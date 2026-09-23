"""The BOX vocabulary: a word is an INTERVAL the track must stay inside.

An arrival is a SENTENCE of EVENTS, each carrying six words — heading, altitude, speed, the
runway they are measured against, how long this event's boxes are held, and whether the approach
ends here. The contract is CONTAINMENT: every row of the track lies inside the boxes in force at
that moment. That replaces the fitting reader's "take the segment's median, snap it to the nearest
centre", and it changes what a vocabulary is for — there is no quantisation error to minimise, so
the design trades SENTENCE LENGTH against BOX WIDTH and nothing else.

Three rules hold for every kind:

* **the boxes tile** — no gap, no overlap. The fitting reader needed the opposite (a tolerance
  under half a bin, so a plateau could not straddle two words); containment needs boxes that cover
  the axis, or a value in a gap has no legal reading at all.
* **the ends are bounded and the outside is REFUSED**, never clamped and never swallowed by a ray.
  A ray contains anything, so a corrupt row would read as a legal word. The bounds are set by what
  is IMPOSSIBLE (30 m/s is under any airliner's stall speed; 250 m/s over the ground is past any
  arrival), never by a quantile of the observed distribution — a 4,900 m arrival and a 200 m/s
  ground speed high in the slice are real and get words.
* **±5 %, with an absolute floor wherever the percentage would vanish** — the heading's course box
  is ±1° because 5 % of zero is no box, and the altitude wedge closes onto ±5 % of ``T + h0``.

The heading and the speed are cut by run-length encoding: their boxes tile, so every row falls in
exactly one and the reading has no choices. The ALTITUDE is different — its word is a TARGET and
its box is that target's own BACKWARD REACHABLE SET, opening backwards from the segment's end at
the aircraft's own flight path angle envelope, so the segments come from a greedy longest reach
read from the END (the target anchors it there). The wedge is what decouples terminal precision
from sentence length: a slab's box is as wide at the start of a segment as at its end, so a tight
terminal box means a fine ladder everywhere and one word per bin crossing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from ts_transformer.manoeuvre.instructions import course_frame, min_rows, smooth

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The reading rule's version. It is part of the spec and therefore of the sha, so an artefact
#: read under an older rule is refused by name rather than reinterpreted.
READING_RULE = "box-v3"
SCHEMA = "ts-box-vocabulary-v1"
KINDS = ("heading", "altitude", "speed", "runway", "duration", "terminal")
TERMINAL_CONTINUE, TERMINAL_LANDED, TERMINAL_GO_AROUND = 0, 1, 2
TERMINAL_WORDS = 3


@dataclass(frozen=True)
class BoxVocabulary:
    """The boxes and the bounds; ``sha256`` is over this spec and nothing else."""

    redundancy_fraction: float = 0.05
    heading_floor_deg: float = 1.0
    heading_range_deg: float = 180.0
    speed_low_mps: float = 30.0
    speed_high_mps: float = 250.0
    altitude_h0_m: float = 50.0
    altitude_top_m: float = 6000.0
    altitude_bottom_m: float = -150.0
    #: How fast the altitude envelope opens BACKWARDS from its target. It must stay under the
    #: fleet's own descent angle (measured p50 2.98°): once the envelope opens as fast as an
    #: aircraft descends, one word covers a whole approach and says nothing — at 3.0° the reading
    #: gives 2 words a flight inside a median 740 m band.
    altitude_down_deg: float = 1.5
    altitude_up_deg: float = 1.0
    duration_bin_s: float = 2.0
    duration_max_s: float = 600.0
    course_smoothing_s: float = 6.0
    smoothing_s: float = 10.0
    reading_rule: str = READING_RULE

    def __post_init__(self) -> None:
        if self.reading_rule != READING_RULE:
            raise ValueError(f"reading_rule {self.reading_rule!r} is not this code's ({READING_RULE!r})")
        if not 0.0 < self.redundancy_fraction < 0.5:
            raise ValueError("redundancy_fraction is a fraction under a half")
        for name in ("heading_floor_deg", "altitude_h0_m", "duration_bin_s", "course_smoothing_s", "smoothing_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} is positive")
        if not self.speed_low_mps < self.speed_high_mps:
            raise ValueError("the speed bounds are ordered")
        if not self.altitude_bottom_m < 0.0 < self.altitude_top_m:
            raise ValueError("the altitude ladder spans the threshold: an arrival below the "
                             "threshold elevation is normal and must have a word")
        if not 0.0 < self.altitude_down_deg < 3.0:
            raise ValueError("altitude_down_deg stays under the fleet's own descent angle (2.98°), "
                             "or one word covers the approach and says nothing")

    # ── the boxes ────────────────────────────────────────────────────────
    @property
    def ratio(self) -> float:
        """The tiling ratio: a box of ±p about its centre abuts the next one at this ratio."""
        return (1.0 + self.redundancy_fraction) / (1.0 - self.redundancy_fraction)

    @property
    def heading_edges(self) -> np.ndarray:
        """Box edges over the relative course, symmetric about the final approach course.

        The half width is ``max(floor, 5 % of the value)``: 2° wide on the course, 3° at 30°, 9° at
        90°, 18° on the reciprocal — the resolution lands where it decides whether the aircraft
        can line up, and is spent nowhere else."""
        p, floor = self.redundancy_fraction, self.heading_floor_deg
        edges = [floor]
        while edges[-1] < self.heading_range_deg:
            half = max(floor, p * edges[-1] / (1.0 - p))
            centre = edges[-1] + floor if half == floor else edges[-1] / (1.0 - p)
            edges.append(centre + half)
        edges[-1] = self.heading_range_deg
        right = np.array(edges)
        return np.unique(np.concatenate([-right[::-1], [-floor, floor], right]))

    @property
    def speed_edges(self) -> np.ndarray:
        edges = [self.speed_low_mps]
        while edges[-1] < self.speed_high_mps:
            edges.append(edges[-1] * self.ratio)
        return np.array(edges)

    @property
    def altitude_targets(self) -> np.ndarray:
        """The target ladder, SIGNED: ``h0·(r^k − 1)`` above the threshold and its mirror below.

        It has to span the threshold. A track that passes under the threshold elevation is an
        ordinary arrival, and a ladder floored at zero refuses it — that was the commonest refusal
        the first reading produced (65 flights, most of them never above 1,320 m)."""
        out, k = [0.0], 1
        while True:
            value = self.altitude_h0_m * (self.ratio ** k - 1.0)
            if value <= self.altitude_top_m:
                out.append(value)
            if -value >= self.altitude_bottom_m:
                out.append(-value)
            if value > self.altitude_top_m and -value < self.altitude_bottom_m:
                return np.array(sorted(out))
            k += 1

    def altitude_half_width(self, target: float | np.ndarray) -> float | np.ndarray:
        """What the wedge closes onto at a segment's end: the target's own ±5 % box, which is also
        the ladder's own spacing — so a one-row segment ALWAYS has a target and the reading cannot
        fail."""
        return self.redundancy_fraction * (np.abs(target) + self.altitude_h0_m)

    @property
    def duration_words(self) -> int:
        return int(round(self.duration_max_s / self.duration_bin_s)) + 1

    @property
    def words(self) -> dict[str, int]:
        """The class count of every kind the SPEC fixes; the runway's is the cohort's."""
        return {"heading": len(self.heading_edges) - 1, "altitude": len(self.altitude_targets),
                "speed": len(self.speed_edges) - 1,
                "duration": self.duration_words, "terminal": TERMINAL_WORDS}

    @property
    def spec(self) -> dict[str, float | str]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()


def _bin(edges: np.ndarray, values: np.ndarray, kind: str) -> np.ndarray:
    """Which box each value falls in. Outside the ends is REFUSED — see the module docstring."""
    if values.min() < edges[0] or values.max() > edges[-1]:
        raise ValueError(f"{kind} leaves the vocabulary's range [{edges[0]:.4g}, {edges[-1]:.4g}]: "
                         f"[{values.min():.4g}, {values.max():.4g}]")
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def altitude_words(height: np.ndarray, remaining_m: np.ndarray, vocabulary: BoxVocabulary,
                   backwards: bool = True) -> np.ndarray:
    """One target per row: the fewest targets whose wedges contain the profile.

    The envelope of target ``T`` at a row with ``r`` metres of PATH still to run inside its
    segment is ``[T − r·tan(up) − f(T), T + r·tan(down) + f(T)]`` with ``f`` the target's ±5 % box.
    ``r`` is the remaining PATH LENGTH (ground speed integrated), never the along-course
    projection: that projection GROWS on a downwind leg, so a vectored approach reads as
    unreachable and is cut one row at a time (measured: it put a third of the fleet past a hundred
    words a flight).

    A segment need not reach its target — a real clearance can be superseded before the aircraft
    gets there — so the target is any ladder point the whole segment admits, not the value at its
    end.
    """
    targets = vocabulary.altitude_targets
    p, h0 = vocabulary.redundancy_fraction, vocabulary.altitude_h0_m
    td = math.tan(math.radians(vocabulary.altitude_down_deg))
    tu = math.tan(math.radians(vocabulary.altitude_up_deg))
    if height.min() < targets[0] or height.max() > targets[-1]:
        raise ValueError(f"altitude leaves the vocabulary's range [{targets[0]:.4g}, {targets[-1]:.4g}]: "
                         f"[{height.min():.4g}, {height.max():.4g}]")
    above, below = height - remaining_m * td, height + remaining_m * tu
    out = np.empty(len(height), dtype=np.int64)

    def pick(hi: float, lo: float, end: float) -> int:
        low = (hi + end * td - p * h0) / (1.0 + p)
        high = (lo - end * tu + p * h0) / (1.0 - p)
        inside = np.flatnonzero((targets >= low) & (targets <= high))
        return int(inside[0]) if len(inside) else -1

    n = len(height)
    step = -1 if backwards else 1
    edge = n - 1 if backwards else 0
    while 0 <= edge < n:
        end = remaining_m[edge]
        hi, lo = above[edge], below[edge]
        word = pick(hi, lo, end)
        if word < 0:
            raise ValueError("no target for a single row: the ladder does not tile its own boxes")
        reach = edge
        while 0 <= reach + step < n:
            hi2, lo2 = max(hi, above[reach + step]), min(lo, below[reach + step])
            nxt = pick(hi2, lo2, end)
            if nxt < 0:
                break
            hi, lo, reach, word = hi2, lo2, reach + step, nxt
        first, last = (reach, edge) if backwards else (edge, reach)
        out[first : last + 1] = word
        edge = reach + step
    return out


def read_boxes(series: "FlightSeries", vocabulary: BoxVocabulary, runway_word: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(event times, words [E, 6], hold seconds)`` for one flight.

    The DURATION sits on the row it describes ("these boxes are held for T"), not on the row after
    it: the chain between two boxes cannot be checked without T at the moment the box is issued,
    and a decoder cannot emit a box and then wait for the next event to learn how long it lasts.
    That also removes the separate landing event the gap-to-the-previous form needed — the last
    box carries its own hold, so the holds sum to the flight.
    """
    frame = course_frame(series)
    times = np.asarray(frame["t"], dtype=np.float64)
    dt = float(np.median(np.diff(times)))
    # The course is smoothed UNWRAPPED and wrapped afterwards. A moving average over the wrapped
    # signal averages +179° and −179° to 0° — the course itself — so a third of the fleet gets
    # heading words pointing the opposite way somewhere on the downwind (measured: 1.8 % of rows,
    # 33.3 % of flights, median error 180°).
    course = (smooth(frame["course_unwrapped_deg"], min_rows(vocabulary.course_smoothing_s, dt)) + 180.0) % 360.0 - 180.0
    window = min_rows(vocabulary.smoothing_s, dt)
    height = smooth(frame["height_m"], window)
    speed = smooth(frame["ground_speed_mps"], window)
    path = np.concatenate([[0.0], np.cumsum(speed[1:] * np.diff(times))])

    columns = {
        "heading": _bin(vocabulary.heading_edges, course, "heading"),
        "altitude": altitude_words(height, path[-1] - path, vocabulary),
        "speed": _bin(vocabulary.speed_edges, speed, "speed"),
    }
    marks = np.unique(np.concatenate([[0]] + [np.flatnonzero(np.diff(c)) + 1 for c in columns.values()]))
    moments = times[marks]
    holds = np.concatenate([np.diff(moments), [float(times[-1] - moments[-1])]])
    duration = np.floor(holds / vocabulary.duration_bin_s + 0.5).astype(np.int64)
    if duration.max() >= vocabulary.duration_words:
        raise ValueError(f"a box is held {holds.max():.0f} s, past the duration ceiling "
                         f"({vocabulary.duration_max_s:g} s)")
    terminal = np.full(len(marks), TERMINAL_CONTINUE, dtype=np.int64)
    terminal[-1] = TERMINAL_LANDED
    words = np.stack([columns["heading"][marks], columns["altitude"][marks], columns["speed"][marks],
                      np.full(len(marks), runway_word, dtype=np.int64), duration, terminal], axis=1)
    return moments, words, holds


def contains(series: "FlightSeries", vocabulary: BoxVocabulary, moments: np.ndarray,
             words: np.ndarray) -> bool:
    """Whether the track really lies inside the words in force — the contract, checked."""
    frame = course_frame(series)
    times = np.asarray(frame["t"], dtype=np.float64)
    dt = float(np.median(np.diff(times)))
    course = (smooth(frame["course_unwrapped_deg"], min_rows(vocabulary.course_smoothing_s, dt)) + 180.0) % 360.0 - 180.0
    window = min_rows(vocabulary.smoothing_s, dt)
    height = smooth(frame["height_m"], window)
    speed = smooth(frame["ground_speed_mps"], window)
    path = np.concatenate([[0.0], np.cumsum(speed[1:] * np.diff(times))])
    remaining = path[-1] - path
    index = np.clip(np.searchsorted(moments, times, side="right") - 1, 0, len(words) - 1)
    heading_edges, speed_edges = vocabulary.heading_edges, vocabulary.speed_edges
    targets = vocabulary.altitude_targets
    if not ((heading_edges[words[index, 0]] <= course) & (course <= heading_edges[words[index, 0] + 1])).all():
        return False
    if not ((speed_edges[words[index, 2]] <= speed) & (speed <= speed_edges[words[index, 2] + 1])).all():
        return False
    # The wedge is anchored on the ALTITUDE segment's end, which is not the event's end: an event
    # opens whenever ANY kind changes, so one altitude word spans several of them. Grouping by the
    # altitude column's own runs is the reading's segmentation (two adjacent segments sharing a
    # target would merge here, which only widens the envelope, never narrows it).
    column = words[index, 1]
    target = targets[column]
    half = vocabulary.altitude_half_width(target)
    ends = np.concatenate([np.flatnonzero(np.diff(column)) + 1, [len(times)]])
    segment_end = np.repeat(remaining[ends - 1], np.diff(np.concatenate([[0], ends])))
    inside = remaining - segment_end
    td = math.tan(math.radians(vocabulary.altitude_down_deg))
    tu = math.tan(math.radians(vocabulary.altitude_up_deg))
    return bool(((height >= target - inside * tu - half - 1e-6)
                 & (height <= target + inside * td + half + 1e-6)).all())
