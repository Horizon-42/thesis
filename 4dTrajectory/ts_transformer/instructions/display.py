"""What a read sentence's envelopes look like, as geometry to draw (vocabulary design §2).

The design's rule is ONE implementation of the envelopes for the labeller's checks, the executor's
limits and the display (§1 principle 4). This module is the display side, built only from the
public functions of `envelope.py` and `labeller/*`:

- a heading word's rows are `labeller.lateral.heading_spans`, its turn region and where its turn
  may end are `labeller.read.turn_ends_at` (`envelope.turn_ends`), and its hold funnel is
  `envelope.hold_funnel` over the same span — the very sets the labeller's hold check judges
  (`labeller.read.span_funnel`, judged with `envelope.inside_convex`), so what is drawn is what was judged;
- the capture corridor widens by `envelope.corridor_half_width_m`, and its rows are judged by
  `envelope.corridor`;
- the altitude tubes ARE `labeller.vertical.tube_bounds`; the speed spans' band rows are judged by
  `envelope.speed_band`, and counted against the labeller's own `labeller.speed.span_checks`;
- every verdict shown is the labeller's own (`Reading.checks`), never recomputed.

It is outside the labeller's source hash (`artefact.LABELLER_MODULES`): changing how a sentence is
drawn does not change the sentence. Everything is in the airport frame (metres east / north of the
airport reference point, geometric MSL, compass degrees true, seconds); the exporter adds the
geodesy (latitude / longitude, the ellipsoid heights Cesium draws in).

One reading the display makes, stated here: the TURN REGION of a turn smaller than
`turn_rate_min_from_deg` is still drawn with the lowest rate, though the labeller applies that rate
only to larger turns (a smaller change of the ground track is mostly wind drift); the word's
verdict says whether it applied (`rate_min_applies`). A word the flight was already flying at entry
has no turn: its funnel is the one cone from the issue point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayCandidate
from ts_transformer.instructions.labeller.lateral import heading_spans
from ts_transformer.instructions.labeller.read import Admitted, Reading, span_funnel, turn_ends_at
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.labeller.vertical import tube_bounds
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ANGLE, ANGLE_LEVEL, SPEED, Words, wrap180

# ---- plane geometry (compass bearings: 0 = north, clockwise; unit vector (sin b, cos b) in (E, N))
def _unit(bearing_deg: float) -> np.ndarray:
    radians = math.radians(bearing_deg)
    return np.array([math.sin(radians), math.cos(radians)])


def _right(bearing_deg: float) -> np.ndarray:
    """The unit vector 90° clockwise of a bearing: 'right of the track'."""
    return _unit(bearing_deg + 90.0)


@dataclass(frozen=True)
class Line:
    """Points in the airport frame, in drawing order."""

    e_m: np.ndarray
    n_m: np.ndarray

    @classmethod
    def of(cls, points: list[np.ndarray] | np.ndarray) -> Line:
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return cls(e_m=array[:, 0].copy(), n_m=array[:, 1].copy())


# ---- turns
@dataclass(frozen=True)
class TurnRegion:
    """Where a turn from its issue point may take the aircraft (§2.3, `envelope.TurnEnds`): between
    the fastest and the slowest turn, flown at the speeds the flight flew, begun on time or as late
    as allowed. ``turn_deg`` is the shorter way to the target (positive = right)."""

    from_track_deg: float
    turn_deg: float
    rate_min_deg_s: float
    rate_max_deg_s: float
    bank_max_deg: float
    start_delay_max_s: float
    fast: Line                   # the fastest turn, begun on time, to where its track enters θ's band
    slow: Line                   # the slowest turn, likewise (or to the flight's end)
    slow_finished: bool          # the slowest turn gets there before the flight ends
    outline: Line                # `envelope.TurnEnds.outline`
    end: Line                    # where the turn may end: `envelope.TurnEnds.corners`, a parallelogram


def turn_region(flight: Admitted, row: int, target_deg: float, spec: VocabularySpec) -> TurnRegion:
    ends = turn_ends_at(flight, row, target_deg, spec)
    start = np.array([flight.signals.e_m[row], flight.signals.n_m[row]])
    track = float(flight.smoothed.track_deg[row])
    return TurnRegion(
        from_track_deg=track % 360.0, turn_deg=float(wrap180(target_deg - track)),
        rate_min_deg_s=spec.turn_rate_min_deg_s, rate_max_deg_s=spec.turn_rate_max_deg_s,
        bank_max_deg=spec.turn_bank_max_deg, start_delay_max_s=spec.turn_start_delay_max_s,
        fast=Line.of(start + ends.fast), slow=Line.of(start + ends.slow), slow_finished=ends.finished,
        outline=Line.of(start + ends.outline), end=Line.of(start + ends.corners))


# ---- holds
@dataclass(frozen=True)
class Funnel:
    """A hold's allowed positions over its span (§2.3, `envelope.hold_funnel`)."""

    target_deg: float
    length_m: float
    #: Where the turn may end, its half extent across θ; and that plus the widening after ``length_m``.
    start_half_width_m: float
    end_half_width_m: float
    axis: Line                   # the nominal line along θ, through the middle of where the turn may end
    outline: Line


def funnel(shape: envelope.HoldFunnel, target_deg: float) -> Funnel:
    """A funnel to draw, with its axis through the middle of where the turn may end."""
    middle = shape.starts.mean(axis=0)
    return Funnel(target_deg=float(target_deg % 360.0), length_m=shape.length_m,
                  start_half_width_m=shape.start_half_width_m, end_half_width_m=shape.end_half_width_m,
                  axis=Line.of([middle, middle + shape.length_m * _unit(target_deg)]), outline=Line.of(shape.outline))


def on_branch(reference_deg: float, target_deg: float) -> float:
    """``target_deg`` on the branch of an unwrapped track at ``reference_deg`` (the shorter way)."""
    return float(reference_deg + wrap180(target_deg - reference_deg))


@dataclass(frozen=True)
class HeadingEnvelope:
    """One heading word's envelope, from its issue point (§2.3)."""

    word: Instruction
    target_deg: float            # θ, the word's own grid value
    #: Where its turn ends: the labeller's turn end for a single turn or a split turn's last part,
    #: the next part's issue row for an earlier part; ``None`` for a word with no turn.
    turn_end_row: int | None
    #: Where its hold begins: its turn's end as the labeller read it (the issue row for a word the
    #: flight was already holding at entry); ``None`` for a split part the next part supersedes
    #: mid-turn.
    hold_start_row: int | None
    #: Where its hold ends: the next heading word's row, or — for the last one — where the
    #: labeller's capture turn begins (the capture itself when there is no capture turn).
    hold_end_row: int
    from_track_deg: float        # the smoothed track at the issue row, unwrapped
    target_on_track_deg: float   # θ on that track's branch
    #: The chart bands. The turn: from the track at issue to θ, each side widened by the tolerance
    #: (the bound `envelope.turn_progress_ok` implies); ``None`` for a word with no turn. The hold:
    #: θ ± the tolerance; ``None`` when the word has no hold.
    turn_band_deg: tuple[float, float] | None
    hold_band_deg: tuple[float, float] | None
    #: The labeller's check of the turn this word belongs to (`Reading.checks["turns"]`) — shared by
    #: the parts of a split turn; ``None`` for a word the flight was already holding.
    turn_check: dict[str, Any] | None
    #: The labeller's check of this word's hold position (`Reading.checks["hold_positions"]`):
    #: ``None`` for a hold it does not judge (`Reading.checks["holds_not_judged"]` counts why).
    hold_check: dict[str, Any] | None
    turn: TurnRegion | None
    funnel: Funnel | None


def heading_envelopes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[HeadingEnvelope]:
    track = flight.smoothed.track_deg
    tolerance = spec.heading_tolerance_deg
    judged = {h["issue_row"]: h for h in reading.checks["hold_positions"]}
    result = []
    for span in heading_spans(reading.instructions, reading.checks["turns"], reading.checks["capture_turn"],
                              reading.capture_row):
        word = span.word
        target = words.heading_deg(word.value)
        start_track = float(track[word.row])
        target_on_track = on_branch(start_track, target)
        if span.turn is None:
            turn_end_row: int | None = None
            region, turn_band = None, None
        else:
            turn_end_row = span.hold_end if span.hold_start is None else span.hold_start
            region = turn_region(flight, word.row, target, spec)
            low, high = sorted((start_track, target_on_track))
            turn_band = (low - tolerance, high + tolerance)
        drawn = funnel(span_funnel(flight, span, target, spec)[1], target) if span.held else None
        result.append(HeadingEnvelope(
            word=word, target_deg=target, turn_end_row=turn_end_row,
            hold_start_row=span.hold_start if span.held else None, hold_end_row=span.hold_end,
            from_track_deg=start_track, target_on_track_deg=target_on_track, turn_band_deg=turn_band,
            hold_band_deg=(target_on_track - tolerance, target_on_track + tolerance) if span.held else None,
            turn_check=span.turn, hold_check=judged.get(word.row) if span.held else None, turn=region,
            funnel=drawn))
    return result


def centreline(candidate: RunwayCandidate, length_m: float) -> Line:
    """The extended centreline, from the threshold out along the approach side."""
    threshold = np.array([candidate.threshold_e_m, candidate.threshold_n_m])
    return Line.of([threshold, threshold - length_m * _unit(candidate.course_deg)])


def runway(candidate: RunwayCandidate) -> Line:
    """The runway itself: the threshold to its far end along the course."""
    threshold = np.array([candidate.threshold_e_m, candidate.threshold_n_m])
    return Line.of([threshold, threshold + candidate.length_m * _unit(candidate.course_deg)])


@dataclass(frozen=True)
class Corridor:
    """After the capture (§2.2): within `envelope.corridor_half_width_m` of the extended centreline
    and `corridor_course_tolerance_deg` of the course, from the capture to the threshold."""

    before_threshold_m: float     # the capture point's distance before the threshold
    half_width_at_capture_m: float
    half_width_at_threshold_m: float
    axis: Line                    # capture distance → threshold, on the centreline
    outline: Line
    #: The rows from the capture to the sentence's end — every one inside `envelope.corridor`, by the
    #: capture's definition (checked).
    rows: int


def corridor(flight: Admitted, reading: Reading, spec: VocabularySpec) -> Corridor:
    candidate, relative = flight.candidate, flight.relative
    rows = slice(reading.capture_row, len(reading.words))
    inside = envelope.corridor(relative.right_of_course_m[rows], relative.track_minus_course_deg[rows],
                               relative.before_threshold_m[rows], spec.corridor_half_width_m,
                               spec.corridor_widening_deg, spec.corridor_course_tolerance_deg)
    if not inside.all():
        raise ValueError(f"{reading.dataset_id}: {int(np.count_nonzero(~inside))} rows after the capture leave the "
                         f"corridor, which the capture's definition excludes")
    before = float(relative.before_threshold_m[reading.capture_row])
    far_width, near_width = (float(envelope.corridor_half_width_m(d, spec.corridor_half_width_m,
                                                                   spec.corridor_widening_deg)) for d in (before, 0.0))
    threshold = np.array([candidate.threshold_e_m, candidate.threshold_n_m])
    along, right = _unit(candidate.course_deg), _right(candidate.course_deg)
    far = threshold - before * along
    return Corridor(
        before_threshold_m=before, half_width_at_capture_m=far_width, half_width_at_threshold_m=near_width,
        axis=Line.of([far, threshold]),
        outline=Line.of([far - far_width * right, threshold - near_width * right,
                         threshold + near_width * right, far + far_width * right]),
        rows=int(len(inside)))


@dataclass(frozen=True)
class CaptureTurn:
    """From the end of the last hold (or of an inserted intercept) onto the course (§2.2)."""

    start_row: int
    course_on_track_deg: float    # the course on the branch of the smoothed track at the start row
    #: The chart band: from the track at the start row to the course, each side widened by the
    #: heading tolerance (the bound the labeller's progress check implies).
    band_deg: tuple[float, float]
    check: dict[str, Any]         # `Reading.checks["capture_turn"]`
    region: TurnRegion


def capture_turn(flight: Admitted, reading: Reading, spec: VocabularySpec) -> CaptureTurn | None:
    check = reading.checks["capture_turn"]
    if check is None:
        return None
    start = int(check["start_row"])
    track = float(flight.smoothed.track_deg[start])
    course = on_branch(track, flight.candidate.course_deg)
    low, high = sorted((track, course))
    return CaptureTurn(start_row=start, course_on_track_deg=course, check=check,
                       band_deg=(low - spec.heading_tolerance_deg, high + spec.heading_tolerance_deg),
                       region=turn_region(flight, start, flight.candidate.course_deg, spec))


def course_band_deg(flight: Admitted, reading: Reading, spec: VocabularySpec) -> tuple[float, float]:
    """The corridor's course tolerance on the heading chart, on the branch of the track at the capture."""
    course = on_branch(float(flight.smoothed.track_deg[reading.capture_row]), flight.candidate.course_deg)
    return course - spec.corridor_course_tolerance_deg, course + spec.corridor_course_tolerance_deg


# ---- vertical
@dataclass(frozen=True)
class AltitudeTube:
    """One altitude word's tube (§2.5), exactly `labeller.vertical.tube_bounds`, with its rows'
    containment and the labeller's own verdict."""

    word: Instruction
    end_row: int                  # one past the last row it covers
    lower_m: np.ndarray
    upper_m: np.ndarray
    inside: np.ndarray            # per row, bool
    check: dict[str, Any]         # `Reading.checks["vertical"]` for this word


def altitude_tubes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[AltitudeTube]:
    distance, altitude = flight.smoothed.distance_m, flight.smoothed.altitude_m
    checks = {check["row"]: check for check in reading.checks["vertical"]}
    tubes = []
    for word, end, low, high in tube_bounds(reading.instructions, distance, altitude, spec, words):
        span = altitude[word.row: end]
        inside = (span >= low) & (span <= high)
        check = checks[word.row]
        if int(np.count_nonzero(inside)) != check["inside"] or end - word.row != check["rows"]:
            raise ValueError(f"altitude word at row {word.row}: the tube's rows disagree with the labeller's check")
        tubes.append(AltitudeTube(word=word, end_row=int(end), lower_m=low, upper_m=high, inside=inside, check=check))
    return tubes


@dataclass(frozen=True)
class AngleWord:
    word: Instruction
    #: The straight piece's angle the class was read from; ``None`` for the level class, which is
    #: a target reached, not a measured slope.
    measured_deg: float | None


def angle_words(reading: Reading) -> list[AngleWord]:
    result = []
    for word in sorted((i for i in reading.instructions if i.column == ANGLE), key=lambda item: item.row):
        if word.value == ANGLE_LEVEL:
            if "angle_deg" in word.info:
                raise ValueError(f"the level angle word at row {word.row} carries a measured angle")
            result.append(AngleWord(word=word, measured_deg=None))
        else:
            result.append(AngleWord(word=word, measured_deg=float(word.info["angle_deg"])))
    return result


# ---- speed
@dataclass(frozen=True)
class SpeedSpan:
    """One speed word's span (§2.6). A target: the monotone transition from the issue row to the
    row the band is first met, then the band V ± tolerance. "Unspecified": the global range only."""

    word: Instruction
    end_row: int                          # one past the last row
    target_mps: float | None
    #: A target's transition bound, rows ``word.row`` .. ``arrival_row`` (inclusive; to the span's
    #: last row when the next word cuts it before the band is met): the speed stays between the
    #: start and the target, each widened by the tolerance (`envelope.speed_transition_ok`'s bound),
    #: and within `speed_accel_max_mps2` × elapsed time of the start (the acceleration bound).
    arrival_row: int | None
    transition_lower_mps: np.ndarray | None
    transition_upper_mps: np.ndarray | None
    band_mps: tuple[float, float] | None
    band_inside: np.ndarray | None        # per band row (arrival .. end), `envelope.speed_band`
    check: dict[str, Any] | None          # `Reading.checks["speed"]` for this word
    #: "unspecified": the only bound left is the speed words' range, as `read.admit` applies it.
    range_mps: tuple[float, float] | None


def speed_spans(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> list[SpeedSpan]:
    speed, time = flight.smoothed.ground_speed_mps, flight.signals.time_s
    ordered = sorted((i for i in reading.instructions if i.column == SPEED), key=lambda item: item.row)
    ends = [item.row for item in ordered[1:]] + [len(speed)]
    checks = {check["row"]: check for check in reading.checks["speed"]}
    tolerance = spec.speed_tolerance_mps
    spans = []
    for word, end in zip(ordered, ends):
        target = words.speed_mps(word.value)
        if target is None:
            spans.append(SpeedSpan(word=word, end_row=int(end), target_mps=None, arrival_row=None,
                                   transition_lower_mps=None, transition_upper_mps=None, band_mps=None,
                                   band_inside=None, check=None,
                                   range_mps=(spec.speed_min_mps - tolerance, spec.speed_max_mps + tolerance)))
            continue
        check = checks[word.row]
        cut = bool(check["cut_before_arrival"])
        arrival = None if cut else word.row + int(check["arrival_rows"])
        last = end - 1 if cut else arrival
        rows = slice(word.row, last + 1)
        start = float(speed[word.row])
        elapsed = time[rows] - time[word.row]
        low = np.maximum(min(start, target) - tolerance, start - spec.speed_accel_max_mps2 * elapsed)
        high = np.minimum(max(start, target) + tolerance, start + spec.speed_accel_max_mps2 * elapsed)
        band_inside = None if cut else envelope.speed_band(speed[arrival:end], target, tolerance)
        if band_inside is not None and (int(np.count_nonzero(band_inside)) != check["band_inside"]
                                        or len(band_inside) != check["band_rows"]):
            raise ValueError(f"speed word at row {word.row}: the band's rows disagree with the labeller's check")
        spans.append(SpeedSpan(word=word, end_row=int(end), target_mps=float(target), arrival_row=arrival,
                               transition_lower_mps=low, transition_upper_mps=high,
                               band_mps=(target - tolerance, target + tolerance), band_inside=band_inside,
                               check=check, range_mps=None))
    return spans


@dataclass(frozen=True)
class FlightEnvelopes:
    heading: list[HeadingEnvelope]
    capture_turn: CaptureTurn | None
    course_band_deg: tuple[float, float]
    corridor: Corridor
    altitude: list[AltitudeTube]
    angle: list[AngleWord]
    speed: list[SpeedSpan]


def flight_envelopes(flight: Admitted, reading: Reading, spec: VocabularySpec, words: Words) -> FlightEnvelopes:
    """Every envelope of one read flight. ``flight`` is `labeller.read.admit` of the signals the
    ``reading`` was read from (the rows the sentence covers, smoothed, relative to its runway)."""
    if flight.signals.n_rows != len(reading.words):
        raise ValueError(f"{reading.dataset_id}: {flight.signals.n_rows} admitted rows but a sentence of "
                         f"{len(reading.words)} steps")
    return FlightEnvelopes(
        heading=heading_envelopes(flight, reading, spec, words), capture_turn=capture_turn(flight, reading, spec),
        course_band_deg=course_band_deg(flight, reading, spec), corridor=corridor(flight, reading, spec),
        altitude=altitude_tubes(flight, reading, spec, words), angle=angle_words(reading),
        speed=speed_spans(flight, reading, spec, words))
