"""The inputs of a row (prior design §2, §7 item 2; milestone B1): from the states on the data's 2 s rows to the inputs of
the Δ rows of one aircraft.

**Scales** (D41): every input is a physical value divided by a constant in SI units, never a statistic of the data.
The distances along and across a candidate go in as asinh(d / `DISTANCE_SCALE_M`): linear within a few hundred metres
of a final, logarithmic out to the 25 km of the slice.

**Motion** (D25, D60): the ground speed, the vertical rate and the direction of motion of a row come from the
displacement between the 2 s row before it and the row, at every Δ — only positions and heights, never the stored
track, ground speed or vertical rate (on the observed rows a fit that reads 7.5 s after the row). Row 0 of an aircraft
has no state before it: its motion is 0 and ``no_motion`` 1.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry, published_glidepath_height_m, relative_to_runway
from ts_transformer.instructions.artefact import STATE_COLUMNS
from ts_transformer.instructions.artefact import SentenceRows as ClosedLoopRows
from ts_transformer.instructions.grammar import InForce, apply
from ts_transformer.instructions.labeller.interval import interval_rows, on_interval_rows
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.batch import IN_FORCE_WORDS, OWN_FEATURES, SentenceRows, variant_features
from ts_transformer.prior.landings import LandingIndex, utc_s

#: Fixed scales of the inputs (D41), SI.
DISTANCE_SCALE_M = 1_000.0
HEIGHT_SCALE_M = 1_000.0
#: The height above a glidepath goes in as asinh(h / `GLIDEPATH_HEIGHT_SCALE_M`): linear near the glidepath, compressed
#: far from it (D65).
GLIDEPATH_HEIGHT_SCALE_M = 100.0
SPEED_SCALE_MPS = 100.0
VERTICAL_RATE_SCALE_MPS = 10.0
#: The runway length of the variant ``constants`` (D39).
LENGTH_SCALE_M = 1_000.0
#: The landings on a candidate in 30 min, divided by this count.
LANDINGS_SCALE = 10.0
#: The time since a word (D17): log(1 + t / `SINCE_TIME_S`) / `SINCE_DIVISOR`.
SINCE_TIME_S = 2.0
SINCE_DIVISOR = 5.0


class Motion(NamedTuple):
    """The motion of rows (module docstring): ``ground_speed_mps``, ``vertical_rate_mps``, ``track_deg`` (compass) and
    ``known`` (False at an aircraft's row 0), each ``[rows]``."""

    ground_speed_mps: np.ndarray
    vertical_rate_mps: np.ndarray
    track_deg: np.ndarray
    known: np.ndarray


def motion(at: np.ndarray, before: np.ndarray, known: np.ndarray, step_s: float) -> Motion:
    """The motion of rows from their states ``at`` and the states of the 2 s row before each (``before``; both ``[rows,
    3]``: e, n, height), where ``known`` (row 0 of an aircraft has none: its motion is 0, D60)."""
    de, dn, dh = (at - before).T
    known = np.asarray(known, dtype=bool)
    return Motion(np.where(known, np.hypot(de, dn) / step_s, 0.0), np.where(known, dh / step_s, 0.0),
                  np.where(known, np.degrees(np.arctan2(de, dn)) % 360.0, 0.0), known)


def since_input(seconds: np.ndarray) -> np.ndarray:
    """The input of D17 of the times ``seconds`` since a column said its word in force."""
    return np.log1p(np.asarray(seconds, dtype=np.float64) / SINCE_TIME_S) / SINCE_DIVISOR


def scaled_distance(d_m: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(d_m, dtype=np.float64) / DISTANCE_SCALE_M)


def direction_inputs(track_deg: np.ndarray, course_deg: float, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The sine and cosine of the direction of motion minus a course, 0 where the motion is not known (D60)."""
    angle = np.radians(np.asarray(track_deg, dtype=np.float64) - course_deg)
    return np.where(known, np.sin(angle), 0.0), np.where(known, np.cos(angle), 0.0)


def state_inputs(at: np.ndarray, before: np.ndarray, known: np.ndarray, landings: np.ndarray, geometry: AirportGeometry,
                 variant: str, step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The inputs of rows that come from the aircraft's states (§2; D13, D23–D25, D58, D60): ``(own [rows,
    len(OWN_FEATURES)], candidates [rows, K, F])``. ``at`` and ``before``: each row's state and the state of the 2 s row
    before it (e, n in the airport frame, MSL height; ``[rows, 3]``), ``known`` whether that row exists (`motion`);
    ``landings``: the landings on each candidate in the 30 min before each row (``[rows, K]``, the candidates' order).
    The one function of §7 item 2 for the states: the training sentences call it on every row at once, a loop on its
    newest row."""
    move = motion(at, before, known, step_s)
    e, n, h = at.T
    count = len(at)
    own = np.zeros((count, len(OWN_FEATURES)))
    own[:, OWN_FEATURES.index("height_above_elevation")] = (h - geometry.elevation_m) / HEIGHT_SCALE_M     # D58
    own[:, OWN_FEATURES.index("ground_speed")] = move.ground_speed_mps / SPEED_SCALE_MPS
    own[:, OWN_FEATURES.index("vertical_rate")] = move.vertical_rate_mps / VERTICAL_RATE_SCALE_MPS
    own[:, OWN_FEATURES.index("no_motion")] = ~move.known
    features = variant_features(variant)
    candidates = np.zeros((count, len(geometry.candidates), len(features)))
    for k, candidate in enumerate(geometry.candidates):
        relative = relative_to_runway(e, n, move.track_deg, h, candidate)
        sine, cosine = direction_inputs(move.track_deg, candidate.course_deg, move.known)
        glidepath = published_glidepath_height_m(geometry, k, relative.before_threshold_m)
        values = {"before_threshold": scaled_distance(relative.before_threshold_m),
                  "right_of_final": scaled_distance(relative.right_of_course_m),
                  "height_above_threshold": relative.height_above_threshold_m / HEIGHT_SCALE_M,
                  "motion_minus_course_sin": sine, "motion_minus_course_cos": cosine,
                  "height_above_glidepath": np.arcsinh((relative.height_above_threshold_m - glidepath)
                                                       / GLIDEPATH_HEIGHT_SCALE_M),
                  "landings_30min": landings[:, k] / LANDINGS_SCALE,
                  "length": np.full(count, candidate.length_m / LENGTH_SCALE_M),
                  "threshold_elevation": np.full(count, candidate.elevation_m / HEIGHT_SCALE_M)}
        candidates[:, k] = np.stack([values[name] for name in features], axis=-1)
    return own, candidates


class Heard:
    """The words in force of one aircraft as a sentence goes on (§2 "Words in force"; D17, D46): the runway in force,
    G, the heading word's track, the altitude, angle and speed words, and when each column said its word. `hear` takes
    a row said on, through the grammar (`instructions.grammar.apply`, which refuses a row it does not pass); `inputs`
    gives the words-in-force inputs of a later row. Before the first predicted step has been said, nothing is in force.
    The one function of §7 item 2 for the words: the training sentences walk it row by row, as a loop does."""

    def __init__(self, geometry: AirportGeometry, words: Words) -> None:
        self.geometry, self.words = geometry, words
        self.state: InForce | None = None
        #: the heading word's track (compass): a heading word keeps the track it said under the runway in force where it
        #: was heard (D46)
        self.track_deg = 0.0
        self.said_s = np.zeros(len(COLUMNS))

    def copy(self) -> Heard:
        """The same words in force, apart from this one's (a speaker's copy, D96 item 5)."""
        out = Heard(self.geometry, self.words)
        out.state, out.track_deg, out.said_s = self.state, self.track_deg, self.said_s.copy()
        return out

    def hear(self, step: np.ndarray, height_m: float, time_s: float) -> None:
        """The row ``step`` said at ``time_s`` (the aircraft's own seconds) to the aircraft ``height_m`` above E."""
        self.state = apply(self.state, step, height_m, self.words, len(self.geometry.candidates))
        if step[HEADING] != UNCHANGED:
            self.track_deg = self.words.heading_track_deg(int(step[HEADING]), self._course())
        # the runway column's time is the time since a candidate was said, not "go-around" (§2, D17)
        heard = [c for c in range(len(COLUMNS)) if step[c] != UNCHANGED and not (c == RUNWAY and step[c] < 0)]
        self.said_s[heard] = time_s

    def inputs(self, time_s: float) -> tuple[int, bool, np.ndarray, np.ndarray, np.ndarray]:
        """``(runway in force, G, heading in force relative to the course of that runway (sine, cosine), the words of
        `batch.IN_FORCE_WORDS`, the `since_input` of each column)`` of a row at ``time_s``: −1, (0, 0) and 0 before
        anything is said."""
        if self.state is None:
            return -1, False, np.zeros(2), np.full(len(IN_FORCE_WORDS), -1), np.zeros(len(COLUMNS))
        relative = np.radians(self.track_deg - self._course())
        return (self.state.runway, self.state.go_around, np.array([np.sin(relative), np.cos(relative)]),
                np.array([self.state.altitude, self.state.angle, self.state.speed]), since_input(time_s - self.said_s))

    def _course(self) -> float:
        return self.geometry.candidates[self.state.runway].course_deg


def sentence_rows(sentence: ClosedLoopRows, flight: Mapping[str, Any], geometry: AirportGeometry,
                  landings: LandingIndex, words: Words, *, interval_s: float, split: str, variant: str) -> SentenceRows:
    """The inputs and targets of one closed-loop sentence on its Δ rows (§2, §7 item 2): ``sentence`` its rows alone —
    what a model may read, never the withheld fields (vocabulary D82) — ``flight`` its record in the split's signals
    (`instructions.artefact.signals_flights`; its entry time and its key, by which its own landing is left out of
    ``landings``, its airport's). Rows before the first predicted step are observed, rows from it are flown (the
    artefact's states, D32)."""
    every = interval_rows(interval_s, words.spec.step_s)
    if not np.array_equal(sentence.on_interval, on_interval_rows(len(sentence.states), every)):
        raise ValueError(f"{flight['dataset_id']}: the sentence's Δ rows are not every {every}-th 2 s row")
    positions = sentence.states[:, [STATE_COLUMNS.index(name) for name in ("e_m", "n_m", "height_m")]]
    rows = np.flatnonzero(sentence.on_interval)
    count, start = len(rows), sentence.start
    # the UTC time of each Δ row: signals row 0 at the flight's entry, 2 s rows
    utc = utc_s(flight["entry_time_utc"]) + (sentence.first_row + rows) * words.spec.step_s
    counts = landings.counts_before(utc, without=own_flight_key(flight))
    own, candidates = state_inputs(positions[rows], positions[np.maximum(rows - 1, 0)], rows > 0,
                                   counts[:, [landings.runways.index(c.ident) for c in geometry.candidates]], geometry,
                                   variant, words.spec.step_s)
    targets = np.full((count, len(COLUMNS)), UNCHANGED, dtype=np.int64)
    targets[start:] = sentence.grid
    runway = np.full(count, -1, dtype=np.int64)
    go_around = np.zeros(count, dtype=bool)
    heading = np.zeros((count, 2))
    levels = np.full((count, len(IN_FORCE_WORDS)), -1, dtype=np.int64)
    since = np.zeros((count, len(COLUMNS)))
    heard = Heard(geometry, words)
    height = positions[rows, 2] - geometry.elevation_m
    for j in range(count):
        runway[j], go_around[j], heading[j], levels[j], since[j] = heard.inputs(j * interval_s)
        if j >= start:
            heard.hear(targets[j], float(height[j]), j * interval_s)
    return SentenceRows(flight_key=flight["dataset_id"], airport=geometry.code, split=split, first_step=start,
                        time_s=(np.arange(count) * interval_s).astype(np.float32), own=own.astype(np.float32),
                        candidates=candidates.astype(np.float32), runway_in_force=runway, go_around=go_around,
                        heading_in_force=heading.astype(np.float32), words_in_force=levels,
                        since=since.astype(np.float32), targets=targets)


def own_flight_key(flight: Mapping[str, Any]) -> str:
    """The tracks roster's key of a flight of the signals: its ``dataset_id`` is ``<airport>:<flight_key>`` (a MIRROR
    of `data.dataset`'s ``dataset_id``, which the prior cannot import: the training plane;
    `tests/test_prior_inputs.py` pins the two)."""
    airport, _, key = str(flight["dataset_id"]).partition(":")
    if airport != flight["airport"] or not key:
        raise ValueError(f"{flight['dataset_id']} is not <airport>:<flight key> of {flight['airport']}")
    return key
