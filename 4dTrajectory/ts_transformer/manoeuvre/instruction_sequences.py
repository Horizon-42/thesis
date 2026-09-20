"""One flight's sentence as the prior reads it (two-tier v3 stage B, plan §5.2.2): at every
position of the sentence (`instructions.Reading`, one every τ) the five words IN FORCE, the
aircraft's state there, and what the position has to predict — the words at the next position
and whether the flight has landed instead.

    position t:  input  = words[t] (heading, altitude, speed, intercept, runway) + state[t]
                 target = words[t + 1]                            (five factorised heads)
                          landed[t] = 1 at the last position      (no next position: the aircraft is down)

A TRUTH sequence carries the truth's own next words as targets. A ROLLED sequence (CAT-K-style
closed-loop fine-tuning, D55) carries states from a history the executor flew and, as targets,
the truth's words at the same absolute time — the sentence the controller would have said had
the aircraft been where it actually is; the two differ only in ``states`` and ``targets``.

The state token is six scaled features — the position RELATIVE TO THE THRESHOLD (the chart's
origin is the threshold only under the threshold-anchored frames, so the target's chart row is
subtracted), the ground speed, cos / sin of the course; its scale is the one definition here
(the intent-code prior's copy is archived). ``ends_at_landing`` says whether a sequence's last
position is the flight's end (a truth sentence, or a rolled prefix that flew the whole flight)
— the batch labels the landing from it, never from a sequence's length.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, Reading
from flight_scenarios.runway_target import airport_reference_point
from ts_transformer.data.channels import target_chart_position
from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS, interpolate_rows

STATE_TOKEN_FEATURES = ("e", "n", "u", "ground_speed", "cos_course", "sin_course")
STATE_TOKEN_SCALE = np.array([20000.0, 20000.0, 2000.0, 100.0, 1.0, 1.0], dtype=np.float32)


def airport_origin(series: FlightSeries) -> np.ndarray:
    """The AIRPORT reference point in this flight's chart — the origin the state tokens take
    since D66, in place of the landing threshold's.

    Why not the threshold: the prior now has to SAY which runway (D62), and a threshold-anchored
    state hands it the answer — the coordinates are centred on the very runway it is being asked
    to name, so the runway head would score ~1.0 while predicting nothing. Anchored at the
    airport it is the genuine runway-intent problem.

    What it costs the other kinds: a shift bounded by the airport's own extent (KRDU 3.7 km,
    KSTL 6.5 km at most between thresholds) against the 20 km position scale below, and the
    threshold-relative position is recoverable from the runway word plus fixed geometry.

    `airport_reference_point` is this repository's one definition — the same `airports[CODE]`
    entry the harvest and the evaluator read — and `target_chart_position` is the same
    projection the track channels use, so the origin sits in exactly their chart.
    """
    point = airport_reference_point(series.airport)
    reference = replace(series.scenario.target, latitude=point["lat"], longitude=point["lon"],
                        altitude=point["elevation_m"])
    return target_chart_position(reference, series.frame)


def state_token(row: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """The six scaled features of one chart row, the position taken from ``origin`` (since D66
    the AIRPORT reference, `airport_origin`, not the threshold). A row slower than
    `MINIMUM_GROUND_SPEED_MPS` has no course and is refused."""
    row = np.asarray(row, dtype=np.float64)
    edot, ndot = row[VELOCITY_IDX[0]], row[VELOCITY_IDX[1]]
    speed = math.hypot(edot, ndot)
    if speed < MINIMUM_GROUND_SPEED_MPS:
        raise ValueError(f"a row moving at {speed:.2f} m/s has no course; a padded or corrupt row is not a state token")
    course = math.atan2(ndot, edot)
    position = row[list(POSITION_IDX)] - np.asarray(origin, dtype=np.float64)[list(POSITION_IDX)]
    features = np.array([*position, speed, math.cos(course), math.sin(course)], dtype=np.float32)
    return features / STATE_TOKEN_SCALE


def state_tokens(times: np.ndarray, values: np.ndarray, query_times: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """``[len(query_times), 6]``: the state token of the polyline ``(times, values)`` read at
    each query time (nothing extrapolated — `interpolate_rows` refuses a time off the span)."""
    rows = interpolate_rows(times, values, query_times)
    return np.stack([state_token(row, origin) for row in rows]).astype(np.float32)


@dataclass(frozen=True)
class InstructionSequence:
    """``positions_s`` ``[P]``, ``words`` ``[P, 5]`` (in force at each position — the input),
    ``states`` ``[P, 6]``, ``targets`` ``[P, 5]`` (the words the position predicts: the next
    position's), the context, and ``ends_at_landing`` — whether the last position is the flight's
    end (its target is then the landing, and its ``targets`` row is unused)."""

    dataset_id: str
    flight_id: str
    positions_s: np.ndarray
    words: np.ndarray
    states: np.ndarray
    targets: np.ndarray
    typecode: str
    ends_at_landing: bool

    def __post_init__(self) -> None:
        count = len(self.positions_s)
        kinds = len(INSTRUCTION_KINDS)
        if count < 1 or self.words.shape != (count, kinds) or self.targets.shape != (count, kinds) \
                or self.states.shape != (count, len(STATE_TOKEN_FEATURES)):
            raise ValueError(
                f"{self.dataset_id}: {count} positions need [{count}, {kinds}] words and targets and "
                f"[{count}, {len(STATE_TOKEN_FEATURES)}] states, got {self.words.shape}, {self.targets.shape}, {self.states.shape}"
            )

    @property
    def length(self) -> int:
        return len(self.positions_s)


def flight_sequence(series: FlightSeries, reading: Reading) -> InstructionSequence:
    """The truth sequence: the reading's words at its positions, the truth's state at each
    position (the supervision polyline, the one the executor's targets come from), the next
    position's words as targets (the last row repeats itself: nothing follows, the landing is
    the target there)."""
    if reading.flight_id != series.flight_id:
        raise ValueError(f"reading {reading.flight_id} is not series {series.flight_id}")
    states = state_tokens(series.supervision_times, series.supervision_values, reading.positions_s, airport_origin(series))
    targets = np.vstack((reading.words[1:], reading.words[-1:]))
    return InstructionSequence(
        dataset_id=series.dataset_id, flight_id=series.flight_id, positions_s=reading.positions_s.copy(),
        words=reading.words.copy(), states=states, targets=targets,
        typecode=str(series.scenario.aircraft.code), ends_at_landing=True,
    )


def rolled_sequence(truth: InstructionSequence, reading: Reading, flown_times: np.ndarray, flown_values: np.ndarray,
                    said_words: np.ndarray, origin: np.ndarray) -> InstructionSequence:
    """A closed-loop sequence (D55): the positions the flown history covers (a prefix of the
    truth's), the words the loop SAID in force at each (``said_words`` ``[P', 5]``, in the
    reading's word space — NOT the prior's shifted classes), the state read off the FLOWN
    polyline (``origin`` = the threshold's chart row), and as targets the TRUTH's words at the
    next position (`Reading.words_at`, by absolute time — what the controller would say to an
    aircraft that is where this one is). It ends at the landing only when it flew every position."""
    if reading.flight_id != truth.flight_id:
        raise ValueError(f"reading {reading.flight_id} is not sequence {truth.flight_id}")
    said = np.asarray(said_words, dtype=np.int64)
    if not 1 <= len(said) <= len(truth.positions_s):
        raise ValueError(f"{truth.flight_id}: {len(said)} said positions; a rolled prefix covers 1 … {len(truth.positions_s)}")
    if len(reading.positions_s) < 2:
        raise ValueError(f"{truth.flight_id}: a one-position reading has no next position to target")
    positions = truth.positions_s[: len(said)]
    states = state_tokens(flown_times, flown_values, positions, origin)
    targets = reading.words_at(positions + (reading.positions_s[1] - reading.positions_s[0]))     # the next position's truth
    return InstructionSequence(
        dataset_id=truth.dataset_id, flight_id=truth.flight_id, positions_s=positions.copy(),
        words=said, states=states, targets=targets,
        typecode=truth.typecode, ends_at_landing=len(said) == truth.length,
    )


__all__ = [
    "STATE_TOKEN_FEATURES", "STATE_TOKEN_SCALE", "InstructionSequence", "flight_sequence", "rolled_sequence",
    "state_token", "state_tokens",
]
