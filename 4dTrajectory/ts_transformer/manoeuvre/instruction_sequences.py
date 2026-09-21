"""One flight's sentence as the prior consumes it (plan §5.2.2).

    event k:  input  = words[k] (heading, vertical, speed, runway, duration, terminal)
                     + the state token at that event's time + the event's index
              target = words[k + 1]

The last event of a sentence that LANDS has no target: its own terminal word says the flight ends
there (D72), so there is no separate landing label. A rolled prefix's last event does have one —
the truth continues past where the loop stopped.

The state token's position is taken from the AIRPORT reference, not the landing threshold (D66):
threshold-relative coordinates would hand the runway head the answer to its own question.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.manoeuvre.instructions import INSTRUCTION_KINDS, TERMINAL_LANDED, Reading
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
    """One flight's EVENT sequence: ``positions_s`` ``[E]`` the event times, ``words`` ``[E, 6]``
    the event's own words (the input), ``states`` ``[E, 6]``, ``targets`` ``[E, 6]`` the words the
    event predicts (the NEXT event's), the context, and ``ends_at_landing``.

    ``ends_at_landing`` is NOT independent: D72 made "does the sentence stop here" one question
    with one answer, the terminal word. It is validated against ``words[-1, terminal]`` below, so
    the two cannot disagree — a decode that stopped because the prior said ``landed`` cannot then
    be supervised toward ``continue`` at exactly the stop it predicted.
    """

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
        terminal = int(self.words[-1, INSTRUCTION_KINDS.index("terminal")])
        landed = terminal == TERMINAL_LANDED
        if landed != self.ends_at_landing:
            raise ValueError(
                f"{self.dataset_id}: ends_at_landing={self.ends_at_landing} but the last event's terminal "
                f"word is {terminal} — D72 made these ONE answer; supervising a predicted stop toward "
                f"'continue' is exactly the contradiction that check exists to stop"
            )
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
    states = state_tokens(series.supervision_times, series.supervision_values, reading.event_times_s, airport_origin(series))
    targets = np.vstack((reading.words[1:], reading.words[-1:]))
    return InstructionSequence(
        dataset_id=series.dataset_id, flight_id=series.flight_id, positions_s=reading.event_times_s.copy(),
        words=reading.words.copy(), states=states, targets=targets,
        typecode=str(series.scenario.aircraft.code), ends_at_landing=True,
    )


def rolled_sequence(truth: InstructionSequence, reading: Reading, flown_times: np.ndarray, flown_values: np.ndarray,
                    said_words: np.ndarray, said_times_s: np.ndarray, origin: np.ndarray) -> InstructionSequence:
    """A closed-loop sequence (D55): the events the LOOP said, at the times the LOOP said them.

    ``said_words`` ``[E', 6]`` are the words in the reading's own space and ``said_times_s`` the
    absolute times it emitted them — which are NOT the truth's event times, because the loop
    chooses its own gaps (that is what the duration word is for). Matching by index would only be
    right if the loop reproduced the truth's instants; it is matched by TIME instead.

    The target of a said event at ``t`` is the truth's NEXT event after ``t`` — what the controller
    would say next to an aircraft that is where this one is. After the truth's last event there is
    no next, so the target is that last event, which carries ``landed``.

    ``ends_at_landing`` comes from the loop's own terminal word, not from how many events it said.
    """
    if reading.flight_id != truth.flight_id:
        raise ValueError(f"reading {reading.flight_id} is not sequence {truth.flight_id}")
    said = np.asarray(said_words, dtype=np.int64)
    times = np.asarray(said_times_s, dtype=np.float64)
    if len(said) != len(times) or not len(said):
        raise ValueError(f"{truth.flight_id}: {len(said)} said events but {len(times)} times")
    if (np.diff(times) <= 0).any():
        raise ValueError(f"{truth.flight_id}: the said event times are not strictly increasing")
    after = np.searchsorted(reading.event_times_s, times, side="right")
    after = np.minimum(after, len(reading.event_times_s) - 1)
    targets = reading.words[after]
    states = state_tokens(flown_times, flown_values, times, origin)
    return InstructionSequence(
        dataset_id=truth.dataset_id, flight_id=truth.flight_id, positions_s=times.copy(),
        words=said, states=states, targets=targets, typecode=truth.typecode,
        ends_at_landing=bool(said[-1, INSTRUCTION_KINDS.index("terminal")] == TERMINAL_LANDED),
    )


__all__ = [
    "STATE_TOKEN_FEATURES", "STATE_TOKEN_SCALE", "InstructionSequence", "flight_sequence", "rolled_sequence",
    "state_token", "state_tokens",
]
