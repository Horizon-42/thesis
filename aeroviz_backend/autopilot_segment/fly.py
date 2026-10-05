"""A word's segment of a Training flight's closed-loop sentence, flown live by the single-flight executor (stage A,
vocabulary §12.1 A23).

THE SENTENCE is the flight's closed-loop sentence at the row interval Δ the view shows (the artefact's, `artefact.
closed_loop_sentences`), set up as the formal replay set it up — from its first predicted step, the observed flight
moved there, its time limit and reserve (`experiments.training_flights`, the export's own setup). A word is one said in
its grid: Δ row ``row`` of ``column``.

THE SEGMENT of a word runs from the cycle it is heard (``row`` × Δ) to the cycle the next word of its column is heard —
for a heading word a lead later (the heading lead L), since the judge reads a heading word to a lead after the next one
is said. The column's last word is flown on to the outcome, and judged there — as is any word whose flight ends before
its stop (a heading word said in the sentence's last rows, its lead past the flight's end). The flight is flown from the sentence's
first predicted step, so the aircraft is where the sentence's earlier words took it; the answer returns the part from the
word on. Nothing is precomputed: the flight is flown again, and then compared with the artefact's stored flown states on
the 2 s rows (`apart_from_stored`), which the export's flown states are too, and refused by name when it is farther from
them than the executor conformance's bound (`refuse_past_bound`), or, flown to its outcome, ends otherwise than the
artefact's stored outcome says. The backend runs no check of the code at its start (vocabulary D73, A43): this is what it
checks of what it shows — the flown positions and heights and the end. What these do not see (a change of the judge's
decision-altitude check or of a limit or mode that moves no state) is seen by the checks that run where the code changes
and by `check_live`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.judge import Verdict, flown_track
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments import training_flights
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.words import HEADING, UNCHANGED, Words

from aeroviz_backend.autopilot_segment.errors import ExecutorDiffers, RequestRefused, Superseded


@dataclass(frozen=True)
class Segment:
    column: int                 # index into `COLUMNS`
    row: int                    # the Δ row its word is said at, in the closed-loop grid
    word: int                   # the word
    correction: bool            # a word the closed-loop reading added
    start_cycle: int            # the cycle it is heard: ``row`` × the cycles of a Δ row
    stop_cycle: int | None      # the cycle the segment stops before; None: flown on to the outcome


def segment_of(sentence: ClosedLoopSentence, column: int, row: int, interval_s: float, params: ExecutorParams,
               words: Words) -> Segment:
    """The segment of the word said at Δ row ``row`` of ``column`` (module docstring); refused when none is said there."""
    grid = sentence.rows.grid
    if not 0 <= row < len(grid) or grid[row, column] == UNCHANGED:
        raise RequestRefused(f"the closed-loop sentence says no word of column {column} at Δ row {row}")
    every = int(round(interval_s / params.cycle_s))
    later = np.flatnonzero(grid[row + 1:, column] != UNCHANGED)
    stop = None
    if len(later):
        lead = int(round(words.spec.heading_lead_s / params.cycle_s)) if column == HEADING else 0
        stop = (row + 1 + int(later[0])) * every + lead
    return Segment(column=column, row=row, word=int(grid[row, column]),
                   correction=bool(sentence.rows.correction[row, column]),
                   start_cycle=row * every, stop_cycle=stop)


@dataclass(frozen=True)
class FlownSegment:
    segment: Segment
    flown: Flown
    stopped: bool               # the segment's stop ended the flight (no verdict)
    verdict: Verdict | None     # the judge's, when flown to its outcome
    fly_s: float
    judge_s: float


def fly_segment(batch: replay.Batch, inputs: FlightInputs, j: int, sentence: ClosedLoopSentence, column: int, row: int,
                params: ExecutorParams, words: Words, superseded: Callable[[], bool]) -> FlownSegment:
    """The segment of the word at Δ row ``row`` of ``column`` in flight ``j``'s closed-loop sentence (``inputs``: the
    batch's), flown and — when flown to its outcome — judged as the replay judges it."""
    segment = segment_of(sentence, column, row, batch.row_interval_s, params, words)
    started = time.perf_counter()
    try:
        flown, stopped = training_flights.fly_single(batch, inputs, j, params, words, stop_cycle=segment.stop_cycle,
                                                     superseded=superseded)
    except InterruptedError as error:
        raise Superseded(f"a newer request came in: {error}") from None
    fly_s = time.perf_counter() - started
    started = time.perf_counter()
    verdict = None if stopped else replay.judge_batch(replay.subset(batch, [j]), flown, words)[0]
    return FlownSegment(segment, flown, stopped, verdict, fly_s, time.perf_counter() - started)


def last_state(result: FlownSegment) -> int:
    """The last flown state drawn: the stop's last state; the outcome's (`training_flights.last_state_cycle`, the
    export's rule)."""
    if result.verdict is None:
        return int(result.flown.done_cycle[0]) + 1
    return training_flights.last_state_cycle(result.verdict.outcome, result.verdict.end_row)


def refuse_past_bound(horizontal_m: float, vertical_m: float, flight: str, against: str) -> None:
    """A live flight's largest horizontal and vertical distance from the flown states it is compared with (``against``:
    what they are) is within the executor conformance's bound, `STATE_BOUND_M`, or refused by name (`ExecutorDiffers`;
    D73, A43): the one definition, for the service (`apart_from_stored`) and for stage B's hook (`prior.py`)."""
    if horizontal_m > STATE_BOUND_M or vertical_m > STATE_BOUND_M:
        raise ExecutorDiffers(
            f"{flight}: the live executor flies {horizontal_m:.3g} m horizontally and {vertical_m:.3g} m vertically from "
            f"{against}, past {STATE_BOUND_M:g} m: the executor code does not fly this set as it was exported; export "
            f"the set again, or restore the code")


def apart_from_stored(result: FlownSegment, sentence: ClosedLoopSentence, batch: replay.Batch, j: int,
                      step_s: float) -> dict[str, float | int]:
    """The live flight against the artefact's stored flown states on the 2 s rows both have (from the first predicted
    step to the last state flown): how many and the largest horizontal and vertical distance. Refused by name
    (`ExecutorDiffers`) when either is past `STATE_BOUND_M` (`refuse_past_bound`) or when a flight flown to its outcome
    ends otherwise than the stored sentence's outcome (D74): the checks of what the service shows (D73, A43)."""
    step_cycles = int(round(step_s / result.flown.cycle_s))
    stored = sentence.rows.flown_states
    rows = min(len(stored), last_state(result) // step_cycles + 1)
    cycles = np.arange(rows) * step_cycles
    track = flown_track(result.flown.states[0, : cycles[-1] + 1].cpu().numpy(), batch.geometries[j])
    horizontal = float(np.hypot(track["e"][cycles] - stored[:rows, 0], track["n"][cycles] - stored[:rows, 1]).max())
    vertical = float(np.abs(track["height"][cycles] - stored[:rows, 2]).max())
    if not (np.isfinite(horizontal) and np.isfinite(vertical)):
        raise ValueError(f"the live flight is {horizontal} m / {vertical} m from the stored states: a non-finite state")
    flight = batch.signals[j].dataset_id
    refuse_past_bound(horizontal, vertical, flight, "the artefact's stored states")
    if result.verdict is not None and result.verdict.outcome != sentence.withheld.outcome:
        raise ExecutorDiffers(f"{flight}: the live flight ends {result.verdict.outcome}, the artefact's stored outcome is "
                              f"{sentence.withheld.outcome}: the executor or the judge code does not fly this set as it "
                              f"was exported; export the set again, or restore the code")
    return {"rows": int(rows), "horizontalM": horizontal, "verticalM": vertical}
