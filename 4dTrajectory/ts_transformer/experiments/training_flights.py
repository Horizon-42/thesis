"""The flights of a Training set of stage A on their closed-loop sentences (vocabulary §12.1 A23): the one setup the
Training export (`training_export`) and the backend's live executor (`aeroviz_backend/autopilot_segment/`) share — so a
live segment is the export's flight, and the two are checked against each other (outline §6 item 6). A shared module of
both, not a runner (`experiments.__main__.NOT_RUNNERS`); here and not in `autopilot/` because it joins the executor to
the Training sets, which the executor does not know.

A flight is drawn by its dataset id, read again by the labeller and checked against its stored sentence, then set up on
its closed-loop sentence at the row interval Δ from its first predicted step — `closed_loop.replay_batch`, as the formal
replay sets it up — after the start's refusals (`autopilot.start.require_startable`, D67: the closed-loop file flown by
the spec's parameters, each sentence's observed rows its flight's by the start rule), so its state there is the start's
(`flights.start_state`, D77). `fly_single` flies one of them with the single-flight executor's one loop
(`autopilot.single.fly`, the executor check's), optionally stopped before a cycle.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay, single
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.frame import ALT, LAT, LON
from ts_transformer.autopilot.judge import Verdict
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.autopilot.start import require_startable
from ts_transformer.instructions.artefact import ClosedLoopSentence, closed_loop_sentences, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import Words

CPU = torch.device("cpu")
#: The outcomes of an approach crossing, whose crossing carries the judge's decision-altitude check (`judge._outcome`).
APPROACH_OUTCOMES = ("landed", "unstable_at_minimums")


@dataclasses.dataclass(frozen=True)
class SetFlights:
    """A set's flights of one split, drawn by id and each read again (`open_flights`): what every row interval shares."""

    drawn: replay.Drawn
    readings: list[Reading]


def open_flights(instructions: Path, split: str, dataset_ids: Sequence[str], words: Words) -> SetFlights:
    """The flights ``dataset_ids`` of ``split``, in that order: drawn (`replay.draw_flights`, the flight's own airframe
    or its stand-in), each read again by the labeller and refused by name unless it gives its stored sentence — its words
    and its runway."""
    spec = words.spec
    signals = load_signals(instructions, split)
    by_id = {flight.dataset_id: i for i, flight in enumerate(signals)}
    missing = [d for d in dataset_ids if d not in by_id]
    if missing:
        raise ValueError(f"{missing[:3]} are not {split} flights of {instructions.name}")
    sentences = load_sentences(instructions, split, spec)
    stored = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    drawn = replay.draw_flights(instructions, split, [by_id[d] for d in dataset_ids], per_airport=0, seed=0,
                                groups=(replay.OWN, replay.STAND_IN))
    # the draw permutes; put the flights back in the order asked, and refuse one it left out (no airframe)
    at = {flight.dataset_id: k for k, flight in enumerate(drawn.signals)}
    if sorted(at) != sorted(dataset_ids):
        raise ValueError(f"{split}: {sorted(set(dataset_ids) - set(at))[:3]} cannot be flown (no airframe)")
    order = [at[d] for d in dataset_ids]
    drawn = dataclasses.replace(drawn, indices=[drawn.indices[k] for k in order],
                                signals=[owned_signals(drawn.signals[k]) for k in order],
                                series=[drawn.series[k] for k in order], groups=[drawn.groups[k] for k in order])
    readings = []
    for i, flight in zip(drawn.indices, drawn.signals):
        reading = read_flight(flight, drawn.geometries[flight.airport], spec, words)
        k = stored[i]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
        if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
            raise ValueError(f"{flight.dataset_id}: the re-read sentence differs from the stored one")
        readings.append(reading)
    return SetFlights(drawn, readings)


def owned_signals(flight: FlightSignals) -> FlightSignals:
    """A flight's signals with arrays of its own: the split's loaded signals are views into one array of every flight
    (`artefact.load_signals`), which a kept view would keep alive whole (~0.5 GB for train)."""
    return dataclasses.replace(flight, **{f.name: getattr(flight, f.name).copy() for f in dataclasses.fields(flight)
                                          if isinstance(getattr(flight, f.name), np.ndarray)})


def owned_sentence(sentence: ClosedLoopSentence) -> ClosedLoopSentence:
    """A closed-loop sentence with arrays of its own (`closed_loop_sentences` gives views into the split's file, ~0.1–0.7
    GB loaded, which a kept view would keep alive whole)."""
    def owned(part: Any) -> Any:
        return dataclasses.replace(part, **{f.name: getattr(part, f.name).copy() for f in dataclasses.fields(part)
                                            if isinstance(getattr(part, f.name), np.ndarray)})
    return ClosedLoopSentence(rows=owned(sentence.rows), withheld=owned(sentence.withheld))


def stored_closed_loop(instructions: Path, split: str, interval_s: float, words: Words) -> dict[int, ClosedLoopSentence]:
    """The split's closed-loop sentences at ``interval_s``, by their flight's place in the split's signals (the reader of
    vocabulary §6 item 3, D82)."""
    return closed_loop_sentences(instructions, split, interval_s, words.spec)


def closed_loop_batch(instructions: Path, split: str, flights: SetFlights, stored: dict[int, ClosedLoopSentence],
                      interval_s: float, params: ExecutorParams, words: Words
                      ) -> tuple[replay.Batch, list[ClosedLoopSentence]]:
    """The flights of ``split`` on their closed-loop sentences at ``interval_s`` from their first predicted step (module
    docstring: `replay.batch_of`, `closed_loop.replay_batch`, after the start's refusals), and each one's stored
    sentence; refused when a flight has none."""
    batch = replay.batch_of(flights.drawn, list(range(len(flights.readings))), flights.readings, interval_s, words)
    # the set's sentences copied BEFORE the batch is built on them: its words grids would be views into the file
    owned = {index: owned_sentence(stored[index]) for index in batch.indices if index in stored}
    if len(owned) != len(flights.readings):
        raise ValueError(f"{len(flights.readings) - len(owned)} of the set's flights have no closed-loop sentence "
                         f"at {interval_s:g} s (refused on the interval: {batch.drawn['refused_on_interval']})")
    require_startable(instructions, split, interval_s, owned, dict(zip(flights.drawn.indices, flights.drawn.signals)),
                      flights.drawn.geometries, params, words)
    batch, _ = closed_loop.replay_batch(batch, owned, words)
    return batch, [owned[index] for index in batch.indices]


def fly_single(batch: replay.Batch, inputs: FlightInputs, j: int, params: ExecutorParams, words: Words, *,
               stop_cycle: int | None = None, superseded: Callable[[], bool] = lambda: False) -> tuple[Flown, bool]:
    """Flight ``j`` of ``batch`` (``inputs``: the batch's, `replay.Batch.inputs`) with the single-flight executor's one
    loop (`single.fly`) — its time limit and reserve — to its end, or stopped before cycle ``stop_cycle``."""
    limits, reserve = replay.time_limits_s(batch, params, words.spec.step_s), replay.reserve_s(batch)
    flight = FlightInputs(**{f.name: getattr(inputs, f.name)[j: j + 1] for f in dataclasses.fields(FlightInputs)})
    return single.fly(flight, batch.geometries[j], batch.approach_ias_mps[j], batch.sentences[j].grid, params, words,
                      step_s=batch.row_interval_s, time_limit_s=limits[j], reserve_s=reserve, stop_cycle=stop_cycle,
                      superseded=superseded)


def last_state_cycle(outcome: str, end_row: int) -> int:
    """The last flown state a flight is drawn to: the judge's outcome row, but for a dynamics failure the state before
    it (the failed state is left out). One rule for the export's ``replay.track`` and the live answer's track; the
    frontend mirrors it (`trainingSample.ts` `lastStateCycle`)."""
    return end_row - 1 if outcome == "dynamics_failure" else end_row


def crossing_payload(verdict: Verdict, flown: Flown, j: int, geometry: AirportGeometry) -> dict[str, Any] | None:
    """A verdict's threshold crossing as the Training view draws it (None: no crossing): where it crossed, at which
    cycle from the first predicted step and which candidate; for an approach crossing the decision-altitude check (D38)
    with the DA point's place and height (None: the judge wrote none — a crossing above the landing screen, or no DA
    point)."""
    if verdict.crossing is None:
        return None
    crossing = verdict.crossing
    out = {"crossM": round(crossing["cross_m"], 2), "heightM": round(crossing["height_m"], 2),
           "atCycle": round(crossing["at_row"], 3), "runwayIndex": int(crossing["runway_index"]), "decision": None}
    # the judge writes a decision-altitude check (None: no DA point) on the crossings it reads as approaches: those
    decision = crossing["decision"] if verdict.outcome in APPROACH_OUTCOMES else None
    if decision is not None:
        state = flown.states[j, decision["row"]].cpu().numpy()
        e, n = geometry.frame.horizontal_from_latlon(float(state[LAT]), float(state[LON]))
        out["decision"] = {
            "cycle": int(decision["row"]), "eM": round(float(e), 1), "nM": round(float(n), 1),
            "latDeg": round(float(state[LAT]), 7), "lonDeg": round(float(state[LON]), 7),
            "heightMslM": round(float(state[ALT]), 1), "rightM": round(decision["right_m"], 2),
            "coneHalfWidthM": round(decision["cone_half_width_m"], 2),
            "aboveGlidepathM": round(decision["above_glidepath_m"], 2), "lateralOk": bool(decision["lateral_ok"]),
            "verticalOk": bool(decision["vertical_ok"]), "passed": bool(decision["passed"])}
    return out
