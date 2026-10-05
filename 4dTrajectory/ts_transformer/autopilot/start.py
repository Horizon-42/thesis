"""The start of a closed loop (vocabulary §6 item 5, D67; §12.1 A26): the executor at the first predicted step of each
flight, then flown a row at a time on the words a caller gives — the closed-loop reading's (`autopilot.closed_loop`) and
a speaker's (the prior's free generation, the post-training) one way in.

`start` takes closed-loop sentences of the artefact as the artefact's reader gives them (`instructions.artefact.
closed_loop_sentences`: a split and a row interval Δ), the directory of the executor spec and the most go-arounds a flight
may say. It opens the spec itself as the replay and the backend do (`replay.open_executor`, D71: the spec measured against
the artefact's vocabulary, the labeller and the executor conformance), so no caller handles executor parameters; the
words are the artefact's vocabulary's. For each sentence's flight: the flight rebuilt from the harvest and compared row by row with the stored signals
(`flights.rebuild_series`, vocabulary §7.2 #4); its aircraft — its own dynamics or a stand-in's — and its approach speed,
by the rule of the replay (`replay.group_of`, `replay.flight_approach_ias_mps`); its state and physical context at the
first predicted step (`flights.flight_inputs`); its time limit (§5.8: the observed time left from the first predicted
step to the end of its labelled sentence × the spec's timeout factor, `replay.time_limit_s`) and the time its go-arounds
may add (`GO_AROUND_EXTRA_S` each, up to the most given). The sentences are tied to what they were read under: each
sentence's observed rows (before its first predicted step) must be its flight's stored signals there, and the artefact's
closed-loop file of the split and Δ must have been flown by the spec's parameters. The artefact holds the signals, not the dynamics (§7.2 #4), so the
start is rebuilt, never stored.

`Loop` is the executor so started (the closed-loop reading builds its own from the same pieces, `Loop.__init__`): at each
Δ row the caller gives every flight's words of the row (`Loop.step`, ``[B, 5]``; the first row every column); the
executor flies Δ seconds in its 1 s cycles and gives the states of the 2 s rows flown (`instructions.artefact.
STATE_COLUMNS`) and the flights done — done where the judge ends a flight (D79). Before anything changes, the row of each
flight still flying is checked (D80): with the grammar (`instructions.grammar.apply`, at the height above the airport
elevation E of the executor's state; a word outside its column refused too) and against the most go-arounds given; a row
refused refuses the whole step by name (`RowRefused`, `GoAroundBeyondMost`) and leaves the loop as it was: the caller
masks such words (a done or halted flight's words are not heard). A done flight's outcome is the judge's on what the
executor recorded, no observed words read (`Loop.outcome`, `judge.outcome_of`).

The start state is the observed row's, by the executor spec's start rule (`flights.start_state`, D77): it reads no sample
after the first predicted step; the observed rows a sentence stores before it are `flights.observed_rows`, by that rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S, Executor
from ts_transformer.autopilot.flights import FlightInputs, compass_track, flight_inputs, observed_rows, rebuild_series
from ts_transformer.autopilot.frame import AirportCharts, Kinematics
from ts_transformer.autopilot.judge import Outcome, outcome_of
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import Spoken
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    CLOSED_LOOP_SCHEMA, ClosedLoopSentence, closed_loop_path, load_candidates, load_sentences, load_signals,
)
from ts_transformer.instructions.grammar import InForce, Ungrammatical, apply, require_values
from ts_transformer.instructions.labeller.interval import OBSERVATION_S, interval_rows
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, Words


class GoAroundBeyondMost(ValueError):
    """A flight said a go-around beyond the most its loop was started with (D67): the caller masks that word."""


class RowRefused(ValueError):
    """A flight's row that the grammar refuses (D80; ``reason`` the grammar's, or a word outside its column): the caller
    masks it."""

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def start_row(row_interval_s: float) -> int:
    """The first predicted step's Δ row: the prior's observation (16 s) after the sentence's first Δ row."""
    return int(round(OBSERVATION_S / row_interval_s))


def state_rows(state: Kinematics) -> np.ndarray:
    """Every flight's state as `STATE_COLUMNS` rows ``[B, 6]``: position in the airport frame, MSL height, track, ground
    speed, vertical rate."""
    speed, gamma = state.speed_mps.cpu().numpy(), state.gamma_rad.cpu().numpy()
    return np.column_stack([state.e_m.cpu().numpy(), state.n_m.cpu().numpy(), state.height_m.cpu().numpy(),
                            compass_track(state.track_deg.cpu().numpy()), state.ground_speed_mps.cpu().numpy(),
                            speed * np.sin(gamma)]).astype(np.float64)


class Loop:
    """Flights flown from their first predicted step a Δ row at a time (module docstring): ``inputs`` their state and
    physical context there, ``geometries`` their airports', ``approach_ias_mps`` their approach speeds,
    ``time_limits_s`` their time limits before the go-arounds, ``most_go_arounds`` the most go-arounds one may say.
    A caller reads the states of the rows flown and which flights are done (`step`), and why a flight ended from the
    judge's outcome (`outcome`; the loop has no ``timed_out()``, D90: the time limit is a function of the observed
    landing time). ``executor`` is public: a caller reads its end cycle (``done_cycle``), its flown record (``flown()``)
    and the aero parameters (``inputs.aero_params``) (vocabulary §6 item 5)."""

    def __init__(self, inputs: FlightInputs, geometries: Sequence[AirportGeometry], approach_ias_mps: Sequence[float],
                 time_limits_s: Sequence[float], params: ExecutorParams, words: Words, *, interval_s: float,
                 most_go_arounds: int, device: torch.device) -> None:
        if most_go_arounds < 0:
            raise ValueError("the most go-arounds of a flight is negative")
        f64 = torch.float64
        self.params, self.words, self.interval_s = params, words, interval_s
        self.geometries = list(geometries)
        self.most_go_arounds = most_go_arounds
        self.executor = Executor(
            inputs, Runways.of(self.geometries, words.spec, dtype=f64, device=device),
            AirportCharts.of(self.geometries, dtype=f64, device=device),
            torch.tensor(list(approach_ias_mps), dtype=f64, device=device), params, words, step_s=interval_s,
            time_limit_s=torch.tensor(list(time_limits_s), dtype=f64, device=device),
            reserve_s=GO_AROUND_EXTRA_S * most_go_arounds)
        self.spoken = Spoken(len(self.geometries), words, step_s=interval_s, device=device)
        self.go_arounds = np.zeros(len(self.geometries), dtype=np.int64)
        #: each flight's words in force by the grammar (None before its first row, D80)
        self.grammar: list[InForce | None] = [None] * len(self.geometries)
        self.step_cycles = int(round(interval_s / params.cycle_s))
        self.row_cycles = int(round(words.spec.step_s / params.cycle_s))      # the cycles of one 2 s row
        if self.step_cycles % self.row_cycles:
            raise ValueError(f"a row interval of {interval_s:g} s is no whole number of {words.spec.step_s:g} s rows")
        self.steps = 0
        self._flown = None

    def rows(self) -> np.ndarray:
        """Every flight's state now (at the start of the next row) as `STATE_COLUMNS` rows, ``[B, 6]``."""
        return state_rows(self.executor.now())

    def step(self, words_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Every flight's words of the next Δ row (``[B, 5]``) flown for Δ seconds: the states of the 2 s rows flown
        (``[B, Δ / 2 s, 6]``, the last at the end of the row) and which flights are done. A row the grammar refuses
        (`RowRefused`) or a go-around beyond the most given (`GoAroundBeyondMost`) refuses the step by name, before
        anything changes (module docstring)."""
        row = np.asarray(words_row, dtype=np.int64)
        executor, cycle_s = self.executor, self.params.cycle_s
        if row.shape != (len(self.geometries), len(COLUMNS)):
            raise ValueError(f"a step is [{len(self.geometries)}, {len(COLUMNS)}] words, got {list(row.shape)}")
        flying = ~(executor.done | executor.halted).cpu().numpy()     # a done or halted flight's words are not heard
        go_around = (row[:, RUNWAY] == RUNWAY_GO_AROUND) & flying
        beyond = np.flatnonzero(go_around & (self.go_arounds >= self.most_go_arounds))
        if len(beyond):
            raise GoAroundBeyondMost(f"flight(s) {beyond.tolist()} said a go-around beyond the most "
                                     f"{self.most_go_arounds} at row {self.steps}")
        for f in range(len(row)):                       # every flight's row, a done or halted one's too (D80)
            try:
                require_values(row[f], self.words, len(self.geometries[f].candidates))
            except ValueError as error:
                raise RowRefused(f"flight {f} at row {self.steps}: {error}", "word outside its column") from None
        height = executor.now().height_m.cpu().numpy()
        grammar = list(self.grammar)
        for f in np.flatnonzero(flying):
            geometry = self.geometries[f]
            try:
                grammar[f] = apply(grammar[f], row[f], float(height[f]) - geometry.elevation_m, self.words,
                                   len(geometry.candidates))
            except Ungrammatical as error:
                raise RowRefused(f"flight {f} at row {self.steps}: {error}", error.reason) from None
        self.spoken.say(row)                              # validates the row before it changes anything
        self.grammar = grammar
        self.go_arounds += go_around
        heard = torch.full((len(row),), self.steps * self.interval_s, dtype=torch.float64,
                           device=executor.state.device)
        flown = []
        for cycle in range(1, self.step_cycles + 1):
            executor.cycle(self.spoken.at(heard), torch.full_like(heard, executor.count * cycle_s))
            if cycle % self.row_cycles == 0:
                flown.append(self.rows())
        self.steps += 1
        self._flown = None
        return np.stack(flown, axis=1), executor.done.cpu().numpy()

    def halt(self, flights: np.ndarray) -> None:
        """Hold ``flights`` (``[B]`` bool) from the next cycle on (`Executor.halt`)."""
        self.executor.halt(torch.as_tensor(flights, device=self.executor.state.device))

    def outcome(self, flight: int) -> Outcome:
        """Flight ``flight``'s outcome, read by the judge off what the executor recorded (`judge.outcome_of`); refused for a
        flight not done (a halted flight is never done)."""
        if not bool(self.executor.done[flight]):
            raise ValueError(f"flight {flight} is not done: it has no outcome yet")
        if self._flown is None:
            self._flown = self.executor.flown()
        return outcome_of(self._flown, flight, self.geometries[flight], self.words.spec)


def require_startable(instructions: Path, split: str, interval_s: float, sentences: Mapping[int, ClosedLoopSentence],
                      signals: Mapping[int, FlightSignals], geometries: Mapping[str, AirportGeometry],
                      params: ExecutorParams, words: Words) -> list[FlightSignals]:
    """The start's refusals (module docstring), for every caller that starts stored closed-loop sentences — `start`, and
    the Training export and the live executor (`experiments.training_flights`): every sentence starts at Δ's first
    predicted step, the artefact's closed-loop sentences of ``split`` at Δ were flown by the parameters ``params``, and
    each sentence's observed rows are its flight's stored signals there (``signals``: the split's, by their place in it;
    ``geometries``: the artefact's candidates), by the start rule (D77). The flights of ``sentences`` (keyed by their
    place in the split's signals), in the order of their keys."""
    if not sentences:
        raise ValueError("no sentence to start")
    first = start_row(interval_s)
    every = interval_rows(interval_s, words.spec.step_s)
    order = sorted(sentences)
    if any(sentences[i].rows.start != first for i in order):
        raise ValueError(f"a sentence does not start at row {first} of {interval_s:g} s")
    path = closed_loop_path(instructions, split, interval_s)
    with np.load(path) as data:
        schema, flown_by = str(data["schema"]), str(data["executor_params_sha256"])
    if schema != CLOSED_LOOP_SCHEMA:
        raise ValueError(f"{path} is not a {CLOSED_LOOP_SCHEMA} file")
    if flown_by != params_sha256(params):
        raise ValueError(f"the {split} closed-loop sentences at {interval_s:g} s were flown by executor parameters "
                         f"{flown_by[:12]}, not {params_sha256(params)[:12]}")
    flights = [signals[i] for i in order]
    for i, flight in zip(order, flights):
        rows = sentences[i].rows.first_row + np.arange(first * every)
        observed = observed_rows(flight, rows, params.start_rule, geometries[flight.airport])
        if not np.array_equal(sentences[i].rows.states[: first * every], observed):
            raise ValueError(f"the sentence of {split} flight {i} is not {flight.dataset_id}'s: its observed rows differ")
    return flights


def start(instructions: Path, split: str, interval_s: float, sentences: Mapping[int, ClosedLoopSentence],
          executor: Path, *, most_go_arounds: int, device: torch.device) -> tuple[Loop, list[int]]:
    """The loop of the flights of ``sentences`` (keyed by their place in the artefact's ``split`` signals, as the reader
    gives them), in the order of their keys, and that order (module docstring), flown by the executor spec in the
    directory ``executor``. Refused unless the spec opens for this artefact (`replay.open_executor`) and the sentences
    may be started (`require_startable`)."""
    params, _, words = replay.open_executor(executor, instructions)
    spec = words.spec
    first = start_row(interval_s)
    every = interval_rows(interval_s, spec.step_s)
    order = sorted(sentences)
    geometries = load_candidates(instructions)
    flights = require_startable(instructions, split, interval_s, sentences, load_signals(instructions, split), geometries,
                                params, words)
    series = rebuild_series(instructions, flights)
    groups = [replay.group_of(item) for item in series]
    unflown = [f.dataset_id for f, g in zip(flights, groups) if g not in (replay.OWN, replay.STAND_IN)]
    if unflown:
        raise ValueError(f"{len(unflown)} flight(s) have no aircraft to fly, e.g. {unflown[:3]}")
    labelled = load_sentences(instructions, split, spec, ("signal_index", "offsets"))
    lengths = dict(zip(labelled["signal_index"].tolist(), np.diff(labelled["offsets"]).tolist()))
    anchors = [sentences[i].rows.first_row + first * every for i in order]
    airports = [geometries[f.airport] for f in flights]
    loop = Loop(flight_inputs(series, flights, anchors, airports, params.start_rule, device=device), airports,
                [replay.flight_approach_ias_mps(s, g) for s, g in zip(series, groups)],
                [replay.time_limit_s(int(lengths[i]), anchor, params, spec.step_s) for i, anchor in zip(order, anchors)],
                params, words, interval_s=interval_s, most_go_arounds=most_go_arounds, device=device)
    return loop, order
