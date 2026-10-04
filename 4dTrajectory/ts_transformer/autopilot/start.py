"""The start of a closed loop (vocabulary §6 item 5, D67; §12.1 A26): the executor at the first predicted step of each
flight, then flown a row at a time on the words a caller gives — the closed-loop reading's (`autopilot.closed_loop`) and
a speaker's (the prior's free generation, the post-training) one way in.

`start` takes closed-loop sentences of the artefact as the artefact's reader gives them (`instructions.artefact.
closed_loop_sentences`: a split and a row interval Δ), the executor spec's parameters and the most go-arounds a flight
may say. For each sentence's flight: the flight rebuilt from the harvest and compared row by row with the stored signals
(`flights.rebuild_series`, vocabulary §7.2 #4); its aircraft — its own dynamics or a stand-in's — and its approach speed,
by the rule of the replay (`replay.group_of`, `replay.flight_approach_ias_mps`); its state and physical context at the
first predicted step (`flights.flight_inputs`); its time limit (§5.8: the observed time left from the first predicted
step to the end of its labelled sentence × the spec's timeout factor, `time_limit_s`) and the time its go-arounds may add
(`GO_AROUND_EXTRA_S` each, up to the most given). The artefact holds the signals, not the dynamics (§7.2 #4), so the
start is rebuilt, never stored.

`Loop` is the executor so started (the closed-loop reading builds its own from the same pieces, `Loop.__init__`): at each
Δ row the caller gives every flight's words of the row (`Loop.step`, ``[B, 5]``; the first row every column); the
executor flies Δ seconds in its 1 s cycles and gives the states of the 2 s rows flown (`instructions.artefact.
STATE_COLUMNS`) and the flights done. A go-around word beyond the most given is refused by name (`GoAroundBeyondMost`):
the caller masks it. A flight's outcome is the judge's on what the executor recorded, no observed words read
(`Loop.outcome`, `judge.outcome_of`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S, Executor
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import AirportCharts, Kinematics
from ts_transformer.autopilot.judge import Outcome, outcome_of
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import Spoken
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import (
    ClosedLoopSentence, load_candidates, load_sentences, load_signals, load_spec,
)
from ts_transformer.instructions.labeller.interval import OBSERVATION_S, interval_rows
from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND, Words


class GoAroundBeyondMost(ValueError):
    """A flight said a go-around beyond the most its loop was started with (D67): the caller masks that word."""


def start_row(row_interval_s: float) -> int:
    """The first predicted step's Δ row: the prior's observation (16 s) after the sentence's first Δ row."""
    return int(round(OBSERVATION_S / row_interval_s))


def time_limit_s(observed_rows: int, first_row: int, params: ExecutorParams, step_s: float) -> float:
    """A flight's time limit before its go-arounds (vocabulary §5.8): the observed time from its 2 s row ``first_row`` to
    the end of its labelled sentence of ``observed_rows`` rows, × the timeout factor."""
    return (observed_rows - first_row) * step_s * params.timeout_factor


def state_rows(state: Kinematics) -> np.ndarray:
    """Every flight's state as `STATE_COLUMNS` rows ``[B, 6]``: position in the airport frame, MSL height, track, ground
    speed, vertical rate."""
    speed, gamma = state.speed_mps.cpu().numpy(), state.gamma_rad.cpu().numpy()
    return np.column_stack([state.e_m.cpu().numpy(), state.n_m.cpu().numpy(), state.height_m.cpu().numpy(),
                            state.track_deg.cpu().numpy(), state.ground_speed_mps.cpu().numpy(),
                            speed * np.sin(gamma)]).astype(np.float64)


class Loop:
    """Flights flown from their first predicted step a Δ row at a time (module docstring): ``inputs`` their state and
    physical context there, ``geometries`` their airports', ``approach_ias_mps`` their approach speeds,
    ``time_limits_s`` their time limits before the go-arounds, ``most_go_arounds`` the most go-arounds one may say."""

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
        (``[B, Δ / 2 s, 6]``, the last at the end of the row) and which flights are done. A go-around beyond the most
        given refuses the row by name, before anything is flown."""
        row = np.asarray(words_row, dtype=np.int64)
        go_around = row[:, RUNWAY] == RUNWAY_GO_AROUND
        beyond = np.flatnonzero(go_around & (self.go_arounds >= self.most_go_arounds))
        if len(beyond):
            raise GoAroundBeyondMost(f"flight(s) {beyond.tolist()} said a go-around beyond the most "
                                     f"{self.most_go_arounds} at row {self.steps}")
        self.go_arounds += go_around
        self.spoken.say(row)
        executor, cycle_s = self.executor, self.params.cycle_s
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

    def timed_out(self) -> np.ndarray:
        """``[B]`` bool: the flights done at their time limit (the executor's own test)."""
        executor = self.executor
        return ((executor.done_cycle.cpu().numpy() + 1) * self.params.cycle_s
                >= executor.time_limit_s.cpu().numpy()) & executor.done.cpu().numpy()

    def outcome(self, flight: int) -> Outcome:
        """Flight ``flight``'s outcome, read by the judge off what the executor recorded (`judge.outcome_of`)."""
        if self._flown is None:
            self._flown = self.executor.flown()
        return outcome_of(self._flown, flight, self.geometries[flight], self.words.spec)


def start(instructions: Path, split: str, interval_s: float, sentences: Mapping[int, ClosedLoopSentence],
          params: ExecutorParams, words: Words, *, most_go_arounds: int, device: torch.device) -> tuple[Loop, list[int]]:
    """The loop of the flights of ``sentences`` (keyed by their place in the artefact's ``split`` signals, as the reader
    gives them), in the order of their keys, and that order (module docstring). Refused unless ``words`` is the
    artefact's vocabulary and every sentence starts at Δ's first predicted step."""
    spec = words.spec
    if load_spec(instructions).sha256 != spec.sha256:
        raise ValueError(f"{instructions} holds another vocabulary than the words given")
    first = start_row(interval_s)
    every = interval_rows(interval_s, spec.step_s)
    order = sorted(sentences)
    if any(sentences[i].start != first for i in order):
        raise ValueError(f"a sentence does not start at row {first} of {interval_s:g} s")
    signals = load_signals(instructions, split)
    geometries = load_candidates(instructions)
    flights = [signals[i] for i in order]
    series = rebuild_series(instructions, flights)
    groups = [replay.group_of(item) for item in series]
    unflown = [f.dataset_id for f, g in zip(flights, groups) if g not in (replay.OWN, replay.STAND_IN)]
    if unflown:
        raise ValueError(f"{len(unflown)} flight(s) have no aircraft to fly, e.g. {unflown[:3]}")
    labelled = load_sentences(instructions, split, spec)
    rows = dict(zip(labelled["signal_index"].tolist(), np.diff(labelled["offsets"]).tolist()))
    anchors = [sentences[i].first_row + first * every for i in order]
    loop = Loop(flight_inputs(series, anchors, device=device), [geometries[f.airport] for f in flights],
                [replay.flight_approach_ias_mps(s, g) for s, g in zip(series, groups)],
                [time_limit_s(int(rows[i]), anchor, params, spec.step_s) for i, anchor in zip(order, anchors)],
                params, words, interval_s=interval_s, most_go_arounds=most_go_arounds, device=device)
    return loop, order

