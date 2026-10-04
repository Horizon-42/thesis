"""The closed-loop reading (design §4.9, D32): the labeller's second pass. It flies each open-loop sentence (the observed
words, `instructions.labeller`) with the executor and adds CORRECTION WORDS where the flown path leaves the observed
one; the prior trains on the flown states of these sentences (§6.1).

THE FLIGHT. A batch of sentences on one row interval Δ (`replay.batch_of`) is flown from the FIRST PREDICTED STEP, Δ row
``OBSERVATION_S / Δ`` (16 s after the sentence's first Δ row), from the observed state there; the rows before it stay
observed and say nothing. The words are said a step at a time, as in free generation (`sentence.Spoken`): at each Δ row
the reading finds the matched point, decides the row's words — the observed words the matched point has reached, with
the corrections — and the executor hears them on the cycle that starts the row (the sentence's own clock: a row is Δ
seconds). The flight ends at the end of the cycle in which the executor is done (a crossing, the ground, the dynamics,
or the replay's time limit: the remaining observed time × the timeout factor, `replay.time_limits_s`, and each
go-around's extra time), not at the open-loop sentence's last row: the closed-loop sentence has the flown rows (§4.9
item 6). A sentence that is in a go-around at its first predicted step, or has fewer than two rows from there, is
refused.

WHEN THE OBSERVED WORDS ARE SAID (D42). The observed time of the matched point is the time at which the observed
aircraft was there (`ObservedPath.match`). At each Δ row the reading says the observed words of every open-loop Δ row
whose time is not later than it and that were not said before; of several such rows, each column's last word
(`last_words`). While the flown aircraft is behind the observed one the observed words wait; ahead of it, they come
sooner. The first predicted step says every column (rule 1): the observed words in force there. A heading word says a
track relative to the course of the runway in force when it is heard (§3.3): a row whose heading word would be heard
under another course than the observed word in force was said under (a runway word passed with it or before it) is
refused, as the row interval refuses it (`labeller.interval` item 4).

THE COMPARISON (`ObservedPath.match`). The observed path is the observed flight at the data's 2 s rows, its positions
as observed and its height and track as the labeller reads them (`labeller.read.smooth`). The matched point is the point
of the path nearest the flown position, searched forward from the row before's: the matched segment advances while the
next segment is no farther from the flown position, so a path that crosses itself does not jump. Past the path's end (an
observed slice stops short of the threshold, the flown aircraft flies on to it) the path goes on along its last segment's
line, its height on that segment's slope (Claude's reading of §4.9 beyond the observed path: without it the frozen last
height reads the last few hundred metres of a descent as "too low"). At the matched point the lateral error
e_y is the flown position's distance from the path perpendicular to the observed track there (right positive), the
vertical error e_h the flown height minus the observed height. A difference along the path (in time) is not corrected,
so the speed words stay the observed ones.

THE CORRECTIONS (`Corrector`), with Y = `closed_loop_lateral_m` and H = `closed_loop_vertical_m` of the spec:

- lateral: when |e_y| > Y and the row says no new observed heading word, the heading class one step (5°)
  from the observed word in force, toward the path; the observed word again when |e_y| < Y / 2 or e_y changes sign; a
  new observed heading word ends a correction (it is said, and the comparison goes on from the next row);
- vertical: only while a descent class of the observed words is in force — when e_h > H the next steeper descent class,
  when e_h < −H the next shallower (none beyond descent 4 or descent 1); the observed class again when |e_h| < H / 2 or
  e_h changes sign; a new observed altitude or angle word ends a correction. A level hold and a climb get none. A
  level reached by a descent says no angle word (the descent class stays in force, `labeller.vertical`), so a LEVEL
  HOLD is the executor's: the level in force captured (`vertical.Vertical.captured`, the level-off begun) — there no
  correction starts and one in force ends (Claude's reading of §4.9 "during a level hold"; the rounding of the level to
  the grid is not corrected).

A correction word is an ordinary word of its column: the grammar (`instructions.grammar`) reads every row at the flown
height, and a row it refuses refuses the flight (counted by reason). A word equal to the one in force is not said (the
sentence's own rule, `labeller.sentence`). The capture, the runway words and the go-around rows are the open-loop
reading's; the executor reads only words.

THE RESULT (`ClosedLoopSentence`): the said words from the first predicted step (each word the reading added marked a
correction: every word that is not an observed word said at its row), the states on every Δ row from the sentence's
first — observed before the first predicted step, flown from it (position in the airport frame, MSL height, track,
ground speed, vertical rate) — e_y, e_h at each flown row, and whether the flight was done at its time limit. Flown
again from the same state on its own clock (the replay: `replay_batch`, the time clock, under the same time limit), a
closed-loop sentence gives the same states.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.conformance import DEVICE, ROUNDOFF, STATE_BOUND_M
from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S, Executor
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs
from ts_transformer.autopilot.frame import AirportCharts, Kinematics
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.replay import Batch, subset
from ts_transformer.autopilot.sentence import Spoken
from ts_transformer.autopilot.spec import executor_source_files, params_sha256
from ts_transformer.instructions.artefact import CLOSED_LOOP_DIRECTORY
from ts_transformer.instructions.conformance import labeller_code_files
from ts_transformer.instructions.grammar import InForce, Ungrammatical, apply
from ts_transformer.instructions.labeller.interval import OBSERVATION_S, in_force, interval_rows
from ts_transformer.instructions.labeller.read import smooth, truncated
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import ALTITUDE, ANGLE, COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, logic_sha256, utc_now, write_json_atomic
from ts_transformer.repo_layout import git_state

#: The columns of a row's state: the airport frame's east and north, MSL height, compass track, ground speed, vertical
#: rate (SI).
STATE_COLUMNS = ("e_m", "n_m", "height_m", "track_deg", "ground_speed_mps", "vertical_rate_mps")


def start_row(row_interval_s: float) -> int:
    """The first predicted step's Δ row: the prior's observation (16 s) after the sentence's first Δ row."""
    return int(round(OBSERVATION_S / row_interval_s))


@dataclass(frozen=True)
class ClosedLoopSentence:
    """One flight's closed-loop sentence on its row interval (module docstring)."""

    grid: np.ndarray            # [M, 5] int16: the words said from the first predicted step (row 0: every column)
    correction: np.ndarray      # [M, 5] bool: a word the closed-loop reading added
    states: np.ndarray          # [start + M, 6] `STATE_COLUMNS` on every Δ row from the sentence's first
    lateral_m: np.ndarray       # [M] e_y at each said row
    vertical_m: np.ndarray      # [M] e_h at each said row
    #: [M, 2] bool: the rows where §4.9 makes no heading / angle correction (`Corrector.row`; the readings of D34)
    uncorrectable: np.ndarray
    #: [M] the open-loop Δ row each said row's observed words reach (D42: the last whose time is not later than the
    #: matched point's; its words and every earlier row's have been said)
    observed_row: np.ndarray
    start: int                  # the first predicted step's Δ row (`start_row`)
    timed_out: bool             # the executor was done in the cycle that reached its time limit


class Match(NamedTuple):
    """A flown position against the observed path (`ObservedPath.match`)."""

    lateral_m: float            # e_y, right of the observed track positive
    vertical_m: float           # e_h, the flown height minus the observed one
    row: float                  # the matched point's observed time, in the path's rows from its first (D42)
    along_m: float              # the matched point's distance along the path from its first row


class ObservedPath:
    """An observed flight's path at the data's rows, and the forward search of the matched point (module docstring)."""

    def __init__(self, e_m: np.ndarray, n_m: np.ndarray, height_m: np.ndarray, first: int) -> None:
        e_m, n_m, height_m = (np.asarray(a, dtype=np.float64) for a in (e_m, n_m, height_m))
        # a row at the position of the one before is no segment: dropped (the matched segment counts the rows kept)
        moved = np.concatenate(([True], (np.diff(e_m) != 0.0) | (np.diff(n_m) != 0.0)))
        if moved.sum() < 2:
            raise ValueError("a path needs two positions")
        self.e, self.n, self.height = e_m[moved], n_m[moved], height_m[moved]
        # the observed times: a segment runs from the last row at its first point (when the observed aircraft left it)
        # to the first row at its second (when it arrived); past the path's end, its last row
        self.arrived = np.flatnonzero(moved)
        self.left = np.append(self.arrived[1:] - 1, len(e_m) - 1)
        #: the distance along the path at every row (a held position adds nothing)
        self.along_rows = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(e_m), np.diff(n_m)))))
        self.segment = min(int(moved[: first + 1].sum()) - 1, len(self.e) - 2)

    def _nearest(self, i: int, e_m: float, n_m: float) -> tuple[float, float]:
        """``(t, distance)``: the point of segment ``i`` nearest the position (``t`` along it, 0 to 1)."""
        de, dn = self.e[i + 1] - self.e[i], self.n[i + 1] - self.n[i]
        t = min(max(((e_m - self.e[i]) * de + (n_m - self.n[i]) * dn) / (de * de + dn * dn), 0.0), 1.0)
        return t, math.hypot(e_m - self.e[i] - t * de, n_m - self.n[i] - t * dn)

    def match(self, e_m: float, n_m: float, height_m: float) -> Match:
        """A flown position and height against the path (module docstring): the matched segment moves forward while the
        next one is no farther from the position, never back."""
        t, distance = self._nearest(self.segment, e_m, n_m)
        while self.segment + 2 < len(self.e):
            t_next, next_distance = self._nearest(self.segment + 1, e_m, n_m)
            if next_distance > distance:
                break
            self.segment, t, distance = self.segment + 1, t_next, next_distance
        i = self.segment
        de, dn = self.e[i + 1] - self.e[i], self.n[i + 1] - self.n[i]
        lateral = ((e_m - self.e[i]) * dn - (n_m - self.n[i]) * de) / math.hypot(de, dn)   # right of the track positive
        last = i == len(self.e) - 2 and t >= 1.0
        row = float(self.left[-1]) if last else float(self.left[i] + t * (self.arrived[i + 1] - self.left[i]))
        if last:                                   # past the end: on along the last segment's line (module docstring)
            t = ((e_m - self.e[i]) * de + (n_m - self.n[i]) * dn) / (de * de + dn * dn)
        along = float(self.along_rows[self.arrived[i]]) + t * math.hypot(de, dn)
        return Match(lateral, height_m - float(self.height[i] + t * (self.height[i + 1] - self.height[i])), row, along)


def reached_row(row: float, every: int) -> int:
    """The last open-loop Δ row (``every`` data rows apart) whose observed time is not later than the matched point's
    observed time ``row`` (in data rows, D42)."""
    return int(math.floor(row / every))


def uncorrected_m(errors: np.ndarray, uncorrectable: np.ndarray) -> float:
    """The largest |error| on the rows where §4.9 makes no correction (D34; 0 where there is none)."""
    return float(np.abs(errors[uncorrectable]).max(initial=0.0))


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1


def last_words(rows: np.ndarray) -> np.ndarray:
    """Each column's last word in ``rows`` (``[n, 5]``; `UNCHANGED` where a column says none, or ``n`` is 0): what one
    said row says of the open-loop rows the matched point passed in it (D42)."""
    out = np.full(len(COLUMNS), UNCHANGED, dtype=np.int64)
    written = rows != UNCHANGED
    for column in np.flatnonzero(written.any(axis=0)):
        out[column] = rows[np.flatnonzero(written[:, column])[-1], column]
    return out


class Corrector:
    """One flight's words, row by row (module docstring): the open-loop sentence on its Δ rows (``observed``), how far
    it has been said, the corrections in force and the grammar's state of the said sentence; ``courses_deg`` the
    candidates' courses."""

    def __init__(self, observed: np.ndarray, start: int, words: Words, courses_deg: Sequence[float]) -> None:
        self.observed, self.held, self.start = observed, in_force(observed), start
        self.words, self.courses_deg = words, list(courses_deg)
        rows = np.arange(len(observed))
        # each open-loop row's runway (its last runway word, the go-around aside) and the row its heading word in force
        # was said at: the course a heading word is relative to (§3.3)
        self.runway = observed[np.maximum.accumulate(np.where(observed[:, RUNWAY] >= 0, rows, 0)), RUNWAY]
        self.heading_row = np.maximum.accumulate(np.where(observed[:, HEADING] != UNCHANGED, rows, 0))
        spec = words.spec
        self.lateral_m, self.vertical_m = spec.closed_loop_lateral_m, spec.closed_loop_vertical_m
        self.next = start        # the first open-loop row not said yet
        self.rows_said = 0
        self.turn = 0            # a heading correction in force: the sign of e_y it answers (0: none)
        self.slope = 0           # an angle correction in force: the sign of e_h it answers (0: none)
        self.said: np.ndarray | None = None     # the said sentence's words in force
        self.grammar: InForce | None = None

    def row(self, reached: int, lateral_m: float, vertical_m: float, height_m: float, *,
            holding: bool) -> tuple[np.ndarray, np.ndarray]:
        """The next said row's words and which are corrections: the observed words of the open-loop rows up to
        ``reached`` (the last whose time is not later than the matched point's; the first predicted step's at the
        least) not said before, and the corrections from the errors, the flown height there and whether the executor
        holds the level in force (``holding``, module docstring); refused when the grammar refuses the row or a heading
        word would be heard under another course than it was said under."""
        first = self.said is None
        last = min(max(reached, self.start) if first else reached, len(self.observed) - 1)
        observed = last_words(self.observed[self.next: last + 1])
        self.next = max(self.next, last + 1)
        self.observed_row = self.next - 1
        held = self.held[self.observed_row]
        wanted = held.astype(np.int64).copy()
        #: rows where §4.9 makes no correction of each column (the readings of D34): the first predicted step, a row
        #: that says a new observed word; vertically also a level hold, no descent class, no class beyond it
        self.uncorrectable = np.ones(2, dtype=bool)
        if not first:
            self.uncorrectable[0] = observed[HEADING] != UNCHANGED
            if observed[HEADING] != UNCHANGED:
                self.turn = 0
            elif self.turn and (abs(lateral_m) < self.lateral_m / 2 or _sign(lateral_m) != self.turn):
                self.turn = 0
            elif not self.turn and abs(lateral_m) > self.lateral_m:
                self.turn = _sign(lateral_m)
            if self.turn:                                  # right of the path: one class to the left
                wanted[HEADING] = (int(held[HEADING]) - self.turn) % self.words.n_heading
            angle = int(held[ANGLE])
            self.uncorrectable[1] = (observed[ALTITUDE] != UNCHANGED or observed[ANGLE] != UNCHANGED or holding
                                     or not self.words.is_descent(angle)
                                     or (abs(vertical_m) > self.vertical_m
                                         and not self.words.is_descent(angle + _sign(vertical_m))))
            if observed[ALTITUDE] != UNCHANGED or observed[ANGLE] != UNCHANGED or holding:
                self.slope = 0
            elif self.slope and (abs(vertical_m) < self.vertical_m / 2 or _sign(vertical_m) != self.slope):
                self.slope = 0
            elif (not self.slope and self.words.is_descent(angle) and abs(vertical_m) > self.vertical_m
                  and self.words.is_descent(angle + _sign(vertical_m))):
                self.slope = _sign(vertical_m)
            if self.slope:                                 # too high: the next steeper class
                wanted[ANGLE] = angle + self.slope
        said = wanted if first else np.where(wanted != self.said, wanted, UNCHANGED)
        k, self.rows_said = self.rows_said, self.rows_said + 1
        if said[HEADING] != UNCHANGED:
            heard = self.courses_deg[self.runway[self.observed_row]]
            meant = self.courses_deg[self.runway[self.heading_row[self.observed_row]]]
            if abs((heard - meant + 180.0) % 360.0 - 180.0) > 1e-9:
                raise Refused("closed loop: heading word across a runway change",
                              f"said row {k}: heard under the course {heard:.1f}°, said under {meant:.1f}°")
        try:
            self.grammar = apply(self.grammar, said, height_m, self.words, len(self.courses_deg))
        except Ungrammatical as error:
            raise Refused(f"closed loop: {error.reason}", f"said row {k}: {error.detail}") from None
        self.said = wanted
        added = (said != UNCHANGED) & ~((observed != UNCHANGED) & (observed == said))
        return said.astype(np.int16), np.zeros(len(COLUMNS), dtype=bool) if first else added


def _state_rows(signals: FlightSignals, rows: np.ndarray) -> np.ndarray:
    """The observed flight's `STATE_COLUMNS` at its 2 s ``rows``."""
    return np.column_stack([signals.e_m[rows], signals.n_m[rows], signals.altitude_m[rows], signals.track_deg[rows],
                            signals.ground_speed_mps[rows], signals.vertical_rate_mps[rows]])


def _flown_row(state: Kinematics, j: int) -> np.ndarray:
    speed, gamma = float(state.speed_mps[j]), float(state.gamma_rad[j])
    return np.array([float(state.e_m[j]), float(state.n_m[j]), float(state.height_m[j]), float(state.track_deg[j]),
                     float(state.ground_speed_mps[j]), speed * math.sin(gamma)])


def start_inputs(batch: Batch, step_s: float, *, device: torch.device) -> FlightInputs:
    """Every flight's physical context at its first predicted step (`flights.flight_inputs` at that 2 s row)."""
    every, start = interval_rows(batch.row_interval_s, step_s), start_row(batch.row_interval_s)
    return flight_inputs(batch.series, [s.first_row + start * every for s in batch.sentences], device=device)


def _rows(inputs: FlightInputs, flights: list[int]) -> FlightInputs:
    return FlightInputs(*(getattr(inputs, name)[flights] for name in
                          ("initial_state", "aero_params", "frame_params", "max_thrust_n")))


def refusal(sentence: replay.Sentence, row_interval_s: float) -> Refused | None:
    """Why a sentence cannot be read in closed loop before it is flown, or None (module docstring)."""
    start = start_row(row_interval_s)
    if len(sentence.grid) - start < 2:
        return Refused("too short for the closed loop", f"{len(sentence.grid)} rows of {row_interval_s:g} s")
    if in_force(sentence.grid)[start, RUNWAY] == RUNWAY_GO_AROUND:
        return Refused("go-around at the first predicted step", f"row {start} of {row_interval_s:g} s")
    return None


def read(batch: Batch, inputs: FlightInputs, params: ExecutorParams, words: Words, *,
         device: torch.device) -> list[ClosedLoopSentence | Refused]:
    """Every flight of ``batch`` read in closed loop (module docstring), in the batch's order: its sentence, or why it
    has none; ``inputs`` every flight's physical context at its first predicted step (`start_inputs`; any rows for the
    flights `refusal` refuses, which are not flown)."""
    spec, interval = words.spec, batch.row_interval_s
    every, start = interval_rows(interval, spec.step_s), start_row(interval)
    out: list[ClosedLoopSentence | Refused | None] = [refusal(s, interval) for s in batch.sentences]
    flying = [j for j, result in enumerate(out) if result is None]
    if not flying:
        return out  # type: ignore[return-value]
    paths, correctors, said, added, blocked, reached, errors, states, limits = [], [], [], [], [], [], [], [], []
    for j in flying:
        signals, reading, sentence = batch.signals[j], batch.readings[j], batch.sentences[j]
        span = len(reading.words) - sentence.first_row          # the observed rows the sentence covers
        smoothed = smooth(truncated(signals, span), spec)
        paths.append(ObservedPath(signals.e_m[:span], signals.n_m[:span], smoothed.altitude_m, start * every))
        correctors.append(Corrector(sentence.grid, start, words,
                                    [candidate.course_deg for candidate in batch.geometries[j].candidates]))
        limits.append(replay.remaining_observed_s(reading, sentence.first_row + start * every, spec.step_s)
                      * params.timeout_factor)
        observed = np.arange(start) * every
        states.append([*_state_rows(signals, observed)])
        said.append([])
        added.append([])
        blocked.append([])
        reached.append([])
        errors.append([])
    f64 = torch.float64
    go_arounds = max(int((batch.sentences[j].grid[start:, RUNWAY] == RUNWAY_GO_AROUND).sum()) for j in flying)
    executor = Executor(
        _rows(inputs, flying),
        Runways.of([batch.geometries[j] for j in flying], spec, dtype=f64, device=device),
        AirportCharts.of([batch.geometries[j] for j in flying], dtype=f64, device=device),
        torch.tensor([batch.approach_ias_mps[j] for j in flying], dtype=f64, device=device), params, words,
        step_s=interval, time_limit_s=torch.tensor(limits, dtype=f64, device=device),
        reserve_s=GO_AROUND_EXTRA_S * go_arounds)
    spoken = Spoken(len(flying), words, step_s=interval, device=device)
    step_cycles = int(round(interval / params.cycle_s))
    live = np.ones(len(flying), dtype=bool)
    timed_out = np.zeros(len(flying), dtype=bool)
    s = 0
    while live.any():
        now = executor.now()
        captured = executor.vertical.captured.cpu().numpy()
        step = np.full((len(flying), len(COLUMNS)), UNCHANGED, dtype=np.int64)
        for f, j in enumerate(flying):
            if not live[f]:
                continue
            flown = _flown_row(now, f)
            match = paths[f].match(flown[0], flown[1], flown[2])
            lateral, vertical = match.lateral_m, match.vertical_m
            try:
                words_row, mask = correctors[f].row(reached_row(match.row, every), lateral, vertical, flown[2],
                                                    holding=bool(captured[f]))
            except Refused as refused:
                out[j] = refused
                live[f] = False
                executor.halt(torch.as_tensor(~live, device=device))
                if s == 0:
                    step[f] = in_force(batch.sentences[j].grid)[start]
                continue
            step[f] = words_row
            said[f].append(words_row)
            added[f].append(mask)
            blocked[f].append(correctors[f].uncorrectable)
            reached[f].append(correctors[f].observed_row)
            errors[f].append((lateral, vertical))
            states[f].append(flown)
        if not live.any():
            break
        spoken.say(step)
        heard = torch.full((len(flying),), s * interval, dtype=f64, device=device)
        for _ in range(step_cycles):
            executor.cycle(spoken.at(heard), torch.full_like(heard, executor.count * params.cycle_s))
        done = executor.done.cpu().numpy()
        cycles = executor.done_cycle.cpu().numpy()
        limit = executor.time_limit_s.cpu().numpy()
        for f in range(len(flying)):
            if live[f] and done[f]:
                live[f] = False
                timed_out[f] = (cycles[f] + 1) * params.cycle_s >= limit[f]   # the executor's own test
        s += 1
    for f, j in enumerate(flying):
        if out[j] is not None:
            continue
        lateral, vertical = (np.array([e[c] for e in errors[f]]) for c in (0, 1))
        out[j] = ClosedLoopSentence(grid=np.array(said[f], dtype=np.int16), correction=np.array(added[f], dtype=bool),
                                    states=np.array(states[f], dtype=np.float64), lateral_m=lateral,
                                    vertical_m=vertical, uncorrectable=np.array(blocked[f], dtype=bool),
                                    observed_row=np.array(reached[f], dtype=np.int64), start=start,
                                    timed_out=bool(timed_out[f]))
    return out  # type: ignore[return-value]


def read_chunked(batch: Batch, params: ExecutorParams, words: Words, *, chunk: int,
                 device: torch.device) -> list[ClosedLoopSentence | Refused]:
    """`read` over ``batch`` in chunks of ``chunk`` flights (each flown on its own executor), in the batch's order."""
    out: list[ClosedLoopSentence | Refused | None] = [refusal(s, batch.row_interval_s) for s in batch.sentences]
    flying = [j for j, result in enumerate(out) if result is None]     # only these have a first predicted step
    for first in range(0, len(flying), chunk):
        members = flying[first: first + chunk]
        part = subset(batch, members)
        for j, result in zip(members, read(part, start_inputs(part, words.spec.step_s, device=device), params, words,
                                           device=device)):
            out[j] = result
    return out  # type: ignore[return-value]


# ---- the conformance (design §9.2 #2): the closed-loop reading checked by what it reads, as the labeller's and the
# executor's are. A fixed REFERENCE sample — the train split's first `REFERENCE_PER_AIRPORT` flown flights of each
# airport in a permutation seeded by `REFERENCE_SEED`, at every row interval the closed-loop sentences were written at — is read by the
# code that wrote the artefact's closed-loop sentences, and stored; the CHECK reads it again with the code on disk and
# compares: the same flights, the same refusals, the same words and corrections, and states and errors no further
# apart than the executor's conformance bound. A check that passes, from a clean checkout, writes
# ``passed-<code>.json``: what `require_conforming_closed_loop` asks for before closed-loop sentences are read. The
# code is named by the logic of the executor's files and the labeller's (`closed_loop_code_sha256`).
REFERENCE_SCHEMA = "ts-closed-loop-conformance-reference-v2"
PASSED_SCHEMA = "ts-closed-loop-conformance-passed-v1"
REFERENCE_SPLIT, REFERENCE_PER_AIRPORT, REFERENCE_SEED = "train", 10, 1337
CONFORMANCE = "conformance"


def closed_loop_code_sha256() -> str:
    return logic_sha256([*executor_source_files(), *labeller_code_files()])


def _reference_results(instructions: Path, params: ExecutorParams, words: Words, intervals: Sequence[float], *,
                       device: torch.device) -> dict[float, tuple[list[str], list[ClosedLoopSentence | Refused]]]:
    drawn, readings = replay.draw_readings(instructions, REFERENCE_SPLIT, words.spec, words,
                                           per_airport=REFERENCE_PER_AIRPORT, seed=REFERENCE_SEED,
                                           groups=(replay.OWN, replay.STAND_IN))
    out = {}
    for interval in intervals:
        batch = replay.batch_of(drawn, list(range(len(readings))), readings, interval, words)
        out[interval] = ([s.dataset_id for s in batch.signals],
                         read_chunked(batch, params, words, chunk=len(batch.sentences), device=device))
    return out


def _result_json(result: ClosedLoopSentence | Refused) -> dict[str, Any]:
    if isinstance(result, Refused):
        return {"refused": result.reason}
    return {"rows": len(result.grid), "timed_out": result.timed_out}


def write_reference(instructions: Path, params: ExecutorParams, words: Words, intervals: Sequence[float], *,
                    git: dict[str, Any], target: Path) -> Path:
    """Write the reference into ``target`` (`CONFORMANCE` inside the closed-loop directory being written), at every row
    interval of ``intervals`` (the closed-loop sentences')."""
    target.mkdir()
    payload: dict[str, Any] = {"schema": REFERENCE_SCHEMA, "written_utc": utc_now(), "git": git,
                               "python": platform.python_version(), "code_sha256": closed_loop_code_sha256(),
                               "spec_sha256": words.spec.sha256, "executor_params_sha256": params_sha256(params),
                               "split": REFERENCE_SPLIT, "per_airport": REFERENCE_PER_AIRPORT, "seed": REFERENCE_SEED,
                               "intervals": {}}
    for interval, (flights, results) in _reference_results(instructions, params, words, intervals,
                                                           device=DEVICE).items():
        payload["intervals"][f"{interval:g}"] = {"flights": flights, "results": [_result_json(r) for r in results]}
        read_ok = [r for r in results if isinstance(r, ClosedLoopSentence)]
        with (target / f"reference_{interval:g}s.npz").open("xb") as handle:
            np.savez_compressed(handle, **_stacked(read_ok))
    (target / "reference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _stacked(results: Sequence[ClosedLoopSentence]) -> dict[str, np.ndarray]:
    def cat(name: str, width: int, dtype: Any) -> np.ndarray:
        rows = [getattr(r, name) for r in results]
        return np.concatenate(rows).astype(dtype) if rows else np.zeros((0, width) if width else 0, dtype=dtype)

    return {"words": cat("grid", 5, np.int16), "correction": cat("correction", 5, bool),
            "uncorrectable": cat("uncorrectable", 2, bool), "observed_row": cat("observed_row", 0, np.int64),
            "states": cat("states", len(STATE_COLUMNS), np.float64), "lateral_m": cat("lateral_m", 0, np.float64),
            "vertical_m": cat("vertical_m", 0, np.float64)}


def _reference_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("reference*")):
        digest.update(path.name.encode("utf-8") + b"\0" + file_sha256(path).encode("utf-8") + b"\0")
    return digest.hexdigest()


@dataclass
class Checked:
    flights: int
    code_sha256: str
    git: dict[str, Any]
    largest_state_difference_m: float = 0.0
    mismatches: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def check(instructions: Path, params: ExecutorParams, words: Words, *, git: dict[str, Any]) -> Checked:
    """Read the reference again with the code on disk and compare (the section's comment)."""
    directory = instructions / CLOSED_LOOP_DIRECTORY / CONFORMANCE
    payload = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
    if payload["schema"] != REFERENCE_SCHEMA:
        raise ValueError(f"{directory / 'reference.json'} is not a {REFERENCE_SCHEMA} file")
    if payload["spec_sha256"] != words.spec.sha256 or payload["executor_params_sha256"] != params_sha256(params):
        raise ValueError(f"{directory} was read with another vocabulary spec or executor spec")
    checked = Checked(flights=0, code_sha256=closed_loop_code_sha256(), git=git)
    intervals = [float(name) for name in payload["intervals"]]
    for interval, (flights, results) in _reference_results(instructions, params, words, intervals,
                                                           device=DEVICE).items():
        stored = payload["intervals"][f"{interval:g}"]
        if flights != stored["flights"]:
            checked.mismatches.setdefault(f"{interval:g} s", []).append(
                "other flights than the reference's: the data moved, or the row interval refuses others")
            continue
        with np.load(directory / f"reference_{interval:g}s.npz") as data:
            reference = {name: data[name] for name in data.files}
        again = _stacked([r for r in results if isinstance(r, ClosedLoopSentence)])
        checked.flights += len(flights)
        for flight, result, expected in zip(flights, results, stored["results"], strict=True):
            if _result_json(result) != expected:
                checked.mismatches.setdefault(f"{flight} at {interval:g} s", []).append(
                    f"{_result_json(result)}, the reference {expected}")
        if checked.mismatches:
            continue
        for name in ("words", "correction", "uncorrectable", "observed_row"):
            if not np.array_equal(again[name], reference[name]):
                checked.mismatches.setdefault(f"{interval:g} s", []).append(f"the {name} differ")
        if any(again[n].shape != reference[n].shape for n in ("states", "lateral_m", "vertical_m")):
            checked.mismatches.setdefault(f"{interval:g} s", []).append("the states have another shape")
            continue

        def apart(a: np.ndarray, b: np.ndarray) -> float:
            """The largest difference, infinite where one is NaN and the other not (both NaN: equal)."""
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                return math.inf
            return float(np.abs(np.nan_to_num(a) - np.nan_to_num(b)).max(initial=0.0))

        difference = apart(again["states"][:, :3], reference["states"][:, :3])
        others = max(apart(again[n], reference[n]) for n in ("lateral_m", "vertical_m"))
        rest = apart(again["states"][:, 3:], reference["states"][:, 3:])
        checked.largest_state_difference_m = max(checked.largest_state_difference_m, difference, others)
        if not (difference <= STATE_BOUND_M and others <= STATE_BOUND_M and rest <= ROUNDOFF):
            checked.mismatches.setdefault(f"{interval:g} s", []).append(
                f"states {difference:.3g} m, errors {others:.3g} m, other columns {rest:.3g}")
    return checked


def passed_path(instructions: Path, code_sha256: str) -> Path:
    return instructions / CLOSED_LOOP_DIRECTORY / CONFORMANCE / f"passed-{code_sha256[:12]}.json"


def write_passed(instructions: Path, checked: Checked) -> Path:
    """The record that the code on disk reads the reference as it was read (a passed check, from a clean checkout)."""
    if not checked.passed:
        raise ValueError(f"the closed-loop reference reads otherwise: {checked.mismatches}")
    git = git_state()
    if checked.git["dirty"] or git != checked.git or closed_loop_code_sha256() != checked.code_sha256:
        raise RuntimeError("a passed record is written from a clean checkout only, for the code the check read with")
    path = passed_path(instructions, checked.code_sha256)
    write_json_atomic(path, {"schema": PASSED_SCHEMA, "written_utc": utc_now(), "git": git,
                             "python": platform.python_version(), "code_sha256": checked.code_sha256,
                             "reference_sha256": _reference_sha256(path.parent), "flights": checked.flights,
                             "largest_state_difference_m": checked.largest_state_difference_m})
    return path


def require_conforming_closed_loop(instructions: Path) -> None:
    """Refused unless the code on disk has read ``instructions``' closed-loop reference as it was read."""
    code = closed_loop_code_sha256()
    path = passed_path(instructions, code)
    command = (f"python run_ts.py instruction_closed_loop --check --instructions {instructions} --executor <its spec> "
               f"(Python {platform.python_version()})")
    if not path.exists():
        raise ValueError(f"the closed-loop code on disk ({code[:12]}) has not been checked against {instructions.name}'s "
                         f"reference: {command}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["schema"] != PASSED_SCHEMA or record["code_sha256"] != code
            or record["reference_sha256"] != _reference_sha256(path.parent)):
        raise ValueError(f"{path} is not a passed record of this code against {instructions.name}'s reference: {command}")


@dataclass(frozen=True)
class Stored:
    """One flight's closed-loop sentence as the artefact holds it (`instructions.artefact.load_closed_loop`)."""

    grid: np.ndarray
    correction: np.ndarray
    first_row: int              # the 2 s row of its sentence's first row on the interval's grid
    states: np.ndarray          # the flown rows only, from the first predicted step
    lateral_m: np.ndarray
    vertical_m: np.ndarray
    uncorrectable: np.ndarray
    observed_row: np.ndarray


def stored_sentences(data: dict[str, np.ndarray]) -> dict[int, Stored]:
    """A loaded closed-loop file's sentences by their signal index."""
    start = int(data["start_row"])
    out = {}
    for k, index in enumerate(data["signal_index"].tolist()):
        rows = slice(int(data["offsets"][k]), int(data["offsets"][k + 1]))
        states = slice(int(data["state_offsets"][k]) + start, int(data["state_offsets"][k + 1]))
        out[index] = Stored(grid=data["words"][rows], correction=data["correction"][rows],
                            first_row=int(data["first_row"][k]), states=data["states"][states],
                            lateral_m=data["lateral_m"][rows], vertical_m=data["vertical_m"][rows],
                            uncorrectable=data["uncorrectable"][rows], observed_row=data["observed_row"][rows])
    return out


def replay_batch(batch: Batch, stored: dict[int, Stored], words: Words) -> tuple[Batch, int]:
    """``batch``'s flights that have a closed-loop sentence in ``stored``, each to be flown on it from its first
    predicted step (its sentence's first row and its observed flight moved there), and how many had none."""
    every, start = interval_rows(batch.row_interval_s, words.spec.step_s), start_row(batch.row_interval_s)
    kept = [j for j, index in enumerate(batch.indices) if index in stored]
    out = subset(batch, kept)
    for position, j in enumerate(kept):
        grid = stored[batch.indices[j]].grid
        if stored[batch.indices[j]].first_row != batch.sentences[j].first_row:
            raise ValueError(f"{batch.signals[j].dataset_id}: the closed-loop sentence starts at 2 s row "
                             f"{stored[batch.indices[j]].first_row}, the labelled one at {batch.sentences[j].first_row}")
        out.sentences[position] = replay.Sentence(
            grid=grid, instructions=replay.instructions_of(grid, batch.geometries[j], words),
            first_row=batch.sentences[j].first_row + start * every)
        out.signals[position] = replay.from_row(batch.signals[j], start * every)
    return out, len(batch.indices) - len(kept)
