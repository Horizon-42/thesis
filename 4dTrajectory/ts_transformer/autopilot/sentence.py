"""Which word of each column is in force at each control cycle (executor design §2.3, §9 method B, §11).

A sentence is a grid of steps (`VocabularySpec.step_s` apart) × the six columns; a cell holds a word or
`UNCHANGED`. The word in force at a SENTENCE TIME is the last one written in its column at or before it —
step 0 writes every column. A word read at step ``r ≥ 1`` takes effect ``delay`` seconds after its step's time,
the column's delay (the labeller's issue row leads the manoeuvre it reads, §9 method B); step 0's words describe
what the aircraft is already doing and are in force from the start. After the sentence's last step every word
stays in force. `Sentences.at` looks the words up at any sentence time.

Which sentence time a control cycle is at is the replay's CLOCK (§11). Past the observed track's end (the executor
slower, or on a longer path) sentence time runs on at the executor's own pace, so a word said in the sentence's last
rows still takes effect after its delay; delays are sentence seconds.

- `TimeClock` — the sentence's own clock: a word is said at the second it was said to the observed aircraft;
- `DistanceClock` — a word is said when the executor has flown as far along its own path as the observed aircraft
  had when the word was said to it;
- `TrackClock` — where the observed aircraft was: a word is said when the executor is at the point of the observed
  track where it was said to the observed aircraft (the nearest point ahead, moving only forward). A tighter turn
  than the observed one reaches the rest of the track sooner, which path length does not see. A word carries a target, not a rate (a
  speed word does not say how fast to slow), so an executor that flies its words at its own pace drifts along its
  path from the observed aircraft, and on the time clock the later words then reach it where they were never
  meant (a base turn two kilometres early, a descent over a shorter final). Air traffic control says a word where
  the aircraft is, and the closed loop's model will say it from the executor's own state: the distance and track
  clocks are the truth sentence said that way.

The columns without a delay (`UNDELAYED`: the runway pointer, the approach column, the heading) are heard once a step,
on the cycles that start a row: a heading word says where the track is a lead after the row it is heard at, and the
judge reads the flown track at rows (`judge.said_at`). On the time clock a row starts on such a cycle anyway; on the
track and distance clocks a word whose row the sentence time reaches between them waits for the next one (a cycle).
The delayed columns are looked up every cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, HEADING, RUNWAY, SPEED, UNCHANGED, Words

#: Which columns each delay moves: the manoeuvre columns method B measures it on. A heading word carries its own
#: timing — it says where the track is `heading_lead_s` later (vocabulary design §10.1) — so it acts when it is
#: said, and the runway pointer and the clearance, which go with it, do too.
DELAY_GROUPS = {"vertical_s": (ALTITUDE, ANGLE), "speed_s": (SPEED,)}
UNDELAYED = (RUNWAY, APPROACH, HEADING)
CLOCKS = ("time", "distance", "track")
#: The track clock looks this far ahead along the observed track for the executor's nearest point (it moves only
#: forward, so a track that loops — an orbit — is followed around, never jumped across) ...
TRACK_WINDOW_S = 60.0
#: ... and moves at most this many rows a cycle: sentence time then runs at most twice as fast as the observed
#: flight's (a row is 2 s, a cycle 1 s) and every row is passed — however far from the observed track the executor is,
#: where "nearest" means little. The delayed columns see every row; the undelayed ones, heard once a step, do not see a
#: row the clock passes within a step (the distance clock, which has no such cap, can pass several): two heading words
#: then arrive together and the first is never flown — the judge judges it on no rows, and the replay counts the one
#: told with it apart (`replay.told_with_skipped`).
TRACK_MAX_ROWS_PER_CYCLE = 1
#: A sentence time this close below a row's start (in rows) is read as that row: float round-off.
ROW_ROUNDING = 1e-9


def row_at(seconds, step_s: float):
    """The sentence row a sentence time falls in (torch or numpy) — the one reading of it the executor's lookup
    (`Sentences.at`) and the judge's filing of a word (`judge.words_said`) share."""
    scaled = seconds / step_s + ROW_ROUNDING
    return torch.floor(scaled).long() if isinstance(scaled, torch.Tensor) else np.floor(scaled).astype(np.int64)


@dataclass(frozen=True)
class Delays:
    """Seconds from a word's step to its effect, per group of columns (`DELAY_GROUPS`)."""

    vertical_s: float
    speed_s: float

    def column(self, index: int) -> float:
        if index in UNDELAYED:
            return 0.0
        (name,) = [name for name, columns in DELAY_GROUPS.items() if index in columns]
        return getattr(self, name)


@dataclass(frozen=True)
class WordsNow:
    """The words in force at one control cycle, ``[B]`` each.

    ``issued_step`` is ``[B, 6]``: the step each column's word in force was written at, so a law can tell a new
    word from the one it was already flying (two words may carry the same value)."""

    runway: torch.Tensor          # long, candidate index
    approach: torch.Tensor        # long, APPROACH_*
    heading_deg: torch.Tensor     # compass target θ
    altitude_m: torch.Tensor      # MSL target (meaningless where ``land``)
    land: torch.Tensor            # bool: "descend to land"
    angle_class: torch.Tensor     # long, the angle column's class
    angle_deg: torch.Tensor       # the class's nominal path angle, descending positive (0 = level)
    speed_mps: torch.Tensor       # ground-speed target (meaningless where ``unspecified``)
    unspecified: torch.Tensor     # bool: the pilot's own speed
    issued_step: torch.Tensor     # long [B, 6]


def _filled(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(value, issued_step)`` per step and column: the last word written at or before each step."""
    if (grid[0] == UNCHANGED).any():
        raise ValueError("a sentence's step 0 must write every column")
    written = grid != UNCHANGED
    steps = np.where(written, np.arange(len(grid))[:, None], 0)
    issued = np.maximum.accumulate(steps, axis=0)
    return np.take_along_axis(grid, issued, axis=0), issued


class Sentences:
    """A batch's sentences (``grids``: each ``[N_i, 6]``, the artefact's word grid), looked up at any sentence time."""

    def __init__(self, grids: Sequence[np.ndarray], words: Words, *, device: torch.device) -> None:
        self.words, self.step_s = words, words.spec.step_s
        width = max(len(grid) for grid in grids)
        value = np.zeros((len(grids), width, 6), dtype=np.int64)
        issued = np.zeros((len(grids), width, 6), dtype=np.int64)
        for flight, grid in enumerate(grids):
            v, i = _filled(np.asarray(grid, dtype=np.int64))
            value[flight, : len(grid)], issued[flight, : len(grid)] = v, i
            value[flight, len(grid):], issued[flight, len(grid):] = v[-1], i[-1]
        self.value = torch.as_tensor(value, device=device)
        self.issued = torch.as_tensor(issued, device=device)
        self.rows = torch.as_tensor([len(grid) for grid in grids], device=device)

        def table(entries: list[float]) -> torch.Tensor:
            return torch.as_tensor(np.asarray(entries, dtype=np.float64), device=device)

        self.heading = table([words.heading_deg(i) for i in range(words.n_heading)])
        self.altitude = table([words.altitude_m(i) if i != words.altitude_land else math.nan
                               for i in range(words.altitude_land + 1)])
        self.angle = table([words.angle_deg(i) for i in range(words.n_descent + 2)])
        self.speed = table([words.speed_mps(i) if i != words.speed_unspecified else math.nan
                            for i in range(words.speed_unspecified + 1)])

    def at(self, sentence_s: torch.Tensor, step_start_s: torch.Tensor, delays: Delays) -> WordsNow:
        """The words in force at each flight's sentence time ``[B]``; the undelayed columns at ``step_start_s``, the
        sentence time of the last cycle that started a row (module docstring)."""
        batch = torch.arange(len(self.rows), device=self.rows.device)
        value, issued = [], []
        for column in range(6):
            heard_s = step_start_s if column in UNDELAYED else sentence_s - delays.column(column)
            row = row_at(heard_s, self.step_s)
            row = torch.minimum(row.clamp(min=0), self.rows - 1)
            value.append(self.value[batch, row, column])
            issued.append(self.issued[batch, row, column])
        v = torch.stack(value, dim=1)
        return WordsNow(
            runway=v[:, RUNWAY], approach=v[:, APPROACH], heading_deg=self.heading[v[:, HEADING]],
            altitude_m=self.altitude[v[:, ALTITUDE]], land=v[:, ALTITUDE] == self.words.altitude_land,
            angle_class=v[:, ANGLE], angle_deg=self.angle[v[:, ANGLE]], speed_mps=self.speed[v[:, SPEED]],
            unspecified=v[:, SPEED] == self.words.speed_unspecified, issued_step=torch.stack(issued, dim=1))


class TimeClock:
    """The sentence's own clock: a cycle's sentence time is the time flown."""

    def __init__(self, cycle_s: float) -> None:
        self.cycle_s = cycle_s

    def now(self, cycle: int, state: Kinematics) -> torch.Tensor:
        return torch.full_like(state.e_m, cycle * self.cycle_s)


class DistanceClock:
    """Where the observed aircraft was (module docstring): a cycle's sentence time is the time at which the observed
    aircraft had flown, along its own path from row 0, as far as the executor has along its own; past the path's end,
    on at the executor's pace (``cycle_s`` a cycle). ``observed_path_m`` is ``[B, N]``: each flight's path length at its
    sentence's rows (``step_s`` apart), padded past the flight's own rows with a path that never ends."""

    def __init__(self, observed_path_m: torch.Tensor, rows: torch.Tensor, step_s: float, cycle_s: float) -> None:
        self.path, self.rows, self.step_s, self.cycle_s = observed_path_m, rows, step_s, cycle_s
        self.flown: torch.Tensor | None = None
        self.last: tuple[torch.Tensor, torch.Tensor] | None = None
        self.beyond_s: torch.Tensor | None = None

    @classmethod
    def of(cls, e_m: Sequence[np.ndarray], n_m: Sequence[np.ndarray], step_s: float, cycle_s: float, *,
           device: torch.device) -> DistanceClock:
        """From each flight's observed positions at its sentence's rows."""
        width = max(len(e) for e in e_m)
        path = np.zeros((len(e_m), width), dtype=np.float64)
        for flight, (e, n) in enumerate(zip(e_m, n_m)):
            steps = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(e), np.diff(n)))))
            path[flight, : len(e)] = steps
            path[flight, len(e):] = steps[-1] + 1e9 * np.arange(1, width - len(e) + 1)
        return cls(torch.as_tensor(path, device=device), torch.as_tensor([len(e) for e in e_m], device=device), step_s,
                   cycle_s)

    def now(self, cycle: int, state: Kinematics) -> torch.Tensor:
        if self.last is None:
            self.flown = torch.zeros_like(state.e_m)
            self.beyond_s = torch.zeros_like(state.e_m)
        else:
            self.flown = self.flown + torch.hypot(state.e_m - self.last[0], state.n_m - self.last[1])
        self.last = (state.e_m.clone(), state.n_m.clone())
        index = torch.searchsorted(self.path, self.flown[:, None], right=True)[:, 0] - 1
        index = torch.minimum(index.clamp(min=0), torch.full_like(index, self.path.shape[1] - 2))
        batch = torch.arange(len(index), device=index.device)
        low, high = self.path[batch, index], self.path[batch, index + 1]
        time = (index + ((self.flown - low) / (high - low)).clamp(0.0, 1.0)) * self.step_s
        end = (self.rows - 1) * self.step_s
        past = time >= end
        time = torch.where(past, end + self.beyond_s, time)
        self.beyond_s = torch.where(past, self.beyond_s + self.cycle_s, self.beyond_s)
        return time


class TrackClock:
    """Where the observed aircraft was (module docstring): a cycle's sentence time is the time of the observed track's
    point nearest the executor, looking `TRACK_WINDOW_S` ahead of the last one, never back, at most
    `TRACK_MAX_ROWS_PER_CYCLE` rows a cycle; past the track's last row, on at the executor's pace. ``observed_e_m`` /
    ``observed_n_m`` are ``[B, N]``, each flight's observed positions at its sentence's rows, padded past its own rows
    with its last position."""

    def __init__(self, observed_e_m: torch.Tensor, observed_n_m: torch.Tensor, rows: torch.Tensor, step_s: float,
                 cycle_s: float) -> None:
        self.e, self.n, self.rows, self.step_s, self.cycle_s = observed_e_m, observed_n_m, rows, step_s, cycle_s
        self.window = int(round(TRACK_WINDOW_S / step_s))
        self.row = torch.zeros(len(rows), dtype=torch.long, device=rows.device)
        self.beyond_s = torch.zeros(len(rows), dtype=torch.float64, device=rows.device)

    @classmethod
    def of(cls, e_m: Sequence[np.ndarray], n_m: Sequence[np.ndarray], step_s: float, cycle_s: float, *,
           device: torch.device) -> TrackClock:
        width = max(len(e) for e in e_m)

        def padded(values: Sequence[np.ndarray]) -> torch.Tensor:
            out = np.zeros((len(values), width), dtype=np.float64)
            for flight, v in enumerate(values):
                out[flight, : len(v)], out[flight, len(v):] = v, v[-1]
            return torch.as_tensor(out, device=device)

        return cls(padded(e_m), padded(n_m), torch.as_tensor([len(e) for e in e_m], device=device), step_s, cycle_s)

    def now(self, cycle: int, state: Kinematics) -> torch.Tensor:
        ahead = self.row[:, None] + torch.arange(self.window + 1, device=self.row.device)[None, :]
        ahead = torch.minimum(ahead, (self.rows - 1)[:, None])
        distance = torch.hypot(self.e.gather(1, ahead) - state.e_m[:, None], self.n.gather(1, ahead) - state.n_m[:, None])
        nearest = ahead.gather(1, distance.argmin(dim=1, keepdim=True))[:, 0]
        self.beyond_s = torch.where(self.row == self.rows - 1, self.beyond_s + self.cycle_s, self.beyond_s)
        self.row = torch.minimum(nearest, self.row + TRACK_MAX_ROWS_PER_CYCLE)
        return self.row.to(torch.float64) * self.step_s + self.beyond_s
