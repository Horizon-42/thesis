"""Which word of each column is in force at each control cycle (executor design §2.3, §11; vocabulary §3, §5).

A sentence is a grid of steps (its ROW INTERVAL ``step_s`` apart: the data's 2 s, or a coarser Δ of the ablation,
vocabulary §4.8) × the five columns; a cell holds a word or `UNCHANGED`. The word in force at a SENTENCE TIME is the last
one written in its column at or before it — step 0 writes every column. The runway column gives two things: the RUNWAY
IN FORCE R, its last candidate (a go-around does not change it), and the GO-AROUND STATE G, true while its last word is
"go-around" (vocabulary §3.2). A heading word is relative to the course of R; the lateral law turns it into a track when it
hears it (`lateral`). A word takes effect when it is said: the vocabulary's own meaning, and the executor
takes nothing else (no delay; method B, which measured one, is archived: `archive/executor_vocabulary_only_2026_09/`).
After the sentence's last step every word stays in force. `Sentences.at` looks the words up at any sentence time.

A sentence is said on its own rows (D57): a cycle's sentence time is the time flown, and every word is heard once a
step, on the cycle that starts its row — a heading word says where the track is a lead after the row it is heard at,
and the judge reads the flown track at rows (`judge.words_said`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)

#: A sentence time this close below a row's start (in rows) is read as that row: float round-off.
ROW_ROUNDING = 1e-9


def row_at(seconds, step_s: float):
    """The sentence row a sentence time falls in (torch or numpy) — the one reading of it the executor's lookup
    (`Sentences.at`) and the judge's filing of a word (`judge.words_said`) share."""
    scaled = seconds / step_s + ROW_ROUNDING
    return torch.floor(scaled).long() if isinstance(scaled, torch.Tensor) else np.floor(scaled).astype(np.int64)


@dataclass(frozen=True)
class WordsNow:
    """The words in force at one control cycle, ``[B]`` each.

    ``issued_step`` is ``[B, 5]``: the step each column's word in force was written at, so a law can tell a new
    word from the one it was already flying (two words may carry the same value); the runway column's is its last
    word's, "go-around" included."""

    runway: torch.Tensor          # long, the runway in force R (a candidate index)
    go_around: torch.Tensor       # bool, the go-around state G
    heading_rel_deg: torch.Tensor  # the heading word's track relative to the course of R, (−180, 180]
    level_m: torch.Tensor         # the level T above the airport elevation E, D58 (meaningless where ``no_level_off``)
    no_level_off: torch.Tensor    # bool: "no level-off"
    angle_class: torch.Tensor     # long, the angle column's class
    angle_deg: torch.Tensor       # the class's nominal path angle, descending positive (0 = level)
    speed_mps: torch.Tensor       # ground-speed target (meaningless where ``unspecified``)
    unspecified: torch.Tensor     # bool: the pilot's own speed
    issued_step: torch.Tensor     # long [B, 5]


def _filled(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(value, issued_step, runway)`` per step: each column's last word written at or before each step and the step it
    was written at, and the runway in force (the runway column's last candidate)."""
    if (grid[0] == UNCHANGED).any() or grid[0, RUNWAY] < 0:
        raise ValueError("a sentence's step 0 must write every column, a candidate in the runway column")
    written = grid != UNCHANGED
    steps = np.where(written, np.arange(len(grid))[:, None], 0)
    issued = np.maximum.accumulate(steps, axis=0)
    candidate = np.maximum.accumulate(np.where(grid[:, RUNWAY] >= 0, np.arange(len(grid)), 0))
    return np.take_along_axis(grid, issued, axis=0), issued, grid[candidate, RUNWAY]


class WordTables:
    """Each column's word values as the laws read them (relative headings, altitudes, angles, speeds), and a step's
    words in force → `WordsNow` — shared by whole sentences (`Sentences`) and sentences said a step at a time
    (`Spoken`)."""

    def __init__(self, words: Words, *, device: torch.device) -> None:
        self.words = words

        def table(entries: list[float]) -> torch.Tensor:
            return torch.as_tensor(np.asarray(entries, dtype=np.float64), device=device)

        self.heading = table([words.heading_relative_deg(i) for i in range(words.n_heading)])
        self.level = table([words.altitude_level_m(i) if i != words.altitude_no_level_off else math.nan
                            for i in range(words.altitude_no_level_off + 1)])
        self.angle = table([words.angle_deg(i) for i in range(words.n_descent + 2)])
        self.speed = table([words.speed_mps(i) if i != words.speed_unspecified else math.nan
                            for i in range(words.speed_unspecified + 1)])

    def now(self, value: torch.Tensor, issued: torch.Tensor, runway: torch.Tensor) -> WordsNow:
        """``value`` / ``issued``: ``[B, 5]``, the word in force per column and the step it was written at; ``runway``
        ``[B]`` the runway in force."""
        return WordsNow(
            runway=runway, go_around=value[:, RUNWAY] == RUNWAY_GO_AROUND, heading_rel_deg=self.heading[value[:, HEADING]],
            level_m=self.level[value[:, ALTITUDE]],
            no_level_off=value[:, ALTITUDE] == self.words.altitude_no_level_off,
            angle_class=value[:, ANGLE], angle_deg=self.angle[value[:, ANGLE]], speed_mps=self.speed[value[:, SPEED]],
            unspecified=value[:, SPEED] == self.words.speed_unspecified, issued_step=issued)


class Sentences:
    """A batch's sentences (``grids``: each ``[N_i, 5]``, a word grid of rows ``step_s`` apart), looked up at any
    sentence time."""

    def __init__(self, grids: Sequence[np.ndarray], words: Words, *, step_s: float, device: torch.device) -> None:
        self.words, self.step_s = words, step_s
        width = max(len(grid) for grid in grids)
        value = np.zeros((len(grids), width, len(COLUMNS)), dtype=np.int64)
        issued = np.zeros((len(grids), width, len(COLUMNS)), dtype=np.int64)
        runway = np.zeros((len(grids), width), dtype=np.int64)
        for flight, grid in enumerate(grids):
            v, i, r = _filled(np.asarray(grid, dtype=np.int64))
            value[flight, : len(grid)], issued[flight, : len(grid)], runway[flight, : len(grid)] = v, i, r
            value[flight, len(grid):], issued[flight, len(grid):], runway[flight, len(grid):] = v[-1], i[-1], r[-1]
        self.value = torch.as_tensor(value, device=device)
        self.issued = torch.as_tensor(issued, device=device)
        self.runway = torch.as_tensor(runway, device=device)
        self.rows = torch.as_tensor([len(grid) for grid in grids], device=device)
        self.tables = WordTables(words, device=device)

    def at(self, heard_s: torch.Tensor) -> WordsNow:
        """The words in force at each flight's sentence time ``[B]`` — the one of the last cycle that started a row
        (module docstring)."""
        batch = torch.arange(len(self.rows), device=self.rows.device)
        row = torch.minimum(row_at(heard_s, self.step_s).clamp(min=0), self.rows - 1)
        return self.tables.now(self.value[batch, row], self.issued[batch, row], self.runway[batch, row])


class Spoken:
    """Sentences said a step at a time (a closed loop: the speaker says a step, the executor flies it). `say` writes
    the next step's words — the first step every column, each later one a word or `UNCHANGED` per column — and `at`
    reads the words in force as `Sentences.at` does: the executor always hears the step just said.

    A multi-aircraft batch (``start_step``, the executor's ``start_cycle`` in steps) gives each flight its own first
    step: before it nothing said for the flight counts, at it every column must be written, and its words are issued at
    its own steps; `sentences` hands each flight's steps from its own first."""

    def __init__(self, batch: int, words: Words, *, step_s: float, device: torch.device,
                 start_step: np.ndarray | None = None) -> None:
        self.step_s, self.device = step_s, device
        self.tables = WordTables(words, device=device)
        self.value = torch.zeros((batch, len(COLUMNS)), dtype=torch.long, device=device)
        self.issued = torch.zeros((batch, len(COLUMNS)), dtype=torch.long, device=device)
        self.runway = torch.zeros(batch, dtype=torch.long, device=device)
        self.start = (np.zeros(batch, dtype=np.int64) if start_step is None
                      else np.asarray(start_step, dtype=np.int64))
        if (self.start < 0).any():
            raise ValueError("a flight's start step is negative")
        self.staggered = bool((self.start != 0).any())
        self.grid: list[np.ndarray] = []             # the steps said, each [B, 5] (UNCHANGED where nothing)

    @property
    def steps(self) -> int:
        """Steps said, from the batch's first."""
        return len(self.grid)

    def say(self, row: np.ndarray) -> None:
        """The next step's words, ``[B, 5]``; a flight's first step must write every column, and before it what is
        said for the flight is not its own."""
        row = np.asarray(row, dtype=np.int64)
        if self.staggered:
            row = np.where((self.start <= self.steps)[:, None], row, UNCHANGED)
            first = self.start == self.steps
            own = torch.as_tensor(self.steps - self.start, device=self.device)[:, None].expand_as(self.issued)
        else:
            first = np.full(len(row), not self.grid)
        if not ((row[first] != UNCHANGED).all() and (row[first, RUNWAY] >= 0).all()):
            raise ValueError("a sentence's step 0 must write every column, a candidate in the runway column")
        written = torch.as_tensor(row != UNCHANGED, device=self.device)
        words = torch.as_tensor(row, device=self.device)
        self.runway = torch.where(words[:, RUNWAY] >= 0, words[:, RUNWAY], self.runway)
        self.value = torch.where(written, words, self.value)
        self.issued = torch.where(written, own if self.staggered else torch.full_like(self.issued, self.steps),
                                  self.issued)
        self.grid.append(row)

    def at(self, heard_s: torch.Tensor) -> WordsNow:
        """The words in force; ``heard_s`` each flight's own sentence time (a flight that has not started: anything)."""
        rows = row_at(heard_s, self.step_s)
        if not self.staggered:
            if bool((rows != self.steps - 1).any()):
                raise ValueError(f"heard at rows {sorted(set(rows.tolist()))}, but the step just said is {self.steps - 1}")
            return self.tables.now(self.value, self.issued, self.runway)
        expected = torch.as_tensor(self.steps - 1 - self.start, device=rows.device)
        started = expected >= 0
        if bool((rows != expected)[started].any()):
            raise ValueError(f"heard at rows {sorted(set(rows[started].tolist()))}, but the steps just said are "
                             f"{sorted(set(expected[started].tolist()))}")
        return self.tables.now(self.value, self.issued, self.runway)

    def sentences(self) -> np.ndarray:
        """``[B, steps, 5]``: every step said, each flight's from its own first (a later starter's last steps
        `UNCHANGED`)."""
        grid = np.stack(self.grid, axis=1)
        if not self.staggered:
            return grid
        out = np.full_like(grid, UNCHANGED)
        for flight, start in enumerate(self.start):
            own = max(grid.shape[1] - int(start), 0)               # none yet for a flight still to start
            out[flight, :own] = grid[flight, grid.shape[1] - own:]
        return out
