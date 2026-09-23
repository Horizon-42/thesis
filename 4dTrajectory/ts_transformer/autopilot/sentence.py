"""Which word of each column is in force at each control cycle (executor design §2.3, §9 method B).

A sentence is a grid of steps (`VocabularySpec.step_s` apart) × the six columns; a cell holds a word
or `UNCHANGED`. The word in force at a step is the last one written in its column at or before it —
step 0 writes every column. A word the labeller read at step ``r ≥ 1`` takes effect ``delay``
seconds after that step's time, the column's delay (the labeller's issue row leads the manoeuvre it
reads, §9 method B); step 0's words describe what the aircraft is already doing and are in force
from t = 0. After the sentence's last step every word stays in force.

The runway pointer and the approach column follow the heading's delay: the clearance is given with
the heading word it goes with (vocabulary design §3.2), and the pointer is written at step 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, HEADING, RUNWAY, SPEED, UNCHANGED, Words


@dataclass(frozen=True)
class Delays:
    """Seconds from a word's step to its effect, per group of columns."""

    heading_s: float
    vertical_s: float
    speed_s: float

    def column(self, index: int) -> float:
        return {RUNWAY: self.heading_s, APPROACH: self.heading_s, HEADING: self.heading_s,
                ALTITUDE: self.vertical_s, ANGLE: self.vertical_s, SPEED: self.speed_s}[index]


@dataclass(frozen=True)
class WordsInForce:
    """``[B, T]`` per field: T control cycles, cycle ``k`` starting at ``k · cycle_s``.

    ``issued_step`` is ``[B, T, 6]``: the step each column's word in force was written at, so a law
    can tell a new word from the one it was already flying (two words may carry the same value)."""

    runway: torch.Tensor          # long, candidate index
    approach: torch.Tensor        # long, APPROACH_*
    heading_deg: torch.Tensor     # compass target θ
    altitude_m: torch.Tensor      # MSL target (meaningless where ``land``)
    land: torch.Tensor            # bool: "descend to land"
    angle_class: torch.Tensor     # long, the angle column's class
    angle_deg: torch.Tensor       # the class's nominal path angle, descending positive (0 = level)
    speed_mps: torch.Tensor       # ground-speed target (meaningless where ``unspecified``)
    unspecified: torch.Tensor     # bool: the pilot's own speed
    issued_step: torch.Tensor     # long [B, T, 6]


def _filled(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(value, issued_step)`` per step and column: the last word written at or before each step."""
    if (grid[0] == UNCHANGED).any():
        raise ValueError("a sentence's step 0 must write every column")
    written = grid != UNCHANGED
    steps = np.where(written, np.arange(len(grid))[:, None], 0)
    issued = np.maximum.accumulate(steps, axis=0)
    return np.take_along_axis(grid, issued, axis=0), issued


def words_in_force(grids: Sequence[np.ndarray], words: Words, delays: Delays, *, cycle_s: float, cycles: int,
                   device: torch.device) -> WordsInForce:
    """The words in force at each of ``cycles`` control cycles, for every flight's sentence ``grid``
    (``[N_i, 6]``, the artefact's word grid)."""
    step_s = words.spec.step_s
    time = np.arange(cycles) * cycle_s
    values = np.zeros((len(grids), cycles, 6), dtype=np.int64)
    issued = np.zeros((len(grids), cycles, 6), dtype=np.int64)
    for flight, grid in enumerate(grids):
        value, step = _filled(np.asarray(grid, dtype=np.int64))
        for column in range(6):
            # the last step whose word is in effect at each cycle's start (step 0 from t = 0)
            row = np.floor((time - delays.column(column)) / step_s + 1e-9).astype(np.int64)
            row = np.clip(row, 0, len(grid) - 1)
            values[flight, :, column] = value[row, column]
            issued[flight, :, column] = step[row, column]

    def table(entries: list[float]) -> np.ndarray:
        return np.asarray(entries, dtype=np.float64)

    heading = table([words.heading_deg(i) for i in range(words.n_heading)])
    altitude = table([words.altitude_m(i) if i != words.altitude_land else math.nan
                      for i in range(words.altitude_land + 1)])
    angle = table([words.angle_deg(i) for i in range(words.n_descent + 2)])
    speed = table([words.speed_mps(i) if i != words.speed_unspecified else math.nan
                   for i in range(words.speed_unspecified + 1)])

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, device=device)

    v = values
    return WordsInForce(
        runway=tensor(v[..., RUNWAY]), approach=tensor(v[..., APPROACH]),
        heading_deg=tensor(heading[v[..., HEADING]]),
        altitude_m=tensor(altitude[v[..., ALTITUDE]]), land=tensor(v[..., ALTITUDE] == words.altitude_land),
        angle_class=tensor(v[..., ANGLE]), angle_deg=tensor(angle[v[..., ANGLE]]),
        speed_mps=tensor(speed[v[..., SPEED]]), unspecified=tensor(v[..., SPEED] == words.speed_unspecified),
        issued_step=tensor(issued),
    )
