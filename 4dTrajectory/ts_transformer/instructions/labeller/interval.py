"""A sentence read at the data's 2 s rows put on a coarser grid of rows Δ apart — the row-interval ablation (design
§4.8, D11). One reading gives every Δ; nothing is read again.

1. The Δ rows are the rows on UTC multiples of Δ (the rule that put artefact `v6_20261002`'s rows on even seconds, so
   the aircraft of a scene stay on one grid): the first is `first_interval_row`, then every Δ / step rows.
2. At each Δ row each column says the word in force at that row if it differs from the word in force at the previous
   Δ row: inside one interval only the last word of a column stays (a go-around and the runway said again inside one
   interval cancel). In the runway column the word in force is the column's last word, "go-around" included.
3. The first Δ row says the five words in force there; a sentence in a go-around at its first Δ row cannot say a
   runway there and is refused. The grammar (`instructions.grammar`) is checked again on the Δ rows, at their altitude.

Δ must be a whole number of steps and divide the prior's 16 s observation (D25: Δ = 2, 4, 8 s; 6 s is refused); at
Δ = the step the sentence comes back unchanged.
"""

from __future__ import annotations

import numpy as np

from ts_transformer.data.day_split import parse_utc
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.labeller.sentence import check_grammar
from ts_transformer.instructions.words import RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words


#: The prior's observation before its first predicted step, s (design §2, `N_LOOK` = 8 rows at 2 s): every Δ divides it,
#: so the first predicted step is a Δ row at every Δ (D25).
OBSERVATION_S = 16.0


def interval_rows(interval_s: float, step_s: float) -> int:
    """Δ as a whole number of rows; refused unless it is a positive whole number of steps that divides
    `OBSERVATION_S`."""
    rows = interval_s / step_s
    if interval_s <= 0.0 or abs(rows - round(rows)) > 1e-9:
        raise ValueError(f"a row interval of {interval_s:g} s is not a positive whole number of {step_s:g} s rows")
    looks = OBSERVATION_S / interval_s
    if abs(looks - round(looks)) > 1e-9:
        raise ValueError(f"a row interval of {interval_s:g} s does not divide the {OBSERVATION_S:g} s observation (D25)")
    return int(round(rows))


def first_interval_row(entry_time_utc: str, interval_s: float, step_s: float) -> int:
    """The first row (row 0 at ``entry_time_utc``, rows ``step_s`` apart) whose UTC second is a multiple of Δ."""
    every = interval_rows(interval_s, step_s)
    seconds = parse_utc(entry_time_utc).timestamp()
    if abs(seconds / step_s - round(seconds / step_s)) > 1e-9:
        raise ValueError(f"row 0 at {entry_time_utc} is not on the {step_s:g} s UTC grid")
    return next(row for row in range(every) if round(seconds + row * step_s) % round(interval_s) == 0)


def in_force(grid: np.ndarray) -> np.ndarray:
    """``[N, 5]``: each column's last word at or before each row (row 0 must say every column)."""
    if (grid[0] == UNCHANGED).any():
        raise ValueError("a sentence's row 0 says every column")
    written = grid != UNCHANGED
    last = np.maximum.accumulate(np.where(written, np.arange(len(grid))[:, None], 0), axis=0)
    return np.take_along_axis(grid, last, axis=0)


def on_interval(grid: np.ndarray, first_row: int, interval_s: float, step_s: float, altitude_m: np.ndarray,
                words: Words, n_candidates: int) -> np.ndarray:
    """``grid`` (a sentence on its ``step_s`` rows) on the Δ rows ``first_row``, ``first_row + Δ / step_s``, …: one
    row each (module docstring); ``altitude_m`` the altitude the grammar is read at, one per ``step_s`` row."""
    every = interval_rows(interval_s, step_s)
    rows = np.arange(first_row, len(grid), every)
    if len(rows) == 0:
        raise Refused("too short", f"no row on the {interval_s:g} s grid")
    held = in_force(np.asarray(grid))[rows]
    if held[0, RUNWAY] == RUNWAY_GO_AROUND:
        raise Refused("go-around at the first row", f"row {first_row} on the {interval_s:g} s grid")
    out = np.where(np.vstack([np.ones((1, held.shape[1]), dtype=bool), held[1:] != held[:-1]]), held, UNCHANGED)
    out = out.astype(np.int16)
    check_grammar(out, np.asarray(altitude_m)[rows], words, n_candidates)
    return out
