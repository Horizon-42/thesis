"""A sentence read at the data's 2 s rows put on a coarser grid of rows Δ apart — the row-interval ablation (design
§4.8, D11). One reading gives every Δ; nothing is read again.

1. The Δ rows are the rows on UTC multiples of Δ (the rule that put artefact `v6_20261002`'s rows on even seconds, so
   the aircraft of a scene stay on one grid): the first is `first_interval_row`, then every Δ / step rows.
2. Each word of the step rows goes to the nearest Δ row; a word exactly between two Δ rows goes to the later one (D45:
   the next Δ row would make a word (Δ − step) / 2 late on average; the nearest makes it step / 2 late at Δ ≥ 4 s, the
   ties going later, and on time at the step). A Δ row says, for each
   column, the last word that goes to it if it differs from the word in force: a go-around and the runway said again
   that go to one Δ row cancel. In the runway column the word in force is the column's last word, "go-around" included.
   The rounding keeps the order of the words, and the words of one step row stay in one Δ row. Words that would go to a
   Δ row past the sentence's last are not said (the sentence ends there).
3. The first Δ row says the five words in force there (every word that goes to it); a sentence in a go-around at its
   first Δ row cannot say a runway there and is refused. The grammar (`instructions.grammar`) is checked again on the Δ
   rows, at their height above the airport elevation E (D58).
4. A heading word is said in the frame where it is heard (D46, §3.3): the step rows' word says an absolute track (its
   class under the course of R at its row); the Δ row that says it gives the class nearest that track minus the course
   of the R in force at the Δ row (the same class when R did not change, or changed to a parallel runway). A Δ row says
   a heading word when a new heading word of the step rows goes to it and its track differs from the track the Δ rows
   have in force — also when its class is the class in force, said under another course (Claude's reading: a class
   says a track only with its course; the step rows say it too, D48, `labeller.sentence`). A change
   of R alone says no heading word (the executor keeps its absolute track, §3.3).

Δ must be a whole number of steps and divide the prior's 16 s observation (D25: Δ = 2, 4, 8 s; 6 s is refused); at
Δ = the step the sentence comes back unchanged.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Sequence

import numpy as np

from ts_transformer.data.day_split import parse_utc
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.labeller.sentence import check_grammar
from ts_transformer.instructions.words import HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words, same_track


#: The prior's observation before its first predicted step, s (vocabulary §2, `N_LOOK` = 8 rows at 2 s): every Δ divides it,
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


def on_interval_rows(n_rows: int, every: int) -> np.ndarray:
    """[n_rows] bool: which of a sentence's step rows, from its first Δ row, are its Δ rows (every ``every``-th, D51)."""
    return np.arange(n_rows) % every == 0


def last_heard_row(row: float, every: int) -> int:
    """The last step row whose time is less than Δ/2 (``every`` step rows) after ``row`` (in step rows): the last word a
    Δ row at ``row`` says (item 2, D45: a word exactly between two Δ rows goes to the later one). The closed-loop reading
    asks it at the matched point's observed time (`autopilot.closed_loop`), the Δ grid at its rows."""
    return int(math.ceil(row + every / 2.0)) - 1


def later_utc(entry_time_utc: str, seconds: float) -> str:
    """The UTC time ``seconds`` after ``entry_time_utc``, in the artefact's form (whole seconds): a sentence's first row
    on a coarser grid starts there."""
    return (parse_utc(entry_time_utc) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def in_force(grid: np.ndarray) -> np.ndarray:
    """``[N, 5]``: each column's last word at or before each row (row 0 must say every column)."""
    if (grid[0] == UNCHANGED).any():
        raise ValueError("a sentence's row 0 says every column")
    written = grid != UNCHANGED
    last = np.maximum.accumulate(np.where(written, np.arange(len(grid))[:, None], 0), axis=0)
    return np.take_along_axis(grid, last, axis=0)


def on_utc_grid(grid: np.ndarray, entry_time_utc: str, interval_s: float, step_s: float, height_m: np.ndarray,
                words: Words, courses_deg: Sequence[float]) -> tuple[int, np.ndarray]:
    """A labelled sentence on its UTC Δ grid (`first_interval_row` of its first row's UTC time, then `on_interval`): the
    2 s row of its first Δ row and its words there. The replay and the labeller conformance read it alike (D49)."""
    first = first_interval_row(entry_time_utc, interval_s, step_s)
    return first, on_interval(grid, first, interval_s, step_s, height_m, words, courses_deg)


def on_interval(grid: np.ndarray, first_row: int, interval_s: float, step_s: float, height_m: np.ndarray,
                words: Words, courses_deg: Sequence[float]) -> np.ndarray:
    """``grid`` (a sentence on its ``step_s`` rows) on the Δ rows ``first_row``, ``first_row + Δ / step_s``, …: one
    row each, every word on the nearest (module docstring); ``height_m`` the height above E the grammar is read at, one per
    ``step_s`` row; ``courses_deg`` the candidates' courses."""
    every = interval_rows(interval_s, step_s)
    rows = np.arange(first_row, len(grid), every)
    if len(rows) == 0:
        raise Refused("too short", f"no row on the {interval_s:g} s grid")
    grid = np.asarray(grid)
    # item 2: Δ row k takes the step rows up to `last[k]` (a tie goes to the later Δ row)
    last = np.minimum([last_heard_row(row, every) for row in rows], len(grid) - 1)
    force = in_force(grid)
    held = force[last]
    if held[0, RUNWAY] == RUNWAY_GO_AROUND:
        raise Refused("go-around at the first row", f"row {first_row} on the {interval_s:g} s grid")
    out = np.where(np.vstack([np.ones((1, held.shape[1]), dtype=bool), held[1:] != held[:-1]]), held, UNCHANGED)
    out = out.astype(np.int16)
    # item 4: each step row's heading word in force as an absolute track (said under the course of R at its row), and
    # the Δ rows' heading words in the frame of the R in force where they are heard
    steps = np.arange(len(grid))
    runway = grid[np.maximum.accumulate(np.where(grid[:, RUNWAY] >= 0, steps, 0)), RUNWAY]
    said = np.maximum.accumulate(np.where(grid[:, HEADING] != UNCHANGED, steps, 0))
    out[:, HEADING] = UNCHANGED
    track = source = None
    for position, step in enumerate(last):
        if said[step] == source:
            continue
        source = said[step]
        wanted = courses_deg[runway[source]] + words.heading_relative_deg(int(force[step, HEADING]))
        if track is None or not same_track(wanted, track):
            out[position, HEADING] = words.heading_class(wanted, courses_deg[runway[step]])
            track = courses_deg[runway[step]] + words.heading_relative_deg(int(out[position, HEADING]))
    check_grammar(out, np.asarray(height_m)[rows], words, len(courses_deg))
    return out
