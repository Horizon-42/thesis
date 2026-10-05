"""Instructions → the sentence: one row of five words per step (vocabulary §3.1, §4.7).

Step 0 carries a concrete word in every column; every later step writes a word only where an instruction was issued,
"unchanged" elsewhere. A word equal to the column's last word is not an instruction and is dropped (so is a second
copy of one word in one step) — in the runway column the last word is "go-around" while one is in force, so the runway
said again after it is kept; in the heading column a word is its absolute track (its class under the course of the
runway in force at its row, §3.3), so after a change of course the class in force is said again (D48). Two different instructions of one column in one step — checked on the full list, before
anything is dropped — refuse the flight, and so does any row the grammar refuses (`instructions.grammar.apply`, read at
each row's height above the airport elevation E, D58), an incomplete step 0 included, and an instruction outside the
sentence's rows.
"""

from __future__ import annotations

import numpy as np

from typing import Sequence

from ts_transformer.instructions.grammar import Ungrammatical, apply
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.words import COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words, same_track


def assemble(n_rows: int, instructions: list[Instruction], height: np.ndarray, words: Words,
             courses_deg: Sequence[float]) -> tuple[np.ndarray, list[Instruction]]:
    """The sentence of ``instructions`` (module docstring); ``height`` each row's height above E, ``courses_deg`` the
    candidates' courses, by candidate."""
    issued: dict[tuple[int, int], int] = {}
    for item in instructions:
        if not 0 <= item.row < n_rows:
            raise Refused("instruction outside the sentence", f"{COLUMNS[item.column]} at row {item.row} of {n_rows}")
        cell = (item.row, item.column)
        if cell in issued and issued[cell] != item.value:
            raise Refused("two instructions in one step",
                          f"{COLUMNS[item.column]} at row {item.row}: {issued[cell]} and {item.value}")
        issued[cell] = item.value
    missing = [COLUMNS[c] for c in range(len(COLUMNS)) if (0, c) not in issued]
    if missing:                              # a heading word reads the course of the runway and the track in force
        raise Refused("first step incomplete", f"no word for {missing}")
    grid = np.full((n_rows, len(COLUMNS)), UNCHANGED, dtype=np.int16)
    last: list[int | None] = [None] * len(COLUMNS)
    track: float | None = None                  # the track the heading word in force says (D48)
    runway = issued[(0, RUNWAY)]
    kept: list[Instruction] = []
    for item in sorted(instructions, key=lambda i: (i.row, i.column)):   # the runway column comes first in a row
        if item.column == RUNWAY and item.value != RUNWAY_GO_AROUND:
            runway = item.value
        if grid[item.row, item.column] != UNCHANGED:
            continue
        if item.column == HEADING:
            said = words.heading_track_deg(item.value, courses_deg[runway])
            if item.row > 0 and same_track(said, track):
                continue
            track = said
        elif item.row > 0 and last[item.column] == item.value:
            continue
        grid[item.row, item.column] = item.value
        last[item.column] = item.value
        kept.append(item)
    check_grammar(grid, height, words, len(courses_deg))
    return grid, kept


def check_grammar(grid: np.ndarray, height: np.ndarray, words: Words, n_candidates: int) -> None:
    """Every row of ``grid`` through the grammar, at its height above E; a refusal names the row."""
    in_force = None
    for row in range(len(grid)):
        try:
            in_force = apply(in_force, grid[row], float(height[row]), words, n_candidates)
        except Ungrammatical as error:
            raise Refused(error.reason, f"row {row}: {error.detail}") from None
