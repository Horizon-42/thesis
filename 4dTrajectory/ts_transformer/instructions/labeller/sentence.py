"""Instructions → the sentence: one row of five words per step (design §3.1, §4.7).

Step 0 carries a concrete word in every column; every later step writes a word only where an instruction was issued,
"unchanged" elsewhere. A word equal to the column's last word is not an instruction and is dropped (so is a second
copy of one word in one step) — in the runway column the last word is "go-around" while one is in force, so the runway
said again after it is kept. Two different instructions of one column in one step — checked on the full list, before
anything is dropped — refuse the flight, and so does any row the grammar refuses (`instructions.grammar.apply`, read at
each row's altitude), an incomplete step 0 included.
"""

from __future__ import annotations

import numpy as np

from ts_transformer.instructions.grammar import Ungrammatical, apply
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words


def assemble(n_rows: int, instructions: list[Instruction], altitude: np.ndarray, words: Words,
             n_candidates: int) -> tuple[np.ndarray, list[Instruction]]:
    issued: dict[tuple[int, int], int] = {}
    for item in instructions:
        cell = (item.row, item.column)
        if cell in issued and issued[cell] != item.value:
            raise Refused("two instructions in one step",
                          f"{COLUMNS[item.column]} at row {item.row}: {issued[cell]} and {item.value}")
        issued[cell] = item.value
    grid = np.full((n_rows, len(COLUMNS)), UNCHANGED, dtype=np.int16)
    last: list[int | None] = [None] * len(COLUMNS)
    kept: list[Instruction] = []
    for item in sorted(instructions, key=lambda i: (i.row, i.column)):
        if grid[item.row, item.column] != UNCHANGED or (item.row > 0 and last[item.column] == item.value):
            continue
        grid[item.row, item.column] = item.value
        last[item.column] = item.value
        kept.append(item)
    check_grammar(grid, altitude, words, n_candidates)
    return grid, kept


def check_grammar(grid: np.ndarray, altitude: np.ndarray, words: Words, n_candidates: int) -> None:
    """Every row of ``grid`` through the grammar, at its altitude; a refusal names the row."""
    in_force = None
    for row in range(len(grid)):
        try:
            in_force = apply(in_force, grid[row], float(altitude[row]), words, n_candidates)
        except Ungrammatical as error:
            raise Refused(error.reason, f"row {row}: {error.detail}") from None
