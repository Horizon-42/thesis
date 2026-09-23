"""Instructions → the sentence: one row of six words per step (vocabulary design §2, §3.5).

Step 0 carries a concrete word in every column; every later step writes a word only where an
instruction was issued, "unchanged" elsewhere. A word equal to the one in force is not an
instruction and is dropped (so is a second copy of one word in one step). Two different
instructions of one column in one step — checked on the full list, before anything is
dropped — an incomplete step 0, or a violation of the compatibility rules (§2.1, §2.5) refuse
the flight.
"""

from __future__ import annotations

import numpy as np

from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, APPROACH, APPROACH_CLEARED, COLUMNS, RUNWAY, UNCHANGED, Words,
)


def assemble(n_rows: int, instructions: list[Instruction], altitude: np.ndarray, spec: VocabularySpec,
             words: Words) -> tuple[np.ndarray, list[Instruction]]:
    issued: dict[tuple[int, int], int] = {}
    for item in instructions:
        cell = (item.row, item.column)
        if cell in issued and issued[cell] != item.value:
            raise Refused("two instructions in one step",
                          f"{COLUMNS[item.column]} at row {item.row}: {issued[cell]} and {item.value}")
        issued[cell] = item.value
    grid = np.full((n_rows, len(COLUMNS)), UNCHANGED, dtype=np.int16)
    in_force: list[int | None] = [None] * len(COLUMNS)
    kept: list[Instruction] = []
    for item in sorted(instructions, key=lambda i: (i.row, i.column)):
        if grid[item.row, item.column] != UNCHANGED or (item.row > 0 and in_force[item.column] == item.value):
            continue
        grid[item.row, item.column] = item.value
        in_force[item.column] = item.value
        kept.append(item)
    missing = [COLUMNS[c] for c in range(len(COLUMNS)) if grid[0, c] == UNCHANGED]
    if missing:
        raise Refused("step 0 incomplete", f"no word for {missing}")
    _check_compatibility(grid, altitude, spec, words)
    return grid, kept


def _check_compatibility(grid: np.ndarray, altitude: np.ndarray, spec: VocabularySpec, words: Words) -> None:
    current = [int(v) for v in grid[0]]
    for row in range(len(grid)):
        changed = grid[row] != UNCHANGED
        previous = list(current)
        for column in np.nonzero(changed)[0]:
            current[column] = int(grid[row, column])
        if changed[RUNWAY] and row > 0 and previous[APPROACH] == APPROACH_CLEARED and not changed[APPROACH]:
            raise Refused("runway changed under a clearance", f"row {row}")
        if not (changed[ALTITUDE] or changed[ANGLE]):
            continue
        target, angle = words.altitude_m(current[ALTITUDE]), current[ANGLE]
        if target is None:
            ok = words.is_descent(angle)
        elif target < altitude[row] - spec.altitude_tolerance_m:
            ok = words.is_descent(angle)
        elif target > altitude[row] + spec.altitude_tolerance_m:
            ok = angle == words.angle_climb
        else:
            ok = True
        if not ok:
            raise Refused("altitude and angle incompatible",
                          f"row {row}: target {target} m at {altitude[row]:.0f} m with angle class {angle}"
                          + (" (level)" if angle == ANGLE_LEVEL else ""))
