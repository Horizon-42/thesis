"""Which words may be said at a step — the vocabulary's compatibility rules (vocabulary design §2.1, §2.5) asked of one
step at a time, for a speaker that masks what they forbid (the design: "checked in labelling, a mask in decoding").

The rules are the labeller's own check (`labeller.sentence._check_compatibility`), called on a two-row sentence: the
words in force before the step, then the step. Row 0 is read at no altitude, so only row 1 is judged — a runway changed
under a clearance with the approach left as it is; an altitude and a descent-angle class that disagree at the
aircraft's altitude, "descend to land" without a descent class among them. Not in the labeller's source hash, so asking
it changes no sentence.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.labeller.sentence import _check_compatibility
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    APPROACH_CLASSES, APPROACH_CLEARED, APPROACH_GO_AROUND, APPROACH_NOT_CLEARED, UNCHANGED, Words,
)

#: The approach column's transitions a speaker may say (multi-aircraft design §6.6 step 8 item 6, the user's 2026-10-02
#: decision): not cleared → cleared → go-around → cleared. No clearance is withdrawn by "not cleared" (after a clearance
#: only a go-around leaves it), "not cleared" and the go-around both come before a clearance and never follow each other,
#: and the word in force is not said again. A DECODING rule only: the labeller never says a go-around nor "not cleared"
#: after a clearance, so its check (and its source hash) is left as it is; words given to a loop are not masked.
APPROACH_NEXT = {APPROACH_NOT_CLEARED: (APPROACH_CLEARED,), APPROACH_CLEARED: (APPROACH_GO_AROUND,),
                 APPROACH_GO_AROUND: (APPROACH_CLEARED,)}
#: what a first step (every column said, nothing in force) may say there
APPROACH_FIRST = (APPROACH_NOT_CLEARED, APPROACH_CLEARED)


def step_allowed(in_force: Sequence[int] | None, step: Sequence[int], altitude_m: float, spec: VocabularySpec,
                 words: Words) -> bool:
    """May ``step`` (six words, `UNCHANGED` where a column says nothing) be said after the words ``in_force`` (six
    words; None before the first step, which must say every column) at ``altitude_m``?"""
    if in_force is None:
        grid, altitude = np.array([step], dtype=np.int64), np.array([altitude_m])
    else:
        grid, altitude = np.array([in_force, step], dtype=np.int64), np.array([math.nan, altitude_m])
    if (grid[0] == UNCHANGED).any():
        raise ValueError("the words in force (or a first step) name every column")
    try:
        _check_compatibility(grid, altitude, spec, words)
    except Refused:
        return False
    return True


def approach_words_allowed(in_force: int | None) -> np.ndarray:
    """``[APPROACH_CLASSES]`` bool: the approach words a step may say after the approach word ``in_force`` (None: the
    first step) — `APPROACH_NEXT`, `APPROACH_FIRST`; "unchanged" is always sayable after a first step and is not this
    function's to answer."""
    out = np.zeros(APPROACH_CLASSES, dtype=bool)
    out[list(APPROACH_FIRST if in_force is None else APPROACH_NEXT[in_force])] = True
    return out

