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
from ts_transformer.instructions.words import UNCHANGED, Words


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
