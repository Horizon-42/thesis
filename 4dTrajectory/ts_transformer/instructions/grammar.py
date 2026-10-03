"""Which words may be said at a step — the vocabulary's grammar (design §3.2 table, §3.7 rules 1–6), in ONE function
(`apply`) that the labeller checks every row with and a speaker masks with ("checked in labelling, a mask in decoding").

A step is one row of five words (`UNCHANGED` where a column says nothing). The words in force (`InForce`) carry the
runway in force R and the go-around state G beside the last word of each other column. The rules:

1. The first step says a value in every column; the runway column says a candidate.
2. The runway column (§3.2): a candidate k after the first step with G false only when k ≠ R (the word in force is not
   said again); with G true any candidate, R included — it ends G; "go-around" only with G false (and never at the
   first step). There is no runway lock (D12).
3. When an altitude or angle word is said, the level T in force and the angle in force agree at the aircraft's height:
   a T more than its band below the aircraft needs a descent class, more than its band above it the climb class.
4. "No level-off" needs a descent class in force or in the same row.
5. "No level-off" is not said while G is true (D14): a runway word ends G first — in the same row (the runway column
   comes first) or earlier.
6. A step that says "go-around" while "no level-off" is in force also says a level more than its band above the
   aircraft (D27): rule 3 then makes it say the climb. With rule 5, "no level-off" is never in force while G is true.
   ("Above the present height" read as rule 3 reads it, more than the level's band above: the reading under which
   rule 3 makes the row climb — Claude's reading of §3.7.)

A rule broken raises `Ungrammatical` with a fixed reason (counted by the labeller as a refusal).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, ANGLE_LEVEL, COLUMNS, HEADING, RUNWAY, RUNWAY_GO_AROUND, SPEED, UNCHANGED, Words,
)


class Ungrammatical(ValueError):
    """A step the grammar refuses; ``reason`` is a fixed category, ``detail`` says what was found."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class InForce:
    """The words in force after a step: the runway in force R (a candidate's index), the go-around state G, and the
    last word said in each of the other columns."""

    runway: int
    go_around: bool
    heading: int
    altitude: int
    angle: int
    speed: int


def apply(in_force: InForce | None, step: Sequence[int], altitude_m: float, words: Words,
          n_candidates: int) -> InForce:
    """The words in force after ``step`` (five words) is said after ``in_force`` (None: ``step`` is the first step) to
    an aircraft at ``altitude_m`` (geometric MSL), at an airport of ``n_candidates`` candidate runways; refused by
    `Ungrammatical` (module docstring)."""
    step = [int(v) for v in step]
    if len(step) != len(COLUMNS):
        raise ValueError(f"a step has {len(COLUMNS)} words, got {len(step)}")
    runway = step[RUNWAY]
    if in_force is None:
        missing = [COLUMNS[c] for c, value in enumerate(step) if value == UNCHANGED]
        if missing:
            raise Ungrammatical("first step incomplete", f"no word for {missing}")
        if not 0 <= runway < n_candidates:
            raise Ungrammatical("runway word not permitted", f"the first step says {runway}, not a candidate")
        after = InForce(runway=runway, go_around=False, heading=step[HEADING], altitude=step[ALTITUDE],
                        angle=step[ANGLE], speed=step[SPEED])
    else:
        after = in_force
        if runway == RUNWAY_GO_AROUND:
            if in_force.go_around:
                raise Ungrammatical("runway word not permitted", "go-around while a go-around is in force")
            if in_force.altitude == words.altitude_no_level_off:
                target = (None if step[ALTITUDE] in (UNCHANGED, words.altitude_no_level_off)
                          else words.altitude_m(step[ALTITUDE]))
                if target is None or target <= altitude_m + words.altitude_tolerance_m(step[ALTITUDE]):
                    raise Ungrammatical("go-around without a level above",
                                        f"\"no level-off\" in force at {altitude_m:.0f} m (rule 6)")
            after = replace(after, go_around=True)
        elif runway != UNCHANGED:
            if not 0 <= runway < n_candidates:
                raise Ungrammatical("runway word not permitted", f"{runway} is not a candidate of {n_candidates}")
            if runway == in_force.runway and not in_force.go_around:
                raise Ungrammatical("runway word not permitted", f"runway {runway} is already in force")
            after = replace(after, runway=runway, go_around=False)
        said = {column: step[column] for column in (HEADING, ALTITUDE, ANGLE, SPEED) if step[column] != UNCHANGED}
        after = replace(after, **{COLUMNS[column]: value for column, value in said.items()})
    if step[ALTITUDE] == words.altitude_no_level_off and after.go_around:
        raise Ungrammatical("no level-off during a go-around", "a runway word ends the go-around first (rule 5)")
    if step[ALTITUDE] != UNCHANGED or step[ANGLE] != UNCHANGED:
        target, angle = words.altitude_m(after.altitude), after.angle
        if target is None:
            if not words.is_descent(angle):
                raise Ungrammatical("no level-off without a descent", f"angle class {angle} in force (rule 4)")
        else:
            band = words.altitude_tolerance_m(after.altitude)
            if target < altitude_m - band:
                ok = words.is_descent(angle)
            elif target > altitude_m + band:
                ok = angle == words.angle_climb
            else:
                ok = True
            if not ok:
                raise Ungrammatical("altitude and angle incompatible",
                                    f"target {target:.0f} m at {altitude_m:.0f} m with angle class {angle}"
                                    + (" (level)" if angle == ANGLE_LEVEL else ""))
    return after


def step_allowed(in_force: InForce | None, step: Sequence[int], altitude_m: float, words: Words,
                 n_candidates: int) -> bool:
    """May ``step`` be said after ``in_force`` (`apply`)?"""
    try:
        apply(in_force, step, altitude_m, words, n_candidates)
    except Ungrammatical:
        return False
    return True


def runway_words_allowed(in_force: InForce | None, n_candidates: int) -> np.ndarray:
    """``[n_candidates + 1]`` bool: which runway words a step may say — each candidate, then "go-around" — after
    ``in_force`` (None: the first step) by rule 2; "unchanged" is always sayable after a first step and is not this
    function's to answer."""
    allowed = np.ones(n_candidates + 1, dtype=bool)
    if in_force is None:
        allowed[n_candidates] = False
    elif in_force.go_around:
        allowed[n_candidates] = False
    else:
        allowed[in_force.runway] = False
    return allowed
