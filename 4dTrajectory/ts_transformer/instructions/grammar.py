"""Which words may be said at a step — the vocabulary's grammar (vocabulary §3.2 table, §3.7 rules 1–6, §6 item 2):
one definition (`_rules`), which `apply` checks a row with (the labeller, the closed-loop reading) and `column_mask` gives
a speaker's mask with ("checked in labelling, a mask in decoding", D62).

A step is one row of five words (`UNCHANGED` where a column says nothing). The words in force (`InForce`) carry the
runway in force R and the go-around state G beside the last word of each other column. The rules:

1. The first step says a value in every column; the runway column says a candidate.
2. The runway column (§3.2): a candidate k after the first step with G false only when k ≠ R (the word in force is not
   said again); with G true any candidate, R included — it ends G; "go-around" only with G false (and never at the
   first step). There is no runway lock (D12).
3. When an altitude or angle word is said, the level T in force and the angle in force agree at the aircraft's height:
   a T more than its band below the aircraft needs a descent class, more than its band above it the climb class. Both
   are heights above the airport elevation E (D58).
4. "No level-off" needs a descent class in force or in the same row.
5. "No level-off" is not said while G is true (D14): a runway word ends G first — in the same row (the runway column
   comes first) or earlier.
6. A step that says "go-around" while "no level-off" is in force also says a level more than its band above the
   aircraft (D27): rule 3 then makes it say the climb. With rule 5, "no level-off" is never in force while G is true.
   ("Above the present height" read as rule 3 reads it, more than the level's band above: the reading under which
   rule 3 makes the row climb — Claude's reading of §3.7.)

A rule broken raises `Ungrammatical` with a fixed reason (counted by the labeller as a refusal).

THE MASK (D62). A speaker says a row column by column. `column_mask` gives, for every word of the column asked
("unchanged" and, in the runway column, "go-around" included), whether some words of the later columns, each among the
words the caller permits there (all, when not given), make the row pass `apply` after the earlier columns' words: a
check with the later columns "unchanged" would refuse good words (a level below the aircraft, which a descent class in
the angle column makes grammatical), and the permitted words of the later columns make sure that a row never reaches a
column with no permitted word. It is `apply` itself: `_rules` is written once over operations that take a single row
(`_ONE`, for `apply`) or numpy arrays (`_MANY`), and the mask evaluates it at once on every completion of the later
columns, for a batch of aircraft. A heading or a speed word enters the rules only as said or "unchanged" (rule 1), so
each of those columns is evaluated at "unchanged" and at one said word that stands for every said word — the tests
check the mask against every completion through `apply`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


#: What `_rules` returns for a broken rule, in the order the rules are read (the first broken one is the answer): the
#: reason (the labeller counts refusals by it) and what was found.
BROKEN = (
    ("first step incomplete", "no word for {missing}"),
    ("runway word not permitted", "the first step says {runway}, not a candidate"),
    ("runway word not permitted", "go-around while a go-around is in force"),
    ("go-around without a level above", "\"no level-off\" in force at {height:.0f} m above the airport (rule 6)"),
    ("runway word not permitted", "{runway} is not a candidate of {candidates}"),
    ("runway word not permitted", "runway {runway} is already in force"),
    ("no level-off during a go-around", "a runway word ends the go-around first (rule 5)"),
    ("no level-off without a descent", "angle class {angle} in force (rule 4)"),
    ("altitude and angle incompatible", "target {target:.0f} m at {height:.0f} m above the airport with angle class "
                                        "{angle}{level}"),
)


class _One:
    """`_rules` on one row: Python scalars."""

    @staticmethod
    def where(condition: Any, yes: Any, no: Any) -> Any:
        return yes if condition else no

    @staticmethod
    def not_(value: Any) -> Any:
        return not value

    @staticmethod
    def lookup(table: np.ndarray, index: int) -> float:
        return float(table[index]) if index >= 0 else math.nan

    @staticmethod
    def zeros(like: Any) -> int:
        return 0

    isnan = staticmethod(math.isnan)


class _Many:
    """`_rules` on numpy arrays, broadcast."""

    where = staticmethod(np.where)
    not_ = staticmethod(np.logical_not)

    @staticmethod
    def lookup(table: np.ndarray, index: np.ndarray) -> np.ndarray:
        return np.where(index >= 0, table[np.clip(index, 0, None)], np.nan)

    @staticmethod
    def zeros(like: Any) -> np.ndarray:
        return np.zeros(np.shape(like), dtype=np.int64)

    isnan = staticmethod(np.isnan)


_ONE, _MANY = _One(), _Many()


def _tables(words: Words) -> tuple[np.ndarray, np.ndarray]:
    """Each altitude word's level above E and the half width of its band, by the word's value ("no level-off": no level;
    its band the lowest level's, `Words.altitude_tolerance_m`)."""
    levels = np.append(words.altitude_levels, np.nan)
    bands = np.append(words.altitude_tolerances, words.altitude_tolerances[0])
    return levels, bands


def _require_altitudes(values: Any, words: Words) -> None:
    """An altitude word outside the grid ("unchanged" to "no level-off") is refused by name, in `apply` and in
    `column_mask` alike: it is no word of the vocabulary."""
    values = np.asarray(values)
    outside = values[(values < UNCHANGED) | (values > words.altitude_no_level_off)]
    if len(outside):
        raise ValueError(f"altitude class {int(outside[0])} outside 0..{words.altitude_no_level_off}")


def _require_words(step: Sequence[int], words: Words) -> None:
    """A heading, altitude, angle or speed word outside its column's words ("unchanged" to the column's last class) is
    refused by name in `apply` (vocabulary D80): it is no word of the vocabulary. The runway column's values are the
    rules' own (rules 1, 2)."""
    counts = words.class_counts()
    for column in (HEADING, ALTITUDE, ANGLE, SPEED):
        if not UNCHANGED <= step[column] < counts[COLUMNS[column]]:
            raise ValueError(f"{COLUMNS[column]} class {step[column]} outside 0..{counts[COLUMNS[column]] - 1}")


def _rules(ops: Any, words: Words, first: Any, in_runway: Any, in_go: Any, in_altitude: Any, in_angle: Any,
           runway: Any, heading: Any, altitude: Any, angle: Any, speed: Any, height: Any, candidates: Any
           ) -> tuple[Any, Any, Any, Any, Any]:
    """Rules 1–6 (module docstring) on a step said after the words in force (``first``: the step is the first; the
    words in force are then not read), to an aircraft ``height`` above E at an airport of ``candidates`` runways:
    ``(code, runway, go-around, altitude, angle)`` after the step, ``code`` 0 when it passes, else 1 + the place in
    `BROKEN` of the first rule it breaks. ``ops`` is `_ONE` or `_MANY`."""
    levels, bands = _tables(words)
    no_level_off, unchanged = words.altitude_no_level_off, UNCHANGED
    code = ops.zeros(height)

    def broken(code: Any, condition: Any, number: int) -> Any:
        return ops.where((code == 0) & condition, number, code)

    later = ops.not_(first)
    missing = ((runway == unchanged) | (heading == unchanged) | (altitude == unchanged) | (angle == unchanged)
               | (speed == unchanged))
    outside = (runway < 0) | (runway >= candidates)
    go_around = runway == RUNWAY_GO_AROUND
    candidate = (runway != unchanged) & ops.not_(go_around)
    climb_to = ops.lookup(levels, altitude)                       # the level said with "go-around" (rule 6)
    not_above = ops.isnan(climb_to) | (climb_to <= height + ops.lookup(bands, altitude))   # no level, or not above
    code = broken(code, first & missing, 1)
    code = broken(code, first & outside, 2)
    code = broken(code, later & go_around & in_go, 3)
    code = broken(code, later & go_around & (in_altitude == no_level_off) & not_above, 4)
    code = broken(code, later & candidate & outside, 5)
    code = broken(code, later & candidate & (runway == in_runway) & ops.not_(in_go), 6)
    after_runway = ops.where(later & ops.not_(candidate), in_runway, runway)
    after_go = later & (go_around | (ops.not_(candidate) & in_go))
    after_altitude = ops.where(later & (altitude == unchanged), in_altitude, altitude)
    after_angle = ops.where(later & (angle == unchanged), in_angle, angle)
    code = broken(code, (altitude == no_level_off) & after_go, 7)
    said = (altitude != unchanged) | (angle != unchanged)
    descent = (after_angle >= ANGLE_LEVEL + 1) & (after_angle <= words.n_descent)
    target = ops.lookup(levels, after_altitude)
    band = ops.lookup(bands, after_altitude)
    agree = ops.where(target < height - band, descent,
                      ops.where(target > height + band, after_angle == words.angle_climb, True))
    code = broken(code, said & (after_altitude == no_level_off) & ops.not_(descent), 8)
    code = broken(code, said & (after_altitude != no_level_off) & ops.not_(agree), 9)
    return code, after_runway, after_go, after_altitude, after_angle


def apply(in_force: InForce | None, step: Sequence[int], height_m: float, words: Words,
          n_candidates: int) -> InForce:
    """The words in force after ``step`` (five words) is said after ``in_force`` (None: ``step`` is the first step) to
    an aircraft ``height_m`` above the airport elevation E (D58), at an airport of ``n_candidates`` candidate runways;
    refused by `Ungrammatical` (module docstring)."""
    step = [int(v) for v in step]
    if len(step) != len(COLUMNS):
        raise ValueError(f"a step has {len(COLUMNS)} words, got {len(step)}")
    _require_words(step, words)
    before = InForce(UNCHANGED, False, UNCHANGED, UNCHANGED, UNCHANGED, UNCHANGED) if in_force is None else in_force
    code, runway, go_around, altitude, angle = _rules(
        _ONE, words, in_force is None, before.runway, before.go_around, before.altitude, before.angle, *step,
        float(height_m), n_candidates)
    if code:
        reason, detail = BROKEN[code - 1]
        target = _ONE.lookup(_tables(words)[0], altitude)
        raise Ungrammatical(reason, detail.format(
            missing=[COLUMNS[c] for c, value in enumerate(step) if value == UNCHANGED], runway=step[RUNWAY],
            candidates=n_candidates, height=height_m, angle=angle, target=target,
            level=" (level)" if angle == ANGLE_LEVEL else ""))
    heading = before.heading if step[HEADING] == UNCHANGED else step[HEADING]
    speed = before.speed if step[SPEED] == UNCHANGED else step[SPEED]
    return InForce(runway=int(runway), go_around=bool(go_around), heading=heading, altitude=int(altitude),
                   angle=int(angle), speed=speed)


def step_allowed(in_force: InForce | None, step: Sequence[int], height_m: float, words: Words,
                 n_candidates: int) -> bool:
    """May ``step`` be said after ``in_force`` (`apply`)?"""
    try:
        apply(in_force, step, height_m, words, n_candidates)
    except Ungrammatical:
        return False
    return True


def column_words(column: int, words: Words, n_candidates: int) -> np.ndarray:
    """The words of ``column`` in the order of `column_mask`'s answer: "unchanged" first; in the runway column then
    "go-around" and the candidates; in the others the column's classes ("no level-off" and "unspecified" last, as
    `Words` numbers them)."""
    if column == RUNWAY:
        return np.array([UNCHANGED, RUNWAY_GO_AROUND, *range(n_candidates)], dtype=np.int64)
    return np.arange(UNCHANGED, words.class_counts()[COLUMNS[column]], dtype=np.int64)


def column_mask(in_force: Sequence[InForce | None], said: np.ndarray, column: int, height_m: np.ndarray,
                n_candidates: Sequence[int], words: Words, permitted: Mapping[int, np.ndarray] | None = None
                ) -> np.ndarray:
    """D62, for a batch of aircraft: ``[B, words]`` (`column_words` of the batch's most candidates; a candidate an
    aircraft's airport does not have is never permitted): whether each word of ``column`` can be said after the words
    in force ``in_force`` (None: the first step) and the earlier columns' words ``said`` (``[B, column]``), to an
    aircraft ``height_m`` above E — whether some words of the later columns, each among ``permitted[c]`` (``[B, words of
    column c]``, `column_words` order; every word where not given), make the row pass `apply` (module docstring)."""
    batch = len(in_force)
    said = np.asarray(said, dtype=np.int64).reshape(batch, column)
    if column > ALTITUDE:
        _require_altitudes(said[:, ALTITUDE], words)
    later = list(range(column + 1, len(COLUMNS)))
    permitted = dict(permitted or {})
    if set(permitted) - set(later):
        raise ValueError(f"permitted words are for the later columns {later}, got {sorted(permitted)}")
    most = int(max(n_candidates))
    width = 2 + len(later)                                     # the batch, the column asked, then each later column
    axis = {c: 2 + k for k, c in enumerate(later)}

    def along(values: np.ndarray, dim: int) -> np.ndarray:
        shape = [1] * width
        shape[0], shape[dim] = values.shape[0], values.shape[1]
        return values.reshape(shape)

    def reduced(c: int) -> bool:                               # read only as said or "unchanged" (module docstring)
        return c in (HEADING, SPEED)

    asked = column_words(column, words, most)
    values = np.array([[UNCHANGED, 0]]) if reduced(column) else asked[None, :]
    row: dict[int, np.ndarray] = {column: along(np.repeat(values, batch, axis=0), 1)}
    allowed = np.ones((batch,) + (1,) * (width - 1), dtype=bool)
    for c in range(column):
        row[c] = said[:, c].reshape((batch,) + (1,) * (width - 1))
    for c in later:
        full = column_words(c, words, most)
        given = np.asarray(permitted.get(c, np.ones((batch, len(full)), dtype=bool)), dtype=bool)
        if given.shape != (batch, len(full)):
            raise ValueError(f"the permitted words of column {COLUMNS[c]} are [{batch}, {len(full)}], got {given.shape}")
        if reduced(c):
            values = np.tile([UNCHANGED, 0], (batch, 1))
            given = np.stack([given[:, 0], given[:, 1:].any(axis=1)], axis=1)
        else:
            values = np.tile(full, (batch, 1))
        row[c] = along(values, axis[c])
        allowed = allowed & along(given, axis[c])
    before = [InForce(UNCHANGED, False, UNCHANGED, UNCHANGED, UNCHANGED, UNCHANGED) if f is None else f for f in in_force]

    def per_aircraft(values: list[Any], dtype: Any) -> np.ndarray:
        return np.asarray(values, dtype=dtype).reshape((batch,) + (1,) * (width - 1))

    code = _rules(_MANY, words, per_aircraft([f is None for f in in_force], bool),
                  per_aircraft([f.runway for f in before], np.int64), per_aircraft([f.go_around for f in before], bool),
                  per_aircraft([f.altitude for f in before], np.int64), per_aircraft([f.angle for f in before], np.int64),
                  *(row[c] for c in range(len(COLUMNS))), per_aircraft(height_m, np.float64),
                  per_aircraft(list(n_candidates), np.int64))[0]
    passes = ((code == 0) & allowed).any(axis=tuple(range(2, width)))
    if reduced(column):
        return np.concatenate([passes[:, :1], np.repeat(passes[:, 1:], len(asked) - 1, axis=1)], axis=1)
    return passes
