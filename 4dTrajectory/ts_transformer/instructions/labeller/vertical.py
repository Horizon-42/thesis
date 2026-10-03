"""Altitude words and descent-angle words (design §3.4, §3.5, §4.4).

The smoothed altitude is fitted against horizontal distance by straight pieces. A piece that lasts the minimum level
time with every row within `level_band_m` of its own median is a LEVEL — a physical test that does not use the grid
(§4.4: the grid's 60–450 m steps would miss a level between two of its levels or swallow a descent) — and its word is
the grid level nearest that median; consecutive level pieces with the same word merge. Every other piece is a MOVE with
a path angle. Consecutive moves in one direction form a run; a run's altitude word is the next level's word (the
turning altitude's when the direction reverses without a level; "no level-off" when the last run descends to the end),
issued where the run begins. An angle word is issued wherever a move's class differs from the class in force. A run
whose word is the altitude word in force — a move between two heights that round to one level — says nothing: it cannot
be said (§3.4). Two levels that meet with no move between them are a STEP: the altitude left the first level inside the
fitted pieces; its word is issued at the first level's last row, its angle read over the rows from there to where the
second level's target is met.

A flight with go-arounds is read approach by approach (D26, §4.6). Each go-around's climb is the first climb — a
climbing run, or a step up — still under way at its lowest point (`climb_after`); the GO-AROUND ROW is the row the climb
is said at, its level word and its climb word in that row. An approach ends there, so the descent run straight before
that climb is the last descent that reaches the end of its approach and says "no level-off", as the last run before
the threshold does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.piecewise import fit_pieces
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ALTITUDE, ANGLE, ANGLE_LEVEL, Words

LEVEL = "level"
MOVE = "move"


@dataclass
class VerticalPiece:
    start: int
    stop: int  # one past the last row
    kind: str                   # LEVEL or MOVE
    target_index: int = -1      # LEVEL: the altitude class
    median_m: float = 0.0       # LEVEL: the height it is held at (its first piece's median)
    angle_deg: float = 0.0      # MOVE: path angle, descending positive

    @property
    def rows(self) -> int:
        return self.stop - self.start

    def descending(self, spec: VocabularySpec) -> bool:
        return self.angle_deg >= spec.descent_angle_edges_deg[0]


@dataclass
class VerticalReading:
    instructions: list[Instruction]
    pieces: list[VerticalPiece]
    #: Each go-around's row, in the order of the lowest points given (module docstring).
    go_around_rows: list[int]


def vertical_pieces(distance: np.ndarray, altitude: np.ndarray, spec: VocabularySpec, words: Words) -> list[VerticalPiece]:
    minimum = spec.rows(spec.level_min_s)
    pieces: list[VerticalPiece] = []
    for piece in fit_pieces(distance, altitude, spec.altitude_fit_tolerance_m):
        rows = altitude[piece.start: piece.stop]
        median = float(np.median(rows))
        if piece.rows >= minimum and bool(envelope.level_band(rows, median, spec.level_band_m).all()):
            try:
                index = words.altitude_index(median)
            except ValueError as error:
                raise Refused("altitude target out of range", str(error)) from None
            previous = pieces[-1] if pieces else None
            if previous is not None and previous.kind == LEVEL and previous.target_index == index:
                previous.stop = piece.stop          # the same level word
            else:
                pieces.append(VerticalPiece(piece.start, piece.stop, LEVEL, target_index=index, median_m=median))
        else:
            pieces.append(VerticalPiece(piece.start, piece.stop, MOVE, angle_deg=math.degrees(math.atan(-piece.slope))))
    return pieces


def held_altitude(pieces: list[VerticalPiece], altitude: np.ndarray) -> np.ndarray:
    """The height the aircraft holds at each row, as the reading has it: a level piece's rows at the height the level is
    held at (its median; a level's rows wander up to `level_band_m` about it), every other row at its own altitude —
    what the grammar reads a row's words at (`labeller.sentence.check_grammar`)."""
    held = np.array(altitude, dtype=np.float64)
    for piece in pieces:
        if piece.kind == LEVEL:
            held[piece.start: piece.stop] = piece.median_m
    return held


def _runs(pieces: list[VerticalPiece], spec: VocabularySpec) -> list[list[VerticalPiece]]:
    """Group the pieces: a level alone, or consecutive moves in one direction."""
    groups: list[list[VerticalPiece]] = []
    for piece in pieces:
        previous = groups[-1][-1] if groups else None
        if (piece.kind == MOVE and previous is not None and previous.kind == MOVE
                and previous.descending(spec) == piece.descending(spec)):
            groups[-1].append(piece)
        else:
            groups.append([piece])
    return groups


def _climb_start(groups: list[list[VerticalPiece]], position: int, spec: VocabularySpec) -> tuple[int, int] | None:
    """``(row said, last row climbing)`` of the climb group ``position`` starts — a climbing run, or a level stepping up
    from the level before it (`_step`'s row) — or None when it starts none."""
    group = groups[position]
    if group[0].kind == MOVE:
        return None if group[0].descending(spec) else (group[0].start, group[-1].stop - 1)
    previous = groups[position - 1][0] if position else None
    if previous is None or previous.kind != LEVEL or group[0].target_index <= previous.target_index:
        return None
    return previous.stop - 1, group[0].start


def climb_after(groups: list[list[VerticalPiece]], point: int, before: int, spec: VocabularySpec) -> int:
    """The group of the go-around climb from the lowest point ``point``: the first climb (`_climb_start`) still under way
    at ``point``, said before row ``before`` (the next go-around's lowest point, or the end); refused when there is
    none."""
    for position in range(len(groups)):
        climb = _climb_start(groups, position, spec)
        if climb is not None and climb[1] >= point and climb[0] < before:
            return position
    raise Refused("go-around without a climb word", f"no climb after the lowest point at row {point}")


def read_vertical(distance: np.ndarray, altitude: np.ndarray, spec: VocabularySpec, words: Words,
                  go_around_points: list[int]) -> VerticalReading:
    """The altitude and angle words; ``go_around_points`` each go-around's lowest point, in row order (module
    docstring)."""
    pieces = vertical_pieces(distance, altitude, spec, words)
    groups = _runs(pieces, spec)
    bounds = [*go_around_points[1:], len(altitude)]
    climbs = [climb_after(groups, point, before, spec) for point, before in zip(go_around_points, bounds)]
    # the last descent of each approach that ends at a go-around row: the run straight before its climb
    final_descents = {position - 1 for position in climbs
                      if position and groups[position - 1][0].kind == MOVE and groups[position - 1][0].descending(spec)}
    reading = VerticalReading(instructions=[], pieces=pieces, go_around_rows=[])
    in_force_angle: int | None = None
    in_force_altitude: int | None = None
    for position, group in enumerate(groups):
        first = group[0]
        if first.kind == LEVEL:
            if position == 0:
                reading.instructions += [
                    Instruction(ALTITUDE, first.target_index, 0, "initial", {"target_m": words.altitude_m(first.target_index)}),
                    Instruction(ANGLE, ANGLE_LEVEL, 0, "initial"),
                ]
                in_force_angle, in_force_altitude = ANGLE_LEVEL, first.target_index
            elif groups[position - 1][0].kind == LEVEL:
                row, angle_class, angle = _step(groups[position - 1][0], first, distance, altitude, spec, words)
                reading.instructions.append(Instruction(ALTITUDE, first.target_index, row, "step",
                                                        {"target_m": words.altitude_m(first.target_index)}))
                in_force_altitude = first.target_index
                if angle_class != in_force_angle:
                    reading.instructions.append(Instruction(ANGLE, angle_class, row, "angle", {"angle_deg": angle}))
                    in_force_angle = angle_class
            continue
        descending = first.descending(spec)
        following = groups[position + 1] if position + 1 < len(groups) else None
        start_altitude = float(altitude[first.start])
        if following is None:
            if not descending:
                raise Refused("climbing at the end", f"{altitude[-1] - altitude[first.start]:+.0f} m over the last run")
            target_index = words.altitude_no_level_off
        elif position in final_descents:
            target_index = words.altitude_no_level_off
        elif following[0].kind == LEVEL:
            target_index = following[0].target_index
        else:                                   # the direction reverses without a level
            turning = float(altitude[group[-1].stop - 1])
            try:
                target_index = words.altitude_index(turning)
            except ValueError as error:
                raise Refused("altitude target out of range", str(error)) from None
        if target_index == in_force_altitude:
            continue                            # a move between two heights of one level: it cannot be said (§3.4)
        target_m = words.altitude_m(target_index)
        band = words.altitude_tolerance_m(target_index)
        if target_m is not None and (target_m > start_altitude + band if descending else target_m < start_altitude - band):
            raise Refused("altitude target on the wrong side",
                          f"{'descent' if descending else 'climb'} from {start_altitude:.0f} m to {target_m:.0f} m")
        kind = "initial" if first.start == 0 else "target"
        reading.instructions.append(Instruction(ALTITUDE, target_index, first.start, kind, {"target_m": target_m}))
        in_force_altitude = target_index
        for piece in group:
            try:
                angle_class = words.angle_index(piece.angle_deg)
            except ValueError as error:
                raise Refused("path angle out of range", str(error)) from None
            if angle_class != in_force_angle:
                reading.instructions.append(Instruction(
                    ANGLE, angle_class, piece.start, "initial" if piece.start == 0 else "angle",
                    {"angle_deg": piece.angle_deg}))
                in_force_angle = angle_class
    for position in climbs:
        row = _climb_start(groups, position, spec)[0]
        said = [item for item in reading.instructions if item.row == row]
        level = any(item.column == ALTITUDE for item in said)
        climb = any(item.column == ANGLE and item.value == words.angle_climb for item in said)
        if not (level and climb):
            raise Refused("go-around without a climb word", f"the climb at row {row} says no level and climb there")
        reading.go_around_rows.append(row)
    return reading


def _step(previous: VerticalPiece, level: VerticalPiece, distance: np.ndarray, altitude: np.ndarray,
          spec: VocabularySpec, words: Words) -> tuple[int, int, float]:
    """A level straight after another of another word: the issue row (the first level's last
    row), the angle class and the angle, read from there to the first row within the fit
    tolerance of the height the second level is held at. The class follows the step's direction."""
    row = previous.stop - 1
    met = np.nonzero(np.abs(altitude[level.start: level.stop] - level.median_m) <= spec.altitude_fit_tolerance_m)[0]
    end = level.start + int(met[0]) if len(met) else level.stop - 1
    angle = math.degrees(math.atan2(float(altitude[row] - altitude[end]), float(distance[end] - distance[row])))
    climbing = level.target_index > previous.target_index
    if climbing != (angle < 0.0):
        raise Refused("altitude target on the wrong side",
                      f"step to {words.altitude_m(level.target_index):.0f} m read at {angle:+.2f}° from row {row}")
    if climbing:
        if -angle > spec.climb_angle_max_deg:
            raise Refused("path angle out of range", f"step climb at {-angle:.1f}°")
        return row, words.angle_climb, angle
    try:
        return row, words.angle_index(angle), angle
    except ValueError as error:
        raise Refused("path angle out of range", str(error)) from None


def tube_bounds(instructions: list[Instruction], distance: np.ndarray, altitude: np.ndarray,
                spec: VocabularySpec, words: Words) -> list[tuple[Instruction, int, np.ndarray, np.ndarray]]:
    """Each altitude word's tube, row by row: ``(word, end, lower, upper)`` over rows
    ``word.row..end-1``, re-anchored at every angle word inside the span (the angle in force at the
    word's own row anchors its start), its margin the word's own (`Words.altitude_tolerance_m`). The
    one reading of §3.4's tube: the labeller's checks and the judge read it from here."""
    altitude_words = sorted((i for i in instructions if i.column == ALTITUDE), key=lambda item: item.row)
    angle_words = sorted((i for i in instructions if i.column == ANGLE), key=lambda item: item.row)
    ends = [item.row for item in altitude_words[1:]] + [len(altitude)]
    tubes = []
    for word, end in zip(altitude_words, ends):
        target, tolerance = words.altitude_m(word.value), words.altitude_tolerance_m(word.value)
        anchors = [a for a in angle_words if word.row <= a.row < end]
        before = [a for a in angle_words if a.row < word.row]
        if not anchors or anchors[0].row != word.row:
            anchors.insert(0, Instruction(ANGLE, before[-1].value, word.row, "in force"))
        lows, highs = [], []
        for anchor, stop in zip(anchors, [a.row for a in anchors[1:]] + [end]):
            rows = slice(anchor.row, stop)
            if anchor.value == ANGLE_LEVEL:
                lows.append(np.full(stop - anchor.row, target - tolerance))
                highs.append(np.full(stop - anchor.row, target + tolerance))
            else:
                low, high = envelope.vertical_tube(distance[rows] - distance[anchor.row], float(altitude[anchor.row]),
                                                   target, words.angle_bounds(anchor.value), tolerance)
                lows.append(low)
                highs.append(high)
        tubes.append((word, end, np.concatenate(lows), np.concatenate(highs)))
    return tubes


def tube_checks(instructions: list[Instruction], distance: np.ndarray, altitude: np.ndarray,
                spec: VocabularySpec, words: Words) -> list[dict[str, Any]]:
    """Each altitude word's span, checked against its tube (`tube_bounds`). Run on the assembled
    sentence, so a word the assembly dropped is not judged."""
    results = []
    for word, end, low, high in tube_bounds(instructions, distance, altitude, spec, words):
        span = altitude[word.row: end]
        inside = int(np.count_nonzero((span >= low) & (span <= high)))
        results.append({"row": word.row, "rows": end - word.row, "inside": inside, "contained": inside == end - word.row,
                        "target_m": words.altitude_m(word.value), "tube_width_end_m": float(high[-1] - low[-1])})
    return results
