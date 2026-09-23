"""Altitude words and descent-angle words (vocabulary design §2.4, §2.5, §3.3).

The smoothed altitude is fitted against horizontal distance by straight pieces. A piece that
stays inside one altitude target's band for at least the minimum level time is a LEVEL (a
target reached); every other piece is a MOVE with a path angle. Consecutive moves in one
direction form a run; a run's altitude word is the next level's target (the turning altitude
when the direction reverses without a level; "descend to land" when the last run reaches the
end), issued where the run begins. An angle word is issued wherever a move's class differs
from the class in force. Two levels that meet with no move between them are a STEP: the
altitude left the first band inside the fitted pieces; its word is issued at the first level's
last row, its angle read over the rows from there to where the second level's target is met.
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


def vertical_pieces(distance: np.ndarray, altitude: np.ndarray, spec: VocabularySpec, words: Words) -> list[VerticalPiece]:
    minimum = spec.rows(spec.level_min_s)
    pieces: list[VerticalPiece] = []
    for piece in fit_pieces(distance, altitude, spec.altitude_fit_tolerance_m):
        rows = altitude[piece.start: piece.stop]
        target = float(np.median(rows))
        index = int(round(target / spec.altitude_step_m))
        level = (piece.rows >= minimum and 0 <= index < words.n_altitude_levels
                 and bool(envelope.level_band(rows, index * spec.altitude_step_m, spec.altitude_tolerance_m).all()))
        if level:
            previous = pieces[-1] if pieces else None
            if (previous is not None and previous.kind == LEVEL
                    and envelope.level_band(rows, words.altitude_m(previous.target_index), spec.altitude_tolerance_m).all()):
                previous.stop = piece.stop          # still inside the band of the level in force
            else:
                pieces.append(VerticalPiece(piece.start, piece.stop, LEVEL, target_index=index))
        else:
            pieces.append(VerticalPiece(piece.start, piece.stop, MOVE, angle_deg=math.degrees(math.atan(-piece.slope))))
    return pieces


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


def read_vertical(distance: np.ndarray, altitude: np.ndarray, spec: VocabularySpec, words: Words) -> VerticalReading:
    pieces = vertical_pieces(distance, altitude, spec, words)
    groups = _runs(pieces, spec)
    reading = VerticalReading(instructions=[], pieces=pieces)
    in_force_angle: int | None = None
    for position, group in enumerate(groups):
        first = group[0]
        if first.kind == LEVEL:
            if position == 0:
                reading.instructions += [
                    Instruction(ALTITUDE, first.target_index, 0, "initial", {"target_m": words.altitude_m(first.target_index)}),
                    Instruction(ANGLE, ANGLE_LEVEL, 0, "initial"),
                ]
                in_force_angle = ANGLE_LEVEL
            elif groups[position - 1][0].kind == LEVEL:
                row, angle_class, angle = _step(groups[position - 1][0], first, distance, altitude, spec, words)
                reading.instructions.append(Instruction(ALTITUDE, first.target_index, row, "step",
                                                        {"target_m": words.altitude_m(first.target_index)}))
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
            target_index = words.altitude_land
        elif following[0].kind == LEVEL:
            target_index = following[0].target_index
        else:                                   # the direction reverses without a level
            turning = float(altitude[group[-1].stop - 1])
            try:
                target_index = words.altitude_index(turning)
            except ValueError as error:
                raise Refused("altitude target out of range", str(error)) from None
        target_m = words.altitude_m(target_index)
        if target_m is not None and (target_m > start_altitude + spec.altitude_tolerance_m if descending
                                     else target_m < start_altitude - spec.altitude_tolerance_m):
            raise Refused("altitude target on the wrong side",
                          f"{'descent' if descending else 'climb'} from {start_altitude:.0f} m to {target_m:.0f} m")
        kind = "initial" if first.start == 0 else "target"
        reading.instructions.append(Instruction(ALTITUDE, target_index, first.start, kind, {"target_m": target_m}))
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
    return reading


def _step(previous: VerticalPiece, level: VerticalPiece, distance: np.ndarray, altitude: np.ndarray,
          spec: VocabularySpec, words: Words) -> tuple[int, int, float]:
    """A level straight after another, outside its band: the issue row (the first level's last
    row), the angle class and the angle, read from there to the first row within the fit
    tolerance of the new target. The class follows the step's direction."""
    row = previous.stop - 1
    target_m = words.altitude_m(level.target_index)
    met = np.nonzero(np.abs(altitude[level.start: level.stop] - target_m) <= spec.altitude_fit_tolerance_m)[0]
    end = level.start + int(met[0]) if len(met) else level.stop - 1
    angle = math.degrees(math.atan2(float(altitude[row] - altitude[end]), float(distance[end] - distance[row])))
    climbing = target_m > words.altitude_m(previous.target_index)
    if climbing != (angle < 0.0):
        raise Refused("altitude target on the wrong side",
                      f"step to {target_m:.0f} m read at {angle:+.2f}° from row {row}")
    if climbing:
        if -angle > spec.climb_angle_max_deg:
            raise Refused("path angle out of range", f"step climb at {-angle:.1f}°")
        return row, words.angle_climb, angle
    try:
        return row, words.angle_index(angle), angle
    except ValueError as error:
        raise Refused("path angle out of range", str(error)) from None


def tube_checks(instructions: list[Instruction], distance: np.ndarray, altitude: np.ndarray,
                spec: VocabularySpec, words: Words) -> list[dict[str, Any]]:
    """Each altitude word's span, checked against the tube (re-anchored at every angle word).
    Run on the assembled sentence, so a word the assembly dropped is not judged."""
    altitude_words = sorted((i for i in instructions if i.column == ALTITUDE), key=lambda item: item.row)
    angle_words = sorted((i for i in instructions if i.column == ANGLE), key=lambda item: item.row)
    ends = [item.row for item in altitude_words[1:]] + [len(altitude)]
    results = []
    for word, end in zip(altitude_words, ends):
        target = words.altitude_m(word.value)
        anchors = [a for a in angle_words if word.row <= a.row < end]
        before = [a for a in angle_words if a.row < word.row]
        if not anchors or anchors[0].row != word.row:
            anchors.insert(0, Instruction(ANGLE, before[-1].value, word.row, "in force"))
        inside_rows, lower_end, upper_end = 0, 0.0, 0.0
        for anchor, stop in zip(anchors, [a.row for a in anchors[1:]] + [end]):
            rows = slice(anchor.row, stop)
            if anchor.value == ANGLE_LEVEL:
                low = np.full(stop - anchor.row, target - spec.altitude_tolerance_m)
                high = np.full(stop - anchor.row, target + spec.altitude_tolerance_m)
            else:
                low, high = envelope.vertical_tube(distance[rows] - distance[anchor.row], float(altitude[anchor.row]),
                                                   target, words.angle_bounds(anchor.value), spec.altitude_tolerance_m)
            inside_rows += int(np.count_nonzero((altitude[rows] >= low) & (altitude[rows] <= high)))
            lower_end, upper_end = float(low[-1]), float(high[-1])
        rows_total = end - word.row
        results.append({"row": word.row, "rows": rows_total, "inside": inside_rows, "contained": inside_rows == rows_total,
                        "target_m": target, "tube_width_end_m": upper_end - lower_end})
    return results
