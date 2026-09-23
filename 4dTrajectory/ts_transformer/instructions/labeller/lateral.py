"""Heading words, the splitting of large turns, the capture of the final and the approach
clearance (vocabulary design §2.2, §2.3, §3.2).

Holds are read by TILING on the heading grid: from each row, the longest run of rows whose
smoothed track stays within the heading tolerance of one grid target and turns no faster than
the hold's largest rate; a run shorter than the minimum hold belongs to a turn. A hold's word
is issued where the turn into it began. Only rows before the capture of the final are read:
after it the lateral path is the line's. The last heading word must converge on the final
(`envelope.heading_converges`), or an intercept word — the course ± the intercept angle,
toward the line — is inserted; the clearance goes with the last heading word.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayRelative
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, Words, wrap180,
)


@dataclass(frozen=True)
class Hold:
    start: int
    stop: int                 # one past the last row
    target_unwrapped: float   # the grid target, on the unwrapped track's branch

    @property
    def target_deg(self) -> float:
        return self.target_unwrapped % 360.0


@dataclass
class LateralReading:
    instructions: list[Instruction]
    capture_row: int
    join_row: int
    holds: list[Hold]
    turns: list[dict[str, Any]] = field(default_factory=list)
    intercept_inserted: bool = False
    #: The capture turn onto the final, from the end of the last hold (or of an inserted intercept)
    #: to the capture; ``None`` for a flight already on the final at row 0.
    capture_turn: dict[str, Any] | None = None


def capture_row(relative: RunwayRelative, spec: VocabularySpec) -> int:
    """First row of the final run of rows that stay in the corridor, before the threshold,
    to the end. A flight that does not end in the corridor is refused."""
    inside = envelope.corridor(relative.right_of_course_m, relative.track_minus_course_deg, relative.before_threshold_m,
                               spec.corridor_half_width_m, spec.corridor_widening_deg, spec.corridor_course_tolerance_deg)
    if not inside[-1]:
        raise Refused("not on the final at the end",
                      f"last row {relative.right_of_course_m[-1]:+.0f} m off the centreline, "
                      f"{relative.track_minus_course_deg[-1]:+.1f}° off the course")
    row = len(inside) - 1
    while row > 0 and inside[row - 1]:
        row -= 1
    return row


def _steady(track: np.ndarray, spec: VocabularySpec) -> bool:
    """A hold is flown straight: its track's least-squares rate stays at or below the hold's
    largest rate (a slow continuous turn inside the band is a turn, not a string of holds)."""
    rows = np.arange(len(track), dtype=np.float64)
    rate = np.polyfit(rows, track, 1)[0] / spec.step_s
    return abs(rate) <= spec.heading_hold_max_rate_deg_s


def find_holds(track_unwrapped: np.ndarray, stop_row: int, spec: VocabularySpec) -> list[Hold]:
    step, tolerance = spec.heading_step_deg, spec.heading_tolerance_deg
    minimum = spec.rows(spec.heading_min_hold_s)
    holds: list[Hold] = []
    row = 0
    while row < stop_row:
        base = track_unwrapped[row]
        best_length, best_target = 0, 0.0
        for k in range(math.ceil((base - tolerance) / step), math.floor((base + tolerance) / step) + 1):
            target = k * step
            inside = envelope.heading_band(track_unwrapped[row:stop_row], target, tolerance)
            length = len(inside) if inside.all() else int(np.argmin(inside))
            if length >= minimum and not _steady(track_unwrapped[row: row + length], spec):
                continue
            if length > best_length or (length == best_length and abs(target - base) < abs(best_target - base)):
                best_length, best_target = length, target
        if best_length >= minimum:
            holds.append(Hold(row, row + best_length, best_target))
            row += best_length
        else:
            row += 1
    return holds


def _departure_row(track: np.ndarray, hold: Hold, direction: float, spec: VocabularySpec) -> int:
    """Where the turn out of ``hold`` began: walking back from the hold's last row while the
    track was already moving in the turn's direction faster than the onset rate."""
    threshold = spec.turn_onset_rate_deg_s * spec.step_s
    row = hold.stop - 1
    while row > hold.start and (track[row] - track[row - 1]) * direction > threshold:
        row -= 1
    return row


def _intercept_end(track: np.ndarray, departure: int, capture: int, target: float, tolerance: float) -> int:
    """Where the turn onto an inserted intercept heading gives way to the capture turn: the first
    row within the tolerance of the target, or, when the capture turn began before the target was
    reached, the row of the turn's furthest progress toward it."""
    span = track[departure: capture + 1]
    direction = 1.0 if target >= span[0] else -1.0
    progress = (span - span[0]) * direction
    reached = np.nonzero(progress >= (target - span[0]) * direction - tolerance)[0]
    return departure + int(reached[0] if len(reached) else np.argmax(progress))


def _turn_check(track: np.ndarray, ground_speed: np.ndarray, start: int, end: int, target: float,
                spec: VocabularySpec) -> dict[str, Any]:
    """The turn envelope over rows ``start..end``: monotone toward the target, bank inside the range."""
    rate = np.diff(track[start: end + 1]) / spec.step_s
    bank = envelope.bank_deg_from_turn_rate(rate, ground_speed[start + 1: end + 1]) if len(rate) else np.zeros(1)
    mean_bank, max_bank = float(bank.mean()), float(bank.max())
    turn = float(target - track[start])
    return {
        "progress_ok": envelope.turn_progress_ok(track[start: end + 1], target, spec.heading_tolerance_deg),
        "mean_bank_deg": mean_bank, "max_bank_deg": max_bank,
        "bank_min_applies": abs(turn) >= spec.turn_bank_min_from_deg,
        "bank_ok": envelope.turn_bank_ok(turn, mean_bank, max_bank, spec.turn_bank_min_deg, spec.turn_bank_max_deg,
                                         spec.turn_bank_min_from_deg),
    }


def _snap(value: float, step: float) -> float:
    return round(value / step) * step


def read_lateral(track: np.ndarray, ground_speed: np.ndarray, relative: RunwayRelative,
                 course_deg: float, spec: VocabularySpec, words: Words) -> LateralReading:
    capture = capture_row(relative, spec)
    reading = LateralReading(instructions=[], capture_row=capture, join_row=0, holds=[])
    if capture == 0:
        reading.instructions += [
            Instruction(HEADING, words.heading_index(track[0]), 0, "initial", {"target_deg": track[0] % 360.0}),
            Instruction(APPROACH, APPROACH_CLEARED, 0, "clear"),
        ]
        return reading

    holds = find_holds(track, capture, spec)
    reading.holds = holds
    step = spec.heading_step_deg

    def emit_turn(start_value: float, target: float, departure: int, arrival: int, kind: str) -> None:
        delta = target - start_value
        if abs(delta) < 1e-9:
            return
        # a turn cannot begin before the word in force was issued
        issued = [item.row for item in reading.instructions if item.column == HEADING]
        departure = max(departure, max(issued) + 1) if issued else departure
        if departure > arrival:
            raise Refused("turn shorter than one step", f"rows {departure}–{arrival}")
        direction = math.copysign(1.0, delta)
        parts = 1 if abs(delta) <= spec.heading_max_turn_deg else math.ceil(abs(delta) / spec.heading_split_part_deg)
        targets = [_snap(start_value + delta * m / parts, step) for m in range(1, parts + 1)]
        targets[-1] = target
        rows = [departure]
        for previous in targets[:-1]:
            later = np.nonzero((track[rows[-1] + 1: arrival + 1] - previous) * direction
                               >= -spec.heading_continue_lead_deg)[0]
            if len(later) == 0:
                raise Refused("split turn not continued", f"the track never came within "
                              f"{spec.heading_continue_lead_deg:g}° of {previous % 360:.0f}°")
            rows.append(rows[-1] + 1 + int(later[0]))
        for part, (row, value) in enumerate(zip(rows, targets), start=1):
            # measured from the track at issue, which may sit anywhere in the previous target's band
            shorter = float(wrap180(value - track[row]))
            if (abs(shorter) > spec.heading_max_turn_deg + spec.heading_tolerance_deg
                    or shorter * direction < -spec.heading_tolerance_deg):
                raise Refused("heading word beyond the largest turn",
                              f"{value % 360:.0f}° issued at {track[row] % 360:.1f}° ({shorter:+.1f}°)")
            reading.instructions.append(Instruction(
                HEADING, words.heading_index(value), row, kind if parts == 1 else f"{kind}-split",
                {"target_deg": value % 360.0, "turn_deg": delta, "part": part, "parts": parts}))
        # an inserted intercept is flown only until the capture turn takes over
        end = _intercept_end(track, departure, arrival, target, spec.heading_tolerance_deg) if kind == "intercept" else arrival
        reading.turns.append({"departure_row": departure, "arrival_row": end, "turn_deg": delta, "parts": parts,
                              "kind": kind, **_turn_check(track, ground_speed, departure, end, target, spec)})

    # the word in force at row 0: the first hold's, reached through a turn from the first row
    # when the flight enters the slice mid-turn
    if holds and holds[0].start == 0:
        current, previous = holds[0].target_unwrapped, holds[0]
        reading.instructions.append(Instruction(HEADING, words.heading_index(current), 0, "initial",
                                                {"target_deg": current % 360.0}))
        remaining = holds[1:]
    else:
        current, previous = float(track[0]), None
        remaining = holds
    for hold in remaining:
        direction = math.copysign(1.0, hold.target_unwrapped - current)
        departure = 0 if previous is None else _departure_row(track, previous, direction, spec)
        emit_turn(current, hold.target_unwrapped, departure, hold.start, "turn")
        current, previous = hold.target_unwrapped, hold

    # the capture: the heading in force must converge on the line, or an intercept word is added
    anchor = 0 if previous is None else previous.stop - 1
    offset = float(relative.right_of_course_m[anchor])
    converging = envelope.heading_converges(current, course_deg, offset, float(relative.before_threshold_m[anchor]),
                                            spec.heading_tolerance_deg, spec.corridor_half_width_m,
                                            spec.corridor_widening_deg)
    if not converging:
        angle = float(wrap180(current - course_deg))
        side = math.copysign(1.0, offset) if offset != 0.0 else math.copysign(1.0, angle)
        intercept = course_deg - side * spec.intercept_angle_deg
        target = _snap(float(track[capture]) + float(wrap180(intercept - track[capture])), step)
        direction = math.copysign(1.0, target - current)
        departure = 0 if previous is None else _departure_row(track, previous, direction, spec)
        emit_turn(current, target, departure, capture, "intercept")
        reading.intercept_inserted = True
        # the intercept must itself reach the final, flown from where the aircraft is on it
        on = reading.turns[-1]["arrival_row"]
        if not envelope.heading_converges(target, course_deg, float(relative.right_of_course_m[on]),
                                          float(relative.before_threshold_m[on]), spec.heading_tolerance_deg,
                                          spec.corridor_half_width_m, spec.corridor_widening_deg):
            raise Refused("no intercept reaches the final",
                          f"{target % 360:.0f}° at row {on}, {relative.right_of_course_m[on]:+.0f} m off the line, "
                          f"{relative.before_threshold_m[on]:.0f} m before the threshold")
    # the capture turn begins where the last hold ends, or where the inserted intercept gives way
    start = reading.turns[-1]["arrival_row"] if reading.intercept_inserted else anchor
    course = float(track[start]) + float(wrap180(course_deg - track[start]))
    reading.capture_turn = {"start_row": start, **_turn_check(track, ground_speed, start, capture, course, spec)}
    if not any(item.column == HEADING and item.row == 0 for item in reading.instructions):
        reading.instructions.insert(0, Instruction(HEADING, words.heading_index(track[0]), 0, "initial",
                                                   {"target_deg": track[0] % 360.0}))
    reading.join_row = max(item.row for item in reading.instructions)
    reading.instructions.append(Instruction(APPROACH, APPROACH_CLEARED, reading.join_row, "clear"))
    if reading.join_row > 0:
        reading.instructions.append(Instruction(APPROACH, APPROACH_NOT_CLEARED, 0, "initial"))
    return reading
