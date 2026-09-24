"""Heading words, the splitting of large turns, the capture of the final and the approach
clearance (vocabulary design §2.2, §2.3, §3.2).

The heading words are read one of two ways (`VocabularySpec.heading_reading`, §10.1 compares them): ``holds``
below, or ``per-step`` (`_read_per_step`): every row labelled with the grid heading the track reaches
`heading_lead_s` later, merged into one word while it stays within `heading_band_deg` of the word in force.

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
    """The turn envelope over rows ``start..end``: monotone toward the target; the turn rate and
    the bank it takes inside the range (§2.3)."""
    turn = float(target - track[start])
    rate = np.diff(track[start: end + 1]) / spec.step_s if end > start else np.zeros(1)
    bank = envelope.bank_deg_from_turn_rate(rate, ground_speed[start + 1: end + 1]) if end > start else np.zeros(1)
    mean_rate = float(np.mean(rate) * math.copysign(1.0, turn)) if turn != 0.0 else 0.0
    max_rate, max_bank = float(np.max(np.abs(rate))), float(np.max(bank))
    return {
        "progress_ok": envelope.turn_progress_ok(track[start: end + 1], target, spec.heading_tolerance_deg),
        "mean_rate_deg_s": mean_rate, "max_rate_deg_s": max_rate, "max_bank_deg": max_bank,
        "rate_min_applies": abs(turn) >= spec.turn_rate_min_from_deg,
        "rate_ok": envelope.turn_rate_ok(turn, mean_rate, max_rate, max_bank, spec.turn_rate_min_deg_s,
                                         spec.turn_rate_max_deg_s, spec.turn_bank_max_deg, spec.turn_rate_min_from_deg),
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

    if spec.heading_reading == "per-step":
        return _read_per_step(track, ground_speed, relative, course_deg, reading, spec, words)
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


def _capture_turn_onset(track: np.ndarray, capture: int, course_deg: float, spec: VocabularySpec) -> int:
    """Where the turn onto the course that ends at the capture began: walking back from the capture while the track
    was already moving toward the course faster than the onset rate (the rule `_departure_row` reads a turn by)."""
    course = float(track[capture]) + float(wrap180(course_deg - track[capture]))
    threshold = spec.turn_onset_rate_deg_s * spec.step_s
    row = capture
    while row > 0 and (track[row] - track[row - 1]) * math.copysign(1.0, course - float(track[row - 1])) > threshold:
        row -= 1
    return row


def _read_per_step(track: np.ndarray, ground_speed: np.ndarray, relative: RunwayRelative, course_deg: float,
                   reading: LateralReading, spec: VocabularySpec, words: Words) -> LateralReading:
    """§10.1's per-step reading of a flight not on the final at row 0: each row before the capture labelled with the
    grid heading nearest the track `heading_lead_s` later (the capture row's track once that lies past it), and a
    new word only where that leaves the word in force by more than `heading_band_deg`. No hold, no split, no
    inserted intercept. Where the clearance goes (`VocabularySpec.heading_clearance`): with the last word (the words
    run to the capture; that word must reach the final, `envelope.heading_converges`, or the flight is refused); at
    the capture turn's onset (`_capture_turn_onset`); or there but not before the word in force reaches the final.
    Under the two capture-turn rules the words stop at the clearance and the capture turn is the executor's."""
    capture, step = reading.capture_row, spec.heading_step_deg
    lead = int(round(spec.heading_lead_s / spec.step_s))
    led = track[np.minimum(np.arange(capture) + lead, capture)]
    current = _snap(float(led[0]), step)
    said = [Instruction(HEADING, words.heading_index(current), 0, "initial", {"target_deg": current % 360.0})]
    in_force = np.empty(capture + 1)                 # the word in force at each row (said at a row before it)
    in_force[:2] = current
    for row in range(1, capture):
        if abs(float(led[row]) - current) > spec.heading_band_deg:
            current = _snap(float(led[row]), step)
            said.append(Instruction(HEADING, words.heading_index(current), row, "per-step", {"target_deg": current % 360.0}))
        in_force[row + 1] = current

    def converges(row: int, heading_deg: float) -> bool:
        return envelope.heading_converges(heading_deg, course_deg, float(relative.right_of_course_m[row]),
                                          float(relative.before_threshold_m[row]), spec.heading_tolerance_deg,
                                          spec.corridor_half_width_m, spec.corridor_widening_deg)

    if spec.heading_clearance == "last-word":
        clear = said[-1].row
        if not converges(clear, current):
            raise Refused("the last heading word does not reach the final",
                          f"{current % 360:.0f}° at row {clear}, {relative.right_of_course_m[clear]:+.0f} m off "
                          f"the line, {relative.before_threshold_m[clear]:.0f} m before the threshold")
    else:
        clear = _capture_turn_onset(track, capture, course_deg, spec)
        if spec.heading_clearance == "capture-turn-converging":
            clear = next(row for row in range(clear, capture + 1)
                         if row == capture or converges(row, float(in_force[row])))
        said = [item for item in said if item.row < max(clear, 1)]
        course = float(track[clear]) + float(wrap180(course_deg - track[clear]))
        reading.capture_turn = {"start_row": clear, **_turn_check(track, ground_speed, clear, capture, course, spec)}
    reading.instructions += said
    reading.join_row = clear
    reading.instructions.append(Instruction(APPROACH, APPROACH_CLEARED, clear, "clear"))
    if clear > 0:
        reading.instructions.append(Instruction(APPROACH, APPROACH_NOT_CLEARED, 0, "initial"))
    return reading


def turn_of_word(word: Instruction, turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The turn check a heading word belongs to: the one it DEPARTS (a single turn, or a split
    turn's first part), or the split turn it continues (a later part, issued mid-turn); ``None``
    for a word the flight was already flying at entry."""
    if word.kind == "initial":
        return None
    kind = word.kind.removesuffix("-split")
    if word.info["part"] == 1:
        found = [t for t in turns if t["kind"] == kind and t["departure_row"] == word.row]
    else:
        found = [t for t in turns if t["kind"] == kind and t["departure_row"] < word.row <= t["arrival_row"]]
    if len(found) != 1:
        raise ValueError(f"heading word {word.kind} at row {word.row}: {len(found)} turn checks cover it")
    return found[0]


@dataclass(frozen=True)
class HeadingSpan:
    """One heading word, from its issue row to the next word's (§2.3)."""

    word: Instruction
    #: The labeller's check of the turn this word belongs to — shared by the parts of a split
    #: turn; ``None`` for a word the flight was already flying at entry.
    turn: dict[str, Any] | None
    #: Where its hold begins: its turn's end as the labeller read it (the issue row for a word
    #: flown from entry); ``None`` for a split part the next part supersedes mid-turn.
    hold_start: int | None
    #: Where its hold ends: the next heading word's row (where the next word takes over), or — for
    #: the last one — where the capture turn begins (the capture itself when there is none).
    hold_end: int

    @property
    def held(self) -> bool:
        return self.hold_start is not None and self.hold_end > self.hold_start


def heading_spans(instructions: list[Instruction], turns: list[dict[str, Any]],
                  capture_turn: dict[str, Any] | None, capture: int) -> list[HeadingSpan]:
    """Every heading word of a sentence with the rows its hold covers — the one definition the
    labeller's hold check and the display share."""
    heading = sorted((item for item in instructions if item.column == HEADING), key=lambda item: item.row)
    last_end = capture if capture_turn is None else int(capture_turn["start_row"])
    spans = []
    for word, end in zip(heading, [w.row for w in heading[1:]] + [last_end]):
        turn = turn_of_word(word, turns)
        if turn is None:
            start: int | None = word.row
        else:
            start = int(turn["arrival_row"]) if word.info["part"] == word.info["parts"] else None
        spans.append(HeadingSpan(word=word, turn=turn, hold_start=start, hold_end=int(end)))
    return spans
