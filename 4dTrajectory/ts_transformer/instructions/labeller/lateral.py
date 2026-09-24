"""Heading words, the capture of the final and the approach clearance (vocabulary design §2.2, §10.1:
instruction-v3's per-step reading).

Every row before the clearance is labelled with the grid heading nearest the smoothed track `heading_lead_s` later
(the capture row's track once that lies past it), and consecutive rows labelled alike are ONE word: a new word is
said wherever the label changes cell (`per_step_words`). A word therefore says where the track will be a lead
later, a continuous turn is a run of words at the pace it was flown, and nothing is inserted: no hold, no split, no
intercept word.

The clearance (§10.1) is said where the capture turn begins — walking back from the capture over the rows already on
the course, then while the track was turning toward the course faster than the onset rate — but not
before the heading word in force reaches the final (`envelope.heading_converges`: within the heading tolerance it
meets the line ahead of the threshold, at no more than 90° plus the tolerance to the course). From there the capture
turn is the executor's; the heading words stop before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayRelative
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import (
    APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, Words, wrap180,
)


@dataclass
class LateralReading:
    instructions: list[Instruction]
    capture_row: int
    join_row: int
    #: The capture turn, from the clearance to the capture (`turn_check`); ``None`` for a flight already on the
    #: final at row 0.
    capture_turn: dict[str, Any] | None
    #: The heading changed before the capture: the sum of the turns from word to word, and on from the last word to
    #: the course (the stratum's measure, `readout.flight_record`).
    turning_deg: float


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


def turn_check(track: np.ndarray, ground_speed: np.ndarray, start: int, end: int, target: float,
               spec: VocabularySpec) -> dict[str, Any]:
    """The capture turn's envelope over rows ``start..end``: monotone toward the target; the turn rate and the bank it
    takes inside the range."""
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


def per_step_words(track: np.ndarray, stop: int, step_deg: float, lead_rows: int) -> list[tuple[int, float]]:
    """``(row, target)`` of every word before ``stop``: each row labelled with the grid heading nearest the track
    ``lead_rows`` later (the row ``stop``'s track once that lies past it), a word wherever the label changes cell. The
    targets lie on the branch of the unwrapped ``track``."""
    led = track[np.minimum(np.arange(stop) + lead_rows, stop)]
    current = _snap(float(led[0]), step_deg)
    said = [(0, current)]
    for row in range(1, stop):
        if abs(float(led[row]) - current) > step_deg / 2:
            current = _snap(float(led[row]), step_deg)
            said.append((row, current))
    return said


def capture_turn_onset(track: np.ndarray, capture: int, course_deg: float, spec: VocabularySpec) -> int:
    """Where the turn onto the course that ends at the capture began: walking back from the capture over the rows
    already on the course (within the corridor's course tolerance: the turn ended before the track entered the
    corridor), then while the track was moving toward the course faster than the onset rate."""
    course = float(track[capture]) + float(wrap180(course_deg - track[capture]))
    threshold = spec.turn_onset_rate_deg_s * spec.step_s
    row = capture
    while row > 0 and abs(float(track[row - 1]) - course) <= spec.corridor_course_tolerance_deg:
        row -= 1
    while row > 0 and (track[row] - track[row - 1]) * math.copysign(1.0, course - float(track[row - 1])) > threshold:
        row -= 1
    return row


def _turning_deg(targets: list[float], course_unwrapped: float) -> float:
    path = [*targets, course_unwrapped]
    return float(sum(abs(b - a) for a, b in zip(path, path[1:])))


def read_lateral(track: np.ndarray, ground_speed: np.ndarray, relative: RunwayRelative,
                 course_deg: float, spec: VocabularySpec, words: Words) -> LateralReading:
    capture = capture_row(relative, spec)
    if capture == 0:
        target = _snap(float(track[0]), spec.heading_step_deg)
        return LateralReading(
            instructions=[Instruction(HEADING, words.heading_index(target), 0, "initial", {"target_deg": target % 360.0}),
                          Instruction(APPROACH, APPROACH_CLEARED, 0, "clear")],
            capture_row=0, join_row=0, capture_turn=None, turning_deg=0.0)

    lead = spec.rows_exact(spec.heading_lead_s)
    said = per_step_words(track, capture, spec.heading_step_deg, lead)
    in_force = np.empty(capture + 1)                 # the word in force at each row (said at a row before it)
    in_force[0] = said[0][1]
    for (row, target), (following, _) in zip(said, [*said[1:], (capture, 0.0)]):
        in_force[row + 1: following + 1] = target

    def converges(row: int) -> bool:
        return envelope.heading_converges(float(in_force[row]), course_deg, float(relative.right_of_course_m[row]),
                                          float(relative.before_threshold_m[row]), spec.heading_tolerance_deg,
                                          spec.corridor_half_width_m, spec.corridor_widening_deg)

    onset = capture_turn_onset(track, capture, course_deg, spec)
    clear = next(row for row in range(onset, capture + 1) if row == capture or converges(row))
    kept = [(row, target) for row, target in said if row < max(clear, 1)]
    course = float(track[clear]) + float(wrap180(course_deg - track[clear]))
    instructions = [Instruction(HEADING, words.heading_index(target), row, "initial" if row == 0 else "per-step",
                                {"target_deg": target % 360.0}) for row, target in kept]
    instructions.append(Instruction(APPROACH, APPROACH_CLEARED, clear, "clear"))
    if clear > 0:
        instructions.append(Instruction(APPROACH, APPROACH_NOT_CLEARED, 0, "initial"))
    last = kept[-1][1]
    return LateralReading(
        instructions=instructions, capture_row=capture, join_row=clear,
        capture_turn={"start_row": clear, **turn_check(track, ground_speed, clear, capture, course, spec)},
        turning_deg=_turning_deg([t for _, t in kept], last + float(wrap180(course_deg - last))))
