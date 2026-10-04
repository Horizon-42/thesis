"""Heading words and the capture of the final (vocabulary §3.3, §4.3).

Every row of the sentence is labelled with the grid heading — RELATIVE to the course of the runway in force R at that
row (D8) — nearest the smoothed track `heading_lead_s` later (the last row's track once that lies past the end), and
consecutive rows labelled alike are ONE word: a new word is said wherever the track a lead later leaves the word in
force by more than half a step (`per_step_words`). A word therefore says where the track will be a lead later, a
continuous turn is a run of words at the pace it was flown, and nothing is inserted: no hold, no split, no intercept
word. The words run from row 0 to the sentence's last row — they describe the turn onto the final and the final itself
(D2); there is no clearance. A word in force keeps the track it said when R changes (the executor keeps it, §5.4): only
a new word reads the new course — and it is said also when its class is the class in force, its track being another
(D48).

The CAPTURE ROW (`capture_row`, vocabulary §2) of an approach (D26) is the first row from which the track stays in the
capture corridor of its runway to the end of the approach. It uses later rows, so it is never an input; only the
labeller reads it: "unspecified" speed starts there (D4), and the readout's stratum counts the turns before the landing
approach's. An approach that ends at a go-around row outside the corridor has none; the landing approach must end in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayRelative
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, Words


@dataclass(frozen=True)
class Approach:
    """An approach's rows ``first`` … ``end - 1`` (D26: from row 0 or the runway word that ends a go-around, to the
    landing or the next go-around row) and the candidate it flies to."""

    first: int
    end: int
    runway_index: int


@dataclass
class LateralReading:
    instructions: list[Instruction]
    #: Each approach's capture row, None where an approach that ends at a go-around row is outside the corridor there;
    #: the last (the landing approach's) is always a row.
    capture_rows: list[int | None]
    #: How far the track turned before the landing approach's capture: the net turn of every run of rows turning one
    #: way faster than the onset rate (`turn_runs`), summed — the stratum's measure (`readout.flight_record`), blind to
    #: a word boundary the track wanders across.
    turning_deg: float


def capture_row(relative: RunwayRelative, first: int, end: int, spec: VocabularySpec) -> int | None:
    """First row of the run of rows ``first`` … ``end - 1`` that stays in the corridor, before the threshold, to
    ``end - 1``; None when row ``end - 1`` is outside it."""
    inside = envelope.corridor(relative.right_of_course_m[first:end], relative.track_minus_course_deg[first:end],
                               relative.before_threshold_m[first:end], spec.corridor_half_width_m,
                               spec.corridor_widening_deg, spec.corridor_course_tolerance_deg)
    if not inside[-1]:
        return None
    row = len(inside) - 1
    while row > 0 and inside[row - 1]:
        row -= 1
    return first + row


def _snap(value: float, step: float) -> float:
    return round(value / step) * step


def per_step_words(track: np.ndarray, course: np.ndarray, step_deg: float, lead_rows: int) -> list[tuple[int, float]]:
    """``(row, target)`` of every word over the rows of ``track`` (unwrapped degrees): each row's word is the grid value
    relative to ``course`` (each row's, degrees) nearest the track ``lead_rows`` later (the last row's once that lies
    past the end), a new word wherever that track leaves the target in force by more than half a step. ``target`` is
    the word's track relative to its row's course, on the branch of the unwrapped ``track``."""
    rows = len(track)
    led = track[np.minimum(np.arange(rows) + lead_rows, rows - 1)]
    relative = _snap(float(led[0] - course[0]), step_deg)
    said, current = [(0, relative)], float(course[0]) + relative
    for row in range(1, rows):
        if abs(float(led[row]) - current) > step_deg / 2:
            relative = _snap(float(led[row] - course[row]), step_deg)
            said.append((row, relative))
            current = float(course[row]) + relative
    return said


def turn_runs(rate: np.ndarray, onset_deg_s: float) -> list[tuple[int, int]]:
    """Runs of rows turning one way faster than the turn onset rate: ``(start, stop)`` over ``rate`` (``rate[i]`` turns
    the track from row ``i`` to ``i + 1``), so the turn runs from track row ``start`` to ``stop``."""
    turning = np.abs(rate) > onset_deg_s
    runs, row = [], 0
    while row < len(rate):
        if not turning[row]:
            row += 1
            continue
        stop, sign = row + 1, np.sign(rate[row])
        while stop < len(rate) and turning[stop] and np.sign(rate[stop]) == sign:
            stop += 1
        runs.append((row, stop))
        row = stop
    return runs


def read_lateral(track: np.ndarray, relatives: list[RunwayRelative], courses_deg: list[float],
                 approaches: list[Approach], runway_rows: np.ndarray, spec: VocabularySpec,
                 words: Words) -> LateralReading:
    """The heading words of the sentence, each approach's capture row and the turning before the landing approach's
    (the last of ``approaches``); ``relatives`` / ``courses_deg`` every candidate's frame and course, ``runway_rows`` the
    runway in force at each row. Each word's ``info`` carries its relative target and the compass track it says."""
    courses = np.asarray(courses_deg, dtype=np.float64)[runway_rows]
    said = per_step_words(track, courses, spec.heading_step_deg, spec.rows_exact(spec.heading_lead_s))
    instructions = [Instruction(HEADING, words.heading_index(target), row, "initial" if row == 0 else "per-step",
                                {"relative_deg": words.heading_relative_deg(words.heading_index(target)),
                                 "target_deg": float((courses[row] + target) % 360.0)})
                    for row, target in said]
    # under one course a new word is always another class; after R changes course it can be the class in force, said
    # again because its track differs (D48: `sentence.assemble` keeps a heading word whose track is new)
    captures = [capture_row(relatives[a.runway_index], a.first, a.end, spec) for a in approaches]
    landing = captures[-1]
    if landing is None:
        relative = relatives[approaches[-1].runway_index]
        raise Refused("not on the final at the end",
                      f"last row {relative.right_of_course_m[-1]:+.0f} m off the centreline, "
                      f"{relative.track_minus_course_deg[-1]:+.1f}° off the course")
    rate = np.diff(track[: landing + 1]) / spec.step_s
    turning = float(sum(abs(track[stop] - track[start]) for start, stop in turn_runs(rate, spec.turn_onset_rate_deg_s)))
    return LateralReading(instructions=instructions, capture_rows=captures, turning_deg=turning)
