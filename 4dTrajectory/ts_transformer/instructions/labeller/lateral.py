"""Heading words and the capture of the final (design §3.3, §4.3).

Every row of the sentence is labelled with the grid heading — RELATIVE to the course of the landed runway, the runway in
force of a labelled sentence (D8) — nearest the smoothed track `heading_lead_s` later (the last row's track once that
lies past the end), and consecutive rows labelled alike are ONE word: a new word is said wherever the label changes
cell (`per_step_words`). A word therefore says where the track will be a lead later, a continuous turn is a run of words
at the pace it was flown, and nothing is inserted: no hold, no split, no intercept word. The words run from row 0 to
the sentence's last row — they describe the turn onto the final and the final itself (D2); there is no clearance.

The CAPTURE ROW (`capture_row`, design §2) is the first row from which the track stays in the capture corridor to the
end. It uses later rows, so it is never an input; only the labeller reads it: "unspecified" speed starts there (D4), and
the readout's stratum counts the turns before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import RunwayRelative
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, Words


@dataclass
class LateralReading:
    instructions: list[Instruction]
    capture_row: int
    #: How far the track turned before the capture: the net turn of every run of rows turning one way faster than the
    #: onset rate (`turn_runs`), summed — the stratum's measure (`readout.flight_record`), blind to a word boundary
    #: the track wanders across.
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


def _snap(value: float, step: float) -> float:
    return round(value / step) * step


def per_step_words(track: np.ndarray, step_deg: float, lead_rows: int) -> list[tuple[int, float]]:
    """``(row, target)`` of every word over the rows of ``track`` (unwrapped degrees, in whatever frame the grid is
    in: relative to the course for heading words): each row labelled with the grid value nearest the track
    ``lead_rows`` later (the last row's once that lies past the end), a word wherever the label changes cell. The
    targets lie on the branch of the unwrapped ``track``."""
    rows = len(track)
    led = track[np.minimum(np.arange(rows) + lead_rows, rows - 1)]
    current = _snap(float(led[0]), step_deg)
    said = [(0, current)]
    for row in range(1, rows):
        if abs(float(led[row]) - current) > step_deg / 2:
            current = _snap(float(led[row]), step_deg)
            said.append((row, current))
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


def read_lateral(track: np.ndarray, relative: RunwayRelative, course_deg: float, spec: VocabularySpec,
                 words: Words) -> LateralReading:
    """The heading words of the sentence (relative to ``course_deg``, the landed runway's), its capture row and the
    turning before it. Each word's ``info`` carries its relative target and the compass track it says."""
    capture = capture_row(relative, spec)
    said = per_step_words(track - course_deg, spec.heading_step_deg, spec.rows_exact(spec.heading_lead_s))
    instructions = [Instruction(HEADING, words.heading_index(target), row, "initial" if row == 0 else "per-step",
                                {"relative_deg": words.heading_relative_deg(words.heading_index(target)),
                                 "target_deg": (course_deg + target) % 360.0})
                    for row, target in said]
    rate = np.diff(track[: capture + 1]) / spec.step_s
    turning = float(sum(abs(track[stop] - track[start]) for start, stop in turn_runs(rate, spec.turn_onset_rate_deg_s)))
    return LateralReading(instructions=instructions, capture_row=capture, turning_deg=turning)
