"""Which segment a word is, and what the executor is told for it — the pure part (numpy and the vocabulary's words).

A SEGMENT is one word of one column in force: from the step it is said (``row``) to the step the next word of its
column is said (``end_row``), or to the sentence's end — the band the sentence bar draws. The executor flies it to
where the word's own envelope ends (``stop_row``): ``end_row``, and for a HEADING word a lead later, since a heading
word is judged from a lead after it is said to a lead after the next heading word is (vocabulary design §10.1) — the
next heading word is told at ``end_row`` as in the sentence and its first lead is flown. When ``stop_row`` is the
sentence's end, the segment is flown on to its outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals
from ts_transformer.instructions.words import COLUMNS, HEADING, UNCHANGED

from aeroviz_backend.autopilot_segment.errors import RequestRefused


@dataclass(frozen=True)
class Segment:
    column: int                     # index into `COLUMNS`
    row: int                        # the step its word is said at
    end_row: int                    # the step the next word of its column is said at, or the sentence's length
    stop_row: int                   # where its own envelope ends: ``end_row``, a heading word's a lead later (capped)
    to_landing: bool                # ``stop_row`` is the sentence's end: flown on to the outcome
    grid: np.ndarray                # [n, 6] the segment's sentence, renumbered from ``row``
    instructions: list[Instruction]  # its words, renumbered the same way


def segment_of(reading: Reading, column: int, row: int, lead_rows: int) -> Segment:
    """The segment of ``column``'s word said at ``row``: step 0 holds the six words in force at ``row`` (each the
    reading's own word, moved to row 0); then the reading's words of the steps before ``stop_row`` (a heading word's:
    ``end_row`` plus ``lead_rows``, the heading lead in steps); and, unless that is the sentence's end, one silent step
    at ``stop_row`` itself — the point the word clock must reach to end it. A word said at the sentence's last step has
    no step to fly and is refused."""
    grid = np.asarray(reading.words)
    rows = len(grid)
    if not 0 <= column < len(COLUMNS):
        raise RequestRefused(f"column {column} is not one of the six {COLUMNS}")
    if not 0 <= row < rows:
        raise RequestRefused(f"step {row} is not a step of this {rows}-step sentence")
    if grid[row, column] == UNCHANGED:
        raise RequestRefused(f"no {COLUMNS[column]} word is said at step {row}: a segment starts where its word is said")
    later = np.nonzero(grid[row + 1:, column] != UNCHANGED)[0]
    end_row = row + 1 + int(later[0]) if len(later) else rows
    stop_row = min(end_row + (lead_rows if column == HEADING else 0), rows)
    to_landing = stop_row == rows
    if to_landing and rows - row < 2:
        raise RequestRefused(f"the {COLUMNS[column]} word said at step {row}, the sentence's last, has no step after it to fly")

    opening = []
    for index in range(len(COLUMNS)):
        said = [word for word in reading.instructions if word.column == index and word.row <= row]
        last = max(word.row for word in said)
        (word,) = [item for item in said if item.row == last]
        if word.value != grid[last, index]:
            raise ValueError(f"{reading.dataset_id}: the {COLUMNS[index]} word at step {last} is {word.value} in the "
                             f"reading's words but {grid[last, index]} in its grid")
        opening.append(replace(word, row=0))
    after = [replace(word, row=word.row - row) for word in reading.instructions if row < word.row < stop_row]

    length = stop_row - row + (0 if to_landing else 1)
    segment = np.full((length, len(COLUMNS)), UNCHANGED, dtype=grid.dtype)
    segment[0] = [word.value for word in opening]
    segment[1: stop_row - row] = grid[row + 1: stop_row]
    return Segment(column=column, row=row, end_row=end_row, stop_row=stop_row, to_landing=to_landing, grid=segment,
                   instructions=opening + after)


def told_words(segment: Segment) -> list[dict[str, int]]:
    """The words the executor was told, at the flight's steps, by step then column: the six in force at the segment's
    first step, then the sentence's words to its stop — as the sentence bar shows them (`trainingAutopilot.segmentWords`)."""
    return [{"row": segment.row + word.row, "column": word.column, "value": int(word.value)}
            for word in sorted(segment.instructions, key=lambda word: (word.row, word.column))]


def segment_reading(reading: Reading, segment: Segment) -> Reading:
    """The reading the executor flies and its judge reads: the segment's words; the flight's clearance, capture and
    "unspecified" rows renumbered the same way (negative: before the segment) — a `Reading` carries them, though the
    judge reads the words; the labeller's checks are the observed flight's, not the segment's, and are left out."""
    return Reading(dataset_id=reading.dataset_id, airport=reading.airport, runway_index=reading.runway_index,
                   words=segment.grid, instructions=segment.instructions,
                   capture_row=reading.capture_row - segment.row, join_row=reading.join_row - segment.row,
                   unspecified_row=reading.unspecified_row - segment.row,
                   cut_at_crossing=reading.cut_at_crossing and segment.to_landing, checks={})


def segment_signals(signals: FlightSignals, segment: Segment) -> FlightSignals:
    """The observed rows the segment's word clock reads: from its first step, one per step of its sentence."""
    stop = segment.row + len(segment.grid)
    return replace(signals, **{name: getattr(signals, name)[segment.row: stop] for name in ROW_FIELDS})
