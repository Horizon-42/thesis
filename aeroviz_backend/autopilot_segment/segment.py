"""Which segment a word is, and what the executor is told for it — the pure part (numpy and the vocabulary's words).

A SEGMENT is one word of one column in force: from the step it is said (``row``) to the step the next word of its
column is said (``end_row``), or to the sentence's end — the band the sentence bar draws. The executor flies it to
where the word's own envelope ends (``stop_row``): ``end_row``, and for a HEADING word a lead later, since a heading
word is judged from a lead after it is said to a lead after the next heading word is (vocabulary design §10.1) — the
next heading word is told at ``end_row`` as in the sentence and its first lead is flown. When ``stop_row`` is the
sentence's end, the segment is flown on to its outcome.

TWO SENTENCES, TWO STARTS (``start_row``, where the executor starts flying):

- THE TRUTH's word (`segment_of`): from the OBSERVED aircraft's state at the word's own step, told the six words in force
  there as its step 0 — the labelled sentence says what the observed aircraft was doing there.
- A MODEL's word (`model_segment`): the model's own sentence (`ModelSentence`, the prior's free generation) was flown by
  the executor from the observed state at its FIRST step (`prior.scene.N_LOOK`), every word heard at the step it was said
  — so at a later word the model's aircraft is wherever the executor flew the model's earlier words, not where the
  observed one was. Its segment is that flight flown again from the sentence's first step, to the word's stop: the part
  from the word on is the segment, the part before it how the aircraft got there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, APPROACH, APPROACH_CLEARED, COLUMNS, HEADING, RUNWAY, SPEED, UNCHANGED, Words,
)
from ts_transformer.prior.augment import LIMITS, TIMEOUT_FACTOR, Augmentation
from ts_transformer.prior.scene import N_LOOK

from aeroviz_backend.autopilot_segment.errors import RequestRefused


@dataclass(frozen=True)
class Segment:
    column: int                     # index into `COLUMNS`
    row: int                        # the step its word is said at
    end_row: int                    # the step the next word of its column is said at, or the sentence's length
    stop_row: int                   # where its own envelope ends: ``end_row``, a heading word's a lead later (capped)
    to_landing: bool                # ``stop_row`` is the sentence's end: flown on to the outcome
    start_row: int                  # the flight's step the executor starts at: ``row`` (the truth's) or the model's first
    grid: np.ndarray                # [n, 6] the sentence the executor is told, renumbered from ``start_row``
    instructions: list[Instruction]  # its words, renumbered the same way

    @property
    def word_step(self) -> int:
        """The selected word's step in the flown sentence (0 for the truth's: its segment starts at it)."""
        return self.row - self.start_row

    def selected(self) -> Instruction:
        """The selected word, as the executor was told it."""
        (word,) = [item for item in self.instructions if item.column == self.column and item.row == self.word_step]
        return word


@dataclass(frozen=True)
class ModelSentence:
    """A model's own sentence of a flight (the prior's free generation, one sample), as the request carries it: its
    words from its first step (``first_row``, where it first spoke: every column said there) to its ``rows``, at the
    flight's own steps; ``overlay_id`` and ``sample`` name it for the answer; ``augmentation``: the move of the
    augmented start it was spoken from (`prior.augment`, its overlay's `prior-generation-augmented` flight), None from
    the flight's own start."""

    overlay_id: str
    sample: int
    first_row: int
    rows: int
    grid: np.ndarray                # [rows - first_row, 6], UNCHANGED where a column says nothing; row 0 complete
    # the procedure's masks it was spoken under, as its overlay records them: each set's name and the digest of the data
    # it read (`prior.masks`); empty: the vocabulary's rules alone
    procedure_masks: tuple[tuple[str, str], ...]
    augmentation: Augmentation | None = None

    @property
    def last_runway(self) -> int:
        """The runway the sentence points at last: the one its free generation's outcome is read on
        (`prior_free_generation.flight_rows`)."""
        said = self.grid[:, RUNWAY][self.grid[:, RUNWAY] != UNCHANGED]
        return int(said[-1])


def _int(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RequestRefused(f"{what} must be a whole number, got {value!r}")
    return value


def model_steps_max(observed_rows: int, first_row: int, timeout_factor: float) -> int:
    """The most steps a model's sentence says from its first step: its free generation speaks a step at a time until
    the executor is done, which is by its time limit (`fly.model_time_limit_s`: the observed flight's steps left from
    ``first_row`` × the timeout factor) — the generation's own cap, `prior.generate.rows_for` of that limit from its first
    predicted step (pinned by `test_autopilot_segment.ModelSegmentTest`)."""
    return 1 + math.ceil((observed_rows - first_row) * timeout_factor)


def model_augmentation(value: object) -> Augmentation | None:
    """The request's ``augmentation``: null (the flight's own start), or ``{rotationDeg, altitudeM, speedScale}`` — a
    move within the draw's limits (`prior.augment.LIMITS`), refused by name otherwise."""
    if value is None:
        return None
    keys = ("rotationDeg", "altitudeM", "speedScale")
    # a JSON number past a double's range (a huge integer) has no float: refused like any other non-number
    if not isinstance(value, dict) or set(value) != set(keys) or not all(
            isinstance(value[key], float) and math.isfinite(value[key])
            or isinstance(value[key], int) and not isinstance(value[key], bool) and abs(value[key]) < 1e300
            for key in keys):
        raise RequestRefused(f"the sentence's augmentation must be null or {{rotationDeg, altitudeM, speedScale}}, got {value!r}")
    move = Augmentation(float(value["rotationDeg"]), float(value["altitudeM"]), float(value["speedScale"]))
    if abs(move.rotation_deg) > LIMITS.rotation_deg or abs(move.altitude_m) > LIMITS.altitude_m \
            or abs(move.speed_scale - 1.0) > LIMITS.speed_fraction + 1e-9:
        raise RequestRefused(f"the sentence's augmentation {value!r} is outside the draw's limits: ±{LIMITS.rotation_deg}°, "
                             f"±{LIMITS.altitude_m} m, ×1 ± {LIMITS.speed_fraction}")
    return move


def model_sentence(record: object, words: Words, runways: int, observed_rows: int, timeout_factor: float) -> ModelSentence:
    """A model's sentence from the request's ``sentence`` (``{overlayId, sample, firstRow, rows, events,
    procedureMasks, augmentation}``; ``procedureMasks``: ``[{name, dataSha256}]``, the sets its overlay says it was spoken
    under; ``augmentation``: `model_augmentation`), refused by name unless its words are the vocabulary's, in (step,
    column) order, from a first step inside the observed flight that says every column — the prior's first predicted
    step (`N_LOOK`), where the free generation starts and which its time limit counts from — over no more steps than its
    flight can say (`model_steps_max`, under an augmented start's time limit when it has one)."""
    if not isinstance(record, dict):
        raise RequestRefused(f"the sentence must be an object, got {record!r}")
    missing = [key for key in ("overlayId", "sample", "firstRow", "rows", "events", "procedureMasks", "augmentation")
               if key not in record]
    if missing:
        raise RequestRefused(f"the sentence has no {missing}")
    overlay_id = record["overlayId"]
    if not isinstance(overlay_id, str):
        raise RequestRefused(f"the sentence's overlayId must be a string, got {overlay_id!r}")
    sample = _int(record["sample"], "the sentence's sample")
    first, rows = _int(record["firstRow"], "the sentence's firstRow"), _int(record["rows"], "the sentence's rows")
    if first != N_LOOK:
        raise RequestRefused(f"the sentence starts at step {first}, not the prior's first predicted step {N_LOOK}")
    if first > observed_rows - 2:
        raise RequestRefused(f"the observed flight's {observed_rows} steps leave none to fly after the sentence's first, {first}")
    if rows <= first:
        raise RequestRefused(f"the sentence ends at step {rows}, not after its first step {first}")
    augmentation = model_augmentation(record["augmentation"])
    most = model_steps_max(observed_rows, first, timeout_factor if augmentation is None else TIMEOUT_FACTOR)
    if rows - first > most:
        raise RequestRefused(f"the sentence says {rows - first} steps from its first step {first}: its flight's time limit "
                             f"lets it say {most}")
    counts = [runways, *(words.class_counts()[name] for name in COLUMNS[1:])]
    events = record["events"]
    if not isinstance(events, list):
        raise RequestRefused("the sentence's events must be a list")
    grid = np.full((rows - first, len(COLUMNS)), UNCHANGED, dtype=np.int64)
    last = (-1, -1)
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"row", "column", "value"}:
            raise RequestRefused(f"the sentence's event {index} is not {{row, column, value}}: {event!r}")
        row, column, value = (_int(event[key], f"the sentence's event {index}'s {key}") for key in ("row", "column", "value"))
        if not first <= row < rows or not 0 <= column < len(COLUMNS) or not 0 <= value < counts[column]:
            raise RequestRefused(f"the sentence's event {index} ({row}, {column}, {value}) is not a word of its steps "
                                 f"{first}…{rows - 1} and the vocabulary")
        if (row, column) <= last:
            raise RequestRefused(f"the sentence's events are not in (step, column) order at {index}")
        last = (row, column)
        grid[row - first, column] = value
    if (grid[0] == UNCHANGED).any():
        raise RequestRefused(f"the sentence's first step {first} does not say every column")
    masks = record["procedureMasks"]
    if not isinstance(masks, list) or not all(isinstance(item, dict) and set(item) == {"name", "dataSha256"}
                                              and all(isinstance(value, str) for value in item.values()) for item in masks):
        raise RequestRefused(f"the sentence's procedureMasks must be a list of {{name, dataSha256}}, got {masks!r}")
    return ModelSentence(overlay_id=overlay_id, sample=sample, first_row=first, rows=rows, grid=grid,
                         procedure_masks=tuple((item["name"], item["dataSha256"]) for item in masks),
                         augmentation=augmentation)


def sentence_instructions(grid: np.ndarray, words: Words) -> list[Instruction]:
    """A said sentence's words as the executor's judge reads them — each cell said, at its step (``grid``'s rows): a
    clearance marked as one (the judge ends heading words and judges the corridor by it), a heading word with its
    target. Nothing else of an `Instruction` is read by the judge; the kind only names why a word was said."""
    out = []
    for row, column in zip(*np.nonzero(np.asarray(grid) != UNCHANGED)):
        value = int(grid[row, column])
        clear = column == APPROACH and value == APPROACH_CLEARED
        info = {"target_deg": words.heading_deg(value)} if column == HEADING else {}
        out.append(Instruction(int(column), value, int(row), "clear" if clear else "said", info))
    return out


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
    return Segment(column=column, row=row, end_row=end_row, stop_row=stop_row, to_landing=to_landing, start_row=row,
                   grid=segment, instructions=opening + after)


def model_segment(sentence: ModelSentence, column: int, row: int, lead_rows: int, words: Words) -> Segment:
    """The segment of ``column``'s word the MODEL said at ``row`` (a flight step): the model's sentence from its first
    step to the word's stop — the flight the model's words made, flown again — and, unless the stop is the sentence's
    end, one silent step at the stop. A word the model did not say at ``row`` is refused. One said at its sentence's
    last step is flown to the outcome: unlike the truth's last step (the landing), a model's is the step its flight was
    still flying in when it ended (a word said after the end is refused once flown, `fly.fly_segment`)."""
    first, rows = sentence.first_row, sentence.rows
    if not 0 <= column < len(COLUMNS):
        raise RequestRefused(f"column {column} is not one of the six {COLUMNS}")
    if not first <= row < rows:
        raise RequestRefused(f"step {row} is not a step of the model's sentence, {first}…{rows - 1}")
    grid = sentence.grid
    if grid[row - first, column] == UNCHANGED:
        raise RequestRefused(f"the model says no {COLUMNS[column]} word at step {row}: a segment starts where its word is said")
    later = np.nonzero(grid[row - first + 1:, column] != UNCHANGED)[0]
    end_row = row + 1 + int(later[0]) if len(later) else rows
    stop_row = min(end_row + (lead_rows if column == HEADING else 0), rows)
    to_landing = stop_row == rows
    flown = np.full((stop_row - first + (0 if to_landing else 1), len(COLUMNS)), UNCHANGED, dtype=grid.dtype)
    flown[: stop_row - first] = grid[: stop_row - first]
    return Segment(column=column, row=row, end_row=end_row, stop_row=stop_row, to_landing=to_landing, start_row=first,
                   grid=flown, instructions=sentence_instructions(flown, words))


def model_reading(signals: FlightSignals, segment: Segment, sentence: ModelSentence, words: Words) -> Reading:
    """The reading the executor flies and its judge reads for a model's segment: its words from the sentence's first
    step; the runway the whole sentence points at last (`ModelSentence.last_runway`) — the landing is judged on it, as
    its free generation judged the sample's, so a segment ends where the sample did. A `Reading`'s clearance, capture and
    "unspecified" rows are the labeller's findings on an observed flight, which a model's sentence has none of — the
    judge reads the words — so they are the sentence's own: where it first clears, never captures, first leaves the
    speed."""
    grid = segment.grid
    return Reading(dataset_id=signals.dataset_id, airport=signals.airport, runway_index=sentence.last_runway, words=grid,
                   instructions=segment.instructions, capture_row=len(grid), join_row=_first(grid[:, APPROACH] == APPROACH_CLEARED),
                   unspecified_row=_first(grid[:, SPEED] == words.speed_unspecified), cut_at_crossing=False, checks={})


def judged_reading(reading: Reading, segment: Segment) -> Reading:
    """The reading the judge reads for a segment: the flown one — but for a model's ANGLE word said after the altitude
    word in force at its step, that altitude word said again there ("in force", as the labeller anchors a tube), so the
    tube the angle word re-anchors is judged from its step on, as the truth's segment has it judged (its step 0 says the
    six words in force: every tube it judges starts at or after the word). Without it the judge counts that tube from
    the altitude word's own, earlier step, on rows another angle word anchored. The executor flies the grid, untouched."""
    step = segment.word_step
    if segment.column != ANGLE or step == 0:
        return reading
    in_force = max((word for word in reading.instructions if word.column == ALTITUDE and word.row <= step),
                   key=lambda word: word.row)
    if in_force.row == step:
        return reading
    return replace(reading, instructions=[*reading.instructions, Instruction(ALTITUDE, in_force.value, step, "in force")])


def _first(flags: np.ndarray) -> int:
    """The first step a flag is set at, or the sentence's length."""
    rows = np.nonzero(flags)[0]
    return int(rows[0]) if len(rows) else len(flags)


def told_words(segment: Segment) -> list[dict[str, int]]:
    """The words the executor was told, at the flight's steps, by step then column — the truth's: the six in force at the
    segment's first step, then the sentence's words to its stop; a model's: its sentence from its first step to the stop
    — as the sentence bar shows them (`trainingAutopilot.segmentWords`)."""
    return [{"row": segment.start_row + word.row, "column": word.column, "value": int(word.value)}
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
    """The observed rows from the step the executor starts at, one per step of its sentence (as many as the observed
    flight has): the truth's word clock reads them, and the judge takes the flight's identity from them."""
    stop = segment.start_row + len(segment.grid)
    return replace(signals, **{name: getattr(signals, name)[segment.start_row: stop] for name in ROW_FIELDS})
