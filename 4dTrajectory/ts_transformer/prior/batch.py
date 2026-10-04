"""One sentence's inputs and targets on its Δ rows, and a batch of them (prior design §2, §3, §7 item 2).

`SentenceRows` is what the prior reads of one flight: the inputs of each row (what a controller knows before the step,
principle 7) and the five words said at it. The input function of milestone B1 makes it from the sentence artefact and
the loops make it from the executor's states; the model reads only `RowTensors`, a batch of them (`collate`).

**Rows and time.** A row every Δ (vocabulary §6, item 7). ``time_s`` is each row's time in seconds from the aircraft's
own row 0: the model reads only differences of it (RoPE, D16). ``first_step`` is the row of the first predicted step;
the rows before it are observed only, never asked.

**Encodings** (the vocabulary's own values, `instructions.words`):

- ``runway_in_force``: −1 "none yet" (up to and with the first predicted step), else the candidate's index. Nothing is
  in force before the first predicted step says every column, so this one value also says "none yet" for the
  heading, which has no index of its own;
- ``go_around``: G, the go-around state;
- ``heading_in_force``: the sine and cosine of the heading word in force relative to the course of R, (0, 0) none yet;
- ``words_in_force``: the altitude, angle and speed words in force (`IN_FORCE_WORDS`), −1 none yet;
- ``since``: for each column, log(1 + t / 2 s) / 5 of the time t since it said its word in force (D17), as B1 writes it;
- ``targets``: the five words said at the row — `UNCHANGED` (−1), `RUNWAY_GO_AROUND` (−2), a candidate's index, a
  word's index. `target_classes` turns them into the heads' classes: in the runway column 0 "unchanged", 1
  "go-around", 2 + k candidate k (`RUNWAY_FIXED_CLASSES` before the candidates, whose number is the airport's own,
  D41); in the other columns 0 "unchanged", 1 + k word k.

**Candidates** (D23, D24, D39). ``candidates`` holds one vector a candidate a row, ``variant_features(variant)`` wide:
the values that change with the aircraft (`CANDIDATE_FEATURES`), and in the variant ``constants`` also the runway's
length and threshold elevation (`RUNWAY_CONSTANT_FEATURES`). A sentence has its airport's candidates, in the order of
the runway word's pointer; a batch pads them, and ``valid`` says which are real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED

#: The own state (D5, D23, D58, D60): no frame, no MSL height; ``no_motion`` is 1 at row 0 only.
OWN_FEATURES = ("height_above_elevation", "ground_speed", "vertical_rate", "no_motion")
#: The inputs that come from the motion in the 2 s before a row (D25): 0 at row 0, which has no state before it (D60).
OWN_MOTION_FEATURES = ("ground_speed", "vertical_rate")
#: A candidate vector (D23, D24): the aircraft in the frame of the candidate, the height above its glidepath, the
#: landings on it in the 30 min before the step.
CANDIDATE_FEATURES = ("before_threshold", "right_of_final", "height_above_threshold", "motion_minus_course_sin",
                      "motion_minus_course_cos", "height_above_glidepath", "landings_30min")
CANDIDATE_MOTION_FEATURES = ("motion_minus_course_sin", "motion_minus_course_cos")
#: The runway constants of the variant ``constants`` (D39).
RUNWAY_CONSTANT_FEATURES = ("length", "threshold_elevation")
VARIANTS = {"full": CANDIDATE_FEATURES, "constants": CANDIDATE_FEATURES + RUNWAY_CONSTANT_FEATURES}
#: The columns whose word in force is an embedding (§2): the runway is its candidate vector, the heading sine and cosine.
IN_FORCE_WORDS = ("altitude", "angle", "speed")
#: The runway head's classes before the candidates: "unchanged", "go-around".
RUNWAY_UNCHANGED_CLASS, RUNWAY_GO_AROUND_CLASS = 0, 1
RUNWAY_FIXED_CLASSES = 2


def variant_features(variant: str) -> tuple[str, ...]:
    """The candidate features of a variant of D39; an unknown name raises."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} is not one of {sorted(VARIANTS)}")
    return VARIANTS[variant]


@dataclass(frozen=True)
class SentenceRows:
    """One flight's rows (module docstring). The constructor checks the shapes and the first predicted step."""

    flight_key: str
    airport: str
    split: str
    first_step: int
    time_s: np.ndarray            # [N] float
    own: np.ndarray               # [N, len(OWN_FEATURES)] float
    candidates: np.ndarray        # [N, K, F] float
    runway_in_force: np.ndarray   # [N] int
    go_around: np.ndarray         # [N] bool
    heading_in_force: np.ndarray  # [N, 2] float
    words_in_force: np.ndarray    # [N, len(IN_FORCE_WORDS)] int
    since: np.ndarray             # [N, len(COLUMNS)] float
    targets: np.ndarray           # [N, len(COLUMNS)] int

    def __post_init__(self) -> None:
        rows = len(self.time_s)
        shapes = {"own": (rows, len(OWN_FEATURES)), "runway_in_force": (rows,), "go_around": (rows,),
                  "heading_in_force": (rows, 2), "words_in_force": (rows, len(IN_FORCE_WORDS)),
                  "since": (rows, len(COLUMNS)), "targets": (rows, len(COLUMNS))}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{self.flight_key}: {name} has shape {getattr(self, name).shape}, not {shape}")
        if self.candidates.ndim != 3 or self.candidates.shape[0] != rows or self.candidates.shape[1] < 1:
            raise ValueError(f"{self.flight_key}: candidates have shape {self.candidates.shape}, not [{rows}, K ≥ 1, F]")
        if not 0 <= self.first_step < rows:
            raise ValueError(f"{self.flight_key}: first predicted step {self.first_step} outside its {rows} rows")
        for name in ("time_s", "own", "candidates", "heading_in_force", "since"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{self.flight_key}: {name} is not finite")
        # the aircraft's own seconds: a UTC time in float32 has a step of 128 s and loses the rows (§2, D16)
        if self.time_s[0] != 0.0 or np.any(np.diff(self.time_s.astype(np.float32)) <= 0.0):
            raise ValueError(f"{self.flight_key}: the row times are not seconds from row 0, increasing in float32")
        # D60: row 0 has no state 2 s before it — its motion inputs are 0 and `no_motion` says so; no other row has it
        no_motion = self.own[:, OWN_FEATURES.index("no_motion")]
        motion = [OWN_FEATURES.index(name) for name in OWN_MOTION_FEATURES]
        candidate_motion = [CANDIDATE_FEATURES.index(name) for name in CANDIDATE_MOTION_FEATURES]
        if (no_motion[0] != 1.0 or np.any(no_motion[1:] != 0.0) or np.any(self.own[0, motion] != 0.0)
                or np.any(self.candidates[0][:, candidate_motion] != 0.0)):
            raise ValueError(f"{self.flight_key}: row 0 must have no motion (0, `no_motion` 1) and no other row "
                             f"`no_motion` (D60)")
        # D23: nothing is in force up to and with the first predicted step, so no input there comes from R
        said = slice(0, self.first_step + 1)
        if (np.any(self.runway_in_force[said] != -1) or np.any(self.words_in_force[said] != -1)
                or np.any(self.heading_in_force[said] != 0.0) or np.any(self.go_around[said])):
            raise ValueError(f"{self.flight_key}: a word is in force at or before the first predicted step")
        slots = self.candidates.shape[1]
        after = self.runway_in_force[self.first_step + 1:]
        if np.any((after < 0) | (after >= slots)):
            raise ValueError(f"{self.flight_key}: a runway in force after the first predicted step is not one of its "
                             f"{slots} candidates")
        runway = self.targets[:, RUNWAY]
        if np.any(((runway < 0) | (runway >= slots)) & (runway != UNCHANGED) & (runway != RUNWAY_GO_AROUND)):
            raise ValueError(f"{self.flight_key}: a runway word is neither a word of the column nor one of its {slots} "
                             f"candidates")
        if np.any(self.targets[:, RUNWAY + 1:] < UNCHANGED) or np.any(self.words_in_force < -1):
            raise ValueError(f"{self.flight_key}: a word below \"unchanged\"")
        first = self.targets[self.first_step]
        if np.any(first == UNCHANGED) or not 0 <= first[RUNWAY] < slots:
            raise ValueError(f"{self.flight_key}: the first predicted step says {first.tolist()}, not a word in "
                             f"every column and a candidate runway")

    @property
    def rows(self) -> int:
        return len(self.time_s)


def require_words(sentences: Sequence[SentenceRows], word_values: Sequence[int]) -> None:
    """Refuse a sentence whose word (said or in force) of a column after the runway is not one of the spec's
    ``word_values`` of that column (`model.PriorConfig.word_values`): checked once where a run's data come in, so a
    word out of range is named by its flight, not met as an index error inside a batch."""
    for s in sentences:
        for k, count in enumerate(word_values):
            if np.any(s.targets[:, RUNWAY + 1 + k] >= count):
                raise ValueError(f"{s.flight_key}: a word of {COLUMNS[RUNWAY + 1 + k]} beyond its {count} values")
        for k, name in enumerate(IN_FORCE_WORDS):
            if np.any(s.words_in_force[:, k] >= word_values[COLUMNS.index(name) - 1]):
                raise ValueError(f"{s.flight_key}: a {name} word in force beyond its values")


def target_classes(targets: np.ndarray) -> np.ndarray:
    """The heads' classes of the vocabulary's words ``targets`` [..., 5] (module docstring)."""
    classes = targets.astype(np.int64) + 1
    runway = targets[..., RUNWAY]
    classes[..., RUNWAY] = np.where(runway == UNCHANGED, RUNWAY_UNCHANGED_CLASS,
                                    np.where(runway == RUNWAY_GO_AROUND, RUNWAY_GO_AROUND_CLASS,
                                             runway + RUNWAY_FIXED_CLASSES))
    return classes


class RowTensors(NamedTuple):
    """A batch of sentences, ``[B, R]`` padded to the longest and ``[B, R, K]`` to the most candidates."""

    time_s: torch.Tensor           # [B, R] float
    own: torch.Tensor              # [B, R, len(OWN_FEATURES)]
    candidates: torch.Tensor       # [B, R, K, F]
    valid: torch.Tensor            # [B, K] bool: a real candidate
    runway_in_force: torch.Tensor  # [B, R] long, −1 none yet
    go_around: torch.Tensor        # [B, R] bool
    heading_in_force: torch.Tensor  # [B, R, 2]
    words_in_force: torch.Tensor   # [B, R, 3] long, −1 none yet
    since: torch.Tensor            # [B, R, 5]
    present: torch.Tensor          # [B, R] bool: a row of the sentence, not padding
    first: torch.Tensor            # [B, R] bool: the first predicted step
    asked: torch.Tensor            # [B, R] bool: present, at or after the first predicted step
    targets: torch.Tensor          # [B, R, 5] long, the heads' classes (`target_classes`)

    def between(self, start: int, end: int) -> RowTensors:
        """Rows ``start`` … ``end − 1`` of every sentence (the candidates' validity is the sentence's)."""
        return RowTensors(*(value if name == "valid" else value[:, start:end]
                            for name, value in zip(self._fields, self)))


def collate(sentences: Sequence[SentenceRows], device: torch.device) -> RowTensors:
    """``sentences`` as one batch on ``device`` (module docstring); every one must have the same candidate width."""
    if not sentences:
        raise ValueError("an empty batch")
    width = sentences[0].candidates.shape[2]
    if any(s.candidates.shape[2] != width for s in sentences):
        raise ValueError("the sentences of a batch have candidate vectors of different widths")
    count, rows = len(sentences), max(s.rows for s in sentences)
    slots = max(s.candidates.shape[1] for s in sentences)
    time_s = np.zeros((count, rows), dtype=np.float32)
    own = np.zeros((count, rows, len(OWN_FEATURES)), dtype=np.float32)
    candidates = np.zeros((count, rows, slots, width), dtype=np.float32)
    valid = np.zeros((count, slots), dtype=bool)
    runway = np.full((count, rows), -1, dtype=np.int64)
    go_around = np.zeros((count, rows), dtype=bool)
    heading = np.zeros((count, rows, 2), dtype=np.float32)
    words = np.full((count, rows, len(IN_FORCE_WORDS)), -1, dtype=np.int64)
    since = np.zeros((count, rows, len(COLUMNS)), dtype=np.float32)
    present = np.zeros((count, rows), dtype=bool)
    first = np.zeros((count, rows), dtype=bool)
    asked = np.zeros((count, rows), dtype=bool)
    targets = np.zeros((count, rows, len(COLUMNS)), dtype=np.int64)
    for b, s in enumerate(sentences):
        n, k = s.rows, s.candidates.shape[1]
        time_s[b, :n], own[b, :n], candidates[b, :n, :k], valid[b, :k] = s.time_s, s.own, s.candidates, True
        runway[b, :n], go_around[b, :n], heading[b, :n] = s.runway_in_force, s.go_around, s.heading_in_force
        words[b, :n], since[b, :n], present[b, :n] = s.words_in_force, s.since, True
        first[b, s.first_step], asked[b, s.first_step: n] = True, True
        targets[b, :n] = target_classes(s.targets)
    arrays = (time_s, own, candidates, valid, runway, go_around, heading, words, since, present, first, asked, targets)
    return RowTensors(*(torch.as_tensor(a, device=device) for a in arrays))
