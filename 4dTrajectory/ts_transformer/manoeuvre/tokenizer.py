"""The tokenizer (plan §2.4): an encoder over one segment's rows and the current state, then
finite scalar quantization; the DECODER is the executor (`outputs/control/heads.py`), so a
code is worth exactly what the executor can fly with it and nothing else.

Two tokenizers, ONE interface (``forward(segment_rows, state_rows) → (z, code)``,
``codes_to_z``, ``z_to_codes``, ``code_count``, ``z_dim``, ``trainable``):

* **learned** (`LearnedTokenizer`): a small transformer over the segment's ``R`` rows in the
  start frame (`manoeuvre/segments.py`) plus one state token (the current chart row: where
  the aircraft is relative to the runway and how fast it goes, so what the state already
  decides need not be encoded), then FSQ [Mentzer et al. 2024, plan §7.1 [5]] — every
  dimension bounded and rounded to one of ``L_d`` levels, ``K = Π L_d`` codes, no codebook
  table, no commitment loss, gradients through the round by the straight-through estimator.
  ``z`` is the grid coordinate ``[B, D]`` in ``[-1, 1]`` — the vector the executor is
  conditioned on and the prior looks a code up as. Trained JOINTLY with the executor: the
  control objective's gradient reaches the encoder through ``z``.
* **command-vocabulary** (`CommandVocabularyTokenizer`, plan §2.4 baseline B): the segment
  read by rule as one of 5 heading classes × 3 vertical classes × 3 speed classes (45 codes,
  ATC-like); ``z`` is the one-hot. No parameters; same interface, same gate T.

Scales are FIXED module constants (`SEGMENT_ROW_SCALE`, `STATE_ROW_SCALE`), not a dataset
normalizer, and the artefact RECORDS them (a codebook written under other scales is refused
on load), so a frozen codebook is self-contained: any segment of any flight encodes to the
same code in any process (plan §2.4, "码在冻结后是稳定身份"). The encoder's size is a module
constant too — ``K`` and ``segment_s`` are the knobs, the encoder is not.

**The codebook artefact** (`write_codebook` / `load_codebook`, plan §2.4): a directory holding
``codebook.pt`` — the tokenizer's kind, shape, scales and weights, the segment length, the
fitted cohort's identity and the source checkpoint, ALL under one ``sha256`` — and
``codebook.json``, the same minus the weights plus that sha, for readers that open no torch
file. ``codebook_sha256`` is what every prior and executor checkpoint binds to and refuses on
mismatch (C12's rule for the conformal table). Writing refuses an existing directory and an
artefact with no cohort identity; loading verifies the sha and that the manifest mirrors
the payload.

A loaded codebook's tokenizer is FROZEN: its parameters take no gradient and its mode stays
``eval`` under the executor's ``train()`` (the encoder carries no dropout or normalisation
statistics today, so the mode changes nothing numerically — and this keeps it so).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.config import (
    FSQ_MINIMUM_LEVELS,
    MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY,
    MANOEUVRE_TOKENIZER_LEARNED,
    MANOEUVRE_TOKENIZERS,
)
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.segments import MINIMUM_GROUND_SPEED_MPS, SEGMENT_CHANNELS, segment_row_count

#: The segment rows' scale, one per `SEGMENT_CHANNELS` entry: a minute of approach flying
#: laterally (5 km), its descent (300 m), the approach speed (100 m/s) and the descent rate
#: (10 m/s) — every scaled entry is O(1). Recorded in every codebook (see `load_codebook`).
SEGMENT_ROW_SCALE = np.array([5000.0, 5000.0, 300.0, 100.0, 100.0, 10.0], dtype=np.float32)
#: The state row's scale, one per chart channel: the 25 km slice (20 km), its height (2 km),
#: the same speeds.
STATE_ROW_SCALE = np.array([20000.0, 20000.0, 2000.0, 100.0, 100.0, 10.0], dtype=np.float32)
SEGMENT_ROW_SCALE.setflags(write=False)
STATE_ROW_SCALE.setflags(write=False)

#: The learned encoder's size. Constants on purpose: the plan's knobs are K and the segment
#: length; the state dict pins the shape into every checkpoint and codebook.
ENCODER_D_MODEL = 64
ENCODER_HEADS = 4
ENCODER_LAYERS = 2
ENCODER_D_FF = 128
#: The bound shrinks the range by this much (the FSQ paper's eps) so that its ends fall
#: strictly inside the outermost levels' rounding cells: a bounded value can then never sit
#: exactly on a .5 boundary, where `round` would be ambiguous.
FSQ_BOUND_EPS = 1e-3

#: The command vocabulary's classes (plan §2.4 baseline B: 航向 ≤±15°、±45°、±90°、反向 ×
#: 高度 × 速度; the thresholds are readings, not regulation values). Heading: the change of
#: ground-track direction over the segment, the course UNWRAPPED along the rows (a 180°
#: reversal is a reversal, never the opposite turn read off a wrapped angle); positive = left.
#: Vertical and speed: the mean rate over the segment, so the classes do not move with the
#: segment length (a 120 s segment is not "descending" because it is long). A segment reaching
#: into the fitted tail reads that tail's HELD velocity columns (`segments.py`): its last
#: segment's speed change and course are measured against a frozen endpoint.
COMMAND_HEADING_CLASSES = (
    "straight", "left-small", "right-small", "left-medium", "right-medium",
    "left-reversal", "right-reversal",
)
COMMAND_HEADING_SMALL_DEG = 15.0
COMMAND_HEADING_MEDIUM_DEG = 45.0
COMMAND_HEADING_REVERSAL_DEG = 90.0
COMMAND_VERTICAL_CLASSES = ("level", "descend", "climb")
COMMAND_LEVEL_RATE_MPS = 1.0
COMMAND_SPEED_CLASSES = ("hold", "decelerate", "accelerate")
COMMAND_HOLD_ACCELERATION_MPS2 = 0.03
COMMAND_VOCABULARY_SIZE = (
    len(COMMAND_HEADING_CLASSES) * len(COMMAND_VERTICAL_CLASSES) * len(COMMAND_SPEED_CLASSES)
)

_ROW_WIDTH = len(SEGMENT_CHANNELS)


# ── FSQ ──────────────────────────────────────────────────────────────────────

def fsq_code_count(levels: Sequence[int]) -> int:
    """``K = Π L_d``."""
    count = 1
    for level in levels:
        count *= int(level)
    return count


class FSQ(nn.Module):
    """Finite scalar quantization over ``D = len(levels)`` dimensions.

    ``quantize(h)`` bounds each pre-activation to ``[-(L-1)/2, (L-1)/2]`` (shifted by half a
    step for an even ``L`` so 0 sits on a level), rounds it (straight-through), and returns
    ``z = q / (L // 2)`` in ``[-1, 1]`` with the code ``c = Σ_d index_d · Π_{d' < d} L_d'``
    (mixed radix; ``index_d = q_d + L_d // 2 ∈ [0, L_d)``). ``codes_to_z`` is the exact inverse.
    """

    def __init__(self, levels: Sequence[int]):
        super().__init__()
        levels = tuple(int(level) for level in levels)
        if not levels or any(level < FSQ_MINIMUM_LEVELS for level in levels):
            raise ValueError(
                f"FSQ levels are at least one dimension of ≥ {FSQ_MINIMUM_LEVELS} levels each, got {levels!r}"
            )
        self.levels = levels
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32), persistent=False)
        self.register_buffer("_half_width", torch.tensor([level // 2 for level in levels], dtype=torch.float32), persistent=False)
        basis = np.cumprod([1, *levels[:-1]]).astype(np.int64)
        self.register_buffer("_basis", torch.tensor(basis, dtype=torch.long), persistent=False)

    @property
    def code_count(self) -> int:
        return fsq_code_count(self.levels)

    @property
    def z_dim(self) -> int:
        return len(self.levels)

    def bound(self, h: torch.Tensor) -> torch.Tensor:
        half_l = (self._levels - 1.0) * (1.0 - FSQ_BOUND_EPS) / 2.0
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0).to(h.dtype)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(h + shift) * half_l - offset

    def continuous(self, h: torch.Tensor) -> torch.Tensor:
        """The UNROUNDED coordinates ``[B, D]`` in z's own space (the bound, scaled like z): what
        the continuous prior regresses (plan §3.3, the discrete-vs-continuous control)."""
        if h.ndim != 2 or h.shape[1] != self.z_dim:
            raise ValueError(f"FSQ pre-activations are [B, {self.z_dim}], got {tuple(h.shape)}")
        return self.bound(h) / self._half_width

    def quantize(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(z [B, D], code [B])`` from pre-activations ``[B, D]``; the round is straight-through."""
        if h.ndim != 2 or h.shape[1] != self.z_dim:
            raise ValueError(f"FSQ pre-activations are [B, {self.z_dim}], got {tuple(h.shape)}")
        bounded = self.bound(h)
        quantized = bounded + (torch.round(bounded) - bounded).detach()
        z = quantized / self._half_width
        return z, self.z_to_codes(z)

    def level_indices(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` per-dimension level indices of ``z``: exact on a grid coordinate, the
        NEAREST level for a z off the grid (a continuous decode, a clamped or averaged z) — so
        a code is always inside the vocabulary the prior embeds."""
        indices = torch.round(z * self._half_width + self._half_width)
        return torch.minimum(indices.clamp(min=0.0), self._levels - 1.0).to(torch.long)

    def z_to_codes(self, z: torch.Tensor) -> torch.Tensor:
        return (self.level_indices(z) * self._basis).sum(dim=-1)

    def codes_to_z(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.to(torch.long)
        if codes.ndim != 1 or bool((codes < 0).any()) or bool((codes >= self.code_count).any()):
            raise ValueError(f"codes are a [B] vector in [0, {self.code_count}), got {tuple(codes.shape)}")
        indices = (codes.unsqueeze(-1) // self._basis) % self._levels.to(torch.long)
        return (indices.to(torch.float32) - self._half_width) / self._half_width


# ── the learned tokenizer ────────────────────────────────────────────────────

class SegmentEncoder(nn.Module):
    """One state token + ``rows`` row tokens (row index as position) through a small
    transformer; the state token's output is projected to the FSQ pre-activations."""

    def __init__(self, rows: int, out_dim: int):
        super().__init__()
        self.rows = int(rows)
        self.row_embedding = nn.Linear(_ROW_WIDTH, ENCODER_D_MODEL)
        self.position = nn.Parameter(torch.empty(self.rows, ENCODER_D_MODEL))
        self.state_embedding = nn.Linear(_ROW_WIDTH, ENCODER_D_MODEL)
        self.state_token = nn.Parameter(torch.empty(1, 1, ENCODER_D_MODEL))
        layer = nn.TransformerEncoderLayer(
            ENCODER_D_MODEL, ENCODER_HEADS, dim_feedforward=ENCODER_D_FF, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, ENCODER_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(ENCODER_D_MODEL)
        self.projection = nn.Linear(ENCODER_D_MODEL, int(out_dim))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.state_token, std=0.02)

    def forward(self, segment_rows: torch.Tensor, state_rows: torch.Tensor) -> torch.Tensor:
        if segment_rows.shape[1:] != (self.rows, _ROW_WIDTH) or state_rows.shape[1:] != (_ROW_WIDTH,):
            raise ValueError(
                f"the encoder reads [B, {self.rows}, {_ROW_WIDTH}] segment rows and [B, {_ROW_WIDTH}] state "
                f"rows, got {tuple(segment_rows.shape)} and {tuple(state_rows.shape)}"
            )
        tokens = self.row_embedding(segment_rows) + self.position
        state = self.state_embedding(state_rows).unsqueeze(1) + self.state_token
        encoded = self.transformer(torch.cat((state, tokens), dim=1))
        return self.projection(self.norm(encoded[:, 0]))


class LearnedTokenizer(nn.Module):
    """Encoder + FSQ over PHYSICAL rows (the scales are applied here)."""

    kind = MANOEUVRE_TOKENIZER_LEARNED
    trainable = True
    #: Set by `load_codebook`: the mode stays `eval` whatever the parent module is put into.
    frozen = False

    def train(self, mode: bool = True):
        return super().train(mode and not self.frozen)

    def __init__(self, levels: Sequence[int], rows: int):
        super().__init__()
        self.fsq = FSQ(levels)
        self.encoder = SegmentEncoder(rows, self.fsq.z_dim)
        self.register_buffer("_segment_scale", torch.tensor(SEGMENT_ROW_SCALE), persistent=False)
        self.register_buffer("_state_scale", torch.tensor(STATE_ROW_SCALE), persistent=False)

    @property
    def levels(self) -> tuple[int, ...]:
        return self.fsq.levels

    @property
    def rows(self) -> int:
        return self.encoder.rows

    @property
    def code_count(self) -> int:
        return self.fsq.code_count

    @property
    def z_dim(self) -> int:
        return self.fsq.z_dim

    def pre_activations(self, segment_rows: torch.Tensor, state_rows: torch.Tensor) -> torch.Tensor:
        scale = self._segment_scale.to(segment_rows.dtype)
        return self.encoder(
            (segment_rows / scale).to(torch.float32),
            (state_rows / self._state_scale.to(state_rows.dtype)).to(torch.float32),
        )

    def forward(self, segment_rows: torch.Tensor, state_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fsq.quantize(self.pre_activations(segment_rows, state_rows))

    def continuous(self, segment_rows: torch.Tensor, state_rows: torch.Tensor) -> torch.Tensor:
        """The unrounded coordinates ``[B, Z]`` (`FSQ.continuous`) of physical rows."""
        return self.fsq.continuous(self.pre_activations(segment_rows, state_rows))

    def codes_to_z(self, codes: torch.Tensor) -> torch.Tensor:
        return self.fsq.codes_to_z(codes)

    def z_to_codes(self, z: torch.Tensor) -> torch.Tensor:
        return self.fsq.z_to_codes(z)


# ── the command-vocabulary baseline ──────────────────────────────────────────

def command_classes(segment_rows: np.ndarray, segment_s: float) -> tuple[int, int, int]:
    """``(heading, vertical, speed)`` class indices of ONE segment's rows in the start frame.

    The heading change is the course unwrapped row by row (2 s rows cannot turn 180° between
    two of them), so a reversal reads as ``±reversal`` and never as the opposite small turn.
    A start row without a course (below `MINIMUM_GROUND_SPEED_MPS`) is refused, as the frame
    refuses it — a padded row must not read as ``straight/level/hold``.
    """
    rows = np.asarray(segment_rows, dtype=np.float64)
    horizontal_dot = list(VELOCITY_IDX[:2])
    ground_speed = np.hypot(rows[:, horizontal_dot[0]], rows[:, horizontal_dot[1]])
    if ground_speed[0] < MINIMUM_GROUND_SPEED_MPS:
        raise ValueError(
            f"the segment's start row has {ground_speed[0]:.3f} m/s of ground speed: no course to "
            "read a heading change from"
        )
    course = np.unwrap(np.arctan2(rows[:, horizontal_dot[1]], rows[:, horizontal_dot[0]]))
    heading_deg = math.degrees(course[-1] - course[0])     # the first row's course is 0
    size = abs(heading_deg)
    if size <= COMMAND_HEADING_SMALL_DEG:
        heading = 0
    elif size <= COMMAND_HEADING_MEDIUM_DEG:
        heading = 1 if heading_deg > 0.0 else 2
    elif size <= COMMAND_HEADING_REVERSAL_DEG:
        heading = 3 if heading_deg > 0.0 else 4
    else:
        heading = 5 if heading_deg > 0.0 else 6
    vertical_rate = rows[-1, POSITION_IDX[2]] / segment_s
    vertical = 0 if abs(vertical_rate) <= COMMAND_LEVEL_RATE_MPS else (1 if vertical_rate < 0.0 else 2)
    acceleration = (ground_speed[-1] - ground_speed[0]) / segment_s
    speed = 0 if abs(acceleration) <= COMMAND_HOLD_ACCELERATION_MPS2 else (1 if acceleration < 0.0 else 2)
    return heading, vertical, speed


def command_code(heading: int, vertical: int, speed: int) -> int:
    return (heading * len(COMMAND_VERTICAL_CLASSES) + vertical) * len(COMMAND_SPEED_CLASSES) + speed


def command_label(code: int) -> str:
    """``heading/vertical/speed`` words of a command code."""
    speed = code % len(COMMAND_SPEED_CLASSES)
    rest = code // len(COMMAND_SPEED_CLASSES)
    vertical = rest % len(COMMAND_VERTICAL_CLASSES)
    heading = rest // len(COMMAND_VERTICAL_CLASSES)
    return f"{COMMAND_HEADING_CLASSES[heading]}/{COMMAND_VERTICAL_CLASSES[vertical]}/{COMMAND_SPEED_CLASSES[speed]}"


class CommandVocabularyTokenizer(nn.Module):
    """The rule-read vocabulary: ``z`` is the code's one-hot, no parameters, no gradient."""

    kind = MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY
    trainable = False
    frozen = False
    levels: tuple[int, ...] = ()

    def train(self, mode: bool = True):
        return super().train(mode and not self.frozen)

    def __init__(self, rows: int, segment_s: float):
        super().__init__()
        self.rows = int(rows)
        self.segment_s = float(segment_s)

    @property
    def code_count(self) -> int:
        return COMMAND_VOCABULARY_SIZE

    @property
    def z_dim(self) -> int:
        return COMMAND_VOCABULARY_SIZE

    def forward(self, segment_rows: torch.Tensor, state_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del state_rows  # the rules read the segment alone
        if segment_rows.shape[1:] != (self.rows, _ROW_WIDTH):
            raise ValueError(f"[B, {self.rows}, {_ROW_WIDTH}] segment rows expected, got {tuple(segment_rows.shape)}")
        rows = segment_rows.detach().cpu().numpy()
        codes = torch.tensor(
            [command_code(*command_classes(item, self.segment_s)) for item in rows],
            dtype=torch.long, device=segment_rows.device,
        )
        return self.codes_to_z(codes), codes

    def codes_to_z(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.to(torch.long)
        if codes.ndim != 1 or bool((codes < 0).any()) or bool((codes >= self.code_count).any()):
            raise ValueError(f"codes are a [B] vector in [0, {self.code_count}), got {tuple(codes.shape)}")
        return nn.functional.one_hot(codes, self.code_count).to(torch.float32)

    def z_to_codes(self, z: torch.Tensor) -> torch.Tensor:
        return z.argmax(dim=-1)


def tokenizer_for(kind: str, *, levels: Sequence[int], segment_s: float, dt_s: float) -> nn.Module:
    """The tokenizer of one ``manoeuvre_tokenizer`` value, sized to the segment."""
    rows = segment_row_count(segment_s, dt_s)
    if kind == MANOEUVRE_TOKENIZER_LEARNED:
        return LearnedTokenizer(levels, rows)
    if kind == MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY:
        if tuple(levels):
            raise ValueError("the command vocabulary has no FSQ levels; manoeuvre_fsq_levels must be empty")
        return CommandVocabularyTokenizer(rows, segment_s)
    raise ValueError(f"unknown manoeuvre_tokenizer {kind!r}; expected one of {MANOEUVRE_TOKENIZERS}")


# ── the codebook artefact ────────────────────────────────────────────────────

CODEBOOK_SCHEMA = "ts-manoeuvre-codebook-v1"
CODEBOOK_WEIGHTS = "codebook.pt"
CODEBOOK_MANIFEST = "codebook.json"


@dataclass(frozen=True)
class Codebook:
    """A FROZEN tokenizer with its identity: ``sha256`` (of the weights file) is what a
    checkpoint binds to as ``codebook_sha256``."""

    tokenizer: nn.Module
    kind: str
    levels: tuple[int, ...]
    rows: int
    segment_s: float
    dt_s: float
    data_identity: dict[str, Any]
    source: dict[str, Any]
    sha256: str
    path: Path

    @property
    def code_count(self) -> int:
        return self.tokenizer.code_count

    @property
    def z_dim(self) -> int:
        return self.tokenizer.z_dim

    def encode(self, segment_rows: np.ndarray, state_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(codes [B], z [B, Z])`` of a batch of physical segment rows ``[B, R, 6]`` and state
        rows ``[B, 6]`` (a single segment ``[R, 6]`` / ``[6]`` is a batch of one), no gradient."""
        segment = np.asarray(segment_rows, dtype=np.float32)
        state = np.asarray(state_rows, dtype=np.float32)
        single = segment.ndim == 2
        if single:
            segment, state = segment[None], state[None]
        with torch.no_grad():
            z, codes = self.tokenizer(torch.from_numpy(segment), torch.from_numpy(state))
        z, codes = z.cpu().numpy(), codes.cpu().numpy().astype(np.int64)
        return (codes[0], z[0]) if single else (codes, z)

    def encode_continuous(self, segment_rows: np.ndarray, state_rows: np.ndarray) -> np.ndarray:
        """The UNROUNDED coordinates ``[B, Z]`` of a batch (a single segment: ``[Z]``): the
        continuous prior's regression target. The command vocabulary has no continuous form."""
        if self.kind != MANOEUVRE_TOKENIZER_LEARNED:
            raise ValueError(f"a {self.kind} codebook has no continuous coordinates")
        segment = np.asarray(segment_rows, dtype=np.float32)
        state = np.asarray(state_rows, dtype=np.float32)
        single = segment.ndim == 2
        if single:
            segment, state = segment[None], state[None]
        with torch.no_grad():
            vector = self.tokenizer.continuous(torch.from_numpy(segment), torch.from_numpy(state)).cpu().numpy()
        return vector[0] if single else vector


def _tokenizer_spec(tokenizer: nn.Module, segment_s: float, dt_s: float) -> dict[str, Any]:
    return {
        "kind": tokenizer.kind,
        "levels": list(tokenizer.levels),
        "rows": int(tokenizer.rows),
        "code_count": int(tokenizer.code_count),
        "z_dim": int(tokenizer.z_dim),
        "segment_s": float(segment_s),
        "dt_s": float(dt_s),
        # The scales the rows were encoded under: part of what decides a code, so part of the
        # artefact and of its sha (a build with other constants refuses the codebook).
        "segment_row_scale": [float(value) for value in SEGMENT_ROW_SCALE],
        "state_row_scale": [float(value) for value in STATE_ROW_SCALE],
    }


def _current_scales() -> dict[str, list[float]]:
    return {
        "segment_row_scale": [float(value) for value in SEGMENT_ROW_SCALE],
        "state_row_scale": [float(value) for value in STATE_ROW_SCALE],
    }


def write_codebook(
    directory: str | Path, tokenizer: nn.Module, *, segment_s: float, dt_s: float,
    data_identity: dict[str, Any], source: dict[str, Any],
) -> Codebook:
    """Freeze ``tokenizer`` into ``directory`` (refused if it exists: an artefact is never
    overwritten). ``data_identity`` is the fitted cohort's identity (C26: the eligible set),
    ``source`` names the checkpoint the weights come from."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"codebook directory {directory} exists; an artefact is never overwritten")
    if int(tokenizer.rows) != segment_row_count(segment_s, dt_s):
        raise ValueError(
            f"the tokenizer reads {tokenizer.rows} rows, a {segment_s:g} s / {dt_s:g} s segment has "
            f"{segment_row_count(segment_s, dt_s)}"
        )
    if not data_identity:
        raise ValueError("a codebook without its fitted cohort's identity is refused (C26)")
    spec = _tokenizer_spec(tokenizer, segment_s, dt_s)
    directory.mkdir(parents=True)
    weights = directory / CODEBOOK_WEIGHTS
    # Everything that identifies the codebook is INSIDE the hashed file; the manifest mirrors it.
    payload = {
        "schema": CODEBOOK_SCHEMA,
        "tokenizer": spec,
        "data_identity": dict(data_identity),
        "source": dict(source),
        "state_dict": {key: value.detach().cpu() for key, value in tokenizer.state_dict().items()},
    }
    torch.save(payload, weights)
    sha256 = file_sha256(weights)
    write_json_atomic(directory / CODEBOOK_MANIFEST, {
        "schema": CODEBOOK_SCHEMA,
        "written_utc": utc_now(),
        "tokenizer": spec,
        "weights_file": CODEBOOK_WEIGHTS,
        "sha256": sha256,
        "data_identity": payload["data_identity"],
        "source": payload["source"],
    })
    return load_codebook(directory)


def load_codebook(directory: str | Path) -> Codebook:
    """The frozen tokenizer of a codebook directory; refuses a manifest/weights sha mismatch."""
    directory = Path(directory)
    manifest = json.loads((directory / CODEBOOK_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != CODEBOOK_SCHEMA:
        raise ValueError(f"{directory}: codebook schema {manifest.get('schema')!r} is not {CODEBOOK_SCHEMA!r}")
    weights = directory / manifest["weights_file"]
    sha256 = file_sha256(weights)
    if sha256 != manifest["sha256"]:
        raise ValueError(f"{directory}: {weights.name} sha256 {sha256[:12]}… is not the manifest's {manifest['sha256'][:12]}…")
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    spec = payload["tokenizer"]
    for key in ("tokenizer", "data_identity", "source"):
        if payload[key] != manifest[key]:
            raise ValueError(f"{directory}: the manifest's {key!r} differs from the hashed payload's")
    scales = {key: spec[key] for key in ("segment_row_scale", "state_row_scale")}
    if scales != _current_scales():
        raise ValueError(
            f"{directory}: the codebook was written under scales {scales}, this build's are "
            f"{_current_scales()}; its codes would not be reproduced"
        )
    tokenizer = tokenizer_for(spec["kind"], levels=spec["levels"], segment_s=spec["segment_s"], dt_s=spec["dt_s"])
    tokenizer.load_state_dict(payload["state_dict"])
    tokenizer.frozen = True
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return Codebook(
        tokenizer=tokenizer, kind=spec["kind"], levels=tuple(spec["levels"]), rows=int(spec["rows"]),
        segment_s=float(spec["segment_s"]), dt_s=float(spec["dt_s"]),
        data_identity=dict(payload["data_identity"]), source=dict(payload["source"]),
        sha256=sha256, path=directory,
    )


__all__ = [
    "CODEBOOK_MANIFEST", "CODEBOOK_SCHEMA", "CODEBOOK_WEIGHTS", "COMMAND_HEADING_CLASSES",
    "COMMAND_SPEED_CLASSES", "COMMAND_VERTICAL_CLASSES", "COMMAND_VOCABULARY_SIZE",
    "ENCODER_D_MODEL", "ENCODER_D_FF", "ENCODER_HEADS", "ENCODER_LAYERS", "FSQ",
    "SEGMENT_ROW_SCALE", "STATE_ROW_SCALE", "Codebook", "CommandVocabularyTokenizer",
    "LearnedTokenizer", "SegmentEncoder", "command_classes", "command_code", "command_label",
    "fsq_code_count", "load_codebook", "tokenizer_for", "write_codebook",
]
