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
normalizer, so a frozen codebook is self-contained: any segment of any flight encodes to the
same code in any process (plan §2.4, "码在冻结后是稳定身份"). The encoder's size is a module
constant too — ``K`` and ``segment_s`` are the knobs, the encoder is not.

**The codebook artefact** (`write_codebook` / `load_codebook`, plan §2.4): a directory holding
``codebook.pt`` (the tokenizer's kind, shape and weights) and ``codebook.json`` (the same
minus the weights, plus the ``sha256`` of ``codebook.pt``, the segment length, the fitted
cohort's identity and the source checkpoint). ``codebook_sha256`` is what every prior and
executor checkpoint binds to and refuses on mismatch (C12's rule for the conformal table).
Writing refuses an existing directory; loading verifies the sha.
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

from ts_transformer.data.channels import CHANNELS
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.segments import SEGMENT_CHANNELS, segment_row_count

MANOEUVRE_TOKENIZER_LEARNED = "learned"
MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY = "command-vocabulary"
MANOEUVRE_TOKENIZERS = (MANOEUVRE_TOKENIZER_LEARNED, MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY)

#: The segment rows' scale, one per `SEGMENT_CHANNELS` entry: a minute of approach flying
#: laterally (5 km), its descent (300 m), the approach speed (100 m/s) and the descent rate
#: (10 m/s) — every scaled entry is O(1).
SEGMENT_ROW_SCALE = np.array([5000.0, 5000.0, 300.0, 100.0, 100.0, 10.0], dtype=np.float32)
#: The state row's scale, one per chart channel: the 25 km slice (20 km), its height (2 km),
#: the same speeds.
STATE_ROW_SCALE = np.array([20000.0, 20000.0, 2000.0, 100.0, 100.0, 10.0], dtype=np.float32)

#: The learned encoder's size. Constants on purpose: the plan's knobs are K and the segment
#: length; the state dict pins the shape into every checkpoint and codebook.
ENCODER_D_MODEL = 64
ENCODER_HEADS = 4
ENCODER_LAYERS = 2
ENCODER_D_FF = 128
#: The bound leaves this much of the outermost half-step unused so the extreme levels are
#: reachable by rounding (the FSQ paper's eps).
FSQ_BOUND_EPS = 1e-3
#: A dimension needs an interior level: at L = 2 the bound's shift is undefined
#: (`atanh(1)`), so the smallest admissible level count is 3.
FSQ_MINIMUM_LEVELS = 3

#: The command vocabulary's classes (plan §2.4 baseline B; the thresholds are readings, not
#: regulation values). Heading: the change of ground-track direction over the segment (the
#: last row's course in the start frame, where the first row's is 0; positive = left).
#: Vertical and speed: the mean rate over the segment, so the classes do not move with the
#: segment length (a 120 s segment is not "descending" because it is long).
COMMAND_HEADING_CLASSES = ("straight", "left-small", "right-small", "left-large", "right-large")
COMMAND_HEADING_SMALL_DEG = 15.0
COMMAND_HEADING_LARGE_DEG = 45.0
COMMAND_VERTICAL_CLASSES = ("level", "descend", "climb")
COMMAND_LEVEL_RATE_MPS = 1.0
COMMAND_SPEED_CLASSES = ("hold", "decelerate", "accelerate")
COMMAND_HOLD_ACCELERATION_MPS2 = 0.03
COMMAND_VOCABULARY_SIZE = (
    len(COMMAND_HEADING_CLASSES) * len(COMMAND_VERTICAL_CLASSES) * len(COMMAND_SPEED_CLASSES)
)

_ROW_WIDTH = len(SEGMENT_CHANNELS)
assert _ROW_WIDTH == len(CHANNELS)


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

    def quantize(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(z [B, D], code [B])`` from pre-activations ``[B, D]``; the round is straight-through."""
        if h.ndim != 2 or h.shape[1] != self.z_dim:
            raise ValueError(f"FSQ pre-activations are [B, {self.z_dim}], got {tuple(h.shape)}")
        bounded = self.bound(h)
        quantized = bounded + (torch.round(bounded) - bounded).detach()
        z = quantized / self._half_width
        return z, self.z_to_codes(z)

    def level_indices(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` per-dimension level indices of grid coordinates ``z``."""
        return torch.round(z * self._half_width + self._half_width).to(torch.long)

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

    def forward(self, segment_rows: torch.Tensor, state_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._segment_scale.to(segment_rows.dtype)
        h = self.encoder(
            (segment_rows / scale).to(torch.float32),
            (state_rows / self._state_scale.to(state_rows.dtype)).to(torch.float32),
        )
        return self.fsq.quantize(h)

    def codes_to_z(self, codes: torch.Tensor) -> torch.Tensor:
        return self.fsq.codes_to_z(codes)

    def z_to_codes(self, z: torch.Tensor) -> torch.Tensor:
        return self.fsq.z_to_codes(z)


# ── the command-vocabulary baseline ──────────────────────────────────────────

def command_classes(segment_rows: np.ndarray, segment_s: float) -> tuple[int, int, int]:
    """``(heading, vertical, speed)`` class indices of ONE segment's rows in the start frame."""
    rows = np.asarray(segment_rows, dtype=np.float64)
    last = rows[-1]
    heading_deg = math.degrees(math.atan2(last[4], last[3]))   # the first row's course is 0
    if abs(heading_deg) <= COMMAND_HEADING_SMALL_DEG:
        heading = 0
    elif abs(heading_deg) <= COMMAND_HEADING_LARGE_DEG:
        heading = 1 if heading_deg > 0.0 else 2
    else:
        heading = 3 if heading_deg > 0.0 else 4
    vertical_rate = last[2] / segment_s
    vertical = 0 if abs(vertical_rate) <= COMMAND_LEVEL_RATE_MPS else (1 if vertical_rate < 0.0 else 2)
    acceleration = (math.hypot(last[3], last[4]) - math.hypot(rows[0, 3], rows[0, 4])) / segment_s
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
    levels: tuple[int, ...] = ()

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


def _tokenizer_spec(tokenizer: nn.Module, segment_s: float, dt_s: float) -> dict[str, Any]:
    return {
        "kind": tokenizer.kind,
        "levels": list(tokenizer.levels),
        "rows": int(tokenizer.rows),
        "code_count": int(tokenizer.code_count),
        "z_dim": int(tokenizer.z_dim),
        "segment_s": float(segment_s),
        "dt_s": float(dt_s),
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
    spec = _tokenizer_spec(tokenizer, segment_s, dt_s)
    directory.mkdir(parents=True)
    weights = directory / CODEBOOK_WEIGHTS
    torch.save({
        "schema": CODEBOOK_SCHEMA,
        "tokenizer": spec,
        "state_dict": {key: value.detach().cpu() for key, value in tokenizer.state_dict().items()},
    }, weights)
    sha256 = file_sha256(weights)
    write_json_atomic(directory / CODEBOOK_MANIFEST, {
        "schema": CODEBOOK_SCHEMA,
        "written_utc": utc_now(),
        "tokenizer": spec,
        "weights_file": CODEBOOK_WEIGHTS,
        "sha256": sha256,
        "data_identity": data_identity,
        "source": source,
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
    if spec != manifest["tokenizer"]:
        raise ValueError(f"{directory}: the weights file's tokenizer spec differs from the manifest's")
    tokenizer = tokenizer_for(spec["kind"], levels=spec["levels"], segment_s=spec["segment_s"], dt_s=spec["dt_s"])
    tokenizer.load_state_dict(payload["state_dict"])
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return Codebook(
        tokenizer=tokenizer, kind=spec["kind"], levels=tuple(spec["levels"]), rows=int(spec["rows"]),
        segment_s=float(spec["segment_s"]), dt_s=float(spec["dt_s"]),
        data_identity=dict(manifest["data_identity"]), source=dict(manifest["source"]),
        sha256=sha256, path=directory,
    )


__all__ = [
    "CODEBOOK_MANIFEST", "CODEBOOK_SCHEMA", "CODEBOOK_WEIGHTS", "COMMAND_HEADING_CLASSES",
    "COMMAND_SPEED_CLASSES", "COMMAND_VERTICAL_CLASSES", "COMMAND_VOCABULARY_SIZE",
    "ENCODER_D_MODEL", "ENCODER_D_FF", "ENCODER_HEADS", "ENCODER_LAYERS", "FSQ",
    "FSQ_MINIMUM_LEVELS", "MANOEUVRE_TOKENIZERS", "MANOEUVRE_TOKENIZER_COMMAND_VOCABULARY",
    "MANOEUVRE_TOKENIZER_LEARNED", "SEGMENT_ROW_SCALE", "STATE_ROW_SCALE", "Codebook",
    "CommandVocabularyTokenizer", "LearnedTokenizer", "SegmentEncoder", "command_classes",
    "command_code", "command_label", "fsq_code_count", "load_codebook", "tokenizer_for",
    "write_codebook",
]
