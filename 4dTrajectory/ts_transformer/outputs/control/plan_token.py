"""The plan token: what the control decoder is told about the segment it flies.

Under ``plan_conditioning = manoeuvre-code`` (plan §2.6) the token is the segment's code
vector ``z`` from the tokenizer (`manoeuvre/tokenizer.py`), which the executor holds as a
submodule (`heads.ControlFeatureModel.manoeuvre_tokenizer`). The dynamics context carries
the tokenizer's INPUTS, and ``z`` has exactly two sources, one per key set:

* ``manoeuvre_segment`` ``[B, R, 6]`` + ``manoeuvre_state`` ``[B, 6]`` — the TRUTH's token
  span (`config.token_span_s` seconds, the horizon unless ``manoeuvre_token_s`` says
  otherwise) in its start frame and the span's start row (`training_plan_context`): training,
  `predict` and protocol C. Reads the future, by design (the tokenizer's job). Under a held
  token (S > the step, two-tier v3 D38) the span a training row reads started ``phase`` steps
  BEFORE the anchor — φ drawn per flight and epoch (`token_phase`), so the executor trains at
  every position inside a span the closed loop holds one token over — and a span the record
  cannot hold is the PREVIOUS span's, as the closed loop holds the last full span's code
  (B-dev2, plan D48: the anchor floor stays the horizon's, not S).
* ``manoeuvre_z`` ``[B, Z]`` — a z handed in: the prior's top-1 in protocol A, or any
  counterfactual code (`plan_z` maps it back to its code).

A batch carries one source or the other, never both. The two-tier v2 tokens are archived
(`archive/two_tier_v2_2026_09/plan_token_v2.py`); under ``off`` nothing is built.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from ts_transformer.config import (
    MANOEUVRE_TOKENIZER_LEARNED,
    PLAN_CONDITIONING_OFF,
    TSConfig,
    token_hold,
    token_span_s,
    token_step_s,
)
from ts_transformer.data.channels import CHANNELS, VELOCITY_IDX
from ts_transformer.manoeuvre.segments import STEP_TOLERANCE_S, segment_row_count, state_row, truth_segment_rows
from ts_transformer.manoeuvre.tokenizer import (
    COMMAND_VOCABULARY_SIZE,
    fsq_code_count,
    load_codebook,
    tokenizer_for,
)

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The dynamics-context keys (beside `condition` and `cta_s`).
MANOEUVRE_SEGMENT_KEY = "manoeuvre_segment"
MANOEUVRE_STATE_KEY = "manoeuvre_state"
MANOEUVRE_Z_KEY = "manoeuvre_z"
#: The forward speed the batch-size probe's empty segment flies at (m/s) — any approach
#: speed; the probe measures memory, not a manoeuvre.
PROBE_SPEED_MPS = 70.0
#: The salt of the per-flight per-epoch phase draw (the anchor draw's pattern,
#: `dataset.RemainingPathUniformAnchorTrajectoryWindows._epoch_draws`): the law's name, so a
#: changed law cannot reproduce an old run's draws under the old name.
TOKEN_PHASE_SAMPLING_VERSION = "per-flight-hash-v1-token-phase"


def plan_token_width(config: TSConfig) -> int:
    """The token's width under this run's ``plan_conditioning`` (the decoder's input size):
    the learned tokenizer's FSQ dimension count, or the command vocabulary's one-hot."""
    if config.plan_conditioning == PLAN_CONDITIONING_OFF:
        return 0
    if config.manoeuvre_tokenizer == MANOEUVRE_TOKENIZER_LEARNED:
        return len(config.manoeuvre_fsq_levels)
    return COMMAND_VOCABULARY_SIZE


def manoeuvre_code_count(config: TSConfig) -> int:
    """K — the vocabulary size under this run's tokenizer; 0 without a plan token."""
    if config.plan_conditioning == PLAN_CONDITIONING_OFF:
        return 0
    if config.manoeuvre_tokenizer == MANOEUVRE_TOKENIZER_LEARNED:
        return fsq_code_count(config.manoeuvre_fsq_levels)
    return COMMAND_VOCABULARY_SIZE


def manoeuvre_tokenizer_for(config: TSConfig) -> tuple[nn.Module, str | None]:
    """The executor's tokenizer and the codebook sha it is bound to: built fresh to train
    JOINTLY (``manoeuvre_codebook`` empty, sha None), or the frozen codebook's — whose kind,
    levels and segment (the token span) must be the config's, or the run would name one
    vocabulary and fly another."""
    if not config.manoeuvre_codebook:
        tokenizer = tokenizer_for(
            config.manoeuvre_tokenizer, levels=config.manoeuvre_fsq_levels,
            segment_s=token_span_s(config), dt_s=config.dt_s,
        )
        return tokenizer, None
    try:
        codebook = load_codebook(config.manoeuvre_codebook)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"this executor was trained against the codebook at {config.manoeuvre_codebook}, which "
            f"is not there ({exc}); the checkpoint binds to it (codebook_sha256 in its "
            "checkpoint_metadata.json) and cannot be rebuilt without it"
        ) from exc
    expected = (
        config.manoeuvre_tokenizer, tuple(config.manoeuvre_fsq_levels),
        token_span_s(config), float(config.dt_s),
    )
    found = (codebook.kind, codebook.levels, codebook.segment_s, codebook.dt_s)
    if expected != found:
        raise ValueError(
            f"the codebook at {config.manoeuvre_codebook} is (tokenizer, levels, segment_s, dt_s) = "
            f"{found}, the config says {expected}"
        )
    return codebook.tokenizer, codebook.sha256


def token_phase(dataset_id: str, anchor: int, epoch_seed: int, config: TSConfig) -> int:
    """The anchor's position inside its token span for one training draw: φ ∈ {0, …, hold−1},
    uniform, one digest per (flight, anchor, epoch) — 0 whenever a token is not held (hold 1:
    every 2026-09-18 arm, S20), so those runs draw nothing and change nothing."""
    hold = token_hold(config)
    if hold == 1:
        return 0
    digest = hashlib.sha256(f"{TOKEN_PHASE_SAMPLING_VERSION}:{int(epoch_seed)}:{dataset_id}:{int(anchor)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % hold


def token_span_start_s(series: FlightSeries, anchor: int, config: TSConfig, phase: int) -> tuple[float, bool]:
    """``(start time, held)`` of the token span a row at ``anchor`` reads: the span that started
    ``phase`` steps before the anchor, or — when the truth ends before that span does — the
    previous span (``held``), which is what the closed loop flies past the truth's last full
    span (`lockstep.fly`, protocol C). A cohort admitting an anchor whose previous span precedes
    the record is refused by name: its floor is the lookback plus the span, not the horizon."""
    span, step = token_span_s(config), token_step_s(config)
    if not 0 <= phase < token_hold(config):
        raise ValueError(f"phase is in [0, {token_hold(config)}), got {phase!r}")
    start = float(series.times[anchor]) - phase * step
    held = start + span > float(series.supervision_times[-1]) + STEP_TOLERANCE_S
    if held:
        start -= span
        if start < float(series.supervision_times[0]) - STEP_TOLERANCE_S:
            raise ValueError(
                f"{series.dataset_id}: anchor {anchor} at phase {phase} holds the previous {span:g} s span, which starts "
                f"{float(series.supervision_times[0]) - start:.1f} s before the record: the cohort's floor is the lookback "
                f"plus the span, not the horizon"
            )
    return start, held


def training_plan_context(series: FlightSeries, anchor: int, config: TSConfig, *, phase: int = 0) -> dict[str, np.ndarray]:
    """The tokenizer's inputs at an observed ``anchor``: the TRUTH's token span (`token_span_s`
    long, in its start frame) and the span's start row — what a training row, `predict` and
    protocol C hand the decoder. Reads the future. ``phase`` (B-dev2) is the anchor's position
    inside the span; at 0 the span starts at the anchor and the start row IS the anchor's."""
    start, held = token_span_start_s(series, anchor, config, phase)
    return {
        MANOEUVRE_SEGMENT_KEY: truth_segment_rows(series, start, token_span_s(config), config.dt_s).astype(np.float32),
        MANOEUVRE_STATE_KEY: (
            np.asarray(series.values[anchor], dtype=np.float32) if phase == 0 and not held
            else state_row(series, start).astype(np.float32)
        ),
    }


def probe_plan_context(config: TSConfig, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """The batch-size probe's inputs: a segment and a state holding nothing but a forward
    speed, of this run's shape — built by the same row count the training rows use."""
    rows = segment_row_count(token_span_s(config), config.dt_s)
    segment = torch.zeros(batch_size, rows, len(CHANNELS), dtype=torch.float32, device=device)
    segment[:, :, VELOCITY_IDX[0]] = PROBE_SPEED_MPS
    state = torch.zeros(batch_size, len(CHANNELS), dtype=torch.float32, device=device)
    state[:, VELOCITY_IDX[0]] = PROBE_SPEED_MPS
    return {MANOEUVRE_SEGMENT_KEY: segment, MANOEUVRE_STATE_KEY: state}


def plan_z(tokenizer: nn.Module, dynamics: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(z [B, Z], code [B])`` for the batch, from its ONE source: a given z, or the truth
    segment and state through the tokenizer."""
    given = MANOEUVRE_Z_KEY in dynamics
    encoded = MANOEUVRE_SEGMENT_KEY in dynamics
    if given == encoded:
        raise ValueError(
            f"a batch carries exactly one z source: {MANOEUVRE_Z_KEY!r}, or "
            f"{MANOEUVRE_SEGMENT_KEY!r} + {MANOEUVRE_STATE_KEY!r}"
        )
    if given:
        z = dynamics[MANOEUVRE_Z_KEY]
        if z.ndim != 2 or z.shape[1] != tokenizer.z_dim:
            raise ValueError(f"a given z is [B, {tokenizer.z_dim}] for this tokenizer, got {tuple(z.shape)}")
        return z, tokenizer.z_to_codes(z)
    return tokenizer(dynamics[MANOEUVRE_SEGMENT_KEY], dynamics[MANOEUVRE_STATE_KEY])


__all__ = [
    "MANOEUVRE_SEGMENT_KEY", "MANOEUVRE_STATE_KEY", "MANOEUVRE_Z_KEY", "PROBE_SPEED_MPS", "TOKEN_PHASE_SAMPLING_VERSION",
    "manoeuvre_code_count", "manoeuvre_tokenizer_for", "plan_token_width", "plan_z",
    "probe_plan_context", "token_phase", "token_span_start_s", "training_plan_context",
]
