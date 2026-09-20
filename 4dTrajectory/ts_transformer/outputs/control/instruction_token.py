"""The instruction token (two-tier v3 stage B, plan §5.2.1 "执行器怎么吃"; B′-dev2).

Under ``plan_conditioning='instruction'`` the executor is handed, beside the aircraft condition
and fused like the CTA, the words IN FORCE at the Δ / τ positions its segment spans — each
position as the vocabulary's bin-centre conditioning (`Vocabulary.conditioning`: cos / sin of
the heading, the altitude and speed as fractions of their ceilings, a graded intercept flag),
``[P, CONDITIONING_WIDTH]`` under `INSTRUCTION_KEY` in the dynamics rows.

ONE source for the three consumers (plan §5.2.1: 训练 / 预测 / 闭环三处一个来源):
`instruction_context` reads the words of ONE flight's `Reading` at a start time — a training
row and `predict` hand it the anchor's own time (the truth's words there); the closed loop
hands it the time of the truth row nearest the FLOWN position (`nearest_truth_time_s`: by
position, not by the flown clock — plan §10 item 3 — so a late aircraft is told what the
controller says where it is, not what was said at that second; the reading is of the WHOLE
record, never of a cohort cut at its first prediction row), or the prior's words (B3′).

The vocabulary is the hashed artefact the config names (`instruction_vocabulary`, the path to
`instruction_vocabulary.json`, read as given like the fitted teacher's); the executor's
checkpoint stores its sha and `load_checkpoint` refuses an artefact whose sha moved
(`ControlStrategy.verify_checkpoint_payload`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ts_transformer.config import PLAN_CONDITIONING_INSTRUCTION, TSConfig
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.manoeuvre.instructions import FT, KT, NO_INTERCEPT, Reading, Vocabulary, load_vocabulary, segment_positions_s

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

INSTRUCTION_KEY = "instructions"


def load_vocabulary_for(config: TSConfig) -> Vocabulary:
    """The artefact `config.instruction_vocabulary` names (refused unless the run conditions on
    instructions), checked against the executor's segment: Δ must be a whole number of τ."""
    if config.plan_conditioning != PLAN_CONDITIONING_INSTRUCTION:
        raise ValueError(f"plan_conditioning={config.plan_conditioning!r} reads no instruction vocabulary")
    path = Path(config.instruction_vocabulary)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: the instruction vocabulary this run names (instruction_vocabulary) is not there — an "
            "instruction executor reads its words through that artefact, at training, at predict and at load"
        )
    vocabulary, _payload = load_vocabulary(path)
    segment_positions_s(0.0, config.control_horizon_s, vocabulary.token_step_s)     # raises when Δ / τ is not whole
    return vocabulary


def instruction_positions(config: TSConfig, vocabulary: Vocabulary) -> int:
    """P — the positions one segment spans (Δ / τ)."""
    return len(segment_positions_s(0.0, config.control_horizon_s, vocabulary.token_step_s))


def instruction_token_width(config: TSConfig, vocabulary: Vocabulary) -> int:
    """The token's flattened width, the encoder's input size: P × the conditioning width."""
    return instruction_positions(config, vocabulary) * Vocabulary.CONDITIONING_WIDTH


def instruction_context(reading: Reading, start_s: float, config: TSConfig, vocabulary: Vocabulary) -> dict[str, np.ndarray]:
    """The token of a segment starting at ``start_s`` on the reading's clock: ``[P, W]`` float32."""
    words = reading.words_at(segment_positions_s(start_s, config.control_horizon_s, vocabulary.token_step_s))
    return {INSTRUCTION_KEY: vocabulary.conditioning(words)}


def nearest_truth_time_s(series: FlightSeries, flown_row: np.ndarray) -> tuple[float, float]:
    """``(time_s, distance_m)`` of the truth row horizontally nearest the flown chart row: where
    the flown aircraft IS on the truth's path, and how far off it. Every row of ``series`` is a
    candidate (the closed loop hands the cohort as it flies it, from its first prediction row)."""
    columns = list(POSITION_IDX[:2])
    rows = np.asarray(series.values, dtype=np.float64)[:, columns]
    flown = np.asarray(flown_row, dtype=np.float64)[columns]
    distance = np.hypot(rows[:, 0] - flown[0], rows[:, 1] - flown[1])
    nearest = int(np.argmin(distance))
    return float(series.times[nearest]), float(distance[nearest])


def probe_instruction_context(config: TSConfig, vocabulary: Vocabulary, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """The batch-size probe's token: every position "on the course, 3 000 ft, 180 kt, no intercept",
    in the vocabulary's own bins."""
    one = [vocabulary.heading_bin(0.0), vocabulary.altitude_bin(3000.0 * FT)[0], vocabulary.speed_bin(180.0 * KT)[0], NO_INTERCEPT]
    words = np.tile(np.array([one], dtype=np.int64), (instruction_positions(config, vocabulary), 1))
    token = torch.from_numpy(vocabulary.conditioning(words)).to(device)
    return {INSTRUCTION_KEY: token.unsqueeze(0).expand(batch_size, -1, -1).contiguous()}


__all__ = [
    "INSTRUCTION_KEY", "instruction_context", "instruction_positions", "instruction_token_width", "load_vocabulary_for",
    "nearest_truth_time_s", "probe_instruction_context",
]
