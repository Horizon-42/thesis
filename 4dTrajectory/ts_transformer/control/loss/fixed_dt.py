"""Fixed-physical-time state supervision for deterministic control prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config import TSConfig
from control.dynamics import rollout as control_rollout
from control.training.curriculum import close_duration_prefix
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction


@dataclass(frozen=True)
class FixedDTStateLossResult:
    per_flight_loss: torch.Tensor
    normalized_segment_end_states: torch.Tensor
    physical_query_states: torch.Tensor
    physical_segment_durations_s: torch.Tensor


def fixed_dt_rollout_channels(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    segment_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query/end states and their exact event-aligned segment clock."""
    durations = prediction.segment_durations.to(control_rollout.ROLLOUT_DTYPE)
    active_segments = (
        torch.ones_like(durations, dtype=torch.bool)
        if segment_valid is None
        else segment_valid.to(device=durations.device)
    )
    durations = close_duration_prefix(
        durations,
        active_segments,
        prediction.final_time_s.to(
            dtype=control_rollout.ROLLOUT_DTYPE, device=durations.device
        ),
    )
    rollout = control_rollout.rollout_control_dense(
        prediction.controls,
        durations,
        dynamics,
        supervision.query_offsets_s,
        supervision.valid,
        config,
        segment_valid=segment_valid,
    )
    return rollout.query_channels, rollout.segment_end_channels, durations


def fixed_dt_control_state_loss(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    segment_valid: torch.Tensor | None = None,
) -> FixedDTStateLossResult:
    """Average each flight over its complete regular-dt reference prefix."""
    query_channels, endpoint_channels, segment_durations = fixed_dt_rollout_channels(
        prediction, supervision, dynamics, config, segment_valid
    )
    dtype, device = query_channels.dtype, query_channels.device
    mean = torch.as_tensor(normalizer.mean, dtype=dtype, device=device)
    scale = torch.as_tensor(normalizer.std, dtype=dtype, device=device)
    normalized_queries = (query_channels - mean) / scale
    normalized_endpoints = (endpoint_channels - mean) / scale
    targets = supervision.states.to(dtype=dtype, device=device)
    weights = supervision.weights.to(dtype=dtype, device=device)
    weights = weights * supervision.valid.to(device=device).unsqueeze(-1)
    squared = (normalized_queries - targets).square() * weights
    denominator = weights.sum(dim=(1, 2)).clamp(min=1.0)
    per_flight = squared.sum(dim=(1, 2)) / denominator
    return FixedDTStateLossResult(
        per_flight_loss=per_flight,
        normalized_segment_end_states=normalized_endpoints,
        physical_query_states=query_channels,
        physical_segment_durations_s=segment_durations,
    )
