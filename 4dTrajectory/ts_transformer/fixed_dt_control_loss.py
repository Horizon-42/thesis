"""Fixed-physical-time state supervision for deterministic control prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aerodynamic_model.torch_dense_rollout import rollout_piecewise_constant_at_times
from aerodynamic_model.torch_dynamics import geodetic_states_to_channels
from config import TSConfig
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction


@dataclass(frozen=True)
class FixedDTStateLossResult:
    per_flight_loss: torch.Tensor
    normalized_segment_end_states: torch.Tensor
    physical_query_states: torch.Tensor


def fixed_dt_rollout_channels(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    segment_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query and segment-end channels from one event-aligned RK4 rollout."""
    rollout_dtype = torch.float64
    rollout = rollout_piecewise_constant_at_times(
        dynamics["initial_state"].to(rollout_dtype),
        prediction.controls.to(rollout_dtype),
        prediction.segment_durations.to(rollout_dtype),
        dynamics["aero_params"].to(rollout_dtype),
        supervision.query_offsets_s.to(rollout_dtype),
        supervision.valid,
        segment_valid=segment_valid,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    frame = dynamics["frame_params"].to(rollout_dtype)
    runway_aligned = config.coordinate_frame == "runway-aligned"
    query_channels = geodetic_states_to_channels(
        rollout.query_states,
        frame,
        runway_aligned=runway_aligned,
    )
    endpoint_channels = geodetic_states_to_channels(
        rollout.segment_end_states,
        frame,
        runway_aligned=runway_aligned,
    )
    return query_channels, endpoint_channels


def fixed_dt_control_state_loss(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    segment_valid: torch.Tensor | None = None,
) -> FixedDTStateLossResult:
    """Average each flight over its complete regular-dt reference prefix."""
    query_channels, endpoint_channels = fixed_dt_rollout_channels(
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
    )
