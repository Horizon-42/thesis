"""Fixed-physical-time state supervision for deterministic control prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ts_transformer.config import TSConfig
from ts_transformer.control.dynamics import rollout as control_rollout
from ts_transformer.dataset import Normalizer
from ts_transformer.fixed_dt_supervision import FixedDTControlSupervision
from ts_transformer.prediction_outputs import ControlPrediction


@dataclass(frozen=True)
class FixedDTStateLossResult:
    per_flight_loss: torch.Tensor
    normalized_segment_end_states: torch.Tensor
    physical_query_states: torch.Tensor
    physical_segment_durations_s: torch.Tensor


def close_duration_prefix(
    durations: torch.Tensor,
    total_duration: torch.Tensor,
) -> torch.Tensor:
    """Close the final segment on the caller's exact physical-time clock.

    The learned durations are a normalized partition scaled by a learned total, so
    accumulating them in the model's float32 leaves the reconstructed total a few
    microseconds off the float64 reference clock the fixed-dt queries live on — enough to
    push a query that lands exactly on the horizon past the end of the rollout. Recomputing
    the last duration from the preceding float64 boundary makes the sum exact instead.
    """
    corrected_last = total_duration - durations[:, :-1].sum(dim=1)
    closed = torch.cat((durations[:, :-1], corrected_last.unsqueeze(1)), dim=1)
    if not torch.allclose(closed.sum(dim=1), total_duration, rtol=1e-6, atol=1e-6):
        raise RuntimeError("control durations do not sum to the physical-time horizon")
    return closed


def fixed_dt_rollout_channels(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query/end states and their exact event-aligned segment clock."""
    durations = close_duration_prefix(
        prediction.segment_durations.to(control_rollout.ROLLOUT_DTYPE),
        prediction.final_time_s.to(
            dtype=control_rollout.ROLLOUT_DTYPE,
            device=prediction.segment_durations.device,
        ),
    )
    rollout = control_rollout.rollout_control_dense(
        prediction.controls,
        durations,
        dynamics,
        supervision.query_offsets_s,
        supervision.valid,
        config,
    )
    return rollout.query_channels, rollout.segment_end_channels, durations


def fixed_dt_control_state_loss(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
) -> FixedDTStateLossResult:
    """Average each flight over its complete regular-dt reference prefix."""
    query_channels, endpoint_channels, segment_durations = fixed_dt_rollout_channels(
        prediction, supervision, dynamics, config
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
