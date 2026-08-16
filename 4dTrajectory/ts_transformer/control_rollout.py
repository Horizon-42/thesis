"""One control/dynamics rollout API shared by training, evaluation and forecasting.

The selected backend owns the state representation and RK4 implementation.  This module
owns the public tensor contract and the mandatory float64 dynamics boundary so callers do
not reproduce dtype/device conversions or reach into backend implementations directly.
"""

from __future__ import annotations

import torch

from config import TSConfig
from control_dynamics_backends import (
    DenseControlRolloutChannels,
    EndpointControlRollout,
    control_dynamics_backend,
)


ROLLOUT_DTYPE = torch.float64


def rollout_control_endpoints(
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
) -> EndpointControlRollout:
    """Roll a batch to every control boundary using the configured dynamics."""
    return control_dynamics_backend(config).endpoint_rollout(
        dynamics["initial_state"].to(ROLLOUT_DTYPE),
        controls.to(ROLLOUT_DTYPE),
        segment_durations_s.to(ROLLOUT_DTYPE),
        dynamics["aero_params"].to(ROLLOUT_DTYPE),
        dynamics["frame_params"].to(ROLLOUT_DTYPE),
        config,
    )


def rollout_control_dense(
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    config: TSConfig,
    *,
    segment_valid: torch.Tensor | None = None,
) -> DenseControlRolloutChannels:
    """Roll a batch once and return exact states at queries and control boundaries."""
    device = controls.device
    return control_dynamics_backend(config).dense_rollout(
        dynamics["initial_state"].to(ROLLOUT_DTYPE),
        controls.to(ROLLOUT_DTYPE),
        segment_durations_s.to(ROLLOUT_DTYPE),
        dynamics["aero_params"].to(ROLLOUT_DTYPE),
        dynamics["frame_params"].to(ROLLOUT_DTYPE),
        query_offsets_s.to(dtype=ROLLOUT_DTYPE, device=device),
        query_valid.to(device=device, dtype=torch.bool),
        config,
        segment_valid=(
            None
            if segment_valid is None
            else segment_valid.to(device=device, dtype=torch.bool)
        ),
    )
