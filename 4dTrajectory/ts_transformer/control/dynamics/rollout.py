"""One control/dynamics rollout API shared by training, evaluation and forecasting.

The selected model/representation pair owns the state and the RK4 implementation. This
module owns the public tensor contract and the mandatory float64 dynamics boundary, so
callers never reproduce dtype/device conversions or reach into a backend directly.
"""

from __future__ import annotations

import torch

from config import TSConfig
from control.dynamics.hooks import CommandHook
from control.dynamics.backends import (
    DenseControlRolloutChannels,
    EndpointControlRollout,
    RolloutInputs,
    control_dynamics_backend,
)


ROLLOUT_DTYPE = torch.float64


def rollout_inputs(
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
) -> RolloutInputs:
    """Assemble the float64 dynamics boundary from a batch's dynamics dict."""
    device = controls.device

    def cast(value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=ROLLOUT_DTYPE, device=device)

    return RolloutInputs(
        initial_state=cast(dynamics["initial_state"]),
        initial_controls=cast(dynamics["initial_controls"]),
        controls=controls.to(ROLLOUT_DTYPE),
        segment_durations_s=segment_durations_s.to(ROLLOUT_DTYPE),
        aero_params=cast(dynamics["aero_params"]),
        frame_params=cast(dynamics["frame_params"]),
        max_thrust_n=cast(dynamics["max_thrust_n"]),
    )


def rollout_control_endpoints(
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    *,
    command_hook: CommandHook | None = None,
) -> EndpointControlRollout:
    """Roll a batch to every control boundary using the configured dynamics.

    ``command_hook`` (``control.dynamics.hooks``) may rewrite each segment's command from
    the state at its start; the result's ``controls`` is then the schedule flown.
    """
    return control_dynamics_backend(config).endpoint_rollout(
        rollout_inputs(controls, segment_durations_s, dynamics),
        config,
        command_hook=command_hook,
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
    command_hook: CommandHook | None = None,
) -> DenseControlRolloutChannels:
    """Roll a batch once and return exact states at queries and control boundaries."""
    device = controls.device
    return control_dynamics_backend(config).dense_rollout(
        rollout_inputs(controls, segment_durations_s, dynamics),
        query_offsets_s.to(dtype=ROLLOUT_DTYPE, device=device),
        query_valid.to(device=device, dtype=torch.bool),
        config,
        segment_valid=(
            None
            if segment_valid is None
            else segment_valid.to(device=device, dtype=torch.bool)
        ),
        command_hook=command_hook,
    )
