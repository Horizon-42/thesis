"""One control/dynamics rollout API shared by training, evaluation and forecasting.

The selected model/representation pair owns the state and the RK4 implementation. This
module owns the public tensor contract and the mandatory float64 dynamics boundary, so
callers never reproduce dtype/device conversions or reach into a backend directly.
"""

from __future__ import annotations

import numpy as np
import torch

from ts_transformer.config import TSConfig
from ts_transformer.outputs.dynamics.hooks import CommandHook
from ts_transformer.outputs.dynamics.backends import (
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
    command_hook: CommandHook | None = None,
) -> DenseControlRolloutChannels:
    """Roll a batch once and return exact states at queries and control boundaries."""
    device = controls.device
    return control_dynamics_backend(config).dense_rollout(
        rollout_inputs(controls, segment_durations_s, dynamics),
        query_offsets_s.to(dtype=ROLLOUT_DTYPE, device=device),
        query_valid.to(device=device, dtype=torch.bool),
        config,
        command_hook=command_hook,
    )


def dense_query_offsets(
    segment_durations_s: np.ndarray, output_dt_s: float
) -> np.ndarray:
    """Return regular output times plus every exact control-switch boundary."""
    durations = np.asarray(segment_durations_s, dtype=np.float64)
    if durations.ndim != 1 or not len(durations):
        raise ValueError("control forecast needs at least one segment duration")
    if not np.isfinite(durations).all() or np.any(durations <= 0.0):
        raise ValueError("control segment durations must be finite and positive")
    if not np.isfinite(output_dt_s) or output_dt_s <= 0.0:
        raise ValueError("dense control output interval must be finite and positive")
    boundaries = np.cumsum(durations)
    total = float(boundaries[-1])
    regular = np.arange(output_dt_s, total, output_dt_s, dtype=np.float64)
    candidates = np.sort(np.concatenate((regular, boundaries)))
    tolerance = np.finfo(np.float64).eps * max(total, 1.0) * 16.0
    keep = np.concatenate(([True], np.diff(candidates) > tolerance))
    offsets = candidates[keep]
    offsets[-1] = total
    return offsets


def padded_dense_queries(
    segment_durations_s: np.ndarray,
    output_dt_s: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    offsets = [
        dense_query_offsets(row, output_dt_s)
        for row in segment_durations_s
    ]
    width = max(len(row) for row in offsets)
    padded = np.zeros((len(offsets), width), dtype=np.float64)
    valid = np.zeros((len(offsets), width), dtype=bool)
    for row, values in enumerate(offsets):
        padded[row, : len(values)] = values
        valid[row, : len(values)] = True
    return offsets, padded, valid
