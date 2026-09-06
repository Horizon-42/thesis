"""Composable tracking-loss components for deterministic control prediction.

This module contains no training loop and no optimizer policy.  It turns one completed
dynamics rollout into named per-flight loss terms, then composes those terms through an
explicit objective registry.  Adding a new tracking recipe therefore does not add another
branch to ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from channels import POSITION_IDX, VELOCITY_IDX
from config import (
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    TSConfig,
)
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from physical_criteria import terminal_position_error_m
from terminal_state_loss import last_reliable_terminal_velocity_target


@dataclass(frozen=True)
class ControlStateLossResult:
    """Backend-neutral state rollout products consumed by tracking objectives."""

    normalized_mse: torch.Tensor
    normalized_segment_end_states: torch.Tensor
    physical_query_states: torch.Tensor | None = None
    physical_position_mse: torch.Tensor | None = None
    physical_velocity_mse: torch.Tensor | None = None
    control_imitation_mse: torch.Tensor | None = None
    # The normalized targets and weights aligned to ``normalized_segment_end_states`` rows
    # (the native grid fills them); the procedure penalty gates on these truth rows.
    aligned_targets: torch.Tensor | None = None
    aligned_weights: torch.Tensor | None = None
    # The command hook's counts over this batch's rollout (empty without a hook).
    hook_diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlTrackingLossTerms:
    """Named, already weighted per-flight tracking contributions."""

    state: torch.Tensor
    terminal_position: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


def terminal_velocity_error_mps(
    normalized_segment_end_states: torch.Tensor,
    normalized_anchor_state: torch.Tensor,
    supervision: FixedDTControlSupervision,
    normalizer: Normalizer,
) -> torch.Tensor:
    """3-D terminal chart-velocity error against the last reliable observation."""

    indices = list(VELOCITY_IDX)
    endpoint = normalized_segment_end_states[:, -1, indices]
    target = last_reliable_terminal_velocity_target(
        normalized_anchor_state.to(dtype=endpoint.dtype, device=endpoint.device),
        supervision,
    ).to(dtype=endpoint.dtype, device=endpoint.device)
    scale = torch.as_tensor(
        normalizer.std[indices], dtype=endpoint.dtype, device=endpoint.device
    )
    return torch.linalg.vector_norm((endpoint - target) * scale, dim=-1)


def normalized_terminal_position_mse(
    normalized_segment_end_states: torch.Tensor,
    normalized_terminal_targets: torch.Tensor,
) -> torch.Tensor:
    """Historical normalized terminal-position MSE, per flight."""

    indices = list(POSITION_IDX)
    delta = (
        normalized_segment_end_states[:, -1, indices]
        - normalized_terminal_targets[:, indices].to(
            dtype=normalized_segment_end_states.dtype,
            device=normalized_segment_end_states.device,
        )
    )
    return delta.square().mean(dim=1)


TrackingObjective = Callable[
    [
        ControlStateLossResult,
        torch.Tensor,
        torch.Tensor,
        TSConfig,
        Normalizer,
        FixedDTControlSupervision | None,
    ],
    ControlTrackingLossTerms,
]


def _normalized_mse_objective(
    result: ControlStateLossResult,
    normalized_anchor_state: torch.Tensor,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dense_supervision: FixedDTControlSupervision | None,
) -> ControlTrackingLossTerms:
    del normalized_anchor_state, normalizer, dense_supervision
    terminal = config.terminal_loss_weight * normalized_terminal_position_mse(
        result.normalized_segment_end_states, terminal_target
    )
    return ControlTrackingLossTerms(result.normalized_mse, terminal)


def _true_time_position_objective(
    result: ControlStateLossResult,
    normalized_anchor_state: torch.Tensor,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dense_supervision: FixedDTControlSupervision | None,
) -> ControlTrackingLossTerms:
    """Minimal physical 3-D path plus soft observed endpoint objective."""

    del normalized_anchor_state, dense_supervision
    if result.physical_position_mse is None:
        raise ValueError(
            "true-time-position requires native uniform-clock physical position loss"
        )
    terminal_position_m = terminal_position_error_m(
        result.normalized_segment_end_states, terminal_target, normalizer
    )
    endpoint_mse = (
        terminal_position_m / config.position_loss_scale_m
    ).square()
    extras: dict[str, torch.Tensor] = {}
    if config.control_velocity_loss_weight:
        if result.physical_velocity_mse is None:
            raise ValueError(
                "the velocity term needs the native uniform-clock physical velocity loss"
            )
        extras["velocity"] = (
            config.control_velocity_loss_weight * result.physical_velocity_mse
        )
    if config.control_imitation_loss_weight:
        if result.control_imitation_mse is None:
            raise ValueError(
                "the imitation term needs the native uniform-clock control inversion"
            )
        extras["imitation"] = (
            config.control_imitation_loss_weight * result.control_imitation_mse
        )
    return ControlTrackingLossTerms(
        state=result.physical_position_mse,
        terminal_position=config.state_endpoint_loss_weight * endpoint_mse,
        extras=extras,
    )


_TRACKING_OBJECTIVES: dict[str, TrackingObjective] = {
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE: _normalized_mse_objective,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION: _true_time_position_objective,
}


def control_tracking_loss_terms(
    result: ControlStateLossResult,
    normalized_anchor_state: torch.Tensor,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dense_supervision: FixedDTControlSupervision | None,
) -> ControlTrackingLossTerms:
    """Compose the configured tracking recipe from independently testable terms."""

    return _TRACKING_OBJECTIVES[config.control_state_objective](
        result,
        normalized_anchor_state,
        terminal_target,
        config,
        normalizer,
        dense_supervision,
    )
