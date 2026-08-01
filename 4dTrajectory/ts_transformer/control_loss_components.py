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

from arc_length_geometry import arc_length_state_loss_terms
from channels import POSITION_IDX, VELOCITY_IDX
from config import (
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_PHYSICAL_CRITERIA,
    CONTROL_STATE_OBJECTIVE_TERMINAL_STATE,
    TSConfig,
)
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from physical_criteria import (
    fixed_dt_position_ade_m,
    physical_criteria_loss,
    terminal_position_error_m,
)


@dataclass(frozen=True)
class ControlStateLossResult:
    """Backend-neutral state rollout products consumed by tracking objectives."""

    normalized_mse: torch.Tensor
    normalized_segment_end_states: torch.Tensor
    physical_query_states: torch.Tensor | None = None


@dataclass(frozen=True)
class ControlTrackingLossTerms:
    """Named, already weighted per-flight tracking contributions."""

    state: torch.Tensor
    terminal_position: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


def last_reliable_terminal_velocity_target(
    normalized_anchor_state: torch.Tensor,
    supervision: FixedDTControlSupervision,
) -> torch.Tensor:
    """Return the last measured velocity target, never a fitted-tail placeholder.

    Fitted approach rows keep velocity-shaped placeholders but give them zero weights.  A
    flight whose fixed-dt future contains no measured velocity row explicitly falls back to
    its observed anchor velocity, which is the last reliable measurement available to the
    deployable model.
    """

    indices = list(VELOCITY_IDX)
    weights = supervision.weights[..., indices].to(
        device=normalized_anchor_state.device
    )
    row_valid = supervision.valid.to(device=weights.device) & torch.all(
        weights > 0.0, dim=-1
    )
    row_numbers = torch.arange(
        row_valid.shape[1], device=row_valid.device, dtype=torch.long
    ).unsqueeze(0).expand_as(row_valid)
    last = torch.where(row_valid, row_numbers, -torch.ones_like(row_numbers)).amax(dim=1)
    safe_last = last.clamp(min=0)
    states = supervision.states.to(
        dtype=normalized_anchor_state.dtype,
        device=normalized_anchor_state.device,
    )
    rows = torch.arange(len(states), device=states.device)
    future_target = states[rows, safe_last][:, indices]
    anchor_target = normalized_anchor_state[:, indices]
    return torch.where((last >= 0).unsqueeze(1), future_target, anchor_target)


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


def _physical_criteria_objective(
    result: ControlStateLossResult,
    normalized_anchor_state: torch.Tensor,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dense_supervision: FixedDTControlSupervision | None,
) -> ControlTrackingLossTerms:
    del normalized_anchor_state, config
    if result.physical_query_states is None or dense_supervision is None:
        raise ValueError("physical-criteria requires fixed-dt control supervision")
    ade_m = fixed_dt_position_ade_m(
        result.physical_query_states, dense_supervision, normalizer
    )
    terminal_m = terminal_position_error_m(
        result.normalized_segment_end_states, terminal_target, normalizer
    )
    zero = terminal_m.new_zeros(terminal_m.shape)
    return ControlTrackingLossTerms(
        physical_criteria_loss(ade_m, terminal_m), zero
    )


def _terminal_state_objective(
    result: ControlStateLossResult,
    normalized_anchor_state: torch.Tensor,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dense_supervision: FixedDTControlSupervision | None,
) -> ControlTrackingLossTerms:
    if dense_supervision is None:
        raise ValueError("terminal-state requires fixed-dt control supervision")
    terminal_position_m = terminal_position_error_m(
        result.normalized_segment_end_states, terminal_target, normalizer
    )
    terminal_velocity_mps = terminal_velocity_error_mps(
        result.normalized_segment_end_states,
        normalized_anchor_state,
        dense_supervision,
        normalizer,
    )
    return ControlTrackingLossTerms(
        state=config.control_dense_state_loss_weight * result.normalized_mse,
        terminal_position=(
            config.control_terminal_position_loss_weight
            * terminal_position_m
            / config.control_terminal_position_scale_m
        ),
        extras={
            "terminal_velocity": (
                config.control_terminal_velocity_loss_weight
                * terminal_velocity_mps
                / config.control_terminal_velocity_scale_mps
            )
        },
    )


def _arc_length_geometry_objective(
    result: ControlStateLossResult,
    normalized_anchor_state: torch.Tensor,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dense_supervision: FixedDTControlSupervision | None,
) -> ControlTrackingLossTerms:
    if dense_supervision is None:
        raise ValueError("arc-length-geometry requires fixed-dt control supervision")
    arc = arc_length_state_loss_terms(
        normalized_anchor_state,
        result.normalized_segment_end_states,
        terminal_target,
        dense_supervision,
        normalizer,
        points=config.n_segments,
    )
    terminal_position_m = terminal_position_error_m(
        result.normalized_segment_end_states, terminal_target, normalizer
    )
    terminal_velocity_mps = terminal_velocity_error_mps(
        result.normalized_segment_end_states,
        normalized_anchor_state,
        dense_supervision,
        normalizer,
    )
    return ControlTrackingLossTerms(
        state=config.control_geometry_loss_weight * arc.position,
        terminal_position=(
            config.control_terminal_position_loss_weight
            * terminal_position_m
            / config.control_terminal_position_scale_m
        ),
        extras={
            "terminal_velocity": (
                config.control_terminal_velocity_loss_weight
                * terminal_velocity_mps
                / config.control_terminal_velocity_scale_mps
            ),
            "arc_horizontal_velocity": (
                config.control_arc_horizontal_velocity_loss_weight
                * arc.horizontal_velocity_mps
                / config.control_arc_horizontal_velocity_scale_mps
            ),
            "arc_vertical_velocity": (
                config.control_arc_vertical_velocity_loss_weight
                * arc.vertical_velocity_mps
                / config.control_arc_vertical_velocity_scale_mps
            ),
        },
    )


_TRACKING_OBJECTIVES: dict[str, TrackingObjective] = {
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE: _normalized_mse_objective,
    CONTROL_STATE_OBJECTIVE_PHYSICAL_CRITERIA: _physical_criteria_objective,
    CONTROL_STATE_OBJECTIVE_TERMINAL_STATE: _terminal_state_objective,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY: _arc_length_geometry_objective,
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
