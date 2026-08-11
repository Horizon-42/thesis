"""Runway-aware terminal-state errors shared by training and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from channels import POSITION_IDX, VELOCITY_IDX
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision


@dataclass(frozen=True)
class TerminalStateErrors:
    """Per-flight physical terminal errors before recipe weights and scales."""

    position_vector_m: torch.Tensor
    velocity_vector_mps: torch.Tensor
    position_runway_components_m: torch.Tensor
    velocity_runway_components_mps: torch.Tensor
    along_position_abs_m: torch.Tensor
    cross_position_abs_m: torch.Tensor
    vertical_position_abs_m: torch.Tensor
    along_velocity_abs_mps: torch.Tensor
    cross_velocity_abs_mps: torch.Tensor
    vertical_velocity_abs_mps: torch.Tensor


def last_reliable_terminal_velocity_target(
    normalized_anchor_state: torch.Tensor,
    supervision: FixedDTControlSupervision,
) -> torch.Tensor:
    """Return the last measured velocity target, never a fitted-tail placeholder."""

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


def _identity_horizontal_torch(
    horizontal: torch.Tensor, heading_rad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del heading_rad
    return horizontal[:, 0], horizontal[:, 1]


def _rotate_horizontal_torch(
    horizontal: torch.Tensor, heading_rad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.cos(heading_rad)
    sine = torch.sin(heading_rad)
    along = horizontal[:, 0] * cosine + horizontal[:, 1] * sine
    cross = -horizontal[:, 0] * sine + horizontal[:, 1] * cosine
    return along, cross


_TORCH_HORIZONTAL_COMPONENTS: dict[
    str, Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
] = {
    "enu": _rotate_horizontal_torch,
    "runway-aligned": _identity_horizontal_torch,
}


def terminal_state_errors(
    normalized_segment_end_states: torch.Tensor,
    normalized_terminal_targets: torch.Tensor,
    normalized_anchor_state: torch.Tensor,
    supervision: FixedDTControlSupervision,
    normalizer: Normalizer,
    runway_heading_rad: torch.Tensor,
    *,
    coordinate_frame: str,
    cross_track_emphasis: float,
    vertical_emphasis: float,
) -> TerminalStateErrors:
    """Decode and decompose terminal errors in runway along/cross/up axes."""

    dtype = normalized_segment_end_states.dtype
    device = normalized_segment_end_states.device
    position_indices = list(POSITION_IDX)
    velocity_indices = list(VELOCITY_IDX)
    endpoint_position = normalized_segment_end_states[:, -1, position_indices]
    target_position = normalized_terminal_targets[:, position_indices].to(
        dtype=dtype, device=device
    )
    position_scale = torch.as_tensor(
        normalizer.std[position_indices], dtype=dtype, device=device
    )
    position_delta = (endpoint_position - target_position) * position_scale

    endpoint_velocity = normalized_segment_end_states[:, -1, velocity_indices]
    target_velocity = last_reliable_terminal_velocity_target(
        normalized_anchor_state.to(dtype=dtype, device=device), supervision
    ).to(dtype=dtype, device=device)
    velocity_scale = torch.as_tensor(
        normalizer.std[velocity_indices], dtype=dtype, device=device
    )
    velocity_delta = (endpoint_velocity - target_velocity) * velocity_scale

    heading = runway_heading_rad.to(dtype=dtype, device=device).reshape(-1)
    if len(heading) != len(position_delta):
        raise ValueError("runway headings must align with the terminal-state batch")
    components = _TORCH_HORIZONTAL_COMPONENTS[coordinate_frame]
    position_along, position_cross = components(position_delta[:, :2], heading)
    velocity_along, velocity_cross = components(velocity_delta[:, :2], heading)
    position_abs = (
        position_along.abs(), position_cross.abs(), position_delta[:, 2].abs()
    )
    velocity_abs = (
        velocity_along.abs(), velocity_cross.abs(), velocity_delta[:, 2].abs()
    )
    return TerminalStateErrors(
        position_vector_m=torch.linalg.vector_norm(position_delta, dim=-1),
        velocity_vector_mps=torch.linalg.vector_norm(velocity_delta, dim=-1),
        position_runway_components_m=(
            position_abs[0]
            + cross_track_emphasis * position_abs[1]
            + vertical_emphasis * position_abs[2]
        ),
        velocity_runway_components_mps=(
            velocity_abs[0]
            + cross_track_emphasis * velocity_abs[1]
            + vertical_emphasis * velocity_abs[2]
        ),
        along_position_abs_m=position_abs[0],
        cross_position_abs_m=position_abs[1],
        vertical_position_abs_m=position_abs[2],
        along_velocity_abs_mps=velocity_abs[0],
        cross_velocity_abs_mps=velocity_abs[1],
        vertical_velocity_abs_mps=velocity_abs[2],
    )


def terminal_state_metrics_numpy(
    predicted_terminal: np.ndarray,
    reference_terminal: np.ndarray,
    terminal_velocity_target: np.ndarray,
    runway_heading_rad: float,
    *,
    coordinate_frame: str,
    cross_track_emphasis: float,
    vertical_emphasis: float,
) -> dict[str, float]:
    """NumPy counterpart for deterministic fixed-anchor validation."""

    position_delta = np.asarray(predicted_terminal)[list(POSITION_IDX)] - np.asarray(
        reference_terminal
    )[list(POSITION_IDX)]
    velocity_delta = np.asarray(predicted_terminal)[list(VELOCITY_IDX)] - np.asarray(
        terminal_velocity_target
    )
    if coordinate_frame == "runway-aligned":
        position_along, position_cross = position_delta[:2]
        velocity_along, velocity_cross = velocity_delta[:2]
    else:
        cosine, sine = np.cos(runway_heading_rad), np.sin(runway_heading_rad)
        position_along = position_delta[0] * cosine + position_delta[1] * sine
        position_cross = -position_delta[0] * sine + position_delta[1] * cosine
        velocity_along = velocity_delta[0] * cosine + velocity_delta[1] * sine
        velocity_cross = -velocity_delta[0] * sine + velocity_delta[1] * cosine
    position_abs = np.abs([position_along, position_cross, position_delta[2]])
    velocity_abs = np.abs([velocity_along, velocity_cross, velocity_delta[2]])
    return {
        "position_vector_m": float(np.linalg.norm(position_delta)),
        "velocity_vector_mps": float(np.linalg.norm(velocity_delta)),
        "position_runway_components_m": float(
            position_abs[0]
            + cross_track_emphasis * position_abs[1]
            + vertical_emphasis * position_abs[2]
        ),
        "velocity_runway_components_mps": float(
            velocity_abs[0]
            + cross_track_emphasis * velocity_abs[1]
            + vertical_emphasis * velocity_abs[2]
        ),
        "along_position_abs_m": float(position_abs[0]),
        "cross_position_abs_m": float(position_abs[1]),
        "vertical_position_abs_m": float(position_abs[2]),
        "along_velocity_abs_mps": float(velocity_abs[0]),
        "cross_velocity_abs_mps": float(velocity_abs[1]),
        "vertical_velocity_abs_mps": float(velocity_abs[2]),
    }
