"""Runway-aware terminal-state reference and errors, shared by training and validation."""

from __future__ import annotations

import numpy as np
import torch

from channels import POSITION_IDX, VELOCITY_IDX
from coordinate_frames import COORDINATE_FRAME_RUNWAY_ALIGNED
from fixed_dt_supervision import FixedDTControlSupervision


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
    if coordinate_frame == COORDINATE_FRAME_RUNWAY_ALIGNED:
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
