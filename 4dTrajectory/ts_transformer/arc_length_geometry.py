"""Arc-aligned position and local-velocity comparison for ordered trajectories.

Both curves are parameterized by normalized horizontal arc length. Position and chart
velocity payloads use the same interpolation plan, so local velocity constrains direction,
speed and descent rate at corresponding geometric progress without coupling to duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from channels import POSITION_IDX, VELOCITY_IDX
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision


@dataclass(frozen=True)
class ArcLengthStateLossTerms:
    """Per-flight unweighted arc-aligned tracking terms."""

    position: torch.Tensor
    horizontal_velocity_mps: torch.Tensor
    horizontal_tangent: torch.Tensor
    horizontal_speed_mps: torch.Tensor
    vertical_velocity_mps: torch.Tensor
    velocity_valid_points: torch.Tensor


@dataclass(frozen=True)
class _TorchArcPlan:
    lower: torch.Tensor
    upper: torch.Tensor
    fraction: torch.Tensor


@dataclass(frozen=True)
class _NumpyArcPlan:
    lower: np.ndarray
    upper: np.ndarray
    fraction: np.ndarray


def _horizontal_arc_plan_torch(
    positions_m: torch.Tensor,
    valid: torch.Tensor,
    *,
    points: int,
) -> _TorchArcPlan:
    if positions_m.ndim != 3 or positions_m.shape[-1] != len(POSITION_IDX):
        raise ValueError("positions_m must be [B,P,3]")
    if valid.shape != positions_m.shape[:2]:
        raise ValueError("valid must align with positions_m")
    if points <= 1:
        raise ValueError("arc-length grid points must be greater than one")
    lengths = valid.sum(dim=1)
    if torch.any(lengths < 2):
        raise ValueError("every arc-length curve must contain at least two valid points")

    segment_valid = valid[:, 1:] & valid[:, :-1]
    segment_length = torch.linalg.vector_norm(
        positions_m[:, 1:, :2] - positions_m[:, :-1, :2], dim=-1
    ) * segment_valid.to(dtype=positions_m.dtype)
    cumulative = torch.cat(
        (positions_m.new_zeros((len(positions_m), 1)), segment_length.cumsum(dim=1)),
        dim=1,
    )
    last = lengths - 1
    total = cumulative.gather(1, last.unsqueeze(1)).squeeze(1)
    progress = torch.linspace(
        0.0, 1.0, points, dtype=positions_m.dtype, device=positions_m.device
    )
    targets = total.unsqueeze(1) * progress.unsqueeze(0)
    upper = torch.searchsorted(
        cumulative.contiguous(), targets.contiguous(), right=False
    )
    upper = torch.maximum(upper, torch.ones_like(upper))
    upper = torch.minimum(upper, last.unsqueeze(1))
    lower = upper - 1
    lower_arc = cumulative.gather(1, lower)
    upper_arc = cumulative.gather(1, upper)
    fraction = (targets - lower_arc) / (upper_arc - lower_arc).clamp_min(
        torch.finfo(positions_m.dtype).eps
    )
    return _TorchArcPlan(lower, upper, fraction.clamp(0.0, 1.0))


def _interpolate_torch(values: torch.Tensor, plan: _TorchArcPlan) -> torch.Tensor:
    def gather(indices: torch.Tensor) -> torch.Tensor:
        return values.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        )

    lower = gather(plan.lower)
    upper = gather(plan.upper)
    return lower + plan.fraction.unsqueeze(-1) * (upper - lower)


def _interpolated_valid_torch(
    node_valid: torch.Tensor, plan: _TorchArcPlan
) -> torch.Tensor:
    lower_valid = node_valid.gather(1, plan.lower)
    upper_valid = node_valid.gather(1, plan.upper)
    epsilon = torch.finfo(plan.fraction.dtype).eps * 8.0
    lower_needed = plan.fraction < 1.0 - epsilon
    upper_needed = plan.fraction > epsilon
    return ((~lower_needed) | lower_valid) & ((~upper_needed) | upper_valid)


def resample_horizontal_arc_length_torch(
    positions_m: torch.Tensor,
    valid: torch.Tensor,
    *,
    points: int,
) -> torch.Tensor:
    """Resample ordered ``[B,P,3]`` curves on a uniform horizontal arc grid."""

    plan = _horizontal_arc_plan_torch(positions_m, valid, points=points)
    return _interpolate_torch(positions_m, plan)


def _horizontal_arc_plan_numpy(
    positions_m: np.ndarray, *, points: int
) -> _NumpyArcPlan:
    positions_m = np.asarray(positions_m, dtype=np.float64)
    if positions_m.ndim != 2 or positions_m.shape[1] != len(POSITION_IDX):
        raise ValueError("positions_m must be [P,3]")
    if len(positions_m) < 2:
        raise ValueError("arc-length curve must contain at least two points")
    if points <= 1:
        raise ValueError("arc-length grid points must be greater than one")

    segment_length = np.linalg.norm(np.diff(positions_m[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_length)))
    targets = np.linspace(0.0, cumulative[-1], points, dtype=np.float64)
    upper = np.searchsorted(cumulative, targets, side="left")
    upper = np.clip(upper, 1, len(positions_m) - 1)
    lower = upper - 1
    denominator = np.maximum(
        cumulative[upper] - cumulative[lower], np.finfo(np.float64).eps
    )
    fraction = np.clip(
        (targets - cumulative[lower]) / denominator, 0.0, 1.0
    )
    return _NumpyArcPlan(lower, upper, fraction)


def _interpolate_numpy(values: np.ndarray, plan: _NumpyArcPlan) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[plan.lower] + plan.fraction[:, None] * (
        values[plan.upper] - values[plan.lower]
    )


def _interpolated_valid_numpy(
    node_valid: np.ndarray, plan: _NumpyArcPlan
) -> np.ndarray:
    node_valid = np.asarray(node_valid, dtype=bool)
    epsilon = np.finfo(np.float64).eps * 8.0
    lower_needed = plan.fraction < 1.0 - epsilon
    upper_needed = plan.fraction > epsilon
    return (
        ((~lower_needed) | node_valid[plan.lower])
        & ((~upper_needed) | node_valid[plan.upper])
    )


def resample_horizontal_arc_length_numpy(
    positions_m: np.ndarray,
    *,
    points: int,
) -> np.ndarray:
    """NumPy equivalent used by deterministic fixed-anchor validation."""

    plan = _horizontal_arc_plan_numpy(positions_m, points=points)
    return _interpolate_numpy(positions_m, plan)


def arc_length_state_loss_terms(
    normalized_anchor_state: torch.Tensor,
    normalized_segment_end_states: torch.Tensor,
    normalized_terminal_target: torch.Tensor,
    supervision: FixedDTControlSupervision,
    normalizer: Normalizer,
    *,
    points: int,
    position_end_weight: float = 1.0,
) -> ArcLengthStateLossTerms:
    """Return position and reliable local-velocity terms on one arc grid."""

    position_indices = list(POSITION_IDX)
    velocity_indices = list(VELOCITY_IDX)
    dtype = normalized_segment_end_states.dtype
    device = normalized_segment_end_states.device
    channel_count = normalized_segment_end_states.shape[-1]
    mean = torch.as_tensor(normalizer.mean, dtype=dtype, device=device)
    scale = torch.as_tensor(normalizer.std, dtype=dtype, device=device)

    predicted_normalized = torch.cat(
        (
            normalized_anchor_state[:, None].to(dtype=dtype, device=device),
            normalized_segment_end_states,
        ),
        dim=1,
    )
    supervision_states = supervision.states.to(dtype=dtype, device=device)
    position_row_valid = supervision.valid.to(device=device) & torch.all(
        supervision.weights[..., position_indices].to(device=device) > 0.0,
        dim=-1,
    )
    velocity_row_valid = supervision.valid.to(device=device) & torch.all(
        supervision.weights[..., velocity_indices].to(device=device) > 0.0,
        dim=-1,
    )
    reference_lengths = position_row_valid.sum(dim=1)
    reference_normalized = normalized_segment_end_states.new_zeros(
        (len(supervision_states), supervision_states.shape[1] + 2, channel_count)
    )
    reference_normalized[:, 0] = normalized_anchor_state.to(
        dtype=dtype, device=device
    )
    source_rows, source_columns = torch.nonzero(
        position_row_valid, as_tuple=True
    )
    compacted_columns = position_row_valid.cumsum(dim=1)[
        source_rows, source_columns
    ]
    reference_normalized[source_rows, compacted_columns] = supervision_states[
        source_rows, source_columns
    ]
    rows = torch.arange(len(reference_normalized), device=device)
    reference_normalized[rows, reference_lengths + 1] = normalized_terminal_target.to(
        dtype=dtype, device=device
    )
    reference_position_valid = (
        torch.arange(reference_normalized.shape[1], device=device).unsqueeze(0)
        < (reference_lengths + 2).unsqueeze(1)
    )
    reference_velocity_valid = torch.zeros_like(reference_position_valid)
    reference_velocity_valid[:, 0] = True
    reference_velocity_valid[source_rows, compacted_columns] = velocity_row_valid[
        source_rows, source_columns
    ]

    predicted_physical = predicted_normalized * scale + mean
    reference_physical = reference_normalized * scale + mean
    predicted_positions = predicted_physical[..., position_indices]
    reference_positions = reference_physical[..., position_indices]
    predicted_plan = _horizontal_arc_plan_torch(
        predicted_positions,
        torch.ones(
            predicted_positions.shape[:2], dtype=torch.bool, device=device
        ),
        points=points,
    )
    reference_plan = _horizontal_arc_plan_torch(
        reference_positions, reference_position_valid, points=points
    )
    predicted_position_grid = _interpolate_torch(predicted_positions, predicted_plan)
    reference_position_grid = _interpolate_torch(reference_positions, reference_plan)
    position_scale = scale[position_indices]
    normalized_delta = (
        predicted_position_grid - reference_position_grid
    ) / position_scale
    position_point_loss = F.smooth_l1_loss(
        normalized_delta,
        torch.zeros_like(normalized_delta),
        reduction="none",
        beta=1.0,
    ).mean(dim=2)
    progress = torch.linspace(0.0, 1.0, points, dtype=dtype, device=device)
    progress_weight = 1.0 + (position_end_weight - 1.0) * progress
    progress_weight = progress_weight / progress_weight.mean()
    position_loss = (position_point_loss * progress_weight.unsqueeze(0)).mean(dim=1)

    predicted_velocity_grid = _interpolate_torch(
        predicted_physical[..., velocity_indices], predicted_plan
    )
    reference_velocity_grid = _interpolate_torch(
        reference_physical[..., velocity_indices], reference_plan
    )
    velocity_valid = _interpolated_valid_torch(
        reference_velocity_valid, reference_plan
    )
    valid_count = velocity_valid.sum(dim=1)
    if torch.any(valid_count == 0):
        raise ValueError("arc-length velocity loss has no reliable reference points")
    velocity_delta = predicted_velocity_grid - reference_velocity_grid
    valid_float = velocity_valid.to(dtype=dtype)
    horizontal_velocity_mps = (
        torch.linalg.vector_norm(velocity_delta[..., :2], dim=-1) * valid_float
    ).sum(dim=1) / valid_count.to(dtype=dtype)
    predicted_horizontal = predicted_velocity_grid[..., :2]
    reference_horizontal = reference_velocity_grid[..., :2]
    tangent_error = 1.0 - F.cosine_similarity(
        predicted_horizontal, reference_horizontal, dim=-1, eps=1e-8
    )
    horizontal_tangent = (tangent_error * valid_float).sum(dim=1) / valid_count.to(
        dtype=dtype
    )
    horizontal_speed_mps = (
        torch.abs(
            torch.linalg.vector_norm(predicted_horizontal, dim=-1)
            - torch.linalg.vector_norm(reference_horizontal, dim=-1)
        )
        * valid_float
    ).sum(dim=1) / valid_count.to(dtype=dtype)
    vertical_velocity_mps = (
        torch.abs(velocity_delta[..., 2]) * valid_float
    ).sum(dim=1) / valid_count.to(dtype=dtype)
    return ArcLengthStateLossTerms(
        position=position_loss,
        horizontal_velocity_mps=horizontal_velocity_mps,
        horizontal_tangent=horizontal_tangent,
        horizontal_speed_mps=horizontal_speed_mps,
        vertical_velocity_mps=vertical_velocity_mps,
        velocity_valid_points=valid_count,
    )


def arc_length_geometry_metrics(
    predicted_positions_m: np.ndarray,
    reference_positions_m: np.ndarray,
    normalizer: Normalizer,
    *,
    points: int,
    position_end_weight: float = 1.0,
) -> dict[str, float]:
    """Physical and normalized position metrics for one ordered curve pair."""

    predicted_grid = resample_horizontal_arc_length_numpy(
        predicted_positions_m, points=points
    )
    reference_grid = resample_horizontal_arc_length_numpy(
        reference_positions_m, points=points
    )
    delta = predicted_grid - reference_grid
    normalized = delta / np.asarray(
        normalizer.std[list(POSITION_IDX)], dtype=np.float64
    )
    absolute = np.abs(normalized)
    smooth_l1 = np.where(absolute < 1.0, 0.5 * absolute**2, absolute - 0.5)
    unweighted_loss = float(smooth_l1.mean())
    progress = np.linspace(0.0, 1.0, points, dtype=np.float64)
    progress_weight = 1.0 + (position_end_weight - 1.0) * progress
    progress_weight /= progress_weight.mean()
    weighted_loss = float((smooth_l1.mean(axis=1) * progress_weight).mean())
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    vertical = np.abs(delta[:, 2])
    distance = np.linalg.norm(delta, axis=1)
    predicted_length_m = float(
        np.linalg.norm(
            np.diff(np.asarray(predicted_positions_m)[:, :2], axis=0), axis=1
        ).sum()
    )
    reference_length_m = float(
        np.linalg.norm(
            np.diff(np.asarray(reference_positions_m)[:, :2], axis=0), axis=1
        ).sum()
    )
    epsilon = np.finfo(np.float64).eps
    path_length_ratio = (predicted_length_m + epsilon) / (
        reference_length_m + epsilon
    )
    terminal = np.linalg.norm(
        np.asarray(predicted_positions_m)[-1]
        - np.asarray(reference_positions_m)[-1]
    )
    return {
        "loss": weighted_loss,
        "unweighted_loss": unweighted_loss,
        "distance_mean_m": float(distance.mean()),
        "predicted_horizontal_length_m": predicted_length_m,
        "reference_horizontal_length_m": reference_length_m,
        "path_length_ratio": float(path_length_ratio),
        "path_length_log_error": float(abs(np.log(path_length_ratio))),
        "horizontal_mean_m": float(horizontal.mean()),
        "horizontal_p95_m": float(np.percentile(horizontal, 95)),
        "vertical_mae_m": float(vertical.mean()),
        "vertical_p95_m": float(np.percentile(vertical, 95)),
        "terminal_position_m": float(terminal),
    }


def arc_length_velocity_metrics(
    predicted_positions_m: np.ndarray,
    predicted_velocity_mps: np.ndarray,
    reference_positions_m: np.ndarray,
    reference_velocity_mps: np.ndarray,
    reference_velocity_valid: np.ndarray,
    *,
    points: int,
) -> dict[str, float]:
    """Velocity error using the same horizontal-arc plans as position alignment."""

    predicted_plan = _horizontal_arc_plan_numpy(predicted_positions_m, points=points)
    reference_plan = _horizontal_arc_plan_numpy(reference_positions_m, points=points)
    predicted_grid = _interpolate_numpy(predicted_velocity_mps, predicted_plan)
    reference_grid = _interpolate_numpy(reference_velocity_mps, reference_plan)
    valid = _interpolated_valid_numpy(reference_velocity_valid, reference_plan)
    if not np.any(valid):
        raise ValueError("arc-length velocity metrics have no reliable reference points")
    delta = predicted_grid[valid] - reference_grid[valid]
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    predicted_horizontal = predicted_grid[valid, :2]
    reference_horizontal = reference_grid[valid, :2]
    predicted_speed = np.linalg.norm(predicted_horizontal, axis=1)
    reference_speed = np.linalg.norm(reference_horizontal, axis=1)
    denominator = np.maximum(
        predicted_speed * reference_speed, np.finfo(np.float64).eps
    )
    tangent = 1.0 - np.clip(
        np.sum(predicted_horizontal * reference_horizontal, axis=1) / denominator,
        -1.0,
        1.0,
    )
    speed = np.abs(predicted_speed - reference_speed)
    vertical = np.abs(delta[:, 2])
    return {
        "horizontal_velocity_mae_mps": float(horizontal.mean()),
        "horizontal_velocity_p95_mps": float(np.percentile(horizontal, 95)),
        "horizontal_tangent_mean": float(tangent.mean()),
        "horizontal_tangent_p95": float(np.percentile(tangent, 95)),
        "horizontal_speed_mae_mps": float(speed.mean()),
        "horizontal_speed_p95_mps": float(np.percentile(speed, 95)),
        "vertical_velocity_mae_mps": float(vertical.mean()),
        "vertical_velocity_p95_mps": float(np.percentile(vertical, 95)),
        "velocity_valid_points": int(valid.sum()),
    }
