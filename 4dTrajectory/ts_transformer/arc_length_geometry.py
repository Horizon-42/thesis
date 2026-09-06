"""Arc-aligned position and local-velocity comparison for ordered trajectories.

Both curves are parameterized by normalized horizontal arc length. Position and chart
velocity payloads use the same interpolation plan, so local velocity constrains direction,
speed and descent rate at corresponding geometric progress without coupling to duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channels import POSITION_IDX
from dataset import Normalizer


@dataclass(frozen=True)
class _NumpyArcPlan:
    lower: np.ndarray
    upper: np.ndarray
    fraction: np.ndarray


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
