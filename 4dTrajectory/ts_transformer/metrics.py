"""Formal true-physical-time trajectory metrics and raw-curve QA.

The accuracy kernel evaluates denormalised predictions on one fixed true-time grid and
reports 3D ADE/FDE plus horizontal, along-track, cross-track, vertical, endpoint and clock
errors.  Raw kinematic summaries are a separate quality-assurance family and never select a
checkpoint.  The former normalized-progress evaluator was removed so production code has one
accuracy definition rather than two similarly named alternatives.
"""

from __future__ import annotations

import numpy as np

from channels import IDX, POSITION_IDX, VELOCITY_IDX

# Reported percentiles. p95 mirrors evaluation/stats.magnitude_spread so the ML-side and
# gate-side summaries can be read against each other.
P95 = 95.0

# Stable scalar keys shared by history.json, CV results, prediction summary.json and the
# kinematic-weight ablation. Keeping this list beside the implementation prevents each
# persistence boundary from inventing a subtly different subset or spelling.
RAW_KINEMATIC_METRIC_KEYS = (
    "position_velocity_rmse_mps",
    "heading_consistency_p95_deg",
    "turn_rate_p95_deg_s",
    "acceleration_p95_mps2",
    "jerk_p95_mps3",
)


def states_with_derived_velocity(
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    segment_durations_s: np.ndarray,
) -> np.ndarray:
    """Return physical states whose velocity is the predicted position derivative.

    The direct-state model's future target is position plus duration.  Velocity remains an
    input feature, but at inference it is derived from consecutive predicted positions so
    exported state rows cannot carry a second, contradictory trajectory.  Each node uses
    the left derivative of the piecewise-linear position curve over its preceding segment.
    """
    anchor = np.asarray(anchor_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    durations = np.asarray(segment_durations_s, dtype=np.float64)
    squeeze = predicted.ndim == 2
    if squeeze:
        predicted = predicted[None, ...]
        anchor = anchor[None, ...] if anchor.ndim == 1 else anchor
        durations = durations[None, ...] if durations.ndim == 1 else durations
    if predicted.ndim != 3 or anchor.ndim != 2 or durations.ndim != 2:
        raise ValueError("derived velocity requires [B,C], [B,N,C], and [B,N]")
    if predicted.shape[:2] != durations.shape or predicted.shape[0] != anchor.shape[0]:
        raise ValueError("derived velocity inputs do not share batch and segment axes")
    if predicted.shape[2] != anchor.shape[1]:
        raise ValueError("anchor and predicted states use different channel counts")
    if not (
        np.isfinite(anchor).all()
        and np.isfinite(predicted).all()
        and np.isfinite(durations).all()
    ):
        raise ValueError("derived velocity inputs must be finite")
    if np.any(durations <= 0.0):
        raise ValueError("derived velocity segment durations must be positive")

    result = predicted.copy()
    positions = result[..., list(POSITION_IDX)]
    previous = np.concatenate(
        (anchor[:, None, list(POSITION_IDX)], positions[:, :-1]), axis=1
    )
    result[..., list(VELOCITY_IDX)] = (
        positions - previous
    ) / durations[..., None]
    return result[0] if squeeze else result


def _validated_path(
    offsets_s: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(offsets_s, dtype=np.float64)
    path = np.asarray(values, dtype=np.float64)
    if offsets.ndim != 1 or path.ndim != 2 or len(offsets) != len(path):
        raise ValueError(f"{label} offsets and values must align as [N] and [N,C]")
    if not len(offsets):
        raise ValueError(f"{label} path must contain at least one future node")
    if path.shape[1] <= max(POSITION_IDX):
        raise ValueError(f"{label} path does not contain three position channels")
    if not np.isfinite(offsets).all() or not np.isfinite(path).all():
        raise ValueError(f"{label} path must be finite")
    if offsets[0] <= 0.0 or np.any(np.diff(offsets) <= 0.0):
        raise ValueError(f"{label} clock must be finite, positive, and strictly increasing")
    return offsets, path


def _sample_positions(
    anchor_values: np.ndarray,
    offsets_s: np.ndarray,
    values: np.ndarray,
    query_offsets_s: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    offsets, path = _validated_path(offsets_s, values, label=label)
    anchor = np.asarray(anchor_values, dtype=np.float64)
    query = np.asarray(query_offsets_s, dtype=np.float64)
    if anchor.ndim != 1 or anchor.shape[0] != path.shape[1]:
        raise ValueError(f"{label} anchor must be [C] and match the path")
    if query.ndim != 1 or not np.isfinite(query).all() or np.any(query < 0.0):
        raise ValueError("common-grid query offsets must be finite and non-negative")
    if not np.isfinite(anchor).all():
        raise ValueError(f"{label} anchor must be finite")
    node_times = np.concatenate(([0.0], offsets))
    positions = np.concatenate(
        (anchor[None, list(POSITION_IDX)], path[:, list(POSITION_IDX)]), axis=0
    )
    # np.interp holds the final node to the right.  This is intentional: a prediction
    # ending early remains accountable over the complete true flight horizon.
    return np.column_stack([
        np.interp(query, node_times, positions[:, axis]) for axis in range(3)
    ])


def common_physical_time_flight_metrics(
    *,
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_offsets_s: np.ndarray,
    predicted_final_time_s: float,
    truth_values: np.ndarray,
    truth_offsets_s: np.ndarray,
    true_final_time_s: float,
    points: int = 64,
) -> dict[str, object]:
    """One flight on a fixed true-physical-time grid, with no overlap truncation."""
    if isinstance(points, bool) or not isinstance(points, int) or points <= 1:
        raise ValueError("common physical-time points must be an integer greater than one")
    if not np.isfinite(predicted_final_time_s) or predicted_final_time_s <= 0.0:
        raise ValueError("predicted final time must be finite and positive")
    if not np.isfinite(true_final_time_s) or true_final_time_s <= 0.0:
        raise ValueError("true final time must be finite and positive")
    truth_offsets, truth_path = _validated_path(
        truth_offsets_s, truth_values, label="truth"
    )
    tolerance = np.finfo(np.float64).eps * max(true_final_time_s, 1.0) * 32.0
    if truth_offsets[-1] < true_final_time_s - tolerance:
        raise ValueError("truth path does not cover its declared final time")

    progress = np.arange(1, points + 1, dtype=np.float64) / points
    query = progress * true_final_time_s
    truth = _sample_positions(
        anchor_values, truth_offsets, truth_path, query, label="truth"
    )
    predicted = _sample_positions(
        anchor_values,
        predicted_offsets_s,
        predicted_values,
        query,
        label="prediction",
    )
    delta = predicted - truth
    displacement = np.linalg.norm(delta, axis=1)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)

    anchor_position = np.asarray(anchor_values, dtype=np.float64)[list(POSITION_IDX)]
    truth_with_anchor = np.concatenate((anchor_position[None, :], truth), axis=0)
    time_with_anchor = np.concatenate(([0.0], query))
    tangent = np.gradient(truth_with_anchor[:, :2], time_with_anchor, axis=0)[1:]
    tangent_norm = np.linalg.norm(tangent, axis=1)
    unit = np.zeros_like(tangent)
    moving = tangent_norm > 1e-9
    unit[moving] = tangent[moving] / tangent_norm[moving, None]
    along = delta[:, 0] * unit[:, 0] + delta[:, 1] * unit[:, 1]
    cross = -delta[:, 0] * unit[:, 1] + delta[:, 1] * unit[:, 0]
    vertical = delta[:, 2]
    time_error = float(predicted_final_time_s - true_final_time_s)
    predicted_arrival = _sample_positions(
        anchor_values,
        predicted_offsets_s,
        predicted_values,
        np.asarray([predicted_final_time_s], dtype=np.float64),
        label="prediction",
    )[0]
    arrival_endpoint_error = float(np.linalg.norm(predicted_arrival - truth[-1]))

    return {
        "metric_grid": "common true physical time",
        "points": points,
        "ade_m": float(displacement.mean()),
        "fde_m": float(displacement[-1]),
        "arrival_endpoint_error_m": arrival_endpoint_error,
        "horizontal_ade_m": float(horizontal.mean()),
        "along_track_m": _spread(along),
        "cross_track_m": _spread(cross),
        "vertical_m": _spread(vertical),
        "final_time_error_s": time_error,
        "true_final_time_s": float(true_final_time_s),
        "predicted_final_time_s": float(predicted_final_time_s),
        "coverage_ratio": float(
            np.asarray(predicted_offsets_s, dtype=np.float64)[-1] / true_final_time_s
        ),
        "n_steps": points,
        "displacement_m": displacement,
    }


def _p95(values: np.ndarray) -> float:
    """p95 of finite magnitudes, or NaN when a trajectory is too short to define it."""
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, P95)) if finite.size else float("nan")


def _wrapped_angle_delta(radians: np.ndarray) -> np.ndarray:
    """Adjacent signed angle changes in [-pi, pi], along the final axis."""
    delta = np.diff(radians, axis=-1)
    return np.arctan2(np.sin(delta), np.cos(delta))


def raw_kinematic_metrics(
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    valid_segments: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Physical smoothness of unfiltered model output on its own time grid.

    ``anchor_values`` is ``[B,C]``, predicted output is ``[B,N,C]`` and durations are
    ``[B,N]``. Durations are explicit rather than reconstructed from ``final_time_s`` so
    the same metric contract works unchanged when the output layer starts emitting
    nonuniform segment durations.

    Position-difference velocity and all higher derivatives come from the raw position
    nodes, because those are the points exported to CZML. The velocity channels are used
    only for the position/velocity RMSE and heading-consistency checks; a model cannot earn
    a smoothness score by predicting smooth velocities beside a jagged position path.
    """
    anchor = np.asarray(anchor_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    durations = np.asarray(segment_durations_s, dtype=np.float64)
    if anchor.ndim != 2 or predicted.ndim != 3 or durations.ndim != 2:
        raise ValueError("raw kinematic metrics require [B,C], [B,N,C], [B,N]")
    if predicted.shape[:2] != durations.shape or anchor.shape != predicted[:, 0].shape:
        raise ValueError("raw kinematic metric shapes do not share B, N and C")
    if valid_segments is None:
        valid = np.ones_like(durations, dtype=bool)
    else:
        valid = np.asarray(valid_segments, dtype=bool)
        if valid.shape != durations.shape:
            raise ValueError("valid segment mask must match [B,N] durations")
        # A padded suffix is the only supported ragged representation. Accepting holes
        # would make acceleration/jerk adjacency ambiguous.
        if np.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("valid segments must form a contiguous prefix")
    if np.any(durations[valid] <= 0.0):
        raise ValueError("valid segment durations must be positive")
    safe_durations = np.where(valid, durations, 1.0)

    nodes = np.concatenate((anchor[:, None, :], predicted), axis=1)
    positions = nodes[..., list(POSITION_IDX)]
    velocity_indices = [IDX["edot"], IDX["ndot"], IDX["udot"]]
    state_velocities = nodes[..., velocity_indices]

    geometric_velocity = np.diff(positions, axis=1) / safe_durations[..., None]
    midpoint_velocity = 0.5 * (state_velocities[:, 1:] + state_velocities[:, :-1])
    velocity_residual = geometric_velocity - midpoint_velocity

    geometric_heading = np.arctan2(
        geometric_velocity[..., 1], geometric_velocity[..., 0]
    )
    state_heading = np.arctan2(midpoint_velocity[..., 1], midpoint_velocity[..., 0])
    geometric_speed = np.linalg.norm(geometric_velocity[..., :2], axis=-1)
    state_speed = np.linalg.norm(midpoint_velocity[..., :2], axis=-1)
    heading_valid = valid & (geometric_speed > 1e-6) & (state_speed > 1e-6)
    heading_delta = np.arctan2(
        np.sin(geometric_heading - state_heading),
        np.cos(geometric_heading - state_heading),
    )

    node_times = np.concatenate(
        (
            np.zeros((len(durations), 1), dtype=np.float64),
            np.cumsum(safe_durations, axis=1),
        ),
        axis=1,
    )
    velocity_times = 0.5 * (node_times[:, 1:] + node_times[:, :-1])
    velocity_time_steps = np.diff(velocity_times, axis=1)

    turn_rate = np.degrees(_wrapped_angle_delta(geometric_heading)) / velocity_time_steps
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    turn_valid = (
        adjacent_valid
        & (geometric_speed[:, 1:] > 1e-6)
        & (geometric_speed[:, :-1] > 1e-6)
    )

    acceleration = np.diff(geometric_velocity, axis=1) / velocity_time_steps[..., None]
    acceleration_magnitude = np.linalg.norm(acceleration, axis=-1)
    acceleration_times = 0.5 * (velocity_times[:, 1:] + velocity_times[:, :-1])
    acceleration_time_steps = np.diff(acceleration_times, axis=1)
    jerk = np.diff(acceleration, axis=1) / acceleration_time_steps[..., None]
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)
    jerk_valid = adjacent_valid[:, 1:] & adjacent_valid[:, :-1]

    velocity_values = velocity_residual[valid]
    position_velocity_rmse = (
        float(np.sqrt(np.mean(velocity_values**2)))
        if velocity_values.size
        else float("nan")
    )

    return {
        # Component RMSE matches the physical residual underlying the training loss.
        "position_velocity_rmse_mps": position_velocity_rmse,
        "heading_consistency_p95_deg": _p95(np.degrees(heading_delta[heading_valid])),
        "turn_rate_p95_deg_s": _p95(turn_rate[turn_valid]),
        "acceleration_p95_mps2": _p95(acceleration_magnitude[adjacent_valid]),
        "jerk_p95_mps3": _p95(jerk_magnitude[jerk_valid]),
        "segments": int(valid.sum()),
    }


def _spread(values: np.ndarray) -> dict[str, float]:
    """Magnitude summary of a signed error array.

    The vectorised twin of ``evaluation/stats.signed_spread`` (same keys, same
    percentile method) — NOT a call into it, because that implementation is stdlib-only
    by design (it judges 101-sample paths) and sorting millions of boxed floats here
    would dominate evaluate_split. A seam test pins the two equal on the same input.
    """
    magnitude = np.abs(values)
    return {
        "mean_abs": float(magnitude.mean()),
        "p95_abs": float(np.percentile(magnitude, P95)),
        "max_abs": float(magnitude.max()),
        "mean_signed": float(values.mean()),
    }
