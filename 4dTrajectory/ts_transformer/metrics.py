"""Trajectory error metrics, in metres, on denormalised channel predictions.

Two families, because they answer different questions:

**Displacement** — ADE (average displacement error, mean over normalized progress) and FDE (final
displacement error, at the last valid step). The standard pair in the trajectory-prediction
literature and what the survey in ``4dTrajectory/docs`` reports for every method.

**Decomposed** — along-track / cross-track / altitude. A 400 m ADE means something very
different when it is 400 m of "arrived early/late along the same path" than when it is
400 m of "flew a different path", and only the second threatens the lateral containment
the evaluation gates check. The decomposition is taken in the frame of the TRUE velocity at
each step: the along-track unit vector is the truth's own horizontal heading, so
along-track error is a timing/speed error and cross-track error is a path error.

All inputs are PHYSICAL units (metres, m/s) — decode through the normalizer first. State
weights can exclude fitted position-only supervision from observed-track headline metrics.
"""

from __future__ import annotations

import numpy as np

from channels import IDX, POSITION_IDX

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


def _positions(values: np.ndarray) -> np.ndarray:
    """[..., C] -> [..., 3] east/north/up."""
    return values[..., list(POSITION_IDX)]


def _horizontal_unit(values: np.ndarray) -> np.ndarray:
    """Unit vector along the truth's horizontal velocity, [..., 2].

    Built from the chart-derivative channels; their direction differs from the physical
    heading by the ratio of the two transport factors (< 0.1 deg over a TMA), which is
    noise for an error DECOMPOSITION frame. Where ground speed is ~0 (a stationary or
    purely vertical sample) the direction is undefined; those steps get a zero vector,
    which sends their whole horizontal error into the cross-track term rather than
    splitting it arbitrarily.
    """
    ve = values[..., IDX["edot"]]
    vn = values[..., IDX["ndot"]]
    speed = np.hypot(ve, vn)
    safe = speed > 1e-6
    unit = np.zeros(values.shape[:-1] + (2,), dtype=np.float64)
    unit[..., 0] = np.where(safe, ve / np.where(safe, speed, 1.0), 0.0)
    unit[..., 1] = np.where(safe, vn / np.where(safe, speed, 1.0), 0.0)
    return unit


def error_components(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Flat, mask-filtered per-step error components in metres.

    ``predicted`` / ``truth`` are ``[B, N, C]`` in physical units, ``mask`` is ``[B, N]``.
    Returns 1-D arrays over the valid progress points: ``displacement`` (3D), ``horizontal``,
    ``along`` (signed, + = predicted ahead of truth), ``cross`` (signed, + = left of the
    true course), ``vertical`` (signed, + = predicted high) — plus ``displacement_grid``,
    the UNMASKED ``[B, H]`` displacement, kept for per-sample indexing (FDE).
    """
    delta = _positions(predicted) - _positions(truth)
    unit = _horizontal_unit(truth)

    de, dn, du = delta[..., 0], delta[..., 1], delta[..., 2]
    along = de * unit[..., 0] + dn * unit[..., 1]
    # Left-normal of (ux, uy) is (-uy, ux); positive cross-track = left of the true course.
    cross = de * -unit[..., 1] + dn * unit[..., 0]

    displacement_grid = np.sqrt(de**2 + dn**2 + du**2)
    valid = mask > 0.5
    return {
        "displacement": displacement_grid[valid],
        "displacement_grid": displacement_grid,
        "horizontal": np.hypot(de, dn)[valid],
        "along": along[valid],
        "cross": cross[valid],
        "vertical": du[valid],
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


def final_index(mask: np.ndarray) -> np.ndarray:
    """Index of the last valid progress point per sample, ``[B]``."""
    return mask.shape[1] - 1 - np.argmax(mask[:, ::-1] > 0.5, axis=1)


def trajectory_metrics(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    """The full metric block for a batch of predictions.

    ``{ade_m, fde_m, ..., along_track_m: {...}, cross_track_m: {...},
       altitude_m: {...}, n_steps, n_samples}``

    ADE averages per-progress-point displacement. FDE is the error at normalized progress
    one, the predicted endpoint of each approach.
    """
    components = error_components(predicted, truth, mask)
    displacement = components["displacement"]

    last = final_index(mask)
    rows = np.arange(predicted.shape[0])
    has_any = mask.sum(axis=1) > 0
    fde = components["displacement_grid"][rows, last][has_any]

    return {
        "ade_m": float(displacement.mean()),
        "fde_m": float(fde.mean()),
        "ade_p95_m": float(np.percentile(displacement, P95)),
        "fde_p95_m": float(np.percentile(fde, P95)),
        "horizontal_m": _spread(components["horizontal"]),
        "along_track_m": _spread(components["along"]),
        "cross_track_m": _spread(components["cross"]),
        "altitude_m": _spread(components["vertical"]),
        "n_steps": int(displacement.size),
        "n_samples": int(predicted.shape[0]),
    }


def error_by_progress(
    predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> list[dict[str, float]]:
    """Displacement error over the shared normalized progress domain ``(0, 1]``."""
    per_step = np.sqrt(((_positions(predicted) - _positions(truth)) ** 2).sum(axis=-1))
    rows = []
    for h in range(predicted.shape[1]):
        valid = mask[:, h] > 0.5
        if not valid.any():
            continue
        errors = per_step[valid, h]
        rows.append({
            "segment": h + 1,
            "progress": (h + 1) / predicted.shape[1],
            "mean_m": float(errors.mean()),
            "p95_m": float(np.percentile(errors, P95)),
            "n": int(valid.sum()),
        })
    return rows
