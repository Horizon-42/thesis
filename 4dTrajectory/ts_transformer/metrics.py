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
