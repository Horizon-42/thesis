"""Reference chart-velocity construction, isolated from trajectory loading.

The historical ``track-fit`` source is produced upstream from the raw ADS-B points.
The two position-derived candidates operate only on the already uniform chart positions,
so they do not alter the geometric reference or the dynamics implementation.
"""

from __future__ import annotations

import numpy as np

from channels import POSITION_IDX, VELOCITY_IDX


REFERENCE_VELOCITY_TRACK_FIT = "track-fit"
REFERENCE_VELOCITY_POSITION_DIFFERENCE = "position-difference"
REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE = (
    "smoothed-position-difference"
)
REFERENCE_VELOCITY_SOURCES = (
    REFERENCE_VELOCITY_TRACK_FIT,
    REFERENCE_VELOCITY_POSITION_DIFFERENCE,
    REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
)


def _causal_position_difference(
    times_s: np.ndarray, positions_m: np.ndarray
) -> np.ndarray:
    interval_velocity = np.diff(positions_m, axis=0) / np.diff(times_s)[:, None]
    return np.concatenate((interval_velocity[:1], interval_velocity), axis=0)


def _causal_local_linear_velocity(
    times_s: np.ndarray, positions_m: np.ndarray, *, window: int = 5
) -> np.ndarray:
    """Slope of a trailing local line; deployable and less noisy than one difference."""
    velocity = np.empty_like(positions_m)
    first_interval = (positions_m[1] - positions_m[0]) / (
        times_s[1] - times_s[0]
    )
    velocity[0] = first_interval
    for end in range(1, len(times_s)):
        start = max(0, end - window + 1)
        local_time = times_s[start : end + 1]
        local_position = positions_m[start : end + 1]
        centered = local_time - local_time.mean()
        velocity[end] = (
            centered[:, None] * local_position
        ).sum(axis=0) / np.square(centered).sum()
    return velocity


def rebuild_reference_velocities(
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    source: str,
    valid_rows: np.ndarray,
) -> np.ndarray:
    """Return channel values with reliable velocity rows rebuilt from position."""
    if source == REFERENCE_VELOCITY_TRACK_FIT:
        return np.asarray(values, dtype=np.float64).copy()

    rows = np.flatnonzero(valid_rows)
    times = np.asarray(times_s, dtype=np.float64)[rows]
    result = np.asarray(values, dtype=np.float64).copy()
    positions = result[rows[:, None], list(POSITION_IDX)]
    if source == REFERENCE_VELOCITY_POSITION_DIFFERENCE:
        velocity = _causal_position_difference(times, positions)
    elif source == REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE:
        velocity = _causal_local_linear_velocity(times, positions)
    else:
        raise ValueError(f"unknown reference velocity source {source!r}")
    result[rows[:, None], list(VELOCITY_IDX)] = velocity
    return result
