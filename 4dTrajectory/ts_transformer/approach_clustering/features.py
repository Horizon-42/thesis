"""Geometry-only features used to group approaches without speed information."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dataset import FlightSeries


def horizontal_arc_feature(
    series: FlightSeries,
    *,
    anchor_index: int,
    points: int,
) -> np.ndarray:
    """Flatten an equal-horizontal-arc resampling of the future E/N path."""
    horizontal = np.asarray(
        series.supervision_values[anchor_index:, :2], dtype=np.float64
    )
    increments = np.linalg.norm(np.diff(horizontal, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    unique = np.concatenate(([True], np.diff(cumulative) > 1e-9))
    cumulative = cumulative[unique]
    horizontal = horizontal[unique]
    target = np.linspace(0.0, cumulative[-1], points)
    sampled = np.column_stack([
        np.interp(target, cumulative, horizontal[:, axis]) for axis in range(2)
    ])
    return sampled.reshape(-1)


def horizontal_arc_features(
    series: Sequence[FlightSeries],
    *,
    anchor_index: int,
    points: int,
) -> np.ndarray:
    return np.stack([
        horizontal_arc_feature(item, anchor_index=anchor_index, points=points)
        for item in series
    ])
