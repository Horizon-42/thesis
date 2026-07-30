"""Ragged reference targets for fixed-physical-time control state loss.

The model still emits ``N`` non-uniform control segments.  This module owns the separate
supervision clock: every complete ``config.dt_s`` interval after the observed anchor.  The
arrival builder has already resampled measured channel values onto that exact grid, so the
targets below are gathered by timestamp rather than interpolated a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FixedDTControlSupervision:
    """Padded dense targets; ``valid`` marks each flight's unpadded prefix."""

    query_offsets_s: torch.Tensor  # [B,M], physical seconds after anchor
    states: torch.Tensor           # [B,M,C], normalized reference channels
    weights: torch.Tensor          # [B,M,C]
    valid: torch.Tensor            # [B,M], bool

    def to(self, device: torch.device) -> "FixedDTControlSupervision":
        return FixedDTControlSupervision(
            query_offsets_s=self.query_offsets_s.to(device),
            states=self.states.to(device),
            weights=self.weights.to(device),
            valid=self.valid.to(device),
        )


def _sample(
    series,
    normalized_values: np.ndarray,
    anchor: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_time = float(series.times[anchor])
    remaining_s = float(series.supervision_times[-1] - anchor_time)
    count = int(np.floor((remaining_s + 1e-7 * dt_s) / dt_s))
    offsets = np.arange(1, count + 1, dtype=np.float64) * dt_s
    if not count:
        channels = normalized_values.shape[1]
        return (
            offsets,
            np.empty((0, channels), dtype=np.float32),
            np.empty((0, channels), dtype=np.float32),
        )

    query_times = anchor_time + offsets
    source_times_all = np.asarray(series.supervision_times, dtype=np.float64)
    if not len(source_times_all):
        raise ValueError("fixed-dt control supervision has no reference rows")
    insertion = np.searchsorted(source_times_all, query_times, side="left")
    left = np.clip(insertion - 1, 0, len(source_times_all) - 1)
    right = np.clip(insertion, 0, len(source_times_all) - 1)
    choose_right = (
        np.abs(source_times_all[right] - query_times)
        < np.abs(source_times_all[left] - query_times)
    )
    source_indices = np.where(choose_right, right, left)
    source_times = source_times_all[source_indices]
    tolerance = max(1e-6, abs(dt_s) * 1e-6)
    if not np.allclose(source_times, query_times, rtol=0.0, atol=tolerance):
        mismatch = int(np.flatnonzero(np.abs(source_times - query_times) > tolerance)[0])
        raise ValueError(
            "fixed-dt control supervision requires the existing regular reference grid; "
            f"query {query_times[mismatch]:.9g}s has nearest row "
            f"{source_times[mismatch]:.9g}s"
        )
    return (
        offsets,
        normalized_values[source_indices].astype(np.float32, copy=False),
        series.supervision_weights[source_indices].astype(np.float32, copy=False),
    )


def build_fixed_dt_supervision(
    series: Sequence,
    normalized_values: Sequence[np.ndarray],
    sample_indices: Sequence[tuple[int, int]],
    *,
    dt_s: float,
) -> FixedDTControlSupervision:
    """Gather and pad a batch without imposing a semantic trajectory-length cap."""
    rows = [
        _sample(series[series_index], normalized_values[series_index], anchor, dt_s)
        for series_index, anchor in sample_indices
    ]
    if not rows:
        raise ValueError("fixed-dt control supervision requires a non-empty batch")
    batch = len(rows)
    max_points = max(len(offsets) for offsets, _states, _weights in rows)
    channels = normalized_values[0].shape[1]
    offsets = np.zeros((batch, max_points), dtype=np.float64)
    states = np.zeros((batch, max_points, channels), dtype=np.float32)
    weights = np.zeros_like(states)
    valid = np.zeros((batch, max_points), dtype=bool)
    for row, (row_offsets, row_states, row_weights) in enumerate(rows):
        count = len(row_offsets)
        offsets[row, :count] = row_offsets
        states[row, :count] = row_states
        weights[row, :count] = row_weights
        valid[row, :count] = True
    return FixedDTControlSupervision(
        query_offsets_s=torch.from_numpy(offsets),
        states=torch.from_numpy(states),
        weights=torch.from_numpy(weights),
        valid=torch.from_numpy(valid),
    )
