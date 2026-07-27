"""Shared output-time grids for training, metrics and inference.

The model head only emits ``pred_len`` state rows and one duration. This module is the
single place that gives those rows physical time meaning. Keeping the schedule here avoids
the former drift where the dataset used one clock while the physics loss and forecast
reconstructed another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from config import HORIZON_FULL, HORIZON_NORMALIZED, HORIZON_WINDOW, TSConfig


@dataclass(frozen=True)
class TimeGrid:
    """One output schedule relative to its observed anchor."""

    offsets_s: np.ndarray
    segment_durations_s: np.ndarray
    active: np.ndarray
    horizon_capped: bool


def _normalized_time_grid(final_time_s: float, config: TSConfig) -> TimeGrid:
    duration = max(float(final_time_s), 0.0)
    progress = np.arange(1, config.pred_len + 1, dtype=np.float64) / config.pred_len
    offsets = progress * duration
    return TimeGrid(
        offsets_s=offsets,
        segment_durations_s=np.diff(np.concatenate(([0.0], offsets))),
        active=np.ones(config.pred_len, dtype=bool),
        horizon_capped=False,
    )


def _fixed_time_grid(final_time_s: float, config: TSConfig) -> TimeGrid:
    duration = max(float(final_time_s), 0.0)
    step_ends = np.arange(1, config.pred_len + 1, dtype=np.float64) * config.dt_s
    active = (step_ends - config.dt_s) < duration
    capped_duration = min(duration, config.pred_len * config.dt_s)
    offsets = np.minimum(step_ends, capped_duration)
    offsets[~active] = capped_duration
    segment_durations = np.zeros(config.pred_len, dtype=np.float64)
    segment_durations[active] = np.diff(
        np.concatenate(([0.0], offsets[active]))
    )
    return TimeGrid(
        offsets_s=offsets,
        segment_durations_s=segment_durations,
        active=active,
        horizon_capped=duration > config.pred_len * config.dt_s,
    )


_NUMPY_GRID_BUILDERS = {
    HORIZON_NORMALIZED: _normalized_time_grid,
    HORIZON_FULL: _fixed_time_grid,
    HORIZON_WINDOW: _fixed_time_grid,
}


def output_time_grid(final_time_s: float, config: TSConfig) -> TimeGrid:
    """Return the complete padded output schedule for one trajectory."""
    return _NUMPY_GRID_BUILDERS[config.horizon_mode](final_time_s, config)


def _normalized_batch_time_grid(
    duration: torch.Tensor, config: TSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    durations = duration.expand(-1, config.pred_len) / config.pred_len
    return durations, torch.ones_like(durations, dtype=torch.bool)


def _fixed_batch_time_grid(
    duration: torch.Tensor, config: TSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.arange(
        config.pred_len,
        dtype=duration.dtype,
        device=duration.device,
    ).reshape(1, -1) * config.dt_s
    durations = (duration - starts).clamp(min=0.0, max=config.dt_s)
    return durations, durations > 0.0


_TORCH_GRID_BUILDERS = {
    HORIZON_NORMALIZED: _normalized_batch_time_grid,
    HORIZON_FULL: _fixed_batch_time_grid,
    HORIZON_WINDOW: _fixed_batch_time_grid,
}


def batch_time_grid(
    final_time_s: torch.Tensor,
    config: TSConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch durations and active mask, both ``[B,pred_len]``."""
    duration = final_time_s.clamp(min=0.0).reshape(-1, 1)
    return _TORCH_GRID_BUILDERS[config.horizon_mode](duration, config)


def _normalized_numpy_batch_time_grid(
    duration: np.ndarray, config: TSConfig
) -> tuple[np.ndarray, np.ndarray]:
    durations = np.broadcast_to(duration / config.pred_len, (len(duration), config.pred_len))
    return durations.copy(), np.ones_like(durations, dtype=bool)


def _fixed_numpy_batch_time_grid(
    duration: np.ndarray, config: TSConfig
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(config.pred_len, dtype=np.float64)[None, :] * config.dt_s
    durations = np.clip(duration - starts, 0.0, config.dt_s)
    return durations, durations > 0.0


_NUMPY_BATCH_GRID_BUILDERS = {
    HORIZON_NORMALIZED: _normalized_numpy_batch_time_grid,
    HORIZON_FULL: _fixed_numpy_batch_time_grid,
    HORIZON_WINDOW: _fixed_numpy_batch_time_grid,
}


def numpy_batch_time_grid(
    final_time_s: np.ndarray,
    config: TSConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy twin used by metric reporting."""
    duration = np.maximum(np.asarray(final_time_s, dtype=np.float64), 0.0)[:, None]
    return _NUMPY_BATCH_GRID_BUILDERS[config.horizon_mode](duration, config)


def numpy_inference_time_grid(
    predicted_final_time_s: np.ndarray,
    config: TSConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical clock carried by raw model nodes at inference.

    Only normalized mode uses the duration head to place state nodes. Full and window
    nodes always remain ``dt_s`` apart; their duration head is an auxiliary TTA estimate.
    """
    if config.horizon_mode == HORIZON_NORMALIZED:
        return numpy_batch_time_grid(predicted_final_time_s, config)
    shape = (len(predicted_final_time_s), config.pred_len)
    return np.full(shape, config.dt_s, dtype=np.float64), np.ones(shape, dtype=bool)
