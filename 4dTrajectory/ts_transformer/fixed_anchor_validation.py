"""Deterministic fixed-anchor metrics on a shared physical-time grid.

Training losses may live on model-specific native clocks. This module owns the deployment
validation contract instead: one ``L-1`` anchor per flight, truth sampled over its physical
remaining time, and every model prediction interpolated onto those same query timestamps.
It contains no training-loop or checkpoint logic so fixed/random training policies can share
the evaluator without depending on one another.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from channels import POSITION_IDX, VELOCITY_IDX
from config import HORIZON_NORMALIZED, TSConfig
from control_loss_components import last_reliable_terminal_velocity_target
from dataset import FlightSeries, Normalizer
from fixed_dt_supervision import build_fixed_dt_supervision
from time_grids import output_time_grid


def fixed_anchor_common_truth(
    series: Sequence[FlightSeries],
    config: TSConfig,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical truth, remaining durations and normalized query progress."""
    if points <= 1:
        raise ValueError("common-grid points must be greater than one")
    progress = np.arange(1, points + 1, dtype=np.float64) / points
    truth = np.empty((len(series), points, len(config.channels)), dtype=np.float32)
    durations = np.empty(len(series), dtype=np.float64)
    anchor = config.seq_len - 1
    for row, item in enumerate(series):
        if item.n_samples <= anchor:
            raise ValueError(
                f"flight {item.dataset_id!r} has no fixed L-1 anchor {anchor}"
            )
        duration = float(item.supervision_times[-1] - item.times[anchor])
        if duration <= 0.0:
            raise ValueError(
                f"flight {item.dataset_id!r} has no future after fixed anchor"
            )
        durations[row] = duration
        query_times = item.times[anchor] + progress * duration
        truth[row] = np.column_stack([
            np.interp(
                query_times,
                item.supervision_times,
                item.supervision_values[:, channel],
            )
            for channel in range(len(config.channels))
        ])
    return truth, durations, progress


def fixed_anchor_common_weights_and_terminal_velocity(
    series: Sequence[FlightSeries],
    config: TSConfig,
    progress: np.ndarray,
    durations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reliable common-grid weights and last observed terminal velocities."""

    points = len(progress)
    weights = np.empty((len(series), points, len(config.channels)), dtype=np.float32)
    anchor = config.seq_len - 1
    for row, item in enumerate(series):
        anchor_time = float(item.times[anchor])
        query_times = anchor_time + progress * durations[row]
        weights[row] = np.column_stack([
            np.interp(
                query_times,
                item.supervision_times,
                item.supervision_weights[:, channel],
            )
            for channel in range(len(config.channels))
        ])
        for channel in VELOCITY_IDX:
            valid = np.flatnonzero(item.supervision_weights[:, channel] > 0.0)
            if not len(valid):
                raise ValueError(
                    f"flight {item.dataset_id!r} has no reliable velocity supervision"
                )
            last = int(valid[-1])
            weights[row, query_times > item.supervision_times[last], channel] = 0.0
    fixed_dt = build_fixed_dt_supervision(
        series,
        [item.supervision_values for item in series],
        [(row, anchor) for row in range(len(series))],
        dt_s=config.dt_s,
    )
    anchor_states = torch.from_numpy(np.stack([
        item.values[anchor] for item in series
    ]).astype(np.float32, copy=False))
    terminal_velocity = last_reliable_terminal_velocity_target(
        anchor_states, fixed_dt
    ).numpy()
    return weights, terminal_velocity


def resample_prediction_to_physical_time(
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_final_time_s: float,
    config: TSConfig,
    query_offsets_s: np.ndarray,
    segment_durations_s: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Interpolate one prediction onto physical offsets after its observed anchor."""
    if config.horizon_mode == HORIZON_NORMALIZED:
        offsets = (
            np.cumsum(np.asarray(segment_durations_s, dtype=np.float64))
            if segment_durations_s is not None
            else output_time_grid(predicted_final_time_s, config).offsets_s
        )
        values = np.asarray(predicted_values)
        capped = False
    else:
        offsets = (
            np.arange(1, len(predicted_values) + 1, dtype=np.float64) * config.dt_s
        )
        closest = int(np.argmin(np.linalg.norm(
            np.asarray(predicted_values)[:, list(POSITION_IDX[:2])], axis=-1
        )))
        capped = closest == len(predicted_values) - 1
        offsets = offsets[: closest + 1]
        values = np.asarray(predicted_values)[: closest + 1]
    if not len(offsets):
        return np.broadcast_to(
            anchor_values, (len(query_offsets_s), len(anchor_values))
        ).copy(), False
    if len(offsets) != len(values):
        raise ValueError("prediction nodes and their physical clock must align")
    if np.any(np.diff(offsets) <= 0.0):
        raise ValueError("prediction clock must be strictly increasing")
    node_times = np.concatenate(([0.0], offsets))
    nodes = np.concatenate((np.asarray(anchor_values)[None, :], values), axis=0)
    sampled = np.column_stack([
        np.interp(query_offsets_s, node_times, nodes[:, channel])
        for channel in range(nodes.shape[1])
    ])
    return sampled, capped


def fixed_anchor_common_grid_metrics(
    series: Sequence[FlightSeries],
    config: TSConfig,
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_final_time_s: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    points: int,
    normalizer: Normalizer | None = None,
) -> dict[str, Any]:
    """Evaluate aligned fixed-anchor predictions in physical metres and seconds."""
    count = len(series)
    anchor_values = np.asarray(anchor_values)
    predicted_values = np.asarray(predicted_values)
    predicted_final_time_s = np.asarray(predicted_final_time_s, dtype=np.float64)
    segment_durations_s = np.asarray(segment_durations_s, dtype=np.float64)
    if anchor_values.shape != (count, len(config.channels)):
        raise ValueError("fixed-anchor values must be [B,C]")
    if predicted_values.ndim != 3 or predicted_values.shape[0] != count:
        raise ValueError("predicted values must be [B,N,C]")
    if predicted_values.shape[2] != len(config.channels):
        raise ValueError("predicted values use the wrong channel count")
    if predicted_final_time_s.shape != (count,):
        raise ValueError("predicted final times must be [B]")
    if segment_durations_s.shape != predicted_values.shape[:2]:
        raise ValueError("segment durations must align with predicted nodes")

    truth, true_duration_s, progress = fixed_anchor_common_truth(series, config, points)
    common = np.empty_like(truth)
    capped = np.zeros(count, dtype=bool)
    for index in range(count):
        common[index], capped[index] = resample_prediction_to_physical_time(
            anchor_values[index],
            predicted_values[index],
            float(predicted_final_time_s[index]),
            config,
            progress * true_duration_s[index],
            segment_durations_s[index],
        )
    delta = common[..., list(POSITION_IDX)] - truth[..., list(POSITION_IDX)]
    error = np.linalg.norm(delta, axis=-1)
    weights, terminal_velocity_target = (
        fixed_anchor_common_weights_and_terminal_velocity(
            series, config, progress, true_duration_s
        )
    )
    terminal_velocity_error = np.linalg.norm(
        common[:, -1, list(VELOCITY_IDX)] - terminal_velocity_target,
        axis=-1,
    )
    time_error = predicted_final_time_s - true_duration_s
    result = {
        "anchor": "fixed L-1",
        "metric_grid": "common physical-time grid",
        "points": points,
        "flights": count,
        "ade_m": float(error.mean()),
        "fde_m": float(error[:, -1].mean()),
        "terminal_velocity_error_mps": float(terminal_velocity_error.mean()),
        "final_time_mae_s": float(np.abs(time_error).mean()),
        "prediction_horizon_cap_rate": float(capped.mean()),
        "ade_per_flight_m": error.mean(axis=1),
        "fde_per_flight_m": error[:, -1],
        "terminal_velocity_error_per_flight_mps": terminal_velocity_error,
        "predicted_final_time_s": predicted_final_time_s,
        "true_final_time_s": true_duration_s,
    }
    if normalizer is not None:
        normalized_delta = (
            common.astype(np.float64) - truth.astype(np.float64)
        ) / np.asarray(normalizer.std, dtype=np.float64)
        weighted_squared = normalized_delta**2 * weights
        denominator = weights.sum(axis=(1, 2)).clip(min=1.0)
        dense_state = weighted_squared.sum(axis=(1, 2)) / denominator
        result["dense_state_loss"] = float(dense_state.mean())
        result["dense_state_loss_per_flight"] = dense_state
    return result
