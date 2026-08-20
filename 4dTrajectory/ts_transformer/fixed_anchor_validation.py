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

from arc_length_geometry import arc_length_geometry_metrics, arc_length_velocity_metrics
from channels import POSITION_IDX, VELOCITY_IDX
from config import HORIZON_NORMALIZED, TSConfig
from control.loss.components import last_reliable_terminal_velocity_target
from dataset import FlightSeries, Normalizer
from fixed_dt_supervision import build_fixed_dt_supervision
from terminal_state_loss import terminal_state_metrics_numpy
from time_grids import output_time_grid


CommonGridTruth = tuple[np.ndarray, np.ndarray, np.ndarray]


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


def fixed_anchor_arc_length_geometry_metrics(
    series: Sequence[FlightSeries],
    config: TSConfig,
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    terminal_velocity_target: np.ndarray,
    normalizer: Normalizer,
    *,
    points: int,
) -> dict[str, Any]:
    """Compare deployable ordered curves after horizontal arc-length alignment."""

    count = len(series)
    per_flight: list[dict[str, float]] = []
    per_flight_velocity: list[dict[str, float]] = []
    per_flight_terminal: list[dict[str, float]] = []
    terminal_velocity_error = np.empty(count, dtype=np.float64)
    anchor_index = config.seq_len - 1
    for row, item in enumerate(series):
        anchor_time = float(item.times[anchor_index])
        future = item.supervision_times > anchor_time
        reference_row_valid = future & np.all(
            item.supervision_weights[:, list(POSITION_IDX)] > 0.0,
            axis=1,
        )
        reference_values = np.concatenate(
            (
                anchor_values[row][None, :],
                item.supervision_values[reference_row_valid],
            ),
            axis=0,
        )
        predicted_curve = np.concatenate(
            (
                anchor_values[row][None, :],
                predicted_values[row],
            ),
            axis=0,
        )
        reference_positions = reference_values[:, list(POSITION_IDX)]
        predicted_positions = predicted_curve[:, list(POSITION_IDX)]
        per_flight.append(
            arc_length_geometry_metrics(
                predicted_positions,
                reference_positions,
                normalizer,
                points=points,
                position_end_weight=config.control_arc_position_end_weight,
            )
        )
        reference_velocity_valid = np.concatenate((
            np.ones(1, dtype=bool),
            np.all(
                item.supervision_weights[reference_row_valid][
                    :, list(VELOCITY_IDX)
                ] > 0.0,
                axis=1,
            ),
        ))
        per_flight_velocity.append(
            arc_length_velocity_metrics(
                predicted_positions,
                predicted_curve[:, list(VELOCITY_IDX)],
                reference_positions,
                reference_values[:, list(VELOCITY_IDX)],
                reference_velocity_valid,
                points=points,
            )
        )
        per_flight_terminal.append(
            terminal_state_metrics_numpy(
                predicted_values[row, -1],
                reference_values[-1],
                terminal_velocity_target[row],
                float(item.scenario.target.psi),
                coordinate_frame=config.coordinate_frame,
                cross_track_emphasis=(
                    config.control_arc_terminal_cross_track_emphasis
                ),
                vertical_emphasis=config.control_arc_terminal_vertical_emphasis,
            )
        )
        terminal_velocity_error[row] = np.linalg.norm(
            predicted_values[row, -1, list(VELOCITY_IDX)]
            - terminal_velocity_target[row]
        )

    result: dict[str, Any] = {
        "arc_length_metric_grid": "normalized horizontal arc length",
        "arc_length_points": points,
        "arc_length_geometry_loss": float(
            np.mean([block["loss"] for block in per_flight])
        ),
        "arc_length_geometry_unweighted_loss": float(
            np.mean([block["unweighted_loss"] for block in per_flight])
        ),
        "arc_length_distance_mean_m": float(
            np.mean([block["distance_mean_m"] for block in per_flight])
        ),
        "arc_length_predicted_horizontal_length_m": float(
            np.mean([block["predicted_horizontal_length_m"] for block in per_flight])
        ),
        "arc_length_reference_horizontal_length_m": float(
            np.mean([block["reference_horizontal_length_m"] for block in per_flight])
        ),
        "arc_length_path_length_ratio": float(
            np.mean([block["path_length_ratio"] for block in per_flight])
        ),
        "arc_length_path_length_log_error": float(
            np.mean([block["path_length_log_error"] for block in per_flight])
        ),
        "arc_length_horizontal_velocity_mae_mps": float(np.mean([
            block["horizontal_velocity_mae_mps"] for block in per_flight_velocity
        ])),
        "arc_length_horizontal_velocity_p95_mps": float(np.mean([
            block["horizontal_velocity_p95_mps"] for block in per_flight_velocity
        ])),
        "arc_length_horizontal_tangent_mean": float(np.mean([
            block["horizontal_tangent_mean"] for block in per_flight_velocity
        ])),
        "arc_length_horizontal_tangent_p95": float(np.mean([
            block["horizontal_tangent_p95"] for block in per_flight_velocity
        ])),
        "arc_length_horizontal_speed_mae_mps": float(np.mean([
            block["horizontal_speed_mae_mps"] for block in per_flight_velocity
        ])),
        "arc_length_horizontal_speed_p95_mps": float(np.mean([
            block["horizontal_speed_p95_mps"] for block in per_flight_velocity
        ])),
        "arc_length_vertical_velocity_mae_mps": float(np.mean([
            block["vertical_velocity_mae_mps"] for block in per_flight_velocity
        ])),
        "arc_length_vertical_velocity_p95_mps": float(np.mean([
            block["vertical_velocity_p95_mps"] for block in per_flight_velocity
        ])),
        "arc_length_velocity_valid_points": int(np.sum([
            block["velocity_valid_points"] for block in per_flight_velocity
        ])),
        "arc_length_horizontal_mean_m": float(
            np.mean([block["horizontal_mean_m"] for block in per_flight])
        ),
        "arc_length_horizontal_p95_m": float(
            np.mean([block["horizontal_p95_m"] for block in per_flight])
        ),
        "arc_length_vertical_mae_m": float(
            np.mean([block["vertical_mae_m"] for block in per_flight])
        ),
        "arc_length_vertical_p95_m": float(
            np.mean([block["vertical_p95_m"] for block in per_flight])
        ),
        "arc_length_terminal_position_m": float(
            np.mean([block["terminal_position_m"] for block in per_flight])
        ),
        "arc_length_terminal_velocity_error_mps": float(
            terminal_velocity_error.mean()
        ),
        "arc_length_terminal_position_runway_components_m": float(np.mean([
            block["position_runway_components_m"] for block in per_flight_terminal
        ])),
        "arc_length_terminal_velocity_runway_components_mps": float(np.mean([
            block["velocity_runway_components_mps"] for block in per_flight_terminal
        ])),
        "arc_length_geometry_loss_per_flight": np.asarray(
            [block["loss"] for block in per_flight], dtype=np.float64
        ),
        "arc_length_distance_mean_per_flight_m": np.asarray(
            [block["distance_mean_m"] for block in per_flight], dtype=np.float64
        ),
        "arc_length_path_length_log_error_per_flight": np.asarray(
            [block["path_length_log_error"] for block in per_flight],
            dtype=np.float64,
        ),
        "arc_length_horizontal_velocity_mae_per_flight_mps": np.asarray(
            [
                block["horizontal_velocity_mae_mps"]
                for block in per_flight_velocity
            ],
            dtype=np.float64,
        ),
        "arc_length_vertical_velocity_mae_per_flight_mps": np.asarray(
            [
                block["vertical_velocity_mae_mps"]
                for block in per_flight_velocity
            ],
            dtype=np.float64,
        ),
        "arc_length_terminal_position_per_flight_m": np.asarray(
            [block["terminal_position_m"] for block in per_flight], dtype=np.float64
        ),
        "arc_length_terminal_velocity_error_per_flight_mps": terminal_velocity_error,
    }
    for key in (
        "along_position_abs_m",
        "cross_position_abs_m",
        "vertical_position_abs_m",
        "along_velocity_abs_mps",
        "cross_velocity_abs_mps",
        "vertical_velocity_abs_mps",
    ):
        result[f"arc_length_terminal_{key}"] = float(np.mean([
            block[key] for block in per_flight_terminal
        ]))
    return result


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


def _fixed_anchor_common_position_arrays(
    series: Sequence[FlightSeries],
    config: TSConfig,
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_final_time_s: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    points: int,
    common_truth: CommonGridTruth | None = None,
):
    """Validate and materialize only the arrays required by formal 3D ADE."""
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
    if not (
        np.isfinite(anchor_values).all()
        and np.isfinite(predicted_values).all()
        and np.isfinite(predicted_final_time_s).all()
        and np.isfinite(segment_durations_s).all()
    ):
        raise ValueError("common-grid prediction inputs must be finite")
    if np.any(predicted_final_time_s <= 0.0) or np.any(segment_durations_s <= 0.0):
        raise ValueError("common-grid prediction clocks must be positive")

    if common_truth is None:
        truth, true_duration_s, progress = fixed_anchor_common_truth(
            series, config, points
        )
    else:
        truth, true_duration_s, progress = common_truth
        if truth.shape != (count, points, len(config.channels)):
            raise ValueError("cached common-grid truth has the wrong state shape")
        if true_duration_s.shape != (count,):
            raise ValueError("cached common-grid truth has the wrong duration shape")
        if progress.shape != (points,):
            raise ValueError("cached common-grid truth has the wrong progress shape")
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
    return truth, common, true_duration_s, progress, capped, error


def fixed_anchor_common_grid_ade_metrics(
    series: Sequence[FlightSeries],
    config: TSConfig,
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_final_time_s: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    points: int,
    common_truth: CommonGridTruth | None = None,
) -> dict[str, Any]:
    """Lean formal selector: common true-time per-flight 3D ADE and FDE only."""
    (
        _truth,
        _common,
        true_duration_s,
        _progress,
        capped,
        error,
    ) = _fixed_anchor_common_position_arrays(
        series,
        config,
        anchor_values,
        predicted_values,
        predicted_final_time_s,
        segment_durations_s,
        points=points,
        common_truth=common_truth,
    )
    return _common_grid_ade_result(
        len(series),
        points,
        predicted_final_time_s,
        true_duration_s,
        capped,
        error,
    )


def _common_grid_ade_result(
    flights: int,
    points: int,
    predicted_final_time_s: np.ndarray,
    true_duration_s: np.ndarray,
    capped: np.ndarray,
    error: np.ndarray,
) -> dict[str, Any]:
    predicted_final_time_s = np.asarray(predicted_final_time_s, dtype=np.float64)
    time_error = predicted_final_time_s - true_duration_s
    return {
        "anchor": "fixed L-1",
        "metric_grid": "common true physical-time grid",
        "points": points,
        "flights": flights,
        "ade_m": float(error.mean()),
        "fde_m": float(error[:, -1].mean()),
        "final_time_mae_s": float(np.abs(time_error).mean()),
        "prediction_horizon_cap_rate": float(capped.mean()),
        "ade_per_flight_m": error.mean(axis=1),
        "fde_per_flight_m": error[:, -1],
        "predicted_final_time_s": predicted_final_time_s,
        "true_final_time_s": true_duration_s,
    }


def _signed_spread(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(values)
    return {
        "mean_signed": float(values.mean()),
        "mean_abs": float(magnitude.mean()),
        "p95_abs": float(np.percentile(magnitude, 95)),
        "max_abs": float(magnitude.max()),
    }


def _magnitude_spread(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def fixed_anchor_common_grid_report_metrics(
    series: Sequence[FlightSeries],
    config: TSConfig,
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_final_time_s: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    points: int,
) -> dict[str, Any]:
    """Formal post-fit metrics: one common true-time grid and interpretable components."""
    truth, common, true_duration_s, progress, capped, error = (
        _fixed_anchor_common_position_arrays(
            series,
            config,
            anchor_values,
            predicted_values,
            predicted_final_time_s,
            segment_durations_s,
            points=points,
        )
    )
    delta = common[..., list(POSITION_IDX)] - truth[..., list(POSITION_IDX)]
    horizontal = np.linalg.norm(delta[..., :2], axis=-1)
    anchor_positions = np.asarray(anchor_values, dtype=np.float64)[
        :, None, list(POSITION_IDX)
    ]
    truth_with_anchor = np.concatenate((anchor_positions, truth[..., list(POSITION_IDX)]), axis=1)
    tangent = np.gradient(truth_with_anchor[..., :2], axis=1)[:, 1:]
    norm = np.linalg.norm(tangent, axis=-1)
    unit = np.zeros_like(tangent)
    moving = norm > 1e-9
    unit[moving] = tangent[moving] / norm[moving, None]
    along = delta[..., 0] * unit[..., 0] + delta[..., 1] * unit[..., 1]
    cross = -delta[..., 0] * unit[..., 1] + delta[..., 1] * unit[..., 0]
    vertical = delta[..., 2]
    predicted_final_time_s = np.asarray(predicted_final_time_s, dtype=np.float64)
    time_error = predicted_final_time_s - true_duration_s
    per_flight_ade = error.mean(axis=1)
    per_flight_fde = error[:, -1]
    # FDE above asks "where is the prediction at the true arrival time?".  Separately,
    # sample the prediction at its own estimated arrival time and compare that position
    # with the observed output endpoint. Keeping both prevents clock and geometry errors
    # from being conflated in diagnosis.
    predicted_arrival = np.empty((len(series), len(POSITION_IDX)), dtype=np.float64)
    for index in range(len(series)):
        arrival, _ = resample_prediction_to_physical_time(
            anchor_values[index],
            predicted_values[index],
            float(predicted_final_time_s[index]),
            config,
            np.asarray([predicted_final_time_s[index]], dtype=np.float64),
            segment_durations_s[index],
        )
        predicted_arrival[index] = arrival[0, list(POSITION_IDX)]
    endpoint_error = np.linalg.norm(
        predicted_arrival - truth[:, -1, list(POSITION_IDX)], axis=-1
    )
    return {
        "anchor": "fixed L-1",
        "metric_grid": "common true physical-time grid",
        "points": points,
        "flights": len(series),
        "ade_m": float(per_flight_ade.mean()),
        "fde_m": float(per_flight_fde.mean()),
        "ade_distribution_m": _magnitude_spread(per_flight_ade),
        "fde_distribution_m": _magnitude_spread(per_flight_fde),
        "arrival_endpoint_error_m": _magnitude_spread(endpoint_error),
        "horizontal_m": _magnitude_spread(horizontal.mean(axis=1)),
        "along_track_m": _signed_spread(along),
        "cross_track_m": _signed_spread(cross),
        "vertical_m": _signed_spread(vertical),
        "final_time_s": {
            "mae": float(np.abs(time_error).mean()),
            "rmse": float(np.sqrt(np.mean(time_error**2))),
            "p95_abs": float(np.percentile(np.abs(time_error), 95)),
            "mean_signed": float(time_error.mean()),
        },
        "prediction_horizon_cap_rate": float(capped.mean()),
        "invalid_flights": 0,
        "by_progress": [
            {
                "progress": float(progress[index]),
                "mean_m": float(error[:, index].mean()),
                "p95_m": float(np.percentile(error[:, index], 95)),
            }
            for index in range(points)
        ],
    }


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
    """Full post-fit diagnostics on the formal common physical-time grid."""
    truth, common, true_duration_s, progress, capped, error = (
        _fixed_anchor_common_position_arrays(
            series,
            config,
            anchor_values,
            predicted_values,
            predicted_final_time_s,
            segment_durations_s,
            points=points,
        )
    )
    result = _common_grid_ade_result(
        len(series),
        points,
        predicted_final_time_s,
        true_duration_s,
        capped,
        error,
    )
    weights, terminal_velocity_target = (
        fixed_anchor_common_weights_and_terminal_velocity(
            series, config, progress, true_duration_s
        )
    )
    terminal_velocity_error = np.linalg.norm(
        common[:, -1, list(VELOCITY_IDX)] - terminal_velocity_target,
        axis=-1,
    )
    result.update({
        "terminal_velocity_error_mps": float(terminal_velocity_error.mean()),
        "terminal_velocity_error_per_flight_mps": terminal_velocity_error,
    })
    if normalizer is not None:
        result.update(
            fixed_anchor_arc_length_geometry_metrics(
                series,
                config,
                anchor_values,
                predicted_values,
                terminal_velocity_target,
                normalizer,
                points=points,
            )
        )
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
