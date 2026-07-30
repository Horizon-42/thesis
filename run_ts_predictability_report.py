#!/usr/bin/env python
"""Validation-only diagnostics for pooled terminal-trajectory predictors.

The report answers three predeclared questions:

1. How does error vary with remaining time, airport, runway and geometric route type?
2. How dispersed are the futures of flights with similar observed anchor histories?
3. How much oracle coverage is gained by stochastic or retrieved multi-candidate outputs?

Outer-test is intentionally unsupported. Multi-candidate ``minADE/minFDE`` values are
oracle set-coverage diagnostics, not deployable selection accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aeroviz-ts-report-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from channels import POSITION_IDX  # noqa: E402
from config import (  # noqa: E402
    HORIZON_FULL, HORIZON_NORMALIZED, HORIZON_WINDOW, TSConfig,
    uses_control_dynamics,
)
from control_prediction_adapters import deployable_control_prediction  # noqa: E402
from dataset import (  # noqa: E402
    FlightSeries,
    arrival_data_provenance,
    build_series,
    dynamics_arrays,
    load_flight_dicts,
    require_matching_data_provenance,
)
from fixed_anchor_validation import (  # noqa: E402
    fixed_anchor_common_truth,
    resample_prediction_to_physical_time,
)
from metrics import raw_kinematic_metrics  # noqa: E402
from models import resolve_device  # noqa: E402
from train import (  # noqa: E402
    FIT_EVALUATION_NAME,
    FIT_EVALUATION_SCHEMA,
    control_rollout_channels,
    load_checkpoint,
    usable_series,
)

REPORT_SCHEMA = "ts-pooled-predictability-report-v4-control-diagnostics-validation-only"
DEFAULT_REMAINING_TIME_EDGES_S = (0.0, 30.0, 60.0, 120.0, 180.0, 300.0, 450.0, 600.0)
ROUTE_TYPES = ("straight-in", "single-turn", "vectoring", "holding-like")


@dataclass
class LoadedRun:
    label: str
    checkpoint: Path
    model: torch.nn.Module
    config: TSConfig
    normalizer: Any
    payload: dict[str, Any]


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--checkpoint requires LABEL=/path/to/checkpoint.pt")
    return label.strip(), Path(raw_path).expanduser().resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_trajectory(future_positions: np.ndarray) -> str:
    """Geometry proxy for route complexity; not an ATC intent label."""
    horizontal = np.asarray(future_positions, dtype=np.float64)[..., :2]
    delta = np.diff(horizontal, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    moving = lengths > 1.0
    if moving.sum() < 2:
        return "straight-in"
    delta = delta[moving]
    lengths = lengths[moving]
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    total_turn_deg = float(np.degrees(np.abs(np.diff(heading)).sum()))
    direct = float(np.linalg.norm(horizontal[-1] - horizontal[0]))
    tortuosity = float(lengths.sum() / max(direct, 1.0))
    if total_turn_deg >= 300.0 or tortuosity >= 2.5:
        return "holding-like"
    if total_turn_deg >= 120.0 or tortuosity >= 1.6:
        return "vectoring"
    if total_turn_deg >= 35.0:
        return "single-turn"
    return "straight-in"


def common_truth(
    series: Sequence[FlightSeries], config: TSConfig, points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    truth, durations, progress = fixed_anchor_common_truth(series, config, points)
    route_types = [
        classify_trajectory(row[:, list(POSITION_IDX)]) for row in truth
    ]
    return truth, durations, progress, route_types


def history_tensor(
    series: Sequence[FlightSeries], config: TSConfig, normalizer: Any
) -> np.ndarray:
    anchor = config.seq_len - 1
    return np.stack([
        normalizer.encode(item.values[anchor - config.seq_len + 1 : anchor + 1])
        for item in series
    ]).astype(np.float32)


def resample_prediction(
    anchor_values: np.ndarray,
    predicted_values: np.ndarray,
    predicted_final_time_s: float,
    config: TSConfig,
    query_offsets_s: np.ndarray,
    segment_durations_s: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    return resample_prediction_to_physical_time(
        anchor_values,
        predicted_values,
        predicted_final_time_s,
        config,
        query_offsets_s,
        segment_durations_s,
    )


def _one_pass_nodes(
    model: torch.nn.Module, histories: torch.Tensor, config: TSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(histories)
    return output.states, output.final_time_s


def _recursive_window_nodes(
    model: torch.nn.Module, histories: torch.Tensor, config: TSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    current = histories
    chunks: list[torch.Tensor] = []
    first_final_time: torch.Tensor | None = None
    produced = 0
    while produced < config.full_horizon_steps:
        output = model(current)
        if first_final_time is None:
            first_final_time = output.final_time_s
        chunks.append(output.states)
        produced += output.states.shape[1]
        current = torch.cat((current, output.states), dim=1)[:, -config.seq_len :]
    assert first_final_time is not None
    return torch.cat(chunks, dim=1)[:, : config.full_horizon_steps], first_final_time


_NODE_PREDICTORS = {
    HORIZON_NORMALIZED: _one_pass_nodes,
    HORIZON_FULL: _one_pass_nodes,
    HORIZON_WINDOW: _recursive_window_nodes,
}


def batch_dynamics_tensors(
    series: Sequence[FlightSeries], config: TSConfig, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build the exact per-flight conditioning/rollout tensors used by training."""
    anchor = config.seq_len - 1
    rows = [dynamics_arrays(item, anchor) for item in series]
    return {
        name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device)
        for name in rows[0]
    }


def predict_batch_nodes(
    run: LoadedRun,
    histories: torch.Tensor,
    series: Sequence[FlightSeries],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return physical nodes, final time, explicit durations and optional controls."""
    if uses_control_dynamics(run.config.prediction_output):
        dynamics = batch_dynamics_tensors(series, run.config, device)
        output = deployable_control_prediction(run.model(histories, dynamics))
        channels, _geodetic = control_rollout_channels(output, dynamics, run.config)
        return (
            channels.detach().cpu().numpy().astype(np.float32),
            output.final_time_s.detach().cpu().numpy().astype(np.float64),
            output.segment_durations.detach().cpu().numpy().astype(np.float64),
            output.controls.detach().cpu().numpy().astype(np.float64),
        )

    states, final_time = _NODE_PREDICTORS[run.config.horizon_mode](
        run.model, histories, run.config
    )
    physical = run.normalizer.decode(
        states.detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    final_time_array = final_time.detach().cpu().numpy().astype(np.float64)
    if run.config.horizon_mode == HORIZON_NORMALIZED:
        durations = np.broadcast_to(
            final_time_array[:, None] / physical.shape[1], physical.shape[:2]
        ).copy()
    else:
        durations = np.full(physical.shape[:2], run.config.dt_s, dtype=np.float64)
    return physical, final_time_array, durations, None


def control_distribution_statistics(
    controls: np.ndarray,
    durations_s: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    """Unweighted validation distributions required by the control experiment guide.

    Bounds may vary by flight. A value is considered near a bound when it lies within one
    percent of that flight's allowed range; the threshold is recorded with the result.
    """
    controls = np.asarray(controls, dtype=np.float64)
    durations_s = np.asarray(durations_s, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if controls.ndim != 3 or controls.shape[-1] != 3:
        raise ValueError("controls must be [B,N,3]")
    if durations_s.shape != controls.shape[:2]:
        raise ValueError("durations must align with control segments")
    if lower.shape != (controls.shape[0], 3) or upper.shape != lower.shape:
        raise ValueError("control bounds must be aligned [B,3] arrays")
    if not (
        np.isfinite(controls).all()
        and np.isfinite(durations_s).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    ):
        raise ValueError("control diagnostics require finite values")
    if np.any(upper <= lower) or np.any(durations_s <= 0.0):
        raise ValueError("control bounds and durations must be strictly positive intervals")

    bound_fraction = 0.01
    ranges = upper - lower
    changes = np.abs(np.diff(controls, axis=1))
    units = ("N", "rad", "1")
    names = ("thrust_N", "bank_rad", "load_factor")
    channel_statistics: dict[str, dict[str, float | str]] = {}
    for channel, (name, unit) in enumerate(zip(names, units)):
        values = controls[..., channel].reshape(-1)
        adjacent = changes[..., channel].reshape(-1)
        low_distance = controls[..., channel] - lower[:, None, channel]
        high_distance = upper[:, None, channel] - controls[..., channel]
        tolerance = bound_fraction * ranges[:, None, channel]
        channel_statistics[name] = {
            "unit": unit,
            "p5": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "near_lower_fraction": float(np.mean(low_distance <= tolerance)),
            "near_upper_fraction": float(np.mean(high_distance <= tolerance)),
            "adjacent_abs_change_median": float(np.median(adjacent)),
            "adjacent_abs_change_p95": float(np.percentile(adjacent, 95)),
            "adjacent_abs_change_max": float(np.max(adjacent)),
        }
    flat_durations = durations_s.reshape(-1)
    return {
        "near_bound_range_fraction": bound_fraction,
        "channels": channel_statistics,
        "durations_s": {
            "min": float(np.min(flat_durations)),
            "p1": float(np.percentile(flat_durations, 1)),
            "median": float(np.median(flat_durations)),
            "p99": float(np.percentile(flat_durations, 99)),
            "max": float(np.max(flat_durations)),
        },
    }


def displacement_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    delta = predicted[..., list(POSITION_IDX)] - truth[..., list(POSITION_IDX)]
    return np.linalg.norm(delta, axis=-1)


def run_deterministic(
    run: LoadedRun,
    series: Sequence[FlightSeries],
    truth: np.ndarray,
    true_duration_s: np.ndarray,
    progress: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    run.model.to(device).eval()
    histories = history_tensor(series, run.config, run.normalizer)
    anchors = np.stack([item.values[run.config.seq_len - 1] for item in series])
    common: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    final_times: list[np.ndarray] = []
    raw_durations: list[np.ndarray] = []
    raw_controls: list[np.ndarray] = []
    capped: list[bool] = []
    with torch.no_grad():
        for start in range(0, len(series), batch_size):
            stop = min(start + batch_size, len(series))
            decoded, predicted_time, durations, controls = predict_batch_nodes(
                run,
                torch.from_numpy(histories[start:stop]).to(device),
                series[start:stop],
                device,
            )
            raw_values.append(decoded)
            final_times.append(predicted_time)
            raw_durations.append(durations)
            if controls is not None:
                raw_controls.append(controls)
            for local, absolute in enumerate(range(start, stop)):
                sampled, was_capped = resample_prediction(
                    anchors[absolute],
                    decoded[local],
                    float(predicted_time[local]),
                    run.config,
                    progress * true_duration_s[absolute],
                    durations[local],
                )
                common.append(sampled)
                capped.append(was_capped)
    common_array = np.stack(common).astype(np.float32)
    raw_array = np.concatenate(raw_values)
    final_time_array = np.concatenate(final_times)
    durations = np.concatenate(raw_durations)
    valid = np.ones_like(durations, dtype=bool)
    kinematics = raw_kinematic_metrics(
        anchors, raw_array, durations, valid_segments=valid
    )
    control_diagnostics = None
    if raw_controls:
        dynamics = [dynamics_arrays(item, run.config.seq_len - 1) for item in series]
        control_diagnostics = control_distribution_statistics(
            np.concatenate(raw_controls),
            durations,
            np.stack([row["control_lower"] for row in dynamics]),
            np.stack([row["control_upper"] for row in dynamics]),
        )
    error = displacement_errors(common_array, truth)
    return {
        "common_prediction": common_array,
        "error_grid_m": error,
        "ade_per_flight_m": error.mean(axis=1),
        "fde_per_flight_m": error[:, -1],
        "predicted_final_time_s": final_time_array,
        "horizon_capped": np.asarray(capped, dtype=bool),
        "raw_kinematics": kinematics,
        "control_diagnostics": control_diagnostics,
    }


def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_coverage(
    run: LoadedRun,
    series: Sequence[FlightSeries],
    truth: np.ndarray,
    true_duration_s: np.ndarray,
    progress: np.ndarray,
    device: torch.device,
    batch_size: int,
    sample_count: int,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Oracle minADE/minFDE for stochastic forward-pass candidate sets."""
    run.model.to(device)
    enable_mc_dropout(run.model)
    histories = history_tensor(series, run.config, run.normalizer)
    anchors = np.stack([item.values[run.config.seq_len - 1] for item in series])
    sums = {k: {"minade": 0.0, "minfde": 0.0} for k in k_values}
    per_flight_minade = np.empty(len(series), dtype=np.float64)
    per_flight_minfde = np.empty(len(series), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(series), batch_size):
            stop = min(start + batch_size, len(series))
            tensor = torch.from_numpy(histories[start:stop]).to(device)
            best_ade = np.full(stop - start, np.inf)
            best_fde = np.full(stop - start, np.inf)
            for draw in range(1, sample_count + 1):
                decoded, predicted_time, durations, _controls = predict_batch_nodes(
                    run, tensor, series[start:stop], device
                )
                sampled = []
                for local, absolute in enumerate(range(start, stop)):
                    candidate, _capped = resample_prediction(
                        anchors[absolute], decoded[local], float(predicted_time[local]),
                        run.config, progress * true_duration_s[absolute], durations[local],
                    )
                    sampled.append(candidate)
                error = displacement_errors(np.stack(sampled), truth[start:stop])
                best_ade = np.minimum(best_ade, error.mean(axis=1))
                best_fde = np.minimum(best_fde, error[:, -1])
                if draw in sums:
                    sums[draw]["minade"] += float(best_ade.sum())
                    sums[draw]["minfde"] += float(best_fde.sum())
            per_flight_minade[start:stop] = best_ade
            per_flight_minfde[start:stop] = best_fde
    run.model.eval()
    return {
        "method": "MC dropout oracle candidate-set coverage",
        "sample_count": sample_count,
        "curve": [
            {
                "k": k,
                "minade_m": sums[k]["minade"] / len(series),
                "minfde_m": sums[k]["minfde"] / len(series),
            }
            for k in k_values
        ],
        "per_flight_minade_m": per_flight_minade,
        "per_flight_minfde_m": per_flight_minfde,
    }


def anchor_descriptor(
    series: Sequence[FlightSeries], config: TSConfig, normalizer: Any
) -> np.ndarray:
    """Five history snapshots, retaining absolute threshold-relative state."""
    lags = np.unique(np.linspace(0, config.seq_len - 1, 5).round().astype(int))
    anchor = config.seq_len - 1
    rows = []
    for item in series:
        encoded = normalizer.encode(item.values[anchor - lags])
        rows.append(encoded.reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def retrieval_coverage(
    train_series: Sequence[FlightSeries],
    validation_series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Any,
    validation_truth: np.ndarray,
    points: int,
    k_values: Sequence[int],
) -> dict[str, Any]:
    train_truth, _duration, progress, _types = common_truth(train_series, config, points)
    train_positions = train_truth[..., list(POSITION_IDX)].astype(np.float32)
    validation_positions = validation_truth[..., list(POSITION_IDX)].astype(np.float32)
    train_descriptor = anchor_descriptor(train_series, config, normalizer)
    validation_descriptor = anchor_descriptor(validation_series, config, normalizer)
    maximum_k = max(k_values)
    search = NearestNeighbors(n_neighbors=maximum_k, algorithm="auto", n_jobs=-1)
    search.fit(train_descriptor)
    distances, indices = search.kneighbors(validation_descriptor)

    curve_sums = {k: {"minade": 0.0, "minfde": 0.0} for k in k_values}
    dispersion_progress = np.zeros(points, dtype=np.float64)
    per_flight_dispersion = np.empty(len(validation_series), dtype=np.float64)
    per_flight_minade = np.empty(len(validation_series), dtype=np.float64)
    per_flight_minfde = np.empty(len(validation_series), dtype=np.float64)
    centroid_ade_sum = 0.0
    example_index = 0
    example_score = -np.inf
    example_candidates = None
    for row in range(len(validation_series)):
        candidates = train_positions[indices[row]]
        error = np.linalg.norm(candidates - validation_positions[row][None, ...], axis=-1)
        candidate_ade = error.mean(axis=1)
        candidate_fde = error[:, -1]
        for k in k_values:
            curve_sums[k]["minade"] += float(candidate_ade[:k].min())
            curve_sums[k]["minfde"] += float(candidate_fde[:k].min())
        per_flight_minade[row] = candidate_ade.min()
        per_flight_minfde[row] = candidate_fde.min()
        centroid = candidates.mean(axis=0)
        centroid_ade_sum += float(
            np.linalg.norm(centroid - validation_positions[row], axis=-1).mean()
        )
        dispersion = np.sqrt(
            np.mean(np.sum((candidates - centroid[None, ...]) ** 2, axis=-1), axis=0)
        )
        dispersion_progress += dispersion
        per_flight_dispersion[row] = float(dispersion.mean())
        if per_flight_dispersion[row] > example_score:
            example_score = per_flight_dispersion[row]
            example_index = row
            example_candidates = candidates.copy()

    count = len(validation_series)
    return {
        "method": "nearest-history retrieval oracle candidate-set coverage",
        "descriptor": "5 equally spaced standardized threshold-relative history states",
        "curve": [
            {
                "k": k,
                "minade_m": curve_sums[k]["minade"] / count,
                "minfde_m": curve_sums[k]["minfde"] / count,
            }
            for k in k_values
        ],
        "centroid_ade_m": centroid_ade_sum / count,
        "mean_dispersion_by_progress_m": dispersion_progress / count,
        "per_flight_dispersion_m": per_flight_dispersion,
        "per_flight_minade_m": per_flight_minade,
        "per_flight_minfde_m": per_flight_minfde,
        "neighbor_distance_mean": float(distances.mean()),
        "example_index": example_index,
        "example_candidates": example_candidates,
        "progress": progress,
    }


def summary_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "n": int(len(finite)),
    }


def grouped_rows(
    flight_rows: Sequence[dict[str, Any]], group_key: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in flight_rows:
        grouped.setdefault((row["model"], str(row[group_key])), []).append(row)
    result = []
    for (model, group), rows in sorted(grouped.items()):
        result.append({
            "model": model,
            group_key: group,
            "flights": len(rows),
            "ade_mean_m": float(np.mean([row["ade_m"] for row in rows])),
            "ade_p95_m": float(np.percentile([row["ade_m"] for row in rows], 95)),
            "fde_mean_m": float(np.mean([row["fde_m"] for row in rows])),
            "time_mae_s": float(np.mean(np.abs([row["final_time_error_s"] for row in rows]))),
        })
    return result


def remaining_time_rows(
    model_results: dict[str, dict[str, Any]],
    durations: np.ndarray,
    progress: np.ndarray,
    edges: Sequence[float] = DEFAULT_REMAINING_TIME_EDGES_S,
) -> list[dict[str, Any]]:
    result = []
    remaining = durations[:, None] * (1.0 - progress[None, :])
    upper_edges = (*edges[1:], float("inf"))
    for label, model in model_results.items():
        error = model["error_grid_m"]
        for lower, upper in zip(edges, upper_edges):
            valid = (remaining >= lower) & (remaining < upper)
            values = error[valid]
            if not len(values):
                continue
            result.append({
                "model": label,
                "remaining_time_bin_s": f"{lower:g}–{'∞' if np.isinf(upper) else f'{upper:g}'}",
                "lower_s": lower,
                "upper_s": upper,
                "mean_error_m": float(values.mean()),
                "p95_error_m": float(np.percentile(values, 95)),
                "points": int(len(values)),
            })
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, plots: Path, name: str) -> None:
    figure.tight_layout()
    figure.savefig(plots / f"{name}.png", dpi=180)
    figure.savefig(plots / f"{name}.svg")
    plt.close(figure)


def plot_reports(
    output: Path,
    model_summary: Sequence[dict[str, Any]],
    fit_summary: Sequence[dict[str, Any]],
    remaining: Sequence[dict[str, Any]],
    airport: Sequence[dict[str, Any]],
    runway: Sequence[dict[str, Any]],
    route: Sequence[dict[str, Any]],
    retrieval: dict[str, Any],
    mc: dict[str, Any],
    deterministic: dict[str, dict[str, Any]],
    validation_truth: np.ndarray,
    validation_series: Sequence[FlightSeries],
    histories: dict[str, list[dict[str, Any]]],
) -> list[str]:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    labels = [row["label"] for row in model_summary]
    x = np.arange(len(labels))
    width = 0.36
    axis.bar(x - width / 2, [row["ade_m"] for row in model_summary], width, label="ADE")
    axis.bar(x + width / 2, [row["fde_m"] for row in model_summary], width, label="FDE")
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_ylabel("Metres")
    axis.set_title("Validation accuracy on a common physical-time grid")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_figure(figure, plots, "model_accuracy")
    generated.append("model_accuracy")

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    fit_x = np.arange(len(labels))
    fit_width = 0.36
    for axis, metric, title in (
        (axes[0], "ade", "Native-grid ADE: fixed-anchor fit replay"),
        (axes[1], "fde", "Native-grid FDE: fixed-anchor fit replay"),
    ):
        axis.bar(
            fit_x - fit_width / 2,
            [row[f"train_{metric}_m"] for row in fit_summary],
            fit_width,
            label="train",
        )
        axis.bar(
            fit_x + fit_width / 2,
            [row[f"val_{metric}_m"] for row in fit_summary],
            fit_width,
            label="validation",
        )
        axis.set_xticks(fit_x, labels, rotation=20, ha="right")
        axis.set(ylabel="Metres", title=title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend()
    save_figure(figure, plots, "fixed_anchor_fit_replay")
    generated.append("fixed_anchor_fit_replay")

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    for label in labels:
        rows = [row for row in remaining if row["model"] == label]
        axis.plot(
            [row["lower_s"] for row in rows],
            [row["mean_error_m"] for row in rows],
            marker="o",
            label=label,
        )
    axis.set(xlabel="Remaining true time (bin lower edge, s)", ylabel="Mean 3-D error (m)")
    axis.set_title("Error versus remaining prediction time")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, plots, "error_by_remaining_time")
    generated.append("error_by_remaining_time")

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    airports = sorted({row["airport"] for row in airport})
    offsets = np.linspace(-0.35, 0.35, len(labels))
    width = 0.7 / max(len(labels), 1)
    for offset, label in zip(offsets, labels):
        lookup = {row["airport"]: row["ade_mean_m"] for row in airport if row["model"] == label}
        axis.bar(np.arange(len(airports)) + offset, [lookup.get(key, np.nan) for key in airports], width, label=label)
    axis.set_xticks(np.arange(len(airports)), airports)
    axis.set(ylabel="Mean ADE (m)", title="Validation ADE by airport")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, plots, "airport_ade")
    generated.append("airport_ade")

    runways = sorted(
        {row["runway"] for row in runway},
        key=lambda key: np.mean([
            row["ade_mean_m"] for row in runway if row["runway"] == key
        ]),
    )
    matrix = np.array([
        [
            next(
                (row["ade_mean_m"] for row in runway
                 if row["runway"] == runway_name and row["model"] == label),
                np.nan,
            )
            for label in labels
        ]
        for runway_name in runways
    ])
    figure, axis = plt.subplots(figsize=(10.2, max(5.2, 0.28 * len(runways))))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(runways)), runways)
    axis.set_title("Validation ADE by airport/runway")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Mean ADE (m)")
    save_figure(figure, plots, "runway_ade_heatmap")
    generated.append("runway_ade_heatmap")

    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    offsets = np.linspace(-0.35, 0.35, len(labels))
    width = 0.7 / max(len(labels), 1)
    for offset, label in zip(offsets, labels):
        lookup = {row["trajectory_type"]: row["ade_mean_m"] for row in route if row["model"] == label}
        axis.bar(np.arange(len(ROUTE_TYPES)) + offset, [lookup.get(key, np.nan) for key in ROUTE_TYPES], width, label=label)
    axis.set_xticks(np.arange(len(ROUTE_TYPES)), ROUTE_TYPES)
    axis.set(ylabel="Mean ADE (m)", title="Validation ADE by geometric trajectory type")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, plots, "trajectory_type_ade")
    generated.append("trajectory_type_ade")

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.plot(
        np.asarray(retrieval["progress"]) * 100.0,
        retrieval["mean_dispersion_by_progress_m"],
        color="#a53f2b",
        linewidth=2.2,
    )
    axis.set(xlabel="Normalized future progress (%)", ylabel="Neighbour-future RMS dispersion (m)")
    axis.set_title("Predictability ceiling: futures after similar observed histories")
    axis.grid(alpha=0.25)
    save_figure(figure, plots, "future_dispersion")
    generated.append("future_dispersion")

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    selected_label = mc["checkpoint_label"]
    deterministic_ade = float(deterministic[selected_label]["ade_per_flight_m"].mean())
    deterministic_fde = float(deterministic[selected_label]["fde_per_flight_m"].mean())
    for axis, metric, baseline, title in (
        (axes[0], "minade_m", deterministic_ade, "Oracle minADE"),
        (axes[1], "minfde_m", deterministic_fde, "Oracle minFDE"),
    ):
        for block, label, marker in ((mc, "MC dropout", "o"), (retrieval, "retrieval", "s")):
            axis.plot([row["k"] for row in block["curve"]], [row[metric] for row in block["curve"]], marker=marker, label=label)
        axis.axhline(baseline, color="#222", linestyle="--", label="deterministic")
        axis.set(xlabel="Candidate count K", ylabel="Metres", title=title)
        axis.grid(alpha=0.25)
    axes[1].legend()
    save_figure(figure, plots, "multi_candidate_coverage")
    generated.append("multi_candidate_coverage")

    example = int(retrieval["example_index"])
    truth_xy = validation_truth[example][:, list(POSITION_IDX)][:, :2]
    figure, axis = plt.subplots(figsize=(7.2, 7.0))
    for candidate in retrieval["example_candidates"][: min(10, len(retrieval["example_candidates"]))]:
        axis.plot(candidate[:, 0], candidate[:, 1], color="#8da0cb", alpha=0.32, linewidth=1)
    axis.plot(truth_xy[:, 0], truth_xy[:, 1], color="#111", linewidth=2.5, label="validation truth")
    model_xy = deterministic[selected_label]["common_prediction"][example][
        :, list(POSITION_IDX)
    ][:, :2]
    axis.plot(model_xy[:, 0], model_xy[:, 1], color="#d95f02", linewidth=2.0, label="deterministic model")
    axis.scatter([0], [0], marker="x", s=65, color="black", label="runway threshold")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set(xlabel="East (m)", ylabel="North (m)", title=f"High-dispersion example: {validation_series[example].flight_id}")
    axis.grid(alpha=0.2)
    axis.legend()
    save_figure(figure, plots, "candidate_example")
    generated.append("candidate_example")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, rows in histories.items():
        if not rows:
            continue
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], label=label)
        axes[1].plot(epochs, [row["val_loss"] for row in rows], label=label)
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation objective", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save_figure(figure, plots, "training_curves")
    generated.append("training_curves")
    return generated


def table_html(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_html(
    output: Path,
    result: dict[str, Any],
    model_summary: Sequence[dict[str, Any]],
    retrieval: dict[str, Any],
    mc: dict[str, Any],
) -> None:
    fit_summary = result["fit_replay"]["summary"]
    model_table = table_html(
        (
            "Model / clock", "ADE (m)", "FDE (m)", "time MAE (s)",
            "predicted cap", "truth > horizon", "epochs",
        ),
        [
            (
                row["label"], f"{row['ade_m']:.1f}", f"{row['fde_m']:.1f}",
                f"{row['final_time_mae_s']:.1f}",
                f"{100 * row['prediction_horizon_cap_rate']:.1f}%",
                f"{100 * row['true_horizon_exceed_rate']:.1f}%",
                row["epochs_run"],
            )
            for row in model_summary
        ],
    )
    fit_table = table_html(
        (
            "Model / clock", "train ADE", "val ADE", "ADE ratio",
            "train FDE", "val FDE", "FDE ratio", "train/val flights",
        ),
        [
            (
                row["label"],
                f"{row['train_ade_m']:.1f}",
                f"{row['val_ade_m']:.1f}",
                f"{row['ade_ratio']:.2f}×",
                f"{row['train_fde_m']:.1f}",
                f"{row['val_fde_m']:.1f}",
                f"{row['fde_ratio']:.2f}×",
                f"{row['train_flights']}/{row['val_flights']}",
            )
            for row in fit_summary
        ],
    )
    kinematic_table = table_html(
        (
            "Model / clock", "pos–vel RMSE (m/s)", "heading p95 (deg)",
            "turn p95 (deg/s)", "accel p95 (m/s²)", "jerk p95 (m/s³)",
        ),
        [
            (
                row["label"],
                f"{row['raw_kinematics']['position_velocity_rmse_mps']:.1f}",
                f"{row['raw_kinematics']['heading_consistency_p95_deg']:.1f}",
                f"{row['raw_kinematics']['turn_rate_p95_deg_s']:.1f}",
                f"{row['raw_kinematics']['acceleration_p95_mps2']:.1f}",
                f"{row['raw_kinematics']['jerk_p95_mps3']:.1f}",
            )
            for row in model_summary
        ],
    )
    candidate_table = table_html(
        ("Method", "K", "minADE (m)", "minFDE (m)"),
        [
            (name, row["k"], f"{row['minade_m']:.1f}", f"{row['minfde_m']:.1f}")
            for name, block in (("MC dropout", mc), ("Nearest-history retrieval", retrieval))
            for row in block["curve"]
        ],
    )
    control_rows = []
    duration_rows = []
    for model_row in model_summary:
        diagnostics = model_row.get("control_diagnostics")
        if diagnostics is None:
            continue
        for name, statistics in diagnostics["channels"].items():
            control_rows.append((
                model_row["label"], name, statistics["unit"],
                f"{statistics['p5']:.4g}", f"{statistics['median']:.4g}",
                f"{statistics['p95']:.4g}",
                f"{100 * statistics['near_lower_fraction']:.2f}%",
                f"{100 * statistics['near_upper_fraction']:.2f}%",
                f"{statistics['adjacent_abs_change_median']:.4g}",
                f"{statistics['adjacent_abs_change_p95']:.4g}",
            ))
        duration = diagnostics["durations_s"]
        duration_rows.append((
            model_row["label"], f"{duration['min']:.4g}", f"{duration['p1']:.4g}",
            f"{duration['median']:.4g}", f"{duration['p99']:.4g}",
            f"{duration['max']:.4g}",
        ))
    control_table = table_html(
        (
            "Model", "Control", "Unit", "p5", "median", "p95", "near lower",
            "near upper", "adjacent |Δ| median", "adjacent |Δ| p95",
        ),
        control_rows,
    ) if control_rows else "<p>No control-output checkpoint was included.</p>"
    duration_table = table_html(
        ("Model", "min (s)", "p1 (s)", "median (s)", "p99 (s)", "max (s)"),
        duration_rows,
    ) if duration_rows else ""
    generated_at = result["generated_at_utc"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pooled trajectory predictability report</title>
<style>
:root{{--ink:#17212b;--muted:#5e6b76;--paper:#fff;--wash:#f1f5f7;--accent:#1e6f8c;--zh:#fff4d6}}
body{{margin:0;background:var(--wash);color:var(--ink);font:16px/1.58 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1120px;margin:auto;background:var(--paper);padding:42px 54px 80px;box-shadow:0 0 28px #24323d18}}
h1{{font-size:2.25rem;line-height:1.15}} h2{{margin-top:2.5rem;border-bottom:2px solid #dce7eb;padding-bottom:.35rem}}
.lead{{font-size:1.12rem;color:var(--muted)}} .zh{{background:var(--zh);border-left:5px solid #e1a93a;padding:14px 18px;margin:18px 0}}
.warning{{background:#fde9e7;border-left:5px solid #c94a3d;padding:14px 18px}} figure{{margin:24px 0 38px}} img{{max-width:100%;height:auto;border:1px solid #d9e0e4}}
figcaption{{color:var(--muted);font-size:.92rem}} .table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{padding:9px 11px;border-bottom:1px solid #dfe5e8;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{background:#edf4f6}}
code{{background:#edf1f3;padding:.1em .3em}} .meta{{font-size:.9rem;color:var(--muted)}}
</style></head><body><main>
<h1>Pooled terminal-trajectory predictability report</h1>
<p class="lead">A validation-only common-clock comparison of state and control-output trajectory predictors, plus conditional future dispersion and multi-candidate oracle coverage.</p>
<p class="meta">Generated {html.escape(generated_at)} · schema <code>{REPORT_SCHEMA}</code> · split <code>validation</code></p>
<div class="warning"><strong>Evaluation boundary.</strong> This report reads outer-train and outer-validation only. Outer-test identities and outputs are not loaded. Oracle minADE/minFDE choose the best candidate after seeing truth; they measure set coverage, not deployable candidate selection.</div>
<div class="zh"><strong>重要说明：</strong>全文没有使用 test 数据。多候选的 minADE/minFDE 是“事后从 K 条候选中挑最好”的覆盖率上限，不能当作在线系统实际精度。</div>

<h2>Executive comparison</h2>{model_table}
<figure><img src="plots/model_accuracy.svg" alt="Model ADE and FDE comparison"><figcaption>All models are evaluated on the same true physical-time grid. Full and window predictions use their restored fixed-dt clock, geometric threshold truncation, and the common H_full cap.</figcaption></figure>
<figure><img src="plots/training_curves.svg" alt="Training and validation loss curves"><figcaption>Objective histories from the checkpoint directories; the retained checkpoint is the best validation epoch.</figcaption></figure>
<div class="zh"><strong>如何读表：</strong>ADE 是整段未来的平均三维位置误差；FDE 是跑道端点时刻的三维误差。full/window 最多递推到 H_full×dt；若预测没有到达跑道，将标记 horizon cap。normalized 则由 final_time_s 把完整航迹映射到归一化进度。</div>

<h2>Best-checkpoint train versus validation replay</h2>{fit_table}
<figure><img src="plots/fixed_anchor_fit_replay.svg" alt="Fixed-anchor train and validation ADE/FDE"><figcaption>The retained best checkpoint is replayed in <code>model.eval()</code> mode on both splits, with dropout disabled, one fixed L-1 anchor per flight, sequential batches and no shuffle. These are native-target-grid measured-data metrics, not the common-grid whole-trajectory metrics above.</figcaption></figure>
<div class="zh"><strong>拟合能力判断：</strong>这一节专门回答模型是否拟合训练集。train 与 validation 使用完全相同的推理状态和固定 anchor；若 train ADE 本身仍很大，说明训练集拟合能力有限；若 train 很低但 validation 明显升高，则说明泛化差距或过拟合。window 的这里是单次短窗误差，整轨递归误差仍看上面的共同时间网格。</div>

<h2>Raw-output kinematics</h2>{kinematic_table}
<p>These metrics are computed before smoothing or terminal post-processing, on each model's native output clock. Lower is better. Position–velocity RMSE is a fleet-level RMS statistic and is deliberately sensitive to catastrophic timing failures; heading, turn, acceleration and jerk use p95 to describe typical tail behaviour.</p>
<div class="zh"><strong>运动学解读：</strong>位置—速度一致性检查“相邻位置差分得到的速度”是否等于模型直接输出的速度。normalized 模式若把 final_time_s 预测得接近 0，会把固定空间位移压进极短时间，从而产生极大的离群值；因此这里不能只看平均 ADE。</div>

<h2>Control and duration distributions</h2>{control_table}{duration_table}
<p>Statistics are unweighted over validation flights and segments. “Near bound” means within 1% of each flight's physical control range. Adjacent changes are absolute segment-to-segment differences in the channel's displayed unit.</p>
<div class="zh"><strong>控制诊断：</strong>这里报告未加权的控制量和分段时长分布，用来识别推力、坡度角、载荷因子贴边，以及大量接近零时长的退化；它们不能由加权 regularizer loss 替代。</div>

<h2>1 — Error structure</h2>
<p>Errors are stratified by true remaining time, airport, runway and a documented geometric route proxy. The proxy uses accumulated heading change and path tortuosity: straight-in, single-turn, vectoring and holding-like. It is not an operational ATC intent label.</p>
<figure><img src="plots/error_by_remaining_time.svg" alt="Error by remaining time"><figcaption>Mean 3-D error at common physical timestamps, grouped by the amount of true time still remaining.</figcaption></figure>
<figure><img src="plots/airport_ade.svg" alt="ADE by airport"><figcaption>Pooled training does not imply uniform airport difficulty; this exposes domain imbalance after airport-macro loss weighting.</figcaption></figure>
<figure><img src="plots/runway_ade_heatmap.svg" alt="ADE heatmap by airport and runway"><figcaption>Runways are airport-qualified so identically named runway ends at different airports are never merged.</figcaption></figure>
<figure><img src="plots/trajectory_type_ade.svg" alt="ADE by trajectory type"><figcaption>Route-complexity groups reveal whether the deterministic average is mainly failing on vectoring or holding-like futures.</figcaption></figure>
<div class="zh"><strong>关键点：</strong>“按机场加权”只让训练目标中五个机场贡献相同，并不会让五个机场同样容易预测。机场、跑道和复杂航迹分组可以定位误差究竟来自数据域差异还是未来多模态性。</div>

<h2>2 — Similar histories, dispersed futures</h2>
<p>Each anchor history is represented by five equally spaced, standardized, threshold-relative state snapshots. For every validation history, the analysis retrieves the K nearest outer-train histories and measures the RMS spatial dispersion of their future trajectories.</p>
<figure><img src="plots/future_dispersion.svg" alt="Conditional future dispersion"><figcaption>A high curve means nearly identical observed histories lead to materially different future routes. This is an empirical predictability ceiling for the chosen descriptor, not a proof of irreducible aleatoric uncertainty.</figcaption></figure>
<figure><img src="plots/candidate_example.svg" alt="High dispersion trajectory example"><figcaption>One high-dispersion validation case: retrieved train futures (blue), truth (black) and the deterministic model (orange).</figcaption></figure>
<p>Mean descriptor distance: <strong>{retrieval['neighbor_distance_mean']:.3f}</strong>. Mean nearest-neighbour future dispersion: <strong>{np.mean(retrieval['per_flight_dispersion_m']):.1f} m</strong>. Neighbour-centroid ADE: <strong>{retrieval['centroid_ade_m']:.1f} m</strong>.</p>
<div class="zh"><strong>中文解释：</strong>如果历史状态很接近的飞机后续却分别左转、右转或继续直飞，单一 MSE 模型通常会输出“平均航迹”。这不是简单增加 epoch 就一定能解决的问题，而是需要 intent、交互信息或概率/多候选输出。</div>

<h2>3 — Deterministic versus multi-candidate coverage</h2>{candidate_table}
<div class="warning"><strong>Endpoint degeneracy in retrieval.</strong> Every complete target future terminates at its own runway-threshold origin by construction. Retrieval FDE therefore collapses to a few metres even when the intermediate route is wrong. Use retrieval minADE and the dispersion curve for the substantive comparison; retrieval minFDE is shown only to expose this target-contract consequence.</div>
<div class="zh"><strong>终点退化：</strong>训练标签都在各自跑道入口坐标原点结束，所以检索候选的 FDE 天然接近 0；它不能证明候选航迹正确。这里真正有意义的是整段 minADE 和中途离散度。</div>
<figure><img src="plots/multi_candidate_coverage.svg" alt="Multi candidate oracle curves"><figcaption>Deterministic baseline versus MC-dropout and nearest-history oracle candidate-set coverage as K grows.</figcaption></figure>
<p>MC dropout changes only stochastic dropout masks in the selected checkpoint. Retrieval returns complete futures from similar outer-train histories. Neither method is presented as a final production multi-modal architecture; together they test whether candidate diversity can cover validation truth better than one deterministic path.</p>
<div class="zh"><strong>结论边界：</strong>若 retrieval 的 minADE 随 K 明显下降而 MC-dropout 几乎不变，说明现有网络的 dropout 不足以表达真实多模态；下一步更适合 CVAE、mixture head 或 diffusion，而不是单纯增加 dropout 次数。</div>

<h2>Artifacts and reproducibility</h2>
<p>Machine-readable outputs are stored beside this page: <code>report.json</code>, <code>flight_metrics.csv</code>, <code>error_by_remaining_time.csv</code>, <code>metrics_by_airport.csv</code>, <code>metrics_by_runway.csv</code>, <code>metrics_by_trajectory_type.csv</code>, and PNG/SVG figures under <code>plots/</code>. Checkpoint SHA-256 values and split digests are embedded in <code>report.json</code>.</p>
</main></body></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


def load_runs(specifications: Sequence[tuple[str, Path]]) -> list[LoadedRun]:
    labels: set[str] = set()
    runs = []
    for label, path in specifications:
        if label in labels:
            raise ValueError(f"duplicate checkpoint label {label!r}")
        labels.add(label)
        model, config, normalizer, payload = load_checkpoint(path)
        runs.append(LoadedRun(label, path, model, config, normalizer, payload))
    return runs


def comparison_identity_error(reference: LoadedRun, candidate: LoadedRun) -> str | None:
    """Return why two checkpoints cannot share a validation-only report."""
    for field in (
        "seq_len",
        "dt_s",
        "seed",
        "coordinate_frame",
        "aircraft_type",
        "aircraft_filter",
    ):
        if getattr(candidate.config, field) != getattr(reference.config, field):
            return f"checkpoint {candidate.label} differs in comparison field {field}"
    # Window mode may exclude a few train/test flights that cannot provide a
    # complete short target.  Those differences do not alter the evaluated
    # validation cohort or the reference checkpoint's retrieval library.
    if candidate.payload["split"]["val"] != reference.payload["split"]["val"]:
        return f"checkpoint {candidate.label} has a different validation split"
    return None


def true_horizon_exceed_rate(durations_s: np.ndarray, config: TSConfig) -> float:
    """Fraction of truths longer than a fixed-clock forecast's total reach."""
    if config.horizon_mode == HORIZON_NORMALIZED:
        return 0.0
    # Recursive window inference repeats H_window-sized passes until H_full, so
    # its total physical reach is H_full just like one-pass full inference.
    horizon_s = config.full_horizon_steps * config.dt_s
    return float((np.asarray(durations_s) > horizon_s).mean())


def history_for_checkpoint(path: Path) -> list[dict[str, Any]]:
    history_path = path.parent / "history.json"
    if not history_path.is_file():
        return []
    return json.loads(history_path.read_text(encoding="utf-8")).get("history", [])


def fit_evaluation_for_checkpoint(path: Path) -> dict[str, Any]:
    evaluation_path = path.parent / FIT_EVALUATION_NAME
    if not evaluation_path.is_file():
        raise ValueError(
            f"missing {evaluation_path}; run ts_transformer evaluate-fit for this checkpoint"
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != FIT_EVALUATION_SCHEMA:
        raise ValueError(f"{evaluation_path} has an obsolete fit-evaluation schema")
    if evaluation.get("checkpoint", {}).get("sha256") != file_sha256(path):
        raise ValueError(f"{evaluation_path} belongs to a different checkpoint")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--multi-candidate-checkpoint", default=None, metavar="LABEL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-points", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.evaluation_points <= 1 or args.candidate_count < 3:
        parser.error("batch size must be positive, evaluation points > 1 and candidates >= 3")

    runs = load_runs(args.checkpoint)
    reference = runs[0]
    reference_split = reference.payload["split"]
    for run in runs[1:]:
        if error := comparison_identity_error(reference, run):
            parser.error(error)

    provenance = reference.payload["data_provenance"]
    airports = tuple(entry["airport"] for entry in provenance["manifests"])
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    current_provenance = arrival_data_provenance(manifests)
    for run in runs:
        require_matching_data_provenance(run.payload, current_provenance)

    development_keys = set(reference_split["train"] + reference_split["val"])
    print(
        f"loading train/validation arrivals for {','.join(airports)}; "
        "outer-test source tracks stay closed",
        flush=True,
    )
    all_series, build_report = build_series(
        load_flight_dicts(manifests, include_flight_keys=development_keys),
        reference.config,
        aircraft_type=reference.config.aircraft_type,
    )
    print(build_report.format(), flush=True)
    all_series = usable_series(all_series, reference.config, verbose=False)
    by_id = {item.dataset_id: item for item in all_series}
    train_series = [by_id[key] for key in reference_split["train"]]
    validation_series = [by_id[key] for key in reference_split["val"]]
    truth, true_duration, progress, route_types = common_truth(
        validation_series, reference.config, args.evaluation_points
    )

    device = resolve_device(args.device)
    deterministic: dict[str, dict[str, Any]] = {}
    for run in runs:
        print(f"evaluating {run.label} on {len(validation_series)} validation flights", flush=True)
        deterministic[run.label] = run_deterministic(
            run, validation_series, truth, true_duration, progress, device, args.batch_size
        )
        run.model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_label = args.multi_candidate_checkpoint or runs[0].label
    selected = next((run for run in runs if run.label == selected_label), None)
    if selected is None:
        parser.error(f"unknown --multi-candidate-checkpoint {selected_label!r}")
    k_values = tuple(k for k in (1, 3, 5, 10, 20) if k <= args.candidate_count)
    if k_values[-1] != args.candidate_count:
        k_values = (*k_values, args.candidate_count)
    print(f"sampling {args.candidate_count} MC-dropout candidates from {selected_label}", flush=True)
    mc = mc_dropout_coverage(
        selected, validation_series, truth, true_duration, progress, device,
        args.batch_size, args.candidate_count, k_values,
    )
    mc["checkpoint_label"] = selected_label
    selected.model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("measuring nearest-history future dispersion", flush=True)
    retrieval = retrieval_coverage(
        train_series, validation_series, reference.config, reference.normalizer,
        truth, args.evaluation_points, k_values,
    )

    flight_rows: list[dict[str, Any]] = []
    for run in runs:
        block = deterministic[run.label]
        for index, item in enumerate(validation_series):
            source = item.scenario.source
            airport = item.airport or source.get("arr_airport") or "?"
            runway = source.get("runway") or "?"
            flight_rows.append({
                "model": run.label,
                "prediction_output": run.config.prediction_output,
                "dataset_id": item.dataset_id,
                "airport": airport,
                "runway": f"{airport}/{runway}",
                "trajectory_type": route_types[index],
                "true_final_time_s": true_duration[index],
                "predicted_final_time_s": block["predicted_final_time_s"][index],
                "final_time_error_s": block["predicted_final_time_s"][index] - true_duration[index],
                "ade_m": block["ade_per_flight_m"][index],
                "fde_m": block["fde_per_flight_m"][index],
                "horizon_capped": bool(block["horizon_capped"][index]),
            })

    remaining = remaining_time_rows(deterministic, true_duration, progress)
    airport_rows = grouped_rows(flight_rows, "airport")
    runway_rows = grouped_rows(flight_rows, "runway")
    route_rows = grouped_rows(flight_rows, "trajectory_type")
    histories = {run.label: history_for_checkpoint(run.checkpoint) for run in runs}
    try:
        fit_evaluations = {
            run.label: fit_evaluation_for_checkpoint(run.checkpoint) for run in runs
        }
    except ValueError as exc:
        parser.error(str(exc))
    fit_summary = []
    for run in runs:
        evaluation = fit_evaluations[run.label]
        train_metrics = evaluation["splits"]["train"]["metrics"]
        val_metrics = evaluation["splits"]["val"]["metrics"]
        gap = evaluation["diagnostics"]["native_generalization"]
        fit_summary.append({
            "label": run.label,
            "train_flights": evaluation["splits"]["train"]["flights"],
            "val_flights": evaluation["splits"]["val"]["flights"],
            "train_ade_m": train_metrics["ade_m"],
            "val_ade_m": val_metrics["ade_m"],
            "ade_ratio": gap["ade_m"]["ratio"],
            "train_fde_m": train_metrics["fde_m"],
            "val_fde_m": val_metrics["fde_m"],
            "fde_ratio": gap["fde_m"]["ratio"],
            "train_final_time_mae_s": train_metrics["final_time_s"]["mae"],
            "val_final_time_mae_s": val_metrics["final_time_s"]["mae"],
        })
    model_summary = []
    for run in runs:
        block = deterministic[run.label]
        model_summary.append({
            "label": run.label,
            "model": run.config.model,
            "prediction_output": run.config.prediction_output,
            "horizon_mode": run.config.horizon_mode,
            "pred_len": run.config.pred_len,
            "ade_m": float(block["ade_per_flight_m"].mean()),
            "fde_m": float(block["fde_per_flight_m"].mean()),
            "final_time_mae_s": float(np.abs(block["predicted_final_time_s"] - true_duration).mean()),
            "prediction_horizon_cap_rate": float(block["horizon_capped"].mean()),
            "true_horizon_exceed_rate": true_horizon_exceed_rate(true_duration, run.config),
            "raw_kinematics": block["raw_kinematics"],
            "control_diagnostics": block["control_diagnostics"],
            "epochs_run": len(histories[run.label]),
        })

    from datetime import datetime, timezone

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "flight_metrics.csv", flight_rows)
    write_csv(output / "error_by_remaining_time.csv", remaining)
    write_csv(output / "metrics_by_airport.csv", airport_rows)
    write_csv(output / "metrics_by_runway.csv", runway_rows)
    write_csv(output / "metrics_by_trajectory_type.csv", route_rows)
    plot_reports(
        output, model_summary, fit_summary, remaining, airport_rows, runway_rows, route_rows,
        retrieval, mc,
        deterministic, truth, validation_series, histories,
    )
    split_digest = hashlib.sha256(
        "\n".join(sorted(reference_split["val"])).encode()
    ).hexdigest()
    result = {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_policy": {
            "evaluated_split": "val",
            "retrieval_reference_split": "train",
            "outer_test_loaded": False,
            "validation_split_sha256": split_digest,
        },
        "airports": list(airports),
        "validation_flights": len(validation_series),
        "train_flights_for_retrieval": len(train_series),
        "evaluation_points": args.evaluation_points,
        "checkpoints": [
            {
                "label": run.label,
                "path": str(run.checkpoint),
                "sha256": file_sha256(run.checkpoint),
                "config": run.config.to_dict(),
            }
            for run in runs
        ],
        "model_summary": model_summary,
        "fit_replay": {
            "metric_scope": (
                "fixed L-1 anchor; measured-data mask; state uses its native clock; "
                "control truth is interpolated to predicted cumulative segment time"
            ),
            "summary": fit_summary,
        },
        "multi_candidate": {
            "warning": "oracle minADE/minFDE are candidate-set coverage, not online selection",
            "mc_dropout": {key: value for key, value in mc.items() if not key.startswith("per_flight")},
            "retrieval": {
                "method": retrieval["method"],
                "descriptor": retrieval["descriptor"],
                "curve": retrieval["curve"],
                "centroid_ade_m": retrieval["centroid_ade_m"],
                "neighbor_distance_mean": retrieval["neighbor_distance_mean"],
                "mean_future_dispersion_m": float(np.mean(retrieval["per_flight_dispersion_m"])),
                "endpoint_future_dispersion_m": float(retrieval["mean_dispersion_by_progress_m"][-1]),
                "endpoint_contract_note": (
                    "Retrieval FDE is degenerate because every target future ends at its "
                    "runway-threshold coordinate origin; use minADE for route coverage."
                ),
            },
        },
    }
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_html(output, result, model_summary, retrieval, mc)
    print(f"✓ wrote validation-only report to {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
