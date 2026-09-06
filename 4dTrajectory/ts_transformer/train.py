"""Training loop: the epoch, the validation pass and the checkpoint it writes.

What a prediction is scored against lives in ``objective.py`` — this module drives the
optimizer over it. The checkpoint carries the config, the fitted normalizer and the flight
ids of each split alongside the weights. That is what makes inference reproducible without
re-deriving anything: ``forecast.py`` loads a checkpoint and knows the resample step,
channel order, output length/time mode, and which flights the model must not be evaluated
on.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from channels import CHANNELS
from batching import resolve_batch_size
from config import (
    CONTROL_HOOK_OFF,
    HOOK_SATURATION_HARD,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_OBJECTIVE,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_STATE,
    TSConfig,
    control_recipe,
    uses_control_dynamics,
)
from control.basis_fit import FittedTeacherTable, load_fitted_teacher
from control.dynamics import rollout as control_rollout
from control.constraints import build_command_hook
from control.training.diagnostics import ControlTrainingDiagnosticsAccumulator
from dataset import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    RandomAnchorTrajectoryWindows,
    TrajectoryWindows,
    iter_batches,
    provenance_eligibility_digests,
    provenance_manifest_digests,
    split_by_flight,
    window_anchors,
)
from evaluation_protocol import (
    TEST_RELEASE_NAME,
    TEST_RELEASE_PROTOCOL_FIELD,
    TEST_RELEASE_SCHEMA,
)
from fixed_anchor_validation import (
    CommonGridTruth,
    fixed_anchor_common_truth,
    fixed_anchor_common_grid_ade_metrics,
    fixed_anchor_common_grid_metrics,
    fixed_anchor_common_grid_report_metrics,
)
from metrics import (
    raw_kinematic_metrics,
    states_with_derived_velocity,
)
from models import build_model, parameter_count, resolve_device
from batch_contract import anchor_state, model_forward, unpack_batch
from io_utils import file_sha256
from objective import (
    PROCEDURE_DIAGNOSTICS,
    ProcedureMultipliers,
    align_control_targets_to_query_clock,
    loss_component_names,
    move_dynamics,
    move_fixed_dt_supervision,
    prediction_loss_components,
    target_contract,
)
from prediction_outputs import ControlPrediction, StatePrediction
from closure_output import (
    ClosurePrediction,
    replay_batch as closure_replay_batch,
)
from time_grids import numpy_inference_time_grid
from training_performance import EpochProfiler

CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "ts-checkpoint-metadata-v35-true-time-endpoint-loss"
HISTORY_NAME = "history.json"
FIT_EVALUATION_NAME = "fit_evaluation.json"
FIT_EVALUATION_SCHEMA = "ts-fit-evaluation-v3-common-true-time-endpoint"


def _split_sha256(series: Sequence[FlightSeries]) -> str:
    """Stable audit digest shared conceptually with cross-validation's split record."""
    return _keys_sha256(item.dataset_id for item in series)


def _keys_sha256(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(keys)).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    optimizer_updates: int
    seconds: float
    val_by_airport: dict[str, float]
    train_components: dict[str, float]
    val_components: dict[str, float]
    validation_selection_metric: str = CHECKPOINT_SELECTION_OBJECTIVE
    validation_selection_value: float | None = None
    validation_selection_by_airport: dict[str, float] = field(default_factory=dict)
    train_anchor_sampling: dict[str, Any] = field(default_factory=dict)
    control_training_diagnostics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    validation_profile_by_airport: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    # The procedure penalty's epoch record: gated rows, violation rates, λ after the update.
    procedure: dict[str, float] = field(default_factory=dict)
    # The command hook's epoch record: every ``hook_*`` count the hook reports, divided by
    # the step count (per-step shares and per-step means, whatever the hook counts), plus
    # ``steps`` itself.
    command_hook: dict[str, float] = field(default_factory=dict)
    # The latent intent's epoch record (control/latent.py): KL nats per flight and the number
    # of latent dimensions carrying more than ACTIVE_UNIT_KL_NATS — the posterior-collapse
    # reading, which looks exactly like "converged" on every other number.
    latent: dict[str, float] = field(default_factory=dict)


@dataclass
class FitResult:
    """In-memory result shared by final training and cross-validation folds."""

    model: nn.Module
    config: TSConfig
    normalizer: Normalizer
    device: torch.device
    history: list[EpochResult]
    best_val_loss: float
    best_validation_selection: float
    train_windows: int
    val_windows: int
    procedure_multipliers: dict[str, float] | None = None
    # The teacher table this fit supervised its imitation term with, parsed ONCE by
    # ``fit_model`` and handed back so the caller can stamp its digest without reopening it.
    fitted_teacher: FittedTeacherTable | None = None


@dataclass(frozen=True)
class SplitPredictionReplay:
    """One immutable deployable prediction pass reused by every metric view."""

    predicted: np.ndarray
    truth: np.ndarray
    mask: np.ndarray
    predicted_time_s: np.ndarray
    truth_time_s: np.ndarray
    anchors: np.ndarray
    segment_durations_s: np.ndarray


def _prediction_batch_replay(
    output: StatePrediction | ControlPrediction,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    final_time_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor] | None,
    dataset: TrajectoryWindows,
) -> SplitPredictionReplay:
    """Materialize deployable physical arrays from an already-computed model output."""
    metric_targets = y
    metric_weights = mask
    if uses_control_dynamics(dataset.config.prediction_output):
        deployable = output
        if dynamics is None:
            raise ValueError("control replay requires per-flight dynamics")
        points = dataset.config.validation_common_grid_points
        progress = torch.arange(
            1,
            points + 1,
            dtype=torch.float64,
            device=deployable.segment_durations.device,
        ) / points
        predicted_total_s = deployable.segment_durations.to(torch.float64).sum(dim=1)
        query_offsets_s = predicted_total_s.unsqueeze(1) * progress.unsqueeze(0)
        query_valid = torch.ones_like(query_offsets_s, dtype=torch.bool)
        rollout = control_rollout.rollout_control_dense(
            deployable.controls,
            deployable.segment_durations,
            dynamics,
            query_offsets_s,
            query_valid,
            dataset.config,
            command_hook=build_command_hook(dataset.config, dynamics),
        )
        predicted_physical = (
            rollout.query_channels.detach().cpu().numpy().astype(np.float32)
        )
        metric_targets, metric_weights = align_control_targets_to_query_clock(
            anchor_state(x, len(dataset.config.channels)),
            y,
            mask,
            query_offsets_s,
            final_time_s,
        )
        segment_durations_s = np.broadcast_to(
            (
                predicted_total_s.detach().cpu().numpy().astype(np.float64)
                / points
            )[:, None],
            (len(x), points),
        ).copy()
        predicted_time_s = deployable.final_time_s.detach().cpu().numpy()
    elif isinstance(output, ClosurePrediction):
        # Drawn, not rolled out: every decision reconstructed in numpy and sampled on the
        # target grid's fractions of its own duration (the context carries the course).
        if dynamics is None:
            raise ValueError("closure replay requires the per-flight label context")
        anchors_physical = dataset.normalizer.decode(
            anchor_state(x, len(dataset.config.channels))
            .detach().cpu().numpy().astype(np.float64)
        )
        predicted_physical, segment_durations_s, predicted_time_s = closure_replay_batch(
            output, anchors_physical, dynamics, dataset.config, dataset.config.pred_len
        )
    else:
        if not isinstance(output, StatePrediction):
            raise TypeError("state replay requires StatePrediction")
        out = output.states.detach().cpu().numpy()
        predicted_physical = dataset.normalizer.decode(
            out.astype(np.float64)
        ).astype(np.float32)
        segment_durations_s = numpy_inference_time_grid(
            output.final_time_s.detach().cpu().numpy(), dataset.config
        )[0]
        predicted_time_s = output.final_time_s.detach().cpu().numpy()

    # Decode in float64 (the normalizer stats' dtype), store float32: a pooled split is
    # tens of thousands of [N,C] windows and metre-scale metrics do not need float64 storage.
    truth = dataset.normalizer.decode(
        metric_targets.detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    anchors = dataset.normalizer.decode(
        anchor_state(x, len(dataset.config.channels))
        .detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    if dataset.config.prediction_output == PREDICTION_STATE:
        # The state output predicts positions + duration only; the control rollout and
        # the closure reconstruction both carry exact velocities.
        predicted_physical = states_with_derived_velocity(
            anchors,
            predicted_physical,
            segment_durations_s,
        ).astype(np.float32)
    raw_mask = metric_weights.detach().cpu().numpy()
    if raw_mask.ndim == 3:
        raw_mask = np.all(raw_mask > 0.0, axis=-1).astype(np.float32)
    return SplitPredictionReplay(
        predicted=predicted_physical,
        truth=truth,
        mask=raw_mask,
        predicted_time_s=predicted_time_s,
        truth_time_s=final_time_s.detach().cpu().numpy(),
        anchors=anchors,
        segment_durations_s=segment_durations_s,
    )


def _merge_prediction_replays(
    chunks: Sequence[tuple[np.ndarray, SplitPredictionReplay]],
    *,
    count: int,
) -> SplitPredictionReplay:
    """Restore dataset order after validation-only duration bucketing."""
    if not chunks:
        raise ValueError("prediction replay requires at least one batch")
    indices = np.concatenate([item[0] for item in chunks])
    order = np.argsort(indices, kind="stable")
    if not np.array_equal(indices[order], np.arange(count, dtype=np.int64)):
        raise ValueError("prediction replay indices must cover the dataset exactly once")

    def merged(name: str) -> np.ndarray:
        return np.concatenate(
            [getattr(item[1], name) for item in chunks], axis=0
        )[order]

    return SplitPredictionReplay(
        predicted=merged("predicted"),
        truth=merged("truth"),
        mask=merged("mask"),
        predicted_time_s=merged("predicted_time_s"),
        truth_time_s=merged("truth_time_s"),
        anchors=merged("anchors"),
        segment_durations_s=merged("segment_durations_s"),
    )


def _predict_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int,
) -> SplitPredictionReplay:
    """Return physical anchor/output arrays, masks, predicted time and true time."""
    model.eval()
    chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    cursor = 0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            (
                x,
                y,
                mask,
                final_time_s,
                _flight_weights,
                dynamics,
                _dense_supervision,
            ) = unpack_batch(raw_batch)
            x_device = x.to(device)
            y_device = y.to(device)
            mask_device = mask.to(device)
            final_time_device = final_time_s.to(device)
            dynamics_device = move_dynamics(dynamics, device)
            output = model_forward(model, x_device, dynamics_device)
            batch_replay = _prediction_batch_replay(
                output,
                x_device,
                y_device,
                mask_device,
                final_time_device,
                dynamics_device,
                dataset,
            )
            batch_count = len(x)
            chunks.append(
                (np.arange(cursor, cursor + batch_count, dtype=np.int64), batch_replay)
            )
            cursor += batch_count
    return _merge_prediction_replays(
        chunks,
        count=len(dataset),
    )


def evaluate_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    replay: SplitPredictionReplay | None = None,
) -> dict[str, Any]:
    """Formal fixed-anchor metrics on the common true-physical-time grid."""
    replay = replay or _predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = fixed_anchor_common_grid_report_metrics(
        dataset.series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
        points=config.validation_common_grid_points,
    )
    indices_by_airport: dict[str, list[int]] = {}
    for index, item in enumerate(dataset.series):
        indices_by_airport.setdefault(item.airport or "<unknown>", []).append(index)
    by_airport: dict[str, dict[str, Any]] = {}
    for airport, indices in sorted(indices_by_airport.items()):
        selected = np.asarray(indices, dtype=np.int64)
        airport_block = fixed_anchor_common_grid_report_metrics(
            [dataset.series[index] for index in indices],
            config,
            replay.anchors[selected],
            replay.predicted[selected],
            replay.predicted_time_s[selected],
            replay.segment_durations_s[selected],
            points=config.validation_common_grid_points,
        )
        by_airport[airport] = {
            key: airport_block[key]
            for key in (
                "flights",
                "ade_m",
                "fde_m",
                "arrival_endpoint_error_m",
                "horizontal_m",
                "along_track_m",
                "cross_track_m",
                "vertical_m",
                "final_time_s",
                "prediction_horizon_cap_rate",
                "invalid_flights",
            )
        }
    block["flight_micro_ade_m"] = block["ade_m"]
    block["flight_micro_fde_m"] = block["fde_m"]
    block["per_airport"] = by_airport
    block["airport_macro"] = {
        "ade_m": float(np.mean([item["ade_m"] for item in by_airport.values()])),
        "fde_m": float(np.mean([item["fde_m"] for item in by_airport.values()])),
        "arrival_endpoint_error_m": float(np.mean([
            item["arrival_endpoint_error_m"]["mean"]
            for item in by_airport.values()
        ])),
        "final_time_mae_s": float(np.mean([
            item["final_time_s"]["mae"] for item in by_airport.values()
        ])),
    }
    # Compatibility scalar names now point at the single formal airport-macro score.
    block["ade_m"] = block["airport_macro"]["ade_m"]
    block["fde_m"] = block["airport_macro"]["fde_m"]
    # Raw model nodes on their own predicted clock: no measured-track interpolation,
    # spline, filtering or CZML resampling. Durations are explicit [B,N] so this call site
    # remains valid when the output layer moves from uniform to nonuniform segments.
    active_segments = replay.segment_durations_s > 0.0
    block["raw_kinematics"] = raw_kinematic_metrics(
        replay.anchors,
        replay.predicted,
        replay.segment_durations_s,
        valid_segments=active_segments,
    )
    observed_nodes, observed_duration_s, _ = fixed_anchor_common_truth(
        dataset.series, config, replay.predicted.shape[1]
    )
    observed_segment_durations_s = np.broadcast_to(
        (observed_duration_s / replay.predicted.shape[1])[:, None],
        replay.segment_durations_s.shape,
    ).copy()
    block["raw_kinematics_observed_baseline"] = raw_kinematic_metrics(
        replay.anchors,
        observed_nodes,
        observed_segment_durations_s,
    )
    return block


def evaluate_fixed_anchor_common_grid(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    replay: SplitPredictionReplay | None = None,
) -> dict[str, Any]:
    """Deployable fixed-anchor metrics on one shared physical-time grid."""
    replay = replay or _predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = fixed_anchor_common_grid_metrics(
        dataset.series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
        points=config.validation_common_grid_points,
        normalizer=normalizer,
    )
    return {
        key: value
        for key, value in block.items()
        if not isinstance(value, np.ndarray)
    }


def _generalization_metric(train_value: float, val_value: float) -> dict[str, float | None]:
    return {
        "train": float(train_value),
        "val": float(val_value),
        "absolute_gap": float(val_value - train_value),
        "ratio": float(val_value / train_value) if train_value != 0.0 else None,
    }


def _training_objective_diagnostics(
    history: Sequence[EpochResult | dict[str, Any]], config: TSConfig
) -> dict[str, Any] | None:
    if not history:
        return None
    rows = [vars(row) if isinstance(row, EpochResult) else row for row in history]
    best = min(
        rows,
        key=lambda row: (
            row.get("validation_selection_value")
            if row.get("validation_selection_value") is not None
            else row["val_loss"]
        ),
    )
    result = {
        "best_epoch": int(best["epoch"]),
        "epochs_run": len(rows),
        "reached_epoch_budget": len(rows) == config.epochs,
        "train_loss_at_best_epoch": float(best["train_loss"]),
        "val_loss_at_best_epoch": float(best["val_loss"]),
        "absolute_gap": float(best["val_loss"] - best["train_loss"]),
        "ratio": (
            float(best["val_loss"] / best["train_loss"])
            if best["train_loss"] != 0.0 else None
        ),
        "checkpoint_selection_metric": best.get(
            "validation_selection_metric", CHECKPOINT_SELECTION_OBJECTIVE
        ),
        "checkpoint_selection_value": float(
            best.get("validation_selection_value")
            if best.get("validation_selection_value") is not None
            else best["val_loss"]
        ),
    }
    last = rows[-1]
    if "learning_rate" in last:
        result["final_learning_rate"] = float(last["learning_rate"])
    if "optimizer_updates" in last:
        result["total_optimizer_updates"] = int(last["optimizer_updates"])
    return result


def _epoch_performance_summary(
    history: Sequence[EpochResult | dict[str, Any]],
    *,
    warmup_epochs: int = 3,
) -> dict[str, Any]:
    """Report median/p90 after a fixed warm-up without affecting model selection."""
    rows = [vars(row) if isinstance(row, EpochResult) else row for row in history]
    excluded = warmup_epochs if len(rows) > warmup_epochs else 0
    measured = rows[excluded:]
    timing_keys = sorted({
        key
        for row in measured
        for key, value in row.get("timing", {}).items()
        if isinstance(value, (int, float))
    })
    metrics = {}
    for key in timing_keys:
        values = np.asarray(
            [row["timing"][key] for row in measured if key in row.get("timing", {})],
            dtype=np.float64,
        )
        if len(values):
            metrics[key] = {
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
            }
    return {
        "warmup_epochs_excluded": excluded,
        "measured_epochs": len(measured),
        "timing": metrics,
    }


def evaluate_fit_splits(
    model: nn.Module,
    train_series: Sequence[FlightSeries],
    val_series: Sequence[FlightSeries],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    history: Sequence[EpochResult | dict[str, Any]] = (),
) -> dict[str, Any]:
    """Replay the best model deterministically on fixed-anchor train and validation.

    This is the shared post-fit/standalone evaluation seam. It deliberately ignores the
    training anchor policy: both splits use one fixed ``L-1`` anchor, sequential batches,
    ``model.eval()`` and no gradients so train/validation metrics are directly comparable.
    """
    if not train_series or not val_series:
        raise ValueError("fit evaluation requires non-empty train and validation splits")
    model.to(device).eval()
    split_series = {"train": train_series, "val": val_series}
    splits: dict[str, dict[str, Any]] = {}
    for split, group in split_series.items():
        splits[split] = evaluate_fixed_anchor_series(
            model, group, normalizer, config, device, split_name=split
        )

    train_metrics = splits["train"]["metrics"]
    val_metrics = splits["val"]["metrics"]
    diagnostics: dict[str, Any] = {
        "generalization": {
            "ade_m": _generalization_metric(train_metrics["ade_m"], val_metrics["ade_m"]),
            "fde_m": _generalization_metric(train_metrics["fde_m"], val_metrics["fde_m"]),
            "final_time_mae_s": _generalization_metric(
                train_metrics["final_time_s"]["mae"],
                val_metrics["final_time_s"]["mae"],
            ),
        }
    }
    objective = _training_objective_diagnostics(history, config)
    if objective is not None:
        diagnostics["training_objective"] = objective

    return {
        "schema_version": FIT_EVALUATION_SCHEMA,
        "evaluation_contract": {
            "model_mode": "eval",
            "dropout": "disabled",
            "anchor": "fixed L-1",
            "batch_order": "sequential (shuffle disabled)",
            "splits": ["train", "val"],
            "metric_grid": (
                f"Q={config.validation_common_grid_points} common true physical time; "
                "prediction endpoint held after early completion"
            ),
        },
        "config": config.to_dict(),
        "splits": splits,
        "diagnostics": diagnostics,
    }


def evaluate_fixed_anchor_series(
    model: nn.Module,
    series: Sequence[FlightSeries],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    split_name: str = "cohort",
) -> dict[str, Any]:
    """Evaluate one explicit flight cohort once on both native and common grids."""
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    if len(dataset) != len(series):
        raise ValueError(
            f"fixed-anchor {split_name} replay covers {len(dataset)}/{len(series)} flights"
        )
    replay = _predict_split(model, dataset, normalizer, device, config.batch_size)
    formal_metrics = evaluate_split(
        model, dataset, normalizer, config, device, replay=replay
    )
    common_grid_metrics = (
        formal_metrics
        if config.checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE
        else evaluate_fixed_anchor_common_grid(
            model, dataset, normalizer, config, device, replay=replay
        )
    )
    return {
        "flights": len(series),
        "windows": len(dataset),
        "split_sha256": _split_sha256(series),
        "metrics": formal_metrics,
        # Retained as an artifact key for existing report readers. Under the formal ADE
        # selector it is the same minimal contract, not a second evaluator.
        "common_grid_metrics": common_grid_metrics,
    }


def write_fit_evaluation(
    evaluation: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Bind a deterministic fit replay to its exact checkpoint and write it atomically."""
    checkpoint = Path(checkpoint_path).resolve()
    document = dict(evaluation)
    document["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / FIT_EVALUATION_NAME
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(output)
    return document


def usable_series(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    minimum_anchor_index: int | None = None,
    verbose: bool = True,
) -> list[FlightSeries]:
    """Drop flights that cannot yield one model window, once and with an audit count."""
    usable = [
        item for item in series
        if len(window_anchors(
            item, config, minimum_anchor_index=minimum_anchor_index
        )) > 0
    ]
    if len(usable) < len(series) and verbose:
        need = config.seq_len + 1
        print(f"  excluded   {len(series) - len(usable)} flight(s) too short to yield one "
              f"training window (need {need} samples = {need * config.dt_s:.0f}s)")
    return usable


def filter_training_cohort(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    verbose: bool = True,
) -> tuple[list[FlightSeries], dict[str, Any]]:
    """Apply a predeclared future-duration floor to train flights only."""
    minimum = float(config.training_cohort_min_future_s)
    anchor = config.seq_len - 1
    retained: list[FlightSeries] = []
    excluded: list[dict[str, Any]] = []
    for item in series:
        remaining = float(item.supervision_times[-1] - item.times[anchor])
        if remaining >= minimum - 1e-9:
            retained.append(item)
        else:
            excluded.append({
                "dataset_id": item.dataset_id,
                "fixed_anchor_remaining_s": remaining,
            })
    audit = {
        "scope": "train only after by-flight split",
        "anchor": "fixed L-1",
        "minimum_future_s": minimum,
        "input_flights": len(series),
        "retained_flights": len(retained),
        "excluded_flights": len(excluded),
        "excluded": excluded,
    }
    if verbose and excluded:
        print(
            f"  cohort     retained {len(retained)}/{len(series)} train flights with "
            f">= {minimum:g}s after fixed L-1 anchor; excluded {len(excluded)}"
        )
    if not retained:
        raise ValueError("training cohort future-duration floor removed every train flight")
    return retained, audit


def _validation_datasets(
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    minimum_anchor_index: int | None = None,
    fitted_teacher: FittedTeacherTable | None = None,
) -> dict[str, TrajectoryWindows]:
    by_airport: dict[str, list[FlightSeries]] = {}
    for item in series:
        by_airport.setdefault(item.airport or "<unknown>", []).append(item)
    return {
        airport: FixedAnchorTrajectoryWindows(
            group,
            config,
            normalizer,
            minimum_anchor_index=minimum_anchor_index,
            fitted_teacher=fitted_teacher,
        )
        for airport, group in sorted(by_airport.items())
    }


VALIDATION_DURATION_BUCKETS_S = (180.0, 360.0, 600.0)


@dataclass(frozen=True)
class ValidationBatch:
    indices: np.ndarray
    raw_batch: tuple
    bucket: str
    query_points: int


@dataclass(frozen=True)
class ValidationBatchPlan:
    """Cached, validation-only batches grouped by fixed-anchor remaining duration."""

    dataset: TrajectoryWindows
    batches: tuple[ValidationBatch, ...]
    bucket_flights: dict[str, int]
    common_truth: CommonGridTruth | None

    @property
    def flights(self) -> int:
        return len(self.dataset)

    @property
    def query_points(self) -> int:
        return sum(batch.query_points for batch in self.batches)


def _duration_bucket_label(bucket: int) -> str:
    lower = 0.0 if bucket == 0 else VALIDATION_DURATION_BUCKETS_S[bucket - 1]
    if bucket < len(VALIDATION_DURATION_BUCKETS_S):
        upper = VALIDATION_DURATION_BUCKETS_S[bucket]
        return f"({lower:g},{upper:g}]s"
    return f"({lower:g},inf)s"


def build_validation_batch_plan(
    dataset: TrajectoryWindows,
    batch_size: int,
    *,
    duration_bucketed: bool = False,
) -> ValidationBatchPlan:
    """Build each fixed validation tensor once; optionally group no-grad rows by duration."""
    if not isinstance(dataset, FixedAnchorTrajectoryWindows):
        raise TypeError("validation batching requires a fixed-anchor dataset")
    durations = np.asarray([
        dataset.series[series_index].supervision_times[-1]
        - dataset.series[series_index].times[anchor]
        for series_index, anchor in dataset.index
    ], dtype=np.float64)
    bucket_ids = (
        np.searchsorted(
            np.asarray(VALIDATION_DURATION_BUCKETS_S, dtype=np.float64),
            durations,
            side="left",
        )
        if duration_bucketed
        else np.zeros(len(dataset), dtype=np.int64)
    )
    batches: list[ValidationBatch] = []
    bucket_flights: dict[str, int] = {}
    bucket_count = len(VALIDATION_DURATION_BUCKETS_S) + 1 if duration_bucketed else 1
    for bucket in range(bucket_count):
        indices = np.flatnonzero(bucket_ids == bucket).astype(np.int64, copy=False)
        if not len(indices):
            continue
        label = _duration_bucket_label(bucket) if duration_bucketed else "unbucketed"
        bucket_flights[label] = len(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            raw_batch = dataset.batch(batch_indices)
            dense = unpack_batch(raw_batch)[-1]
            query_points = (
                int(dense.valid.sum())
                if dense is not None
                else len(batch_indices) * dataset.config.pred_len
            )
            batches.append(
                ValidationBatch(
                    indices=batch_indices,
                    raw_batch=raw_batch,
                    bucket=label,
                    query_points=query_points,
                )
            )
    covered = (
        np.concatenate([batch.indices for batch in batches])
        if batches else np.array([], dtype=np.int64)
    )
    if not np.array_equal(np.sort(covered), np.arange(len(dataset), dtype=np.int64)):
        raise ValueError("validation duration buckets must cover every row exactly once")
    return ValidationBatchPlan(
        dataset=dataset,
        batches=tuple(batches),
        bucket_flights=bucket_flights,
        common_truth=(
            fixed_anchor_common_truth(
                dataset.series,
                dataset.config,
                dataset.config.validation_common_grid_points,
            )
            if dataset.config.checkpoint_selection_metric
            == CHECKPOINT_SELECTION_COMMON_GRID_ADE
            else None
        ),
    )


@dataclass(frozen=True)
class ValidationAirportEvaluation:
    components: dict[str, float]
    replay: SplitPredictionReplay
    profile: dict[str, Any]


def _evaluate_validation_airport(
    model: nn.Module,
    plan: ValidationBatchPlan,
    device: torch.device,
    *,
    profiler: EpochProfiler | None = None,
    multipliers: ProcedureMultipliers | None = None,
) -> ValidationAirportEvaluation:
    """Evaluate both validation clocks from one model forward per cached batch."""
    dataset = plan.dataset
    names = loss_component_names(dataset.config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    replay_chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in plan.batches:
            section = profiler.section if profiler is not None else None
            data_context = section("val_data_s") if section else nullcontext()
            with data_context:
                (
                    x,
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    dynamics,
                    dense_supervision,
                ) = unpack_batch(batch.raw_batch)
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                final_time_s = final_time_s.to(device)
                flight_weights = flight_weights.to(device)
                dynamics = move_dynamics(dynamics, device)
                dense_supervision = move_fixed_dt_supervision(
                    dense_supervision, device
                )
            objective_context = (
                section("val_objective_s") if section else nullcontext()
            )
            with objective_context:
                prediction = model_forward(model, x, dynamics, future=(y, final_time_s))
                components = prediction_loss_components(
                    prediction,
                    anchor_state(x, len(dataset.config.channels)),
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    dataset.config,
                    dataset.normalizer,
                    dynamics,
                    dense_supervision,
                    multipliers=multipliers,
                )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
            selection_context = (
                section("val_checkpoint_selection_s") if section else nullcontext()
            )
            with selection_context:
                # The DEPLOYABLE replay: what the checkpoint predicts without the
                # truth's future. A latent model's objective forward above decoded a
                # posterior sample (it read the future); the replay the fixed-anchor
                # selection metrics score must be the prior top-1 decode. (The
                # objective-based selection metric would still read the posterior —
                # config refuses it for a latent run.)
                deployable = (
                    model_forward(model, x, dynamics)
                    if getattr(model, "consumes_future", False) else prediction
                )
                replay_chunks.append((
                    batch.indices,
                    _prediction_batch_replay(
                        deployable,
                        x,
                        y,
                        mask,
                        final_time_s,
                        dynamics,
                        dataset,
                    ),
                ))
    denominator = max(flight_weight_total, 1.0)
    replay = _merge_prediction_replays(replay_chunks, count=len(dataset))
    profile = {
        "flights": plan.flights,
        "batches": len(plan.batches),
        "query_points": plan.query_points,
        "wall_s": time.perf_counter() - started,
        "duration_bucket_flights": dict(plan.bucket_flights),
    }
    return ValidationAirportEvaluation(
        components={
            name: value / denominator for name, value in component_totals.items()
        },
        replay=replay,
        profile=profile,
    )


def _dataset_loss_components(
    model: nn.Module,
    dataset: TrajectoryWindows,
    device: torch.device,
    batch_size: int,
    *,
    multipliers: ProcedureMultipliers | None = None,
) -> dict[str, float]:
    names = loss_component_names(dataset.config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            (
                x,
                y,
                mask,
                final_time_s,
                flight_weights,
                dynamics,
                dense_supervision,
            ) = unpack_batch(raw_batch)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            dense_supervision = move_fixed_dt_supervision(dense_supervision, device)
            prediction = model_forward(model, x, dynamics, future=(y, final_time_s))
            components = prediction_loss_components(
                prediction,
                anchor_state(x, len(dataset.config.channels)),
                y,
                mask,
                final_time_s,
                flight_weights,
                dataset.config,
                dataset.normalizer,
                dynamics,
                dense_supervision,
                multipliers=multipliers,
            )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
    denominator = max(flight_weight_total, 1.0)
    return {name: value / denominator for name, value in component_totals.items()}


@dataclass(frozen=True)
class ValidationSelection:
    """One deterministic checkpoint-selection result over fixed-anchor validation."""

    metric: str
    value: float
    by_airport: dict[str, float]
    details_by_airport: dict[str, dict[str, Any]] = field(default_factory=dict)


def _objective_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    del model, val_sets, normalizer, config, device, precomputed_details_by_airport
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_OBJECTIVE,
        value=float(np.mean(list(val_by_airport.values()))),
        by_airport=dict(val_by_airport),
    )


def _common_grid_validation_details(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
    replays_by_airport: dict[str, SplitPredictionReplay] | None = None,
    common_truth_by_airport: dict[str, CommonGridTruth] | None = None,
) -> dict[str, dict[str, Any]]:
    del val_by_airport
    if precomputed_details_by_airport is not None:
        if set(precomputed_details_by_airport) != set(val_sets):
            raise ValueError("precomputed validation details do not match airport sets")
        return precomputed_details_by_airport
    if replays_by_airport is not None and set(replays_by_airport) != set(val_sets):
        raise ValueError("validation replays do not match airport sets")
    if (
        common_truth_by_airport is not None
        and set(common_truth_by_airport) != set(val_sets)
    ):
        raise ValueError("validation common-truth caches do not match airport sets")
    details: dict[str, dict[str, Any]] = {}
    for airport, dataset in val_sets.items():
        replay = (
            replays_by_airport[airport]
            if replays_by_airport is not None
            else _predict_split(
                model, dataset, normalizer, device, config.batch_size
            )
        )
        if config.checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE:
            block = fixed_anchor_common_grid_ade_metrics(
                dataset.series,
                config,
                replay.anchors,
                replay.predicted,
                replay.predicted_time_s,
                replay.segment_durations_s,
                points=config.validation_common_grid_points,
                common_truth=(
                    common_truth_by_airport[airport]
                    if common_truth_by_airport is not None
                    else None
                ),
            )
            details[airport] = {
                "ade_m": block["ade_m"],
                "fde_m": block["fde_m"],
                "final_time_mae_s": block["final_time_mae_s"],
                "flights": block["flights"],
            }
            continue
        block = evaluate_fixed_anchor_common_grid(
            model,
            dataset,
            normalizer,
            config,
            device,
            replay=replay,
        )
        details[airport] = {
            "ade_m": block["ade_m"],
            "fde_m": block["fde_m"],
            "dense_state_loss": block["dense_state_loss"],
            "terminal_velocity_error_mps": block[
                "terminal_velocity_error_mps"
            ],
            "final_time_mae_s": block["final_time_mae_s"],
            "flights": block["flights"],
            "arc_length_geometry_loss": block["arc_length_geometry_loss"],
            "arc_length_geometry_unweighted_loss": block[
                "arc_length_geometry_unweighted_loss"
            ],
            "arc_length_distance_mean_m": block["arc_length_distance_mean_m"],
            "arc_length_path_length_ratio": block[
                "arc_length_path_length_ratio"
            ],
            "arc_length_path_length_log_error": block[
                "arc_length_path_length_log_error"
            ],
            "arc_length_horizontal_velocity_mae_mps": block[
                "arc_length_horizontal_velocity_mae_mps"
            ],
            "arc_length_horizontal_velocity_p95_mps": block[
                "arc_length_horizontal_velocity_p95_mps"
            ],
            "arc_length_horizontal_tangent_mean": block[
                "arc_length_horizontal_tangent_mean"
            ],
            "arc_length_horizontal_tangent_p95": block[
                "arc_length_horizontal_tangent_p95"
            ],
            "arc_length_horizontal_speed_mae_mps": block[
                "arc_length_horizontal_speed_mae_mps"
            ],
            "arc_length_horizontal_speed_p95_mps": block[
                "arc_length_horizontal_speed_p95_mps"
            ],
            "arc_length_vertical_velocity_mae_mps": block[
                "arc_length_vertical_velocity_mae_mps"
            ],
            "arc_length_vertical_velocity_p95_mps": block[
                "arc_length_vertical_velocity_p95_mps"
            ],
            "arc_length_horizontal_mean_m": block[
                "arc_length_horizontal_mean_m"
            ],
            "arc_length_horizontal_p95_m": block[
                "arc_length_horizontal_p95_m"
            ],
            "arc_length_vertical_mae_m": block["arc_length_vertical_mae_m"],
            "arc_length_vertical_p95_m": block["arc_length_vertical_p95_m"],
            "arc_length_terminal_position_m": block[
                "arc_length_terminal_position_m"
            ],
            "arc_length_terminal_velocity_error_mps": block[
                "arc_length_terminal_velocity_error_mps"
            ],
            "arc_length_terminal_position_runway_components_m": block[
                "arc_length_terminal_position_runway_components_m"
            ],
            "arc_length_terminal_velocity_runway_components_mps": block[
                "arc_length_terminal_velocity_runway_components_mps"
            ],
        }
    return details


def _common_grid_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    details = _common_grid_validation_details(
        model=model, val_sets=val_sets, normalizer=normalizer, config=config,
        device=device, val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    by_airport = {
        airport: float(block["ade_m"]) for airport, block in details.items()
    }
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        value=float(np.mean(list(by_airport.values()))),
        by_airport=by_airport,
        details_by_airport=details,
    )


_VALIDATION_SELECTIONS: dict[str, Callable[..., ValidationSelection]] = {
    CHECKPOINT_SELECTION_OBJECTIVE: _objective_validation_selection,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE: _common_grid_validation_selection,
}


def fit_model(
    train_series: Sequence[FlightSeries],
    val_series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    auto_batch_size: bool = False,
    minimum_anchor_index: int | None = None,
    verbose: bool = True,
) -> FitResult:
    """Fit one model against explicit train/validation flights, without touching test."""
    if not train_series or not val_series:
        raise ValueError("fit_model requires non-empty train and validation flights")
    cohort_floor = config.training_cohort_min_future_s
    cohort_anchor = config.seq_len - 1
    cohort_ineligible = [
        item for item in train_series
        if item.supervision_times[-1] - item.times[cohort_anchor] < cohort_floor - 1e-9
    ]
    if cohort_ineligible:
        raise ValueError(
            f"fit_model received {len(cohort_ineligible)} train flight(s) below the "
            f"predeclared {cohort_floor:g} s cohort floor; filter the train cohort before fit"
        )

    if (
        config.control_command_hook != CONTROL_HOOK_OFF
        and config.control_hook_saturation == HOOK_SATURATION_HARD
    ):
        raise ValueError(
            "hard hook saturation is for inference-only arms: a clamped command has no "
            "gradient, so training would learn nothing on the clamped steps"
        )
    device = resolve_device(config.device)
    batch_size = resolve_batch_size(config, device, auto=auto_batch_size, verbose=verbose)
    config = replace(config, batch_size=batch_size)

    # Reset after the isolated auto-batch probe so probing cannot change final initialisation.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    normalizer = Normalizer.fit(train_series, balance_airports_and_flights=True)
    # The imitation term's teacher table is a TRAINING input and is opened exactly here —
    # the one place that builds supervised window sets. Every replay path (evaluate-fit,
    # the z-oracle forecast, the approach-cohort comparison) builds its own window set
    # without one and must keep working when the table is gone.
    fitted_teacher = (
        load_fitted_teacher(config.control_fitted_teacher_path)
        if config.uses_fitted_teacher
        else None
    )
    training_dataset_class = {
        False: FixedAnchorTrajectoryWindows,
        True: RandomAnchorTrajectoryWindows,
    }[config.random_train_anchor]
    train_set = training_dataset_class(
        train_series,
        config,
        normalizer,
        minimum_anchor_index=minimum_anchor_index,
        fitted_teacher=fitted_teacher,
    )
    if verbose and train_set.closure_coverage is not None:
        present, valid, total = train_set.closure_coverage
        print(f"  closure labels: {present} of {total} training flights in the file, {valid} valid "
              f"({valid / max(total, 1):.1%} regress; the rest are in the batch, out of the loss)")
    val_sets = _validation_datasets(
        val_series,
        config,
        normalizer,
        minimum_anchor_index=minimum_anchor_index,
        fitted_teacher=fitted_teacher,
    )
    val_batch_plans = {
        airport: build_validation_batch_plan(dataset, config.batch_size)
        for airport, dataset in val_sets.items()
    }
    val_common_truth_by_airport: dict[str, CommonGridTruth] | None = None
    if config.checkpoint_selection_metric == CHECKPOINT_SELECTION_COMMON_GRID_ADE:
        val_common_truth_by_airport = {}
        for airport, plan in val_batch_plans.items():
            if plan.common_truth is None:
                raise RuntimeError(
                    f"validation plan for {airport} omitted required common-grid truth"
                )
            val_common_truth_by_airport[airport] = plan.common_truth
    val_window_count = sum(len(dataset) for dataset in val_sets.values())
    if not len(train_set) or not val_window_count:
        raise ValueError(
            f"empty window set (train={len(train_set)}, val={val_window_count}) — "
            f"seq_len={config.seq_len}, minimum_anchor_index={minimum_anchor_index!r} "
            "leaves no future remainder in these tracks"
        )

    model = build_model(config, normalizer).to(device)
    multipliers = ProcedureMultipliers.from_config(config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
    )
    flights_per_epoch = sum(
        count > 0 for _start, count in train_set.series_ranges.values()
    )
    if config.random_train_anchor and flights_per_epoch != len(train_series):
        raise ValueError(
            f"random-anchor {config.random_train_anchor_min_future_s:g} s future contract "
            f"covers {flights_per_epoch}/"
            f"{len(train_series)} train flights; adjust the train roster explicitly "
            "instead of silently changing experiment membership"
        )
    if verbose:
        print(
            f"  model      {config.model}/{config.prediction_output} "
            f"({parameter_count(model):,} params) on {device}"
        )
        output_grid = {
            HORIZON_NORMALIZED: f"N={config.n_segments} normalized progress segments",
            HORIZON_FULL: (
                f"H={config.full_horizon_steps} physical {config.dt_s:g}s steps, one pass"
            ),
            HORIZON_WINDOW: (
                f"H={config.window_horizon_steps} physical {config.dt_s:g}s steps per pass"
            ),
        }[config.horizon_mode]
        print(
            f"  prediction L={config.seq_len} ({config.lookback_s:.0f}s history) -> "
            f"{output_grid} + final_time_s"
        )
        print(f"  flights    train {len(train_series)} / val {len(val_series)}")
        print(f"  windows    train {len(train_set)} / val {val_window_count} "
              "(validation anchor: fixed L-1)")
        print(f"  anchors    {train_set.anchor_description}")
        if config.random_train_anchor:
            print(
                f"  anchor min {config.random_train_anchor_min_future_s:g}s future; "
                f"eligible {flights_per_epoch}/{len(train_series)} train flights"
            )
        print(
            f"  selection  {config.checkpoint_selection_metric} on fixed L-1 validation"
        )
        if minimum_anchor_index is not None:
            print(f"  anchor     common minimum index {minimum_anchor_index} "
                  f"({minimum_anchor_index * config.dt_s:.0f}s after track entry)")
        print(
            f"  sampling   one shuffled sample/flight; {flights_per_epoch} flight(s)/epoch; "
            "airport-macro loss weights"
        )

    history: list[EpochResult] = []
    best_val = math.inf
    best_val_loss_at_selection = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_multipliers: dict[str, float] | None = None
    epochs_without_improvement = 0
    optimizer_updates = 0
    component_names = loss_component_names(config)

    for epoch in range(1, config.epochs + 1):
        profiler = EpochProfiler(device)
        epoch_start_optimizer_updates = optimizer_updates
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        train_anchor_sampling = train_set.anchor_statistics(config.seed + epoch)

        model.train()
        train_component_totals = {name: 0.0 for name in component_names}
        train_diagnostic_totals: dict[str, float] = {name: 0.0 for name in PROCEDURE_DIAGNOSTICS}
        train_weight_total = 0.0
        control_diagnostics = (
            ControlTrainingDiagnosticsAccumulator(
                config.control_gradient_clip_norm
            )
            if config.control_gradient_clip_norm > 0.0
            else None
        )
        train_batches = iter(iter_batches(
            train_set, config.batch_size, shuffle=True, seed=config.seed + epoch
        ))
        while True:
            data_started = time.perf_counter()
            try:
                raw_batch = next(train_batches)
            except StopIteration:
                break
            (
                x,
                y,
                mask,
                final_time_s,
                flight_weights,
                dynamics,
                dense_supervision,
            ) = unpack_batch(raw_batch)
            batch_count = len(flight_weights)
            batch_weight = float(flight_weights.sum())
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            dense_supervision = move_fixed_dt_supervision(dense_supervision, device)
            profiler.add_cpu_seconds(
                "train_data_s", time.perf_counter() - data_started
            )
            optimizer.zero_grad()
            with profiler.section("train_forward_s"):
                prediction = model_forward(model, x, dynamics, future=(y, final_time_s))
            with profiler.section("train_rollout_loss_s"):
                if control_diagnostics is not None:
                    if not isinstance(prediction, ControlPrediction) or dynamics is None:
                        raise RuntimeError(
                            "control gradient diagnostics require deterministic control output"
                        )
                    control_diagnostics.record_prediction(prediction, dynamics)
                components = prediction_loss_components(
                    prediction,
                    anchor_state(x, len(config.channels)),
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    config,
                    normalizer,
                    dynamics,
                    dense_supervision,
                    multipliers=multipliers,
                )
            loss = components.total
            with profiler.section("train_backward_step_s"):
                loss.backward()
                if control_diagnostics is not None:
                    control_diagnostics.record_gradients_and_clip(model)
                optimizer.step()
            optimizer_updates += 1
            for name, value in components.tensors().items():
                train_component_totals[name] += float(value.detach()) * batch_count
            for name, value in components.diagnostics.items():
                train_diagnostic_totals[name] = train_diagnostic_totals.get(name, 0.0) + float(value)
            train_weight_total += batch_weight

        model.eval()
        val_evaluations = {
            airport: _evaluate_validation_airport(
                model,
                plan,
                device,
                profiler=profiler,
                multipliers=multipliers,
            )
            for airport, plan in val_batch_plans.items()
        }
        val_components_by_airport = {
            airport: evaluation.components
            for airport, evaluation in val_evaluations.items()
        }
        # The dual update, AFTER the validation pass so this epoch's train and val
        # ``procedure`` components were both scored with the same λ (``lambda_*``); the
        # updated value (``lambda_*_next``) is what the next epoch trains with.
        procedure_epoch: dict[str, float] = {}
        if multipliers is not None:
            gated = train_diagnostic_totals["procedure_gated_rows"]
            lateral_rate = train_diagnostic_totals["procedure_lateral_violations"] / max(gated, 1.0)
            vertical_rate = train_diagnostic_totals["procedure_vertical_violations"] / max(gated, 1.0)
            procedure_epoch = {
                "train_gated_rows": gated,
                "train_lateral_violation_rate": lateral_rate,
                "train_vertical_violation_rate": vertical_rate,
                "lambda_lateral": multipliers.lateral,
                "lambda_vertical": multipliers.vertical,
            }
            epoch_multipliers = multipliers.to_dict()
            multipliers.update(lateral_rate, vertical_rate, config)
            procedure_epoch["lambda_lateral_next"] = multipliers.lateral
            procedure_epoch["lambda_vertical_next"] = multipliers.vertical
        else:
            epoch_multipliers = None
        # The command hook's epoch record: how often it was gated on and how hard it acted
        # (per-step shares over the epoch's rollouts; the step count beside them).
        hook_steps = train_diagnostic_totals.get("hook_steps", 0.0)
        hook_epoch: dict[str, float] = {}
        if hook_steps > 0.0:
            hook_epoch = {
                name.removeprefix("hook_"): value / hook_steps
                for name, value in train_diagnostic_totals.items()
                if name.startswith("hook_") and name != "hook_steps"
            }
            hook_epoch["steps"] = hook_steps
        latent_epoch: dict[str, float] = {}
        if config.latent_dim > 0:
            # Both totals are UNWEIGHTED sums over flights (control/latent.py), so they are
            # divided by the unweighted flight count they were summed over, never by the
            # airport-weighted total the objective components use.
            flights = max(train_diagnostic_totals.get("latent_flights", 0.0), 1.0)
            latent_epoch = {
                # what the objective charged (free bits applied; MC for a mixture)
                "kl_nats_per_flight": train_diagnostic_totals.get("latent_kl_nats", 0.0) / flights,
                # analytic KL per flight against the most responsible component — the
                # quantity active_units is read from
                "component_kl_nats_per_flight": (
                    train_diagnostic_totals.get("latent_component_kl_nats", 0.0) / flights
                ),
                "active_units": train_diagnostic_totals.get("latent_active_units", 0.0) / flights,
            }
        train_components = {
            name: value / max(train_weight_total, 1.0)
            for name, value in train_component_totals.items()
        }
        train_loss = sum(train_components.values())
        val_by_airport = {
            airport: sum(components.values())
            for airport, components in val_components_by_airport.items()
        }
        val_components = {
            name: float(np.mean([
                components[name] for components in val_components_by_airport.values()
            ]))
            for name in component_names
        }
        # Equal airport weight: a large/long airport cannot control early stopping alone.
        val_loss = float(np.mean(list(val_by_airport.values())))
        control_training_diagnostics = (
            control_diagnostics.summary() if control_diagnostics is not None else {}
        )
        replay_by_airport = {
            airport: evaluation.replay
            for airport, evaluation in val_evaluations.items()
        }
        selection_metrics_started = time.perf_counter()
        common_grid_details = _common_grid_validation_details(
            model=model,
            val_sets=val_sets,
            normalizer=normalizer,
            config=config,
            device=device,
            val_by_airport=val_by_airport,
            replays_by_airport=replay_by_airport,
            common_truth_by_airport=val_common_truth_by_airport,
        )
        profiler.add_cpu_seconds(
            "val_checkpoint_selection_s",
            time.perf_counter() - selection_metrics_started,
        )
        validation_selection = _VALIDATION_SELECTIONS[
            config.checkpoint_selection_metric
        ](
            model=model,
            val_sets=val_sets,
            normalizer=normalizer,
            config=config,
            device=device,
            val_by_airport=val_by_airport,
            precomputed_details_by_airport=common_grid_details,
        )
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"training diverged at epoch {epoch} (train {train_loss}, val {val_loss}) "
                f"— no checkpoint written; lower --learning-rate"
            )
        if not math.isfinite(validation_selection.value):
            raise RuntimeError(
                f"validation selection metric {validation_selection.metric} is not finite"
            )
        scheduler.step(validation_selection.value)
        timing = profiler.finish(
            optimizer_updates=optimizer_updates - epoch_start_optimizer_updates
        )
        history.append(EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=epoch_learning_rate,
            optimizer_updates=optimizer_updates,
            seconds=timing["epoch_total_s"],
            val_by_airport=val_by_airport,
            train_components=train_components,
            val_components=val_components,
            validation_selection_metric=validation_selection.metric,
            validation_selection_value=validation_selection.value,
            validation_selection_by_airport=validation_selection.by_airport,
            train_anchor_sampling=train_anchor_sampling,
            control_training_diagnostics=control_training_diagnostics,
            timing=timing,
            validation_profile_by_airport={
                airport: evaluation.profile
                for airport, evaluation in val_evaluations.items()
            },
            procedure=procedure_epoch,
            command_hook=hook_epoch,
            latent=latent_epoch,
        ))

        if validation_selection.value < best_val - 1e-9:
            best_val = validation_selection.value
            best_val_loss_at_selection = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_multipliers = epoch_multipliers
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        if verbose:
            print(f"  epoch {epoch:3d}/{config.epochs}  train {train_loss:.6f}  "
                  f"val-macro {val_loss:.6f}  lr {epoch_learning_rate:.2e}  "
                  f"updates {optimizer_updates:5d}  {history[-1].seconds:5.1f}s"
                  f"{marker}")
            if validation_selection.metric != CHECKPOINT_SELECTION_OBJECTIVE:
                print(
                    f"             checkpoint  "
                    f"{validation_selection.metric}="
                    f"{validation_selection.value:.1f}"
                )
            print(
                "             val parts  "
                + "  ".join(
                    f"{name}={val_components[name]:.4f}" for name in component_names
                )
            )
            if control_training_diagnostics:
                gradients = control_training_diagnostics["gradient_norm_pre_clip"]
                clip = control_training_diagnostics["clip"]
                saturation = control_training_diagnostics["control_saturation"]
                print(
                    "             gradients  "
                    f"total mean/max={gradients['mean']['total']:.2f}/"
                    f"{gradients['max']['total']:.2f}  "
                    f"backbone={gradients['max']['backbone']:.2f}  "
                    f"control={gradients['max']['control_head']:.2f}  "
                    f"time={gradients['max']['final_time_head']:.2f}"
                )
                print(
                    "             stability "
                    f"clip {clip['triggered_batches']}/{clip['batches']}  "
                    f"saturation={saturation['overall_rate']:.3%}"
                )

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"  early stop: {config.patience} epochs without improvement")
            break

    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)
    return FitResult(
        model=model,
        config=config,
        normalizer=normalizer,
        device=device,
        history=history,
        best_val_loss=best_val_loss_at_selection,
        best_validation_selection=best_val,
        train_windows=len(train_set),
        val_windows=val_window_count,
        # The λ the SELECTED epoch trained with, i.e. the one belonging to the restored
        # weights (the history carries the whole trajectory).
        procedure_multipliers=best_multipliers,
        fitted_teacher=fitted_teacher,
    )


def train(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    output_dir: str | Path,
    data_provenance: dict[str, Any],
    reserved_test_keys: Sequence[str] | None = None,
    data_selection: dict[str, Any] | None = None,
    auto_batch_size: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model on ``series``; write ``checkpoint.pt`` + ``history.json``.

    Returns the run summary (also embedded in the history file).
    """
    out = Path(output_dir)
    release_path = out / TEST_RELEASE_NAME
    if release_path.exists():
        raise RuntimeError(
            f"refusing to train in {out}: test release ledger {release_path} protects "
            "the checkpoint already published from this directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    if data_provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a TS arrival-data fingerprint")
    manifest_digests = provenance_manifest_digests(data_provenance)
    eligibility_digests = provenance_eligibility_digests(data_provenance)

    series = usable_series(series, config, verbose=verbose)
    train_series, val_series, test_series = split_by_flight(series, config)
    if reserved_test_keys is not None and test_series:
        raise ValueError(
            "development training received outer-test series even though sealed test "
            "identities were supplied separately"
        )
    checkpoint_test_keys = (
        list(reserved_test_keys)
        if reserved_test_keys is not None
        else [s.dataset_id for s in test_series]
    )
    train_series, training_cohort = filter_training_cohort(
        train_series, config, verbose=verbose
    )
    fit = fit_model(
        train_series,
        val_series,
        config,
        auto_batch_size=auto_batch_size,
        verbose=verbose,
    )
    model, config, normalizer, device = (
        fit.model, fit.config, fit.normalizer, fit.device
    )
    # Which teacher table supervised the imitation term: the file, its digest, its width,
    # its anchor, its airports and how many flights it carries — read off the table
    # ``fit_model`` already parsed, so the file is opened exactly once per run. Recorded
    # BESIDE `data_provenance`, never inside it: that object is compared for EQUALITY by
    # `evaluate-fit` and `freeze-test` against a provenance rebuilt from the arrival
    # manifests alone, and the teacher is a TRAINING input — no replay or prediction path
    # builds a window set that reads it, so a checkpoint must stay usable with the table
    # gone.
    fitted_teacher = None if fit.fitted_teacher is None else fit.fitted_teacher.provenance
    test_window_count = (
        None
        if reserved_test_keys is not None
        else sum(min(len(window_anchors(item, config)), 1) for item in test_series)
    )
    anchor_audit = fit.history[0].train_anchor_sampling
    training_anchor_contract = {
        key: anchor_audit[key]
        for key in (
            "policy",
            "sampling_version",
            "minimum_future_s",
            "eligibility_policy",
            "temporal_candidate_anchors",
            "eligible_candidate_anchors",
            "excluded_candidate_anchors",
        )
    }

    checkpoint_payload = {
        "target_contract": target_contract(config),
        TEST_RELEASE_PROTOCOL_FIELD: TEST_RELEASE_SCHEMA,
        "config": config.to_dict(),
        # The model INPUT contract: state channels + input-only conditioning columns.
        # Stored beside ``channels`` for the same reason — load_checkpoint refuses a
        # mismatch, so a renamed or reordered conditioning channel cannot load silently.
        "input_channels": list(config.input_channels),
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "split": {
            "train": [s.dataset_id for s in train_series],
            "val": [s.dataset_id for s in val_series],
            "test": checkpoint_test_keys,
        },
        "best_val_loss": fit.best_val_loss,
        "validation_selection": {
            "metric": config.checkpoint_selection_metric,
            "best_value": fit.best_validation_selection,
            "anchor": "fixed L-1",
            "common_grid_points": config.validation_common_grid_points,
        },
        "training_anchor_contract": training_anchor_contract,
        "training_cohort": training_cohort,
        "data_provenance": data_provenance,
        "data_selection": data_selection,
    }
    if fitted_teacher is not None:
        checkpoint_payload["fitted_teacher"] = fitted_teacher
    if fit.procedure_multipliers is not None:
        # The λ the selected epoch trained with: what a reader of the history needs to
        # weigh the logged ``procedure`` component, and where the dual run stood.
        checkpoint_payload["procedure_multipliers"] = fit.procedure_multipliers
    checkpoint_path = out / CHECKPOINT_NAME
    checkpoint_tmp = out / f"{CHECKPOINT_NAME}.tmp"
    # Freeze-test may be run by another process while fitting. Recheck immediately before
    # the first checkpoint write so an already-bound checkpoint is never replaced.
    if release_path.exists():
        raise RuntimeError(
            f"refusing to replace {checkpoint_path}: test release ledger {release_path} "
            "was created while training"
        )
    torch.save(checkpoint_payload, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    # The model artifact is complete before any derived metric/report replay starts. A
    # reporting failure can therefore be resumed with ``evaluate-fit`` without retraining.
    fit_evaluation = evaluate_fit_splits(
        model,
        train_series,
        val_series,
        normalizer,
        config,
        device,
        history=fit.history,
    )
    fit_evaluation = write_fit_evaluation(
        fit_evaluation,
        checkpoint_path=checkpoint_path,
        output_dir=out,
    )
    split_metrics = {
        split: block["metrics"]
        for split, block in fit_evaluation["splits"].items()
    }
    common_grid_metrics = {
        split: block["common_grid_metrics"]
        for split, block in fit_evaluation["splits"].items()
    }
    checkpoint_metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        TEST_RELEASE_PROTOCOL_FIELD: TEST_RELEASE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "arrival_manifests": manifest_digests,
        "random_train_anchor": config.random_train_anchor,
        "training_anchor_contract": training_anchor_contract,
        "training_cohort_min_future_s": config.training_cohort_min_future_s,
        "training_cohort_excluded_flights": training_cohort["excluded_flights"],
        "random_train_anchor_min_future_s": config.random_train_anchor_min_future_s,
        "checkpoint_selection_metric": config.checkpoint_selection_metric,
        "validation_common_grid_points": config.validation_common_grid_points,
        "horizon_mode": config.horizon_mode,
        "prediction_output": config.prediction_output,
        "aircraft_filter": config.aircraft_filter,
        "pred_len": config.pred_len,
        "full_horizon_steps": config.full_horizon_steps,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": config.lr_plateau_factor,
            "patience": config.lr_plateau_patience,
        },
        "split_sha256": {
            "train": _split_sha256(train_series),
            "val": _split_sha256(val_series),
            "test": _keys_sha256(checkpoint_test_keys),
        },
    }
    if eligibility_digests:
        checkpoint_metadata["eligibility_rosters"] = eligibility_digests
    if fitted_teacher is not None:
        checkpoint_metadata["fitted_teacher"] = fitted_teacher
    if data_selection is not None:
        selection_path = out / "data_selection.json"
        selection_tmp = out / "data_selection.json.tmp"
        selection_tmp.write_text(json.dumps(data_selection, indent=2), encoding="utf-8")
        selection_tmp.replace(selection_path)
        checkpoint_metadata["data_selection_sha256"] = file_sha256(selection_path)
    if uses_control_dynamics(config.prediction_output):
        checkpoint_metadata["control_recipe"] = control_recipe(config)
    metadata_path = out / CHECKPOINT_METADATA_NAME
    metadata_tmp = out / f"{CHECKPOINT_METADATA_NAME}.tmp"
    metadata_tmp.write_text(json.dumps(checkpoint_metadata, indent=2), encoding="utf-8")
    metadata_tmp.replace(metadata_path)

    summary = {
        "config": config.to_dict(),
        "parameters": parameter_count(model),
        "device": str(device),
        "epochs_run": len(fit.history),
        "optimizer_updates": fit.history[-1].optimizer_updates,
        "final_learning_rate": fit.history[-1].learning_rate,
        "best_val_loss": fit.best_val_loss,
        "validation_selection": {
            "metric": config.checkpoint_selection_metric,
            "best_value": fit.best_validation_selection,
            "anchor": "fixed L-1",
            "common_grid_points": config.validation_common_grid_points,
        },
        "training_anchor_contract": training_anchor_contract,
        "training_cohort": training_cohort,
        "flights": {
            "train": len(train_series),
            "val": len(val_series),
            "test": len(checkpoint_test_keys),
        },
        "windows": {
            "train": fit.train_windows,
            "val": fit_evaluation["splits"]["val"]["windows"],
            "test": test_window_count,
        },
        "metrics": split_metrics,
        "common_grid_metrics": common_grid_metrics,
        "fit_diagnostics": fit_evaluation["diagnostics"],
        "performance": _epoch_performance_summary(fit.history),
        "data_provenance": {
            "schema_version": data_provenance["schema_version"],
            "arrival_manifests": manifest_digests,
            "source_record_count": sum(
                len(entry["source_records"]) for entry in data_provenance["manifests"]
            ),
        },
        "data_selection": data_selection,
        "history": [vars(h) for h in fit.history],
    }
    if eligibility_digests:
        summary["data_provenance"]["eligibility_rosters"] = eligibility_digests
    (out / HISTORY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        for split, block in split_metrics.items():
            print(f"  {split:5s}  ADE {block['ade_m']:7.1f} m   FDE {block['fde_m']:7.1f} m   "
                  f"endpoint {block['airport_macro']['arrival_endpoint_error_m']:7.1f} m   "
                  f"cross-track p95 {block['cross_track_m']['p95_abs']:7.1f} m   "
                  f"alt p95 {block['vertical_m']['p95_abs']:6.1f} m   "
                  f"time MAE {block['final_time_s']['mae']:5.1f} s")
            raw = block["raw_kinematics"]
            observed_raw = block["raw_kinematics_observed_baseline"]
            print(
                "         raw kinematics prediction / observed baseline  "
                f"turn {raw['turn_rate_p95_deg_s']:.2f}/"
                f"{observed_raw['turn_rate_p95_deg_s']:.2f} deg/s   "
                f"accel {raw['acceleration_p95_mps2']:.2f}/"
                f"{observed_raw['acceleration_p95_mps2']:.2f} m/s²   "
                f"jerk {raw['jerk_p95_mps3']:.2f}/"
                f"{observed_raw['jerk_p95_mps3']:.2f} m/s³"
            )
        print(
            f"✓ wrote {out / CHECKPOINT_NAME}, {out / HISTORY_NAME} and "
            f"{out / FIT_EVALUATION_NAME}"
        )

    return summary


def load_checkpoint(path: str | Path) -> tuple[nn.Module, TSConfig, Normalizer, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :func:`train`."""
    # weights_only=True: the payload is tensors + primitives only (config/normalizer are
    # plain dicts and lists), so nothing here needs — or should get — pickle execution.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = TSConfig.from_dict(payload["config"])
    expected_contract = target_contract(config)
    if payload.get("target_contract") != expected_contract:
        raise ValueError(
            f"checkpoint target contract {payload.get('target_contract')!r} does not match "
            f"the configured {config.prediction_output!r} output contract "
            f"{expected_contract!r}; retrain it"
        )
    if config.channels != CHANNELS:
        raise ValueError(
            f"checkpoint channel contract {config.channels} != this build's "
            f"channels.CHANNELS {CHANNELS} — the model and normalizer index the old "
            f"contract, the data build the new one, and a same-length mismatch would load "
            f"cleanly but silently mis-map (or mis-scale: ve/vn/vu -> edot/ndot/udot was a "
            f"semantics change) every channel. Re-train, or run the matching code version."
        )
    # Checkpoints written before target conditioning existed carry no input_channels
    # key; they were trained with none, so their input contract IS the channel contract.
    stored_inputs = payload.get("input_channels", list(config.channels))
    if list(stored_inputs) != list(config.input_channels):
        raise ValueError(
            f"checkpoint input channel contract {list(stored_inputs)} != this build's "
            f"{list(config.input_channels)} for target_conditioning="
            f"{config.target_conditioning!r}, intent_conditioning="
            f"{config.intent_conditioning!r} — the conditioning columns the model was "
            "trained on are not the ones this build would feed it. Re-train, or run the "
            "matching code version."
        )
    normalizer = Normalizer.from_dict(payload["normalizer"])
    model = build_model(config)
    # ``StateOutputLayer.offset_mask`` is a pure function of the channel contract and is
    # no longer persisted; checkpoints of the 2026-09-03 generation (2f7f746 … 388574f:
    # the state-v2 anchor-relative arms) stored it. Drop that one key, keep the load
    # strict for everything else.
    state = dict(payload["model_state"])
    state.pop("offset_mask", None)
    model.load_state_dict(state)
    model.eval()
    return model, config, normalizer, payload
