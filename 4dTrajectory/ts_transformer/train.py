"""The training loop: the epoch, the cohort it runs on, and the checkpoint it writes.

Two modules under it carry what used to be inline here. What a prediction is scored
against is ``objective.py``; how a fitted model is replayed on a split and which epoch is
kept is ``validation.py``. This module drives the optimizer over the first and calls the
second once per epoch.

The checkpoint carries the config, the fitted normalizer and the flight ids of each split
alongside the weights. That is what makes inference reproducible without re-deriving
anything: ``forecast.py`` loads a checkpoint and knows the resample step, channel order,
output length/time mode, and which flights the model must not be evaluated on.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.channels import CHANNELS
from ts_transformer.batching import resolve_batch_size
from ts_transformer.config import (
    CONTROL_HOOK_OFF,
    HOOK_SATURATION_HARD,
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_METRICS,
    CHECKPOINT_SELECTION_OBJECTIVE,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    LR_PLATEAU_METRIC_OBJECTIVE,
    LR_PLATEAU_METRIC_SELECTION,
    TSConfig,
    control_recipe,
    default_anchor,
    uses_control_dynamics,
)
from ts_transformer.control.basis_fit import FittedTeacherTable, load_fitted_teacher
from ts_transformer.control.dynamics.hooks import HOOK_DIAGNOSTIC_PREFIX, HOOK_STEPS_KEY
from ts_transformer.control.training.diagnostics import ControlTrainingDiagnosticsAccumulator
from ts_transformer.data_provenance import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    provenance_eligible_set_digests,
    provenance_manifest_digests,
)
from ts_transformer.dataset import (
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    fixed_anchor_index,
    iter_batches,
    training_window_class,
    window_anchors,
)
from ts_transformer.splits import split_by_flight
from ts_transformer.evaluation_protocol import (
    TEST_RELEASE_NAME,
    TEST_RELEASE_PROTOCOL_FIELD,
    TEST_RELEASE_SCHEMA,
)
from ts_transformer.control.latent import effective_latent_beta, latent_epoch_record
from ts_transformer.fixed_anchor_validation import CommonGridTruth
from ts_transformer.models import build_model, parameter_count, resolve_device
from ts_transformer.batch_contract import anchor_state, model_forward, unpack_batch
from ts_transformer.io_utils import file_sha256
from ts_transformer.objective import (
    PROCEDURE_DIAGNOSTICS,
    ProcedureMultipliers,
    loss_component_names,
    move_dynamics,
    move_fixed_dt_supervision,
    prediction_loss_components,
    target_contract,
)
from ts_transformer.prediction_outputs import ControlPrediction
from ts_transformer.training_performance import EpochProfiler
from ts_transformer.validation import (
    VALIDATION_SELECTIONS,
    anchor_grid_coverage,
    build_anchor_grid_validation_plans,
    common_grid_validation_details,
    evaluate_validation_airport,
    predict_split,
    validation_datasets,
    build_validation_batch_plan,
    evaluate_fixed_anchor_common_grid,
    evaluate_split,
)

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
    # The anchor-grid metric's per-anchor-set record: the five sets it averages, each with
    # the flights that HAVE that anchor and the common-grid ADE over them, so the shape of
    # the anytime curve is visible during training and not only after a replay. Empty for
    # every other selection metric.
    validation_anchor_grid: dict[str, Any] = field(default_factory=dict)
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
    # The latent intent's epoch record (control/latent.latent_epoch_record): the KL charged
    # and its analytic per-dimension form, the KL's mean/variance split, the posterior mean's
    # displacement from the prior mean in prior sigmas, and the active-unit counts — the
    # posterior-collapse reading, which looks exactly like "converged" on every other number.
    latent: dict[str, Any] = field(default_factory=dict)


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
    training_objective = _training_objective_diagnostics(history, config)
    if training_objective is not None:
        diagnostics["training_objective"] = training_objective

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
    replay = predict_split(model, dataset, normalizer, device, config.batch_size)
    formal_metrics = evaluate_split(
        model, dataset, normalizer, config, device, replay=replay
    )
    common_grid_metrics = (
        formal_metrics
        if config.checkpoint_selection_metric in CHECKPOINT_SELECTION_COMMON_GRID_METRICS
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
    minimum_anchor_index: int | None = None,
    verbose: bool = True,
) -> tuple[list[FlightSeries], dict[str, Any]]:
    """Apply a predeclared future-duration floor to train flights only.

    The floor is measured from the anchor the fixed-anchor windows will be built at
    (:func:`dataset.fixed_anchor_index`): ``L-1``, or the experiment's common floor.
    """
    minimum = float(config.training_cohort_min_future_s)
    anchor = fixed_anchor_index(config, minimum_anchor_index)
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
        "anchor": "fixed L-1" if anchor == default_anchor(config) else f"fixed index {anchor}",
        "minimum_future_s": minimum,
        "input_flights": len(series),
        "retained_flights": len(retained),
        "excluded_flights": len(excluded),
        "excluded": excluded,
    }
    if verbose and excluded:
        print(
            f"  cohort     retained {len(retained)}/{len(series)} train flights with "
            f">= {minimum:g}s after the fixed anchor (index {anchor}); excluded {len(excluded)}"
        )
    if not retained:
        raise ValueError("training cohort future-duration floor removed every train flight")
    return retained, audit




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
    cohort_anchor = fixed_anchor_index(config, minimum_anchor_index)
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
    train_set = training_window_class(config)(
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
    val_sets = validation_datasets(
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
    if config.checkpoint_selection_metric in CHECKPOINT_SELECTION_COMMON_GRID_METRICS:
        val_common_truth_by_airport = {}
        for airport, plan in val_batch_plans.items():
            if plan.common_truth is None:
                raise RuntimeError(
                    f"validation plan for {airport} omitted required common-grid truth"
                )
            val_common_truth_by_airport[airport] = plan.common_truth
    # The anchor-grid metric's extra anchor sets: built once here beside the L-1 plans
    # (which are the metric's first set), replayed every epoch. The L-1 pass is NOT
    # rebuilt for them, and a candidate bin this cohort cannot cover is dropped here with
    # a printed notice rather than averaged in.
    anchor_grid_plans = (
        build_anchor_grid_validation_plans(
            val_sets, config.batch_size, minimum_anchor_index=minimum_anchor_index
        )
        if config.checkpoint_selection_metric == CHECKPOINT_SELECTION_ANCHOR_GRID_ADE
        else None
    )
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
        if config.random_train_anchor_l1_share:
            # Bounded coverage is stated, never assumed: the share reserves the anchor the
            # FIXED-anchor arms train at, and this says how many flights have no such
            # anchor (output eligibility removed it) and are reserved at their earliest
            # admissible one instead. The same count is in every epoch's record.
            print(
                f"  L-1 share  {config.random_train_anchor_l1_share:g} of the draws "
                f"reserved for anchor {default_anchor(config)}; "
                f"{train_set.flights_without_default_anchor}/{flights_per_epoch} flights "
                "store no such anchor and are reserved at their earliest admissible one"
            )
        print(
            f"  selection  {config.checkpoint_selection_metric} on fixed L-1 validation"
        )
        if anchor_grid_plans is not None:
            coverage = anchor_grid_coverage(anchor_grid_plans)
            print("  grid       L-1 + " + ", ".join(
                f"{label} ({sum(flights.values())} flights)"
                for label, flights in coverage.items()
            ) + (
                "" if not anchor_grid_plans.dropped else
                "; dropped " + ", ".join(
                    item["bin"] for item in anchor_grid_plans.dropped
                ) + f" under {anchor_grid_plans.minimum_coverage:.0%} coverage"
            ))
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
        # The objective THIS epoch optimizes: the run's config carrying the annealed KL
        # weight (identical to `config` itself once the warm-up is over, and always when
        # there is none). The training batches and the validation pass below are both
        # scored under it, so an epoch's train and val `latent_kl` mean the same thing —
        # the rule the procedure penalty's λ already follows.
        beta_effective = effective_latent_beta(config, epoch)
        epoch_config = replace(config, latent_beta=beta_effective)

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
                    epoch_config,
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
            airport: evaluate_validation_airport(
                model,
                plan,
                device,
                config=epoch_config,
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
        hook_steps = train_diagnostic_totals.get(HOOK_STEPS_KEY, 0.0)
        hook_epoch: dict[str, float] = {}
        if hook_steps > 0.0:
            hook_epoch = {
                name.removeprefix(HOOK_DIAGNOSTIC_PREFIX): value / hook_steps
                for name, value in train_diagnostic_totals.items()
                if name.startswith(HOOK_DIAGNOSTIC_PREFIX) and name != HOOK_STEPS_KEY
            }
            hook_epoch["steps"] = hook_steps
        latent_epoch: dict[str, Any] = {}
        if config.latent_dim > 0:
            latent_epoch = latent_epoch_record(
                train_diagnostic_totals, config, beta_effective=beta_effective
            )
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
        common_grid_details = common_grid_validation_details(
            model=model,
            val_sets=val_sets,
            normalizer=normalizer,
            config=config,
            device=device,
            val_by_airport=val_by_airport,
            replays_by_airport=replay_by_airport,
            common_truth_by_airport=val_common_truth_by_airport,
        )
        validation_selection = VALIDATION_SELECTIONS[
            config.checkpoint_selection_metric
        ](
            model=model,
            val_sets=val_sets,
            normalizer=normalizer,
            config=config,
            device=device,
            val_by_airport=val_by_airport,
            precomputed_details_by_airport=common_grid_details,
            anchor_grid_plans=anchor_grid_plans,
        )
        # Everything the selection metric costs, including the anchor grid's four extra
        # replays — a metric whose price is not in its own timer is a metric nobody can
        # budget for.
        profiler.add_cpu_seconds(
            "val_checkpoint_selection_s",
            time.perf_counter() - selection_metrics_started,
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
        # The plateau is measured on ONE of the two numbers this epoch already produced —
        # never on a third one computed here, so the scheduler and the record cannot
        # disagree about what stalled. Which one is `lr_plateau_metric`; the kept epoch is
        # `validation_selection` either way.
        scheduler.step({
            LR_PLATEAU_METRIC_SELECTION: validation_selection.value,
            LR_PLATEAU_METRIC_OBJECTIVE: val_loss,
        }[config.lr_plateau_metric])
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
            validation_anchor_grid=validation_selection.anchor_grid,
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
            if validation_selection.anchor_grid:
                # The curve's shape, per epoch: the mean is one number and hides which
                # anchors moved. `n` rides along because bins hold different flights.
                print("             anchor set  " + "  ".join(
                    f"{name}={block['ade_m']:.0f}(n{block['flights']})"
                    for name, block in
                    validation_selection.anchor_grid["anchor_sets"].items()
                ))
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
    eligible_sets = provenance_eligible_set_digests(data_provenance)

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
            # WHICH of the epoch record's two numbers the plateau was measured on:
            # `selection` = `validation_selection_value`, `objective` = `val_loss`. The
            # values themselves are already in every epoch; this is the link between them
            # and the learning-rate column beside them.
            "metric": config.lr_plateau_metric,
        },
        "split_sha256": {
            "train": _split_sha256(train_series),
            "val": _split_sha256(val_series),
            "test": _keys_sha256(checkpoint_test_keys),
        },
    }
    if eligible_sets:
        # The eligible SET, not the roster file's bytes: regenerating the observed
        # evaluation moves those bytes with the set unchanged (data_provenance).
        checkpoint_metadata["eligible_sets"] = eligible_sets
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
    if eligible_sets:
        summary["data_provenance"]["eligible_sets"] = eligible_sets
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


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """The stored payload alone — no model build, no contract checks.

    For a preflight that only needs the checkpoint's `data_provenance` / `data_selection`
    blocks (the publisher, the pipeline's reuse check): such a reader must not pay the
    model build, nor be refused by a model contract it never touches.
    """
    # weights_only=True: the payload is tensors + primitives only (config/normalizer are
    # plain dicts and lists), so nothing here needs — or should get — pickle execution.
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_checkpoint(path: str | Path) -> tuple[nn.Module, TSConfig, Normalizer, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :func:`train`."""
    payload = load_checkpoint_payload(path)
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
    stored_inputs = payload.get("input_channels") or list(config.channels)
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
