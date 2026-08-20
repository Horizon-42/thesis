#!/usr/bin/env python
"""Run a fold-local teacher/no-teacher ablation on sealed outer-train flights.

This is deliberately not a hyperparameter search.  It keeps the frozen ``simple-v1``
recipe unchanged, partitions only the usable outer-train population, regenerates each
fold's teacher schedules from that fold's training rows, and evaluates both arms on the
same fold validation rows.  Outer-validation and outer-test trajectory values are never
opened.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
import random
from statistics import fmean, pstdev
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import torch  # noqa: E402

from config import (  # noqa: E402
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_RECIPE_SIMPLE_V1,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    PREDICTION_CONTROL,
    TSConfig,
    control_simple_v1_overrides,
)
from dataset import (  # noqa: E402
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    arrival_data_provenance,
    build_series,
    cross_validation_folds,
    flight_keys_by_split,
    load_flight_dicts,
    provenance_manifest_digests,
)
from models import parameter_count, resolve_device  # noqa: E402
from control.oracle.evaluation import evaluate_schedule, move_dynamics  # noqa: E402
from train import prediction_loss_components  # noqa: E402
from control.oracle.optimization import (  # noqa: E402
    BatchedOracleTeacher,
    teacher_optimization_stages,
    optimize_teacher_controls,
)
from control.oracle.pretraining import CachedSchedulePretrainer  # noqa: E402
from control.oracle.targets import build_inverse_dynamics_target  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
from train import (  # noqa: E402
    evaluate_fixed_anchor_series,
    fit_model,
    unpack_batch,
    usable_series,
)
from train_only_diagnostics import rank_outer_train_candidates  # noqa: E402


RESULTS_SCHEMA = "ts-simple-v1-teacher-paired-cv-v1-outer-train-only"
CONTRACT_SCHEMA = "ts-simple-v1-teacher-paired-cv-contract-v1"
ARM_SCHEMA = "ts-simple-v1-teacher-paired-cv-arm-v1"
TEACHER_SCHEMA = "ts-oracle-teacher-fold-local-v1-outer-train-only"
RESULTS_NAME = "paired_cv_results.json"
CONTRACT_NAME = "run_contract.json"
TEACHER_NAMESPACE = "inverse-dynamics-oracle-teacher-paired-cv-v1"
ARM_ORDER = ("teacher", "no_teacher")


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact {path} must contain an object")
    return payload


def _simple_config(*, seed: int, split_seed: int, device: str) -> TSConfig:
    return TSConfig(
        control_recipe_name=CONTROL_RECIPE_SIMPLE_V1,
        seed=seed,
        split_seed=split_seed,
        device=device,
        **control_simple_v1_overrides(),
    )


def _teacher_config(*, seed: int, split_seed: int, device: str) -> TSConfig:
    """The already-audited schedule optimization recipe used by the paired run."""
    return TSConfig(
        prediction_output=PREDICTION_CONTROL,
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        control_dynamics_backend=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        random_train_anchor=False,
        n_segments=64,
        seed=seed,
        split_seed=split_seed,
        device=device,
    )


def select_fold_teacher_series(
    fold_train: Sequence[FlightSeries],
    *,
    fold_index: int,
    cohort_size: int,
    split_seed: int,
) -> tuple[FlightSeries, ...]:
    """Select a stable teacher cohort solely from one fold's training rows."""
    if cohort_size < 1:
        raise ValueError("teacher cohort size must be positive")
    if len(fold_train) < cohort_size:
        raise ValueError(
            f"fold {fold_index} has only {len(fold_train)} training flights for a "
            f"teacher cohort of {cohort_size}"
        )
    indexed = {item.dataset_id: item for item in fold_train}
    if len(indexed) != len(fold_train):
        raise ValueError(f"fold {fold_index} training identities are not unique")
    ranked = rank_outer_train_candidates(
        list(indexed),
        ranking_namespace=f"{TEACHER_NAMESPACE}:fold={fold_index}",
        split_seed=split_seed,
    )
    return tuple(indexed[key] for key in ranked[:cohort_size])


def validate_fold_isolation(
    *,
    usable_outer_train_ids: Sequence[str],
    outer_validation_ids: Sequence[str],
    outer_test_ids: Sequence[str],
    folds: Sequence[Sequence[FlightSeries]],
    teacher_ids_by_fold: Sequence[Sequence[str]],
) -> None:
    """Reject overlap, incomplete coverage, and fold-validation teacher leakage."""
    outer_train = set(usable_outer_train_ids)
    if len(outer_train) != len(usable_outer_train_ids):
        raise ValueError("usable outer-train identities are not unique")
    if outer_train & set(outer_validation_ids):
        raise ValueError("usable outer-train overlaps outer-validation identities")
    if outer_train & set(outer_test_ids):
        raise ValueError("usable outer-train overlaps outer-test identities")
    if len(folds) != len(teacher_ids_by_fold):
        raise ValueError("fold and teacher-cohort counts differ")

    fold_sets = [{item.dataset_id for item in fold} for fold in folds]
    if any(len(ids) != len(fold) for ids, fold in zip(fold_sets, folds)):
        raise ValueError("a CV fold contains duplicate flight identities")
    if set().union(*fold_sets) != outer_train:
        raise ValueError("CV folds do not cover the usable outer-train population")
    for left in range(len(fold_sets)):
        for right in range(left + 1, len(fold_sets)):
            if fold_sets[left] & fold_sets[right]:
                raise ValueError("CV validation folds overlap")
    for index, teacher_ids in enumerate(teacher_ids_by_fold):
        teachers = set(teacher_ids)
        fold_train = outer_train - fold_sets[index]
        if len(teachers) != len(teacher_ids):
            raise ValueError(f"fold {index} teacher identities are not unique")
        if not teachers <= fold_train:
            raise ValueError(
                f"fold {index} teacher cohort is not confined to fold training"
            )


@dataclass(frozen=True)
class RNGNeutralPretrainer:
    """Let pretraining change weights but not the subsequent training RNG stream."""

    delegate: Callable[..., dict[str, Any]]

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            result = self.delegate(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        result = dict(result)
        result["downstream_rng_policy"] = (
            "CPU, CUDA, NumPy and Python RNG states restored after pretraining"
        )
        return result


def _move_teacher_batch(batch: tuple[Any, ...], device: torch.device):
    x, target, weights, final_time, _flight_weights, dynamics, supervision = (
        unpack_batch(batch)
    )
    return (
        x.to(device),
        target.to(device),
        weights.to(device),
        final_time.to(device),
        move_dynamics(dynamics, device),
        supervision.to(device),
    )


def _median_schedule_metrics(
    rows: Sequence[dict[str, Any]], mode: str
) -> dict[str, float]:
    return {
        metric: float(np.median([row[mode][metric] for row in rows]))
        for metric in ("ade_m", "fde_at_last_complete_dt_m", "terminal_distance_m")
    }


def _load_completed_teacher(
    directory: Path,
    *,
    contract_sha256: str,
    fold_index: int,
    expected_ids: Sequence[str],
    n_segments: int,
) -> Path | None:
    audit_path = directory / "teacher_optimization.json"
    schedule_path = directory / "teacher_schedules.npz"
    if not audit_path.exists() and not schedule_path.exists():
        return None
    if not audit_path.is_file() or not schedule_path.is_file():
        raise ValueError(f"partial teacher artifact in {directory}; remove or repair it")
    audit = _read_json_object(audit_path)
    if (
        audit.get("schema_version") != TEACHER_SCHEMA
        or audit.get("run_contract_sha256") != contract_sha256
        or audit.get("fold") != fold_index
        or audit.get("dataset_ids") != list(expected_ids)
        or audit.get("schedule_sha256") != _file_sha256(schedule_path)
    ):
        raise ValueError(f"teacher artifact {directory} does not match this run contract")
    with np.load(schedule_path, allow_pickle=False) as source:
        actual_ids = [str(value) for value in source["dataset_ids"].tolist()]
        controls = source["controls"]
        durations = source["segment_durations_s"]
    expected_shape = (len(expected_ids), n_segments)
    if (
        actual_ids != list(expected_ids)
        or controls.shape != (*expected_shape, 3)
        or durations.shape != expected_shape
    ):
        raise ValueError(f"teacher schedule {schedule_path} has an invalid shape or roster")
    return schedule_path


def optimize_fold_teacher(
    teacher_series: Sequence[FlightSeries],
    *,
    config: TSConfig,
    directory: Path,
    fold_index: int,
    contract_sha256: str,
    prefix_steps: int,
    full_steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    log_every: int,
) -> Path:
    """Optimize and atomically cache one fold-local schedule cohort."""
    expected_ids = [item.dataset_id for item in teacher_series]
    completed = _load_completed_teacher(
        directory,
        contract_sha256=contract_sha256,
        fold_index=fold_index,
        expected_ids=expected_ids,
        n_segments=int(config.n_segments),
    )
    if completed is not None:
        print(f"  fold {fold_index + 1}: restored teacher schedules from {completed}")
        return completed

    directory.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer.fit(teacher_series, balance_airports_and_flights=True)
    dataset = FixedAnchorTrajectoryWindows(teacher_series, config, normalizer)
    if len(dataset) != len(teacher_series):
        raise ValueError(
            f"fold {fold_index} teacher fixed-anchor dataset covers "
            f"{len(dataset)}/{len(teacher_series)} flights"
        )
    targets = [build_inverse_dynamics_target(dataset, index) for index in range(len(dataset))]
    device = resolve_device(config.device)
    x, target, weights, final_time, dynamics, supervision = _move_teacher_batch(
        dataset.batch(np.arange(len(dataset))), device
    )
    initial_controls = torch.as_tensor(
        np.stack([item.controls for item in targets]), dtype=torch.float32, device=device
    )
    teacher = BatchedOracleTeacher(
        initial_controls,
        dynamics["control_lower"],
        dynamics["control_upper"],
        final_time,
    ).to(device)
    stages = teacher_optimization_stages(prefix_steps, full_steps)
    history = optimize_teacher_controls(
        teacher,
        x=x,
        target=target,
        weights=weights,
        final_time_s=final_time,
        dynamics=dynamics,
        supervision=supervision,
        config=config,
        normalizer=normalizer,
        stages=stages,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        log_every=log_every,
        loss_components=prediction_loss_components,
    )
    with torch.no_grad():
        optimized = teacher()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(targets):
        initial_metrics = evaluate_schedule(
            item.prediction(device), dataset, index, config, device
        )
        optimized_prediction = ControlPrediction(
            controls=optimized.controls[index : index + 1],
            segment_durations=optimized.segment_durations[index : index + 1],
            final_time_s=optimized.final_time_s[index : index + 1],
        )
        optimized_metrics = evaluate_schedule(
            optimized_prediction, dataset, index, config, device
        )
        rows.append(
            {
                "dataset_id": item.dataset_id,
                "initial": initial_metrics,
                "optimized": optimized_metrics,
            }
        )
        print(
            f"  teacher {index + 1:02d}/{len(targets)} {item.dataset_id}: "
            f"ADE {initial_metrics['ade_m']:.1f} -> {optimized_metrics['ade_m']:.1f} m"
        )

    controls = optimized.controls.detach().cpu().numpy()
    durations = optimized.segment_durations.detach().cpu().numpy()
    temporary = directory / "teacher_schedules.tmp.npz"
    schedule_path = directory / "teacher_schedules.npz"
    np.savez_compressed(
        temporary,
        dataset_ids=np.asarray(expected_ids),
        controls=controls,
        segment_durations_s=durations,
    )
    temporary.replace(schedule_path)
    audit = {
        "schema_version": TEACHER_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "fold": fold_index,
        "scope": "fold training only",
        "dataset_ids": expected_ids,
        "dataset_ids_sha256": _ids_sha256(expected_ids),
        "schedule_sha256": _file_sha256(schedule_path),
        "config": config.to_dict(),
        "recipe": {
            "initialization": "inverse-dynamics",
            "duration": "uniform true fold-train final time / N; frozen",
            "objective": "production arc-length-geometry 2+4",
            "stages": [vars(stage) for stage in stages],
            "learning_rate": learning_rate,
            "gradient_clip_norm": gradient_clip_norm,
        },
        "median_metrics": {
            "initial": _median_schedule_metrics(rows, "initial"),
            "optimized": _median_schedule_metrics(rows, "optimized"),
        },
        "history": history,
        "flights": rows,
    }
    _write_json_atomic(directory / "teacher_optimization.json", audit)
    del teacher, optimized, x, target, weights, final_time, dynamics, supervision
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return schedule_path


def _best_epoch(fit: Any) -> Any:
    eligible = [
        row
        for row in fit.history
        if not row.training_stage or row.training_stage.get("is_full_horizon", True)
    ]
    return min(
        eligible,
        key=lambda row: (
            row.validation_selection_value
            if row.validation_selection_value is not None
            else row.val_loss
        ),
    )


def _metric_snapshot(metrics: dict[str, Any]) -> dict[str, float | int]:
    return {
        "ade_m": float(metrics["ade_m"]),
        "fde_m": float(metrics["fde_m"]),
        "arrival_endpoint_error_mean_m": float(
            metrics["arrival_endpoint_error_m"]["mean"]
        ),
        "cross_track_p95_abs_m": float(metrics["cross_track_m"]["p95_abs"]),
        "vertical_p95_abs_m": float(metrics["vertical_m"]["p95_abs"]),
        "final_time_mae_s": float(metrics["final_time_s"]["mae"]),
        "invalid_flights": int(metrics["invalid_flights"]),
    }


def _load_completed_arm(
    path: Path,
    *,
    contract_sha256: str,
    fold_index: int,
    arm: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json_object(path)
    if (
        payload.get("schema_version") != ARM_SCHEMA
        or payload.get("run_contract_sha256") != contract_sha256
        or payload.get("fold") != fold_index
        or payload.get("arm") != arm
        or payload.get("train_sha256") != _ids_sha256(train_ids)
        or payload.get("validation_sha256") != _ids_sha256(val_ids)
    ):
        raise ValueError(f"completed arm {path} does not match this run contract")
    value = payload.get("validation_metrics", {}).get("ade_m")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"completed arm {path} has no finite validation ADE")
    return payload


def fit_arm(
    *,
    arm: str,
    fold_index: int,
    fold_train: Sequence[FlightSeries],
    fold_val: Sequence[FlightSeries],
    config: TSConfig,
    schedule_path: Path,
    output_path: Path,
    contract_sha256: str,
    pretraining_steps: int,
    pretraining_learning_rate: float,
    pretraining_gradient_clip_norm: float,
) -> dict[str, Any]:
    """Fit one paired arm and persist enough evidence to resume at arm boundaries."""
    train_ids = [item.dataset_id for item in fold_train]
    val_ids = [item.dataset_id for item in fold_val]
    completed = _load_completed_arm(
        output_path,
        contract_sha256=contract_sha256,
        fold_index=fold_index,
        arm=arm,
        train_ids=train_ids,
        val_ids=val_ids,
    )
    if completed is not None:
        print(
            f"  fold {fold_index + 1} {arm}: restored completed arm "
            f"(ADE {completed['validation_metrics']['ade_m']:.1f} m)"
        )
        return completed

    pretrainer = None
    if arm == "teacher":
        pretrainer = RNGNeutralPretrainer(
            CachedSchedulePretrainer(
                schedule_path=schedule_path,
                steps=pretraining_steps,
                learning_rate=pretraining_learning_rate,
                gradient_clip_norm=pretraining_gradient_clip_norm,
            )
        )
    elif arm != "no_teacher":
        raise ValueError(f"unknown paired arm {arm!r}")

    started = time.monotonic()
    fit = fit_model(
        fold_train,
        fold_val,
        config,
        auto_batch_size=False,
        verbose=True,
        model_pretrainer=pretrainer,
    )
    evaluation = evaluate_fixed_anchor_series(
        fit.model,
        fold_val,
        fit.normalizer,
        fit.config,
        fit.device,
        split_name=f"fold-{fold_index}-validation",
    )
    best = _best_epoch(fit)
    result = {
        "schema_version": ARM_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "fold": fold_index,
        "arm": arm,
        "leakage_guard": {
            "training_population": "this CV fold's outer-train training subset",
            "evaluation_population": "this CV fold's outer-train validation subset",
            "outer_validation_values_used": False,
            "outer_test_values_used": False,
        },
        "config": fit.config.to_dict(),
        "parameters": parameter_count(fit.model),
        "train_flights": len(fold_train),
        "validation_flights": len(fold_val),
        "train_sha256": _ids_sha256(train_ids),
        "validation_sha256": _ids_sha256(val_ids),
        "train_windows": fit.train_windows,
        "validation_windows": fit.val_windows,
        "epochs_run": len(fit.history),
        "best_epoch": int(best.epoch),
        "best_validation_selection": float(fit.best_validation_selection),
        "validation_selection_metric": fit.config.checkpoint_selection_metric,
        "validation_metrics": _metric_snapshot(evaluation["metrics"]),
        "model_pretraining": fit.model_pretraining,
        "wall_time_s": float(time.monotonic() - started),
        "history": [vars(row) for row in fit.history],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, result)
    print(
        f"  fold {fold_index + 1} {arm}: "
        f"ADE {result['validation_metrics']['ade_m']:.1f} m, "
        f"best epoch {result['best_epoch']}, epochs {result['epochs_run']}"
    )
    del fit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_pairs(fold_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired fold effects without treating three folds as independent tests."""
    metric_names = tuple(
        key
        for key in fold_results[0]["arms"]["teacher"]["validation_metrics"]
        if key != "invalid_flights"
    )
    metrics: dict[str, Any] = {}
    for metric in metric_names:
        teacher = [
            float(row["arms"]["teacher"]["validation_metrics"][metric])
            for row in fold_results
        ]
        baseline = [
            float(row["arms"]["no_teacher"]["validation_metrics"][metric])
            for row in fold_results
        ]
        improvements = [base - taught for base, taught in zip(baseline, teacher)]
        percentages = [
            100.0 * improvement / base if base != 0.0 else 0.0
            for improvement, base in zip(improvements, baseline)
        ]
        metrics[metric] = {
            "teacher_by_fold": teacher,
            "no_teacher_by_fold": baseline,
            "teacher_mean": fmean(teacher),
            "no_teacher_mean": fmean(baseline),
            "paired_improvement_by_fold": improvements,
            "paired_improvement_mean": fmean(improvements),
            "paired_improvement_std_population": pstdev(improvements),
            "paired_percent_improvement_by_fold": percentages,
            "paired_percent_improvement_mean": fmean(percentages),
            "teacher_wins": sum(value > 0.0 for value in improvements),
            "folds": len(fold_results),
        }
    return metrics


def _build_contract(
    *,
    simple_config: TSConfig,
    teacher_base_config: TSConfig,
    provenance: dict[str, Any],
    split_keys: dict[str, list[str]],
    outer_train: Sequence[FlightSeries],
    folds: Sequence[Sequence[FlightSeries]],
    teacher_ids_by_fold: Sequence[Sequence[str]],
    cohort_size: int,
    prefix_steps: int,
    full_steps: int,
    teacher_learning_rate: float,
    teacher_gradient_clip_norm: float,
    pretraining_steps: int,
    pretraining_learning_rate: float,
    pretraining_gradient_clip_norm: float,
) -> dict[str, Any]:
    outer_train_ids = [item.dataset_id for item in outer_train]
    fold_rows = []
    for index, fold_val in enumerate(folds):
        val_ids = [item.dataset_id for item in fold_val]
        val_set = set(val_ids)
        train_ids = [key for key in outer_train_ids if key not in val_set]
        fold_rows.append(
            {
                "fold": index,
                "seed": simple_config.seed + index,
                "train_flights": len(train_ids),
                "validation_flights": len(val_ids),
                "train_sha256": _ids_sha256(train_ids),
                "validation_sha256": _ids_sha256(val_ids),
                "teacher_dataset_ids": list(teacher_ids_by_fold[index]),
                "teacher_dataset_ids_sha256": _ids_sha256(
                    teacher_ids_by_fold[index]
                ),
            }
        )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "fixed simple-v1 teacher/no-teacher paired CV",
        "simple_config": simple_config.to_dict(),
        "teacher_optimization_config": teacher_base_config.to_dict(),
        "arrival_manifests": provenance_manifest_digests(provenance),
        "outer_split_identity_only": {
            name: {
                "flights": len(keys),
                "sha256": _ids_sha256(keys),
            }
            for name, keys in split_keys.items()
        },
        "usable_outer_train": {
            "flights": len(outer_train_ids),
            "sha256": _ids_sha256(outer_train_ids),
        },
        "n_splits": len(folds),
        "folds": fold_rows,
        "arms": list(ARM_ORDER),
        "teacher_optimization": {
            "cohort_size": cohort_size,
            "selection_namespace": TEACHER_NAMESPACE,
            "prefix_steps_per_stage": prefix_steps,
            "full_steps": full_steps,
            "learning_rate": teacher_learning_rate,
            "gradient_clip_norm": teacher_gradient_clip_norm,
        },
        "teacher_pretraining": {
            "steps": pretraining_steps,
            "learning_rate": pretraining_learning_rate,
            "gradient_clip_norm": pretraining_gradient_clip_norm,
            "downstream_rng_policy": "restore after pretraining",
        },
        "leakage_guard": {
            "trajectory_values_loaded": "usable outer-train only",
            "outer_validation_values_used": False,
            "outer_test_values_used": False,
            "teacher_scope": "fold training only",
        },
    }
    return json.loads(json.dumps(contract, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-prefix-steps", type=int, default=30)
    parser.add_argument("--teacher-full-steps", type=int, default=150)
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-gradient-clip-norm", type=float, default=20.0)
    parser.add_argument("--teacher-log-every", type=int, default=10)
    parser.add_argument("--pretraining-steps", type=int, default=1000)
    parser.add_argument("--pretraining-learning-rate", type=float, default=1e-4)
    parser.add_argument("--pretraining-gradient-clip-norm", type=float, default=20.0)
    args = parser.parse_args(argv)

    for name in (
        "folds",
        "cohort_size",
        "teacher_prefix_steps",
        "teacher_full_steps",
        "teacher_log_every",
        "pretraining_steps",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    for name in (
        "teacher_learning_rate",
        "teacher_gradient_clip_norm",
        "pretraining_learning_rate",
        "pretraining_gradient_clip_norm",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    airport = args.airport.strip().upper()
    manifest = (
        args.data.expanduser().resolve()
        if args.data is not None
        else (
            REPO_ROOT
            / "trajectory_data_process"
            / "outputs"
            / "harvest"
            / airport
            / "arrivals"
            / "manifest.json"
        ).resolve()
    )
    if not manifest.is_file():
        parser.error(f"missing arrivals manifest {manifest}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_config = _simple_config(
        seed=args.seed, split_seed=args.split_seed, device=args.device
    )
    teacher_base_config = _teacher_config(
        seed=args.seed, split_seed=args.split_seed, device=args.device
    )
    provenance = arrival_data_provenance([manifest])
    split_keys = flight_keys_by_split(provenance, simple_config)
    print(
        f"loading only {len(split_keys['train'])} outer-train rostered arrivals; "
        "outer-validation/test source tracks stay closed"
    )
    built, report = build_series(
        load_flight_dicts([manifest], include_flight_keys=set(split_keys["train"])),
        simple_config,
        airport=airport,
        aircraft_type=simple_config.aircraft_type,
    )
    print(report.format())
    outer_train = usable_series(built, simple_config, verbose=True)
    if not outer_train:
        raise ValueError("no usable outer-train flights")
    folds = cross_validation_folds(outer_train, args.folds, seed=simple_config.seed)
    outer_train_ids = [item.dataset_id for item in outer_train]
    teacher_series_by_fold: list[tuple[FlightSeries, ...]] = []
    for fold_index, fold_val in enumerate(folds):
        val_ids = {item.dataset_id for item in fold_val}
        fold_train = [item for item in outer_train if item.dataset_id not in val_ids]
        teacher_series_by_fold.append(
            select_fold_teacher_series(
                fold_train,
                fold_index=fold_index,
                cohort_size=args.cohort_size,
                split_seed=simple_config.resolved_split_seed,
            )
        )
    teacher_ids_by_fold = [
        [item.dataset_id for item in cohort] for cohort in teacher_series_by_fold
    ]
    validate_fold_isolation(
        usable_outer_train_ids=outer_train_ids,
        outer_validation_ids=split_keys["val"],
        outer_test_ids=split_keys["test"],
        folds=folds,
        teacher_ids_by_fold=teacher_ids_by_fold,
    )
    contract = _build_contract(
        simple_config=simple_config,
        teacher_base_config=teacher_base_config,
        provenance=provenance,
        split_keys=split_keys,
        outer_train=outer_train,
        folds=folds,
        teacher_ids_by_fold=teacher_ids_by_fold,
        cohort_size=args.cohort_size,
        prefix_steps=args.teacher_prefix_steps,
        full_steps=args.teacher_full_steps,
        teacher_learning_rate=args.teacher_learning_rate,
        teacher_gradient_clip_norm=args.teacher_gradient_clip_norm,
        pretraining_steps=args.pretraining_steps,
        pretraining_learning_rate=args.pretraining_learning_rate,
        pretraining_gradient_clip_norm=args.pretraining_gradient_clip_norm,
    )
    contract_sha256 = _canonical_sha256(contract)
    contract_path = output_dir / CONTRACT_NAME
    if contract_path.exists():
        existing = _read_json_object(contract_path)
        if existing != contract:
            raise ValueError(
                f"existing run contract {contract_path} differs; choose a new output directory"
            )
        print(f"resuming exact run contract {contract_sha256}")
    else:
        _write_json_atomic(contract_path, contract)

    print(
        f"paired CV: {len(outer_train)} usable outer-train flights, {args.folds} folds, "
        f"teacher cohort {args.cohort_size} per fold"
    )
    fold_results: list[dict[str, Any]] = []
    for fold_index, fold_val in enumerate(folds):
        val_ids = {item.dataset_id for item in fold_val}
        fold_train = [item for item in outer_train if item.dataset_id not in val_ids]
        fold_dir = output_dir / f"fold_{fold_index}"
        print(
            f"\n=== fold {fold_index + 1}/{args.folds}: "
            f"train {len(fold_train)} / validate {len(fold_val)} ==="
        )
        teacher_config = replace(teacher_base_config, seed=args.seed + fold_index)
        schedule_path = optimize_fold_teacher(
            teacher_series_by_fold[fold_index],
            config=teacher_config,
            directory=fold_dir / "teacher_schedule",
            fold_index=fold_index,
            contract_sha256=contract_sha256,
            prefix_steps=args.teacher_prefix_steps,
            full_steps=args.teacher_full_steps,
            learning_rate=args.teacher_learning_rate,
            gradient_clip_norm=args.teacher_gradient_clip_norm,
            log_every=args.teacher_log_every,
        )
        fold_config = replace(simple_config, seed=args.seed + fold_index)
        arms: dict[str, dict[str, Any]] = {}
        for arm in ARM_ORDER:
            print(f"\n--- fold {fold_index + 1} / {arm} ---")
            arms[arm] = fit_arm(
                arm=arm,
                fold_index=fold_index,
                fold_train=fold_train,
                fold_val=fold_val,
                config=fold_config,
                schedule_path=schedule_path,
                output_path=fold_dir / arm / "result.json",
                contract_sha256=contract_sha256,
                pretraining_steps=args.pretraining_steps,
                pretraining_learning_rate=args.pretraining_learning_rate,
                pretraining_gradient_clip_norm=args.pretraining_gradient_clip_norm,
            )
        fold_results.append(
            {
                "fold": fold_index,
                "seed": fold_config.seed,
                "train_flights": len(fold_train),
                "validation_flights": len(fold_val),
                "train_sha256": _ids_sha256(
                    [item.dataset_id for item in fold_train]
                ),
                "validation_sha256": _ids_sha256(
                    [item.dataset_id for item in fold_val]
                ),
                "teacher_schedule": {
                    "path": str(schedule_path.resolve()),
                    "sha256": _file_sha256(schedule_path),
                    "dataset_ids": teacher_ids_by_fold[fold_index],
                },
                "arms": arms,
            }
        )

    results = {
        "schema_version": RESULTS_SCHEMA,
        "run_contract_sha256": contract_sha256,
        "selection_role": (
            "paired robustness evaluation of the frozen teacher initialization; "
            "not hyperparameter search and not final test"
        ),
        "leakage_guard": contract["leakage_guard"],
        "folds": fold_results,
        "paired_summary": summarize_pairs(fold_results),
    }
    _write_json_atomic(output_dir / RESULTS_NAME, results)
    primary = results["paired_summary"]["ade_m"]
    print(
        f"\n✓ paired CV complete: teacher ADE {primary['teacher_mean']:.1f} m vs "
        f"no-teacher {primary['no_teacher_mean']:.1f} m; mean paired improvement "
        f"{primary['paired_improvement_mean']:.1f} m "
        f"({primary['paired_percent_improvement_mean']:.1f}%), "
        f"wins {primary['teacher_wins']}/{primary['folds']}"
    )
    print(f"  wrote {output_dir / RESULTS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
