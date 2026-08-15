"""Leak-free hyperparameter search inside the locked outer training split.

The outer validation and test flights are counted for the audit record but never passed to
``fit_model``. Candidate selection therefore cannot observe either set; final training later
uses outer validation for early stopping, and only the frozen final checkpoint predicts test.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

import torch

from batching import resolve_batch_size
from config import (
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_CRITERIA,
    CHECKPOINT_SELECTION_OBJECTIVE,
    CHECKPOINT_SELECTION_TERMINAL_STATE,
    HORIZON_NORMALIZED,
    TSConfig,
)
from dataset import (
    FixedAnchorTrajectoryWindows, FlightSeries,
    cross_validation_folds,
    provenance_eligibility_digests,
    provenance_manifest_digests,
    split_by_flight,
)
from models import resolve_device
from train import evaluate_split, filter_training_cohort, fit_model, usable_series

RESULTS_NAME = "cv_results.json"
BEST_CONFIG_NAME = "best_config.json"
PROGRESS_NAME = "cv_candidate_progress.json"
RESULTS_SCHEMA = "ts-cross-validation-v13-candidate-checkpoints"
PROGRESS_SCHEMA = "ts-cross-validation-progress-v1"
SELECTION_METRIC = (
    "mean outer-train-fold airport-macro weighted sum of normalized state MSE, "
    "scaled final-time MSE, position/velocity displacement-consistency MSE, and "
    "terminal-position MSE"
)
SELECTION_METRIC_DESCRIPTIONS = {
    CHECKPOINT_SELECTION_OBJECTIVE: SELECTION_METRIC,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE: (
        "mean outer-train-fold airport-macro fixed-anchor common physical-time ADE"
    ),
    CHECKPOINT_SELECTION_COMMON_GRID_CRITERIA: (
        "mean outer-train-fold airport-macro smooth maximum of fixed-anchor "
        "common physical-time ADE/100 m and FDE/100 m"
    ),
    CHECKPOINT_SELECTION_TERMINAL_STATE: (
        "mean outer-train-fold airport-macro fixed-anchor weighted dense-state, "
        "terminal-position and terminal-velocity criterion"
    ),
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY: (
        "mean outer-train-fold airport-macro fixed-anchor horizontal-arc position, "
        "local-velocity and terminal-state criterion"
    ),
}
CV_PARAMETER_GRIDS: dict[str, tuple[Any, ...]] = {
    "n_segments": (64, 128, 256),
    "learning_rate": (1e-4, 3e-4, 5e-4),
    "d_model": (64, 128, 256),
    "e_layers": (2, 3),
    "n_heads": (4, 8),
    "dropout": (0.05, 0.1, 0.2),
    "weight_decay": (0.0, 1e-4),
}
CV_OVERRIDE_FIELDS = (*CV_PARAMETER_GRIDS, "d_ff")
DEFAULT_CV_PARAMETERS = (
    "n_segments",
    "learning_rate",
    "d_model",
)
DEFAULT_CV_EPOCHS = 36
DEFAULT_CV_PATIENCE = 6


def validate_cv_parameters(parameters: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(parameters)
    if not selected:
        raise ValueError("at least one CV parameter is required")
    if len(set(selected)) != len(selected):
        raise ValueError("CV parameters must not contain duplicates")
    unknown = [name for name in selected if name not in CV_PARAMETER_GRIDS]
    if unknown:
        raise ValueError(
            f"unsupported CV parameter {unknown[0]!r}; choose from "
            f"{tuple(CV_PARAMETER_GRIDS)}"
        )
    return selected


def applicable_cv_parameters(
    parameters: Sequence[str], horizon_mode: str
) -> tuple[str, ...]:
    """Return only parameters that affect the selected prediction contract."""
    selected = validate_cv_parameters(parameters)
    if horizon_mode != HORIZON_NORMALIZED:
        selected = tuple(name for name in selected if name != "n_segments")
    if not selected:
        raise ValueError(
            "n_segments only affects normalized-time output; fixed-horizon CV needs "
            "at least one applicable parameter"
        )
    return selected


def parameter_grid(parameters: Sequence[str]) -> dict[str, list[Any]]:
    selected = validate_cv_parameters(parameters)
    return {name: list(CV_PARAMETER_GRIDS[name]) for name in selected}


def _candidate_overrides(
    base: TSConfig,
    parameters: Sequence[str] = DEFAULT_CV_PARAMETERS,
) -> list[dict[str, Any]]:
    """Return the complete, architecture-valid grid for the selected parameters."""
    selected = applicable_cv_parameters(parameters, base.horizon_mode)
    candidates: list[dict[str, Any]] = []
    value_grid = (CV_PARAMETER_GRIDS[name] for name in selected)
    for values in itertools.product(*value_grid):
        overrides = dict(zip(selected, values))
        d_model = int(overrides.get("d_model", base.d_model))
        n_heads = int(overrides.get("n_heads", base.n_heads))
        if d_model % n_heads:
            continue
        if "d_model" in overrides:
            overrides["d_ff"] = d_model * 2
        candidates.append(overrides)
    return candidates


def _split_digest(series: Sequence[FlightSeries]) -> str:
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def _airport_counts(series: Sequence[FlightSeries]) -> dict[str, int]:
    return dict(sorted(Counter(item.airport or "<unknown>" for item in series).items()))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"candidate checkpoint {path} is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"candidate checkpoint {path} must contain a JSON object")
    return payload


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_audit(
    outer_train: Sequence[FlightSeries],
    outer_val: Sequence[FlightSeries],
    outer_test: Sequence[FlightSeries],
) -> dict[str, Any]:
    return {
        "train_flights": len(outer_train),
        "validation_flights": len(outer_val),
        "test_flights": len(outer_test),
        "train_sha256": _split_digest(outer_train),
        "validation_sha256": _split_digest(outer_val),
        "test_sha256": _split_digest(outer_test),
        "train_by_airport": _airport_counts(outer_train),
        "validation_by_airport": _airport_counts(outer_val),
        "test_by_airport": _airport_counts(outer_test),
    }


def _candidate_run_contract(
    *,
    base_config: TSConfig,
    outer_split: dict[str, Any],
    training_cohort: dict[str, Any],
    folds: Sequence[Sequence[FlightSeries]],
    selected_parameters: Sequence[str],
    candidates: Sequence[dict[str, Any]],
    cv_epochs: int,
    cv_patience: int,
    auto_batch_size: bool,
    data_provenance: dict[str, Any],
) -> dict[str, Any]:
    contract = {
        "progress_schema_version": PROGRESS_SCHEMA,
        "results_schema_version": RESULTS_SCHEMA,
        "base_config": base_config.to_dict(),
        "checkpoint_selection_metric": base_config.checkpoint_selection_metric,
        "outer_split": outer_split,
        "training_cohort": training_cohort,
        "folds": [
            {
                "fold": fold_index,
                "validation_flights": len(fold),
                "validation_by_airport": _airport_counts(fold),
                "validation_split_sha256": _split_digest(fold),
            }
            for fold_index, fold in enumerate(folds)
        ],
        "n_splits": len(folds),
        "tuned_parameters": list(selected_parameters),
        "candidate_overrides": list(candidates),
        "cv_epochs": cv_epochs,
        "cv_patience": cv_patience,
        "auto_batch_size": auto_batch_size,
        "arrival_manifests": provenance_manifest_digests(data_provenance),
        "eligibility_rosters": provenance_eligibility_digests(data_provenance),
    }
    # Keep the in-memory contract in the same JSON-native representation that is
    # restored on resume (notably, dataclass tuple fields become JSON arrays).
    return json.loads(json.dumps(contract, allow_nan=False))


def _require_finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate checkpoint {label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"candidate checkpoint {label} must be finite")


def _load_candidate_progress(
    path: Path,
    *,
    run_contract: dict[str, Any],
    run_contract_sha256: str,
    candidates: Sequence[dict[str, Any]],
    folds: Sequence[Sequence[FlightSeries]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json_strict(path)
    if payload.get("schema_version") != PROGRESS_SCHEMA:
        raise ValueError(
            "candidate checkpoint schema does not match the current implementation"
        )
    if (
        payload.get("run_contract_sha256") != run_contract_sha256
        or payload.get("run_contract") != run_contract
    ):
        raise ValueError(
            "candidate checkpoint does not match the current CV run contract"
        )
    completed = payload.get("completed_candidates")
    rows = payload.get("candidates")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or completed > len(candidates)
        or not isinstance(rows, list)
        or len(rows) != completed
        or payload.get("candidate_count") != len(candidates)
    ):
        raise ValueError("candidate checkpoint has an invalid completion boundary")
    for expected_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("candidate checkpoint candidate row must be an object")
        if row.get("candidate") != expected_index:
            raise ValueError("candidate checkpoint candidates must be sequential")
        if row.get("overrides") != candidates[expected_index]:
            raise ValueError("candidate checkpoint overrides do not match run contract")
        _require_finite_number(
            row.get("mean_val_macro_loss"), "mean_val_macro_loss"
        )
        _require_finite_number(
            row.get("std_val_macro_loss"), "std_val_macro_loss"
        )
        fold_rows = row.get("folds")
        if not isinstance(fold_rows, list) or len(fold_rows) != len(folds):
            raise ValueError("candidate checkpoint has incomplete fold results")
        for fold_index, fold_row in enumerate(fold_rows):
            if not isinstance(fold_row, dict) or fold_row.get("fold") != fold_index:
                raise ValueError("candidate checkpoint fold order is invalid")
            if (
                fold_row.get("validation_flights") != len(folds[fold_index])
                or fold_row.get("validation_split_sha256")
                != _split_digest(folds[fold_index])
            ):
                raise ValueError(
                    "candidate checkpoint fold does not match current CV run contract"
                )
            _require_finite_number(
                fold_row.get("best_validation_selection"),
                "best_validation_selection",
            )
    return rows


def _write_candidate_progress(
    path: Path,
    *,
    run_contract: dict[str, Any],
    run_contract_sha256: str,
    candidate_count: int,
    candidate_results: Sequence[dict[str, Any]],
) -> None:
    _write_json_atomic(path, {
        "schema_version": PROGRESS_SCHEMA,
        "run_contract_sha256": run_contract_sha256,
        "run_contract": run_contract,
        "candidate_count": candidate_count,
        "completed_candidates": len(candidate_results),
        "candidates": list(candidate_results),
    })


def cross_validate(
    series: Sequence[FlightSeries],
    base_config: TSConfig,
    *,
    output_dir: str | Path,
    data_provenance: dict[str, Any],
    n_splits: int = 3,
    cv_parameters: Sequence[str] = DEFAULT_CV_PARAMETERS,
    cv_epochs: int = DEFAULT_CV_EPOCHS,
    cv_patience: int = DEFAULT_CV_PATIENCE,
    auto_batch_size: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Search hyperparameters on outer-train only and persist an auditable result."""
    if cv_epochs <= 0 or cv_patience <= 0:
        raise ValueError("cv_epochs and cv_patience must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    usable = usable_series(series, base_config, verbose=verbose)
    outer_train, outer_val, outer_test = split_by_flight(usable, base_config)
    outer_train, training_cohort = filter_training_cohort(
        outer_train, base_config, verbose=verbose
    )
    folds = cross_validation_folds(outer_train, n_splits, seed=base_config.seed)
    selected_parameters = applicable_cv_parameters(
        cv_parameters, base_config.horizon_mode
    )
    candidates = _candidate_overrides(base_config, selected_parameters)
    outer_split = _split_audit(outer_train, outer_val, outer_test)
    eligibility_digests = provenance_eligibility_digests(data_provenance)
    manifest_digests = provenance_manifest_digests(data_provenance)
    run_contract = _candidate_run_contract(
        base_config=base_config,
        outer_split=outer_split,
        training_cohort=training_cohort,
        folds=folds,
        selected_parameters=selected_parameters,
        candidates=candidates,
        cv_epochs=cv_epochs,
        cv_patience=cv_patience,
        auto_batch_size=auto_batch_size,
        data_provenance=data_provenance,
    )
    run_contract_sha256 = _canonical_sha256(run_contract)
    progress_path = out / PROGRESS_NAME
    candidate_results = _load_candidate_progress(
        progress_path,
        run_contract=run_contract,
        run_contract_sha256=run_contract_sha256,
        candidates=candidates,
        folds=folds,
    )

    if verbose:
        print(f"  outer split (locked): train {len(outer_train)} / val {len(outer_val)} / "
              f"test {len(outer_test)}")
        print(f"  CV isolation: {n_splits} folds over outer-train only; outer-val/test untouched")
        print(f"  candidates: {len(candidates)}; epochs {cv_epochs}; patience {cv_patience}")
        if candidate_results:
            print(
                f"  resume: {len(candidate_results)}/{len(candidates)} complete "
                f"candidate(s) restored from {progress_path}"
            )

    for candidate_index in range(len(candidate_results), len(candidates)):
        overrides = candidates[candidate_index]
        candidate_config = replace(
            base_config,
            **overrides,
            epochs=cv_epochs,
            patience=min(cv_patience, cv_epochs),
        )
        device = resolve_device(candidate_config.device)
        resolved_batch = resolve_batch_size(
            candidate_config, device, auto=auto_batch_size, verbose=verbose
        )
        candidate_config = replace(candidate_config, batch_size=resolved_batch)
        if verbose:
            print(
                f"\n  CV candidate {candidate_index + 1}/{len(candidates)}: "
                f"{overrides}"
            )

        fold_results: list[dict[str, Any]] = []
        for fold_index, fold_val in enumerate(folds):
            fold_train = [
                item
                for other_index, fold in enumerate(folds)
                if other_index != fold_index
                for item in fold
            ]
            fold_config = replace(candidate_config, seed=base_config.seed + fold_index)
            if verbose:
                print(f"  fold {fold_index + 1}/{n_splits}: train {len(fold_train)} / "
                      f"validate {len(fold_val)}")
            fit = fit_model(
                fold_train,
                fold_val,
                fold_config,
                auto_batch_size=False,
                verbose=verbose,
            )
            best_epoch = min(
                fit.history,
                key=lambda row: (
                    row.validation_selection_value
                    if getattr(row, "validation_selection_value", None) is not None
                    else row.val_loss
                ),
            )
            best_selection = getattr(
                fit, "best_validation_selection", fit.best_val_loss
            )
            validation_metrics = evaluate_split(
                fit.model,
                FixedAnchorTrajectoryWindows(fold_val, fit.config, fit.normalizer),
                fit.normalizer,
                fit.config,
                fit.device,
            )
            fold_results.append({
                "fold": fold_index,
                "train_flights": len(fold_train),
                "validation_flights": len(fold_val),
                "validation_by_airport": _airport_counts(fold_val),
                "validation_split_sha256": _split_digest(fold_val),
                "best_val_macro_loss": fit.best_val_loss,
                "best_validation_selection": best_selection,
                "validation_selection_metric": candidate_config.checkpoint_selection_metric,
                "best_epoch": best_epoch.epoch,
                "val_by_airport": best_epoch.val_by_airport,
                "batch_size": fit.config.batch_size,
                # Independent physical-unit diagnostics. Candidate selection remains the
                # declared airport-macro objective above; these fields make accuracy and
                # smoothness trade-offs auditable without changing that objective.
                "validation_metrics": validation_metrics,
            })
            del fit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores = [row["best_validation_selection"] for row in fold_results]
        candidate_results.append({
            "candidate": candidate_index,
            "overrides": overrides,
            "mean_val_macro_loss": fmean(scores),
            "std_val_macro_loss": pstdev(scores),
            "folds": fold_results,
        })
        _write_candidate_progress(
            progress_path,
            run_contract=run_contract,
            run_contract_sha256=run_contract_sha256,
            candidate_count=len(candidates),
            candidate_results=candidate_results,
        )

    best = min(
        candidate_results,
        key=lambda row: (row["mean_val_macro_loss"], row["candidate"]),
    )
    best_overrides = dict(best["overrides"])
    results = {
        "schema_version": RESULTS_SCHEMA,
        "run_contract_sha256": run_contract_sha256,
        "selection_metric": SELECTION_METRIC_DESCRIPTIONS[
            base_config.checkpoint_selection_metric
        ],
        "leakage_guard": {
            "search_population": "outer_train_only",
            "outer_validation_used": False,
            "outer_test_used": False,
        },
        "outer_split": outer_split,
        "training_cohort": training_cohort,
        "n_splits": n_splits,
        "search_strategy": "exhaustive_grid",
        "tuned_parameters": list(selected_parameters),
        "parameter_grid": parameter_grid(selected_parameters),
        "candidate_count": len(candidates),
        "cv_epochs": cv_epochs,
        "cv_patience": cv_patience,
        "auto_batch_size": auto_batch_size,
        "base_config": base_config.to_dict(),
        "arrival_manifests": manifest_digests,
        "candidates": candidate_results,
        "best_candidate": best["candidate"],
        "best_mean_val_macro_loss": best["mean_val_macro_loss"],
        "best_overrides": best_overrides,
    }
    if eligibility_digests:
        results["eligibility_rosters"] = eligibility_digests
    _write_json_atomic(out / RESULTS_NAME, results)
    _write_json_atomic(out / BEST_CONFIG_NAME, best_overrides)
    if verbose:
        print(f"✓ cross validation selected candidate {best['candidate']}: {best_overrides}")
        print(f"  wrote {out / RESULTS_NAME} and {out / BEST_CONFIG_NAME}")
    return results
