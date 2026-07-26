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
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

import torch

from batching import resolve_batch_size
from config import TSConfig
from dataset import (
    FlightSeries,
    cross_validation_folds,
    provenance_manifest_digests,
    split_by_flight,
)
from models import resolve_device
from train import fit_model, usable_series

RESULTS_NAME = "cv_results.json"
BEST_CONFIG_NAME = "best_config.json"
TUNED_FIELDS = (
    "learning_rate",
    "d_model",
    "d_ff",
    "e_layers",
    "n_heads",
    "dropout",
    "weight_decay",
)


def _candidate_overrides(base: TSConfig, max_trials: int) -> list[dict[str, Any]]:
    """A deterministic random subset of a bounded, architecture-valid grid."""
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    baseline = {field: getattr(base, field) for field in TUNED_FIELDS}
    grid: list[dict[str, Any]] = []
    for learning_rate, d_model, e_layers, n_heads, dropout, weight_decay in itertools.product(
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
        (2, 3),
        (4, 8),
        (0.05, 0.1, 0.2),
        (0.0, 1e-4),
    ):
        if d_model % n_heads:
            continue
        grid.append({
            "learning_rate": learning_rate,
            "d_model": d_model,
            "d_ff": d_model * 2,
            "e_layers": e_layers,
            "n_heads": n_heads,
            "dropout": dropout,
            "weight_decay": weight_decay,
        })
    grid = [candidate for candidate in grid if candidate != baseline]
    random.Random(base.seed).shuffle(grid)
    return [baseline, *grid[: max(0, max_trials - 1)]]


def _split_digest(series: Sequence[FlightSeries]) -> str:
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def _airport_counts(series: Sequence[FlightSeries]) -> dict[str, int]:
    return dict(sorted(Counter(item.airport or "<unknown>" for item in series).items()))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def cross_validate(
    series: Sequence[FlightSeries],
    base_config: TSConfig,
    *,
    output_dir: str | Path,
    data_provenance: dict[str, Any],
    n_splits: int = 3,
    max_trials: int = 4,
    cv_epochs: int = 12,
    cv_patience: int = 4,
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
    folds = cross_validation_folds(outer_train, n_splits, seed=base_config.seed)
    candidates = _candidate_overrides(base_config, max_trials)

    if verbose:
        print(f"  outer split (locked): train {len(outer_train)} / val {len(outer_val)} / "
              f"test {len(outer_test)}")
        print(f"  CV isolation: {n_splits} folds over outer-train only; outer-val/test untouched")
        print(f"  candidates: {len(candidates)}; epochs {cv_epochs}; patience {cv_patience}")

    trial_results: list[dict[str, Any]] = []
    for trial_index, overrides in enumerate(candidates, 1):
        trial_config = replace(
            base_config,
            **overrides,
            epochs=cv_epochs,
            patience=min(cv_patience, cv_epochs),
        )
        device = resolve_device(trial_config.device)
        resolved_batch = resolve_batch_size(
            trial_config, device, auto=auto_batch_size, verbose=verbose
        )
        trial_config = replace(trial_config, batch_size=resolved_batch)
        if verbose:
            print(f"\n  CV trial {trial_index}/{len(candidates)}: {overrides}")

        fold_results: list[dict[str, Any]] = []
        for fold_index, fold_val in enumerate(folds):
            fold_train = [
                item
                for other_index, fold in enumerate(folds)
                if other_index != fold_index
                for item in fold
            ]
            fold_config = replace(trial_config, seed=base_config.seed + fold_index)
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
            best_epoch = min(fit.history, key=lambda row: row.val_loss)
            fold_results.append({
                "fold": fold_index,
                "train_flights": len(fold_train),
                "validation_flights": len(fold_val),
                "validation_by_airport": _airport_counts(fold_val),
                "validation_split_sha256": _split_digest(fold_val),
                "best_val_macro_loss": fit.best_val_loss,
                "best_epoch": best_epoch.epoch,
                "val_by_airport": best_epoch.val_by_airport,
                "batch_size": fit.config.batch_size,
            })
            del fit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores = [row["best_val_macro_loss"] for row in fold_results]
        trial_results.append({
            "trial": trial_index - 1,
            "overrides": overrides,
            "mean_val_macro_loss": fmean(scores),
            "std_val_macro_loss": pstdev(scores),
            "folds": fold_results,
        })

    best = min(trial_results, key=lambda row: (row["mean_val_macro_loss"], row["trial"]))
    best_overrides = dict(best["overrides"])
    results = {
        "schema_version": "ts-cross-validation-v1",
        "selection_metric": "mean outer-train-fold airport-macro normalized MSE",
        "leakage_guard": {
            "search_population": "outer_train_only",
            "outer_validation_used": False,
            "outer_test_used": False,
        },
        "outer_split": {
            "train_flights": len(outer_train),
            "validation_flights": len(outer_val),
            "test_flights": len(outer_test),
            "train_sha256": _split_digest(outer_train),
            "validation_sha256": _split_digest(outer_val),
            "test_sha256": _split_digest(outer_test),
            "train_by_airport": _airport_counts(outer_train),
            "validation_by_airport": _airport_counts(outer_val),
            "test_by_airport": _airport_counts(outer_test),
        },
        "n_splits": n_splits,
        "max_trials": max_trials,
        "cv_epochs": cv_epochs,
        "cv_patience": cv_patience,
        "auto_batch_size": auto_batch_size,
        "base_config": base_config.to_dict(),
        "arrival_manifests": provenance_manifest_digests(data_provenance),
        "trials": trial_results,
        "best_trial": best["trial"],
        "best_mean_val_macro_loss": best["mean_val_macro_loss"],
        "best_overrides": best_overrides,
    }
    _write_json_atomic(out / RESULTS_NAME, results)
    _write_json_atomic(out / BEST_CONFIG_NAME, best_overrides)
    if verbose:
        print(f"✓ cross validation selected trial {best['trial']}: {best_overrides}")
        print(f"  wrote {out / RESULTS_NAME} and {out / BEST_CONFIG_NAME}")
    return results
