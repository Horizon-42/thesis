#!/usr/bin/env python
"""Compare TS history lengths with identical flights, folds, anchors, and futures."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from dataclasses import fields, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from batching import resolve_batch_size  # noqa: E402
from config import (  # noqa: E402
    COORDINATE_FRAMES,
    DEFAULT_AIRCRAFT_TYPE,
    MODELS,
    SAMPLING_AIRPORT_FLIGHT_BALANCED,
    TSConfig,
)
from dataset import (  # noqa: E402
    FlightSeries,
    TrajectoryWindows,
    arrival_data_provenance,
    build_series,
    cross_validation_folds,
    load_flight_dicts,
    provenance_manifest_digests,
    split_by_flight,
    window_anchors,
)
from models import build_model, parameter_count, resolve_device  # noqa: E402
from train import evaluate_split, fit_model, usable_series  # noqa: E402

RESULT_SCHEMA = "ts-history-length-ablation-v1-common-anchor"
RESULT_NAME = "history_length_ablation.json"
BEST_CONFIG_NAME = "best_history_length.json"
DEFAULT_SEQ_LENS = (30, 60, 90)


def _parse_seq_lens(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(token.strip()) for token in raw.split(",") if token.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seq-lens requires comma-separated integers") from exc
    if len(values) < 2 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--seq-lens requires at least two positive values")
    return values


def _parse_airports(raw: str) -> tuple[str, ...]:
    airports = tuple(sorted({token.strip().upper() for token in raw.split(",") if token.strip()}))
    if not airports:
        raise argparse.ArgumentTypeError("--airports requires at least one ICAO code")
    return airports


def _batch_size(raw: str) -> int | str:
    if raw == "auto":
        return raw
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch-size must be a positive integer or auto") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be a positive integer or auto")
    return value


def _config_overrides(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config overrides {path}: {exc}") from exc
    allowed = {field.name for field in fields(TSConfig)}
    protected = {
        "seq_len", "model", "coordinate_frame", "seed", "device", "batch_size",
        "epochs", "patience", "sampling_strategy", "train_samples_per_epoch",
        "eval_anchor_policy",
    }
    if not isinstance(payload, dict) or any(key not in allowed for key in payload):
        raise ValueError("config overrides must be a JSON object containing TSConfig fields")
    conflict = sorted(set(payload) & protected)
    if conflict:
        raise ValueError("config overrides cannot set experiment control " + conflict[0])
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _series_digest(series: Sequence[FlightSeries]) -> str:
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def _airport_counts(series: Sequence[FlightSeries]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in series:
        airport = item.airport or "<unknown>"
        counts[airport] = counts.get(airport, 0) + 1
    return dict(sorted(counts.items()))


def anchor_signature(
    series: Sequence[FlightSeries],
    config: TSConfig,
    common_anchor_index: int,
) -> tuple[int, str]:
    """Return the count and identity digest of a candidate's admissible anchors."""
    digest = hashlib.sha256()
    count = 0
    for item in sorted(series, key=lambda row: row.dataset_id):
        for anchor in window_anchors(
            item, config, minimum_anchor_index=common_anchor_index
        ):
            digest.update(f"{item.dataset_id}:{anchor}\n".encode())
            count += 1
    return count, digest.hexdigest()


def select_history_length(candidates: Sequence[dict[str, Any]]) -> int:
    """Select minimum loss; an exact tie keeps the shorter history."""
    winner = min(
        candidates,
        key=lambda row: (row["mean_val_macro_loss"], row["seq_len"]),
    )
    return int(winner["seq_len"])


def _metric_summary(folds: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "ade_m": fmean(fold["metrics"]["ade_m"] for fold in folds),
        "fde_m": fmean(fold["metrics"]["fde_m"] for fold in folds),
        "cross_track_p95_m": fmean(
            fold["metrics"]["cross_track_m"]["p95_abs"] for fold in folds
        ),
        "final_time_mae_s": fmean(
            fold["metrics"]["final_time_s"]["mae"] for fold in folds
        ),
    }


def run_history_ablation(
    series: Sequence[FlightSeries],
    base_config: TSConfig,
    *,
    seq_lens: Sequence[int],
    data_provenance: dict[str, Any],
    output_dir: Path,
    n_splits: int,
    epochs: int,
    patience: int,
    auto_batch_size: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run outer-train-only CV over L while holding the anchor population fixed."""
    candidates_l = tuple(sorted(set(seq_lens)))
    maximum_l = max(candidates_l)
    common_anchor_index = maximum_l - 1
    population_config = replace(base_config, seq_len=maximum_l)
    usable = usable_series(
        series,
        population_config,
        minimum_anchor_index=common_anchor_index,
        verbose=verbose,
    )
    outer_train, outer_val, outer_test = split_by_flight(usable, population_config)
    folds = cross_validation_folds(outer_train, n_splits, seed=base_config.seed)

    device = resolve_device(base_config.device)
    resolved_batch = resolve_batch_size(
        population_config,
        device,
        auto=auto_batch_size,
        verbose=verbose,
    )
    reference_count, reference_digest = anchor_signature(
        outer_train, population_config, common_anchor_index
    )

    if verbose:
        print(
            f"history ablation: L={','.join(map(str, candidates_l))}; "
            f"common anchor index={common_anchor_index} "
            f"({common_anchor_index * base_config.dt_s:.0f}s after entry)"
        )
        print(
            f"  outer split (locked): train {len(outer_train)} / val {len(outer_val)} / "
            f"test {len(outer_test)}"
        )
        print(
            f"  {n_splits} folds; {len(candidates_l) * n_splits} fits; "
            f"shared batch size {resolved_batch}"
        )

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, seq_len in enumerate(candidates_l):
        config = replace(
            base_config,
            seq_len=seq_len,
            batch_size=resolved_batch,
            epochs=epochs,
            patience=min(patience, epochs),
            eval_anchor_policy="first",
        )
        anchor_count, anchor_digest = anchor_signature(
            outer_train, config, common_anchor_index
        )
        if (anchor_count, anchor_digest) != (reference_count, reference_digest):
            raise RuntimeError(
                f"L={seq_len} produced a different anchor population; ablation rejected"
            )

        if verbose:
            elapsed = (seq_len - 1) * config.dt_s
            print(
                f"\n=== candidate {candidate_index + 1}/{len(candidates_l)}: "
                f"L={seq_len} ({elapsed:.0f}s observed span) ==="
            )

        fold_results: list[dict[str, Any]] = []
        for fold_index, fold_val in enumerate(folds):
            fold_train = [
                item
                for other_index, fold in enumerate(folds)
                if other_index != fold_index
                for item in fold
            ]
            fold_config = replace(config, seed=base_config.seed + fold_index)
            if verbose:
                print(
                    f"  fold {fold_index + 1}/{n_splits}: "
                    f"train {len(fold_train)} / validate {len(fold_val)}"
                )
            fit = fit_model(
                fold_train,
                fold_val,
                fold_config,
                auto_batch_size=False,
                minimum_anchor_index=common_anchor_index,
                verbose=verbose,
            )
            best_epoch = min(fit.history, key=lambda row: row.val_loss)
            validation = TrajectoryWindows(
                fold_val,
                fit.config,
                fit.normalizer,
                anchor_policy="first",
                minimum_anchor_index=common_anchor_index,
            )
            metrics = evaluate_split(
                fit.model, validation, fit.normalizer, fit.config, fit.device
            )
            fold_results.append({
                "fold": fold_index,
                "train_flights": len(fold_train),
                "validation_flights": len(fold_val),
                "validation_by_airport": _airport_counts(fold_val),
                "validation_split_sha256": _series_digest(fold_val),
                "best_val_macro_loss": fit.best_val_loss,
                "best_epoch": best_epoch.epoch,
                "val_by_airport": best_epoch.val_by_airport,
                "batch_size": fit.config.batch_size,
                "train_windows": fit.train_windows,
                "validation_windows": fit.val_windows,
                "metrics": metrics,
                "history": [vars(row) for row in fit.history],
            })
            del validation, fit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        scores = [fold["best_val_macro_loss"] for fold in fold_results]
        candidate_results.append({
            "candidate": candidate_index,
            "seq_len": seq_len,
            "history_samples": seq_len,
            "history_elapsed_s": (seq_len - 1) * base_config.dt_s,
            "mean_val_macro_loss": fmean(scores),
            "std_val_macro_loss": pstdev(scores),
            "parameters": parameter_count(build_model(config)),
            "metric_means": _metric_summary(fold_results),
            "folds": fold_results,
        })

    selected_l = select_history_length(candidate_results)
    result = {
        "schema_version": RESULT_SCHEMA,
        "selection_metric": (
            "mean outer-train-fold airport-macro joint normalized state/time MSE"
        ),
        "tie_break": "shorter history wins an exact tie",
        "selected_seq_len": selected_l,
        "selected_history_elapsed_s": (selected_l - 1) * base_config.dt_s,
        "candidate_seq_lens": list(candidates_l),
        "common_anchor": {
            "index": common_anchor_index,
            "elapsed_s_after_entry": common_anchor_index * base_config.dt_s,
            "defined_by": "max(seq_lens) - 1",
            "outer_train_window_count": reference_count,
            "outer_train_anchor_sha256": reference_digest,
        },
        "controls": {
            "n_splits": n_splits,
            "epochs": epochs,
            "patience": patience,
            "batch_size": resolved_batch,
            "train_samples_per_epoch": base_config.train_samples_per_epoch,
            "base_config": population_config.to_dict(),
        },
        "outer_split": {
            "train_flights": len(outer_train),
            "validation_flights": len(outer_val),
            "test_flights": len(outer_test),
            "train_sha256": _series_digest(outer_train),
            "validation_sha256": _series_digest(outer_val),
            "test_sha256": _series_digest(outer_test),
            "train_by_airport": _airport_counts(outer_train),
            "validation_by_airport": _airport_counts(outer_val),
            "test_by_airport": _airport_counts(outer_test),
        },
        "arrival_manifests": provenance_manifest_digests(data_provenance),
        "leakage_guard": {
            "selection_population": "outer_train_only",
            "outer_validation_used": False,
            "outer_test_used": False,
        },
        "candidates": candidate_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / RESULT_NAME, result)
    _write_json_atomic(output_dir / BEST_CONFIG_NAME, {"seq_len": selected_l})
    write_reports(output_dir, result)
    if verbose:
        print(f"\n✓ selected L={selected_l}; wrote {output_dir / RESULT_NAME}")
        print(f"  report: {output_dir / 'plots' / 'index.md'}")
    return result


def write_reports(output_dir: Path, result: dict[str, Any]) -> None:
    """Write flat tables and compact thesis-ready plots beside the JSON result."""
    candidates = result["candidates"]
    with (output_dir / "history_length_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=(
            "seq_len", "history_elapsed_s", "parameters", "mean_val_macro_loss",
            "std_val_macro_loss", "ade_m", "fde_m", "cross_track_p95_m",
            "final_time_mae_s",
        ))
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({
                "seq_len": candidate["seq_len"],
                "history_elapsed_s": candidate["history_elapsed_s"],
                "parameters": candidate["parameters"],
                "mean_val_macro_loss": candidate["mean_val_macro_loss"],
                "std_val_macro_loss": candidate["std_val_macro_loss"],
                **candidate["metric_means"],
            })

    with (output_dir / "history_length_folds.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=(
            "seq_len", "fold", "best_epoch", "best_val_macro_loss", "batch_size",
            "train_flights", "validation_flights", "train_windows",
            "validation_windows", "ade_m", "fde_m", "cross_track_p95_m",
            "final_time_mae_s",
        ))
        writer.writeheader()
        for candidate in candidates:
            for fold in candidate["folds"]:
                writer.writerow({
                    "seq_len": candidate["seq_len"],
                    "fold": fold["fold"],
                    "best_epoch": fold["best_epoch"],
                    "best_val_macro_loss": fold["best_val_macro_loss"],
                    "batch_size": fold["batch_size"],
                    "train_flights": fold["train_flights"],
                    "validation_flights": fold["validation_flights"],
                    "train_windows": fold["train_windows"],
                    "validation_windows": fold["validation_windows"],
                    "ade_m": fold["metrics"]["ade_m"],
                    "fde_m": fold["metrics"]["fde_m"],
                    "cross_track_p95_m": fold["metrics"]["cross_track_m"]["p95_abs"],
                    "final_time_mae_s": fold["metrics"]["final_time_s"]["mae"],
                })

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots / ".matplotlib-cache"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    x = [candidate["seq_len"] for candidate in candidates]
    loss = [candidate["mean_val_macro_loss"] for candidate in candidates]
    error = [candidate["std_val_macro_loss"] for candidate in candidates]
    selected = result["selected_seq_len"]

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.errorbar(x, loss, yerr=error, marker="o", capsize=5)
    axis.scatter([selected], [loss[x.index(selected)]], color="tab:red", zorder=3, label="selected")
    axis.set(xlabel="History samples L", ylabel="CV validation loss",
             title="History-length ablation at a common anchor")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(plots / f"history_length_cv_loss.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].plot(x, [row["metric_means"]["ade_m"] for row in candidates], marker="o", label="ADE")
    axes[0].plot(x, [row["metric_means"]["fde_m"] for row in candidates], marker="o", label="FDE")
    axes[0].set(xlabel="History samples L", ylabel="Metres", title="Validation displacement")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        x,
        [row["metric_means"]["final_time_mae_s"] for row in candidates],
        marker="o",
        color="tab:green",
    )
    axes[1].set(xlabel="History samples L", ylabel="Seconds", title="Final-time MAE")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(plots / f"history_length_metrics.{suffix}", dpi=180)
    plt.close(figure)

    (plots / "index.md").write_text(
        "# History-length ablation\n\n"
        f"Selected `L={selected}` at fixed anchor index "
        f"`{result['common_anchor']['index']}`.\n\n"
        "![CV loss](history_length_cv_loss.png)\n\n"
        "![Validation metrics](history_length_metrics.png)\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", type=_parse_airports, default=None,
                        help="comma-separated airports; default: all discovered K-airports")
    parser.add_argument("--seq-lens", type=_parse_seq_lens, default=DEFAULT_SEQ_LENS,
                        help="history candidates; default: 30,60,90")
    parser.add_argument("--model", choices=MODELS, default="itransformer")
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES, default="enu")
    parser.add_argument("--n-segments", type=int, default=None)
    parser.add_argument("--config-overrides", type=Path, default=None,
                        help="optional fixed TSConfig overrides, such as best_config.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=pipeline.DEFAULT_CV_EPOCHS)
    parser.add_argument("--patience", type=int, default=pipeline.DEFAULT_CV_PATIENCE)
    parser.add_argument("--samples-per-epoch", type=int,
                        default=pipeline.DEFAULT_CV_SAMPLES_PER_EPOCH)
    parser.add_argument("--batch-size", type=_batch_size, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aircraft-type", default=DEFAULT_AIRCRAFT_TYPE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for name in ("epochs", "patience", "samples_per_epoch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.n_segments is not None and args.n_segments <= 0:
        parser.error("--n-segments must be positive")
    try:
        overrides = _config_overrides(args.config_overrides)
    except ValueError as exc:
        parser.error(str(exc))

    airports = args.airports or tuple(pipeline.discover_k_airports())
    if not airports:
        parser.error(f"no K-airport arrivals found under {pipeline.HARVEST_ROOT}")
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        parser.error(f"missing arrivals manifest {missing[0]}")

    output_dir = (args.output_dir or (
        pipeline.OPT_OUTPUTS_ROOT / "POOLED" /
        f"ts_{args.model}_normalized_time_history_length_ablation"
    )).resolve()
    maximum_l = max(args.seq_lens)
    common_anchor_index = maximum_l - 1
    print(f"history-length ablation: L={','.join(map(str, args.seq_lens))}")
    print(f"airports (locked): {','.join(airports)}")
    print(
        f"common anchor: index {common_anchor_index}; all candidates predict the same future"
    )
    print(
        f"budget: {len(args.seq_lens)} candidates × {args.folds} folds = "
        f"{len(args.seq_lens) * args.folds} fits; epochs={args.epochs}, "
        f"patience={args.patience}"
    )
    print(f"output: {output_dir}")
    if args.dry_run:
        return 0

    config_values = {
        **overrides,
        "model": args.model,
        "coordinate_frame": args.coordinate_frame,
        "seq_len": maximum_l,
        "seed": args.seed,
        "device": args.device,
        "aircraft_type": args.aircraft_type,
        "sampling_strategy": SAMPLING_AIRPORT_FLIGHT_BALANCED,
        "train_samples_per_epoch": args.samples_per_epoch,
        "eval_anchor_policy": "first",
    }
    if args.n_segments is not None:
        config_values["n_segments"] = args.n_segments
    if isinstance(args.batch_size, int):
        config_values["batch_size"] = args.batch_size
    base_config = TSConfig(**config_values)

    provenance = arrival_data_provenance(manifests)
    flights = load_flight_dicts(manifests)
    series, report = build_series(
        flights,
        base_config,
        aircraft_type=base_config.aircraft_type,
    )
    print(report.format())
    if not series:
        parser.error("no usable trajectory series")
    run_history_ablation(
        series,
        base_config,
        seq_lens=args.seq_lens,
        data_provenance=provenance,
        output_dir=output_dir,
        n_splits=args.folds,
        epochs=args.epochs,
        patience=args.patience,
        auto_batch_size=args.batch_size == "auto",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
