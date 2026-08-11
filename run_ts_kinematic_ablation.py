#!/usr/bin/env python
"""Select a TS kinematic-loss weight with held-out accuracy and raw smoothness metrics.

The default screening search trains on a deterministic airport-balanced subset of the
locked outer-train split and selects on a disjoint subset of outer-validation;
``--full-outer-split`` confirms finalists on both complete outer partitions. Outer-test
identities are recorded but never evaluated. Unlike ordinary CV loss, the selection metric
does not change when the candidate loss weight changes: accuracy is physical ADE, and
smoothness is measured from raw decoded model nodes against the observed tracks' baseline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from config import DEFAULT_AIRCRAFT_TYPE, MODELS, TSConfig  # noqa: E402
from dataset import (  # noqa: E402
    FlightSeries, arrival_data_provenance, build_series, flight_keys_by_split,
    load_flight_dicts,
    provenance_manifest_digests, split_by_flight,
)
from export import accuracy_block, observed_series_metrics  # noqa: E402
from forecast import forecast_approach  # noqa: E402
from metrics import RAW_KINEMATIC_METRIC_KEYS  # noqa: E402
from train import fit_model, usable_series  # noqa: E402

RESULT_SCHEMA = "ts-kinematic-weight-ablation-v3-robust-physics-capacity-grid"
DEFAULT_WEIGHTS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)


def _parse_airports(raw: str) -> tuple[str, ...]:
    airports = tuple(sorted({token.strip().upper() for token in raw.split(",") if token.strip()}))
    if not airports:
        raise argparse.ArgumentTypeError("--airports requires at least one ICAO code")
    return airports


def _parse_weights(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--kinematic-weights requires comma-separated numbers"
        ) from exc
    if len(values) < 2 or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "--kinematic-weights requires at least two non-negative values"
        )
    return tuple(dict.fromkeys(values))


def _parse_n_segments(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--n-segment-candidates requires comma-separated integers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--n-segment-candidates must be positive")
    return tuple(dict.fromkeys(values))


def _parse_d_models(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--d-model-candidates requires comma-separated integers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("--d-model-candidates must be positive")
    return tuple(dict.fromkeys(values))


def _series_digest(series: Sequence[FlightSeries]) -> str:
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def _key_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def select_balanced_subset(
    series: Sequence[FlightSeries],
    airports: Sequence[str],
    *,
    samples_per_airport: int,
    seed: int,
    label: str,
) -> list[FlightSeries]:
    """Deterministic, airport-balanced sample from one already-locked outer split."""
    selected: list[FlightSeries] = []
    for airport in airports:
        candidates = [item for item in series if item.airport == airport]
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"kinematic-ablation:{label}:{seed}:{item.dataset_id}".encode()
            ).digest(),
        )
        if len(ordered) < samples_per_airport:
            raise ValueError(
                f"airport {airport} has only {len(ordered)} {label} flights, "
                f"need {samples_per_airport}"
            )
        selected.extend(ordered[:samples_per_airport])
    return sorted(selected, key=lambda item: item.dataset_id)


def raw_metric_ratios(accuracy: dict[str, Any]) -> dict[str, float]:
    """Prediction/observed fleet-p95 ratios on the five stable raw metrics."""
    raw = accuracy["raw_kinematics"]
    return {
        key: raw["predicted"][key]["p95"] / raw["observed_baseline"][key]["p95"]
        for key in RAW_KINEMATIC_METRIC_KEYS
    }


def smoothness_score(ratios: dict[str, float]) -> float:
    """Unitless geometric mean; every metric contributes as a relative baseline ratio."""
    return float(math.exp(np.mean([math.log(max(value, 1e-12)) for value in ratios.values()])))


def select_candidate(
    candidates: Sequence[dict[str, Any]], *, accuracy_tolerance: float,
    smoothness_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Accuracy-first knee rule, fixed before looking at experiment results.

    Admit candidates whose validation ADE is within ``accuracy_tolerance`` of the best,
    then find the lowest observed-normalized raw-kinematic score. Scores within the
    ``smoothness_tolerance`` practical-equivalence band are treated as tied and resolved by
    ADE then model size. This prevents a very smooth but wrong straight line from winning,
    while also preventing noise-level physics differences from selecting a larger model.
    """
    best_ade = min(row["validation_accuracy"]["ade_m"]["mean"] for row in candidates)
    admitted = [
        row for row in candidates
        if row["validation_accuracy"]["ade_m"]["mean"]
        <= best_ade * (1.0 + accuracy_tolerance)
    ]
    best_smoothness = min(row["smoothness_score"] for row in admitted)
    physics_equivalent = [
        row for row in admitted
        if row["smoothness_score"] <= best_smoothness * (1.0 + smoothness_tolerance)
    ]
    return min(
        physics_equivalent,
        key=lambda row: (
            row["validation_accuracy"]["ade_m"]["mean"],
            row.get("d_model", 0),
            row.get("n_segments", 0),
            row["kinematic_weight"],
            row["smoothness_score"],
        ),
    )


def _validation_accuracy(
    model: torch.nn.Module,
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer,
    device: torch.device,
) -> dict[str, Any]:
    overlap = []
    for item in series:
        forecast = forecast_approach(model, item, config, normalizer, device=device)
        overlap.append(observed_series_metrics(item, forecast))
    return accuracy_block(overlap)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_reports(output_dir: Path, result: dict[str, Any]) -> None:
    candidates = result["candidates"]
    selected = result["selected_kinematic_weight"]
    fieldnames = (
        "n_segments", "d_model", "kinematic_weight", "best_epoch", "epochs_run", "validation_ade_m",
        "validation_fde_m", "final_time_mae_s", "smoothness_score",
        *RAW_KINEMATIC_METRIC_KEYS,
    )
    with (output_dir / "kinematic_weight_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            ratios = row["raw_metric_ratios"]
            writer.writerow({
                "n_segments": row["n_segments"],
                "d_model": row["d_model"],
                "kinematic_weight": row["kinematic_weight"],
                "best_epoch": row["best_epoch"],
                "epochs_run": row["epochs_run"],
                "validation_ade_m": row["validation_accuracy"]["ade_m"]["mean"],
                "validation_fde_m": row["validation_accuracy"]["fde_m"]["mean"],
                "final_time_mae_s": row["validation_accuracy"]["final_time_s"]["mae"],
                "smoothness_score": row["smoothness_score"],
                **ratios,
            })

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots / ".matplotlib-cache"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n_values = sorted({row["n_segments"] for row in candidates})
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    for n_segments in n_values:
        for d_model in sorted({row["d_model"] for row in candidates}):
            rows_n = [
                row for row in candidates
                if row["n_segments"] == n_segments and row["d_model"] == d_model
            ]
            if not rows_n:
                continue
            weights = [row["kinematic_weight"] for row in rows_n]
            label = f"N={n_segments}, d={d_model}"
            axis.plot(weights, [row["validation_accuracy"]["ade_m"]["mean"] for row in rows_n],
                      marker="o", label=f"ADE {label}")
            axis.plot(weights, [row["validation_accuracy"]["fde_m"]["mean"] for row in rows_n],
                      marker="s", linestyle=":", label=f"FDE {label}")
    axis.set(xlabel="Kinematic loss weight", ylabel="Metres",
             title="Held-out physical accuracy")
    axis.set_xscale("symlog", linthresh=0.1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(plots / f"validation_accuracy.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for n_segments in n_values:
        for d_model in sorted({row["d_model"] for row in candidates}):
            rows_n = [
                row for row in candidates
                if row["n_segments"] == n_segments and row["d_model"] == d_model
            ]
            if rows_n:
                axis.plot(
                    [row["kinematic_weight"] for row in rows_n],
                    [row["smoothness_score"] for row in rows_n],
                    marker="o", label=f"N={n_segments}, d={d_model}",
                )
    axis.axhline(1.0, color="black", linestyle=":", label="observed baseline")
    axis.set(xlabel="Kinematic loss weight", ylabel="Prediction / observed fleet p95",
             title="Raw-node kinematic geometric-mean score")
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(plots / f"raw_kinematic_ratios.{suffix}", dpi=180)
    plt.close(figure)

    rows = []
    for row in candidates:
        accuracy = row["validation_accuracy"]
        marker = " **selected**" if (
            row["kinematic_weight"] == selected
            and row["n_segments"] == result["selected_n_segments"]
            and row["d_model"] == result["selected_d_model"]
        ) else ""
        rows.append(
            f"| {row['n_segments']} | {row['d_model']} | {row['kinematic_weight']:g}{marker} | "
            f"{row['best_epoch']}{'+' if row['epoch_search_censored'] else ''} | "
            f"{accuracy['ade_m']['mean']:.1f} | {accuracy['fde_m']['mean']:.1f} | "
            f"{accuracy['final_time_s']['mae']:.1f} | {row['smoothness_score']:.3f} |"
        )
    (plots / "index.md").write_text(
        "# Kinematic loss weight ablation\n\n"
        "Selection is outer-validation only: admit candidates within "
        f"{result['selection_rule']['accuracy_tolerance'] * 100:.0f}% of the best ADE, "
        "then minimize the geometric mean of five prediction/observed fleet-p95 "
        "raw-kinematic ratios. Scores within "
        f"{result['selection_rule']['smoothness_practical_equivalence'] * 100:.0f}% "
        "are treated as practically equivalent and resolved by ADE/model size.\n\n"
        "| N | d_model | Weight | Best epoch | ADE (m) | FDE (m) | Time MAE (s) | Raw score |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n![Validation accuracy](validation_accuracy.png)\n\n"
        "![Raw kinematic ratios](raw_kinematic_ratios.png)\n",
        encoding="utf-8",
    )


def run_ablation(
    series: Sequence[FlightSeries],
    base_config: TSConfig,
    *,
    airports: Sequence[str],
    train_per_airport: int,
    val_per_airport: int,
    weights: Sequence[float],
    n_segments_candidates: Sequence[int],
    d_model_candidates: Sequence[int],
    accuracy_tolerance: float,
    smoothness_tolerance: float,
    full_outer_split: bool,
    output_dir: Path,
    manifest_digests: dict[str, str],
    outer_split_keys: dict[str, list[str]],
    verbose: bool,
) -> dict[str, Any]:
    usable = usable_series(series, base_config, verbose=verbose)
    outer_train, outer_val, outer_test = split_by_flight(usable, base_config)
    if full_outer_split:
        train_sample, val_sample = list(outer_train), list(outer_val)
        sampling_mode = "complete outer-train and outer-validation"
    else:
        train_sample = select_balanced_subset(
            outer_train, airports, samples_per_airport=train_per_airport,
            seed=base_config.seed, label="outer-train",
        )
        val_sample = select_balanced_subset(
            outer_val, airports, samples_per_airport=val_per_airport,
            seed=base_config.seed, label="outer-validation",
        )
        sampling_mode = "deterministic airport-balanced screening subset"
    print(
        f"outer roster: train {len(outer_split_keys['train'])} / "
        f"val {len(outer_split_keys['val'])} / test {len(outer_split_keys['test'])}; "
        "opened train/validation source tracks only"
    )
    print(
        f"experiment population: train {len(train_sample)} / val {len(val_sample)} "
        f"({sampling_mode}); "
        f"test untouched"
    )

    candidates = []
    candidate_specs = [
        (n_segments, d_model, weight)
        for n_segments in n_segments_candidates
        for d_model in d_model_candidates
        for weight in weights
    ]
    for index, (n_segments, d_model, weight) in enumerate(candidate_specs):
        config = replace(
            base_config,
            n_segments=n_segments,
            d_model=d_model,
            d_ff=d_model * 2,
            kinematic_consistency_loss_weight=weight,
        )
        print(
            f"\n=== candidate {index + 1}/{len(candidate_specs)}: "
            f"N={n_segments}, d_model={d_model}, kinematic={weight:g} ==="
        )
        fit = fit_model(train_sample, val_sample, config, verbose=verbose)
        best = min(fit.history, key=lambda row: row.val_loss)
        accuracy = _validation_accuracy(
            fit.model, val_sample, fit.config, fit.normalizer, fit.device
        )
        ratios = raw_metric_ratios(accuracy)
        candidates.append({
            "n_segments": n_segments,
            "d_model": d_model,
            "kinematic_weight": weight,
            "epochs_run": len(fit.history),
            "best_epoch": best.epoch,
            "epoch_search_censored": (
                best.epoch == len(fit.history) == config.epochs
            ),
            "best_val_macro_loss": fit.best_val_loss,
            "best_val_components": best.val_components,
            "validation_accuracy": accuracy,
            "raw_metric_ratios": ratios,
            "smoothness_score": smoothness_score(ratios),
            "history": [vars(row) for row in fit.history],
        })
        del fit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected = select_candidate(
        candidates,
        accuracy_tolerance=accuracy_tolerance,
        smoothness_tolerance=smoothness_tolerance,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "purpose": "kinematic loss selection by held-out accuracy and raw-node physics",
        "selected_kinematic_weight": selected["kinematic_weight"],
        "selected_n_segments": selected["n_segments"],
        "selected_d_model": selected["d_model"],
        "recommended_epochs": selected["best_epoch"],
        "recommended_epochs_censored": selected["epoch_search_censored"],
        "selection_rule": {
            "primary": "outer-validation observed-overlap ADE",
            "accuracy_tolerance": accuracy_tolerance,
            "smoothness_practical_equivalence": smoothness_tolerance,
            "secondary": (
                "minimum geometric mean of prediction/observed fleet-p95 ratios across "
                + ", ".join(RAW_KINEMATIC_METRIC_KEYS)
            ),
            "tie_break": "lowest ADE, then smallest d_model, N, and weight",
            "test_used": False,
        },
        "controls": {
            "base_config": base_config.to_dict(),
            "weights": list(weights),
            "n_segments_candidates": list(n_segments_candidates),
            "d_model_candidates": list(d_model_candidates),
        },
        "population": {
            "sampling_mode": sampling_mode,
            "airports": list(airports),
            "train_per_airport": None if full_outer_split else train_per_airport,
            "validation_per_airport": None if full_outer_split else val_per_airport,
            "train_flights": len(train_sample),
            "validation_flights": len(val_sample),
            "train_sha256": _series_digest(train_sample),
            "validation_sha256": _series_digest(val_sample),
        },
        "outer_split": {
            "train": len(outer_split_keys["train"]),
            "validation": len(outer_split_keys["val"]),
            "test": len(outer_split_keys["test"]),
            "train_sha256": _key_digest(outer_split_keys["train"]),
            "validation_sha256": _key_digest(outer_split_keys["val"]),
            "test_sha256": _key_digest(outer_split_keys["test"]),
            "loaded_usable_train": len(outer_train),
            "loaded_usable_validation": len(outer_val),
        },
        "arrival_manifests": manifest_digests,
        "candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "kinematic_weight_ablation.json", result)
    _write_json_atomic(output_dir / "best_kinematic_config.json", {
        "n_segments": selected["n_segments"],
        "d_model": selected["d_model"],
        "d_ff": selected["d_model"] * 2,
        "kinematic_consistency_loss_weight": selected["kinematic_weight"],
        "epochs": selected["best_epoch"],
    })
    write_reports(output_dir, result)
    print(
        f"\n✓ selected N={selected['n_segments']}, "
        f"d_model={selected['d_model']}, kinematic={selected['kinematic_weight']:g}, "
        f"recommended epochs={'≥' if selected['epoch_search_censored'] else ''}"
        f"{selected['best_epoch']}"
    )
    print(f"  report: {output_dir / 'plots' / 'index.md'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="itransformer")
    parser.add_argument("--airports", type=_parse_airports, default=None)
    parser.add_argument("--kinematic-weights", type=_parse_weights, default=DEFAULT_WEIGHTS)
    parser.add_argument("--train-per-airport", type=int, default=64)
    parser.add_argument("--val-per-airport", type=int, default=16)
    parser.add_argument(
        "--full-outer-split", action="store_true",
        help="confirm candidates on every outer-train/validation flight; test remains untouched",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-segments", type=int, default=None)
    parser.add_argument("--n-segment-candidates", type=_parse_n_segments, default=None)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-model-candidates", type=_parse_d_models, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.10)
    parser.add_argument("--smoothness-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for name in (
        "train_per_airport", "val_per_airport", "epochs", "patience", "batch_size",
        "d_model",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_segments is not None and args.n_segments <= 0:
        parser.error("--n-segments must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if not 0.0 <= args.accuracy_tolerance <= 1.0:
        parser.error("--accuracy-tolerance must be in [0, 1]")
    if not 0.0 <= args.smoothness_tolerance <= 1.0:
        parser.error("--smoothness-tolerance must be in [0, 1]")

    airports = args.airports or tuple(pipeline.discover_k_airports())
    if not airports:
        parser.error("no K-airport arrivals found")
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        parser.error(f"missing arrivals manifest {missing[0]}; run prepare_scenario_inputs.py")
    output_dir = args.output_dir or (
        REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" /
        f"ts_{args.model}_kinematic_weight_ablation"
    )
    base_n_segments = args.n_segments or TSConfig(model=args.model).n_segments
    n_segment_candidates = args.n_segment_candidates or (base_n_segments,)
    d_model_candidates = args.d_model_candidates or (args.d_model,)
    population_label = (
        "full outer split"
        if args.full_outer_split
        else f"train={args.train_per_airport}/airport, val={args.val_per_airport}/airport"
    )
    print(
        f"kinematic ablation: model={args.model}, airports={','.join(airports)}, "
        f"N={','.join(map(str, n_segment_candidates))}, "
        f"d_model={','.join(map(str, d_model_candidates))}, "
        f"weights={','.join(f'{weight:g}' for weight in args.kinematic_weights)}, "
        f"{population_label}, "
        f"epochs={args.epochs}, dropout={args.dropout:g}"
    )
    print(f"output: {output_dir}")
    if args.dry_run:
        return 0

    config = TSConfig(
        model=args.model,
        n_segments=base_n_segments,
        d_model=args.d_model,
        d_ff=args.d_model * 2,
        epochs=args.epochs,
        patience=min(args.patience, args.epochs),
        batch_size=args.batch_size,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        aircraft_type=DEFAULT_AIRCRAFT_TYPE,
        coordinate_frame="enu",
        random_train_anchor=False,
    )
    provenance = arrival_data_provenance(manifests)
    outer_split_keys = flight_keys_by_split(provenance, config)
    development_keys = set(outer_split_keys["train"] + outer_split_keys["val"])
    series, report = build_series(
        load_flight_dicts(manifests, include_flight_keys=development_keys),
        config,
        aircraft_type=config.aircraft_type,
    )
    print(report.format())
    run_ablation(
        series,
        config,
        airports=airports,
        train_per_airport=args.train_per_airport,
        val_per_airport=args.val_per_airport,
        weights=args.kinematic_weights,
        n_segments_candidates=n_segment_candidates,
        d_model_candidates=d_model_candidates,
        accuracy_tolerance=args.accuracy_tolerance,
        smoothness_tolerance=args.smoothness_tolerance,
        full_outer_split=args.full_outer_split,
        output_dir=output_dir,
        manifest_digests=provenance_manifest_digests(provenance),
        outer_split_keys=outer_split_keys,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
