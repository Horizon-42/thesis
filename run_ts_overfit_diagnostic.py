#!/usr/bin/env python
"""Test whether the TS model can memorize a small, fixed outer-train subset.

This is deliberately not a validation experiment. The same flights are used for gradient
updates and for the no-dropout replay measured after each epoch. If the model cannot drive
that replay loss and the physical trajectory errors down, adding more data is not the next
remedy: the bottleneck is the model, target, or objective.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from channels import POSITION_IDX  # noqa: E402
from config import DEFAULT_AIRCRAFT_TYPE, TSConfig  # noqa: E402
from dataset import (  # noqa: E402
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    arrival_data_provenance,
    build_series,
    flight_keys_by_split,
    iter_batches,
    load_flight_dicts,
    provenance_manifest_digests,
    split_name_for_dataset_id,
)
from models import parameter_count  # noqa: E402
from train import evaluate_split, fit_model, usable_series  # noqa: E402

RESULT_SCHEMA = "ts-small-sample-overfit-diagnostic-v1"
DEFAULT_WEIGHTS = (10.0, 0.0)


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
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("--kinematic-weights must be non-negative")
    return tuple(dict.fromkeys(values))


def _series_digest(series: Sequence[FlightSeries]) -> str:
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def select_balanced_subset(
    series: Sequence[FlightSeries],
    airports: Sequence[str],
    *,
    samples_per_airport: int,
    seed: int,
) -> list[FlightSeries]:
    """Select a deterministic, airport-balanced memorization population."""
    selected: list[FlightSeries] = []
    for airport in airports:
        candidates = [item for item in series if item.airport == airport]
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"overfit:{seed}:{item.dataset_id}".encode()
            ).digest(),
        )
        if len(ordered) < samples_per_airport:
            raise ValueError(
                f"airport {airport} has only {len(ordered)} outer-train flights, "
                f"need {samples_per_airport}"
            )
        selected.extend(ordered[:samples_per_airport])
    return sorted(selected, key=lambda item: item.dataset_id)


def _terminal_target_metrics(
    model: torch.nn.Module,
    dataset: FixedAnchorTrajectoryWindows,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Physical error at progress=1, including fitted runway-crossing targets."""
    horizontal, vertical, distance = [], [], []
    model.eval()
    with torch.no_grad():
        for x, target, _mask, _final_time_s, _flight_weights in iter_batches(
            dataset, batch_size, shuffle=False, seed=0
        ):
            prediction = model(x.to(device)).states.cpu().numpy()
            predicted = dataset.normalizer.decode(prediction.astype(np.float64))
            truth = dataset.normalizer.decode(target.numpy().astype(np.float64))
            delta = predicted[:, -1, list(POSITION_IDX)] - truth[:, -1, list(POSITION_IDX)]
            horizontal.extend(np.linalg.norm(delta[:, :2], axis=1).tolist())
            vertical.extend(np.abs(delta[:, 2]).tolist())
            distance.extend(np.linalg.norm(delta, axis=1).tolist())
    return {
        "horizontal_mean_m": float(np.mean(horizontal)),
        "horizontal_p95_m": float(np.percentile(horizontal, 95)),
        "vertical_mean_abs_m": float(np.mean(vertical)),
        "vertical_p95_abs_m": float(np.percentile(vertical, 95)),
        "distance_mean_m": float(np.mean(distance)),
        "distance_p95_m": float(np.percentile(distance, 95)),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_reports(output_dir: Path, result: dict[str, Any]) -> None:
    candidates = result["candidates"]
    with (output_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=(
            "kinematic_weight", "epochs_run", "best_epoch", "first_replay_loss",
            "best_replay_loss", "loss_reduction_pct", "ade_m", "fde_m",
            "final_time_mae_s", "terminal_horizontal_mean_m",
            "terminal_vertical_mean_abs_m",
        ))
        writer.writeheader()
        for candidate in candidates:
            metrics = candidate["training_replay_metrics"]
            terminal = candidate["terminal_target_metrics"]
            writer.writerow({
                "kinematic_weight": candidate["kinematic_weight"],
                "epochs_run": candidate["epochs_run"],
                "best_epoch": candidate["best_epoch"],
                "first_replay_loss": candidate["first_replay_loss"],
                "best_replay_loss": candidate["best_replay_loss"],
                "loss_reduction_pct": candidate["loss_reduction_pct"],
                "ade_m": metrics["ade_m"],
                "fde_m": metrics["fde_m"],
                "final_time_mae_s": metrics["final_time_s"]["mae"],
                "terminal_horizontal_mean_m": terminal["horizontal_mean_m"],
                "terminal_vertical_mean_abs_m": terminal["vertical_mean_abs_m"],
            })

    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plots / ".matplotlib-cache"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    for candidate in candidates:
        axis.plot(
            [row["epoch"] for row in candidate["history"]],
            [row["val_loss"] for row in candidate["history"]],
            label=f"kinematic={candidate['kinematic_weight']:g}",
        )
    axis.set(xlabel="Epoch", ylabel="Training replay loss",
             title="Small-sample memorization curve")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(plots / f"replay_loss.{suffix}", dpi=180)
    plt.close(figure)

    labels = [f"k={row['kinematic_weight']:g}" for row in candidates]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(
        x - width / 2,
        [row["training_replay_metrics"]["ade_m"] for row in candidates],
        width,
        label="ADE",
    )
    axis.bar(
        x + width / 2,
        [row["terminal_target_metrics"]["horizontal_mean_m"] for row in candidates],
        width,
        label="Terminal horizontal",
    )
    axis.set(xticks=x, xticklabels=labels, ylabel="Metres",
             title="Memorized trajectory and runway-target error")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(plots / f"physical_errors.{suffix}", dpi=180)
    plt.close(figure)

    rows = []
    for candidate in candidates:
        metrics = candidate["training_replay_metrics"]
        terminal = candidate["terminal_target_metrics"]
        rows.append(
            f"| {candidate['kinematic_weight']:g} | {candidate['best_epoch']} | "
            f"{candidate['best_replay_loss']:.6f} | {metrics['ade_m']:.1f} | "
            f"{metrics['fde_m']:.1f} | {metrics['final_time_s']['mae']:.1f} | "
            f"{terminal['horizontal_mean_m']:.1f} | "
            f"{terminal['vertical_mean_abs_m']:.1f} |"
        )
    (plots / "index.md").write_text(
        "# Small-sample overfit diagnostic\n\n"
        "The replay population is exactly the training population; these are memorization, "
        "not generalization, metrics.\n\n"
        "| Kinematic weight | Best epoch | Replay loss | ADE (m) | FDE (m) | "
        "Time MAE (s) | Terminal horizontal (m) | Terminal vertical abs. (m) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n![Replay loss](replay_loss.png)\n\n"
        "![Physical errors](physical_errors.png)\n",
        encoding="utf-8",
    )


def run_diagnostic(
    series: Sequence[FlightSeries],
    base_config: TSConfig,
    *,
    airports: Sequence[str],
    samples_per_airport: int,
    kinematic_weights: Sequence[float],
    output_dir: Path,
    manifest_digests: dict[str, str],
    outer_split_keys: dict[str, list[str]],
) -> dict[str, Any]:
    outer_train = usable_series(series, base_config)
    wrong_split = [
        item.dataset_id for item in outer_train
        if split_name_for_dataset_id(item.dataset_id, base_config) != "train"
    ]
    if wrong_split:
        raise ValueError(f"overfit diagnostic loaded non-training flight {wrong_split[0]!r}")
    sample = select_balanced_subset(
        outer_train,
        airports,
        samples_per_airport=samples_per_airport,
        seed=base_config.seed,
    )
    print(
        f"outer roster: train {len(outer_split_keys['train'])} / "
        f"val {len(outer_split_keys['val'])} / test {len(outer_split_keys['test'])}; "
        "opened train source tracks only"
    )
    print(
        f"memorization population: {len(sample)} flights "
        f"({samples_per_airport} per airport), sha256={_series_digest(sample)}"
    )

    candidates = []
    for index, weight in enumerate(kinematic_weights):
        config = replace(base_config, kinematic_consistency_loss_weight=weight)
        print(
            f"\n=== candidate {index + 1}/{len(kinematic_weights)}: "
            f"kinematic={weight:g} ==="
        )
        # Intentional leakage: the validation argument is the training sample itself. It is
        # used only as a deterministic no-dropout replay for checkpoint selection.
        fit = fit_model(sample, sample, config, verbose=True)
        replay = FixedAnchorTrajectoryWindows(sample, fit.config, fit.normalizer)
        metrics = evaluate_split(fit.model, replay, fit.normalizer, fit.config, fit.device)
        terminal = _terminal_target_metrics(
            fit.model,
            replay,
            batch_size=fit.config.batch_size,
            device=fit.device,
        )
        best = min(fit.history, key=lambda row: row.val_loss)
        first_loss = fit.history[0].val_loss
        candidates.append({
            "kinematic_weight": weight,
            "parameters": parameter_count(fit.model),
            "epochs_run": len(fit.history),
            "best_epoch": best.epoch,
            "first_replay_loss": first_loss,
            "best_replay_loss": best.val_loss,
            "loss_reduction_pct": 100.0 * (1.0 - best.val_loss / first_loss),
            "best_loss_components": best.val_components,
            "training_replay_metrics": metrics,
            "terminal_target_metrics": terminal,
            "history": [vars(row) for row in fit.history],
        })
        del replay, fit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "schema_version": RESULT_SCHEMA,
        "purpose": "small-sample training-set memorization capacity, not generalization",
        "population": {
            "airports": list(airports),
            "samples_per_airport": samples_per_airport,
            "flights": len(sample),
            "dataset_ids_sha256": _series_digest(sample),
            "dataset_ids": [item.dataset_id for item in sample],
            "source": "locked outer-train only",
        },
        "outer_split": {
            "train": len(outer_split_keys["train"]),
            "validation": len(outer_split_keys["val"]),
            "test": len(outer_split_keys["test"]),
        },
        "controls": {
            "base_config": base_config.to_dict(),
            "kinematic_weights": list(kinematic_weights),
            "training_replay_is_training_population": True,
        },
        "arrival_manifests": manifest_digests,
        "candidates": candidates,
    }
    _write_json_atomic(output_dir / "overfit_diagnostic.json", result)
    write_reports(output_dir, result)
    print(f"\n✓ wrote {output_dir / 'overfit_diagnostic.json'}")
    print(f"  report: {output_dir / 'plots' / 'index.md'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", type=_parse_airports, default=None,
                        help="comma-separated airports; default: all discovered K-airports")
    parser.add_argument("--samples-per-airport", type=int, default=32)
    parser.add_argument("--kinematic-weights", type=_parse_weights, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-segments", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for name in ("samples_per_airport", "epochs", "patience", "batch_size", "n_segments"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")

    airports = args.airports or tuple(pipeline.discover_k_airports())
    if not airports:
        parser.error(f"no K-airport arrivals found under {pipeline.HARVEST_ROOT}")
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        parser.error(f"missing arrivals manifest {missing[0]}; run prepare_scenario_inputs.py")

    output_dir = args.output_dir or (
        REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" /
        "ts_itransformer_small_sample_overfit"
    )
    print(
        f"small-sample overfit diagnostic: airports={','.join(airports)}, "
        f"samples={args.samples_per_airport}/airport, epochs={args.epochs}, "
        f"dropout={args.dropout:g}, "
        f"kinematic={','.join(f'{value:g}' for value in args.kinematic_weights)}"
    )
    print(f"output: {output_dir}")
    if args.dry_run:
        return 0

    config = TSConfig(
        model="itransformer",
        n_segments=args.n_segments,
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
    flights = load_flight_dicts(
        manifests,
        include_flight_keys=set(outer_split_keys["train"]),
    )
    series, report = build_series(
        flights,
        config,
        aircraft_type=config.aircraft_type,
    )
    print(report.format())
    if not series:
        parser.error("no usable trajectory series")
    run_diagnostic(
        series,
        config,
        airports=airports,
        samples_per_airport=args.samples_per_airport,
        kinematic_weights=args.kinematic_weights,
        output_dir=output_dir,
        manifest_digests=provenance_manifest_digests(provenance),
        outer_split_keys=outer_split_keys,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
