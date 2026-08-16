#!/usr/bin/env python
"""Validation-only clock attribution for a control-output TS checkpoint.

The diagnostic separates errors caused by the learned total duration, the learned
non-uniform duration partition, and the control/dynamics rollout.  It deliberately has no
test-split option and only opens the validation flight identities stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
import run_ts_predictability_report as common_report  # noqa: E402
from config import PREDICTION_CONTROL  # noqa: E402
from control_rollout import rollout_control_endpoints  # noqa: E402
from dataset import (  # noqa: E402
    FlightSeries,
    arrival_data_provenance,
    build_series,
    load_flight_dicts,
    require_matching_data_provenance,
)
from models import resolve_device  # noqa: E402
from train import load_checkpoint, usable_series  # noqa: E402


SCHEMA_VERSION = "ts-control-clock-attribution-v1-validation-only"
VARIANT_LABELS = (
    "predicted_clock",
    "true_total_timewarp_only",
    "true_total_learned_partition_rerollout",
    "predicted_total_uniform_partition_rerollout",
    "true_total_uniform_partition_rerollout",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def duration_variants(
    predicted_durations_s: torch.Tensor,
    true_total_s: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the learned/scaled/uniform clocks used by the attribution report."""
    if predicted_durations_s.ndim != 2:
        raise ValueError("predicted durations must be [B,N]")
    if true_total_s.shape != predicted_durations_s.shape[:1]:
        raise ValueError("true total durations must be [B]")
    if torch.any(predicted_durations_s <= 0.0) or torch.any(true_total_s <= 0.0):
        raise ValueError("clock attribution requires strictly positive durations")

    predicted_total = predicted_durations_s.sum(dim=1, keepdim=True)
    learned_fractions = predicted_durations_s / predicted_total
    segments = predicted_durations_s.shape[1]
    uniform_fraction = torch.full_like(predicted_durations_s, 1.0 / segments)
    true_total = true_total_s.unsqueeze(1).to(
        dtype=predicted_durations_s.dtype,
        device=predicted_durations_s.device,
    )
    return {
        "predicted_clock": predicted_durations_s,
        "true_total_timewarp_only": learned_fractions * true_total,
        "true_total_learned_partition_rerollout": learned_fractions * true_total,
        "predicted_total_uniform_partition_rerollout": uniform_fraction * predicted_total,
        "true_total_uniform_partition_rerollout": uniform_fraction * true_total,
    }


def _rerollout(
    controls: torch.Tensor,
    durations_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    run: common_report.LoadedRun,
) -> np.ndarray:
    channels = rollout_control_endpoints(
        controls, durations_s, dynamics, run.config
    ).channels
    return channels.detach().cpu().numpy().astype(np.float32)


def evaluate_clock_variants(
    run: common_report.LoadedRun,
    series: Sequence[FlightSeries],
    truth: np.ndarray,
    true_duration_s: np.ndarray,
    progress: np.ndarray,
    route_types: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    run.model.to(device).eval()
    histories = common_report.history_tensor(series, run.config, run.normalizer)
    anchors = np.stack([item.values[run.config.seq_len - 1] for item in series])
    predictions: dict[str, list[np.ndarray]] = {label: [] for label in VARIANT_LABELS}
    final_times: dict[str, list[np.ndarray]] = {label: [] for label in VARIANT_LABELS}

    with torch.no_grad():
        for start in range(0, len(series), batch_size):
            stop = min(start + batch_size, len(series))
            batch_series = series[start:stop]
            dynamics = common_report.batch_dynamics_tensors(batch_series, run.config, device)
            output = run.model(torch.from_numpy(histories[start:stop]).to(device), dynamics)
            variants = duration_variants(
                output.segment_durations,
                torch.from_numpy(true_duration_s[start:stop]).to(device),
            )
            baseline = _rerollout(output.controls, output.segment_durations, dynamics, run)

            raw_nodes = {
                "predicted_clock": baseline,
                # This condition changes only the timestamps assigned to the already rolled
                # out nodes; it is intentionally not a physically consistent new rollout.
                "true_total_timewarp_only": baseline,
                "true_total_learned_partition_rerollout": _rerollout(
                    output.controls,
                    variants["true_total_learned_partition_rerollout"],
                    dynamics,
                    run,
                ),
                "predicted_total_uniform_partition_rerollout": _rerollout(
                    output.controls,
                    variants["predicted_total_uniform_partition_rerollout"],
                    dynamics,
                    run,
                ),
                "true_total_uniform_partition_rerollout": _rerollout(
                    output.controls,
                    variants["true_total_uniform_partition_rerollout"],
                    dynamics,
                    run,
                ),
            }
            for label in VARIANT_LABELS:
                durations = variants[label].detach().cpu().numpy().astype(np.float64)
                totals = durations.sum(axis=1)
                sampled_rows = []
                for local, absolute in enumerate(range(start, stop)):
                    sampled, _capped = common_report.resample_prediction(
                        anchors[absolute],
                        raw_nodes[label][local],
                        float(totals[local]),
                        run.config,
                        progress * true_duration_s[absolute],
                        durations[local],
                    )
                    sampled_rows.append(sampled)
                predictions[label].append(np.stack(sampled_rows).astype(np.float32))
                final_times[label].append(totals)

    blocks: dict[str, dict[str, Any]] = {}
    flight_rows: list[dict[str, Any]] = []
    for label in VARIANT_LABELS:
        prediction = np.concatenate(predictions[label])
        predicted_total = np.concatenate(final_times[label])
        errors = common_report.displacement_errors(prediction, truth)
        ade = errors.mean(axis=1)
        fde = errors[:, -1]
        blocks[label] = {
            "common_prediction": prediction,
            "error_grid_m": errors,
            "ade_per_flight_m": ade,
            "fde_per_flight_m": fde,
            "predicted_final_time_s": predicted_total,
            "summary": {
                "ade_m": float(ade.mean()),
                "fde_m": float(fde.mean()),
                "final_time_mae_s": float(np.abs(predicted_total - true_duration_s).mean()),
            },
        }
        for index, item in enumerate(series):
            source = item.scenario.source
            airport = item.airport or source.get("arr_airport") or "?"
            runway = source.get("runway") or "?"
            flight_rows.append({
                "variant": label,
                "dataset_id": item.dataset_id,
                "airport": airport,
                "runway": f"{airport}/{runway}",
                "aircraft_type": item.scenario.aircraft.code,
                "trajectory_type": route_types[index],
                "true_final_time_s": true_duration_s[index],
                "predicted_final_time_s": predicted_total[index],
                "final_time_error_s": predicted_total[index] - true_duration_s[index],
                "ade_m": ade[index],
                "fde_m": fde[index],
            })
    return blocks, flight_rows


def _group_rows(rows: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), str(row[field])), []).append(row)
    result = []
    for (variant, value), group in sorted(grouped.items()):
        ade = np.asarray([row["ade_m"] for row in group], dtype=np.float64)
        fde = np.asarray([row["fde_m"] for row in group], dtype=np.float64)
        time_error = np.asarray([row["final_time_error_s"] for row in group], dtype=np.float64)
        result.append({
            "variant": variant,
            field: value,
            "flights": len(group),
            "ade_mean_m": float(ade.mean()),
            "ade_p95_m": float(np.percentile(ade, 95)),
            "fde_mean_m": float(fde.mean()),
            "time_mae_s": float(np.abs(time_error).mean()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-points", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.evaluation_points <= 1:
        parser.error("batch size must be positive and evaluation points must exceed one")

    checkpoint = args.checkpoint.expanduser().resolve()
    model, config, normalizer, payload = load_checkpoint(checkpoint)
    if config.prediction_output != PREDICTION_CONTROL:
        parser.error("clock attribution requires a control-output checkpoint")
    run = common_report.LoadedRun("control", checkpoint, model, config, normalizer, payload)

    provenance = payload["data_provenance"]
    airports = tuple(entry["airport"] for entry in provenance["manifests"])
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    require_matching_data_provenance(payload, arrival_data_provenance(manifests))
    validation_keys = list(payload["split"]["val"])
    print(
        f"loading {len(validation_keys)} validation identities for {','.join(airports)}; "
        "outer-test source tracks stay closed",
        flush=True,
    )
    all_series, build_report = build_series(
        load_flight_dicts(manifests, include_flight_keys=set(validation_keys)),
        config,
        aircraft_type=config.aircraft_type,
    )
    print(build_report.format(), flush=True)
    all_series = usable_series(all_series, config, verbose=False)
    by_id = {item.dataset_id: item for item in all_series}
    missing = [key for key in validation_keys if key not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} checkpoint validation flights could not be rebuilt")
    validation_series = [by_id[key] for key in validation_keys]
    truth, true_duration, progress, route_types = common_report.common_truth(
        validation_series, config, args.evaluation_points
    )

    device = resolve_device(args.device)
    blocks, flight_rows = evaluate_clock_variants(
        run,
        validation_series,
        truth,
        true_duration,
        progress,
        route_types,
        device,
        args.batch_size,
    )
    baseline_ade = blocks["predicted_clock"]["summary"]["ade_m"]
    summary = []
    for label in VARIANT_LABELS:
        row = {"variant": label, **blocks[label]["summary"]}
        row["ade_change_vs_predicted_clock_percent"] = (
            (row["ade_m"] / baseline_ade - 1.0) * 100.0
        )
        summary.append(row)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    common_report.write_csv(output / "flight_metrics.csv", flight_rows)
    common_report.write_csv(output / "metrics_by_airport.csv", _group_rows(flight_rows, "airport"))
    common_report.write_csv(
        output / "metrics_by_aircraft_type.csv", _group_rows(flight_rows, "aircraft_type")
    )
    common_report.write_csv(
        output / "metrics_by_trajectory_type.csv",
        _group_rows(flight_rows, "trajectory_type"),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_policy": {
            "evaluated_split": "val",
            "outer_test_loaded": False,
            "validation_split_sha256": split_sha256(validation_keys),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "config": config.to_dict(),
        },
        "validation_flights": len(validation_series),
        "evaluation_points": args.evaluation_points,
        "variants": {
            "predicted_clock": "learned controls, learned total time and learned partition",
            "true_total_timewarp_only": (
                "unchanged predicted nodes; learned partition rescaled to true total time"
            ),
            "true_total_learned_partition_rerollout": (
                "learned controls re-integrated with learned partition rescaled to true total"
            ),
            "predicted_total_uniform_partition_rerollout": (
                "learned controls re-integrated on a uniform partition of predicted total"
            ),
            "true_total_uniform_partition_rerollout": (
                "learned controls re-integrated on a uniform partition of true total"
            ),
        },
        "summary": summary,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote validation-only clock attribution to {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
