#!/usr/bin/env python
"""Validation-only selector/oracle diagnostics for a control-mixture checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import run_ts_pipeline as pipeline
from channels import POSITION_IDX
from config import PREDICTION_CONTROL_MIXTURE
from control_mixture import ControlMixturePrediction
from control_rollout import rollout_control_endpoints
from dataset import (
    arrival_data_provenance,
    build_series,
    load_flight_dicts,
    require_matching_data_provenance,
)
from models import resolve_device
from train import usable_series
from run_ts_predictability_report import (
    common_truth,
    displacement_errors,
    file_sha256,
    history_tensor,
    load_runs,
    resample_prediction,
)

REPORT_SCHEMA = "ts-control-mixture-validation-report-v1"


def candidate_path_diversity(candidates: np.ndarray) -> np.ndarray:
    """Mean pairwise 3D path separation for each flight in [B,K,P,C]."""
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        raise ValueError("candidates must be [B,K,P,C] with at least two experts")
    positions = np.take(candidates, POSITION_IDX, axis=-1)
    pairwise = [
        np.linalg.norm(positions[:, left] - positions[:, right], axis=-1).mean(axis=1)
        for left in range(candidates.shape[1])
        for right in range(left + 1, candidates.shape[1])
    ]
    return np.stack(pairwise, axis=1).mean(axis=1)


def evaluate_candidates(
    run,
    series,
    truth: np.ndarray,
    true_duration_s: np.ndarray,
    progress: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Evaluate every expert; truth is used only for validation diagnostics."""
    from run_ts_predictability_report import batch_dynamics_tensors

    run.model.to(device).eval()
    histories = history_tensor(series, run.config, run.normalizer)
    anchors = np.stack([item.values[run.config.seq_len - 1] for item in series])
    ade_rows: list[np.ndarray] = []
    fde_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    selector_rows: list[np.ndarray] = []
    path_diversity_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(series), batch_size):
            stop = min(start + batch_size, len(series))
            batch_series = series[start:stop]
            dynamics = batch_dynamics_tensors(batch_series, run.config, device)
            output = run.model(
                torch.from_numpy(histories[start:stop]).to(device), dynamics
            )
            if not isinstance(output, ControlMixturePrediction):
                raise TypeError("control-mixture checkpoint returned the wrong output type")
            sampled_experts: list[np.ndarray] = []
            for expert in range(output.expert_count):
                candidate = output.candidate(expert)
                channels = rollout_control_endpoints(
                    candidate.controls,
                    candidate.segment_durations,
                    dynamics,
                    run.config,
                ).channels
                decoded = channels.cpu().numpy().astype(np.float32)
                durations = candidate.segment_durations.cpu().numpy().astype(np.float64)
                sampled = []
                for local, absolute in enumerate(range(start, stop)):
                    values, _capped = resample_prediction(
                        anchors[absolute],
                        decoded[local],
                        float(candidate.final_time_s[local].cpu()),
                        run.config,
                        progress * true_duration_s[absolute],
                        durations[local],
                    )
                    sampled.append(values)
                sampled_experts.append(np.stack(sampled))
            candidates = np.stack(sampled_experts, axis=1)
            error = displacement_errors(candidates, truth[start:stop, None])
            ade_rows.append(error.mean(axis=2))
            fde_rows.append(error[:, :, -1])
            time_rows.append(output.final_time_s.cpu().numpy())
            selector_rows.append(output.selection_logits.argmax(dim=1).cpu().numpy())
            path_diversity_rows.append(candidate_path_diversity(candidates))
    return {
        "ade_m": np.concatenate(ade_rows),
        "fde_m": np.concatenate(fde_rows),
        "final_time_s": np.concatenate(time_rows),
        "selected_expert": np.concatenate(selector_rows),
        "path_diversity_m": np.concatenate(path_diversity_rows),
    }


def _usage(indices: np.ndarray, expert_count: int) -> list[dict[str, float | int]]:
    counts = np.bincount(indices, minlength=expert_count)
    return [
        {"expert": expert, "flights": int(count), "fraction": float(count / len(indices))}
        for expert, count in enumerate(counts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-points", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.evaluation_points <= 1:
        parser.error("batch size must be positive and evaluation points > 1")

    run = load_runs([("control-mixture", args.checkpoint.resolve())])[0]
    if run.config.prediction_output != PREDICTION_CONTROL_MIXTURE:
        parser.error("checkpoint is not a control-mixture run")
    split = run.payload["split"]
    provenance = run.payload["data_provenance"]
    airports = tuple(entry["airport"] for entry in provenance["manifests"])
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    require_matching_data_provenance(run.payload, arrival_data_provenance(manifests))
    development_keys = set(split["train"] + split["val"])
    print(
        f"loading train/validation arrivals for {','.join(airports)}; "
        "outer-test source tracks stay closed",
        flush=True,
    )
    all_series, build_report = build_series(
        load_flight_dicts(manifests, include_flight_keys=development_keys),
        run.config,
        aircraft_type=run.config.aircraft_type,
    )
    print(build_report.format(), flush=True)
    all_series = usable_series(all_series, run.config, verbose=False)
    by_id = {item.dataset_id: item for item in all_series}
    validation = [by_id[key] for key in split["val"]]
    truth, true_duration, progress, _route_types = common_truth(
        validation, run.config, args.evaluation_points
    )
    metrics = evaluate_candidates(
        run,
        validation,
        truth,
        true_duration,
        progress,
        resolve_device(args.device),
        args.batch_size,
    )

    rows = np.arange(len(validation))
    selected = metrics["selected_expert"]
    oracle_ade = metrics["ade_m"].argmin(axis=1)
    oracle_fde = metrics["fde_m"].argmin(axis=1)
    selected_ade = metrics["ade_m"][rows, selected]
    selected_fde = metrics["fde_m"][rows, selected]
    selected_time = metrics["final_time_s"][rows, selected]
    expert_count = run.config.control_expert_count
    flight_rows = [
        {
            "dataset_id": item.dataset_id,
            "airport": item.airport,
            "selected_expert": int(selected[index]),
            "oracle_ade_expert": int(oracle_ade[index]),
            "selected_ade_m": float(selected_ade[index]),
            "oracle_ade_m": float(metrics["ade_m"][index, oracle_ade[index]]),
            "selected_fde_m": float(selected_fde[index]),
            "oracle_fde_m": float(metrics["fde_m"][index, oracle_fde[index]]),
            "selected_final_time_error_s": float(selected_time[index] - true_duration[index]),
            "candidate_path_diversity_m": float(metrics["path_diversity_m"][index]),
        }
        for index, item in enumerate(validation)
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "flight_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flight_rows[0]))
        writer.writeheader()
        writer.writerows(flight_rows)
    summary = {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_policy": {
            "evaluated_split": "val",
            "outer_test_loaded": False,
            "validation_split_sha256": hashlib.sha256(
                "\n".join(sorted(split["val"])).encode()
            ).hexdigest(),
        },
        "checkpoint": {
            "path": str(run.checkpoint),
            "sha256": file_sha256(run.checkpoint),
        },
        "validation_flights": len(validation),
        "expert_count": expert_count,
        "deployable_selector": {
            "ade_m": float(selected_ade.mean()),
            "fde_m": float(selected_fde.mean()),
            "final_time_mae_s": float(np.abs(selected_time - true_duration).mean()),
            "oracle_ade_top1_accuracy": float(np.mean(selected == oracle_ade)),
            "usage": _usage(selected, expert_count),
        },
        "oracle_candidate_set": {
            "warning": "validation-only oracle coverage; not deployable selection",
            "minade_m": float(metrics["ade_m"].min(axis=1).mean()),
            "minfde_m": float(metrics["fde_m"].min(axis=1).mean()),
            "ade_expert_usage": _usage(oracle_ade, expert_count),
            "fde_expert_usage": _usage(oracle_fde, expert_count),
        },
        "experts": [
            {
                "expert": expert,
                "ade_m": float(metrics["ade_m"][:, expert].mean()),
                "fde_m": float(metrics["fde_m"][:, expert].mean()),
                "final_time_mae_s": float(
                    np.abs(metrics["final_time_s"][:, expert] - true_duration).mean()
                ),
            }
            for expert in range(expert_count)
        ],
        "candidate_path_diversity_m": {
            "mean": float(metrics["path_diversity_m"].mean()),
            "median": float(np.median(metrics["path_diversity_m"])),
        },
    }
    (output / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote validation-only mixture report to {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
