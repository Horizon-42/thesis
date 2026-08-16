#!/usr/bin/env python
"""Validation-safe per-flight control reconstruction ceiling diagnostic.

The trained network supplies only the initialization.  Each selected development flight
then receives its own oracle control parameters, optimized directly against that flight's
known future while the total duration is fixed to truth.  This is not a deployable model;
it asks whether the current OpenAP dynamics and control parameterization can represent the
observed trajectories at all.

The script deliberately supports only checkpoint train/validation identities.  It never
opens outer-test tracks or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
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
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    arrival_data_provenance,
    build_series,
    load_flight_dicts,
    require_matching_data_provenance,
)
from models import resolve_device  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
from train import (  # noqa: E402
    load_checkpoint,
    move_dynamics,
    move_fixed_dt_supervision,
    prediction_loss_components,
    unpack_batch,
    usable_series,
)


SCHEMA_VERSION = "ts-control-capacity-ceiling-v1-development-only"
ORACLE_MODES = ("optimized_partition", "uniform_partition")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keys_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def balanced_key_sample(
    keys: Sequence[str], *, per_airport: int, seed: int, split: str
) -> list[str]:
    """Deterministically sample the same number of checkpoint identities per airport."""
    if per_airport <= 0:
        raise ValueError("per_airport must be positive")
    by_airport: dict[str, list[str]] = {}
    for key in keys:
        airport, separator, _flight_id = key.partition(":")
        if not separator:
            raise ValueError(f"airport-qualified dataset identity required, got {key!r}")
        by_airport.setdefault(airport, []).append(key)

    selected: list[str] = []
    for airport, group in sorted(by_airport.items()):
        if len(group) < per_airport:
            raise ValueError(
                f"airport {airport} has {len(group)} {split} identities, fewer than "
                f"requested {per_airport}"
            )
        ordered = sorted(
            group,
            key=lambda key: hashlib.sha256(f"{seed}:{split}:{key}".encode()).digest(),
        )
        selected.extend(ordered[:per_airport])
    return selected


def bounded_control_logits(
    controls: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    unit = ((controls - lower.unsqueeze(1)) / (upper - lower).unsqueeze(1)).clamp(
        min=epsilon, max=1.0 - epsilon
    )
    return torch.logit(unit)


def oracle_prediction(
    control_logits: torch.Tensor,
    duration_logits: torch.Tensor | None,
    true_total_s: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> ControlPrediction:
    controls = lower.unsqueeze(1) + torch.sigmoid(control_logits) * (
        upper - lower
    ).unsqueeze(1)
    segments = controls.shape[1]
    if duration_logits is None:
        fractions = torch.full_like(control_logits[..., 0], 1.0 / segments)
    else:
        if duration_logits.shape != control_logits.shape[:2]:
            raise ValueError("duration logits must align with control segments")
        fractions = torch.softmax(duration_logits, dim=1)
    durations = fractions * true_total_s.unsqueeze(1)
    return ControlPrediction(
        controls=controls,
        segment_durations=durations,
        final_time_s=durations.sum(dim=1),
    )


def nonfinite_gradient_diagnostics(
    parameters: Sequence[torch.nn.Parameter],
) -> list[dict[str, Any]]:
    """Locate non-finite oracle gradients without mutating the optimizer state."""
    diagnostics = []
    for parameter_index, parameter in enumerate(parameters):
        if parameter.grad is None:
            continue
        nonfinite = ~torch.isfinite(parameter.grad)
        if not torch.any(nonfinite):
            continue
        affected_rows = torch.nonzero(
            nonfinite.reshape(nonfinite.shape[0], -1).any(dim=1), as_tuple=False
        ).flatten()
        diagnostics.append({
            "parameter_index": parameter_index,
            "nonfinite_values": int(nonfinite.sum().detach().cpu()),
            "affected_batch_rows": affected_rows.detach().cpu().tolist(),
        })
    return diagnostics


def clip_grad_norm_float64_(
    parameters: Sequence[torch.nn.Parameter], max_norm: float
) -> float:
    """Clip finite FP32 gradients using a float64 norm accumulator.

    Oracle reconstruction can transiently produce finite gradients whose squared FP32
    sum overflows.  PyTorch's default clipper then reports ``inf`` and scales every value
    to zero.  A float64 accumulator preserves the intended global-norm clipping rule.
    """
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    total_square = torch.zeros(
        (), dtype=torch.float64, device=gradients[0].device
    )
    for gradient in gradients:
        total_square = total_square + gradient.detach().to(torch.float64).square().sum()
    total_norm = total_square.sqrt()
    if not torch.isfinite(total_norm):
        return float(total_norm.detach().cpu())
    coefficient = torch.clamp(
        max_norm / (total_norm + 1e-12), max=1.0
    )
    for gradient in gradients:
        gradient.mul_(coefficient.to(dtype=gradient.dtype))
    return float(total_norm.detach().cpu())


def prepare_capacity_batch(
    raw_batch: tuple, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    Any,
]:
    """Normalize the shared 5/6/7-field batch contract for this experiment runner."""
    (
        x,
        y,
        mask,
        final_time_s,
        flight_weights,
        dynamics,
        dense_supervision,
    ) = unpack_batch(raw_batch)
    return (
        x.to(device),
        y.to(device),
        mask.to(device),
        final_time_s.to(device),
        flight_weights.to(device),
        move_dynamics(dynamics, device),
        move_fixed_dt_supervision(dense_supervision, device),
    )


def _common_metrics(
    physical_nodes: np.ndarray,
    durations_s: np.ndarray,
    series: Sequence[FlightSeries],
    truth: np.ndarray,
    true_duration_s: np.ndarray,
    progress: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.stack([item.values[config.seq_len - 1] for item in series])
    sampled = []
    for index in range(len(series)):
        row, _capped = common_report.resample_prediction(
            anchors[index],
            physical_nodes[index],
            float(durations_s[index].sum()),
            config,
            progress * true_duration_s[index],
            durations_s[index],
        )
        sampled.append(row)
    errors = common_report.displacement_errors(np.stack(sampled), truth)
    return errors.mean(axis=1), errors[:, -1]


def _prediction_metrics(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    series: Sequence[FlightSeries],
    truth: np.ndarray,
    true_duration_s: np.ndarray,
    progress: np.ndarray,
    config: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with torch.no_grad():
        physical = rollout_control_endpoints(
            prediction.controls,
            prediction.segment_durations,
            dynamics,
            config,
        ).channels
    durations = prediction.segment_durations.detach().cpu().numpy().astype(np.float64)
    ade, fde = _common_metrics(
        physical.detach().cpu().numpy().astype(np.float32),
        durations,
        series,
        truth,
        true_duration_s,
        progress,
        config,
    )
    flat = durations.reshape(-1)
    return ade, fde, {
        "duration_min_s": float(flat.min()),
        "duration_p1_s": float(np.percentile(flat, 1)),
        "duration_median_s": float(np.median(flat)),
        "duration_p99_s": float(np.percentile(flat, 99)),
        "duration_max_s": float(flat.max()),
    }


def optimize_oracle(
    mode: str,
    initial: ControlPrediction,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    true_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    config: Any,
    normalizer: Any,
    dense_supervision: Any,
    *,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
) -> tuple[ControlPrediction, dict[str, Any]]:
    if mode not in ORACLE_MODES:
        raise ValueError(f"unknown oracle mode {mode!r}")
    control_logits = torch.nn.Parameter(bounded_control_logits(
        initial.controls.detach(), dynamics["control_lower"], dynamics["control_upper"]
    ))
    duration_logits = None
    parameters: list[torch.nn.Parameter] = [control_logits]
    if mode == "optimized_partition":
        fractions = (
            initial.segment_durations.detach()
            / initial.segment_durations.detach().sum(dim=1, keepdim=True)
        ).clamp(min=1e-8)
        duration_logits = torch.nn.Parameter(fractions.log())
        parameters.append(duration_logits)

    # This is a representational ceiling, so scalar control regularizers and time loss are
    # deliberately removed.  The total time itself is fixed to truth, while the existing
    # small terminal term remains part of the physical reconstruction objective.
    ceiling_config = replace(
        config,
        final_time_loss_weight=0.0,
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    best_loss = math.inf
    best_step = 0
    best_parameters: list[torch.Tensor] = []
    history: list[dict[str, float | int]] = []
    termination: dict[str, Any] = {
        "reason": "requested_steps_completed",
        "step": steps,
    }
    steps_completed = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = oracle_prediction(
            control_logits,
            duration_logits,
            true_final_time_s,
            dynamics["control_lower"],
            dynamics["control_upper"],
        )
        components = prediction_loss_components(
            prediction,
            x[:, -1],
            y,
            mask,
            true_final_time_s,
            flight_weights,
            ceiling_config,
            normalizer,
            dynamics,
            dense_supervision,
        )
        loss = components.total
        if not torch.isfinite(loss):
            if not best_parameters:
                raise RuntimeError(f"non-finite oracle objective at step {step}")
            termination = {"reason": "nonfinite_objective", "step": step}
            print(
                f"{mode}: stopping before step {step} update because the objective "
                "became non-finite; retaining the best finite parameters",
                flush=True,
            )
            break
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_step = step
            # Retain the exact parameter values whose forward pass produced best_loss.
            best_parameters = [parameter.detach().clone() for parameter in parameters]
        loss.backward()
        gradient_diagnostics = nonfinite_gradient_diagnostics(parameters)
        if gradient_diagnostics:
            if not best_parameters:
                raise RuntimeError(f"non-finite oracle gradient at step {step}")
            termination = {
                "reason": "nonfinite_gradient",
                "step": step,
                "parameters": gradient_diagnostics,
            }
            print(
                f"{mode}: stopping before step {step} update because gradients became "
                "non-finite; retaining the best finite parameters",
                flush=True,
            )
            break
        grad_norm = clip_grad_norm_float64_(parameters, max_grad_norm)
        if not math.isfinite(grad_norm):
            raise RuntimeError(
                f"gradient norm became non-finite at step {step} despite finite values"
            )
        optimizer.step()
        steps_completed = step
        if step == 1 or step % 10 == 0 or step == steps:
            durations = prediction.segment_durations.detach()
            history.append({
                "step": step,
                "loss": value,
                "state_loss": float(components.state.detach()),
                "terminal_loss": float(components.terminal.detach()),
                "grad_norm": grad_norm,
                "duration_min_s": float(durations.min()),
                "duration_p1_s": float(torch.quantile(durations, 0.01)),
                "duration_median_s": float(torch.median(durations)),
                "duration_p99_s": float(torch.quantile(durations, 0.99)),
                "duration_max_s": float(durations.max()),
            })

    if not best_parameters:
        raise RuntimeError("oracle optimization did not retain a finite parameter state")
    with torch.no_grad():
        control_logits.copy_(best_parameters[0])
        if duration_logits is not None:
            duration_logits.copy_(best_parameters[1])
        prediction = oracle_prediction(
            control_logits,
            duration_logits,
            true_final_time_s,
            dynamics["control_lower"],
            dynamics["control_upper"],
        )
    return prediction, {
        "mode": mode,
        "steps_requested": steps,
        "steps_completed": steps_completed,
        "best_step": best_step,
        "best_loss": best_loss,
        "learning_rate": learning_rate,
        "max_grad_norm": max_grad_norm,
        "elapsed_seconds": time.perf_counter() - started,
        "termination": termination,
        "history": history,
    }


def _summary_rows(
    rows: Sequence[dict[str, Any]], split: str | None = None
) -> list[dict[str, Any]]:
    selected = [row for row in rows if split is None or row["split"] == split]
    result = []
    modes = sorted({str(row["mode"]) for row in selected})
    for mode in modes:
        group = [row for row in selected if row["mode"] == mode]
        ade = np.asarray([row["ade_m"] for row in group], dtype=np.float64)
        fde = np.asarray([row["fde_m"] for row in group], dtype=np.float64)
        result.append({
            "split": split or "development_sample",
            "mode": mode,
            "flights": len(group),
            "ade_mean_m": float(ade.mean()),
            "ade_median_m": float(np.median(ade)),
            "ade_p95_m": float(np.percentile(ade, 95)),
            "fde_mean_m": float(fde.mean()),
            "fde_median_m": float(np.median(fde)),
            "fde_p95_m": float(np.percentile(fde, 95)),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", required=True, type=Path)
    parser.add_argument("--state-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--per-airport", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--evaluation-points", type=int, default=64)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0.0 or args.max_grad_norm <= 0.0:
        parser.error("steps, learning rate and max grad norm must be positive")

    checkpoint = args.control_checkpoint.expanduser().resolve()
    model, config, normalizer, payload = load_checkpoint(checkpoint)
    if config.prediction_output != PREDICTION_CONTROL:
        parser.error("capacity ceiling requires a control-output checkpoint")
    control_run = common_report.LoadedRun(
        "control_model", checkpoint, model, config, normalizer, payload
    )

    train_keys = balanced_key_sample(
        payload["split"]["train"],
        per_airport=args.per_airport,
        seed=args.sample_seed,
        split="train",
    )
    val_keys = balanced_key_sample(
        payload["split"]["val"],
        per_airport=args.per_airport,
        seed=args.sample_seed,
        split="val",
    )
    selected_keys = train_keys + val_keys
    provenance = payload["data_provenance"]
    airports = tuple(entry["airport"] for entry in provenance["manifests"])
    manifests = [pipeline.arrival_manifest_path(airport) for airport in airports]
    require_matching_data_provenance(payload, arrival_data_provenance(manifests))
    print(
        f"loading {len(train_keys)} train + {len(val_keys)} validation identities; "
        "outer-test source tracks stay closed",
        flush=True,
    )
    all_series, build_report = build_series(
        load_flight_dicts(manifests, include_flight_keys=set(selected_keys)),
        config,
        aircraft_type=config.aircraft_type,
    )
    print(build_report.format(), flush=True)
    all_series = usable_series(all_series, config, verbose=False)
    by_id = {item.dataset_id: item for item in all_series}
    missing = [key for key in selected_keys if key not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} selected development flights could not be rebuilt")
    selected_series = [by_id[key] for key in selected_keys]
    split_labels = np.asarray(["train"] * len(train_keys) + ["val"] * len(val_keys))
    truth, true_duration, progress, route_types = common_report.common_truth(
        selected_series, config, args.evaluation_points
    )

    dataset = FixedAnchorTrajectoryWindows(selected_series, config, normalizer)
    if len(dataset) != len(selected_series):
        raise ValueError("capacity sample must provide exactly one fixed anchor per flight")
    raw_batch = dataset.batch(np.arange(len(dataset)))
    device = resolve_device(args.device)
    (
        x,
        y,
        mask,
        final_time_s,
        flight_weights,
        dynamics,
        dense_supervision,
    ) = prepare_capacity_batch(raw_batch, device)
    control_run.model.to(device).eval()
    with torch.no_grad():
        initial = control_run.model(x, dynamics)

    flight_rows: list[dict[str, Any]] = []

    def append_metrics(mode: str, prediction: ControlPrediction) -> dict[str, float]:
        ade, fde, duration_stats = _prediction_metrics(
            prediction,
            dynamics,
            selected_series,
            truth,
            true_duration,
            progress,
            config,
        )
        for index, item in enumerate(selected_series):
            flight_rows.append({
                "split": split_labels[index],
                "mode": mode,
                "dataset_id": item.dataset_id,
                "airport": item.airport,
                "aircraft_type": item.scenario.aircraft.code,
                "trajectory_type": route_types[index],
                "ade_m": ade[index],
                "fde_m": fde[index],
            })
        return duration_stats

    mode_durations: dict[str, dict[str, float]] = {}
    mode_durations["control_model"] = append_metrics("control_model", initial)
    optimization: dict[str, Any] = {}
    for mode in ORACLE_MODES:
        print(f"optimizing {mode} for {args.steps} steps", flush=True)
        prediction, record = optimize_oracle(
            mode,
            initial,
            x,
            y,
            mask,
            final_time_s,
            flight_weights,
            dynamics,
            config,
            normalizer,
            dense_supervision,
            steps=args.steps,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
        )
        label = f"oracle_{mode}"
        mode_durations[label] = append_metrics(label, prediction)
        optimization[label] = record

    state_checkpoint_info = None
    if args.state_checkpoint is not None:
        state_path = args.state_checkpoint.expanduser().resolve()
        state_model, state_config, state_normalizer, state_payload = load_checkpoint(state_path)
        state_run = common_report.LoadedRun(
            "state_reference", state_path, state_model, state_config, state_normalizer, state_payload
        )
        if error := common_report.comparison_identity_error(control_run, state_run):
            parser.error(error)
        print("evaluating state reference on the same development sample", flush=True)
        state_block = common_report.run_deterministic(
            state_run,
            selected_series,
            truth,
            true_duration,
            progress,
            device,
            batch_size=len(selected_series),
        )
        for index, item in enumerate(selected_series):
            flight_rows.append({
                "split": split_labels[index],
                "mode": "state_reference",
                "dataset_id": item.dataset_id,
                "airport": item.airport,
                "aircraft_type": item.scenario.aircraft.code,
                "trajectory_type": route_types[index],
                "ade_m": state_block["ade_per_flight_m"][index],
                "fde_m": state_block["fde_per_flight_m"][index],
            })
        state_checkpoint_info = {
            "path": str(state_path),
            "sha256": file_sha256(state_path),
        }

    summary = _summary_rows(flight_rows, "train") + _summary_rows(flight_rows, "val")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    common_report.write_csv(output / "flight_metrics.csv", flight_rows)
    common_report.write_csv(output / "summary.csv", summary)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_policy": {
            "evaluated_splits": ["train", "val"],
            "outer_test_loaded": False,
            "full_train_split_sha256": keys_sha256(payload["split"]["train"]),
            "full_validation_split_sha256": keys_sha256(payload["split"]["val"]),
            "sample_seed": args.sample_seed,
            "per_airport_per_split": args.per_airport,
            "selected_train_sha256": keys_sha256(train_keys),
            "selected_validation_sha256": keys_sha256(val_keys),
            "selected_train_ids": train_keys,
            "selected_validation_ids": val_keys,
        },
        "control_checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "state_checkpoint": state_checkpoint_info,
        "objective": {
            "total_time": "fixed to observed truth",
            "optimized": "per-flight bounded controls and, where named, duration partition",
            "disabled_weights": [
                "final_time_loss_weight",
                "control_effort_loss_weight",
                "control_smoothness_loss_weight",
            ],
            "terminal_loss_weight": config.terminal_loss_weight,
            "warning": "oracle reconstruction ceiling; not a deployable prediction result",
        },
        "summary": summary,
        "duration_statistics": mode_durations,
        "optimization": optimization,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote development-only capacity ceiling to {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
