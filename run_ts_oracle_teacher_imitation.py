#!/usr/bin/env python
"""Memorization gate for a train-only optimized oracle-teacher cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from config import (  # noqa: E402
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    PREDICTION_CONTROL,
    TSConfig,
)
from dataset import FixedAnchorTrajectoryWindows, Normalizer  # noqa: E402
from models import build_model, resolve_device  # noqa: E402
from oracle_teacher.cohort import select_outer_train_cohort  # noqa: E402
from oracle_teacher.evaluation import (  # noqa: E402
    evaluate_schedule,
    move_dynamics,
    observed_clock_prediction,
)
from oracle_teacher.imitation import control_imitation_loss  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
from train import model_forward  # noqa: E402


SCHEMA = "ts-oracle-teacher-imitation-gate-v1-train-only"


def _one_prediction(prediction: ControlPrediction, index: int) -> ControlPrediction:
    return ControlPrediction(
        controls=prediction.controls[index : index + 1],
        segment_durations=prediction.segment_durations[index : index + 1],
        final_time_s=prediction.final_time_s[index : index + 1],
    )


def _rollout_rows(
    prediction: ControlPrediction,
    dataset: FixedAnchorTrajectoryWindows,
    config: TSConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    rows = []
    for index in range(len(dataset)):
        one = _one_prediction(prediction, index)
        batch = dataset.batch(np.array([index]))
        target_final_time_s = batch[3].to(device)
        state_clock = observed_clock_prediction(one, target_final_time_s)
        metrics = evaluate_schedule(state_clock, dataset, index, config, device)
        metrics["final_time_abs_error_s"] = float(
            (one.final_time_s - target_final_time_s).abs().cpu()
        )
        rows.append(metrics)
    return rows


def _median_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.median([row[key] for row in rows]))
        for key in (
            "ade_m",
            "fde_at_last_complete_dt_m",
            "terminal_distance_m",
            "final_time_abs_error_s",
        )
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--teacher-schedules", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("steps", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0 or args.gradient_clip_norm <= 0.0:
        parser.error("learning rate and gradient clip norm must be positive")
    if not args.teacher_schedules.is_file():
        parser.error(f"missing teacher schedule file {args.teacher_schedules}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    with np.load(args.teacher_schedules, allow_pickle=False) as source:
        if set(source.files) != {
            "dataset_ids",
            "controls",
            "segment_durations_s",
        }:
            parser.error("teacher schedule file has an unexpected contract")
        dataset_ids = [str(value) for value in source["dataset_ids"].tolist()]
        target_controls_np = np.asarray(source["controls"], dtype=np.float32)
        target_durations_np = np.asarray(
            source["segment_durations_s"], dtype=np.float32
        )

    airport = args.airport.strip().upper()
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        control_dynamics_backend=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        random_train_anchor=False,
        n_segments=target_controls_np.shape[1],
        dropout=0.1,
        seed=args.seed,
        split_seed=args.split_seed,
        device=args.device,
    )
    cohort = select_outer_train_cohort(
        pipeline.arrival_manifest_path(airport),
        config,
        airport=airport,
        cohort_size=len(dataset_ids),
    )
    rebuilt_ids = [item.dataset_id for item in cohort.series]
    if rebuilt_ids != dataset_ids:
        parser.error("teacher schedules do not match the deterministic train cohort")
    normalizer = Normalizer.fit(cohort.series, balance_airports_and_flights=True)
    dataset = FixedAnchorTrajectoryWindows(cohort.series, config, normalizer)
    device = resolve_device(args.device)
    x, _target, _weights, final_time, _flight_weights, dynamics, _supervision = (
        dataset.batch(np.arange(len(dataset)))
    )
    x = x.to(device)
    final_time = final_time.to(device)
    dynamics = move_dynamics(dynamics, device)
    target_controls = torch.as_tensor(target_controls_np, device=device)
    target_durations = torch.as_tensor(target_durations_np, device=device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []

    model.eval()
    with torch.no_grad():
        initial_prediction = model_forward(model, x, dynamics)
    initial_rows = _rollout_rows(initial_prediction, dataset, config, device)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model_forward(model, x, dynamics)
        loss = control_imitation_loss(
            prediction,
            target_controls,
            target_durations,
            final_time,
            dynamics["control_lower"],
            dynamics["control_upper"],
            final_time_scale_s=config.final_time_scale_s,
        )
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip_norm
        )
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.total.detach()),
                "control": float(loss.control.detach()),
                "duration_fraction": float(loss.duration_fraction.detach()),
                "final_time": float(loss.final_time.detach()),
                "gradient_norm": float(gradient_norm.detach()),
            }
            history.append(row)
            print(
                f"imitation {step:4d}/{args.steps}: loss={row['loss']:.7f} "
                f"control={row['control']:.7f} time={row['final_time']:.7f}"
            )

    model.eval()
    with torch.no_grad():
        trained_prediction = model_forward(model, x, dynamics)
        final_loss = control_imitation_loss(
            trained_prediction,
            target_controls,
            target_durations,
            final_time,
            dynamics["control_lower"],
            dynamics["control_upper"],
            final_time_scale_s=config.final_time_scale_s,
        )
    trained_rows = _rollout_rows(trained_prediction, dataset, config, device)
    teacher_prediction = ControlPrediction(
        controls=target_controls,
        segment_durations=target_durations,
        final_time_s=final_time,
    )
    teacher_rows = _rollout_rows(teacher_prediction, dataset, config, device)

    result = {
        "schema_version": SCHEMA,
        "test_policy": "outer-train values only; validation/test values unopened",
        "config": config.to_dict(),
        "teacher_schedule_path": str(args.teacher_schedules.resolve()),
        "dataset_ids": dataset_ids,
        "recipe": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "loss": "unit-box control MSE + N-scaled duration-fraction MSE + time/600 MSE",
        },
        "final_imitation_loss": {
            "total": float(final_loss.total.cpu()),
            "control": float(final_loss.control.cpu()),
            "duration_fraction": float(final_loss.duration_fraction.cpu()),
            "final_time": float(final_loss.final_time.cpu()),
        },
        "median_observed_clock_rollout_metrics": {
            "untrained": _median_metrics(initial_rows),
            "imitation": _median_metrics(trained_rows),
            "teacher": _median_metrics(teacher_rows),
        },
        "history": history,
        "flight_metrics": [
            {
                "dataset_id": dataset_id,
                "untrained": initial,
                "imitation": trained,
                "teacher": teacher,
            }
            for dataset_id, initial, trained, teacher in zip(
                dataset_ids, initial_rows, trained_rows, teacher_rows
            )
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "imitation_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / "pretrained_model_state.pt")
    print(json.dumps(result["median_observed_clock_rollout_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
