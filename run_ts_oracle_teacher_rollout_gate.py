#!/usr/bin/env python
"""Fine-tune an imitated oracle teacher model on its train-only rollout objective."""

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
from oracle_teacher.optimization import TeacherOptimizationStage  # noqa: E402
from oracle_teacher.rollout_finetuning import fine_tune_model_on_rollout  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
from train import model_forward  # noqa: E402


SCHEMA = "ts-oracle-teacher-rollout-gate-v1-train-only"


def _single(prediction: ControlPrediction, index: int) -> ControlPrediction:
    return ControlPrediction(
        controls=prediction.controls[index : index + 1],
        segment_durations=prediction.segment_durations[index : index + 1],
        final_time_s=prediction.final_time_s[index : index + 1],
    )


def _evaluate(
    prediction: ControlPrediction,
    dataset: FixedAnchorTrajectoryWindows,
    config: TSConfig,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows = []
    for index in range(len(dataset)):
        one = _single(prediction, index)
        target_time = dataset.batch(np.array([index]))[3].to(device)
        metrics = evaluate_schedule(
            observed_clock_prediction(one, target_time),
            dataset,
            index,
            config,
            device,
        )
        metrics["final_time_abs_error_s"] = float(
            (one.final_time_s - target_time).abs().cpu()
        )
        rows.append(metrics)
    medians = {
        key: float(np.median([row[key] for row in rows]))
        for key in (
            "ade_m",
            "fde_at_last_complete_dt_m",
            "terminal_distance_m",
            "final_time_abs_error_s",
        )
    }
    return rows, medians


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--pretrained-state", type=Path, required=True)
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--prefix-steps", type=int, default=10)
    parser.add_argument("--full-steps", type=int, default=170)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("cohort_size", "prefix_steps", "full_steps", "log_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0 or args.gradient_clip_norm <= 0.0:
        parser.error("learning rate and gradient clip norm must be positive")
    if not args.pretrained_state.is_file():
        parser.error(f"missing pretrained state {args.pretrained_state}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

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
        n_segments=64,
        dropout=0.1,
        seed=args.seed,
        split_seed=args.split_seed,
        device=args.device,
    )
    cohort = select_outer_train_cohort(
        pipeline.arrival_manifest_path(airport),
        config,
        airport=airport,
        cohort_size=args.cohort_size,
    )
    normalizer = Normalizer.fit(cohort.series, balance_airports_and_flights=True)
    dataset = FixedAnchorTrajectoryWindows(cohort.series, config, normalizer)
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(
        torch.load(args.pretrained_state, map_location=device, weights_only=True)
    )
    batch = dataset.batch(np.arange(len(dataset)))
    x, target, weights, final_time, _flight_weights, dynamics, supervision = batch
    x, target, weights, final_time = (
        x.to(device),
        target.to(device),
        weights.to(device),
        final_time.to(device),
    )
    dynamics = move_dynamics(dynamics, device)
    supervision = supervision.to(device)

    model.eval()
    with torch.no_grad():
        before_prediction = model_forward(model, x, dynamics)
    before_rows, before_median = _evaluate(
        before_prediction, dataset, config, device
    )
    stages = (
        TeacherOptimizationStage("60s", 60.0, args.prefix_steps),
        TeacherOptimizationStage("120s", 120.0, args.prefix_steps),
        TeacherOptimizationStage("240s", 240.0, args.prefix_steps),
        TeacherOptimizationStage("full", None, args.full_steps),
    )
    history = fine_tune_model_on_rollout(
        model,
        x=x,
        target=target,
        weights=weights,
        final_time_s=final_time,
        dynamics=dynamics,
        supervision=supervision,
        config=config,
        normalizer=normalizer,
        stages=stages,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        log_every=args.log_every,
    )
    model.eval()
    with torch.no_grad():
        after_prediction = model_forward(model, x, dynamics)
    after_rows, after_median = _evaluate(after_prediction, dataset, config, device)

    result = {
        "schema_version": SCHEMA,
        "test_policy": "outer-train values only; validation/test values unopened",
        "config": config.to_dict(),
        "pretrained_state": str(args.pretrained_state.resolve()),
        "dataset_ids": [item.dataset_id for item in cohort.series],
        "recipe": {
            "stages": [vars(stage) for stage in stages],
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "objective": "production arc-length-geometry 2+4",
        },
        "median_observed_clock_metrics": {
            "before": before_median,
            "after": after_median,
        },
        "history": history,
        "flight_metrics": [
            {
                "dataset_id": item.dataset_id,
                "before": before,
                "after": after,
            }
            for item, before, after in zip(cohort.series, before_rows, after_rows)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "rollout_gate_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), output_dir / "rollout_finetuned_model_state.pt")
    print(json.dumps(result["median_observed_clock_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
