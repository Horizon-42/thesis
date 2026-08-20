#!/usr/bin/env python
"""Audit inverse-dynamics teacher quality on a deterministic outer-train cohort."""

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
from models import resolve_device  # noqa: E402
from control.oracle.cohort import select_outer_train_cohort  # noqa: E402
from control.oracle.evaluation import evaluate_schedule  # noqa: E402
from control.oracle.targets import (  # noqa: E402
    build_inverse_dynamics_target,
    neutral_prediction,
)


SCHEMA = "ts-oracle-teacher-quality-audit-v1-train-only"


def _summary(rows: list[dict[str, object]], key: str, mode: str) -> dict[str, float]:
    values = np.asarray([row[mode][key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cohort_size < 1:
        parser.error("--cohort-size must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    airport = args.airport.strip().upper()
    manifest = pipeline.arrival_manifest_path(airport)
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
        seed=args.seed,
        split_seed=args.split_seed,
        device=args.device,
    )
    cohort = select_outer_train_cohort(
        manifest,
        config,
        airport=airport,
        cohort_size=args.cohort_size,
    )
    normalizer = Normalizer.fit(cohort.series, balance_airports_and_flights=True)
    dataset = FixedAnchorTrajectoryWindows(cohort.series, config, normalizer)
    device = resolve_device(args.device)

    rows: list[dict[str, object]] = []
    for index in range(len(dataset)):
        target = build_inverse_dynamics_target(dataset, index)
        _x, _y, _weights, _time, _flight_weights, dynamics, _supervision = (
            dataset.batch(np.array([index]))
        )
        teacher = evaluate_schedule(
            target.prediction(device), dataset, index, config, device
        )
        neutral = evaluate_schedule(
            neutral_prediction(
                dynamics,
                n_segments=int(config.n_segments),
                final_time_s=target.final_time_s,
                device=device,
            ),
            dataset,
            index,
            config,
            device,
        )
        rows.append(
            {
                "dataset_id": target.dataset_id,
                "final_time_s": target.final_time_s,
                "reference_points": target.reference_points,
                "clipped_fraction": target.clipped_fraction.tolist(),
                "teacher": teacher,
                "neutral": neutral,
            }
        )
        print(
            f"{index + 1:02d}/{len(dataset)} {target.dataset_id}: "
            f"teacher ADE={teacher['ade_m']:.1f} m, neutral={neutral['ade_m']:.1f} m"
        )

    metric_names = ("ade_m", "fde_at_last_complete_dt_m", "terminal_distance_m")
    summary = {
        mode: {metric: _summary(rows, metric, mode) for metric in metric_names}
        for mode in ("teacher", "neutral")
    }
    teacher_better = np.mean(
        [row["teacher"]["ade_m"] < row["neutral"]["ade_m"] for row in rows]
    )
    result = {
        "schema_version": SCHEMA,
        "test_policy": "outer-train values only; validation/test values unopened",
        "config": config.to_dict(),
        "cohort": {
            "airport": airport,
            "size": len(rows),
            "dataset_ids": [row["dataset_id"] for row in rows],
            "outer_split_identity_counts": {
                name: len(keys) for name, keys in cohort.split_keys.items()
            },
            "ranked_candidates_opened": list(cohort.ranked_candidates_opened),
        },
        "teacher_recipe": {
            "initializer": "inverse-dynamics",
            "velocity_source": "smoothed-position-difference over known future positions",
            "duration": "uniform true outer-train final time / N",
            "optimization_steps": 0,
        },
        "summary": summary,
        "teacher_ade_better_fraction": float(teacher_better),
        "flights": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "teacher_quality.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "teacher_ade_better_fraction": teacher_better}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
