#!/usr/bin/env python
"""Refine a deterministic inverse-dynamics teacher cohort using outer-train only.

With no ``--airport`` argument the cohort is balanced across every discovered K-airport.
Repeating ``--airport`` restricts the same pooled workflow to an explicit airport set.
"""

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
from oracle_teacher.cohort import select_outer_train_cohort  # noqa: E402
from oracle_teacher.evaluation import evaluate_schedule, move_dynamics  # noqa: E402
from oracle_teacher.optimization import (  # noqa: E402
    BatchedOracleTeacher,
    TeacherOptimizationStage,
    optimize_teacher_controls,
)
from oracle_teacher.targets import build_inverse_dynamics_target  # noqa: E402


SCHEMA = "ts-oracle-teacher-optimized-cohort-v1-train-only"


def _balanced_cohort_sizes(airports: list[str], total: int) -> dict[str, int]:
    """Allocate one fixed-size teacher cohort as evenly as possible by airport."""
    if not airports:
        raise ValueError("at least one airport is required")
    ordered = sorted(set(airports))
    if total < len(ordered):
        raise ValueError(
            f"cohort size {total} cannot cover all {len(ordered)} airports"
        )
    base, remainder = divmod(total, len(ordered))
    return {
        airport: base + (index < remainder)
        for index, airport in enumerate(ordered)
    }


def _move_batch(batch, device: torch.device):
    x, target, weights, final_time, flight_weights, dynamics, supervision = batch
    del flight_weights
    return (
        x.to(device),
        target.to(device),
        weights.to(device),
        final_time.to(device),
        move_dynamics(dynamics, device),
        supervision.to(device),
    )


def _aggregate(rows: list[dict[str, object]], mode: str) -> dict[str, float]:
    return {
        metric: float(np.median([row[mode][metric] for row in rows]))
        for metric in ("ade_m", "fde_at_last_complete_dt_m", "terminal_distance_m")
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--airport",
        action="append",
        default=None,
        help=(
            "airport to include; repeat for a selected pooled set "
            "(default: every discovered K-airport)"
        ),
    )
    parser.add_argument("--cohort-size", type=int, default=32)
    parser.add_argument("--prefix-steps", type=int, default=30)
    parser.add_argument("--full-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    airports = sorted(
        {
            value.strip().upper()
            for value in (args.airport or pipeline.discover_k_airports())
            if value.strip()
        }
    )
    if not airports:
        parser.error("no airports with arrivals manifests were selected")
    try:
        cohort_sizes = _balanced_cohort_sizes(airports, args.cohort_size)
    except ValueError as exc:
        parser.error(str(exc))
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
    cohort_series = []
    cohort_audit = []
    for airport in airports:
        cohort = select_outer_train_cohort(
            pipeline.arrival_manifest_path(airport),
            config,
            airport=airport,
            cohort_size=cohort_sizes[airport],
        )
        cohort_series.extend(cohort.series)
        cohort_audit.append({
            "airport": airport,
            "size": len(cohort.series),
            "dataset_ids": [item.dataset_id for item in cohort.series],
            "outer_split_identity_counts": {
                name: len(keys) for name, keys in cohort.split_keys.items()
            },
            "ranked_candidates_opened": list(cohort.ranked_candidates_opened),
        })
    print(
        "teacher cohort: "
        + ", ".join(f"{row['airport']}={row['size']}" for row in cohort_audit)
    )
    normalizer = Normalizer.fit(cohort_series, balance_airports_and_flights=True)
    dataset = FixedAnchorTrajectoryWindows(cohort_series, config, normalizer)
    targets = [build_inverse_dynamics_target(dataset, index) for index in range(len(dataset))]
    device = resolve_device(args.device)
    x, target, weights, final_time, dynamics, supervision = _move_batch(
        dataset.batch(np.arange(len(dataset))), device
    )
    initial_controls = torch.as_tensor(
        np.stack([item.controls for item in targets]), dtype=torch.float32, device=device
    )
    teacher = BatchedOracleTeacher(
        initial_controls,
        dynamics["control_lower"],
        dynamics["control_upper"],
        final_time,
    ).to(device)
    stages = (
        TeacherOptimizationStage("60s", 60.0, args.prefix_steps),
        TeacherOptimizationStage("120s", 120.0, args.prefix_steps),
        TeacherOptimizationStage("240s", 240.0, args.prefix_steps),
        TeacherOptimizationStage("full", None, args.full_steps),
    )
    history = optimize_teacher_controls(
        teacher,
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
    with torch.no_grad():
        optimized = teacher()
    rows: list[dict[str, object]] = []
    for index, item in enumerate(targets):
        initial_metrics = evaluate_schedule(
            item.prediction(device), dataset, index, config, device
        )
        optimized_prediction = type(optimized)(
            controls=optimized.controls[index : index + 1],
            segment_durations=optimized.segment_durations[index : index + 1],
            final_time_s=optimized.final_time_s[index : index + 1],
        )
        optimized_metrics = evaluate_schedule(
            optimized_prediction, dataset, index, config, device
        )
        rows.append(
            {
                "dataset_id": item.dataset_id,
                "initial": initial_metrics,
                "optimized": optimized_metrics,
            }
        )
        print(
            f"{index + 1:02d}/{len(targets)} {item.dataset_id}: "
            f"ADE {initial_metrics['ade_m']:.1f} -> {optimized_metrics['ade_m']:.1f} m"
        )

    controls = optimized.controls.detach().cpu().numpy()
    durations = optimized.segment_durations.detach().cpu().numpy()
    result = {
        "schema_version": SCHEMA,
        "test_policy": "outer-train values only; validation/test values unopened",
        "config": config.to_dict(),
        "cohort": {
            "airports": airports,
            "size": len(targets),
            "dataset_ids": [item.dataset_id for item in targets],
            "allocation": cohort_sizes,
            "by_airport": cohort_audit,
        },
        "recipe": {
            "initialization": "inverse-dynamics",
            "duration": "uniform true outer-train final time / N; frozen",
            "objective": "production arc-length-geometry 2+4",
            "stages": [vars(stage) for stage in stages],
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "median_metrics": {
            "initial": _aggregate(rows, "initial"),
            "optimized": _aggregate(rows, "optimized"),
        },
        "history": history,
        "flights": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "teacher_optimization.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "teacher_schedules.npz",
        dataset_ids=np.asarray([item.dataset_id for item in targets]),
        controls=controls,
        segment_durations_s=durations,
    )
    print(json.dumps(result["median_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
