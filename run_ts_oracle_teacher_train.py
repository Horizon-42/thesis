#!/usr/bin/env python
"""Formal KSJC train/validation run initialized by train-only oracle schedules."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

from config import (  # noqa: E402
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_BACKENDS,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    MODELS,
    PREDICTION_CONTROL,
    TSConfig,
)
from dataset import (  # noqa: E402
    arrival_data_provenance,
    build_series,
    data_selection_audit,
    flight_keys_by_split,
    load_flight_dicts,
)
from oracle_teacher.pretraining import CachedSchedulePretrainer  # noqa: E402
from oracle_teacher.progressive_pretraining import (  # noqa: E402
    ProgressiveSchedulePretrainer,
)
from train import train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--teacher-schedules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=MODELS, default="itransformer")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--control-dynamics-backend",
        choices=CONTROL_DYNAMICS_BACKENDS,
        default=CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    )
    parser.add_argument(
        "--teacher-pretraining",
        choices=("direct", "progressive"),
        default="direct",
    )
    args = parser.parse_args(argv)
    for name in ("d_model", "d_ff", "e_layers", "n_heads", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.data.is_file():
        parser.error(f"missing arrivals manifest {args.data}")
    if not args.teacher_schedules.is_file():
        parser.error(f"missing teacher schedules {args.teacher_schedules}")
    airport = args.airport.strip().upper()
    config = TSConfig(
        model=args.model,
        prediction_output=PREDICTION_CONTROL,
        aircraft_filter=AIRCRAFT_FILTER_OPENAP_DIRECT,
        control_dynamics_backend=args.control_dynamics_backend,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_state_duration_gradient=False,
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
        control_horizon_curriculum_s=(60.0, 120.0, 240.0),
        control_horizon_curriculum_stage_epochs=10,
        control_gradient_clip_norm=20.0,
        n_segments=64,
        batch_size=args.batch_size,
        epochs=180,
        learning_rate=3e-5,
        lr_plateau_patience=8,
        patience=20,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
        random_train_anchor=False,
        seed=1337,
        split_seed=1337,
        device=args.device,
    )
    provenance = arrival_data_provenance([args.data])
    split_keys = flight_keys_by_split(provenance, config)
    development_keys = set(split_keys["train"] + split_keys["val"])
    print(
        f"loading {len(development_keys)} train/validation arrivals; "
        "outer-test source tracks stay closed"
    )
    series, report = build_series(
        load_flight_dicts([args.data], include_flight_keys=development_keys),
        config,
        airport=airport,
        aircraft_type=config.aircraft_type,
    )
    print(report.format())
    selection = data_selection_audit(series, report, config, split_keys)
    pretrainer = {
        "direct": CachedSchedulePretrainer,
        "progressive": ProgressiveSchedulePretrainer,
    }[args.teacher_pretraining](args.teacher_schedules)
    train(
        series,
        config,
        output_dir=args.output_dir,
        data_provenance=provenance,
        reserved_test_keys=split_keys["test"],
        data_selection=selection,
        model_pretrainer=pretrainer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
