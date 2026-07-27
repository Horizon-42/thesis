#!/usr/bin/env python
"""Run the pooled TS cross-validation stage with the project defaults."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import run_ts_pipeline as pipeline


def _airports(value: str) -> tuple[str, ...]:
    airports = tuple(sorted({
        item.strip().upper() for item in value.split(",") if item.strip()
    }))
    if not airports:
        raise argparse.ArgumentTypeError("--airports requires at least one ICAO code")
    return airports


def _batch_size(value: str) -> str:
    if value == "auto":
        return value
    try:
        batch_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--batch-size must be a positive integer or auto"
        ) from exc
    if batch_size <= 0:
        raise argparse.ArgumentTypeError("--batch-size must be a positive integer or auto")
    return str(batch_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", type=_airports, default=None,
                        help="comma-separated airports; default: all discovered K-airports")
    parser.add_argument("--model", choices=pipeline.MODELS, default="itransformer")
    parser.add_argument("--frame", choices=pipeline.COORDINATE_FRAMES, default="enu")
    parser.add_argument("--batch-size", type=_batch_size, default="2048",
                        help="positive integer or auto (default: 2048)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="run directory; CV artifacts go under cross_validation/")
    parser.add_argument(
        "--random-train-anchor",
        action="store_true",
        help="train from random valid anchors; default: fixed full-trajectory anchor L-1",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    airports = args.airports or tuple(pipeline.discover_k_airports())
    if not airports:
        parser.error(f"no K-airport arrivals found under {pipeline.HARVEST_ROOT}")
    missing = [
        airport for airport in airports
        if not pipeline.arrival_manifest_path(airport).is_file()
    ]
    if missing:
        parser.error(f"missing arrivals manifest for {missing[0]}")

    plan = pipeline.TrainingPlan(
        airports,
        args.model,
        training_mode="pooled",
        seed=1337,
        coordinate_frame=args.frame,
        batch_size=args.batch_size,
        random_train_anchor=args.random_train_anchor,
        output_dir=args.output_dir,
    )
    label, command = plan.cv_step()
    plot_command = [
        sys.executable,
        str(pipeline.REPO_ROOT / "plot_ts_results.py"),
        str(plan.train_dir),
    ]
    candidate_count = math.prod(
        len(values) for values in pipeline.parameter_grid(plan.cv_parameters).values()
    )

    print(f"{label}: {','.join(airports)} · {args.model} · {args.frame}")
    print(f"grid: {','.join(plan.cv_parameters)} ({candidate_count} candidates)")
    print(f"output: {plan.cv_dir}")
    if args.dry_run:
        print(" ".join(command))
        print("after CV: " + " ".join(plot_command))
        return 0

    subprocess.run(command, cwd=pipeline.REPO_ROOT, check=True)
    subprocess.run(plot_command, cwd=pipeline.REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
