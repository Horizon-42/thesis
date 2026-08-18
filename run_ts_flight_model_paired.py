#!/usr/bin/env python
"""Paired point-mass vs first-order-lag comparison, published end to end.

The tau_bank cross-validation answered "which time constant" and could not answer "does
the lagged flight model help at all": every CV arm IS the lagged model, and CV discards its
fold models, so nothing it produces can be replayed or drawn. This runs the one comparison
that matters instead — the two frozen recipes, which differ in exactly one field — and
keeps every artifact:

    train      -> checkpoint.pt (+ history, fit_evaluation, checkpoint_metadata)
    predict    -> per-flight prediction records: states, controls, observed lookback
    evaluate   -> evaluation_report.json / .html on the same gates as the optimizer
    publish    -> comparison CZML under aeroviz-4d/public/data/airports/<ICAO>/comparison

Pairing is by construction: both arms take the same manifest, the same eligibility roster
and the same ``--split-seed``, so the outer train/validation/test flights are identical and
the two runs differ only in ``control_dynamics_model``.

Development scope: publishes the validation split. Outer-test is never opened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent
TS_SCRIPT = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"
OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"


@dataclass(frozen=True)
class Arm:
    """One side of the pair: a frozen recipe plus the identity it publishes under."""

    key: str
    recipe: str
    label: str
    extra_train_args: tuple[str, ...] = ()


ARMS = (
    Arm("point_mass", "simple-v1", "point-mass controls"),
    Arm(
        "first_order_lag",
        "simple-v1-lag",
        "first-order-lag controls (tau_bank=2s)",
        ("--control-bank-tau-s", "2.0"),
    ),
)


def arm_commands(
    arm: Arm, *, airport: str, campaign: Path, split: str, device: str, seed: int,
    split_seed: int,
) -> list[tuple[str, list[str]]]:
    manifest = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    roster = HARVEST_ROOT / airport / "arrivals" / "lateral_pass_eligibility.json"
    train_dir = campaign / arm.key
    pred_dir = campaign / f"{arm.key}_pred_{split}"
    report = pred_dir / "evaluation_report.json"
    category = f"ts_{airport.lower()}_flight_model_{arm.key}"
    py = sys.executable
    return [
        (
            f"{arm.key}: train",
            [
                py, str(TS_SCRIPT), "train",
                "--data", str(manifest),
                "--eligibility-roster", str(roster),
                "--airport", airport,
                "--control-recipe", arm.recipe,
                "--seed", str(seed),
                "--split-seed", str(split_seed),
                "--device", device,
                "--output-dir", str(train_dir),
                *arm.extra_train_args,
            ],
        ),
        (
            f"{arm.key}: predict ({split})",
            [
                py, str(TS_SCRIPT), "predict",
                "--checkpoint", str(train_dir / "checkpoint.pt"),
                "--data", str(manifest),
                "--eligibility-roster", str(roster),
                "--output-dir", str(pred_dir),
                "--split", split,
                "--device", device,
            ],
        ),
        (
            f"{arm.key}: evaluation report",
            [py, "-m", "evaluation", "--input", str(pred_dir), "--output", str(report)],
        ),
        (
            f"{arm.key}: evaluation HTML",
            [
                py, "-m", "evaluation.visualize",
                "--input", str(pred_dir),
                "--output", str(pred_dir / "evaluation_report.html"),
            ],
        ),
        (
            f"{arm.key}: comparison CZML",
            [
                py, str(CZML_SCRIPT),
                "--summary", str(pred_dir / "summary.json"),
                "--output-dir",
                str(COMPARISON_AIRPORTS_ROOT / airport / "comparison" / category),
                "--airport", airport,
                "--category", category,
                "--category-label", arm.label,
                "--dataset-split", split,
                "--evaluation-report", str(report),
            ],
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--campaign", type=Path, default=None)
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val"),
        help="publication split; outer-test is deliberately not offered",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    airport = args.airport.upper()
    campaign = args.campaign or (
        OUTPUTS_ROOT / airport / "experiments" / "flight_model_paired"
    )
    steps = [
        step
        for arm in ARMS
        for step in arm_commands(
            arm,
            airport=airport,
            campaign=campaign,
            split=args.split,
            device=args.device,
            seed=args.seed,
            split_seed=args.split_seed,
        )
    ]
    print(f"paired flight-model comparison · {airport} · split={args.split}")
    print(f"campaign: {campaign}")
    for index, (label, command) in enumerate(steps, 1):
        header = f"[{index}/{len(steps)}] {label}"
        if args.dry_run:
            print(f"  {header}\n    {' '.join(command)}")
            continue
        print(f"\n=== {header} ===\n{' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
