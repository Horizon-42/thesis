#!/usr/bin/env python
"""Train, predict and evaluate a set of STATE-output arms that differ by one axis.

The airport-center frame ablation (``4dTrajectory/ts_transformer/docs/
2026-09-03_airport_frame_ablation_plan.md``) compares one recipe under three charts —
threshold-anchored, airport-anchored, airport-anchored + target conditioning. Every arm
starts from the same base config, overrides only the fields the arm names, and goes
through the chain that produces comparable numbers::

    train    -> checkpoint.pt (+ history, fit_evaluation, checkpoint_metadata)
    predict  -> per-flight records on the validation split (observed lookback included)
    evaluate -> evaluation_report.json / .html on the optimizer's gates

Arms are declared in a JSON file::

    {"base": {"prediction_output": "state", "model": "itransformer", "horizon_mode": "full"},
     "arms": [{"key": "B_airport_enu", "label": "airport-anchored ENU",
               "overrides": {"coordinate_frame": "airport-enu"}}]}

Every arm shares the manifest, the eligibility roster and ``--split-seed``, so the outer
split is identical and the arms are paired flight-by-flight. Development scope: predicts
train or validation, never outer-test. Deliberately NO per-arm cross-validation (a
search selecting different hyperparameters per arm would confound the axis under test)
and NO comparison-CZML publication (the ts experiment trees are what moves free disk;
publish the arms you want to look at afterwards with
``publish_ts_experiment_trajectories.py``).

Resumable: an arm whose artifact already exists skips that step, so a crash or a stop
costs only the step in flight — ``run_ts_control_arms.py`` had to be re-declared to resume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
TS_SCRIPT = TS_DIR / "__main__.py"
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"

if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import TSConfig  # noqa: E402
from run_naming import run_display_name, run_slug  # noqa: E402

# A state arm on one airport: ~50 MB of checkpoint/history + ~0.3 GB of validation records.
# Refuse to start a campaign the disk cannot hold rather than die mid-arm.
ESTIMATED_BYTES_PER_ARM = 400 * 1024**2
MINIMUM_FREE_BYTES = 2 * 1024**3


def arm_config(base: dict, overrides: dict, destination: Path) -> tuple[Path, TSConfig]:
    """Write the arm's complete override set and return the resolved config for naming."""
    settings = dict(base)
    settings.update(overrides)
    config = TSConfig(**settings)  # validates: an unrunnable arm fails here, before training
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    return destination, config


def arm_steps(
    key: str, label: str, config_path: Path, config: TSConfig, *, airport: str,
    campaign: Path, split: str, device: str, seed: int | None, split_seed: int | None,
) -> list[tuple[str, list[str], Path]]:
    """(step label, command, artifact whose existence means the step is done)."""
    manifest = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    roster = HARVEST_ROOT / airport / "arrivals" / "lateral_pass_eligibility.json"
    train_dir = campaign / key
    pred_dir = campaign / f"{key}_pred_{split}"
    report = pred_dir / "evaluation_report.json"
    py = sys.executable
    identity: list[str] = []
    # ``seed``/``split_seed`` inside the arm's overrides win; the CLI supplies defaults.
    if "seed" not in json.loads(config_path.read_text()) and seed is not None:
        identity += ["--seed", str(seed)]
    if "split_seed" not in json.loads(config_path.read_text()) and split_seed is not None:
        identity += ["--split-seed", str(split_seed)]
    return [
        (f"{key}: train", [
            py, str(TS_SCRIPT), "train",
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--airport", airport, "--config-overrides", str(config_path),
            *identity, "--device", device, "--output-dir", str(train_dir),
            "--campaign-id", campaign.name, "--experiment-id", key,
        ], train_dir / "checkpoint.pt"),
        (f"{key}: predict ({split})", [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(train_dir / "checkpoint.pt"),
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--airport", airport,
            "--output-dir", str(pred_dir), "--split", split, "--device", device,
        ], pred_dir / "summary.json"),
        (f"{key}: evaluation report",
         [py, "-m", "evaluation", "--input", str(pred_dir), "--output", str(report)],
         report),
        (f"{key}: evaluation HTML", [
            py, "-m", "evaluation.visualize", "--input", str(pred_dir),
            "--output", str(pred_dir / "evaluation_report.html"),
        ], pred_dir / "evaluation_report.html"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True, help="JSON arm declaration")
    parser.add_argument("--campaign", type=Path, required=True,
                        help="output directory; one per airport, shared by every arm")
    parser.add_argument("--airport", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"),
                        help="prediction split; outer-test is deliberately not offered")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    declaration = json.loads(args.arms.read_text(encoding="utf-8"))
    base = declaration.get("base", {})
    arms = declaration["arms"]
    if not arms:
        parser.error("the arm declaration is empty")
    airport = args.airport.upper()
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    for name in ("manifest.json", "lateral_pass_eligibility.json"):
        path = HARVEST_ROOT / airport / "arrivals" / name
        if not path.is_file():
            parser.error(f"{path} is missing (a harvest rebuild deletes the roster — "
                         "see trajectory_data_process/CLAUDE.md)")
    print(f"frame-ablation campaign · {airport} · split={args.split}\ncampaign: {campaign}")

    steps: list[tuple[str, list[str], Path]] = []
    for arm in arms:
        key = arm["key"]
        config_path, config = arm_config(
            base, arm.get("overrides", {}), campaign / key / "config.json"
        )
        print(f"  arm {key:<26s} {run_display_name(config.to_dict(), extra=(key,))}")
        print(f"      slug {run_slug(config.to_dict())}")
        steps += arm_steps(
            key, arm.get("label", key), config_path, config, airport=airport,
            campaign=campaign, split=args.split, device=args.device,
            seed=args.seed, split_seed=args.split_seed,
        )

    pending = [step for step in steps if not step[2].exists()]
    free = shutil.disk_usage(campaign if campaign.exists() else REPO_ROOT).free
    needed = ESTIMATED_BYTES_PER_ARM * len({s[0].split(":")[0] for s in pending})
    print(f"  {len(pending)}/{len(steps)} steps pending · free {free / 1024**3:.1f} GiB · "
          f"estimated need {needed / 1024**3:.1f} GiB")
    if not args.dry_run and free - needed < MINIMUM_FREE_BYTES:
        parser.error("not enough free disk for the pending arms; clean up first")

    for index, (label, command, artifact) in enumerate(steps, 1):
        header = f"[{index}/{len(steps)}] {label}"
        if artifact.exists():
            print(f"  {header}: done ({artifact.name} exists), skipping")
            continue
        if args.dry_run:
            print(f"  {header}\n    {' '.join(command)}")
            continue
        print(f"\n=== {header} ===\n{' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
