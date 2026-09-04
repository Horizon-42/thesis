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

A control campaign names ``"base_recipe": "simple-v3"`` instead of (or under) ``base``:
the recipe's content becomes the base and the arms override only fields it leaves open.

A PREDICT-ONLY arm reuses an existing checkpoint (no training) and may add predict
options — the inference-time projection arm of the final-approach constraint campaign::

    {"key": "A_project_on_final", "label": "arm A + corridor projection",
     "checkpoint": "4dTrajectory/outputs/{airport}/experiments/airport_frame_20260903/A_threshold_enu/checkpoint.pt",
     "predict_args": ["--project-final", "on-final"]}

``{airport}`` in the checkpoint path is substituted with the campaign's airport.

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

from config import TSConfig, recipe_settings  # noqa: E402
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


def _evaluation_steps(key: str, pred_dir: Path) -> list[tuple[str, list[str], Path]]:
    report = pred_dir / "evaluation_report.json"
    py = sys.executable
    return [
        (f"{key}: evaluation report",
         [py, "-m", "evaluation", "--input", str(pred_dir), "--output", str(report)],
         report),
        (f"{key}: evaluation HTML", [
            py, "-m", "evaluation.visualize", "--input", str(pred_dir),
            "--output", str(pred_dir / "evaluation_report.html"),
        ], pred_dir / "evaluation_report.html"),
    ]


def predict_only_steps(
    key: str, checkpoint: Path, predict_args: list[str], *, airport: str, campaign: Path,
    split: str, device: str,
) -> list[tuple[str, list[str], Path]]:
    """Predict + evaluate from an EXISTING checkpoint: no training, no config of its own."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{key}: checkpoint {checkpoint} does not exist")
    manifest = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    roster = HARVEST_ROOT / airport / "arrivals" / "lateral_pass_eligibility.json"
    pred_dir = campaign / f"{key}_pred_{split}"
    return [
        (f"{key}: predict ({split}, from {checkpoint.parent.name})", [
            sys.executable, str(TS_SCRIPT), "predict",
            "--checkpoint", str(checkpoint),
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--airport", airport,
            "--output-dir", str(pred_dir), "--split", split, "--device", device,
            *predict_args,
        ], pred_dir / "summary.json"),
        *_evaluation_steps(key, pred_dir),
    ]


def arm_steps(
    key: str, label: str, config_path: Path, config: TSConfig, *, airport: str,
    campaign: Path, split: str, device: str, seed: int | None, split_seed: int | None,
    formal: bool = True,
) -> list[tuple[str, list[str], Path]]:
    """(step label, command, artifact whose existence means the step is done)."""
    manifest = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    roster = HARVEST_ROOT / airport / "arrivals" / "lateral_pass_eligibility.json"
    train_dir = campaign / key
    pred_dir = campaign / f"{key}_pred_{split}"
    py = sys.executable
    identity: list[str] = []
    # ``seed``/``split_seed`` inside the arm's overrides win; the CLI supplies defaults.
    if "seed" not in json.loads(config_path.read_text()) and seed is not None:
        identity += ["--seed", str(seed)]
    if "split_seed" not in json.loads(config_path.read_text()) and split_seed is not None:
        identity += ["--split-seed", str(split_seed)]
    # A formal run stamps an experiment manifest (git commit, command, data selection) and
    # refuses a dirty worktree — including untracked files that are not this campaign's.
    formal_identity = ["--campaign-id", campaign.name, "--experiment-id", key] if formal else []
    return [
        (f"{key}: train", [
            py, str(TS_SCRIPT), "train",
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--airport", airport, "--config-overrides", str(config_path),
            *identity, "--device", device, "--output-dir", str(train_dir),
            *formal_identity,
        ], train_dir / "checkpoint.pt"),
        (f"{key}: predict ({split})", [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(train_dir / "checkpoint.pt"),
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--airport", airport,
            "--output-dir", str(pred_dir), "--split", split, "--device", device,
        ], pred_dir / "summary.json"),
        *_evaluation_steps(key, pred_dir),
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
    parser.add_argument(
        "--informal", action="store_true",
        help="skip the experiment manifest (and its clean-worktree guard); the checkpoint "
             "metadata still records the data and recipe, but not the git commit",
    )
    args = parser.parse_args(argv)

    declaration = json.loads(args.arms.read_text(encoding="utf-8"))
    base = declaration.get("base", {})
    # A control campaign starts from a NAMED recipe's content and keeps the name, so an
    # arm may only touch fields the recipe leaves open (TSConfig refuses the rest).
    base_recipe = declaration.get("base_recipe")
    if base_recipe:
        base = {**recipe_settings(base_recipe, keep_name=True), **base}
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
    trained_arms = 0
    for arm in arms:
        key = arm["key"]
        if "checkpoint" in arm:
            if "overrides" in arm:
                parser.error(f"arm {key}: a predict-only arm takes the checkpoint's config, "
                             "not overrides")
            checkpoint = REPO_ROOT / arm["checkpoint"].format(airport=airport)
            print(f"  arm {key:<26s} predict-only from {checkpoint} "
                  f"{' '.join(arm.get('predict_args', []))}")
            steps += predict_only_steps(
                key, checkpoint, list(arm.get("predict_args", [])), airport=airport,
                campaign=campaign, split=args.split, device=args.device,
            )
            continue
        trained_arms += 1
        config_path, config = arm_config(
            base, arm.get("overrides", {}), campaign / key / "config.json"
        )
        print(f"  arm {key:<26s} {run_display_name(config.to_dict(), extra=(key,))}")
        print(f"      slug {run_slug(config.to_dict())}")
        steps += arm_steps(
            key, arm.get("label", key), config_path, config, airport=airport,
            campaign=campaign, split=args.split, device=args.device,
            seed=args.seed, split_seed=args.split_seed, formal=not args.informal,
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
