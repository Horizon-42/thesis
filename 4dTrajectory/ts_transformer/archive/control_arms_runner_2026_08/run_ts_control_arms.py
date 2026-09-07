#!/usr/bin/env python
"""Train, predict, evaluate and publish a set of control-output arms that differ by one axis.

The bank-wiggle investigation (``docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md``)
has a queue of candidate causes, each decided by retraining with ONE field changed. This
runs such a set: every arm starts from the same frozen recipe content, overrides only the
fields the arm names, and goes through the full chain so nothing is lost —

    train    -> checkpoint.pt (+ history, fit_evaluation, checkpoint_metadata)
    predict  -> per-flight records: states, controls, observed lookback
    evaluate -> evaluation_report.json / .html on the optimizer's gates
    publish  -> comparison CZML under aeroviz-4d/public/data/airports/<ICAO>/comparison

Arms are declared in a JSON file::

    {"base_recipe": "simple-v1-lag",
     "arms": [{"key": "velocity_0p25",
               "label": "velocity term 0.25",
               "overrides": {"control_velocity_loss_weight": 0.25}}]}

Every arm shares the manifest, the eligibility roster and ``--split-seed``, so the outer
split is identical and the arms are paired flight-by-flight. Development scope: publishes
train or validation, never outer-test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
TS_SCRIPT = TS_DIR / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"
OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"

if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CONTROL_RECIPE_NAMES, recipe_settings  # noqa: E402


def arm_config_file(base_recipe: str, overrides: dict, destination: Path) -> Path:
    """Write the arm's complete TSConfig overrides, recipe content plus its one change.

    Arms carry ``custom`` deliberately: a named recipe freezes the very fields an
    experiment varies, so running under the name would be refused. The recipe CONTENT is
    still what every arm starts from, and the base name is recorded beside it.
    """
    settings = recipe_settings(base_recipe, keep_name=False)
    unknown = sorted(set(overrides) - set(settings))
    settings.update(overrides)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    if unknown:
        print(f"   note: {', '.join(unknown)} not in the base recipe; taken as-is")
    return destination


def arm_steps(
    key: str, label: str, config_path: Path, *, airport: str, campaign: Path,
    split: str, device: str, seed: int, split_seed: int,
) -> list[tuple[str, list[str]]]:
    manifest = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    roster = HARVEST_ROOT / airport / "arrivals" / "lateral_pass_eligibility.json"
    train_dir = campaign / key
    pred_dir = campaign / f"{key}_pred_{split}"
    report = pred_dir / "evaluation_report.json"
    category = f"ts_{airport.lower()}_{campaign.name}_{key}"
    py = sys.executable
    return [
        (f"{key}: train", [
            py, str(TS_SCRIPT), "train",
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--airport", airport, "--config-overrides", str(config_path),
            "--seed", str(seed), "--split-seed", str(split_seed),
            "--device", device, "--output-dir", str(train_dir),
        ]),
        (f"{key}: predict ({split})", [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(train_dir / "checkpoint.pt"),
            "--data", str(manifest), "--eligibility-roster", str(roster),
            "--output-dir", str(pred_dir), "--split", split, "--device", device,
        ]),
        (f"{key}: evaluation report",
         [py, "-m", "evaluation", "--input", str(pred_dir), "--output", str(report)]),
        (f"{key}: evaluation HTML", [
            py, "-m", "evaluation.visualize", "--input", str(pred_dir),
            "--output", str(pred_dir / "evaluation_report.html"),
        ]),
        (f"{key}: comparison CZML", [
            py, str(CZML_SCRIPT),
            "--summary", str(pred_dir / "summary.json"),
            "--output-dir", str(COMPARISON_AIRPORTS_ROOT / airport / "comparison" / category),
            "--airport", airport, "--category", category, "--category-label", label,
            "--dataset-split", split, "--evaluation-report", str(report),
        ]),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True, help="JSON arm declaration")
    parser.add_argument("--campaign", type=Path, required=True,
                        help="output directory; its name also tags the published category")
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--split", default="val", choices=("train", "val"),
                        help="publication split; outer-test is deliberately not offered")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    declaration = json.loads(args.arms.read_text(encoding="utf-8"))
    base_recipe = declaration["base_recipe"]
    if base_recipe not in CONTROL_RECIPE_NAMES:
        parser.error(f"unknown base recipe {base_recipe!r}; choose from {CONTROL_RECIPE_NAMES}")
    arms = declaration["arms"]
    if not arms:
        parser.error("the arm declaration is empty")

    airport = args.airport.upper()
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    print(f"control-arm campaign · {airport} · base {base_recipe} · split={args.split}")
    print(f"campaign: {campaign}")

    steps: list[tuple[str, list[str]]] = []
    for arm in arms:
        config_path = arm_config_file(
            base_recipe, arm.get("overrides", {}), campaign / arm["key"] / "config.json"
        )
        print(f"  arm {arm['key']:<22s} {arm.get('overrides', {})}")
        steps += arm_steps(
            arm["key"], arm.get("label", arm["key"]), config_path,
            airport=airport, campaign=campaign, split=args.split, device=args.device,
            seed=args.seed, split_seed=args.split_seed,
        )

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
