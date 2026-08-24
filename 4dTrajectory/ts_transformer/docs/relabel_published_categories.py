#!/usr/bin/env python
"""One-off: rewrite legacy ``ts_*`` comparison-category labels to the canonical grammar.

The frontend's published categories carry three label dialects from before
``run_naming`` existed. Publisher-managed categories (``prediction_*`` /
``experiment_*``) are refreshed by ``publish_ts_experiment_trajectories.py
--refresh-labels-only``; this script covers the hand-published remainder:

- ``ts_pooled_{itr,ptst}_normalized_time_{train,val}`` — the pooled state baselines,
  resolved to their POOLED run directories.
- ``ts_{icao}_{campaign}_{arm}`` — per-airport experiment arms, resolved by scanning
  ``4dTrajectory/outputs/<ICAO>/experiments/<campaign>/<arm>/history.json`` (plus an
  explicit override table for keys whose spelling dropped part of the campaign name).

Only the ``label`` field of ``comparison/categories.json`` is rewritten — no CZML, no
records, no checkpoints, no keys/dirs. Unmatched ``ts_*`` keys abort the run loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR, REPO_ROOT):  # repo root: config -> channels -> aerodynamic_model
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_naming import category_display_label, run_display_name  # noqa: E402

AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"

POOLED_STATE_RUNS = {
    "itr": OUTPUTS_ROOT / "POOLED" / "ts_itransformer_normalized_time",
    "ptst": OUTPUTS_ROOT / "POOLED" / "ts_patchtst_normalized_time",
}

# Keys whose spelling dropped part of the campaign directory name.
KEY_OVERRIDES = {
    "ts_ksjc_flight_model_first_order_lag": ("KSJC", "flight_model_paired", "first_order_lag"),
    "ts_ksjc_flight_model_point_mass": ("KSJC", "flight_model_paired", "point_mass"),
}


def _config(run_dir: Path) -> dict:
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    config = history.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{run_dir}/history.json carries no config object")
    return config


def _experiment_key_map(icao: str) -> dict[str, tuple[str, str, Path]]:
    """Every ts_<icao>_<campaign>_<arm> key derivable from the experiments tree."""
    experiments = OUTPUTS_ROOT / icao / "experiments"
    mapping: dict[str, tuple[str, str, Path]] = {}
    if not experiments.is_dir():
        return mapping
    for campaign in sorted(p for p in experiments.iterdir() if p.is_dir()):
        for arm in sorted(p for p in campaign.iterdir() if p.is_dir()):
            if not (arm / "history.json").is_file():
                continue
            key = f"ts_{icao.lower()}_{campaign.name}_{arm.name}"
            mapping[key] = (campaign.name, arm.name, arm)
    for key, (over_icao, campaign, arm) in KEY_OVERRIDES.items():
        if over_icao == icao:
            mapping[key] = (campaign, arm, experiments / campaign / arm)
    return mapping


def _new_label(key: str, icao: str, split: str | None) -> str | None:
    if key.startswith("ts_pooled_"):
        # ts_pooled_<model>_normalized_time_<split>
        parts = key.split("_")
        run_dir = POOLED_STATE_RUNS.get(parts[2])
        if run_dir is None or not (run_dir / "history.json").is_file():
            return None
        return category_display_label(
            split or parts[-1],
            run_display_name(_config(run_dir), extra=("pooled cohort",)),
        )
    campaign_map = _experiment_key_map(icao)
    resolved = campaign_map.get(key)
    if resolved is None:
        return None
    campaign, arm, run_dir = resolved
    return category_display_label(
        split or "val",
        run_display_name(_config(run_dir), extra=(f"{campaign}/{arm}",)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print changes, write nothing")
    args = parser.parse_args()

    unmatched: list[str] = []
    changed = 0
    for manifest_path in sorted(AIRPORTS_ROOT.glob("*/comparison/categories.json")):
        icao = manifest_path.parents[1].name
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        touched = False
        for category in document["categories"]:
            key = category.get("key", "")
            if not key.startswith("ts_"):
                continue
            label = _new_label(key, icao, category.get("datasetSplit"))
            if label is None:
                unmatched.append(f"{icao}/{key}")
                continue
            if category.get("label") != label:
                print(f"{icao}/{key}\n  - {category.get('label')}\n  + {label}")
                category["label"] = label
                touched = True
                changed += 1
        if touched and not args.dry_run:
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)

    if unmatched:
        print("\nERROR: unmatched ts_* categories (no run directory found):", file=sys.stderr)
        for entry in unmatched:
            print(f"  {entry}", file=sys.stderr)
        return 1
    print(f"\n{'would relabel' if args.dry_run else 'relabelled'} {changed} categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
