#!/usr/bin/env python
"""One-off: backfill per-category ``accuracy`` (mean/p95 ADE/FDE) into categories.json.

``build_scenario_comparison_czml.py`` stamps a compact ``accuracy`` block onto every
prediction category entry (2026-08-24) so the frontend can rank a split's results
without fetching each category's full comparison index. Already-published entries
predate the field; this patches them from each category's own published
``comparison_index.json`` ``prediction`` block — the same numbers the builder would
stamp (it subsets, never recomputes).

Metadata-only: only ``categories.json`` files change. A category that looks like a
prediction result but has no ``prediction`` block is reported loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_DIR = REPO_ROOT / "aeroviz-4d" / "python"
if str(BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_DIR))

from build_scenario_comparison_czml import category_accuracy_summary  # noqa: E402

AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
PREDICTION_KEY_PREFIXES = ("ts_", "prediction_", "experiment_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print changes, write nothing")
    args = parser.parse_args()

    problems: list[str] = []
    patched = 0
    for manifest_path in sorted(AIRPORTS_ROOT.glob("*/comparison/categories.json")):
        icao = manifest_path.parents[1].name
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        touched = False
        for category in document["categories"]:
            index_path = manifest_path.parent / category["dir"] / "comparison_index.json"
            if not index_path.is_file():
                continue  # report-only category (observed)
            index = json.loads(index_path.read_text(encoding="utf-8"))
            accuracy = category_accuracy_summary(index.get("prediction"))
            if accuracy is None:
                if category.get("key", "").startswith(PREDICTION_KEY_PREFIXES):
                    problems.append(
                        f"{icao}/{category['key']}: comparison index has no prediction block"
                    )
                continue  # optimizer category — no such metric
            if category.get("accuracy") != accuracy:
                category["accuracy"] = accuracy
                touched = True
                patched += 1
                print(f"{icao}/{category['key']}: "
                      f"ADE mean {accuracy.get('adeM', {}).get('mean', float('nan')):.0f} m")
        if touched and not args.dry_run:
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)

    if problems:
        print("\nERROR: prediction categories without accuracy:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"\n{'would patch' if args.dry_run else 'patched'} {patched} categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
