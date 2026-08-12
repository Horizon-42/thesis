"""Publish the observed evaluation to where the frontend reads it.

The frontend already fetches and renders this schema — ``airportEvaluationReportUrl``
→ ``EvaluationReportWindow`` — so publishing is just putting the report where the
comparison-category machinery looks and registering the category.

REPORT-ONLY, NO CZML
--------------------
Every other comparison category ships CZML because it has model trajectories to draw.
The observed category has none: the flown track is already on screen (that IS the
observed layer), and its verdict colouring is applied there from this very report. A
second copy of the same geometry would be one more thing to keep in sync for no
picture the user cannot already see. So ``groups: 0`` — the category exists to carry a
report for the evaluation summary and is excluded from the drawable comparison selector.

CATEGORY SHAPE
--------------
``categories.json`` entries must satisfy the frontend's ``isComparisonCategory``:
``key``/``label``/``dir``/``groups``/``constrained``, all required. ``constrained`` is
false and means what it says elsewhere — whether the SOLVES enforced the procedure as
path constraints. Nothing was solved here, so false is the honest value, not a
placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OBSERVED_CATEGORY_KEY = "observed"
OBSERVED_CATEGORY_LABEL = "Observed ADS-B"
REPORT_NAME = "evaluation_report.json"
CATEGORIES_NAME = "categories.json"


def comparison_root(frontend_data_root: Path, airport: str) -> Path:
    """``public/data/airports/<ICAO>/comparison``."""
    return frontend_data_root / "airports" / airport / "comparison"


def publish_observed_report(
    report: dict[str, Any],
    *,
    frontend_data_root: Path,
    airport: str,
    key: str = OBSERVED_CATEGORY_KEY,
    label: str = OBSERVED_CATEGORY_LABEL,
) -> Path:
    """Write the report into its category dir and register the category. Returns the path."""
    root = comparison_root(frontend_data_root, airport)
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
    _upsert_category(root / CATEGORIES_NAME, key=key, label=label, directory=key)
    return report_path


def _upsert_category(manifest_path: Path, *, key: str, label: str, directory: str) -> None:
    """Add/replace one entry in the shared categories manifest, preserving the others.

    Deliberately a read-modify-write of the same file the CZML builder maintains: the
    observed category has to sit alongside the optimizer ones in a single selector, and
    rewriting the file wholesale would drop whichever categories this run did not build.
    """
    manifest: dict[str, Any] = {"categories": []}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("categories"), list):
            manifest = loaded
    kept = [c for c in manifest["categories"] if c.get("key") != key]
    kept.append(
        {"key": key, "label": label, "dir": directory, "groups": 0, "constrained": False}
    )
    manifest["categories"] = sorted(kept, key=lambda c: c["key"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
