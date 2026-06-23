"""dynamics_comparison_history.py
================================
Persistence + averaging for Dynamics Comparison runs.

Each run is stored as one JSON file under ``HISTORY_DIR`` (a record = the run's
meta + its chart series).  The averaging endpoint reads every stored record and
returns a single averaged chart — ALL of the averaging math lives here on the
backend (the frontend only plots the result).

Runs have different horizons (so different distance grids and lengths), so to
average them they are resampled onto one common distance grid (0 .. the *shortest*
run's max distance, the range every run covers) by linear interpolation, then
averaged per system per field.  Final deviations are averaged from each run's own
end-of-flight value.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from aeroviz_backend import paths

# One JSON file per run, kept out of the source tree's tracked code.
HISTORY_DIR = paths.REPO_ROOT / "dynamics_comparison_history"

COMPARED_KEYS = ("A", "C", "D")
_ERROR_FIELDS = ("horiz", "alt", "head", "speed", "fpa")
_AVERAGE_GRID_POINTS = 120

# Writes use unique filenames so the threading HTTP server never collides; the
# lock only serialises the (rare) clear-all.
_LOCK = threading.Lock()


def save_record(record: dict[str, Any], now_iso: str) -> int:
    """Persist one run record; return the new total record count."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stored = {"version": 1, "savedAtIso": now_iso, **record}
    path = HISTORY_DIR / f"run_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(stored))
    return _count()


def history_count() -> int:
    return _count()


def clear_history() -> int:
    with _LOCK:
        if HISTORY_DIR.exists():
            for path in HISTORY_DIR.glob("run_*.json"):
                path.unlink()
    return 0


def average_history() -> dict[str, Any] | None:
    """Average every stored run onto a common distance grid.

    Returns ``{"runCount", "chart"}`` (chart matching the run chart shape so the
    frontend reuses the same plot), or ``None`` when there is no usable history.
    """
    records = _load_records()
    usable = [r for r in records if _chart_distance(r)]
    if not usable:
        return None

    common_max = min(_chart_distance(r)[-1] for r in usable)
    if common_max <= 0.0:
        grid = [0.0]
    else:
        grid = [common_max * i / (_AVERAGE_GRID_POINTS - 1) for i in range(_AVERAGE_GRID_POINTS)]

    count = len(usable)
    series: dict[str, dict[str, list[float]]] = {}
    final: dict[str, dict[str, float]] = {}
    for key in COMPARED_KEYS:
        series[key] = {}
        final[key] = {}
        for field in _ERROR_FIELDS:
            stacked = np.zeros(len(grid))
            final_sum = 0.0
            for record in usable:
                chart = record["chart"]
                x = chart["distanceKm"]
                # Records written before a field existed (e.g. "fpa") simply
                # contribute zeros for it — the only sensible value for a metric
                # that run never recorded.
                y = chart["series"][key].get(field, [0.0] * len(x))
                stacked += np.interp(grid, x, y)
                final_sum += chart["final"][key].get(field, 0.0)
            series[key][field] = [round(float(v), 6) for v in (stacked / count)]
            final[key][field] = round(final_sum / count, 6)

    return {
        "runCount": count,
        "chart": {
            "distanceKm": [round(g, 5) for g in grid],
            "timeS": [],  # averaged over distance, not time
            "series": series,
            "final": final,
        },
    }


def _chart_distance(record: dict[str, Any]) -> list[float]:
    return record.get("chart", {}).get("distanceKm", [])


def _load_records() -> list[dict[str, Any]]:
    if not HISTORY_DIR.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(HISTORY_DIR.glob("run_*.json")):
        records.append(json.loads(path.read_text()))
    return records


def _count() -> int:
    if not HISTORY_DIR.exists():
        return 0
    return sum(1 for _ in HISTORY_DIR.glob("run_*.json"))
