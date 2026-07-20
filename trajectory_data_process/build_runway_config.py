#!/usr/bin/env python3
"""Build the runway-threshold mapping JSON from the OurAirports reference data.

The output (config/runway_thresholds.json) is the reusable, maintainable source of
runway thresholds for the project's main airports. Re-run this whenever the airport
set changes or the OurAirports CSVs are updated.

    python trajectory_data_process/build_runway_config.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.acquisition.airports import FT_TO_M, airports_csv_path
from trajectory_data_process.acquisition.runways import landing_thresholds_from_row, runways_csv_path

DEFAULT_AIRPORTS = ["KRDU", "KMSY", "KSJC", "KSMF", "KSTL"]


def _airport_rows(csv_path: Path, codes: set[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for key in ((row.get("ident") or ""), (row.get("gps_code") or ""), (row.get("icao_code") or "")):
                if key.upper() in codes:
                    rows[key.upper()] = row
    return rows


def build_config(airports: list[str], aeroviz_root: Path) -> dict[str, Any]:
    codes = {a.upper() for a in airports}
    airport_rows = _airport_rows(airports_csv_path(aeroviz_root), codes)

    # v2: ``thresholds[].lat/lon/elevation_m`` are the LANDING threshold (displaced where the
    # source data says so), not the pavement end; ``displaced_threshold_m`` records the shift.
    out: dict[str, Any] = {"schema_version": "runway-thresholds-v2", "airports": {}}
    runways_by_airport: dict[str, list[dict[str, Any]]] = {code: [] for code in codes}

    with runways_csv_path(aeroviz_root).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            code = (row.get("airport_ident") or "").upper()
            if code not in codes or row.get("closed") == "1":
                continue
            if not (row.get("le_latitude_deg") and row.get("he_latitude_deg")):
                continue
            runways_by_airport[code].append(
                {
                    "name": f'{row["le_ident"]}/{row["he_ident"]}',
                    "length_ft": int(row["length_ft"]) if row.get("length_ft") else None,
                    "surface": row.get("surface") or None,
                    # Landing thresholds (displaced where published) — the same computation
                    # resolve_runway_threshold uses, so the two harvest paths agree.
                    "thresholds": landing_thresholds_from_row(row),
                }
            )

    for code in airports:
        code = code.upper()
        meta = airport_rows.get(code)
        if meta is None:
            raise RuntimeError(f"Airport {code} not found in airports.csv")
        elev_ft = meta.get("elevation_ft")
        out["airports"][code] = {
            "name": meta.get("name"),
            "lat": float(meta["latitude_deg"]),
            "lon": float(meta["longitude_deg"]),
            "elevation_m": round(float(elev_ft) * FT_TO_M, 2) if elev_ft else 0.0,
            "runways": sorted(runways_by_airport[code], key=lambda r: r["name"]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build runway_thresholds.json from OurAirports CSVs")
    p.add_argument("--airports", nargs="+", default=DEFAULT_AIRPORTS)
    p.add_argument("--aeroviz-root", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    script_path = Path(__file__).resolve()
    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else script_path.parents[1] / "aeroviz-4d"
    output = Path(args.output) if args.output else script_path.parent / "config" / "runway_thresholds.json"

    config = build_config(args.airports, aeroviz_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    thresholds = sum(len(rw["thresholds"]) for a in config["airports"].values() for rw in a["runways"])
    print(f"[config] airports: {len(config['airports'])}  thresholds: {thresholds}")
    print(f"[config] output: {output}")


if __name__ == "__main__":
    main()
