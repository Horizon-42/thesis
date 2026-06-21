#!/usr/bin/env python3
"""Download N historical landing trajectories for every runway threshold.

Reads the runway-threshold mapping (config/runway_thresholds.json) and, for each
airport and each threshold, collects ``--count`` landings into a CZML-input file.
All the work lives in ``landings.py``; this entry point only wires config to disk.

    python trajectory_data_process/download_landings.py --count 20
    python trajectory_data_process/download_landings.py --count 30 --airports KRDU KSJC
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.landings import (
    download_airport_landings,
    iter_airport_entries,
    load_runway_config,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Download landing trajectories per runway threshold")
    p.add_argument("--count", type=int, default=20, help="Landings to collect per runway threshold")
    p.add_argument("--config", default=str(here / "config" / "runway_thresholds.json"))
    p.add_argument("--airports", nargs="+", default=None, help="Subset of airport codes (default: all in config)")
    p.add_argument("--start", default=None, help="Scan backward from this UTC time (ISO, default: now)")
    p.add_argument("--max-lookback-days", type=float, default=30.0)
    p.add_argument("--chunk-hours", type=float, default=6.0)
    p.add_argument("--runway-threshold-radius-m", type=float, default=600.0)
    p.add_argument("--output-root", default=str(here / "outputs" / "landings"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runway_config(Path(args.config))
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")) if args.start else datetime.now(timezone.utc)
    output_root = Path(args.output_root)
    wanted = {a.upper() for a in args.airports} if args.airports else None

    summary: dict[str, dict[str, int]] = {}
    for profile, thresholds in iter_airport_entries(config):
        if wanted and profile.code not in wanted:
            continue
        collected = download_airport_landings(
            profile=profile,
            thresholds=thresholds,
            count=args.count,
            start=start,
            max_lookback_days=args.max_lookback_days,
            chunk_hours=args.chunk_hours,
            runway_threshold_radius_m=args.runway_threshold_radius_m,
        )
        airport_dir = output_root / profile.code
        airport_dir.mkdir(parents=True, exist_ok=True)
        summary[profile.code] = {}
        for ident, flights in collected.items():
            path = airport_dir / f"{profile.code}_{ident}_landings.json"
            path.write_text(json.dumps(flights, indent=2), encoding="utf-8")
            summary[profile.code][ident] = len(flights)
            print(f"[landings] {profile.code} {ident}: {len(flights)}/{args.count} -> {path}")

    summary_path = output_root / f"summary_{start.strftime('%Y%m%dT%H%M%SZ')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[landings] summary: {summary_path}")


if __name__ == "__main__":
    main()
