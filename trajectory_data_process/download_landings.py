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
    check_history_access,
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
    p.add_argument("--chunk-hours", type=float, default=6.0, help="Hours per history query")
    p.add_argument("--dry-give-up-days", type=float, default=4.0, help="Give up a runway after this many days scanned with no new landing")
    p.add_argument("--bbox-radius-km", type=float, default=30.0, help="Terminal-area query box radius around the airport")
    p.add_argument("--runway-threshold-radius-m", type=float, default=1000.0)
    p.add_argument("--output-root", default=str(here / "outputs" / "landings"))
    p.add_argument("--overwrite", action="store_true", help="Refetch from scratch instead of resuming existing files")
    return p.parse_args()


def _existing_flights(path: Path) -> list[dict]:
    """Load already-downloaded landings, or an empty list."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def main() -> None:
    args = parse_args()
    config = load_runway_config(Path(args.config))
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")) if args.start else datetime.now(timezone.utc)
    output_root = Path(args.output_root)
    wanted = {a.upper() for a in args.airports} if args.airports else None

    entries = [(p, t) for p, t in iter_airport_entries(config) if not wanted or p.code in wanted]
    if not entries:
        raise SystemExit("No matching airports in config for the requested --airports.")

    # By default, resume: load existing files and only work airports still short of --count.
    def landing_path(code: str, ident: str) -> Path:
        return output_root / code / f"{code}_{ident}_landings.json"

    plans = []
    for profile, thresholds in entries:
        preloaded = (
            {}
            if args.overwrite
            else {t.ident: _existing_flights(landing_path(profile.code, t.ident)) for t in thresholds}
        )
        needs_work = args.overwrite or any(len(preloaded.get(t.ident, [])) < args.count for t in thresholds)
        plans.append((profile, thresholds, preloaded, needs_work))

    if not any(needs_work for *_, needs_work in plans):
        print("[landings] all thresholds already satisfied; nothing to do (use --overwrite to refetch).")
        return

    first = next(profile for profile, _t, _p, needs in plans if needs)
    print("[landings] preflight: checking OpenSky history access...", flush=True)
    check_history_access(profile=first, reference=start, bbox_radius_km=args.bbox_radius_km)

    summary: dict[str, dict[str, int]] = {}
    for profile, thresholds, preloaded, needs_work in plans:
        airport_dir = output_root / profile.code
        summary[profile.code] = {}
        if not needs_work:
            for t in thresholds:
                summary[profile.code][t.ident] = len(preloaded[t.ident])
            print(f"[landings] {profile.code}: already complete, skipped (use --overwrite to refetch)")
            continue

        collected = download_airport_landings(
            profile=profile,
            thresholds=thresholds,
            count=args.count,
            start=start,
            max_lookback_days=args.max_lookback_days,
            chunk_hours=args.chunk_hours,
            bbox_radius_km=args.bbox_radius_km,
            runway_threshold_radius_m=args.runway_threshold_radius_m,
            dry_give_up_days=args.dry_give_up_days,
            preloaded=None if args.overwrite else preloaded,
        )
        airport_dir.mkdir(parents=True, exist_ok=True)
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
