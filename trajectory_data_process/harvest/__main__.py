"""CLI: harvest one airport, then judge its observed approaches.

    python -m trajectory_data_process.harvest --airport KRDU --count 200
    python -m trajectory_data_process.harvest --airport KRDU --evaluate-only

Two stages, separately runnable, because they cost very differently: ``tracks/`` is
re-derived only by re-downloading, while ``approach/`` is a pure recomputation that any
change to the fit window, the established criteria or the TCH source invalidates.
``--evaluate-only`` rebuilds the second from the first.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import evaluate_batch

from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.observed import (
    REPORT_NAME,
    load_observed_records,
    write_observed_records,
)
from trajectory_data_process.harvest.runner import HarvestPlan, harvest_airport
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json"
DEFAULT_OUTPUT = REPO_ROOT / "trajectory_data_process/outputs/harvest"
DEFAULT_CIFP = REPO_ROOT / "data/CIFP/CIFP_260319/FAACIFP18"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trajectory_data_process.harvest")
    parser.add_argument("--airport", required=True, help="ICAO code, e.g. KRDU")
    parser.add_argument("--count", type=int, default=200,
                        help="assigned landings wanted per runway (default: 200)")
    parser.add_argument("--start", help="ISO UTC instant to scan backward from (default: now)")
    parser.add_argument("--max-lookback-days", type=float, default=30.0)
    parser.add_argument("--chunk-hours", type=float, default=6.0)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cifp", type=Path, default=DEFAULT_CIFP,
                        help="ARINC 424 CIFP file supplying per-runway published TCH")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="skip the download; rebuild approach/ from the stored tracks")
    parser.add_argument("--no-cache", action="store_true", help="bypass the history query cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    airport = load_airport(args.airport, config_file=args.config, cifp_file=args.cifp)
    paths = HarvestPaths(root=args.output, code=args.airport)

    if args.evaluate_only:
        manifest = read_manifest(paths)
        print(f"[harvest] reusing stored tracks: {manifest['counts']}")
    else:
        result = harvest_airport(
            airport,
            paths,
            HarvestPlan(
                target_per_runway=args.count,
                start=_parse_start(args.start),
                chunk_hours=args.chunk_hours,
                max_lookback_days=args.max_lookback_days,
                radius_km=args.radius_km,
                cached=not args.no_cache,
            ),
        )
        manifest = result.manifest
        print(f"[harvest] {args.airport}: {manifest['counts']} over "
              f"{result.chunks_fetched} chunks")

    summary = write_observed_records(airport, paths)
    records = load_observed_records(paths)
    report = evaluate_batch(records)
    (paths.approach / REPORT_NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")

    _print_digest(args.airport, manifest, summary, report)
    return 0


def _print_digest(code: str, manifest: dict, summary: dict, report: dict) -> None:
    counts = manifest["counts"]
    print(f"\n{code} harvest")
    print(f"  tracks      : {manifest['total']} total — "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  per runway  : {manifest['per_runway']}")
    if summary["skipped"]:
        print(f"  skipped     : {len(summary['skipped'])} (no published LPV TCH)")
    observed = report.get("observed", {})
    if observed:
        print(f"  established : {observed['established']}/{summary['total']} "
              f"({observed['established_rate']:.0%})  "
              f"not established {observed['not_established']}")
        print(f"  gates       : {report['successful']} pass of {report['measured']} measured "
              f"— {observed['marginal']} marginal (the data cannot decide)")
    if report.get("lateral_m"):
        print(f"  lateral  m  : {report['lateral_m']}")
        print(f"  vertical m  : {report['vertical_m']}")


def _parse_start(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
