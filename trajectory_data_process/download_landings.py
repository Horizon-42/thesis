#!/usr/bin/env python3
"""Batch entry point for the canonical harvest downloader.

``python -m trajectory_data_process.harvest`` is the single-airport implementation.
This file deliberately contains no acquisition, reconstruction, or runway-assignment
logic; it only expands an airport list and invokes that implementation once per airport.

Examples::

    python trajectory_data_process/download_landings.py --airports KRDU KSJC --count 200
    python trajectory_data_process/download_landings.py --count 200  # every configured airport
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.harvest.__main__ import main as harvest_main

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "runway_thresholds.json"
DEFAULT_OUTPUT = HERE / "outputs" / "harvest"
DEFAULT_CIFP = HERE.parents[0] / "data" / "CIFP" / "CIFP_260319" / "FAACIFP18"
DEFAULT_FRONTEND_DATA = HERE.parents[0] / "aeroviz-4d" / "public" / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest ADS-B tracks and derive model-ready arrivals for airports"
    )
    parser.add_argument(
        "--airports", nargs="+", default=None,
        help="ICAO codes; omit to harvest every airport in runway_thresholds.json",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--start", default=None, help="ISO UTC instant to scan backward from")
    parser.add_argument("--max-lookback-days", type=float, default=30.0)
    parser.add_argument("--chunk-hours", type=float, default=6.0)
    parser.add_argument("--radius-km", type=float, default=30.0)
    parser.add_argument("--entry-radius-km", type=float, default=25.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cifp", type=Path, default=DEFAULT_CIFP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frontend-data", type=Path, default=DEFAULT_FRONTEND_DATA)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="reuse each airport's tracks/manifest.json and rebuild derived outputs",
    )
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--no-czml", action="store_true")
    parser.add_argument("--multiplier", type=int, default=None)
    return parser


def configured_airports(config_path: Path, requested: list[str] | None) -> list[str]:
    configured = json.loads(config_path.read_text(encoding="utf-8"))["airports"]
    available = {code.upper() for code in configured}
    if requested is None:
        return sorted(available)
    wanted = [code.strip().upper() for code in requested]
    unknown = sorted(set(wanted) - available)
    if unknown:
        raise SystemExit(
            f"airport(s) not present in {config_path}: {', '.join(unknown)}"
        )
    return wanted


def harvest_argv(args: argparse.Namespace, airport: str) -> list[str]:
    argv = [
        "--airport", airport,
        "--count", str(args.count),
        "--max-lookback-days", str(args.max_lookback_days),
        "--chunk-hours", str(args.chunk_hours),
        "--radius-km", str(args.radius_km),
        "--entry-radius-km", str(args.entry_radius_km),
        "--config", str(args.config),
        "--cifp", str(args.cifp),
        "--output", str(args.output),
        "--frontend-data", str(args.frontend_data),
    ]
    if args.start:
        argv += ["--start", args.start]
    if args.multiplier is not None:
        argv += ["--multiplier", str(args.multiplier)]
    for enabled, flag in (
        (args.no_cache, "--no-cache"),
        (args.evaluate_only, "--evaluate-only"),
        (args.no_publish, "--no-publish"),
        (args.no_czml, "--no-czml"),
    ):
        if enabled:
            argv.append(flag)
    return argv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    airports = configured_airports(args.config, args.airports)
    for index, airport in enumerate(airports, 1):
        print(f"\n=== harvest {airport} ({index}/{len(airports)}) ===", flush=True)
        result = harvest_main(harvest_argv(args, airport))
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
