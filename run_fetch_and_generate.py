#!/usr/bin/env python3
"""
Run the OpenSky fetch stage and CZML generation stage as two decoupled steps.

This script intentionally orchestrates through CLI boundaries:
1) opensky_data_query/fetch_cylw_opensky.py
2) aeroviz-4d/python/generate_czml.py

All unknown CLI arguments are forwarded to the fetch script so this wrapper
stays low-coupling when fetch options evolve.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _extract_option(args: list[str], flag: str, default: str) -> str:
    value = default
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            value = args[idx + 1]
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
    return value


def _latest_matching_file(folder: Path, pattern: str, before: set[Path]) -> Path | None:
    candidates = sorted(
        folder.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    created_now = [p for p in candidates if p.resolve() not in before]
    if created_now:
        return created_now[-1]
    if candidates:
        return candidates[-1]
    return None


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run fetch/normalization and CZML generation in one command",
        epilog=(
            "Any unknown arguments are forwarded to opensky_data_query/fetch_cylw_opensky.py.\n"
            "Common forwarded options include: --mode, --allow-partial, --approach-window-min, --altitude-mode."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--airport",
        default="CYYC",
        help="Airport ICAO for output file matching (default: CYYC)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder for fetch stage (defaults to opensky_data_query/outputs)",
    )
    parser.add_argument(
        "--aeroviz-root",
        default=None,
        help="Path to aeroviz-4d folder (defaults to ./aeroviz-4d)",
    )
    parser.add_argument(
        "--czml-output",
        default=None,
        help="Output CZML path (defaults to <aeroviz-root>/public/data/trajectories.czml)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=None,
        help="Optional CZML clock multiplier passed to generate_czml.py",
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help=(
            "Existing JSON to bypass live fetch. Accepts either:\n"
            "- *_raw_*.json: runs normalization/conversion then CZML generation\n"
            "- *_czml_input_*.json: skips straight to CZML generation"
        ),
    )
    parser.add_argument(
        "--disable-normalization",
        action="store_true",
        help="Forwarded to fetch stage: export raw track waypoints without normalization/filtering",
    )
    parser.add_argument(
        "--hours-ago",
        type=float,
        default=None,
        help=(
            "Historical window start offset in hours before now (decimal, e.g. 2.5). "
            "Must be paired with --range-hours. Triggers --mode historical on the fetch stage."
        ),
    )
    parser.add_argument(
        "--range-hours",
        type=float,
        default=None,
        help="Historical window duration in hours (decimal, e.g. 1.5). Must be paired with --hours-ago.",
    )

    # Everything not recognized here is forwarded to fetch_cylw_opensky.py.
    args, fetch_passthrough = parser.parse_known_args()
    return args, fetch_passthrough


def _resolve_historical_window(args: argparse.Namespace) -> tuple[int, int] | None:
    """Translate --hours-ago/--range-hours into (begin, end) unix seconds, or return None."""
    if args.hours_ago is None and args.range_hours is None:
        return None
    if args.hours_ago is None or args.range_hours is None:
        raise RuntimeError("--hours-ago and --range-hours must be provided together.")
    if args.hours_ago <= 0 or args.range_hours <= 0:
        raise RuntimeError("--hours-ago and --range-hours must be positive decimals.")

    now = int(time.time())
    begin = now - int(round(args.hours_ago * 3600))
    end = begin + int(round(args.range_hours * 3600))
    if end <= begin:
        raise RuntimeError("Computed historical window is empty; check --range-hours.")
    return begin, end


def main() -> None:
    args, fetch_passthrough = parse_args()

    historical_window = _resolve_historical_window(args)
    if historical_window is not None:
        if args.input_json:
            raise RuntimeError(
                "--hours-ago/--range-hours cannot be combined with --input-json (fetch stage is bypassed)."
            )
        for conflicting in ("--mode", "--begin", "--end"):
            if _has_flag(fetch_passthrough, conflicting):
                raise RuntimeError(
                    f"--hours-ago/--range-hours conflicts with forwarded {conflicting}; "
                    "remove one of them."
                )

    repo_root = Path(__file__).resolve().parent
    fetch_script = repo_root / "opensky_data_query" / "fetch_cylw_opensky.py"

    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else repo_root / "aeroviz-4d"
    output_root = Path(args.output_root) if args.output_root else repo_root / "opensky_data_query" / "outputs"

    generator_script = aeroviz_root / "python" / "generate_czml.py"
    czml_output_path = (
        Path(args.czml_output)
        if args.czml_output
        else aeroviz_root / "public" / "data" / "trajectories.czml"
    )

    if not fetch_script.exists():
        raise RuntimeError(f"Fetch script not found: {fetch_script}")
    if not generator_script.exists():
        raise RuntimeError(f"CZML generator script not found: {generator_script}")

    output_root.mkdir(parents=True, exist_ok=True)

    czml_input_path: Path
    effective_airport = args.airport.upper()

    if args.input_json:
        input_json_path = Path(args.input_json)
        if not input_json_path.exists():
            raise RuntimeError(f"Input JSON not found: {input_json_path}")

        loaded = json.loads(input_json_path.read_text(encoding="utf-8"))

        if isinstance(loaded, list):
            # Already in CZML-input schema, so go directly to generator.
            czml_input_path = input_json_path
            print("[Pipeline] Fetch stage bypassed: using existing CZML-input JSON.")
        elif isinstance(loaded, dict) and isinstance(loaded.get("tracks"), list):
            # Reuse existing raw payload and run conversion/normalization only.
            effective_airport = str(loaded.get("airport") or args.airport).upper()
            airport_tag = effective_airport.lower()
            pattern = f"{airport_tag}_czml_input_*.json"
            existing = {p.resolve() for p in output_root.glob(pattern)}

            fetch_cmd = [
                sys.executable,
                str(fetch_script),
                "--input-raw-json",
                str(input_json_path),
            ]
            if not _has_flag(fetch_passthrough, "--airport"):
                fetch_cmd.extend(["--airport", effective_airport])
            if not _has_flag(fetch_passthrough, "--output-root"):
                fetch_cmd.extend(["--output-root", str(output_root)])
            if not _has_flag(fetch_passthrough, "--aeroviz-root"):
                fetch_cmd.extend(["--aeroviz-root", str(aeroviz_root)])
            if args.disable_normalization and not _has_flag(fetch_passthrough, "--disable-normalization"):
                fetch_cmd.append("--disable-normalization")
            fetch_cmd.extend(fetch_passthrough)

            print("[Pipeline] Running normalization stage from existing raw JSON...")
            subprocess.run(fetch_cmd, check=True)

            czml_input_path = _latest_matching_file(output_root, pattern, existing)
            if not czml_input_path:
                raise RuntimeError(
                    f"No CZML input JSON produced for airport {effective_airport}. "
                    f"Checked pattern {pattern} under {output_root}."
                )
        else:
            raise RuntimeError(
                "--input-json must be either a raw payload object (with key 'tracks') "
                "or a CZML-input list of flight records."
            )
    else:
        fetch_cmd = [sys.executable, str(fetch_script)]
        if not _has_flag(fetch_passthrough, "--airport"):
            fetch_cmd.extend(["--airport", args.airport])
        if not _has_flag(fetch_passthrough, "--output-root"):
            fetch_cmd.extend(["--output-root", str(output_root)])
        if not _has_flag(fetch_passthrough, "--aeroviz-root"):
            fetch_cmd.extend(["--aeroviz-root", str(aeroviz_root)])
        if args.disable_normalization and not _has_flag(fetch_passthrough, "--disable-normalization"):
            fetch_cmd.append("--disable-normalization")
        if historical_window is not None:
            begin_unix, end_unix = historical_window
            fetch_cmd.extend([
                "--mode", "historical",
                "--begin", str(begin_unix),
                "--end", str(end_unix),
            ])
            print(
                f"[Pipeline] Historical window: {args.hours_ago}h ago for {args.range_hours}h "
                f"(begin={begin_unix}, end={end_unix})."
            )
        fetch_cmd.extend(fetch_passthrough)

        effective_airport = _extract_option(fetch_cmd, "--airport", args.airport).upper()
        airport_tag = effective_airport.lower()
        pattern = f"{airport_tag}_czml_input_*.json"
        existing = {p.resolve() for p in output_root.glob(pattern)}

        print("[Pipeline] Running fetch stage...")
        subprocess.run(fetch_cmd, check=True)

        czml_input_path = _latest_matching_file(output_root, pattern, existing)
        if not czml_input_path:
            raise RuntimeError(
                f"No CZML input JSON produced for airport {effective_airport}. "
                f"Checked pattern {pattern} under {output_root}."
            )

    generate_cmd = [
        sys.executable,
        str(generator_script),
        "--input",
        str(czml_input_path),
        "--output",
        str(czml_output_path),
    ]
    if args.multiplier is not None:
        generate_cmd.extend(["--multiplier", str(args.multiplier)])

    print("[Pipeline] Running CZML generation stage...")
    subprocess.run(generate_cmd, check=True)

    print(f"[Pipeline] airport:      {effective_airport}")
    print(f"[Pipeline] czml input:   {czml_input_path}")
    print(f"[Pipeline] czml output:  {czml_output_path}")


if __name__ == "__main__":
    main()
