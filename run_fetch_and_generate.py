#!/usr/bin/env python3
"""
Run the OpenSky fetch stage and CZML generation stage as two decoupled steps.

This script intentionally orchestrates through CLI boundaries:
1) opensky_cylw/fetch_cylw_opensky.py
2) aeroviz-4d/python/generate_czml.py

All unknown CLI arguments are forwarded to the fetch script so this wrapper
stays low-coupling when fetch options evolve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
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
        description="Run OpenSky fetch and CZML generation in one command"
    )
    parser.add_argument(
        "--airport",
        default="CYYC",
        help="Airport ICAO for output file matching (default: CYYC)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output folder for fetch stage (defaults to opensky_cylw/outputs)",
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

    # Everything not recognized here is forwarded to fetch_cylw_opensky.py.
    args, fetch_passthrough = parser.parse_known_args()
    return args, fetch_passthrough


def main() -> None:
    args, fetch_passthrough = parse_args()

    repo_root = Path(__file__).resolve().parent
    fetch_script = repo_root / "opensky_cylw" / "fetch_cylw_opensky.py"

    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else repo_root / "aeroviz-4d"
    output_root = Path(args.output_root) if args.output_root else repo_root / "opensky_cylw" / "outputs"

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

    fetch_cmd = [sys.executable, str(fetch_script)]
    if not _has_flag(fetch_passthrough, "--airport"):
        fetch_cmd.extend(["--airport", args.airport])
    if not _has_flag(fetch_passthrough, "--output-root"):
        fetch_cmd.extend(["--output-root", str(output_root)])
    if not _has_flag(fetch_passthrough, "--aeroviz-root"):
        fetch_cmd.extend(["--aeroviz-root", str(aeroviz_root)])
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
