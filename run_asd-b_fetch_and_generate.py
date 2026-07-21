#!/usr/bin/env python3
"""Harvest one airport and publish its observed CZML in one command.

The canonical downloader is ``trajectory_data_process.harvest``. Unknown arguments are
forwarded to its multi-airport wrapper, so ``--count``, ``--start`` and scan-radius knobs
remain available here. An explicit ``--input-json`` retains the standalone CZML-rendering
utility without introducing a second acquisition path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Harvest ADS-B arrivals and publish the observed CZML",
        epilog="Unknown arguments are forwarded to trajectory_data_process/download_landings.py",
    )
    parser.add_argument("--airport", required=True)
    parser.add_argument("--output-root", default=None, help="harvest output root")
    parser.add_argument("--aeroviz-root", default=None)
    parser.add_argument(
        "--frontend-data",
        default=None,
        help="frontend public/data root used by the harvest and as the CZML copy source",
    )
    parser.add_argument("--czml-output", default=None)
    parser.add_argument("--multiplier", type=int, default=None)
    parser.add_argument("--input-json", default=None,
                        help="render an existing flight-array JSON instead of harvesting")
    parser.add_argument("--generate-procedures", action="store_true")
    parser.add_argument("--cifp-root", default=None)
    parser.add_argument("--procedure-type", default="SIAP")
    parser.add_argument("--procedure-output", default=None)
    parser.add_argument("--include-procedure-transitions", action="store_true")
    parser.add_argument("--procedure-charts-root", default=None)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    repo_root = Path(__file__).resolve().parent
    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else repo_root / "aeroviz-4d"
    harvest_root = (
        Path(args.output_root)
        if args.output_root
        else repo_root / "trajectory_data_process" / "outputs" / "harvest"
    )
    frontend_data = (
        Path(args.frontend_data)
        if args.frontend_data
        else aeroviz_root / "public" / "data"
    )
    published_czml = (
        frontend_data / "airports" / args.airport.upper() / "trajectories.czml"
    )
    requested_czml = Path(args.czml_output) if args.czml_output else published_czml

    generator = aeroviz_root / "python" / "generate_czml.py"
    if args.input_json:
        source = Path(args.input_json)
        if not source.exists():
            raise RuntimeError(f"Input JSON not found: {source}")
        cmd = [
            sys.executable, str(generator), "--airport", args.airport.upper(),
            "--input", str(source), "--output", str(requested_czml),
        ]
        if args.multiplier is not None:
            cmd += ["--multiplier", str(args.multiplier)]
        subprocess.run(cmd, check=True)
    else:
        downloader = repo_root / "trajectory_data_process" / "download_landings.py"
        cmd = [sys.executable, str(downloader), "--airports", args.airport.upper()]
        if not _has_flag(passthrough, "--output"):
            cmd += ["--output", str(harvest_root)]
        cmd += ["--frontend-data", str(frontend_data)]
        if args.multiplier is not None and not _has_flag(passthrough, "--multiplier"):
            cmd += ["--multiplier", str(args.multiplier)]
        cmd += passthrough
        subprocess.run(cmd, cwd=repo_root, check=True)
        if requested_czml.resolve() != published_czml.resolve():
            requested_czml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(published_czml, requested_czml)

    if args.generate_procedures:
        procedure_script = aeroviz_root / "python" / "preprocess_procedures.py"
        procedure_output = (
            Path(args.procedure_output)
            if args.procedure_output
            else frontend_data / "airports" / args.airport.upper() / "procedures.geojson"
        )
        cifp_root = (
            Path(args.cifp_root)
            if args.cifp_root
            else repo_root / "data" / "CIFP" / "CIFP_260319"
        )
        command = [
            sys.executable, str(procedure_script),
            "--cifp-root", str(cifp_root),
            "--airport", args.airport.upper(),
            "--procedure-type", args.procedure_type,
            "--include-all-rnav",
            "--output", str(procedure_output),
        ]
        if args.include_procedure_transitions:
            command.append("--include-transitions")
        if args.procedure_charts_root:
            command.extend(["--charts-root", args.procedure_charts_root])
        subprocess.run(command, check=True)

    print(f"[pipeline] airport: {args.airport.upper()}")
    print(f"[pipeline] CZML:   {requested_czml}")


if __name__ == "__main__":
    main()
