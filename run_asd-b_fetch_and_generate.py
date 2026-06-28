#!/usr/bin/env python3
"""Run the two decoupled pipeline stages in one command.

1) trajectory_data_process/download_trajectories.py  (OpenSky history DB -> *_czml_input_*.json)
2) aeroviz-4d/python/generate_czml.py                (*_czml_input_*.json -> trajectories.czml)

Unknown arguments are forwarded to the download stage, so options such as
``--begin/--end``, ``--runway``, ``--fetch-profile`` flow straight through.

Examples:
    python run_asd-b_fetch_and_generate.py --airport KRDU \\
      --begin 2026-04-19T10:00:00Z --end 2026-04-19T10:30:00Z --runway 23R

    # Skip the download stage and render an existing CZML-input JSON:
    python run_asd-b_fetch_and_generate.py --airport KRDU \\
      --input-json trajectory_data_process/outputs/krdu/krdu_czml_input_20260419T225427Z.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _latest_czml_input(folder: Path, pattern: str, before: set[Path]) -> Path | None:
    candidates = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    created_now = [p for p in candidates if p.resolve() not in before]
    return (created_now or candidates)[-1] if (created_now or candidates) else None


def _airport_output_dir(output_root: Path, airport: str) -> Path:
    """Download output subdirectory for an airport (e.g. <root>/cyvr)."""
    return output_root / airport.lower()


def _default_czml_output(aeroviz_root: Path, airport: str) -> Path:
    """Default front-end CZML path for an airport's trajectories."""
    return aeroviz_root / "public" / "data" / "airports" / airport.upper() / "trajectories.czml"


def _default_procedure_output(aeroviz_root: Path, airport: str) -> Path:
    """Default front-end GeoJSON path for an airport's RNAV/RNP procedures."""
    return aeroviz_root / "public" / "data" / "airports" / airport.upper() / "procedures.geojson"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run trajectory download and CZML generation in one command",
        epilog="Unknown arguments are forwarded to trajectory_data_process/download_trajectories.py "
        "(e.g. --begin, --end, --runway, --fetch-profile).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--airport", required=True, help="Airport ICAO code, e.g. KRDU")
    parser.add_argument("--output-root", default=None, help="Download output root (default: trajectory_data_process/outputs)")
    parser.add_argument("--aeroviz-root", default=None, help="Path to aeroviz-4d (default: ./aeroviz-4d)")
    parser.add_argument("--czml-output", default=None, help="Output CZML path (default: <aeroviz>/public/data/airports/<AIRPORT>/trajectories.czml)")
    parser.add_argument("--multiplier", type=int, default=None, help="Optional CZML clock multiplier")
    parser.add_argument("--input-json", default=None, help="Existing *_czml_input_*.json to render directly (skips download)")
    parser.add_argument("--generate-procedures", action="store_true", help="Also regenerate RNAV/RNP procedure assets")
    parser.add_argument("--cifp-root", default=None, help="CIFP cycle directory for --generate-procedures")
    parser.add_argument("--procedure-type", default="SIAP", help="CIFP procedure type for --generate-procedures")
    parser.add_argument("--procedure-output", default=None, help="Output GeoJSON for --generate-procedures")
    parser.add_argument("--include-procedure-transitions", action="store_true",
                        help="Pass --include-transitions to procedure preprocessing")
    parser.add_argument("--procedure-charts-root", default=None,
                        help="Optional charts root forwarded to procedure preprocessing as --charts-root")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main() -> None:
    args, passthrough = parse_args()
    repo_root = Path(__file__).resolve().parent
    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else repo_root / "aeroviz-4d"
    output_root = Path(args.output_root) if args.output_root else repo_root / "trajectory_data_process" / "outputs"

    download_script = repo_root / "trajectory_data_process" / "download_trajectories.py"
    generator_script = aeroviz_root / "python" / "generate_czml.py"
    for script in (download_script, generator_script):
        if not script.exists():
            raise RuntimeError(f"Required script not found: {script}")

    airport = args.airport.upper()

    if args.input_json:
        czml_input_path = Path(args.input_json)
        if not czml_input_path.exists():
            raise RuntimeError(f"Input JSON not found: {czml_input_path}")
        print("[pipeline] Using existing CZML-input JSON; download stage skipped.", flush=True)
    else:
        airport_dir = _airport_output_dir(output_root, airport)
        pattern = f"{airport.lower()}_czml_input_*.json"
        before = {p.resolve() for p in airport_dir.glob(pattern)}

        download_cmd = [sys.executable, str(download_script), "--airport", airport]
        if not _has_flag(passthrough, "--output-root"):
            download_cmd += ["--output-root", str(output_root)]
        if not _has_flag(passthrough, "--aeroviz-root"):
            download_cmd += ["--aeroviz-root", str(aeroviz_root)]
        download_cmd += passthrough

        print("[pipeline] Running download stage...", flush=True)
        subprocess.run(download_cmd, check=True)

        czml_input_path = _latest_czml_input(airport_dir, pattern, before)
        if not czml_input_path:
            raise RuntimeError(f"No CZML-input JSON produced under {airport_dir} ({pattern}).")

    czml_output_path = (
        Path(args.czml_output)
        if args.czml_output
        else _default_czml_output(aeroviz_root, airport)
    )
    generate_cmd = [
        sys.executable, str(generator_script),
        "--airport", airport,
        "--input", str(czml_input_path),
        "--output", str(czml_output_path),
    ]
    if args.multiplier is not None:
        generate_cmd += ["--multiplier", str(args.multiplier)]

    print("[pipeline] Running CZML generation stage...", flush=True)
    subprocess.run(generate_cmd, check=True)

    if args.generate_procedures:
        procedure_script = aeroviz_root / "python" / "preprocess_procedures.py"
        if not procedure_script.exists():
            raise RuntimeError(f"Procedure script not found: {procedure_script}")
        procedure_output = (
            Path(args.procedure_output)
            if args.procedure_output
            else _default_procedure_output(aeroviz_root, airport)
        )
        cifp_root = Path(args.cifp_root) if args.cifp_root else repo_root / "data" / "CIFP" / "CIFP_260319"
        procedure_cmd = [
            sys.executable, str(procedure_script),
            "--cifp-root", str(cifp_root),
            "--airport", airport,
            "--procedure-type", args.procedure_type,
            "--include-all-rnav",
            "--output", str(procedure_output),
        ]
        if args.include_procedure_transitions:
            procedure_cmd.append("--include-transitions")
        if args.procedure_charts_root:
            procedure_cmd.extend(["--charts-root", str(args.procedure_charts_root)])
        print("[pipeline] Running procedure asset generation stage...", flush=True)
        subprocess.run(procedure_cmd, check=True)
        print(f"[pipeline] procedures:  {procedure_output}")

    print(f"[pipeline] airport:     {airport}")
    print(f"[pipeline] czml input:  {czml_input_path}")
    print(f"[pipeline] czml output: {czml_output_path}")


if __name__ == "__main__":
    main()
