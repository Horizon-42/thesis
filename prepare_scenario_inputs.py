#!/usr/bin/env python
"""Prepare observed outputs and scenario JSON for later optimization.

This is the data-preparation half of the scenario pipeline.  It never invokes the
optimizer.

Steps per airport:

  1. trajectory_data_process.harvest --evaluate-only
       stored tracks ─► arrivals/manifest.json + observed evaluation/CZML
  2. flight_scenarios
       arrivals/manifest.json ─► fitted-ADS-B and/or runway-target scenario JSON

The two prepared target datasets feed three optimizer modes: ``fitted_adsb`` uses the
fitted-ADS-B dataset, while ``runway`` and ``runway_cons`` share the runway-target
dataset.

Usage:
    # prepare both target datasets for one airport:
    python prepare_scenario_inputs.py --airport KRDU
    # prepare fitted ADS-B targets only:
    python prepare_scenario_inputs.py --airport KRDU --target-type fitted-adsb
    # reuse the existing arrivals manifest without rebuilding observed outputs:
    python prepare_scenario_inputs.py --airport KRDU --skip-observed
    # prepare every K-airport with stored harvest data:
    python prepare_scenario_inputs.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SCENARIOS_DIR = REPO_ROOT / "flight_scenarios" / "outputs"
HARVEST_TRACKS_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"

TARGET_TYPES = ("fitted-adsb", "runway")
PROGRESS_INTERVAL_S = 30.0


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, remaining = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _run_command_with_progress(
    cmd: list[str],
    *,
    label: str,
    interval_s: float = PROGRESS_INTERVAL_S,
) -> None:
    """Run one inherited-output subprocess with a periodic elapsed-time heartbeat."""
    if interval_s <= 0.0:
        raise ValueError(f"progress interval must be positive, got {interval_s}")
    started = time.monotonic()
    print(f"[progress] {label}: started", flush=True)
    process = subprocess.Popen(cmd, cwd=REPO_ROOT)
    while True:
        try:
            returncode = process.wait(timeout=interval_s)
            break
        except subprocess.TimeoutExpired:
            elapsed = _format_elapsed(time.monotonic() - started)
            print(f"[progress] {label}: still running ({elapsed} elapsed)", flush=True)

    elapsed = _format_elapsed(time.monotonic() - started)
    if returncode != 0:
        print(f"[progress] {label}: failed after {elapsed}", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)
    print(f"[progress] {label}: completed in {elapsed}", flush=True)


def arrival_manifest_path(airport: str) -> Path:
    return HARVEST_TRACKS_ROOT / airport.upper() / "arrivals" / "manifest.json"


def scenario_output_path(airport: str, target_type: str) -> Path:
    code = airport.strip().upper()
    if target_type not in TARGET_TYPES:
        raise ValueError(f"unknown target type {target_type!r}; expected one of {TARGET_TYPES}")
    tag = "_fitted_adsb" if target_type == "fitted-adsb" else "_threshold"
    return SCENARIOS_DIR / f"{code}_arrivals{tag}_scenarios.json"


def discover_k_airports() -> list[str]:
    """Every K-airport with stored tracks or an already-derived arrival manifest."""
    if not HARVEST_TRACKS_ROOT.exists():
        return []
    airports = []
    for child in sorted(HARVEST_TRACKS_ROOT.iterdir()):
        code = child.name.upper()
        if (
            child.is_dir()
            and code.startswith("K")
            and (
                (child / "tracks" / "manifest.json").exists()
                or arrival_manifest_path(code).exists()
            )
        ):
            airports.append(code)
    return airports


def scenario_command(airport: str, target_type: str) -> list[str]:
    """Resolve the scenario-builder command for one airport and target dataset."""
    code = airport.strip().upper()
    output = scenario_output_path(code, target_type)
    cmd = [
        sys.executable,
        "-m",
        "flight_scenarios",
        "--input",
        str(arrival_manifest_path(code)),
        "--output",
        str(output),
    ]
    cmd.append(
        "--target-from-fitted-adsb"
        if target_type == "fitted-adsb"
        else "--target-from-threshold"
    )
    return cmd


def run_observed(airport: str, *, dry_run: bool) -> bool:
    """Rebuild arrivals plus the observed evaluation/CZML from stored tracks."""
    code = airport.strip().upper()
    manifest = HARVEST_TRACKS_ROOT / code / "tracks" / "manifest.json"
    if not manifest.exists():
        print(
            f"   ⚠ {code}: no stored tracks at {manifest.parent} — skipping observed "
            f"rebuild"
        )
        return False
    cmd = [
        sys.executable,
        "-m",
        "trajectory_data_process.harvest",
        "--airport",
        code,
        "--evaluate-only",
    ]
    if dry_run:
        print(f"   [observed] {' '.join(cmd)}")
        return True
    print(f"\n=== [{code} · observed baseline] ===\n{' '.join(cmd)}", flush=True)
    _run_command_with_progress(cmd, label=f"{code} observed rebuild")
    return True


def run_for_airport(
    airport: str,
    target_type: str,
    *,
    dry_run: bool,
    input_will_exist: bool = False,
) -> bool:
    """Build one prepared scenario dataset; return False when its input is missing."""
    code = airport.strip().upper()
    manifest = arrival_manifest_path(code)
    output = scenario_output_path(code, target_type)
    print(f"\n━━ {code}  [{target_type}]  ·  prepare scenario input")
    print(f"   arrivals : {manifest}")
    print(f"   scenarios: {output}")

    if not manifest.exists() and not (dry_run and input_will_exist):
        print(f"   ⚠ skip: missing input {manifest}")
        return False

    cmd = scenario_command(code, target_type)
    if dry_run:
        print(f"   [scenarios] {' '.join(cmd)}")
        return True
    print(f"\n=== [{code} · scenarios] ===\n{' '.join(cmd)}", flush=True)
    _run_command_with_progress(cmd, label=f"{code} {target_type} scenario build")
    print(f"✓ {code} [{target_type}] prepared")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--airport",
        default=None,
        help="airport ICAO; omit to prepare every K-prefixed airport with stored harvest data",
    )
    parser.add_argument(
        "--target-type",
        choices=TARGET_TYPES,
        default=None,
        help="prepare one target dataset; omit to prepare fitted-adsb and runway",
    )
    parser.add_argument(
        "--skip-observed",
        action="store_true",
        help="reuse the existing arrivals manifest without rebuilding observed evaluation/CZML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved paths and commands without running them",
    )
    args = parser.parse_args()

    target_types = TARGET_TYPES if args.target_type is None else (args.target_type,)
    if args.target_type is None:
        print("no --target-type given → preparing fitted-adsb and runway inputs")

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(
                f"no K-prefixed airports with stored harvest data under "
                f"{HARVEST_TRACKS_ROOT}"
            )
        print(
            f"no --airport given → preparing {len(airports)} K-airport(s): "
            f"{', '.join(airports)}"
        )

    prepared = 0
    total = len(airports) * len(target_types)
    stage_total = total + (0 if args.skip_observed else len(airports))
    stage = 0
    for airport in airports:
        observed_scheduled = False
        if not args.skip_observed:
            stage += 1
            print(
                f"\n[progress] stage {stage}/{stage_total}: "
                f"{airport} observed baseline"
            )
            observed_scheduled = run_observed(airport, dry_run=args.dry_run)
        for target_type in target_types:
            stage += 1
            print(
                f"\n[progress] stage {stage}/{stage_total}: "
                f"{airport} {target_type} scenario input"
            )
            if run_for_airport(
                airport,
                target_type,
                dry_run=args.dry_run,
                input_will_exist=args.dry_run and observed_scheduled,
            ):
                prepared += 1

    verb = "previewed" if args.dry_run else "prepared"
    print(
        f"\n✓ {verb} {prepared}/{total} scenario input(s) "
        f"for {len(airports)} airport(s)"
    )


if __name__ == "__main__":
    main()
