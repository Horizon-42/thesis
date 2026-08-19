"""CLI: build scenario datasets from harvest arrival manifests.

    python -m flight_scenarios --airport KRDU --target-from-threshold
    python -m flight_scenarios \
      --input trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json

Every manifest is already a de-duplicated, model-ready, all-runway roster.  There is no
file globbing and no separate "combined" mode.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from trajectory_data_process.harvest.arrivals import resolve_arrival_manifest

from .dataset import build_scenario_dataset, write_selection
from .scenario import save_scenarios
from .start_state import DEFAULT_WINDOW_S

DEFAULT_HARVEST_ROOT = Path("trajectory_data_process/outputs/harvest")
DEFAULT_OUTPUT_DIR = Path("flight_scenarios/outputs")


def discover_arrival_manifests(
    *,
    input_path: str | Path | None = None,
    airport: str | None = None,
    harvest_root: str | Path = DEFAULT_HARVEST_ROOT,
) -> list[Path]:
    if input_path:
        return [resolve_arrival_manifest(input_path)]
    root = Path(harvest_root)
    if airport:
        return [resolve_arrival_manifest(root / airport.upper())]
    return sorted(root.glob("*/arrivals/manifest.json"))


def airport_for_manifest(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    code = manifest.get("airport") if isinstance(manifest, dict) else None
    if not code:
        raise ValueError(f"{path} does not declare an airport")
    return str(code).upper()


def scenario_output_name(
    airport: str, *, threshold: bool, fitted_adsb: bool = False
) -> str:
    if threshold and fitted_adsb:
        raise ValueError("threshold and fitted ADS-B targets are mutually exclusive")
    tag = "_threshold" if threshold else "_fitted_adsb" if fitted_adsb else ""
    return f"{airport}_arrivals{tag}_scenarios.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build flight scenarios from harvest arrivals/manifest.json"
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--input", default=None, help="arrival manifest or airport harvest dir")
    target.add_argument("--airport", default=None, help="airport under --harvest-root")
    parser.add_argument("--harvest-root", default=str(DEFAULT_HARVEST_ROOT))
    parser.add_argument("--output", default=None, help="output path; requires one input/airport")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--aircraft-type", default="A320",
        help="fallback aircraft code when icao24 cannot be resolved",
    )
    target_mode = parser.add_mutually_exclusive_group()
    target_mode.add_argument("--target-from-threshold", action="store_true")
    target_mode.add_argument(
        "--target-from-fitted-adsb", action="store_true",
        help="target the final_approach OLS threshold crossing (position and approach "
             "kinematics fitted)",
    )
    parser.add_argument("--mass-kg", type=float, default=None)
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument(
        "--max-per-runway", type=int, default=None, metavar="N",
        help="keep at most N arrivals per runway, evenly spaced over landing time "
             "(omit for every rostered arrival). The selection depends only on the "
             "roster, so both target datasets pick the SAME flights; it is written to "
             "<output>.selection.json and printed",
    )
    args = parser.parse_args(argv)

    manifests = discover_arrival_manifests(
        input_path=args.input, airport=args.airport, harvest_root=args.harvest_root
    )
    if not manifests:
        parser.error(
            f"no */arrivals/manifest.json under {args.harvest_root}; run the harvest first"
        )
    if args.output and len(manifests) != 1:
        parser.error("--output requires exactly one --input or --airport")

    if args.max_per_runway is not None and args.max_per_runway < 1:
        parser.error(f"--max-per-runway must be >= 1, got {args.max_per_runway}")
    target = (
        "runway" if args.target_from_threshold
        else "fitted-adsb" if args.target_from_fitted_adsb
        else "track-end"
    )

    total = 0
    for manifest in manifests:
        airport = airport_for_manifest(manifest)
        scenarios, selection = build_scenario_dataset(
            manifest,
            args.aircraft_type,
            target=target,
            max_per_runway=args.max_per_runway,
            mass_kg=args.mass_kg,
            window_s=args.window_s,
        )
        output = (
            Path(args.output)
            if args.output
            else Path(args.output_dir)
            / scenario_output_name(
                airport,
                threshold=args.target_from_threshold,
                fitted_adsb=args.target_from_fitted_adsb,
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        save_scenarios(scenarios, output)
        selection_file = write_selection(selection, output)
        distribution = Counter(scenario.aircraft.code for scenario in scenarios)
        print(f"✓ {airport}: {len(scenarios)} scenario(s) -> {output}")
        print(f"    population: {selection.summary_line()}")
        for row in selection.excluded_unfittable:
            print(f"      dropped {row['flight_key']} ({row['runway']}): {row['reason']}")
        print(f"    selection: {selection_file}")
        print(f"    aircraft: {dict(distribution)}")
        total += len(scenarios)

    print(f"✓ done: {len(manifests)} airport(s), {total} scenario(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
