"""CLI: build scenario datasets from CZML-input landings files.

    # all runways of one airport (aircraft auto-resolved per flight from icao24):
    python -m flight_scenarios --airport KRDU
    # every runway of every airport under the landings dir:
    python -m flight_scenarios
    # a single explicit file:
    python -m flight_scenarios --input outputs/landings/KRDU/KRDU_05L_landings.json --output scen.json

Each flight becomes one FlightScenario; each landings file yields one scenario JSON (one per
runway), written under --output-dir (or --output for a single --input).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .build import build_scenarios_from_czml_input
from .scenario import save_scenarios
from .start_state import DEFAULT_WINDOW_S

DEFAULT_LANDINGS_DIR = Path("trajectory_data_process/outputs/landings")
DEFAULT_OUTPUT_DIR = Path("flight_scenarios/outputs")


def discover_landings(
    *,
    input_path: str | Path | None = None,
    airport: str | None = None,
    landings_dir: str | Path = DEFAULT_LANDINGS_DIR,
) -> list[Path]:
    """The landings files to process, by mode.

    - ``input_path`` → just that file.
    - ``airport`` → ``<landings_dir>/<AIRPORT>/*_landings.json`` (all runways of one airport).
    - neither → ``<landings_dir>/*/*_landings.json`` (all runways of all airports).

    The ``*_combined_czml_input.json`` files are skipped (only ``*_landings.json`` matches).
    """
    if input_path:
        return [Path(input_path)]
    landings_dir = Path(landings_dir)
    pattern = f"{airport}/*_landings.json" if airport else "*/*_landings.json"
    return sorted(landings_dir.glob(pattern))


def _output_path(landings_file: Path, *, output: str | None, single_input: bool, output_dir: str | Path) -> Path:
    if single_input and output:
        return Path(output)
    stem = landings_file.stem.replace("_landings", "")
    return Path(output_dir) / f"{stem}_scenarios.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flight scenarios from CZML-input landings data")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--input", default=None,
        help="A single CZML-input landings JSON (*_landings.json). Omit to run over the landings dir.",
    )
    target.add_argument(
        "--airport", default=None,
        help="Airport code (e.g. KRDU): process every runway of this airport. "
             "If neither --input nor --airport is given, process every runway of every airport.",
    )
    parser.add_argument(
        "--landings-dir", default=str(DEFAULT_LANDINGS_DIR),
        help=f"Root landings dir for --airport / all-airports discovery (default: {DEFAULT_LANDINGS_DIR})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output scenario JSON path (only valid with --input; batch modes use --output-dir)",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for per-runway scenario files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--aircraft-type", default="A320",
        help="Fallback aircraft code for flights whose icao24 can't be resolved (default: A320); "
             "aircraft is normally auto-resolved per flight from its icao24",
    )
    parser.add_argument("--mass-kg", type=float, default=None, help="Override mass (defaults to the aircraft's max-take-off mass)")
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S, help="Finite-difference window for the start state")
    args = parser.parse_args()

    if args.output and not args.input:
        parser.error("--output is only valid with --input; batch modes write to --output-dir")

    landings_files = discover_landings(input_path=args.input, airport=args.airport, landings_dir=args.landings_dir)
    if not landings_files:
        parser.error(
            "no *_landings.json files found to process "
            f"(input={args.input!r}, airport={args.airport!r}, landings-dir={args.landings_dir!r})"
        )

    total = 0
    for landings_file in landings_files:
        scenarios = build_scenarios_from_czml_input(
            str(landings_file), args.aircraft_type, mass_kg=args.mass_kg, window_s=args.window_s
        )
        out_path = _output_path(
            landings_file, output=args.output, single_input=bool(args.input), output_dir=args.output_dir
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_scenarios(scenarios, out_path)
        distribution = Counter(scenario.aircraft.code for scenario in scenarios)
        total += len(scenarios)
        print(f"✓ {landings_file.name}: {len(scenarios)} scenario(s) -> {out_path}")
        print(f"    aircraft: {dict(distribution)}")

    print(f"✓ done: {len(landings_files)} file(s), {total} scenario(s) total (fallback type: {args.aircraft_type})")


if __name__ == "__main__":
    main()
