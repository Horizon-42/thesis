"""CLI: build a scenario dataset from a CZML-input file.

    # aircraft auto-resolved per flight from its icao24 (OpenAP):
    python -m flight_scenarios --input outputs/landings/KRDU/KRDU_05L_landings.json \
        --output scenarios_krdu_05l.json
    # ...with a fallback type for flights whose icao24 can't be resolved:
    python -m flight_scenarios --input … --aircraft-type A320 --output …

Each flight in the input becomes one FlightScenario; they are written as a JSON list
(usable directly by the optimizer or a data-driven model).
"""

from __future__ import annotations

import argparse
from collections import Counter

from .build import build_scenarios_from_czml_input
from .scenario import save_scenarios
from .start_state import DEFAULT_WINDOW_S


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flight scenarios from CZML-input data")
    parser.add_argument("--input", required=True, help="CZML-input JSON (a *_czml_input_*.json / *_landings.json)")
    parser.add_argument("--aircraft-type", default="A320", help="Fallback aircraft code for flights whose icao24 can't be resolved (default: A320); aircraft is normally auto-resolved per flight from its icao24")
    parser.add_argument("--output", required=True, help="Output scenario JSON path")
    parser.add_argument("--mass-kg", type=float, default=None, help="Override mass (defaults to the aircraft's max-take-off mass)")
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S, help="Finite-difference window for the start state")
    args = parser.parse_args()

    scenarios = build_scenarios_from_czml_input(
        args.input, args.aircraft_type, mass_kg=args.mass_kg, window_s=args.window_s
    )
    save_scenarios(scenarios, args.output)
    distribution = Counter(scenario.aircraft.code for scenario in scenarios)
    print(f"✓ wrote {len(scenarios)} scenario(s) -> {args.output}")
    print(f"  aircraft: {dict(distribution)}  (fallback type for unresolved icao24: {args.aircraft_type})")


if __name__ == "__main__":
    main()
