"""CLI: build a scenario dataset from a CZML-input file.

    python -m flight_scenarios --input outputs/landings/KRDU/KRDU_05L_landings.json \
        --aircraft A320 --output scenarios_krdu_05l.json

Each flight in the input becomes one FlightScenario; they are written as a JSON list
(usable directly by the optimizer or a data-driven model).
"""

from __future__ import annotations

import argparse

from .build import build_scenarios_from_czml_input
from .scenario import save_scenarios
from .start_state import DEFAULT_WINDOW_S


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flight scenarios from CZML-input data")
    parser.add_argument("--input", required=True, help="CZML-input JSON (a *_czml_input_*.json / *_landings.json)")
    parser.add_argument("--aircraft", required=True, help="Aircraft code for the spec, e.g. A320 / B77W / C172")
    parser.add_argument("--output", required=True, help="Output scenario JSON path")
    parser.add_argument("--mass-kg", type=float, default=None, help="Override mass (defaults to the aircraft spec mass)")
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S, help="Finite-difference window for the start state")
    args = parser.parse_args()

    scenarios = build_scenarios_from_czml_input(
        args.input, args.aircraft, mass_kg=args.mass_kg, window_s=args.window_s
    )
    save_scenarios(scenarios, args.output)
    print(f"✓ wrote {len(scenarios)} scenario(s) -> {args.output}")


if __name__ == "__main__":
    main()
