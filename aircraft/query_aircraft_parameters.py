#!/usr/bin/env python3
"""Resolve an aircraft (by ICAO24, registration, or typecode) to an ``Aircraft``.

The OpenAP cache supplies geometry / mass / engine / drag. OpenAP has no *approach*
envelope (reference speeds, final-approach geometry), so a category-based default fills
that group — refine per type as needed.

CLI:
    python aircraft/query_aircraft_parameters.py 4951d9

Python:
    from aircraft.query_aircraft_parameters import get_aircraft_parameters
    aircraft = get_aircraft_parameters("A320")
    print(aircraft.geometry.wing_area_m2, aircraft.engine.max_thrust_total_n)
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from aircraft.aircraft_sets import Aircraft, Approach, Drag, Engine, Geometry, Mass

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMETERS_PATH = SCRIPT_DIR / "openap_aircraft_parameters.json"
LOOKUP_PATH = SCRIPT_DIR / "aircraft_id_lookup.json"


# OpenAP carries no approach envelope; default one by MTOW class (mirrors the hand-tuned
# presets in aircraft_sets). OpenAP's own ``category`` (transport_jet /
# business_or_general_aviation / unknown) can't discriminate — it lumps the A318 and the
# 777 into "transport_jet" — so weight is the right key. Refine per type when better data
# is available.
_GENERAL_AVIATION_APPROACH = Approach(65.0, 60.0, 75.0, 2.0, 5.0, 0.5, 3.0, 15.0, 800.0)
_NARROW_BODY_APPROACH = Approach(145.0, 135.0, 155.0, 5.0, 10.0, 0.8, 3.0, 15.0, 40000.0)
_WIDE_BODY_APPROACH = Approach(155.0, 145.0, 165.0, 6.0, 12.0, 1.0, 3.0, 15.0, 140000.0)

# MTOW class boundaries (kg): 5 700 = the light/large-aircraft regulatory split;
# 150 000 ≈ the narrow-body/wide-body split (A321 ~93 t … B767 ~186 t).
_LIGHT_MAX_TAKEOFF_KG = 5_700.0
_WIDE_BODY_MIN_TAKEOFF_KG = 150_000.0


def _default_approach(max_takeoff_kg: float | None) -> Approach:
    """Pick a default approach envelope by maximum take-off weight."""
    if max_takeoff_kg is None:
        return _NARROW_BODY_APPROACH
    if max_takeoff_kg < _LIGHT_MAX_TAKEOFF_KG:
        return _GENERAL_AVIATION_APPROACH
    if max_takeoff_kg >= _WIDE_BODY_MIN_TAKEOFF_KG:
        return _WIDE_BODY_APPROACH
    return _NARROW_BODY_APPROACH


class AircraftLookupError(LookupError):
    """Raised when a requested aircraft id cannot be resolved to OpenAP data."""


def fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def normalize_id(value: str | None) -> str:
    return (value or "").strip().upper()


@lru_cache(maxsize=None)
def load_json(path: Path) -> dict[str, Any]:
    """Load + cache a JSON file (the OpenAP cache is large and read once per aircraft)."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_typecode(aircraft_id: str, parameters: dict[str, Any], lookup: dict[str, Any]) -> tuple[str, str]:
    normalized = normalize_id(aircraft_id)
    if normalized in parameters.get("typecodes", {}):
        return normalized, "typecode"

    typecode = lookup.get("icao24_to_typecode", {}).get(normalized)
    if typecode:
        return normalize_id(typecode), "icao24"

    typecode = lookup.get("registration_to_typecode", {}).get(normalized)
    if typecode:
        return normalize_id(typecode), "registration"

    raise AircraftLookupError(f"Aircraft id {normalized} was not found in {LOOKUP_PATH.name}.")


def get_aircraft_parameters(aircraft_id: str) -> Aircraft:
    """Resolve ``aircraft_id`` to an :class:`Aircraft` built from the OpenAP cache."""
    parameters = load_json(PARAMETERS_PATH)
    lookup = load_json(LOOKUP_PATH)
    typecode, _resolved_by = resolve_typecode(normalize_id(aircraft_id), parameters, lookup)

    record = parameters.get("typecodes", {}).get(typecode)
    if not record:
        raise AircraftLookupError(f"Typecode {typecode} is not present in {PARAMETERS_PATH.name}.")
    if not record.get("openap_supported"):
        reason = record.get("error", "not supported by OpenAP")
        raise AircraftLookupError(f"{aircraft_id} resolves to {typecode}, but {reason}")

    data = record["parameters"]
    geometry = data.get("geometry", {})
    mass = data.get("mass", {})
    drag = data.get("drag", {})
    engine = data.get("engine", {})
    category = data.get("category")

    return Aircraft(
        code=typecode,
        name=data.get("aircraft_name") or typecode,
        category=category or "unknown",
        geometry=Geometry(
            wing_area_m2=geometry.get("wing_area_m2"),
            wing_span_m=geometry.get("wing_span_m"),
            wing_mean_chord_m=geometry.get("wing_mac_m"),
            wing_sweep_deg=geometry.get("wing_sweep_deg"),
            fuselage_length_m=geometry.get("fuselage_length_m"),
            fuselage_width_m=geometry.get("fuselage_width_m"),
            fuselage_height_m=geometry.get("fuselage_height_m"),
        ),
        mass=Mass(
            max_takeoff_kg=mass.get("mtow_kg"),
            max_landing_kg=mass.get("mlw_kg"),
            operating_empty_kg=mass.get("oew_kg"),
            max_fuel_kg=mass.get("maximum_fuel_capacity_kg"),
        ),
        engine=Engine(
            count=engine.get("number"),
            max_thrust_n_each=engine.get("max_thrust_n_each"),
            model=engine.get("default") or engine.get("type"),
            cruise_thrust_n_each=engine.get("cruise_thrust_n_each"),
            cruise_sfc=engine.get("cruise_sfc"),
        ),
        approach=_default_approach(mass.get("mtow_kg")),
        drag=Drag(
            zero_lift_cd0=drag.get("cd0"),
            induced_drag_factor=drag.get("k"),
            oswald_efficiency=drag.get("e"),
            landing_gear_drag_increment=drag.get("landing_gear_drag_increment"),
        ),
    )


def format_aircraft(aircraft: Aircraft) -> str:
    g, m, e = aircraft.geometry, aircraft.mass, aircraft.engine
    return "\n".join([
        f"Aircraft {aircraft.code} ({aircraft.name}, {aircraft.category})",
        f"  geometry  wing_area_m2={fmt(g.wing_area_m2)}  wing_span_m={fmt(g.wing_span_m)}",
        f"  mass      max_takeoff_kg={fmt(m.max_takeoff_kg)}  max_landing_kg={fmt(m.max_landing_kg)}",
        f"  engine    count={fmt(e.count)}  max_thrust_n_each={fmt(e.max_thrust_n_each)}  total_n={fmt(e.max_thrust_total_n)}",
        f"  approach  Vref_kt={fmt(aircraft.approach.reference_speed_kt)} (category default)",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the resolved Aircraft for one ICAO24, registration, or typecode."
    )
    parser.add_argument("aircraft_id", help="Example: 4951d9, CS-TNY, or A320.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(format_aircraft(get_aircraft_parameters(args.aircraft_id)))
    except AircraftLookupError as exc:
        print(f"Aircraft lookup failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
