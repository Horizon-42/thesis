#!/usr/bin/env python3
"""Read aircraft parameters by ICAO24, registration, or typecode.

CLI:
    python aircraft/query_aircraft_parameters.py 4951d9

Python:
    from aircraft.query_aircraft_parameters import get_aircraft_parameters

    aircraft = get_aircraft_parameters("4951d9")
    print(aircraft.typecode)
    print(aircraft.geometry.wing_area_m2)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PARAMETERS_PATH = SCRIPT_DIR / "openap_aircraft_parameters.json"
LOOKUP_PATH = SCRIPT_DIR / "aircraft_id_lookup.json"


@dataclass(frozen=True, slots=True)
class AircraftGeometry:
    wing_area_m2: float | None
    wing_span_m: float | None
    wing_mac_m: float | None
    wing_sweep_deg: float | None
    fuselage_length_m: float | None
    fuselage_width_m: float | None
    fuselage_height_m: float | None


@dataclass(frozen=True, slots=True)
class AircraftMass:
    mtow_kg: float | None
    oew_kg: float | None
    mlw_kg: float | None
    maximum_fuel_capacity_kg: float | None


@dataclass(frozen=True, slots=True)
class AircraftDrag:
    cd0: float | None
    k: float | None
    e: float | None
    landing_gear_drag_increment: float | None


@dataclass(frozen=True, slots=True)
class AircraftEngine:
    default: str | None
    type: str | None
    number: int | None
    max_thrust_n_each: float | None
    cruise_thrust_n_each: float | None
    cruise_sfc: float | None


@dataclass(frozen=True, slots=True)
class AircraftParameters:
    input_id: str
    resolved_by: str
    typecode: str
    name: str | None
    category: str | None
    geometry: AircraftGeometry
    mass: AircraftMass
    drag: AircraftDrag
    engine: AircraftEngine
    representative_metadata: dict[str, Any] | None

    def __str__(self) -> str:
        lines = [
            "AircraftParameters",
            f"  input_id     : {self.input_id}",
            f"  resolved_by  : {self.resolved_by}",
            f"  typecode     : {self.typecode}",
            f"  name         : {self.name or 'n/a'}",
            f"  category     : {self.category or 'n/a'}",
            "  geometry",
            f"    wing_area_m2 : {fmt(self.geometry.wing_area_m2)}",
            f"    wing_span_m  : {fmt(self.geometry.wing_span_m)}",
            f"    wing_mac_m   : {fmt(self.geometry.wing_mac_m)}",
            f"    sweep_deg    : {fmt(self.geometry.wing_sweep_deg)}",
            "  mass",
            f"    mtow_kg      : {fmt(self.mass.mtow_kg)}",
            f"    oew_kg       : {fmt(self.mass.oew_kg)}",
            f"    mlw_kg       : {fmt(self.mass.mlw_kg)}",
            f"    fuel_cap_kg  : {fmt(self.mass.maximum_fuel_capacity_kg)}",
            "  drag",
            f"    cd0          : {fmt(self.drag.cd0)}",
            f"    k            : {fmt(self.drag.k)}",
            f"    e            : {fmt(self.drag.e)}",
            "  engine",
            f"    default      : {self.engine.default or 'n/a'}",
            f"    number       : {fmt(self.engine.number)}",
            f"    thrust_each_N: {fmt(self.engine.max_thrust_n_each)}",
        ]
        return "\n".join(lines)


class AircraftLookupError(LookupError):
    """Raised when a requested aircraft id cannot be resolved to OpenAP data."""


def fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def normalize_id(value: str | None) -> str:
    return (value or "").strip().upper()


def load_json(path: Path) -> dict[str, Any]:
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


def get_aircraft_parameters(aircraft_id: str) -> AircraftParameters:
    parameters = load_json(PARAMETERS_PATH)
    lookup = load_json(LOOKUP_PATH)
    input_id = normalize_id(aircraft_id)
    typecode, resolved_by = resolve_typecode(input_id, parameters, lookup)

    typecode_record = parameters.get("typecodes", {}).get(typecode)
    if not typecode_record:
        raise AircraftLookupError(f"Typecode {typecode} is not present in {PARAMETERS_PATH.name}.")
    if not typecode_record.get("openap_supported"):
        reason = typecode_record.get("error", "not supported by OpenAP")
        raise AircraftLookupError(f"{input_id} resolves to {typecode}, but {reason}")

    data = typecode_record["parameters"]
    geometry = data.get("geometry", {})
    mass = data.get("mass", {})
    drag = data.get("drag", {})
    engine = data.get("engine", {})

    return AircraftParameters(
        input_id=input_id,
        resolved_by=resolved_by,
        typecode=typecode,
        name=data.get("aircraft_name"),
        category=data.get("category"),
        geometry=AircraftGeometry(
            wing_area_m2=geometry.get("wing_area_m2"),
            wing_span_m=geometry.get("wing_span_m"),
            wing_mac_m=geometry.get("wing_mac_m"),
            wing_sweep_deg=geometry.get("wing_sweep_deg"),
            fuselage_length_m=geometry.get("fuselage_length_m"),
            fuselage_width_m=geometry.get("fuselage_width_m"),
            fuselage_height_m=geometry.get("fuselage_height_m"),
        ),
        mass=AircraftMass(
            mtow_kg=mass.get("mtow_kg"),
            oew_kg=mass.get("oew_kg"),
            mlw_kg=mass.get("mlw_kg"),
            maximum_fuel_capacity_kg=mass.get("maximum_fuel_capacity_kg"),
        ),
        drag=AircraftDrag(
            cd0=drag.get("cd0"),
            k=drag.get("k"),
            e=drag.get("e"),
            landing_gear_drag_increment=drag.get("landing_gear_drag_increment"),
        ),
        engine=AircraftEngine(
            default=engine.get("default"),
            type=engine.get("type"),
            number=engine.get("number"),
            max_thrust_n_each=engine.get("max_thrust_n_each"),
            cruise_thrust_n_each=engine.get("cruise_thrust_n_each"),
            cruise_sfc=engine.get("cruise_sfc"),
        ),
        representative_metadata=typecode_record.get("representative_metadata"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print cached aircraft parameters for one ICAO24, registration, or typecode."
    )
    parser.add_argument("aircraft_id", help="Example: 4951d9, CS-TNY, or A320.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(get_aircraft_parameters(args.aircraft_id))
    except AircraftLookupError as exc:
        print(f"Aircraft lookup failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
