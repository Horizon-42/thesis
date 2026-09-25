#!/usr/bin/env python3
"""Resolve an aircraft (by ICAO24, registration, or typecode) to an ``Aircraft``.

The OpenAP cache supplies geometry / mass / engine / drag for the types OpenAP models
DIRECTLY. OpenAP has no *approach* envelope: its speeds are the type's published approach speed
(``aircraft/reference_speeds.json``, FAA Aircraft Characteristics Database), and its procedure
geometry and thrust guess are MTOW-class defaults (``aircraft_sets.class_procedure``). Refused:
a type with no published approach speed (B3XM), and an OpenAP SYNONYM -- a type OpenAP only
covers with another type's data. Those are decided once, per type, in
``aircraft/performance_index.json`` (own parameters, an explicit substitute, or excluded).

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

from aircraft.aircraft_sets import Aircraft, Approach, Drag, Engine, Geometry, Mass, class_procedure
from aircraft.identity import OPENSKY_LOOKUP_PATH, OPENSKY_LOOKUP_SCHEMA
from aircraft.reference_speeds import reference_speed

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMETERS_PATH = SCRIPT_DIR / "openap_aircraft_parameters.json"
LOOKUP_PATH = OPENSKY_LOOKUP_PATH
# The OpenAP parameters cache's schema (``aircraft.build_openap_aircraft_database`` writes it).
OPENAP_PARAMETERS_SCHEMA = 1
# What each cache's schema_version must read; a file of another schema is refused, not read.
_CACHE_SCHEMAS = {PARAMETERS_PATH: OPENAP_PARAMETERS_SCHEMA, LOOKUP_PATH: OPENSKY_LOOKUP_SCHEMA}


class AircraftLookupError(LookupError):
    """Raised when a requested aircraft id cannot be resolved to OpenAP data."""


def fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def normalize_id(value: str | None) -> str:
    return (value or "").strip().upper()


@lru_cache(maxsize=None)
def load_json(path: Path) -> dict[str, Any]:
    """Load + cache one of the two OpenAP caches (large, read once per aircraft), refusing a file
    whose ``schema_version`` is not the one this reader knows."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = _CACHE_SCHEMAS[path]
    if payload.get("schema_version") != expected:
        raise ValueError(
            f"{path} has schema {payload.get('schema_version')!r}, not {expected}; rebuild it "
            "(python -m aircraft.build_openap_aircraft_database)"
        )
    return payload


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
    surrogate = normalize_id(data.get("openap_performance_typecode") or typecode)
    if surrogate != typecode:
        raise AircraftLookupError(
            f"{aircraft_id} resolves to {typecode}, which OpenAP covers only with {surrogate}'s data; "
            "the model does not fly that data under another type's code"
        )
    speeds = reference_speed(typecode)
    if speeds is None:
        raise AircraftLookupError(
            f"{aircraft_id} resolves to {typecode}, which has no published approach speed in "
            "aircraft/reference_speeds.json"
        )
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
            max_takeoff_kg=mass["mtow_kg"],
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
        approach=Approach(speeds=speeds, **class_procedure(mass["mtow_kg"])),
        drag=Drag(
            zero_lift_cd0=drag.get("cd0"),
            induced_drag_factor=drag.get("k"),
            oswald_efficiency=drag.get("e"),
            landing_gear_drag_increment=drag.get("landing_gear_drag_increment"),
        ),
    )


def openap_source_label() -> str:
    """Stable audit label for the cached OpenAP performance provider."""
    source = load_json(PARAMETERS_PATH).get("source", {})
    version = source.get("openap_version") or "unknown"
    return f"openap-{version}"


def openap_performance_metadata(typecode: str) -> dict[str, Any]:
    """Describe the direct or synonym performance model behind one ICAO typecode."""
    normalized = normalize_id(typecode)
    payload = load_json(PARAMETERS_PATH)
    record = payload.get("typecodes", {}).get(normalized, {})
    parameters = record.get("parameters", {})
    return {
        "source": openap_source_label(),
        "performance_typecode": parameters.get("openap_performance_typecode", normalized),
        "uses_synonym": parameters.get("openap_performance_typecode", normalized) != normalized,
    }


def openap_support_kind(typecode: str | None) -> str | None:
    """Return ``direct``, ``synonym``, or ``None`` for one ICAO designator.

    Identity resolution is deliberately kept outside this provider helper.  Callers first
    standardize a flight to an ICAO Doc 8643 typecode, then use this function to state the
    exact OpenAP performance policy they accept.  In particular, a strict experiment can
    exclude both unsupported aircraft and OpenAP synonym/surrogate models without
    maintaining a second hard-coded aircraft list.

    ``direct`` means :func:`get_aircraft_parameters` builds it: OpenAP models the type itself
    AND it has a published approach speed (B3XM, absent from the FAA table, returns ``None``).
    ``synonym`` means OpenAP covers it only with another type's data; the model does not fly
    that data directly (``aircraft/performance_index.json`` decides those types).
    """
    normalized = normalize_id(typecode)
    if not normalized:
        return None
    record = load_json(PARAMETERS_PATH).get("typecodes", {}).get(normalized, {})
    if not record.get("openap_supported"):
        return None
    performance_typecode = normalize_id(
        record.get("parameters", {}).get("openap_performance_typecode") or normalized
    )
    if performance_typecode != normalized:
        return "synonym"
    return "direct" if reference_speed(normalized) is not None else None


def openap_direct_typecodes() -> tuple[str, ...]:
    """Sorted ICAO typecodes backed by their own OpenAP performance model."""
    records = load_json(PARAMETERS_PATH).get("typecodes", {})
    return tuple(sorted(code for code in records if openap_support_kind(code) == "direct"))


def format_aircraft(aircraft: Aircraft) -> str:
    g, m, e = aircraft.geometry, aircraft.mass, aircraft.engine
    return "\n".join([
        f"Aircraft {aircraft.code} ({aircraft.name}, {aircraft.category})",
        f"  geometry  wing_area_m2={fmt(g.wing_area_m2)}  wing_span_m={fmt(g.wing_span_m)}",
        f"  mass      max_takeoff_kg={fmt(m.max_takeoff_kg)}  max_landing_kg={fmt(m.max_landing_kg)}",
        f"  engine    count={fmt(e.count)}  max_thrust_n_each={fmt(e.max_thrust_n_each)}  total_n={fmt(e.max_thrust_total_n)}",
        f"  approach  Vref_kt={fmt(aircraft.approach.speeds.approach_speed_kt)} at MALW "
        f"{fmt(aircraft.approach.speeds.malw_kg)} kg (published, reference_speeds.json)",
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
