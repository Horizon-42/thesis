#!/usr/bin/env python3
"""Build the local aircraft parameter cache once.

Run from the repository root:

    python aircraft/build_openap_aircraft_database.py

Inputs:
    data/AIRCRAFT/aircraftDatabase.csv

Outputs:
    aircraft/openap_aircraft_parameters.json
    aircraft/aircraft_id_lookup.json
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_DATABASE = REPO_ROOT / "data" / "AIRCRAFT" / "aircraftDatabase.csv"
PARAMETERS_OUTPUT = Path(__file__).resolve().parent / "openap_aircraft_parameters.json"
LOOKUP_OUTPUT = Path(__file__).resolve().parent / "aircraft_id_lookup.json"

METADATA_FIELDS = [
    "icao24",
    "registration",
    "manufacturericao",
    "manufacturername",
    "model",
    "typecode",
    "icaoaircrafttype",
    "operator",
    "operatoricao",
    "operatoriata",
    "owner",
    "registered",
    "reguntil",
    "status",
    "built",
    "engines",
    "categoryDescription",
]


def normalize_id(value: str | None) -> str:
    return (value or "").strip().upper()


def clean_value(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def metadata_from_row(row: Mapping[str, str]) -> dict[str, str | None]:
    return {field: clean_value(row.get(field)) for field in METADATA_FIELDS}


def iter_aircraft_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def scan_aircraft_database(csv_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    icao24_to_typecode: dict[str, str] = {}
    registration_to_typecode: dict[str, str] = {}
    typecode_counts: dict[str, int] = {}
    rows_with_typecode = 0

    for row in iter_aircraft_rows(csv_path):
        typecode = normalize_id(row.get("typecode"))
        if not typecode:
            continue

        rows_with_typecode += 1
        typecode_counts[typecode] = typecode_counts.get(typecode, 0) + 1
        representatives.setdefault(typecode, metadata_from_row(row))

        icao24 = normalize_id(row.get("icao24"))
        if icao24:
            icao24_to_typecode.setdefault(icao24, typecode)

        registration = normalize_id(row.get("registration"))
        if registration:
            registration_to_typecode.setdefault(registration, typecode)

    lookup = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "source": {
            "aircraft_database_csv": str(csv_path),
            "primary_adsb_identifier": "icao24",
            "note": "OpenSky/ADS-B state vectors normally use ICAO24; registrations are included for convenience.",
        },
        "counts": {
            "rows_with_typecode": rows_with_typecode,
            "unique_typecodes": len(representatives),
            "icao24_mappings": len(icao24_to_typecode),
            "registration_mappings": len(registration_to_typecode),
        },
        "icao24_to_typecode": icao24_to_typecode,
        "registration_to_typecode": registration_to_typecode,
        "typecode_counts": dict(sorted(typecode_counts.items())),
    }
    return representatives, lookup


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_openap_import_environment() -> None:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir())) / "aeroviz-openap-cache"
    matplotlib_cache = cache_root / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))


def load_openap_prop_module():
    prepare_openap_import_environment()
    try:
        from openap import prop
    except ImportError as exc:
        raise SystemExit(
            "OpenAP is not installed. Install it in the aviation conda environment with: pip install openap"
        ) from exc
    return prop


def package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_nested(data: Mapping[str, Any], *path: str) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def build_openap_typecode_record(typecode: str, prop_module) -> tuple[dict[str, Any] | None, str | None]:
    try:
        aircraft = prop_module.aircraft(typecode)
    except Exception as exc:
        return None, str(exc)

    default_engine = get_nested(aircraft, "engine", "default")
    engine = None
    engine_error = None
    if default_engine:
        try:
            engine = prop_module.engine(default_engine)
        except Exception as exc:
            engine_error = str(exc)

    record = {
        "typecode": typecode,
        "aircraft_name": aircraft.get("aircraft"),
        "category": classify_aircraft(typecode),
        "geometry": {
            "wing_area_m2": get_nested(aircraft, "wing", "area"),
            "wing_span_m": get_nested(aircraft, "wing", "span"),
            "wing_mac_m": get_nested(aircraft, "wing", "mac"),
            "wing_sweep_deg": get_nested(aircraft, "wing", "sweep"),
            "wing_thickness_to_chord": get_nested(aircraft, "wing", "t/c"),
            "fuselage_length_m": get_nested(aircraft, "fuselage", "length"),
            "fuselage_width_m": get_nested(aircraft, "fuselage", "width"),
            "fuselage_height_m": get_nested(aircraft, "fuselage", "height"),
        },
        "mass": {
            "mtow_kg": aircraft.get("mtow") or get_nested(aircraft, "limits", "MTOW"),
            "oew_kg": aircraft.get("oew") or get_nested(aircraft, "limits", "OEW"),
            "mlw_kg": aircraft.get("mlw") or get_nested(aircraft, "limits", "MLW"),
            "maximum_fuel_capacity_kg": aircraft.get("mfc") or get_nested(aircraft, "limits", "MFC"),
            "note": "OpenAP provides type-level mass limits, not actual per-flight mass.",
        },
        "drag": {
            "cd0": get_nested(aircraft, "drag", "cd0"),
            "k": get_nested(aircraft, "drag", "k"),
            "e": get_nested(aircraft, "drag", "e"),
            "landing_gear_drag_increment": get_nested(aircraft, "drag", "gears"),
        },
        "limits": {
            "ceiling_m": aircraft.get("ceiling") or get_nested(aircraft, "limits", "ceiling"),
            "vmo_kts": aircraft.get("vmo") or get_nested(aircraft, "limits", "VMO"),
            "mmo": aircraft.get("mmo") or get_nested(aircraft, "limits", "MMO"),
        },
        "cruise": {
            "height_m": get_nested(aircraft, "cruise", "height"),
            "mach": get_nested(aircraft, "cruise", "mach"),
        },
        "engine": {
            "default": default_engine,
            "type": get_nested(aircraft, "engine", "type"),
            "mount": get_nested(aircraft, "engine", "mount"),
            "number": get_nested(aircraft, "engine", "number"),
            "max_thrust_n_each": get_nested(engine or {}, "max_thrust"),
            "cruise_thrust_n_each": get_nested(engine or {}, "cruise_thrust"),
            "cruise_sfc": get_nested(engine or {}, "cruise_sfc"),
            "lookup_error": engine_error,
        },
        "flaps": aircraft.get("flaps"),
        "raw_openap": {
            "aircraft": aircraft,
            "default_engine": engine,
        },
    }
    return json_safe(record), None


def classify_aircraft(typecode: str) -> str:
    typecode = typecode.upper()
    if typecode.startswith(("A3", "B7", "B3")) or typecode in {"E170", "E190", "E195"}:
        return "transport_jet"
    if typecode.startswith(("C", "GLF")):
        return "business_or_general_aviation"
    return "unknown"


def build_parameter_database(csv_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prop_module = load_openap_prop_module()
    supported_typecodes = {normalize_id(typecode) for typecode in prop_module.available_aircraft()}
    representatives, lookup = scan_aircraft_database(csv_path)

    typecodes: dict[str, Any] = {}
    for typecode, metadata in sorted(representatives.items()):
        if typecode not in supported_typecodes:
            typecodes[typecode] = {
                "typecode": typecode,
                "openap_supported": False,
                "representative_metadata": metadata,
                "error": f"Typecode {typecode} is not supported by this OpenAP installation.",
            }
            continue

        openap_record, error = build_openap_typecode_record(typecode, prop_module)
        if error:
            typecodes[typecode] = {
                "typecode": typecode,
                "openap_supported": False,
                "representative_metadata": metadata,
                "error": error,
            }
            continue

        typecodes[typecode] = {
            "openap_supported": True,
            "representative_metadata": metadata,
            "parameters": openap_record,
        }

    parameters = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "source": {
            "aircraft_database_csv": str(csv_path),
            "openap_version": package_version("openap"),
            "openap_supported_typecode_count": len(supported_typecodes),
            "aircraft_database_unique_typecode_count": len(representatives),
        },
        "typecodes": typecodes,
    }
    return parameters, lookup


def write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    if not AIRCRAFT_DATABASE.exists():
        raise SystemExit(f"Missing aircraft metadata CSV: {AIRCRAFT_DATABASE}")

    parameters, lookup = build_parameter_database(AIRCRAFT_DATABASE)
    write_json(parameters, PARAMETERS_OUTPUT)
    write_json(lookup, LOOKUP_OUTPUT)

    supported_count = sum(1 for item in parameters["typecodes"].values() if item.get("openap_supported"))
    print(f"Wrote {PARAMETERS_OUTPUT}")
    print(f"Wrote {LOOKUP_OUTPUT}")
    print(f"Unique typecodes: {len(parameters['typecodes'])}")
    print(f"OpenAP-supported typecodes: {supported_count}")
    print(f"ICAO24 mappings: {lookup['counts']['icao24_mappings']}")
    print(f"Registration mappings: {lookup['counts']['registration_mappings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
