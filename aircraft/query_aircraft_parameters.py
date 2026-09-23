#!/usr/bin/env python3
"""Resolve an aircraft (by ICAO24, registration, or typecode) to an ``Aircraft``.

The OpenAP cache supplies geometry / mass / engine / drag. OpenAP has no *approach*
envelope: its speeds are the type's published approach speed (``aircraft/reference_speeds.json``,
FAA Aircraft Characteristics Database), and its procedure geometry and thrust guess are
MTOW-class defaults. A type with OpenAP data but no published approach speed is refused.

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
from aircraft.reference_speeds import reference_speed

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMETERS_PATH = SCRIPT_DIR / "openap_aircraft_parameters.json"
LOOKUP_PATH = SCRIPT_DIR / "aircraft_id_lookup.json"


# OpenAP carries no approach envelope. Its SPEEDS are the type's published approach speed
# (``Approach.speeds``, the published row); the procedure geometry and the thrust initial guess below are
# MTOW-class defaults mirroring the hand-tuned presets in aircraft_sets. OpenAP's own
# ``category`` (transport_jet / business_or_general_aviation / unknown) can't discriminate —
# it lumps the A318 and the 777 into "transport_jet" — so weight is the right key.
_GENERAL_AVIATION_PROCEDURE = dict(
    final_segment_min_nm=2.0, final_segment_max_nm=5.0, protection_half_width_nm=0.5,
    glide_angle_deg=3.0, threshold_crossing_height_m=15.0, thrust_guess_n=800.0,
)
_NARROW_BODY_PROCEDURE = dict(
    final_segment_min_nm=5.0, final_segment_max_nm=10.0, protection_half_width_nm=0.8,
    glide_angle_deg=3.0, threshold_crossing_height_m=15.0, thrust_guess_n=40000.0,
)
_WIDE_BODY_PROCEDURE = dict(
    final_segment_min_nm=6.0, final_segment_max_nm=12.0, protection_half_width_nm=1.0,
    glide_angle_deg=3.0, threshold_crossing_height_m=15.0, thrust_guess_n=140000.0,
)

# MTOW class boundaries (kg): 5 700 = the light/large-aircraft regulatory split;
# 150 000 ≈ the narrow-body/wide-body split (A321 ~93 t … B767 ~186 t).
_LIGHT_MAX_TAKEOFF_KG = 5_700.0
_WIDE_BODY_MIN_TAKEOFF_KG = 150_000.0


def _class_procedure(max_takeoff_kg: float) -> dict[str, float]:
    """The approach procedure defaults (not the speeds) of a maximum take-off weight's class."""
    if max_takeoff_kg < _LIGHT_MAX_TAKEOFF_KG:
        return _GENERAL_AVIATION_PROCEDURE
    if max_takeoff_kg >= _WIDE_BODY_MIN_TAKEOFF_KG:
        return _WIDE_BODY_PROCEDURE
    return _NARROW_BODY_PROCEDURE


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


# OpenAP substitutes a SURROGATE performance model for types it lacks
# (``openap_performance_typecode`` in the cache). That is fine for what the
# surrogate exists for — engine/drag performance — but it must not misstate the
# AIRFRAME's identity facts. For C56X the surrogate is the much smaller Citation II
# (C550): MTOW 6,849 kg against the real Citation Excel/XLS's ~9,100 kg — a 25 %
# mass understatement that placed the speed gate's stall-anchored window ~13 kt
# low and mislabelled half the type's real crossings "too fast"
# (evaluation/docs/BASELINE_SPEED_GATE_RESULTS.md §5). The corrections below
# restore the certificated airframe facts (Cessna 560XL series: MTOW 20,000 lb =
# 9,072 kg, MLW 18,700 lb = 8,482 kg, wing area 369.7 sq ft = 34.35 m²); the
# surrogate keeps supplying performance, and
# ``aircraft_dynamics_surrogate_typecode`` still reports it for audit.
_AIRFRAME_IDENTITY_CORRECTIONS: dict[str, dict[str, dict[str, float]]] = {
    "C56X": {
        "geometry": {"wing_area_m2": 34.35},
        "mass": {"mtow_kg": 9072.0, "mlw_kg": 8482.0},
    },
}


def _mass_airframe(typecode: str, parameters: dict[str, Any]) -> str:
    """The typecode whose MASS a cached record carries, i.e. whose published approach speed applies.

    A direct model carries its own mass. A synonym record carries its surrogate's (LJ45 is
    cached with GLF6's 37.9 t), unless an airframe-identity correction restored the type's own
    mass (C56X). The speed must belong to the same airframe as the mass, or rescaling it by
    sqrt(m / MALW) produces nonsense (LJ45's 123 kt at GLF6's mass would read 257 kt).
    """
    performance = normalize_id(parameters.get("openap_performance_typecode") or typecode)
    if performance == typecode or "mass" in _AIRFRAME_IDENTITY_CORRECTIONS.get(typecode, {}):
        return typecode
    return performance


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
    speed_airframe = _mass_airframe(typecode, data)
    speeds = reference_speed(speed_airframe)
    if speeds is None:
        raise AircraftLookupError(
            f"{aircraft_id} resolves to {typecode}, whose airframe {speed_airframe} has no published "
            "approach speed in aircraft/reference_speeds.json"
        )
    geometry = dict(data.get("geometry", {}))
    mass = dict(data.get("mass", {}))
    drag = data.get("drag", {})
    engine = data.get("engine", {})
    category = data.get("category")
    correction = _AIRFRAME_IDENTITY_CORRECTIONS.get(typecode)
    if correction is not None:
        geometry.update(correction.get("geometry", {}))
        mass.update(correction.get("mass", {}))

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
        approach=Approach(speeds=speeds, **_class_procedure(mass["mtow_kg"])),
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

    "Supported" means the model can build it: a record whose airframe has no published
    approach speed (B3XM, absent from the FAA table) returns ``None`` like an unsupported one,
    because :func:`get_aircraft_parameters` refuses it.
    """
    normalized = normalize_id(typecode)
    if not normalized:
        return None
    record = load_json(PARAMETERS_PATH).get("typecodes", {}).get(normalized, {})
    if not record.get("openap_supported"):
        return None
    if reference_speed(_mass_airframe(normalized, record.get("parameters", {}))) is None:
        return None
    performance_typecode = normalize_id(
        record.get("parameters", {}).get("openap_performance_typecode") or normalized
    )
    return "direct" if performance_typecode == normalized else "synonym"


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
