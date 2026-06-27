"""Assemble FlightScenarios from CZML-input flights + an aircraft identity.

Orchestration only — it wires the pieces together:

    CZML-input flight ──► initial_state_from_track ──► GeodeticState
    aircraft id       ──► aircraft_for_code        ──► AircraftSpec ──► aero_params_for_aircraft
                                                                            └──► AeroParams
    => FlightScenario(initial, aircraft, aero, source)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aircraft.aero_params import aero_params_for_aircraft

from .scenario import FlightScenario, aircraft_for_code
from .start_state import DEFAULT_WINDOW_S, final_state_from_track, initial_state_from_track


def build_scenario(
    flight: dict[str, Any],
    aircraft_id: str,
    *,
    mass_kg: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
) -> FlightScenario:
    """Build one :class:`FlightScenario` from a single CZML-input ``flight`` dict.

    ``flight`` is one element of a CZML-input file: ``{id, callsign, icao24, runway,
    waypoints: [[t, lon, lat, alt], ...], ...}``. ``aircraft_id`` selects the spec
    (CZML-input's own ``type`` is ``"UNK"``, so the type must be supplied here). ``mass_kg``
    defaults to the aircraft's spec mass.
    """
    aircraft = aircraft_for_code(aircraft_id)
    mass = mass_kg if mass_kg is not None else aircraft.mass_kg

    waypoints = flight["waypoints"]
    # The optimizer flies initial -> target. Both ends of the observed track give the
    # boundary states (so the optimizer reproduces the observed approach, and the result
    # can be compared against the real flight).
    initial = initial_state_from_track(waypoints, mass_kg=mass, window_s=window_s)
    target = final_state_from_track(waypoints, mass_kg=mass, window_s=window_s)
    aero = aero_params_for_aircraft(aircraft)

    source = {
        "id": flight.get("id"),
        "callsign": flight.get("callsign"),
        "icao24": flight.get("icao24"),
        "runway": flight.get("runway"),
        "landing_time_utc": flight.get("landing_time_utc"),
        "n_samples": len(waypoints),
        "window_s": window_s,
    }
    return FlightScenario(initial=initial, aircraft=aircraft, aero=aero, source=source, target=target)


def build_scenarios_from_czml_input(
    czml_input: str | Path | list[dict[str, Any]],
    aircraft_id: str,
    *,
    mass_kg: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[FlightScenario]:
    """Build a scenario per flight in a CZML-input file (or already-loaded list).

    ``czml_input`` may be a path to a ``*_czml_input_*.json`` / ``*_landings.json`` file,
    or the parsed list of flight dicts.
    """
    flights = _load_flights(czml_input)
    return [
        build_scenario(flight, aircraft_id, mass_kg=mass_kg, window_s=window_s)
        for flight in flights
    ]


def _load_flights(czml_input: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(czml_input, (str, Path)):
        return json.loads(Path(czml_input).read_text(encoding="utf-8"))
    return czml_input
