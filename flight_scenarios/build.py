"""Assemble FlightScenarios from CZML-input flights + an aircraft identity.

Orchestration only — it wires the pieces together:

    CZML-input flight ──► initial_state_from_track ──► GeodeticState
    flight icao24     ──► aircraft_for_code (OpenAP) ──► Aircraft ──► aero_params_for_aircraft
      (else --aircraft-type fallback)                                       └──► AeroParams
    => FlightScenario(initial, aircraft, aero, source)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aircraft.aero_params import aero_params_for_aircraft

from .datum import flight_to_msl, flights_to_msl
from .runway_target import threshold_target_state
from .scenario import FlightScenario, aircraft_for_code
from .start_state import DEFAULT_WINDOW_S, final_state_from_track, initial_state_from_track


def build_scenario(
    flight: dict[str, Any],
    aircraft_type: str | None = None,
    *,
    airport: str | None = None,
    mass_kg: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    target_from_threshold: bool = False,
) -> FlightScenario:
    """Build one :class:`FlightScenario` from a single CZML-input ``flight`` dict.

    ``flight`` is one element of a CZML-input file: ``{id, callsign, icao24, arr_airport,
    runway, waypoints: [[t, lon, lat, alt], ...], ...}``. The aircraft is resolved from the
    flight's transponder address (``icao24``) via the OpenAP lookup; ``aircraft_type`` (e.g.
    ``"A320"``) is an explicit fallback, used only when the ``icao24`` can't be resolved.
    ``mass_kg`` defaults to the aircraft's landing mass (these are approach scenarios).

    ``target_from_threshold`` chooses the **target** state: when ``False`` (default) it is
    derived from the end of the observed track; when ``True`` it is the published runway
    threshold (a clean approach endpoint — at the threshold-crossing height, Vref, runway
    heading, glidepath), falling back to the track end if the threshold isn't in the config.
    ``airport`` supplies the arrival airport for the threshold lookup (the CZML-input flight's
    own ``arr_airport`` is often empty — the airport lives in the file path).

    The track's altitudes are converted from ellipsoidal (HAE) to MSL here rather than
    assumed, so a scenario cannot be built on the wrong vertical datum no matter which
    loader produced the dict — this bug reached three separate load paths. The conversion
    is idempotent (keyed on ``altitude_source``), so callers that already used
    :func:`load_observed_flights` pay only a tag check. See ``flight_scenarios/datum.py``.
    """
    flight = flight_to_msl(flight)
    aircraft = _resolve_aircraft(flight, aircraft_type)
    mass = mass_kg if mass_kg is not None else aircraft.landing_mass
    arr_airport = airport or flight.get("arr_airport")

    waypoints = flight["waypoints"]
    # The optimizer flies initial -> target. The initial state is the start of the observed
    # track; the target is either the track end or the runway threshold (see below).
    initial = initial_state_from_track(waypoints, mass_kg=mass, window_s=window_s)
    target_source = "track_end"
    target = None
    if target_from_threshold:
        target = threshold_target_state(arr_airport, flight.get("runway"), aircraft, mass_kg=mass)
        if target is not None:
            target_source = "runway_threshold"
    if target is None:
        target = final_state_from_track(waypoints, mass_kg=mass, window_s=window_s)
    aero = aero_params_for_aircraft(aircraft)

    source = {
        "id": flight.get("id"),
        "callsign": flight.get("callsign"),
        "icao24": flight.get("icao24"),
        "arr_airport": arr_airport,
        "runway": flight.get("runway"),
        "landing_time_utc": flight.get("landing_time_utc"),
        # Terminal-ring entry time (absolute UTC, from the arrival-segment cut) — the
        # co-temporal boundary condition multi-aircraft interaction studies place by.
        "entry_time_utc": flight.get("entry_time_utc"),
        "n_samples": len(waypoints),
        "window_s": window_s,
        "target_source": target_source,
        # Datum provenance: after flight_to_msl above this is the MSL tag, so a saved
        # scenarios file records which vertical datum it was built on — pre-datum-fix
        # (HAE-era) files carry no such key and are thereby distinguishable.
        "altitude_source": flight.get("altitude_source"),
    }
    return FlightScenario(initial=initial, aircraft=aircraft, aero=aero, source=source, target=target)


def build_scenarios_from_czml_input(
    czml_input: str | Path | list[dict[str, Any]],
    aircraft_type: str | None = None,
    *,
    airport: str | None = None,
    mass_kg: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    target_from_threshold: bool = False,
) -> list[FlightScenario]:
    """Build a scenario per flight in a CZML-input file (or already-loaded list).

    ``czml_input`` may be a path to a ``*_czml_input_*.json`` / ``*_landings.json`` file,
    or the parsed list of flight dicts. Each flight's aircraft is resolved from its own
    ``icao24`` (so a mixed-fleet file gets per-flight types); ``aircraft_type`` is the
    fallback when an ``icao24`` can't be resolved. ``airport`` and ``target_from_threshold``
    are forwarded to :func:`build_scenario`.
    """
    flights = load_observed_flights(czml_input)
    return [
        build_scenario(
            flight, aircraft_type, airport=airport, mass_kg=mass_kg, window_s=window_s,
            target_from_threshold=target_from_threshold,
        )
        for flight in flights
    ]


def _resolve_aircraft(flight: dict[str, Any], fallback_type: str | None):
    """Resolve a flight's :class:`Aircraft`, preferring its transponder address.

    Order: the flight's declared ``type`` (if real — CZML-input writes ``"UNK"``), then its
    ``icao24`` via the OpenAP lookup, then ``fallback_type`` (the CLI ``--aircraft-type``).
    Raises if none resolve — fail loudly rather than guess a type.
    """
    candidates: list[str] = []
    declared = (flight.get("type") or "").strip().upper()
    if declared and declared != "UNK":
        candidates.append(declared)
    icao24 = (flight.get("icao24") or "").strip()
    if icao24:
        candidates.append(icao24)
    if fallback_type:
        candidates.append(fallback_type)

    for candidate in candidates:
        try:
            return aircraft_for_code(candidate)
        except KeyError:
            continue
    raise KeyError(
        f"could not resolve an aircraft for flight {flight.get('id')!r}: tried "
        f"{candidates or ['<nothing>']}; pass --aircraft-type as a fallback"
    )


def load_observed_flights(czml_input: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """CZML-input flights, read INTO THE MODELING PLANE: altitudes converted HAE -> MSL.

    Every consumer that turns an observed track into modeling state must come through here
    -- both the scenario builder and ``write_reference_records`` -- because a scenario built
    on MSL and a reference record built on HAE would be 30 m apart while looking identical.
    See ``flight_scenarios/datum.py`` for why the conversion is here and not in the harvest.
    """
    flights = (
        json.loads(Path(czml_input).read_text(encoding="utf-8"))
        if isinstance(czml_input, (str, Path))
        else czml_input
    )
    return flights_to_msl(flights)
