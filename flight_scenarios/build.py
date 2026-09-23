"""Assemble FlightScenarios from CZML-input flights + an aircraft identity.

Orchestration only — it wires the pieces together:

    CZML-input flight ──► initial_state_from_track ──► GeodeticState
    declared type / ICAO24 ──► ICAO Doc 8643 identity ─┐
                                                      ├─► preset / performance index / OpenAP
      (no dynamics: NoAircraftDynamics, or an explicit ┘      Aircraft ─► AeroParams
       caller fallback -- only the ts `all` filter passes one)
    => FlightScenario(initial, aircraft, aero, source)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import Aircraft
from aircraft.identity import AircraftIdentity, get_default_identity_resolver
from aircraft.performance_index import index_entry, performance_index_identity

from .datum import flight_to_msl, flights_to_msl
from .fitted_approach import UnusableFittedApproach, fit_flight_final_approach
from .identity import flight_key
from .runway_target import threshold_target_state
from .scenario import (
    FlightScenario,
    aircraft_dynamics_source,
    aircraft_dynamics_surrogate_typecode,
    aircraft_for_code,
)
from .start_state import DEFAULT_WINDOW_S, final_state_from_track, initial_state_from_track


def build_scenario(
    flight: dict[str, Any],
    aircraft_type: str | None = None,
    *,
    airport: str | None = None,
    mass_kg: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    target_from_threshold: bool = False,
    target_from_fitted_adsb: bool = False,
    aircraft_provider: str = "auto",
) -> FlightScenario:
    """Build one :class:`FlightScenario` from a single CZML-input ``flight`` dict.

    ``flight`` is one element of a CZML-input file: ``{id, callsign, icao24, arr_airport,
    runway, waypoints: [[t, lon, lat, alt], ...], ...}``. Identity is resolved separately
    from performance: declared designators and registry results are validated against ICAO
    Doc 8643, then ``scenario.aircraft_for_code`` picks the dynamics (a preset, the performance
    index's decision, or the type's own OpenAP model). Without dynamics the build raises
    :class:`NoAircraftDynamics`, unless ``aircraft_type`` names an explicit fallback to fly
    instead; it never overwrites the audited real identity. ``mass_kg`` defaults to the selected
    dynamics model's landing mass (these are approach scenarios).

    ``target_from_threshold`` chooses the published runway endpoint.  The mutually exclusive
    ``target_from_fitted_adsb`` chooses the OLS-extrapolated flown threshold crossing:
    fitted position and fitted approach kinematics.  With neither flag the target remains
    the last observed sample for generic callers.
    ``airport`` supplies the arrival airport for the threshold lookup (the CZML-input flight's
    own ``arr_airport`` is often empty — the airport lives in the file path).

    The track's altitudes are converted from ellipsoidal (HAE) to MSL here rather than
    assumed, so a scenario cannot be built on the wrong vertical datum no matter which
    loader produced the dict — this bug reached three separate load paths. The conversion
    is idempotent (keyed on ``altitude_source``), so callers that already used
    :func:`load_model_arrivals` pay only a tag check. See ``flight_scenarios/datum.py``.
    """
    if target_from_threshold and target_from_fitted_adsb:
        raise ValueError("threshold and fitted ADS-B targets are mutually exclusive")
    raw_flight = flight
    fitted_hae = (
        fit_flight_final_approach(raw_flight)
        if target_from_fitted_adsb else None
    )
    flight = flight_to_msl(flight)
    aircraft_selection = _resolve_aircraft(
        flight, aircraft_type, aircraft_provider=aircraft_provider
    )
    aircraft = aircraft_selection.aircraft
    mass = mass_kg if mass_kg is not None else aircraft.landing_mass
    arr_airport = airport or flight.get("arr_airport")

    waypoints = flight["waypoints"]
    # The optimizer flies initial -> target. The initial state is the start of the observed
    # track; the target is either the track end or the runway threshold (see below).
    initial = initial_state_from_track(waypoints, mass_kg=mass, window_s=window_s)
    target_source = "track_end"
    target = None
    if target_from_threshold:
        target = threshold_target_state(
            arr_airport,
            flight.get("runway"),
            aircraft,
            mass_kg=mass,
            published_target=flight.get("runway_target"),
        )
        if target is not None:
            target_source = "runway_threshold"
    elif target_from_fitted_adsb:
        fitted = fitted_hae
        if fitted is None:
            raise UnusableFittedApproach(
                f"flight {flight.get('id')!r} has no usable fitted final approach for "
                f"runway {flight.get('runway')!r}"
            )
        target = fitted.target_state(
            mass_kg=mass,
            hae_minus_msl_m=float(
                (flight.get("runway_target") or {})["hae_minus_msl_m"]
            ),
        )
        target_source = "fitted_adsb_crossing"
    if target is None:
        target = final_state_from_track(waypoints, mass_kg=mass, window_s=window_s)
    aero = aero_params_for_aircraft(aircraft)

    source = {
        "id": flight.get("id"),
        # The project's actual flight identity (``id`` is only the callsign). Carried on
        # the scenario so every downstream record — including the optimizer's evaluation
        # rows, which reported ``flight_key: null`` while the observed rows carried it —
        # names WHICH flight it is without having to re-derive the key from a filename.
        # Only when the flight HAS an id: ``flight_key``'s fallback is the caller's list
        # index, which this function does not have, and a key built on the wrong index
        # would disagree with the record filename ``_scenario_filename`` derives.
        "flight_key": flight_key(flight, 0) if flight.get("id") else None,
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
        "threshold_crossing_height_m": (
            (flight.get("runway_target") or {}).get("threshold_crossing_height_m")
        ),
        "published_glidepath_deg": (
            (flight.get("runway_target") or {}).get("published_glidepath_deg")
        ),
        # Datum provenance: after flight_to_msl above this is the MSL tag, so a saved
        # scenarios file records which vertical datum it was built on — pre-datum-fix
        # (HAE-era) files carry no such key and are thereby distinguishable.
        "altitude_source": flight.get("altitude_source"),
        "hae_minus_msl_m": (flight.get("runway_target") or {}).get("hae_minus_msl_m"),
        "vertical_source": (flight.get("runway_target") or {}).get("vertical_source"),
        # The stall-model facts evaluation's threshold speed gate anchors on (same
        # AeroParams the optimizer and replay fly with, so the gate judges the record
        # against the model that produced it). Producer-supplied like hae_minus_msl_m:
        # a computed record without this block grades speed-indeterminate, loudly.
        "landing_aero": {
            "wing_area_m2": aero.S,
            "cl_max_landing": aero.Cl_max,
        },
        **aircraft_selection.audit_fields(),
    }
    return FlightScenario(initial=initial, aircraft=aircraft, aero=aero, source=source, target=target)


def build_scenarios_from_arrivals(
    arrivals: str | Path | list[dict[str, Any]],
    aircraft_type: str | None = None,
    *,
    airport: str | None = None,
    mass_kg: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    target_from_threshold: bool = False,
    target_from_fitted_adsb: bool = False,
    aircraft_provider: str = "auto",
) -> list[FlightScenario]:
    """Build a scenario per flight in an arrival manifest (or already-loaded list).

    ``arrivals`` may be an airport harvest root, ``arrivals/manifest.json``, or an
    already-loaded list. Each flight's identity is resolved from its own declared type or
    ``icao24`` (so a mixed-fleet file gets per-flight types); ``aircraft_type`` is only the
    dynamics fallback. ``airport`` and ``target_from_threshold`` are forwarded to
    :func:`build_scenario`.
    """
    flights = load_model_arrivals(arrivals)
    return [
        build_scenario(
            flight, aircraft_type, airport=airport, mass_kg=mass_kg, window_s=window_s,
            target_from_threshold=target_from_threshold,
            target_from_fitted_adsb=target_from_fitted_adsb,
            aircraft_provider=aircraft_provider,
        )
        for flight in flights
    ]


class NoAircraftDynamics(KeyError):
    """The flight's aircraft has no dynamics the model may fly: no ICAO type, a type the
    performance index excludes, or a type nothing models. Batch layers drop the flight and
    name ``typecode`` and ``reason``; nothing is flown in its place."""

    def __init__(self, typecode: str | None, reason: str, flight_id: Any = None) -> None:
        super().__init__(f"no aircraft dynamics for flight {flight_id!r} (type {typecode}): {reason}")
        self.typecode = typecode
        self.reason = reason

    def __str__(self) -> str:          # KeyError would quote the message
        return str(self.args[0])


@dataclass(frozen=True, slots=True)
class _AircraftSelection:
    aircraft: Aircraft
    identity: AircraftIdentity
    fallback_used: bool
    fallback_reason: str | None
    provider: str

    def audit_fields(self) -> dict[str, Any]:
        identity = self.identity
        return {
            "resolved_typecode": identity.typecode,
            "identity_source": identity.identity_source,
            "identity_source_date": identity.identity_source_date,
            "typecode_source": identity.typecode_source,
            "typecode_standard": identity.typecode_standard,
            "typecode_standard_date": identity.typecode_standard_date,
            "typecode_method": identity.typecode_method,
            "typecode_confidence": identity.confidence,
            "registry_registration": identity.registration,
            "registry_manufacturer": identity.manufacturer,
            "registry_model": identity.model,
            "faa_model_code": identity.faa_model_code,
            "dynamics_typecode": self.aircraft.code,
            "dynamics_source": aircraft_dynamics_source(
                self.aircraft.code, provider=self.provider
            ),
            "dynamics_surrogate_typecode": aircraft_dynamics_surrogate_typecode(
                self.aircraft.code, provider=self.provider
            ),
            # What the performance index decided for the identity's type (own / substitute /
            # exclude), or None when the type is flown natively or the provider bypasses it.
            "performance_index_decision": (
                entry.decision
                if self.provider == "auto" and identity.typecode is not None
                and (entry := index_entry(identity.typecode)) is not None
                else None
            ),
            "performance_index_sha256": performance_index_identity()["sha256"],
            "aircraft_fallback_used": self.fallback_used,
            "aircraft_fallback_reason": self.fallback_reason,
        }


def resolve_airframe(
    icao24: str | None,
    *,
    declared_type: str | None = None,
    aircraft_provider: str = "auto",
) -> tuple[float | None, str] | None:
    """``(landing_mass_kg or None, typecode)`` for one airframe, or None.

    For the harvest's observed records. The IDENTITY (icao24 → ICAO type, the same
    resolver ``build_scenario`` uses) is what the evaluation speed gate looks its
    PUBLISHED approach-speed window up by (``evaluation.speed_gate.TYPECODE_KEYS``),
    so it is returned whenever it resolves. The mass is the type's OWN landing mass when
    the model flies the type as itself (a preset, an OpenAP-direct type, an own-parameter
    row of the performance index), else None -- a substitute airframe's mass is not the
    observed type's, and an observed record's mass is an assumption the states carry, not
    something the gate reads, so a type without its own dynamics must not lose its type
    over it. No fallback type on purpose: an
    unresolvable identity returns None and the caller grades speed indeterminate,
    loudly, instead of judging a bizjet against an A320 window.
    """
    identity = get_default_identity_resolver().resolve(
        declared_type=declared_type, icao24=icao24
    )
    if identity.typecode is None:
        return None
    try:
        aircraft = aircraft_for_code(identity.typecode, provider=aircraft_provider)
    except KeyError:
        return None, identity.typecode
    if aircraft.code != identity.typecode:
        return None, identity.typecode
    return aircraft.landing_mass, identity.typecode


def _resolve_aircraft(
    flight: dict[str, Any],
    fallback_type: str | None,
    *,
    aircraft_provider: str = "auto",
) -> _AircraftSelection:
    """Resolve identity first, then obtain OpenAP/preset dynamics independently."""
    resolver = get_default_identity_resolver()
    identity = resolver.resolve(
        declared_type=flight.get("type"),
        icao24=flight.get("icao24"),
    )
    failure = identity.failure_reason
    if identity.typecode is not None:
        try:
            aircraft = aircraft_for_code(identity.typecode, provider=aircraft_provider)
        except KeyError as exc:
            failure = exc.args[0]
        else:
            return _AircraftSelection(
                aircraft=aircraft,
                identity=identity,
                fallback_used=False,
                fallback_reason=None,
                provider=aircraft_provider,
            )

    if not fallback_type:
        raise NoAircraftDynamics(
            identity.typecode, failure or "identity has no ICAO typecode", flight.get("id")
        )

    fallback_code = resolver.catalog.normalize_typecode(fallback_type)
    try:
        aircraft = aircraft_for_code(fallback_code, provider=aircraft_provider)
    except KeyError as exc:
        raise KeyError(
            f"fallback aircraft {fallback_code!r} has no usable dynamics: {exc}"
        ) from None
    return _AircraftSelection(
        aircraft=aircraft,
        identity=identity,
        fallback_used=True,
        fallback_reason=failure or "identity has no ICAO typecode",
        provider=aircraft_provider,
    )


def load_model_arrivals(arrivals: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Manifest-rostered arrivals entering the modeling plane: HAE converted to MSL.

    Every consumer that turns an observed track into modeling state must come through here
    -- both the scenario builder and ``write_reference_records`` -- because a scenario built
    on MSL and a reference record built on HAE would be 30 m apart while looking identical.
    See ``flight_scenarios/datum.py`` for why the conversion is here and not in the harvest.
    """
    if isinstance(arrivals, (str, Path)):
        # Local import keeps the data-plane package from becoming part of
        # flight_scenarios' import graph. ``harvest.airports`` imports only the datum
        # conversion below; an eager arrival import here would loop back into the
        # partially initialized airport module.
        from trajectory_data_process.harvest.arrivals import load_arrival_flights

        flights = load_arrival_flights(arrivals)
    else:
        flights = arrivals
    return flights_to_msl(flights)
