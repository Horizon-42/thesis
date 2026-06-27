"""The FlightScenario record + JSON (de)serialization.

This is plumbing: the neutral, serializable container that both the optimizer and a
data-driven model consume. It carries the domain types from the modeling plane
(``GeodeticState``, ``AircraftSpec``, ``AeroParams``) and round-trips through plain JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aerodynamic_model.common import GeodeticState
from aircraft.aero_params import AeroParams
from aircraft.aircraft_sets import AIRCRAFT_PRESETS, AircraftSpec
from aircraft.query_aircraft_parameters import AircraftGeometry, AircraftMass, AircraftDrag, AircraftEngine, AircraftParameters, get_aircraft_parameters

def aircraft_for_code(aircraft_id: str) -> AircraftSpec:
    """Resolve an aircraft code (e.g. ``"A320"``) to its :class:`AircraftSpec` preset.

    The single place that maps an identifier to a spec. Extend here (e.g. an
    OpenAP-derived spec via ``aircraft/query_aircraft_parameters.py``) when types beyond
    the presets are needed; callers and serialization stay unchanged.
    """
    code = aircraft_id.strip().upper()

    aircraft_params: AircraftParameters = get_aircraft_parameters(code)
    
    if aircraft_params is None:
        raise ValueError(f"unknown aircraft code: {code}")
    
    # Extract the relevant parameters from the AircraftParameters object
    geometry: AircraftGeometry = aircraft_params.geometry
    mass: AircraftMass = aircraft_params.mass
    drag: AircraftDrag = aircraft_params.drag
    engine: AircraftEngine = aircraft_params.engine

    return AircraftSpec(
        code=code,
        name=aircraft_params.name,
        category=aircraft_params.category,
        wing_area_m2=geometry.wing_area_m2,
        mass_kg=mass.max_takeoff_mass_kg,
        max_thrust_n=engine.max_thrust_n,
        approach_thrust_guess_n=engine.approach_thrust_guess_n,
        terminal_speed_kt=drag.terminal_speed_kt,
        terminal_speed_min_kt=drag.terminal_speed_min_kt,
        terminal_speed_max_kt=drag.terminal_speed_max_kt,
        final_approach_min_nm=drag.final_approach_min_nm,
        final_approach_max_nm=drag.final_approach_max_nm,
        final_approach_lateral_half_width_nm=drag.final_approach_lateral_half_width_nm,
        final_approach_glide_angle_deg=drag.final_approach_glide_angle_deg,
        threshold_crossing_height_m=drag.threshold_crossing_height_m,   
    )


@dataclass
class FlightScenario:
    """One modeling input: an initial state + aircraft, derived from an observed flight.

    ``target`` is optional (e.g. a runway threshold or the track's final state); the
    optimizer fills it in if not provided here. ``source`` carries provenance metadata
    (flight id, callsign, icao24, runway, sample count, …).
    """

    initial: GeodeticState
    aircraft: AircraftSpec
    aero: AeroParams
    source: dict[str, Any] = field(default_factory=dict)
    target: GeodeticState | None = None

    def to_dict(self) -> dict[str, Any]:
        # The aircraft is stored by code (presets are the source of truth); aero params
        # are stored explicitly so a data-driven model sees them without a lookup.
        return {
            "initial": asdict(self.initial),
            "target": asdict(self.target) if self.target is not None else None,
            "aircraft_code": self.aircraft.code,
            "aero": asdict(self.aero),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlightScenario":
        target = data.get("target")
        return cls(
            initial=GeodeticState(**data["initial"]),
            target=GeodeticState(**target) if target is not None else None,
            aircraft=aircraft_for_code(data["aircraft_code"]),
            aero=AeroParams(**data["aero"]),
            source=data.get("source", {}),
        )


def save_scenarios(scenarios: list[FlightScenario], path: str | Path) -> None:
    """Write a list of scenarios to one JSON file (a small dataset)."""
    payload = [scenario.to_dict() for scenario in scenarios]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scenarios(path: str | Path) -> list[FlightScenario]:
    """Read a scenario JSON file back into :class:`FlightScenario` objects."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FlightScenario.from_dict(item) for item in payload]
