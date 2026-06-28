"""The FlightScenario record + JSON (de)serialization.

This is plumbing: the neutral, serializable container that both the optimizer and a
data-driven model consume. It carries the domain types from the modeling plane
(``GeodeticState``, ``Aircraft``, ``AeroParams``) and round-trips through plain JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aerodynamic_model.common import GeodeticState
from aircraft.aero_params import AeroParams
from aircraft.aircraft_sets import AIRCRAFT_PRESETS, Aircraft
from aircraft.query_aircraft_parameters import AircraftLookupError, get_aircraft_parameters


def aircraft_for_code(aircraft_id: str) -> Aircraft:
    """Resolve an aircraft code (e.g. ``"A320"``) to an :class:`Aircraft`.

    Hand-tuned presets win (they carry a calibrated approach envelope); otherwise fall
    back to the OpenAP cache (geometry/mass/engine/drag + a category-default approach).
    """
    code = aircraft_id.strip().upper()
    preset = AIRCRAFT_PRESETS.get(code)
    if preset is not None:
        return preset
    try:
        return get_aircraft_parameters(code)
    except AircraftLookupError as exc:
        available = ", ".join(sorted(AIRCRAFT_PRESETS))
        raise KeyError(
            f"unknown aircraft '{aircraft_id}'; known presets: {available}; OpenAP: {exc}"
        ) from None


@dataclass
class FlightScenario:
    """One modeling input: an initial state + aircraft, derived from an observed flight.

    ``target`` is optional (e.g. a runway threshold or the track's final state); the
    optimizer fills it in if not provided here. ``source`` carries provenance metadata
    (flight id, callsign, icao24, runway, sample count, …).
    """

    initial: GeodeticState
    aircraft: Aircraft
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
