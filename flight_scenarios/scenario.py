"""The FlightScenario record + JSON (de)serialization.

This is plumbing: the neutral, serializable container that both the optimizer and a
data-driven model consume. It carries the domain types from the modeling plane
(``GeodeticState``, ``Aircraft``, ``AeroParams``) and round-trips through plain JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aerodynamic_model.common import GeodeticState
from aircraft.aero_params import AeroParams
from aircraft.aircraft_sets import AIRCRAFT_PRESETS, Aircraft
from aircraft.performance_index import (
    PERFORMANCE_INDEX_SCHEMA,
    index_entry,
    performance_index_identity,
)
from aircraft.query_aircraft_parameters import (
    AircraftLookupError,
    get_aircraft_parameters,
    openap_performance_metadata,
    openap_source_label,
)


AIRCRAFT_PROVIDERS = ("auto", "openap")

#: The mass and target speed of a scenario WITHOUT dynamics: unknown, never a stand-in. Such a
#: scenario exists only for trajectory-only consumers (ts state series, the instruction
#: labeller); anything that needs dynamics asks :meth:`FlightScenario.dynamics`, which refuses.
UNKNOWN_WITHOUT_DYNAMICS = math.nan


class NoAircraftDynamics(KeyError):
    """The flight's aircraft has no dynamics the model may fly: no ICAO type, a type the
    performance index excludes, or a type nothing models. Batch layers that need dynamics
    drop the flight and name ``typecode`` and ``reason``; nothing is flown in its place."""

    def __init__(self, typecode: str | None, reason: str, flight_id: Any = None) -> None:
        super().__init__(f"no aircraft dynamics for flight {flight_id!r} (type {typecode}): {reason}")
        self.typecode = typecode
        self.reason = reason

    def __str__(self) -> str:          # KeyError would quote the message
        return str(self.args[0])


def aircraft_for_code(aircraft_id: str, *, provider: str = "auto") -> Aircraft:
    """Resolve an aircraft code (e.g. ``"A320"``) to an :class:`Aircraft`.

    ``auto`` (the default): a hand-tuned preset wins; then the performance index
    (``aircraft/performance_index.json``) decides the types with no native model -- the
    type's own parameters, the airframe it is flown as (returned under THAT airframe's code,
    so its mass, approach speed and speed gate belong together), or excluded (``KeyError``
    naming why); every other type is flown by its own OpenAP model. ``openap`` uses OpenAP
    only, bypassing presets and the index, so a fleet-filtered experiment cannot claim OpenAP
    provenance while silently using other dynamics.
    """
    if provider not in AIRCRAFT_PROVIDERS:
        raise ValueError(f"unknown aircraft provider {provider!r}; expected {AIRCRAFT_PROVIDERS}")
    code = aircraft_id.strip().upper()
    if provider == "auto":
        preset = AIRCRAFT_PRESETS.get(code)
        if preset is not None:
            return preset
        entry = index_entry(code)
        if entry is not None:
            if entry.decision == "own":
                return entry.aircraft
            if entry.decision == "substitute":
                return aircraft_for_code(entry.substitute)
            raise KeyError(f"aircraft '{aircraft_id}' is excluded by the performance index: {entry.reason}")
    try:
        return get_aircraft_parameters(code)
    except AircraftLookupError as exc:
        where = "" if provider == "openap" else "not a preset, no performance-index row; "
        raise KeyError(f"no dynamics for aircraft '{aircraft_id}': {where}OpenAP: {exc}") from None


def _index_own(code: str) -> bool:
    entry = index_entry(code)
    return entry is not None and entry.decision == "own"


def aircraft_dynamics_source(aircraft_code: str, *, provider: str = "auto") -> str:
    """Return the provider label used for a resolved scenario aircraft (the FLOWN code)."""
    code = aircraft_code.strip().upper()
    if provider == "auto" and code in AIRCRAFT_PRESETS:
        return "aircraft_preset"
    if provider == "auto" and _index_own(code):
        return PERFORMANCE_INDEX_SCHEMA
    return openap_source_label()


def aircraft_provider_of(dynamics_source: str) -> str:
    """The provider that resolves a record's ``dynamics_typecode`` back to the aircraft it flew,
    from the record's ``dynamics_source`` (``aircraft_dynamics_source``'s label): ``openap`` for an
    OpenAP-flown record (an ``openap`` run flies the OpenAP A320, not the preset), else ``auto``."""
    return "openap" if dynamics_source == openap_source_label() else "auto"


def aircraft_dynamics_surrogate_typecode(
    aircraft_code: str, *, provider: str = "auto"
) -> str | None:
    """OpenAP's performance type for an OpenAP-flown aircraft (its own code, since synonyms are
    not flown directly), or ``None`` for a preset or an own-parameter index type."""
    code = aircraft_code.strip().upper()
    if provider == "auto" and (code in AIRCRAFT_PRESETS or _index_own(code)):
        return None
    return str(openap_performance_metadata(code)["performance_typecode"])


@dataclass
class FlightScenario:
    """One modeling input: an initial state + aircraft, derived from an observed flight.

    ``target`` is optional (e.g. a runway threshold or the track's final state); the
    optimizer fills it in if not provided here. ``source`` carries provenance metadata
    (flight id, callsign, icao24, runway, sample count, …).
    """

    initial: GeodeticState
    #: ``None`` (with ``aero``) for a scenario WITHOUT dynamics: its masses and target speed
    #: are then `UNKNOWN_WITHOUT_DYNAMICS`; built only on request
    #: (``build_scenario(..., require_dynamics=False)``) for trajectory-only consumers.
    aircraft: Aircraft | None
    aero: AeroParams | None
    source: dict[str, Any] = field(default_factory=dict)
    target: GeodeticState | None = None

    @property
    def has_dynamics(self) -> bool:
        return self.aircraft is not None

    def dynamics(self, purpose: str) -> tuple[Aircraft, AeroParams]:
        """``(aircraft, aero)`` for a consumer that needs them; a scenario without dynamics
        raises, naming ``purpose``, instead of being computed on unknown numbers."""
        if self.aircraft is None or self.aero is None:
            raise NoAircraftDynamics(
                self.source.get("resolved_typecode"),
                f"{purpose} needs aircraft dynamics; {self.source.get('no_dynamics_reason')}",
                self.source.get("flight_key") or self.source.get("id"),
            )
        return self.aircraft, self.aero

    def to_dict(self) -> dict[str, Any]:
        # The aircraft is stored by code (presets are the source of truth); aero params
        # are stored explicitly so a data-driven model sees them without a lookup.
        return {
            "initial": asdict(self.initial),
            "target": asdict(self.target) if self.target is not None else None,
            "aircraft_code": self.aircraft.code if self.aircraft is not None else None,
            "aero": asdict(self.aero) if self.aero is not None else None,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlightScenario":
        """Rebuild a saved scenario; refuse one whose runway-threshold target is stale.

        The aircraft is rebuilt from its code, so it carries TODAY's approach envelope, while
        the saved target carries the speed it was built with. A ``runway_threshold`` target
        must fly the airframe's published approach speed at the target mass
        (``Approach.reference_speed_ms``); a file prepared under another rule or another table
        (before 2026-09-24 every 5.7-150 t type targeted 145 kt) would pin the old speed, so it
        is refused by name instead of being flown mixed. Likewise the stored aircraft code is
        re-resolved through today's performance index, so a scenario must carry the index it
        was built with (``source["performance_index_sha256"]``); a file built under another
        index -- or before there was one -- could come back as another airframe under its
        stored aero, and is refused.
        """
        target = data.get("target")
        code = data["aircraft_code"]
        scenario = cls(
            initial=GeodeticState(**data["initial"]),
            target=GeodeticState(**target) if target is not None else None,
            aircraft=aircraft_for_code(code) if code is not None else None,
            aero=AeroParams(**data["aero"]) if code is not None else None,
            source=data.get("source", {}),
        )
        built_with = scenario.source.get("performance_index_sha256")
        if built_with != performance_index_identity()["sha256"]:
            raise ValueError(
                f"scenario {scenario.source.get('flight_key')!r} was built with performance index "
                f"{built_with!r}, not today's {performance_index_identity()['sha256']!r}; its aircraft "
                "would be re-resolved under different decisions — regenerate it with "
                "prepare_scenario_inputs.py"
            )
        if (scenario.has_dynamics and scenario.target is not None
                and scenario.source["target_source"] == "runway_threshold"):
            expected = scenario.aircraft.approach.reference_speed_ms(scenario.target.m)
            if not math.isclose(scenario.target.V, expected, rel_tol=1e-9):
                raise ValueError(
                    f"scenario {scenario.source.get('flight_key')!r}: its runway-threshold target "
                    f"flies {scenario.target.V:.3f} m/s, but {scenario.aircraft.code}'s published "
                    f"approach speed at {scenario.target.m:.0f} kg is {expected:.3f} m/s; the file "
                    "was prepared under another approach-speed rule or table — regenerate it "
                    "with prepare_scenario_inputs.py"
                )
        return scenario


def save_scenarios(scenarios: list[FlightScenario], path: str | Path) -> None:
    """Write a list of scenarios to one JSON file (a small dataset)."""
    payload = [scenario.to_dict() for scenario in scenarios]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scenarios(path: str | Path) -> list[FlightScenario]:
    """Read a scenario JSON file back into :class:`FlightScenario` objects."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FlightScenario.from_dict(item) for item in payload]
