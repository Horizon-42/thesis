"""FlightScenario record + serialization (plumbing — passes without the start_state TODO)."""

import pytest

from aerodynamic_model.common import GeodeticState
from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import A320
from flight_scenarios.scenario import (
    FlightScenario,
    aircraft_for_code,
    load_scenarios,
    save_scenarios,
)


def test_aircraft_for_code_is_case_insensitive():
    assert aircraft_for_code("a320") is A320
    assert aircraft_for_code("A320") is A320


def test_aircraft_for_code_unknown_raises():
    with pytest.raises(KeyError):
        aircraft_for_code("ZZZZ")


def _sample_scenario() -> FlightScenario:
    initial = GeodeticState(35.6, -78.5, 1500.0, 80.0, 1.2, -0.05, 78000.0)
    return FlightScenario(
        initial=initial,
        aircraft=A320,
        aero=aero_params_for_aircraft(A320),
        source={"id": "AFR074", "icao24": "3949ea", "n_samples": 2},
    )


def test_to_from_dict_round_trip():
    scen = _sample_scenario()
    restored = FlightScenario.from_dict(scen.to_dict())
    assert restored.initial == scen.initial
    assert restored.aircraft is A320
    assert restored.aero == scen.aero
    assert restored.source["id"] == "AFR074"
    assert restored.target is None


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "scenarios.json"
    save_scenarios([_sample_scenario()], path)
    loaded = load_scenarios(path)
    assert len(loaded) == 1
    assert loaded[0].initial == _sample_scenario().initial
    assert loaded[0].aircraft is A320
