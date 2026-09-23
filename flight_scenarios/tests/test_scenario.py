"""FlightScenario record + serialization (plumbing — passes without the start_state TODO)."""

import pytest

from aerodynamic_model.common import GeodeticState
from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import A320
from aircraft.performance_index import performance_index_identity
from geokit import kt_to_ms
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
        source={"id": "AFR074", "icao24": "3949ea", "n_samples": 2,
                "performance_index_sha256": performance_index_identity()["sha256"]},
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


def _threshold_scenario(target_speed_ms: float) -> FlightScenario:
    scen = _sample_scenario()
    scen.target = GeodeticState(35.87, -78.80, 126.0, target_speed_ms, 0.9, -0.0524, 60000.0)
    scen.source = {**scen.source, "flight_key": "AFR074_05L", "target_source": "runway_threshold"}
    return scen


def test_a_runway_threshold_target_at_the_published_speed_round_trips():
    speed = A320.approach.reference_speed_ms(60000.0)
    restored = FlightScenario.from_dict(_threshold_scenario(speed).to_dict())
    assert restored.target.V == speed


def test_a_runway_threshold_target_from_before_the_published_speeds_is_refused():
    # Files prepared before 2026-09-23 pin every 5.7-150 t type at 145 kt.
    with pytest.raises(ValueError, match="another approach-speed rule or table"):
        FlightScenario.from_dict(_threshold_scenario(kt_to_ms(145.0)).to_dict())


def test_a_scenario_built_under_another_performance_index_is_refused():
    # The stored aircraft code is re-resolved through TODAY's index; a file built before it
    # (or under other decisions) could come back as another airframe under its stored aero.
    stale = _sample_scenario().to_dict()
    stale["source"]["performance_index_sha256"] = None
    with pytest.raises(ValueError, match="performance index"):
        FlightScenario.from_dict(stale)

