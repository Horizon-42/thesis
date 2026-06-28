"""End-to-end build wiring: CZML-input flight + aircraft resolution -> FlightScenario."""

import pytest

from aircraft.aircraft_sets import A320
from flight_scenarios.build import build_scenario, build_scenarios_from_czml_input

# A minimal CZML-input flight (one element of a *_czml_input_*.json / *_landings.json).
# icao24 3949ea is a real Air France transponder address -> resolves to B772 via OpenAP.
FLIGHT = {
    "id": "AFR074",
    "callsign": "AFR074",
    "type": "UNK",
    "icao24": "3949ea",
    "runway": "05L",
    "waypoints": [
        [0.0, -78.45, 35.74, 2500.0],
        [5.0, -78.46, 35.74, 2450.0],
        [10.0, -78.47, 35.74, 2400.0],
    ],
}


def test_build_scenario_resolves_aircraft_from_icao24():
    # No fallback supplied: the aircraft is resolved from the flight's own icao24.
    scen = build_scenario(FLIGHT)
    assert scen.aircraft.code == "B772"
    assert scen.aero.S == scen.aircraft.geometry.wing_area_m2
    assert scen.initial.m == scen.aircraft.mass.max_takeoff_kg
    assert scen.source["id"] == "AFR074"
    assert scen.source["runway"] == "05L"
    assert scen.source["n_samples"] == 3
    # both boundary states are populated (initial = track start, target = track end)
    assert scen.initial.longitude == FLIGHT["waypoints"][0][1]
    assert scen.target is not None
    assert scen.target.longitude == FLIGHT["waypoints"][-1][1]


def test_build_scenario_falls_back_to_aircraft_type_when_icao24_unresolvable():
    flight = {**FLIGHT, "icao24": None}  # no transponder address -> use the explicit fallback
    scen = build_scenario(flight, "A320")
    assert scen.aircraft is A320


def test_build_scenario_raises_when_unresolvable_and_no_fallback():
    flight = {**FLIGHT, "icao24": None}
    with pytest.raises(KeyError):
        build_scenario(flight)


def test_build_scenarios_from_list_resolves_each_flight():
    scens = build_scenarios_from_czml_input([FLIGHT, FLIGHT])
    assert len(scens) == 2
    assert all(s.aircraft.code == "B772" for s in scens)
