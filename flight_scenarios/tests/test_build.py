"""End-to-end build wiring: CZML-input flight + aircraft id -> FlightScenario."""

from aircraft.aircraft_sets import A320
from flight_scenarios.build import build_scenario, build_scenarios_from_czml_input

# A minimal CZML-input flight (one element of a *_czml_input_*.json / *_landings.json).
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


def test_build_scenario_wires_aircraft_source_and_endpoints():
    scen = build_scenario(FLIGHT, "A320")
    assert scen.aircraft is A320
    assert scen.aero.S == A320.wing_area_m2
    assert scen.initial.m == A320.mass_kg
    assert scen.source["id"] == "AFR074"
    assert scen.source["runway"] == "05L"
    assert scen.source["n_samples"] == 3
    # both boundary states are populated (initial = track start, target = track end)
    assert scen.initial.longitude == FLIGHT["waypoints"][0][1]
    assert scen.target is not None
    assert scen.target.longitude == FLIGHT["waypoints"][-1][1]


def test_build_scenarios_from_list():
    scens = build_scenarios_from_czml_input([FLIGHT, FLIGHT], "C172")
    assert len(scens) == 2
    assert all(s.aircraft.code == "C172" for s in scens)
