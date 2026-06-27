"""Tests for the scenario-comparison CZML builder.

``_last_time`` passes already. The rest are the TODO targets (xfail until the two TODOs in
build_scenario_comparison_czml are implemented), since the builder is gated on them.
"""

import pytest

from build_scenario_comparison_czml import (
    OPTIMIZER_COLOR,
    REFERENCE_COLOR,
    SIMULATOR_COLOR,
    _last_time,
    _reference_entity_from_adsb,
    _states_to_waypoints,
    build_comparison_czml,
)

STATES = [
    {"t": 0.0, "lat": 35.74, "lon": -78.45, "alt": 2500.0, "V": 130.0, "psi": 1.0, "gamma": -0.05, "m": 78000.0},
    {"t": 5.0, "lat": 35.73, "lon": -78.46, "alt": 2400.0, "V": 128.0, "psi": 1.0, "gamma": -0.05, "m": 78000.0},
]
STATE_DATA = {
    "source": {"id": "AFR074"},
    "final_time_s": 5.0,
    "optimizer_states": STATES,
    "simulator_states": STATES,
}
ADSB_CZML = [
    {"id": "document", "clock": {}},
    {
        "id": "AFR074",
        "name": "AFR074",
        "position": {"cartographicDegrees": [0, -78.45, 35.74, 2500.0]},
        "path": {"material": {"solidColor": {"color": {"rgba": [255, 140, 0, 200]}}}},
    },
]


def test_last_time():
    assert _last_time(STATES) == 5.0
    assert _last_time([]) == 0.0


@pytest.mark.xfail(reason="implement TODO ① in _states_to_waypoints", strict=False)
def test_states_to_waypoints_order():
    waypoints = _states_to_waypoints(STATES)
    assert waypoints[0] == (0.0, -78.45, 35.74, 2500.0)   # (t, lon, lat, alt) — lon before lat
    assert waypoints[1] == (5.0, -78.46, 35.73, 2400.0)


@pytest.mark.xfail(reason="implement TODO ② in _reference_entity_from_adsb", strict=False)
def test_reference_entity_copies_and_recolors():
    entity = _reference_entity_from_adsb(ADSB_CZML, "AFR074", REFERENCE_COLOR)
    assert entity is not None
    assert entity["id"] == "scenario-reference"
    assert entity["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    # the source CZML must not be mutated
    assert ADSB_CZML[1]["path"]["material"]["solidColor"]["color"]["rgba"] == [255, 140, 0, 200]


@pytest.mark.xfail(reason="depends on TODO ① + ②", strict=False)
def test_build_comparison_czml_has_three_trajectories():
    czml = build_comparison_czml(STATE_DATA, ADSB_CZML)
    assert czml[0]["id"] == "document"
    ids = [packet["id"] for packet in czml[1:]]
    assert ids == ["scenario-reference", "scenario-optimizer", "scenario-simulator"]
    # colours wired through
    opt = next(p for p in czml if p["id"] == "scenario-optimizer")
    assert opt["path"]["material"]["solidColor"]["color"]["rgba"] == list(OPTIMIZER_COLOR)


@pytest.mark.xfail(reason="depends on TODO ② (reference lookup)", strict=False)
def test_no_reference_when_flight_missing():
    czml = build_comparison_czml({**STATE_DATA, "source": {"id": "NOPE"}}, ADSB_CZML)
    ids = [packet["id"] for packet in czml[1:]]
    assert "scenario-reference" not in ids
    assert "scenario-optimizer" in ids and "scenario-simulator" in ids
