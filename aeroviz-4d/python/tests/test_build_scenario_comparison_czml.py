"""Tests for the scenario-comparison CZML builder (single + per-runway batch)."""

import json

from build_scenario_comparison_czml import (
    FAILED_COLOR,
    OPTIMIZER_COLOR,
    REFERENCE_COLOR,
    SIMULATOR_COLOR,
    _last_time,
    _reference_entity_from_adsb,
    _states_to_waypoints,
    build_comparison_czml,
    build_runway_comparison,
    group_results_by_runway,
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


def test_states_to_waypoints_order():
    waypoints = _states_to_waypoints(STATES)
    assert waypoints[0] == (0.0, -78.45, 35.74, 2500.0)   # (t, lon, lat, alt) — lon before lat
    assert waypoints[1] == (5.0, -78.46, 35.73, 2400.0)


def test_reference_entity_copies_and_recolors():
    entity = _reference_entity_from_adsb(ADSB_CZML, "AFR074", REFERENCE_COLOR)
    assert entity is not None
    assert entity["id"] == "scenario-reference"
    assert entity["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    # the source CZML must not be mutated
    assert ADSB_CZML[1]["path"]["material"]["solidColor"]["color"]["rgba"] == [255, 140, 0, 200]


def test_build_comparison_czml_has_three_trajectories():
    czml = build_comparison_czml(STATE_DATA, ADSB_CZML)
    assert czml[0]["id"] == "document"
    ids = [packet["id"] for packet in czml[1:]]
    assert ids == ["scenario-reference", "scenario-optimizer", "scenario-simulator"]
    opt = next(p for p in czml if p["id"] == "scenario-optimizer")
    assert opt["path"]["material"]["solidColor"]["color"]["rgba"] == list(OPTIMIZER_COLOR)


def test_no_reference_when_flight_missing():
    czml = build_comparison_czml({**STATE_DATA, "source": {"id": "NOPE"}}, ADSB_CZML)
    ids = [packet["id"] for packet in czml[1:]]
    assert "scenario-reference" not in ids
    assert "scenario-optimizer" in ids and "scenario-simulator" in ids


# ── Batch: per-runway combined CZML + failed coloring ─────────────────────────

def test_group_results_by_runway_keys_by_airport_and_runway():
    summary = {"results": [
        {"id": "A", "arr_airport": "KRDU", "runway": "05L", "status": "solved"},
        {"id": "B", "arr_airport": "KRDU", "runway": "05L", "status": "failed"},
        {"id": "C", "arr_airport": "KRDU", "runway": "23R", "status": "solved"},
    ]}
    groups = group_results_by_runway(summary)
    assert set(groups) == {("KRDU", "05L"), ("KRDU", "23R")}
    assert len(groups[("KRDU", "05L")]) == 2


def test_build_runway_comparison_solved_three_paths_failed_red(tmp_path):
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"},
        {"id": "DAL1312", "runway": "05L", "status": "failed", "states_file": None},
    ]
    adsb = [
        ADSB_CZML[0],
        ADSB_CZML[1],
        {**ADSB_CZML[1], "id": "DAL1312", "name": "DAL1312"},
    ]
    czml = build_runway_comparison(results, tmp_path, adsb)
    ids = [p["id"] for p in czml[1:]]

    # solved flight -> three namespaced entities
    assert {"ref-AFR074", "opt-AFR074", "sim-AFR074"} <= set(ids)
    sim = next(p for p in czml if p["id"] == "sim-AFR074")
    assert sim["path"]["material"]["solidColor"]["color"]["rgba"] == list(SIMULATOR_COLOR)

    # failed flight -> reference only, dark red, no optimizer/simulator
    assert "ref-DAL1312" in ids
    assert "opt-DAL1312" not in ids and "sim-DAL1312" not in ids
    failed_ref = next(p for p in czml if p["id"] == "ref-DAL1312")
    assert failed_ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(FAILED_COLOR)
