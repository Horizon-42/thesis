"""Tests for the scenario-comparison CZML builder (single + per-runway batch)."""

import json
import re

import pytest
from pathlib import Path

import build_scenario_comparison_czml as comparison_builder
from build_scenario_comparison_czml import (
    FAILED_COLOR,
    LOOKBACK_COLOR,
    PREDICTION_COLOR,
    OFF_TARGET_COLOR,
    OFF_TARGET_REF_COLOR,
    OPTIMIZER_COLOR,
    REFERENCE_COLOR,
    SIMULATOR_COLOR,
    _TRAIL_TIME_S,
    _last_time,
    _reference_entity_from_adsb,
    _states_to_waypoints,
    _upsert_category,
    build_comparison_czml,
    build_runway_comparison,
    evaluation_batch_stats,
    group_results_by_runway,
    load_verdicts,
    optimization_stats,
    prediction_accuracy_stats,
    publish_comparison_batch,
    prune_unreferenced_outputs,
    states_schema,
)

STATES = [
    {"t": 0.0, "lat": 35.74, "lon": -78.45, "alt": 2500.0, "V": 130.0, "psi": 1.0, "gamma": -0.05, "m": 78000.0},
    {"t": 5.0, "lat": 35.73, "lon": -78.46, "alt": 2400.0, "V": 128.0, "psi": 1.0, "gamma": -0.05, "m": 78000.0},
]
STATE_DATA = {
    "source": {"id": "AFR074", "runway": "05L", "hae_minus_msl_m": -33.5},
    "final_time_s": 5.0,
    "optimizer_states": STATES,
    "simulator_states": STATES,
}
# The observed layer's entity id IS the flight identity (flight_key of the same source
# fields the record stems carry) — the bare callsign lives only in ``name``.
ADSB_CZML = [
    {"id": "document", "clock": {}},
    {
        "id": "AFR074_05L",
        "name": "AFR074",
        "position": {"cartographicDegrees": [0, -78.45, 35.74, 2500.0]},
        # A short trailTime like the real trajectories.czml — the reference builder must OVERRIDE it.
        "path": {"leadTime": 0, "trailTime": 300, "material": {"solidColor": {"color": {"rgba": [255, 140, 0, 200]}}}},
    },
]


def _offsets(entity):
    """The time offsets (seconds from the document epoch) of a built entity's position samples."""
    cd = entity["position"]["cartographicDegrees"]
    return cd[0::4]


def _sample_at(entity, index):
    """One (lon, lat, alt) sample of a built entity's position, by sample index."""
    cd = entity["position"]["cartographicDegrees"]
    start = (index % (len(cd) // 4)) * 4
    return cd[start + 1 : start + 4]


def test_last_time():
    assert _last_time(STATES) == 5.0
    assert _last_time([]) == 0.0


def test_states_to_waypoints_order():
    waypoints = _states_to_waypoints(STATES, -33.5)

    assert waypoints[0][:3] == (0.0, -78.45, 35.74)
    assert waypoints[1][:3] == (5.0, -78.46, 35.73)
    assert waypoints[0][3] == 2466.5
    assert waypoints[1][3] == 2366.5
    # Sanity, non-circularly: over the eastern US the geoid sits 25-40 m below the
    # ellipsoid, so the CZML altitude must come out that far BELOW the record's MSL.
    assert 2500.0 - 40.0 < waypoints[0][3] < 2500.0 - 25.0


def test_states_to_waypoints_empty_is_empty():
    assert _states_to_waypoints([], -33.5) == []


def test_vertical_datum_uses_record_fixed_offset():
    assert _states_to_waypoints(
        [{"t": 0.0, "lon": -78.0, "lat": 35.0, "alt": 100.0}], -33.5
    )[0][3] == 66.5


def test_reference_entity_copies_and_recolors():
    entity = _reference_entity_from_adsb(ADSB_CZML, "AFR074_05L", REFERENCE_COLOR)
    assert entity is not None
    assert entity["id"] == "scenario-reference"
    assert entity["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    # the reference inherits the observed-track fading trail (it is copied, not rebuilt)
    assert entity["path"]["trailTime"] == 300
    # the source CZML must not be mutated
    assert ADSB_CZML[1]["path"]["material"]["solidColor"]["color"]["rgba"] == [255, 140, 0, 200]


def test_all_three_trajectories_share_a_fading_trail():
    # All three comparison paths must FADE (finite trailTime), so the moving aircraft head is
    # distinguishable from the tail. The optimizer/simulator used to persist (trailTime 100000)
    # while the reference faded — that mismatch was the bug. They must now trail identically.
    czml = build_comparison_czml(STATE_DATA, ADSB_CZML)
    ref, opt, sim = czml[1], czml[2], czml[3]
    assert _TRAIL_TIME_S < 100000                              # a trailing tail, not "keep whole path"
    assert opt["path"]["trailTime"] == _TRAIL_TIME_S
    assert sim["path"]["trailTime"] == _TRAIL_TIME_S
    assert ref["path"]["trailTime"] == opt["path"]["trailTime"]   # reference fades the same way


def test_build_comparison_czml_has_three_trajectories():
    czml = build_comparison_czml(STATE_DATA, ADSB_CZML)
    assert czml[0]["id"] == "document"
    ids = [packet["id"] for packet in czml[1:]]
    assert ids == ["scenario-reference", "scenario-optimizer", "scenario-simulator"]
    opt = next(p for p in czml if p["id"] == "scenario-optimizer")
    assert opt["path"]["material"]["solidColor"]["color"]["rgba"] == list(OPTIMIZER_COLOR)


def test_no_reference_when_flight_missing():
    czml = build_comparison_czml({
        **STATE_DATA,
        "source": {"id": "NOPE", "runway": "05L", "hae_minus_msl_m": -33.5},
    }, ADSB_CZML)
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


def test_same_callsign_same_runway_on_different_days_are_two_groups(tmp_path):
    # `id` is the callsign (a copy of it, despite the name) and repeats every day, so keying
    # groups on id_runway made one flight silently overwrite the other. Measured on the KRDU
    # harvest that lost 218 of 996 arrivals. The record filename carries the full flight_key
    # (callsign_runway_icao24_landingTime), which is what the group is keyed by now.
    stems = ("ASA677_05R_a54aae_20260629T093123Z", "ASA677_05R_a9e8ce_20260630T093925Z")
    for stem in stems:
        (tmp_path / f"{stem}_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "ASA677", "runway": "05R", "status": "solved",
         "states_file": "ASA677_05R_a54aae_20260629T093123Z_states.json"},
        {"id": "ASA677", "runway": "05R", "status": "solved",
         "states_file": "ASA677_05R_a9e8ce_20260630T093925Z_states.json"},
    ]
    # The observed layer ids the namesakes by their full flight_keys, with distinct
    # positions — so the test can pin that each group copied ITS OWN track, not
    # whichever namesake happened to come first (the old callsign-lookup bug).
    adsb = [
        ADSB_CZML[0],
        {**ADSB_CZML[1], "id": stems[0], "name": "ASA677",
         "position": {"cartographicDegrees": [0, -78.45, 35.74, 2500.0]}},
        {**ADSB_CZML[1], "id": stems[1], "name": "ASA677",
         "position": {"cartographicDegrees": [0, -78.99, 35.99, 3000.0]}},
    ]
    czml, index = build_runway_comparison(results, tmp_path, adsb, airport="KRDU")

    assert len(index) == 2, "two distinct flights must not collapse into one group"
    assert {r["group"] for r in index} == set(stems)
    # Both keep the callsign for display; only the grouping key is the full identity.
    assert {r["flightId"] for r in index} == {"ASA677"}
    assert len([p for p in czml[1:] if p["id"].startswith("opt-")]) == 2
    # Each reference is the RIGHT namesake's track (matched by flight_key, not callsign).
    refs = {p["id"]: p for p in czml[1:] if p["id"].startswith("ref-")}
    assert set(refs) == {f"ref-{stems[0]}", f"ref-{stems[1]}"}
    assert refs[f"ref-{stems[0]}"]["position"]["cartographicDegrees"][1] == -78.45
    assert refs[f"ref-{stems[1]}"]["position"]["cartographicDegrees"][1] == -78.99


def test_failed_and_solved_rows_of_one_flight_still_share_a_group(tmp_path):
    # A failed row has no states_file but always has an eval_file, and the two names share
    # the flight_key stem — so the dedup that prefers the solved row keeps working.
    stem = "AFR074_05L_abc123_20260629T101112Z"
    (tmp_path / f"{stem}_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "failed",
         "states_file": None, "eval_file": f"{stem}_eval.json"},
        {"id": "AFR074", "runway": "05L", "status": "solved",
         "states_file": f"{stem}_states.json", "eval_file": f"{stem}_eval.json"},
    ]
    _, index = build_runway_comparison(results, tmp_path, ADSB_CZML, airport="KRDU")
    assert len(index) == 1
    assert index[0]["status"] == "solved"       # solved wins regardless of row order


def test_build_runway_comparison_solved_three_paths_failed_red(tmp_path):
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"},
        {"id": "DAL1312", "runway": "05L", "status": "failed", "states_file": None},
    ]
    adsb = [
        ADSB_CZML[0],
        ADSB_CZML[1],
        {**ADSB_CZML[1], "id": "DAL1312_05L", "name": "DAL1312"},
    ]
    # The scenario map carries every flight's initial V + mass (built before optimization), so a
    # FAILED optimization still gets V + mass in the index. Keyed by flight_key (= the group).
    scenario_initial = {
        "AFR074_05L": {"V": 130.0, "m": 78000.0},
        "DAL1312_05L": {"V": 122.5, "m": 61000.0},
    }
    czml, index = build_runway_comparison(
        results, tmp_path, adsb, airport="KRDU", scenario_initial=scenario_initial
    )
    ids = [p["id"] for p in czml[1:]]

    # solved flight -> three entities, ids namespaced by the group (the record stem)
    assert {"ref-AFR074_05L", "opt-AFR074_05L", "sim-AFR074_05L"} <= set(ids)
    sim = next(p for p in czml if p["id"] == "sim-AFR074_05L")
    assert sim["path"]["material"]["solidColor"]["color"]["rgba"] == list(SIMULATOR_COLOR)
    assert sim["properties"]["kind"] == "simulator" and sim["properties"]["group"] == "AFR074_05L"
    assert sim["show"] is False                         # hidden by default (sampling mode)

    # failed flight -> reference only, dark red, no optimizer/simulator
    assert "ref-DAL1312_05L" in ids
    assert "opt-DAL1312_05L" not in ids and "sim-DAL1312_05L" not in ids
    failed_ref = next(p for p in czml if p["id"] == "ref-DAL1312_05L")
    assert failed_ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(FAILED_COLOR)

    # index: one record per group, carrying its entity ids + initial state
    by_group = {r["group"]: r for r in index}
    assert set(by_group) == {"AFR074_05L", "DAL1312_05L"}
    solved_rec = by_group["AFR074_05L"]
    assert solved_rec["status"] == "solved"
    assert solved_rec["entities"] == ["ref-AFR074_05L", "opt-AFR074_05L", "sim-AFR074_05L"]
    assert solved_rec["initialState"]["lat"] == STATES[0]["lat"]
    # mass is carried so the frontend flight list can show the optimizer's aircraft mass
    assert solved_rec["initialState"]["m"] == STATES[0]["m"]
    # solved: top-level V + mass come from the optimizer's own initial state
    assert solved_rec["initialVMps"] == STATES[0]["V"]
    assert solved_rec["massKg"] == STATES[0]["m"]

    # failed: no states, but the scenario still supplies V + mass (so the list shows + reds them)
    failed_rec = by_group["DAL1312_05L"]
    assert failed_rec["status"] == "failed"
    assert failed_rec["initialState"] is None
    assert failed_rec["initialVMps"] == 122.5
    assert failed_rec["massKg"] == 61000.0


def test_batch_can_anchor_references_in_canonical_observed_czml(tmp_path):
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "solved",
         "states_file": "AFR074_05L_states.json"},
        {"id": "DAL1312", "runway": "05L", "status": "failed", "states_file": None},
    ]

    czml, index = build_runway_comparison(
        results,
        tmp_path,
        [],
        airport="KRDU",
        include_reference_entities=False,
    )

    physical_ids = {packet["id"] for packet in czml[1:]}
    assert physical_ids == {"opt-AFR074_05L", "sim-AFR074_05L"}
    by_group = {row["group"]: row for row in index}
    assert by_group["AFR074_05L"]["entities"] == [
        "ref-AFR074_05L", "opt-AFR074_05L", "sim-AFR074_05L"
    ]
    assert by_group["DAL1312_05L"]["entities"] == ["ref-DAL1312_05L"]


def test_build_runway_comparison_dedupes_collided_rows(tmp_path):
    # Two summary rows for the same (flightId, runway) point at the SAME states file
    # (the filename collides on overwrite). They must collapse to ONE group/one entity set.
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"},
        {"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"},
    ]
    czml, index = build_runway_comparison(results, tmp_path, [ADSB_CZML[0]], airport="KRDU")
    assert [p["id"] for p in czml if p["id"].startswith("opt-")] == ["opt-AFR074_05L"]
    assert len(index) == 1


def test_build_runway_comparison_solved_wins_over_failed_same_group(tmp_path):
    # A flight can appear as BOTH a failed and a solved row for the same (flightId, runway).
    # Solved must win regardless of order -> exactly one group, one set of entities, no
    # duplicate ids (the failed row must not also emit a colliding ref-).
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "failed", "states_file": None},
        {"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"},
    ]
    czml, index = build_runway_comparison(results, tmp_path, [ADSB_CZML[0], ADSB_CZML[1]], airport="KRDU")
    entity_ids = [p["id"] for p in czml[1:]]
    assert len(entity_ids) == len(set(entity_ids))            # no colliding ids
    assert len(index) == 1 and index[0]["status"] == "solved"
    assert {"ref-AFR074_05L", "opt-AFR074_05L", "sim-AFR074_05L"} == set(entity_ids)


def test_build_runway_comparison_start_visible(tmp_path):
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"}]
    czml, _ = build_runway_comparison(results, tmp_path, [ADSB_CZML[0]], airport="KRDU", start_hidden=False)
    assert next(p for p in czml if p["id"] == "opt-AFR074_05L")["show"] is True


def test_clock_spans_reference_on_failed_only_runway(tmp_path):
    """A runway with ONLY failed scenarios must still span its reference tracks' full duration.

    The clock used to be derived from solved opt/sim states only, so a failed-only runway
    collapsed to a ~1s clock and every (much longer) reference froze at its start point.
    """
    from datetime import datetime

    long_ref = {
        # A file-less row's group is flight_key over the row fields — here just id_runway —
        # and the observed entity id must be that same identity for the lookup to hit.
        "id": "N999XX_32", "name": "N999XX",
        "position": {"epoch": "2026-04-01T08:00:00Z",
                     "cartographicDegrees": [0.0, -78.7, 35.8, 2000.0, 900.0, -78.6, 35.9, 500.0]},
        "path": {"material": {"solidColor": {"color": {"rgba": [235, 235, 235, 200]}}}},
    }
    adsb = [{"id": "document", "clock": {}}, long_ref]
    results = [{"id": "N999XX", "runway": "32", "status": "failed", "states_file": None}]
    czml, _ = build_runway_comparison(results, tmp_path, adsb, airport="KRDU")

    start_s, end_s = czml[0]["clock"]["interval"].split("/")
    duration = (datetime.fromisoformat(end_s.replace("Z", "+00:00"))
                - datetime.fromisoformat(start_s.replace("Z", "+00:00"))).total_seconds()
    assert duration == 900.0   # spans the 900s reference, not ~1s


def test_clock_spans_long_reference_past_short_optimizer(tmp_path):
    """When the reference outlasts the optimizer/simulator, the clock follows the reference."""
    from datetime import datetime

    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")  # opt/sim only 5s
    long_ref = {**ADSB_CZML[1], "position": {"epoch": "2026-04-01T08:00:00Z",
                "cartographicDegrees": [0.0, -78.45, 35.74, 2500.0, 600.0, -78.5, 35.8, 400.0]}}
    adsb = [ADSB_CZML[0], long_ref]
    results = [{"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json"}]
    czml, _ = build_runway_comparison(results, tmp_path, adsb, airport="KRDU")
    start_s, end_s = czml[0]["clock"]["interval"].split("/")
    duration = (datetime.fromisoformat(end_s.replace("Z", "+00:00"))
                - datetime.fromisoformat(start_s.replace("Z", "+00:00"))).total_seconds()
    assert duration == 600.0   # the 600s reference, not the 5s opt/sim


def test_off_target_group_yellow_reference_and_verdict_metrics(tmp_path):
    # A solved flight whose evaluation verdict says success=false is OFF TARGET: its
    # reference renders yellow with status "offTarget" (on all three entities + the index
    # record), and the verdict's final deviations are copied onto the record. A flight whose
    # verdict says success=true stays plain "solved" with the white reference.
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    miss_data = {**STATE_DATA, "source": {
        "id": "DAL1312", "runway": "05L", "hae_minus_msl_m": -33.5
    }}
    (tmp_path / "DAL1312_05L_states.json").write_text(json.dumps(miss_data), encoding="utf-8")
    indeterminate_data = {**STATE_DATA, "source": {
        "id": "UAL55", "runway": "05L", "hae_minus_msl_m": -33.5
    }}
    (tmp_path / "UAL55_05L_states.json").write_text(
        json.dumps(indeterminate_data), encoding="utf-8"
    )
    results = [
        {"id": "AFR074", "runway": "05L", "status": "solved",
         "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"},
        {"id": "DAL1312", "runway": "05L", "status": "solved",
         "states_file": "DAL1312_05L_states.json", "eval_file": "DAL1312_05L_eval.json"},
        {"id": "UAL55", "runway": "05L", "status": "solved",
         "states_file": "UAL55_05L_states.json", "eval_file": "UAL55_05L_eval.json"},
    ]
    adsb = [ADSB_CZML[0], ADSB_CZML[1],
            {**ADSB_CZML[1], "id": "DAL1312_05L", "name": "DAL1312"},
            {**ADSB_CZML[1], "id": "UAL55_05L", "name": "UAL55"}]
    verdicts = load_verdicts({"trajectories": [
        {"file": "AFR074_05L_eval.json", "solved": True, "success": True, "verdict": "pass",
         "lateral_m": 2.8, "vertical_m": 0.1},
        {"file": "DAL1312_05L_eval.json", "solved": True, "success": False, "verdict": "fail",
         "lateral_m": 179.5, "vertical_m": -25.4},
        {"file": "UAL55_05L_eval.json", "solved": True, "success": False,
         "verdict": "indeterminate", "lateral_m": 2.0, "vertical_m": 1.0},
    ]})

    czml, index = build_runway_comparison(results, tmp_path, adsb, airport="KRDU",
                                          verdicts=verdicts)
    by_id = {p["id"]: p for p in czml[1:]}

    off_ref = by_id["ref-DAL1312_05L"]
    assert off_ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(OFF_TARGET_REF_COLOR)
    assert off_ref["name"].endswith("(off target)")
    assert off_ref["properties"]["status"] == "offTarget"
    # The RESULT path is the trajectory that missed the target — IT carries the bright
    # yellow (the frontend keeps the CZML colour for off-target groups); the reference is
    # dark amber. The optimizer PLAN keeps its legend orange.
    off_sim = by_id["sim-DAL1312_05L"]
    assert off_sim["properties"]["status"] == "offTarget"
    assert off_sim["path"]["material"]["solidColor"]["color"]["rgba"] == list(OFF_TARGET_COLOR)
    assert off_sim["name"].endswith("(off target)")
    assert by_id["opt-DAL1312_05L"]["path"]["material"]["solidColor"]["color"]["rgba"] \
        == list(OPTIMIZER_COLOR)

    ok_ref = by_id["ref-AFR074_05L"]
    assert ok_ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    assert ok_ref["properties"]["status"] == "solved"

    undecided_ref = by_id["ref-UAL55_05L"]
    assert undecided_ref["properties"]["status"] == "indeterminate"
    assert undecided_ref["path"]["material"]["solidColor"]["color"]["rgba"] \
        == list(REFERENCE_COLOR)

    by_group = {r["group"]: r for r in index}
    assert by_group["UAL55_05L"]["status"] == "indeterminate"
    assert by_group["UAL55_05L"]["terminalVerdict"] == "indeterminate"
    assert by_group["DAL1312_05L"]["status"] == "offTarget"
    assert by_group["DAL1312_05L"]["lateralErrM"] == 179.5
    assert by_group["DAL1312_05L"]["verticalErrM"] == -25.4
    assert by_group["AFR074_05L"]["status"] == "solved"
    assert by_group["AFR074_05L"]["lateralErrM"] == 2.8


def test_no_verdicts_keeps_solved_plain(tmp_path):
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]
    czml, index = build_runway_comparison(results, tmp_path, [ADSB_CZML[0], ADSB_CZML[1]],
                                          airport="KRDU")
    ref = next(p for p in czml if p["id"] == "ref-AFR074_05L")
    assert ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    assert index[0]["status"] == "solved"
    assert "lateralErrM" not in index[0]


def test_scenario_initial_map_keys_namesakes_apart_by_flight_key(tmp_path):
    # Two landings by the same callsign on the same runway (different aircraft/days) must
    # keep their own V/mass — the old (id, runway) key served one flight's numbers for both.
    from build_scenario_comparison_czml import scenario_initial_map

    scenarios = [
        {"source": {"id": "ASA677", "runway": "05R", "icao24": "a54aae", "hae_minus_msl_m": -33.5,
                    "landing_time_utc": "2026-06-29T09:31:23Z"},
         "initial": {"V": 130.0, "m": 78000.0}},
        {"source": {"id": "ASA677", "runway": "05R", "icao24": "a9e8ce", "hae_minus_msl_m": -33.5,
                    "landing_time_utc": "2026-06-30T09:39:25Z"},
         "initial": {"V": 118.0, "m": 64000.0}},
    ]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(scenarios), encoding="utf-8")

    initial = scenario_initial_map([path])
    assert initial == {
        "ASA677_05R_a54aae_20260629T093123Z": {"V": 130.0, "m": 78000.0},
        "ASA677_05R_a9e8ce_20260630T093925Z": {"V": 118.0, "m": 64000.0},
    }


def test_optimization_stats_merges_summary_and_report():
    summary = {"total": 4, "solved": 3, "failed": 1}
    report = {
        "successful": 2, "success_rate": 0.5,
        "lateral_m": {"mean": 61.4, "p95": 170.0, "max": 179.5},
        "final_time_s": {"mean": 335.9, "min": 284.1, "max": 393.5},
    }
    stats = optimization_stats(summary, report)
    assert stats["solveRate"] == 0.75
    assert stats["successful"] == 2 and stats["successRate"] == 0.5
    assert stats["avgStateErrorM"] == 61.4
    assert stats["avgTimeS"] == 335.9
    # Without a report: solve stats only (no evaluation keys).
    bare = optimization_stats(summary, None)
    assert bare["solveRate"] == 0.75 and "successRate" not in bare
    # An all-unsolved report has null spreads — the means stay None, not a crash.
    empty = optimization_stats(summary, {"successful": 0, "success_rate": 0.0,
                                         "lateral_m": None, "final_time_s": None})
    assert empty["avgStateErrorM"] is None and empty["avgTimeS"] is None


def test_prediction_accuracy_stats_publishes_existing_ade_and_fde():
    summary = {
        "mode": "tsTransformer:itransformer:full:test",
        "accuracy": {
            "flights": 152,
            "flights_without_overlap": 0,
            "final_time_s": {"mae": 58.1, "p95_abs": 120.0, "mean_signed": -4.0},
            "ade_m": {"median": 1400.0, "mean": 1755.6, "p95": 4656.2, "max": 9012.0},
            "fde_m": {"median": 1600.0, "mean": 2082.4, "p95": 6002.1, "max": 10024.0},
            "arrival_endpoint_error_m": {
                "median": 900.0, "mean": 1200.0, "p95": 3500.0, "max": 7000.0
            },
            "cross_track_p95_m": {"mean": 700.0, "p95": 1800.0},
            "altitude_p95_m": {"mean": 90.0, "p95": 210.0},
        },
    }
    assert prediction_accuracy_stats(summary) == {
        "flights": 152,
        "finalTimeS": {"mae": 58.1, "p95Abs": 120.0, "meanSigned": -4.0},
        "adeM": {"median": 1400.0, "mean": 1755.6, "p95": 4656.2, "max": 9012.0},
        "fdeM": {"median": 1600.0, "mean": 2082.4, "p95": 6002.1, "max": 10024.0},
        "arrivalEndpointErrorM": {
            "median": 900.0, "mean": 1200.0, "p95": 3500.0, "max": 7000.0
        },
        "crossTrackP95M": {"mean": 700.0, "p95": 1800.0},
        "altitudeP95M": {"mean": 90.0, "p95": 210.0},
        "rawKinematics": {"predicted": None, "observedBaseline": None, "delta": None},
    }
    assert prediction_accuracy_stats({"mode": "runway", "total": 4}) is None
    assert prediction_accuracy_stats({
        "mode": "tsTransformer:itransformer:full:test",
        "accuracy": {"flights": 0},
    }) == {
        "flights": 0,
        "finalTimeS": None,
        "adeM": None,
        "fdeM": None,
        "arrivalEndpointErrorM": None,
        "crossTrackP95M": None,
        "altitudeP95M": None,
        "rawKinematics": {"predicted": None, "observedBaseline": None, "delta": None},
    }


def test_evaluation_batch_stats_excludes_only_per_flight_details():
    report = {
        # Not a real report version: the builder copies it through without reading it (a pass-through is
        # what is pinned below), so no producer's version is restated here.
        "schema_version": "any-report-version",
        "total": 10,
        "solved": 9,
        "solve_rate": 0.9,
        "successful": 4,
        "success_rate": 0.4,
        "failed": 2,
        "indeterminate": 4,
        "verdict_counts": {"pass": 4, "fail": 2, "indeterminate": 4},
        "lateral_m": {"mean": 12.0, "p95": 30.0, "max": 42.0},
        "vertical_m": {"mean_abs": 3.0, "p95_abs": 7.0},
        "final_time_s": {"mean": 300.0, "min": 200.0, "max": 400.0},
        "trajectories": [{"file": "one_eval.json"}],
    }
    stats = evaluation_batch_stats(report)
    assert stats["schemaVersion"] == "any-report-version"
    assert stats["solveRate"] == 0.9
    assert stats["indeterminate"] == 4
    assert stats["verdictCounts"]["fail"] == 2
    assert stats["lateralM"]["p95"] == 30.0
    assert "trajectories" not in stats


def test_prediction_accuracy_stats_rejects_incomplete_predictor_summary():
    import pytest

    with pytest.raises(ValueError, match="missing its accuracy block"):
        prediction_accuracy_stats({"mode": "tsTransformer:patchtst:window:test"})


def test_prediction_batch_commits_accuracy_to_comparison_index(monkeypatch, tmp_path):
    summary = {
        "mode": "tsTransformer:itransformer:full:test",
        "total": 1,
        "solved": 1,
        "failed": 0,
        "results": [{"id": "ONE", "runway": "05L"}],
        "accuracy": {
            "flights": 1,
            "flights_without_overlap": 0,
            "ade_m": {"mean": 125.0, "p95": 125.0},
            "fde_m": {"mean": 240.0, "p95": 240.0},
        },
    }

    monkeypatch.setattr(
        comparison_builder,
        "build_runway_comparison",
        lambda results, *_args, airport, **_kwargs: (
            [{"id": "document"}],
            [{
                "group": results[0]["id"],
                "flightId": results[0]["id"],
                "runway": results[0]["runway"],
                "airport": airport,
                "status": "solved",
                "entities": [],
            }],
        ),
    )

    index = publish_comparison_batch(
        summary=summary,
        states_dir=tmp_path,
        out_dir=tmp_path,
        airport="KRDU",
        category="ts_itr_full_test",
        start_hidden=True,
        scenario_initial=None,
        evaluation_report={"total": 1, "successful": 0, "success_rate": 0.0,
                           "lateral_m": None, "final_time_s": None, "trajectories": []},
        generation="prediction123",
    )

    assert index["prediction"] == {
        "flights": 1,
        "finalTimeS": None,
        "adeM": {"mean": 125.0, "p95": 125.0},
        "fdeM": {"mean": 240.0, "p95": 240.0},
        "arrivalEndpointErrorM": None,
        "crossTrackP95M": None,
        "altitudeP95M": None,
        "rawKinematics": {"predicted": None, "observedBaseline": None, "delta": None},
    }
    assert json.loads((tmp_path / "comparison_index.json").read_text())["prediction"] \
        == index["prediction"]


def test_publish_evaluation_report_copies_verbatim(tmp_path):
    # The frontend's evaluation view reads this published copy — every number in it comes
    # from `python -m evaluation` (the one backend exit), so the copy must be verbatim.
    from build_scenario_comparison_czml import publish_evaluation_report

    report = {"total": 3, "solve_rate": 0.5, "trajectories": [{"id": "X"}]}
    path = publish_evaluation_report(
        report,
        tmp_path,
        filename="evaluation_report_batch123.json",
    )
    assert path == tmp_path / "evaluation_report_batch123.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_prune_unreferenced_outputs_runs_against_the_committed_roster(tmp_path):
    (tmp_path / "comparison_KRDU_36_old.czml").write_text("[]")
    (tmp_path / "comparison_KRDU_05L_current.czml").write_text("[]")
    (tmp_path / "evaluation_report.json").write_text("{}")
    (tmp_path / "comparison_index.json").write_text("{}")

    deleted = prune_unreferenced_outputs(
        tmp_path, {"comparison_KRDU_05L_current.czml"}
    )
    assert not (tmp_path / "comparison_KRDU_36_old.czml").exists()
    assert (tmp_path / "comparison_KRDU_05L_current.czml").exists()
    assert not (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "comparison_index.json").exists()
    assert len(deleted) == 2


def test_batch_failure_preserves_the_previous_committed_generation(monkeypatch, tmp_path):
    import pytest

    old_file = tmp_path / "comparison_KRDU_05L_previous.czml"
    old_file.write_text("[{\"id\":\"old\"}]")
    old_index = {
        "epoch": "2026-01-01T00:00:00Z",
        "groups": [{"czml": old_file.name}],
    }
    index_path = tmp_path / "comparison_index.json"
    index_path.write_text(json.dumps(old_index))

    summary = {
        "total": 2,
        "solved": 2,
        "failed": 0,
        "failure_rate": 0.0,
        "results": [
            {"id": "ONE", "runway": "05L"},
            {"id": "TWO", "runway": "23R"},
        ],
    }

    def fake_build(results, *_args, airport, **_kwargs):
        runway = results[0]["runway"]
        return ([{"id": "document"}], [{
            "group": results[0]["id"],
            "flightId": results[0]["id"],
            "runway": runway,
            "airport": airport,
            "status": "solved",
            "entities": [],
        }])

    original_write = comparison_builder._write_json_atomic

    def fail_on_second_runway(path, value, *, pretty=False):
        if "23R" in path.name:
            raise RuntimeError("injected second-runway failure")
        original_write(path, value, pretty=pretty)

    monkeypatch.setattr(comparison_builder, "build_runway_comparison", fake_build)
    monkeypatch.setattr(comparison_builder, "_write_json_atomic", fail_on_second_runway)

    with pytest.raises(RuntimeError, match="second-runway"):
        publish_comparison_batch(
            summary=summary,
            states_dir=tmp_path,
            out_dir=tmp_path,
            airport="KRDU",
            category="runway",
            start_hidden=True,
            scenario_initial=None,
            evaluation_report={"total": 2, "trajectories": []},
            generation="next",
        )

    assert json.loads(index_path.read_text()) == old_index
    assert old_file.exists()
    assert not (tmp_path / "comparison_KRDU_05L_next.czml").exists()


def test_batch_requires_an_evaluation_report_for_the_committed_generation(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="evaluation report"):
        publish_comparison_batch(
            summary={
                "total": 0,
                "solved": 0,
                "failed": 0,
                "failure_rate": 0.0,
                "results": [],
            },
            states_dir=tmp_path,
            out_dir=tmp_path,
            airport="KRDU",
            category="runway",
            start_hidden=True,
            scenario_initial=None,
            evaluation_report=None,
            generation="batch123",
        )


def test_batch_index_commits_one_generation_and_then_prunes_old_files(
    monkeypatch, tmp_path
):
    old_file = tmp_path / "comparison_KRDU_23R_previous.czml"
    old_report = tmp_path / "evaluation_report.json"
    old_file.write_text("[]")
    old_report.write_text("{}")
    summary = {
        "total": 1,
        "solved": 1,
        "failed": 0,
        "failure_rate": 0.0,
        "results": [{"id": "ONE", "runway": "05L"}],
    }

    monkeypatch.setattr(
        comparison_builder,
        "build_runway_comparison",
        lambda results, *_args, airport, **_kwargs: (
            [{"id": "document"}],
            [{
                "group": results[0]["id"],
                "flightId": results[0]["id"],
                "runway": results[0]["runway"],
                "airport": airport,
                "status": "solved",
                "entities": [],
            }],
        ),
    )

    published = publish_comparison_batch(
        summary=summary,
        states_dir=tmp_path,
        out_dir=tmp_path,
        airport="KRDU",
        category="runway",
        start_hidden=True,
        scenario_initial=None,
        evaluation_report={"total": 1, "trajectories": []},
        generation="batch123",
    )

    committed = json.loads((tmp_path / "comparison_index.json").read_text())
    assert committed == published
    assert committed["schemaVersion"] == "comparison-v2-generation"
    assert committed["generation"] == "batch123"
    assert committed["groups"][0]["czml"] == "comparison_KRDU_05L_batch123.czml"
    assert committed["evaluationReport"] == "evaluation_report_batch123.json"
    assert (tmp_path / committed["groups"][0]["czml"]).exists()
    assert (tmp_path / committed["evaluationReport"]).exists()
    assert not old_file.exists()
    assert not old_report.exists()


def test_upsert_category_adds_and_replaces(tmp_path):
    manifest = tmp_path / "categories.json"
    _upsert_category(manifest, key="runway", label="Runway target", directory="runway",
                     group_count=10, constrained=False)
    total = _upsert_category(manifest, key="fitted_adsb", label="Fitted ADS-B crossing",
                             directory="fitted_adsb",
                             group_count=20, constrained=False)
    assert total == 2
    cats = json.loads(manifest.read_text())["categories"]
    assert [c["key"] for c in cats] == ["fitted_adsb", "runway"]  # sorted by key
    # re-registering an existing key replaces it in place (no duplicate)
    _upsert_category(manifest, key="fitted_adsb", label="Fitted ADS-B crossing",
                     directory="fitted_adsb",
                     group_count=21, constrained=False)
    cats = json.loads(manifest.read_text())["categories"]
    assert len(cats) == 2
    assert next(c for c in cats if c["key"] == "fitted_adsb")["groups"] == 21


def test_fitted_adsb_category_replaces_legacy_asdb_alias(tmp_path):
    manifest = tmp_path / "categories.json"
    manifest.write_text(json.dumps({"categories": [
        {"key": "asdb", "label": "ADS-B target", "dir": "asdb", "groups": 9,
         "constrained": False},
        {"key": "asdb_cons", "label": "ADS-B target (constrained)", "dir": "asdb_cons",
         "groups": 8, "constrained": True},
        {"key": "runway", "label": "Runway target", "dir": "runway", "groups": 10,
         "constrained": False},
    ]}))

    _upsert_category(
        manifest, key="fitted_adsb", label="Fitted ADS-B crossing",
        directory="fitted_adsb", group_count=11, constrained=False,
    )
    keys = [c["key"] for c in json.loads(manifest.read_text())["categories"]]
    assert keys == ["fitted_adsb", "runway"]


def test_upsert_category_stamps_the_explicit_constrained_field(tmp_path):
    # Constrained-ness is a manifest FIELD the frontend keys off — never derived
    # from the key/dir spelling (docs once said "runwayConstrained", which no
    # suffix check would match).
    manifest = tmp_path / "categories.json"
    _upsert_category(manifest, key="runway", label="Runway target", directory="runway",
                     group_count=10, constrained=False)
    _upsert_category(manifest, key="runway_cons", label="Runway (constrained)",
                     directory="runway_cons", group_count=9, constrained=True)
    cats = {c["key"]: c for c in json.loads(manifest.read_text())["categories"]}
    assert cats["runway"]["constrained"] is False
    assert cats["runway_cons"]["constrained"] is True


def test_upsert_prediction_category_stamps_dataset_split(tmp_path):
    manifest = tmp_path / "categories.json"
    _upsert_category(
        manifest,
        key="ts_pooled_itr_normalized_time_train",
        label="Training split (in-sample)",
        directory="ts_pooled_itr_normalized_time_train",
        group_count=12,
        constrained=False,
        dataset_split="train",
    )
    category = json.loads(manifest.read_text())["categories"][0]
    assert category["datasetSplit"] == "train"


def test_dataset_splits_mirror_the_frontend():
    """`DATASET_SPLITS` is a MIRROR of the frontend's list (a split only one side knows is either
    refused here or dropped there as an invalid category)."""
    import re

    from build_scenario_comparison_czml import DATASET_SPLITS

    source = (Path(__file__).resolve().parents[2] / "src" / "data" / "airportData.ts").read_text(encoding="utf-8")
    literal = re.search(r"export const DATASET_SPLITS = \[([^\]]*)\] as const", source).group(1)
    assert tuple(re.findall(r'"([^"]+)"', literal)) == DATASET_SPLITS
    assert "dayval" in DATASET_SPLITS


def test_upsert_category_stamps_compact_accuracy_for_result_ranking(tmp_path):
    # The frontend ranks a split's results by mean/p95 ADE/FDE from categories.json
    # alone — without fetching each category's full comparison index.
    from build_scenario_comparison_czml import category_accuracy_summary

    prediction_block = {
        "flights": 1404,
        "adeM": {"mean": 1406.6, "p95": 4113.4, "median": 662.8, "max": 10651.5},
        "fdeM": {"mean": 1697.0, "p95": 5922.9},
        "finalTimeS": {"mae": 26.2},
    }
    accuracy = category_accuracy_summary(prediction_block)
    assert accuracy == {
        "adeM": {"mean": 1406.6, "p95": 4113.4},
        "fdeM": {"mean": 1697.0, "p95": 5922.9},
    }
    # Optimizer batches have no prediction block and must not grow the field.
    assert category_accuracy_summary(None) is None

    manifest = tmp_path / "categories.json"
    _upsert_category(
        manifest,
        key="ts_pooled_itr_normalized_time_val",
        label="Validation split",
        directory="ts_pooled_itr_normalized_time_val",
        group_count=12,
        constrained=False,
        dataset_split="val",
        accuracy=accuracy,
    )
    _upsert_category(
        manifest,
        key="runway",
        label="Runway target",
        directory="runway",
        group_count=12,
        constrained=False,
    )
    categories = {c["key"]: c for c in json.loads(manifest.read_text())["categories"]}
    assert categories["ts_pooled_itr_normalized_time_val"]["accuracy"] == accuracy
    assert "accuracy" not in categories["runway"]


def test_upsert_experiment_category_stamps_grouped_checkpoint_metadata(tmp_path):
    manifest = tmp_path / "categories.json"
    experiment = {
        "id": "campaign/stage/run",
        "group": "campaign",
        "checkpoint": "4dTrajectory/outputs/POOLED/experiments/campaign/stage/run/checkpoint.pt",
        "model": "itransformer",
        "predictionOutput": "control",
        "horizonMode": "normalized",
        "seed": 1337,
    }
    _upsert_category(
        manifest,
        key="experiment_run_abc_val",
        label="Validation — Experiment run",
        directory="experiment_run_abc_val",
        group_count=12,
        constrained=False,
        dataset_split="val",
        result_source="experiment",
        experiment=experiment,
    )
    category = json.loads(manifest.read_text())["categories"][0]
    assert category["resultSource"] == "experiment"
    assert category["experiment"] == experiment


def test_cli_rejects_a_category_split_that_disagrees_with_the_summary(monkeypatch, tmp_path):
    import pytest
    import sys

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"split": "test"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "build_scenario_comparison_czml.py",
        "--summary", str(summary),
        "--output-dir", str(tmp_path / "comparison"),
        "--evaluation-report", str(tmp_path / "evaluation_report.json"),
        "--dataset-split", "train",
    ])

    with pytest.raises(SystemExit, match="2"):
        comparison_builder.main()


# ── Prediction schema (4dTrajectory/ts_transformer) ──────────────────────────

# A prediction record rebases time so t=0 is the ANCHOR: `predicted_states` runs forward from
# there, and `observed_states` is the WHOLE observed track, so it starts at negative t. Here the
# model was shown 4 s of lookback (t = -4 … 0) before forecasting, i.e. the anchor sits 4 s into
# the observed track (`anchorTimeS`).
LOOKBACK_STATES = [
    {**STATES[0], "t": -4.0, "lat": 35.76, "lon": -78.43},
    {**STATES[0], "t": -2.0, "lat": 35.75, "lon": -78.44},
]
PREDICTION_STATE_DATA = {
    "source": {
        "id": "AFR074", "predictor": "itransformer", "anchorTimeS": 4.0,
        "hae_minus_msl_m": -33.5,
    },
    "final_time_s": 5.0,
    "predicted_states": STATES,
    # `+ STATES`, not `+ STATES[1:]`: the anchor sample belongs to both halves — the writer
    # emits the very same state object at t=0 in each — which is what makes the drawn lookback
    # meet the forecast with no gap.
    "observed_states": LOOKBACK_STATES + STATES,
}
# The observed CZML a comparison overlay is fed is the ARRIVAL WINDOW: the model slice,
# t=0 at terminal-ring entry — the same origin every record carries. Its first sample is
# therefore the group's first OBSERVED sample (the lookback's), not the anchor's. The
# airport-wide trajectories.czml (ADSB_CZML above) is the other window and starts earlier
# and further out; feeding it here is the misalignment `_require_reference_aligned` exists
# to catch.
ARRIVAL_WINDOW_CZML = [
    {"id": "document", "clock": {}},
    {
        "id": "AFR074_05L",
        "name": "AFR074",
        "position": {
            "cartographicDegrees": [
                0.0, LOOKBACK_STATES[0]["lon"], LOOKBACK_STATES[0]["lat"], 2600.0,
                2.0, LOOKBACK_STATES[1]["lon"], LOOKBACK_STATES[1]["lat"], 2550.0,
                4.0, STATES[0]["lon"], STATES[0]["lat"], 2500.0,
                9.0, STATES[1]["lon"], STATES[1]["lat"], 2400.0,
            ]
        },
        "path": {
            "leadTime": 0, "trailTime": 300,
            "material": {"solidColor": {"color": {"rgba": [255, 140, 0, 200]}}},
        },
    },
]


def test_states_schema_distinguishes_the_two_producers():
    assert states_schema(STATE_DATA) == "optimizer"
    assert states_schema(PREDICTION_STATE_DATA) == "predicted"


def test_states_schema_rejects_a_file_matching_neither():
    import pytest

    with pytest.raises(KeyError, match="neither the optimizer schema"):
        states_schema({"source": {}, "final_time_s": 1.0})


def test_prediction_states_render_as_one_purple_path_plus_the_reference(tmp_path):
    # A learned predictor has no plan-vs-replay split: one trajectory, prefixed `pred-` so
    # the frontend's kindOfEntityId maps it to the "predicted" kind (and its own legend
    # colour) rather than mislabelling it as an optimizer plan. It comes with `look-`, the
    # faded lookback window it was conditioned on.
    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]

    czml, index = build_runway_comparison(results, tmp_path, ARRIVAL_WINDOW_CZML, airport="KRDU")
    ids = [p["id"] for p in czml if p.get("id") != "document"]
    assert ids == ["ref-AFR074_05L", "look-AFR074_05L", "pred-AFR074_05L"]

    prediction = next(p for p in czml if p["id"] == "pred-AFR074_05L")
    assert prediction["path"]["material"]["solidColor"]["color"]["rgba"] == list(PREDICTION_COLOR)
    assert prediction["properties"]["kind"] == "predicted"
    assert index[0]["entities"] == ["ref-AFR074_05L", "look-AFR074_05L", "pred-AFR074_05L"]


def test_prediction_is_shifted_onto_the_references_timeline(tmp_path):
    # The record's t=0 is the ANCHOR, but the reference copied from the ADS-B CZML starts at
    # t=0 = the START of the observed track. Writing the record's times through unshifted drew
    # the forecast `anchorTimeS` seconds early — on real KRDU 05L data the forecast's first
    # sample (bit-identical to the reference's t=118 s sample) landed at t=0, 12 km from where
    # the reference was then. Every prediction-schema entity is shifted by anchorTimeS.
    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]

    czml, _ = build_runway_comparison(results, tmp_path, ARRIVAL_WINDOW_CZML, airport="KRDU")
    lookback = next(p for p in czml if p["id"] == "look-AFR074_05L")
    prediction = next(p for p in czml if p["id"] == "pred-AFR074_05L")

    # anchorTimeS = 4.0: the lookback occupies [0, 4] and the forecast starts exactly where it
    # ends, so the two draw as one continuous track rather than a line beginning in mid-air.
    assert _offsets(lookback) == [0.0, 2.0, 4.0]
    assert _offsets(prediction) == [4.0, 9.0]
    # The shared anchor sample is the same position in both — the join is exact, not merely close.
    assert _sample_at(lookback, -1) == _sample_at(prediction, 0)
    # ...and the group now starts where its reference starts, which is the whole point of
    # shifting: the lookback's first sample IS the reference's first sample.
    reference = next(p for p in czml if p["id"] == "ref-AFR074_05L")
    assert _offsets(reference)[0] == _offsets(lookback)[0] == 0.0
    assert _sample_at(reference, 0)[:2] == _sample_at(lookback, 0)[:2]


# An "anytime" prediction: the SAME record contract, re-anchored MID-APPROACH. The anytime
# runner (`run_ts.py anytime_curve --write-records`) forecasts each flight from where it still
# had ~12 km of path left to fly, which on a real arrival is a few hundred seconds into the
# 25 km slice — not the 120 s L−1 anchor every other prediction batch shares. Nothing here is
# a second schema: t=0 is still the anchor and the anchor is still `source.anchorTimeS`, which
# is exactly why the placement has to come off that field and never off an assumed L−1.
MID_APPROACH_LOOKBACK = [
    {**STATES[0], "t": -240.0, "lat": 35.90, "lon": -78.30},
    {**STATES[0], "t": -120.0, "lat": 35.85, "lon": -78.35},
    {**STATES[0], "t": -60.0, "lat": 35.80, "lon": -78.40},
]
MID_APPROACH_STATE_DATA = {
    "source": {
        "id": "AFR074", "predictor": "itransformer", "anchorTimeS": 240.0,
        "hae_minus_msl_m": -33.5,
    },
    "final_time_s": 5.0,
    "predicted_states": STATES,
    "observed_states": MID_APPROACH_LOOKBACK + STATES,
}
# The same arrival window (t=0 at terminal-ring entry) as ARRIVAL_WINDOW_CZML, long enough to
# reach the anchor: the reference must still cover the forecast after the shift.
MID_APPROACH_WINDOW_CZML = [
    {"id": "document", "clock": {}},
    {
        "id": "AFR074_05L",
        "name": "AFR074",
        "position": {
            "cartographicDegrees": [
                0.0, -78.30, 35.90, 2900.0,
                120.0, -78.35, 35.85, 2800.0,
                180.0, -78.40, 35.80, 2700.0,
                240.0, STATES[0]["lon"], STATES[0]["lat"], 2500.0,
                245.0, STATES[1]["lon"], STATES[1]["lat"], 2400.0,
            ]
        },
        "path": {
            "leadTime": 0, "trailTime": 300,
            "material": {"solidColor": {"color": {"rgba": [255, 140, 0, 200]}}},
        },
    },
]


def test_a_mid_approach_anchor_is_drawn_where_the_flight_was_then(tmp_path):
    # The anytime delivery form. A record anchored at 240 s must be drawn 240 s along the
    # observed flight's clock — the shift is whatever `anchorTimeS` says, not the L−1 value
    # every other batch happens to share. Drawn from t=0 instead, the forecast would begin at
    # the 25 km slice entry, kilometres from where the aircraft actually was, and would read
    # as a gross model error rather than as a publication bug.
    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(MID_APPROACH_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]

    czml, _ = build_runway_comparison(
        results, tmp_path, MID_APPROACH_WINDOW_CZML, airport="KRDU")
    lookback = next(p for p in czml if p["id"] == "look-AFR074_05L")
    prediction = next(p for p in czml if p["id"] == "pred-AFR074_05L")
    reference = next(p for p in czml if p["id"] == "ref-AFR074_05L")

    # The forecast starts MID-approach, at the anchor — never at the window's start.
    assert _offsets(prediction) == [240.0, 245.0]
    # The faded segment is the observed track BEFORE the anchor — at a re-anchored record that
    # is more than the model's input window, which is why it is no longer named as one
    # (`_lookback_states`, and follow-up 21 in docs/code-health-followups.md). What it must do
    # is meet the forecast exactly, so the pair draws as one track.
    assert _offsets(lookback) == [0.0, 120.0, 180.0, 240.0]
    assert _sample_at(lookback, -1) == _sample_at(prediction, 0)
    # ...and where it starts is where the observed flight was at 240 s — the reference's own
    # sample at that offset — so the purple line leaves the white one instead of crossing it.
    assert _offsets(reference) == [0.0, 120.0, 180.0, 240.0, 245.0]
    assert _sample_at(reference, 3)[:2] == _sample_at(prediction, 0)[:2]
    # The reference (the ARRIVAL window, the model's origin) still spans the whole group.
    assert _offsets(reference)[0] == _offsets(lookback)[0] == 0.0
    assert max(_offsets(reference)) >= max(_offsets(prediction))
    # ...and the clock is derived from the SHIFTED entities, so playback reaches the forecast.
    assert comparison_builder._entity_last_time(prediction) == 245.0


def test_a_mid_approach_group_still_requires_the_arrival_window_reference(tmp_path):
    # The later the anchor, the further ahead a full-track reference puts the group — and the
    # check has to keep firing, because a mid-approach forecast drawn against the wrong origin
    # is entirely plausible-looking.
    import pytest

    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(MID_APPROACH_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]

    with pytest.raises(ValueError, match="requires the arrival window"):
        build_runway_comparison(results, tmp_path, ADSB_CZML, airport="KRDU")


def test_a_full_track_reference_is_refused_for_a_prediction_group(tmp_path):
    # The other half of the same alignment. `anchorTimeS` puts the group on the ARRIVAL
    # window's origin (t=0 = terminal-ring entry); a reference copied from the airport-wide
    # trajectories.czml is on the full track's origin (t=0 = first reception), a median 45 s
    # and 5 km earlier on real KRDU data. Both start at t=0 and both name the right flight,
    # so nothing downstream can tell — the group simply renders ahead of its own truth.
    import pytest

    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]

    with pytest.raises(ValueError, match="requires the arrival window"):
        build_runway_comparison(results, tmp_path, ADSB_CZML, airport="KRDU")


def test_lookback_is_faded_and_carries_its_own_kind(tmp_path):
    # Same hue as the forecast (it is one track), lower alpha (this half was GIVEN to the model,
    # not produced by it). The frontend keys the fade off `kind`, so both must be right.
    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]

    czml, _ = build_runway_comparison(results, tmp_path, ARRIVAL_WINDOW_CZML, airport="KRDU")
    lookback = next(p for p in czml if p["id"] == "look-AFR074_05L")

    assert lookback["path"]["material"]["solidColor"]["color"]["rgba"] == list(LOOKBACK_COLOR)
    assert lookback["properties"]["kind"] == "lookback"
    assert LOOKBACK_COLOR[:3] == PREDICTION_COLOR[:3]
    assert LOOKBACK_COLOR[3] < PREDICTION_COLOR[3]


def test_prediction_schema_requires_the_observed_track(tmp_path):
    # Without `observed_states` there is no lookback to draw and the forecast would start in
    # mid-air. That is a broken record, not a variant to degrade gracefully around.
    import pytest

    with pytest.raises(KeyError, match="neither the optimizer schema"):
        states_schema({"source": {}, "final_time_s": 1.0, "predicted_states": STATES})


def test_a_prediction_missing_the_gates_keeps_its_own_colour(tmp_path):
    # An optimizer result that misses the gates is recoloured off-target yellow so the few
    # bad ones stand out. A forecast essentially NEVER makes the 106.75 m lateral gate, so
    # ~100% of a prediction batch would go yellow — the marking would carry no information
    # and would erase the kind's colour. Status stays accurate; only the colouring is skipped.
    (tmp_path / "AFR074_05L_states.json").write_text(
        json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]
    verdicts = {"AFR074_05L_eval.json": {"solved": True, "success": False,
                                         "verdict": "fail",
                                         "lateral_m": 413.5, "vertical_m": 12.0}}

    czml, index = build_runway_comparison(results, tmp_path, ARRIVAL_WINDOW_CZML, airport="KRDU",
                                          verdicts=verdicts)
    prediction = next(p for p in czml if p["id"] == "pred-AFR074_05L")
    reference = next(p for p in czml if p["id"] == "ref-AFR074_05L")

    assert prediction["path"]["material"]["solidColor"]["color"]["rgba"] == list(PREDICTION_COLOR)
    assert reference["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    assert "off target" not in prediction["name"]
    # ...but the verdict is NOT hidden: status and the per-flight deviation still carry it.
    assert prediction["properties"]["status"] == "offTarget"
    assert index[0]["lateralErrM"] == 413.5


def test_an_optimizer_result_missing_the_gates_still_goes_off_target_yellow(tmp_path):
    # The guard above must not have disabled off-target marking for the optimizer path.
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved",
                "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"}]
    verdicts = {"AFR074_05L_eval.json": {"solved": True, "success": False,
                                         "verdict": "fail",
                                         "lateral_m": 900.0, "vertical_m": 30.0}}

    czml, _ = build_runway_comparison(results, tmp_path, ADSB_CZML, airport="KRDU",
                                      verdicts=verdicts)
    simulator = next(p for p in czml if p["id"] == "sim-AFR074_05L")
    reference = next(p for p in czml if p["id"] == "ref-AFR074_05L")
    assert simulator["path"]["material"]["solidColor"]["color"]["rgba"] == list(OFF_TARGET_COLOR)
    assert reference["path"]["material"]["solidColor"]["color"]["rgba"] == list(OFF_TARGET_REF_COLOR)
    assert "(off target)" in simulator["name"]


def test_czml_precision_mirrors_the_record_contract():
    # The CZML rounding is a deliberate MIRROR of evaluation_export.STATE_DECIMALS (this
    # package must not import the modeling tree). If the record contract moves and this
    # does not, the plotted path and the judged path start disagreeing in the last digits.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(
        0, str(_Path(__file__).resolve().parents[3] / "4dTrajectory" / "optimization")
    )
    from evaluation_export import STATE_DECIMALS

    assert "t" not in STATE_DECIMALS and not hasattr(comparison_builder, "_T_DECIMALS")
    assert (comparison_builder._DEG_DECIMALS
            == STATE_DECIMALS["lat"] == STATE_DECIMALS["lon"])
    assert comparison_builder._ALT_DECIMALS == STATE_DECIMALS["alt"]


def test_a_flight_graded_on_the_runway_it_landed_on_takes_that_verdict(tmp_path):
    # The two-tier generation records' grading (ts runner R38): a sentence that landed on another runway than the
    # observed flight's is graded again there. Passing there is its own status, `otherRunway` (drawn in its own
    # colour), and the record names both runways' verdicts; failing there is plain offTarget; a flight not graded
    # again keeps the category's verdict.
    for flight in ("AFR074", "DAL1312", "UPS22"):
        (tmp_path / f"{flight}_05L_states.json").write_text(json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": flight, "runway": "05L", "status": "solved", "states_file": f"{flight}_05L_states.json",
                "eval_file": f"{flight}_05L_eval.json"} for flight in ("AFR074", "DAL1312", "UPS22")]
    verdicts = load_verdicts({"trajectories": [
        {"file": f"{flight}_05L_eval.json", "solved": True, "verdict": verdict, "lateral_m": 1400.0,
         "vertical_m": 3.0} for flight, verdict in (("AFR074", "fail"), ("DAL1312", "fail"), ("UPS22", "pass"))]})
    landed = load_verdicts({"trajectories": [
        {"file": "AFR074_05L_eval.json", "runway": "05R", "solved": True, "verdict": "pass", "lateral_m": 2.5,
         "vertical_m": 1.0},
        {"file": "DAL1312_05L_eval.json", "runway": "05R", "solved": True, "verdict": "fail", "lateral_m": 30.0,
         "vertical_m": 1.0}]})
    _czml, index = build_runway_comparison(results, tmp_path, ARRIVAL_WINDOW_CZML, airport="KRDU",
                                           verdicts=verdicts, landed_verdicts=landed)
    by_flight = {record["flightId"]: record for record in index}
    assert by_flight["AFR074"]["status"] == "otherRunway"
    assert (by_flight["AFR074"]["landedRunway"], by_flight["AFR074"]["observedRunwayVerdict"]) == ("05R", "fail")
    assert (by_flight["AFR074"]["terminalVerdict"], by_flight["AFR074"]["lateralErrM"]) == ("pass", 2.5)
    assert by_flight["DAL1312"]["status"] == "offTarget" and by_flight["DAL1312"]["landedRunway"] == "05R"
    assert by_flight["UPS22"]["status"] == "solved" and "landedRunway" not in by_flight["UPS22"]


def test_a_landed_runway_verdict_that_is_not_another_runways_graded_verdict_is_refused(tmp_path):
    (tmp_path / "AFR074_05L_states.json").write_text(json.dumps(PREDICTION_STATE_DATA), encoding="utf-8")
    results = [{"id": "AFR074", "runway": "05L", "status": "solved", "states_file": "AFR074_05L_states.json",
                "eval_file": "AFR074_05L_eval.json"}]
    verdicts = load_verdicts({"trajectories": [{"file": "AFR074_05L_eval.json", "solved": True, "verdict": "fail"}]})
    for landed in ({"runway": "05R", "solved": False, "verdict": "fail"},      # unsolved: never "passed there"
                   {"runway": "05R", "solved": True, "verdict": None},         # no verdict
                   {"runway": "05L", "solved": True, "verdict": "pass"}):      # its own runway is not another
        with pytest.raises(ValueError, match="not one of another runway's graded verdicts"):
            build_runway_comparison(results, tmp_path, ARRIVAL_WINDOW_CZML, airport="KRDU", verdicts=verdicts,
                                    landed_verdicts={"AFR074_05L_eval.json": {"file": "AFR074_05L_eval.json", **landed}})


def test_a_landed_runway_report_grading_a_flight_the_category_lacks_is_refused(tmp_path):
    summary = {"results": [], "split": "val"}
    with pytest.raises(ValueError, match="grades flights the category does not hold"):
        comparison_builder.publish_comparison_batch(
            summary=summary, states_dir=tmp_path, out_dir=tmp_path / "out", airport="KRDU", category=None,
            start_hidden=True, scenario_initial=None, evaluation_report={"trajectories": []},
            landed_runway_report={"trajectories": [{"file": "X_05L_eval.json", "runway": "05R", "solved": True,
                                                    "verdict": "pass"}]})



# ── Traffic windows (4dTrajectory/optimization/traffic_optimization.py m1) ────

COMMANDED = "AFR074_05L_a1b2c3_20260501T000300Z"
EARLIER = "DAL1312_05L_d4e5f6_20260501T000100Z"      # already in the scene when the commanded one enters
LATER = "UPS22_05R_a7b8c9_20260501T000500Z"         # enters after it
ENTRY_UTC = {
    COMMANDED: "2026-05-01T00:00:10.250Z",
    EARLIER: "2026-05-01T00:00:00.901Z",
    LATER: "2026-05-01T00:01:41.000Z",
}


def _write_traffic_window(directory: Path, *, entry: str = ENTRY_UTC[COMMANDED], sidecar: dict | None = None,
                          write_sidecar: bool = True) -> list[dict]:
    """One window in ``directory``: its states file, its sidecar, and the summary row naming them."""
    state_data = {**STATE_DATA, "source": {**STATE_DATA["source"], "entry_time_utc": entry}}
    (directory / f"{COMMANDED}_states.json").write_text(json.dumps(state_data), encoding="utf-8")
    if write_sidecar:
        (directory / f"{COMMANDED}_traffic.json").write_text(json.dumps(sidecar or {
            "schema": comparison_builder.TRAFFIC_SCHEMA, "flight_key": COMMANDED,
            "outcome": "separated", "recorded": [EARLIER, LATER],
        }), encoding="utf-8")
    return [{"id": "AFR074", "runway": "05L", "status": "solved",
             "states_file": f"{COMMANDED}_states.json", "eval_file": f"{COMMANDED}_eval.json"}]


def _write_arrivals_manifest(directory: Path, entries: dict[str, str] = ENTRY_UTC,
                             schema_version: str | None = comparison_builder.ARRIVALS_SCHEMA_VERSION) -> Path:
    path = directory / "manifest.json"
    path.write_text(json.dumps({"schema_version": schema_version, "records": [
        {"flight_key": key, "entry_time_utc": entry} for key, entry in entries.items()]}), encoding="utf-8")
    return path


def _traffic_index(directory: Path, results: list[dict], entries: dict[str, str] = ENTRY_UTC) -> list[dict]:
    _czml, index = build_runway_comparison(
        results, directory, [], airport="KRDU", include_reference_entities=False,
        traffic_entry_times=comparison_builder.load_entry_times(_write_arrivals_manifest(directory, entries)))
    return index


def test_a_traffic_window_carries_its_outcome_the_recorded_aircraft_and_when_each_enters(tmp_path):
    index = _traffic_index(tmp_path, _write_traffic_window(tmp_path))

    assert [record["group"] for record in index] == [COMMANDED]
    # `recorded` is the logical reference id the group's own reference carries (`ref-<flight_key>`);
    # the offsets are the seconds each enters after the commanded aircraft, to the millisecond.
    assert index[0]["traffic"] == {
        "outcome": "separated",
        "recorded": [f"ref-{EARLIER}", f"ref-{LATER}"],
        "startOffsetsS": [-9.349, 90.75],
    }
    assert f"ref-{COMMANDED}" in index[0]["entities"]


def test_a_batch_without_traffic_entry_times_has_no_traffic_block(tmp_path):
    results = _write_traffic_window(tmp_path)
    _czml, index = build_runway_comparison(results, tmp_path, [], airport="KRDU", include_reference_entities=False)
    assert "traffic" not in index[0]


def test_a_missing_traffic_sidecar_is_refused_by_name(tmp_path):
    results = _write_traffic_window(tmp_path, write_sidecar=False)
    with pytest.raises(ValueError, match=f"{COMMANDED}: traffic sidecar {COMMANDED}_traffic.json is missing"):
        _traffic_index(tmp_path, results)


@pytest.mark.parametrize("schema", ["optimization-traffic-v1", "optimization-traffic-block-v2", None])
def test_a_traffic_sidecar_of_another_schema_is_refused_by_name(tmp_path, schema):
    sidecar = {"flight_key": COMMANDED, "outcome": "separated", "recorded": []}
    if schema is not None:
        sidecar["schema"] = schema
    results = _write_traffic_window(tmp_path, sidecar=sidecar)
    with pytest.raises(ValueError, match=f"{COMMANDED}_traffic.json has schema {schema!r}, this builder reads "
                                         f"'optimization-traffic-v2' only"):
        _traffic_index(tmp_path, results)


def test_a_traffic_sidecar_of_another_flight_is_refused(tmp_path):
    results = _write_traffic_window(tmp_path, sidecar={
        "schema": comparison_builder.TRAFFIC_SCHEMA, "flight_key": EARLIER, "outcome": "separated", "recorded": []})
    with pytest.raises(ValueError, match=f"belongs to {EARLIER!r}"):
        _traffic_index(tmp_path, results)


def test_a_recorded_flight_the_arrival_window_does_not_hold_is_refused_by_name(tmp_path):
    results = _write_traffic_window(tmp_path)
    entries = {key: entry for key, entry in ENTRY_UTC.items() if key != LATER}
    with pytest.raises(ValueError, match=f"1 recorded flight_key.*not in the arrivals manifest.*{LATER}"):
        _traffic_index(tmp_path, results, entries)


def test_a_manifest_that_is_not_the_one_the_run_flew_is_refused(tmp_path):
    results = _write_traffic_window(tmp_path)
    entries = {**ENTRY_UTC, COMMANDED: "2026-05-01T00:00:08.000Z"}      # an older roster: entry 2 s early
    with pytest.raises(ValueError, match="the manifest is not the one this run flew"):
        _traffic_index(tmp_path, results, entries)


def test_an_unsolved_row_is_a_group_without_traffic_and_is_not_refused(tmp_path):
    # Its M1 baseline failed: a failed record and no sidecar. The solved window beside it keeps its block.
    results = _write_traffic_window(tmp_path) + [{
        "id": "BAW9", "runway": "05L", "status": "failed", "eval_file": "BAW9_05L_b00009_20260501T000900Z_eval.json"}]
    index = {record["flightId"]: record for record in _traffic_index(tmp_path, results)}
    assert index["BAW9"]["status"] == "failed" and "traffic" not in index["BAW9"]
    assert index["AFR074"]["traffic"]["outcome"] == "separated"


def test_the_traffic_mirrors_are_the_sources_constants():
    # Read as text, not imported: the optimizer modules pull in casadi, and this package must not
    # depend on the modeling tree (same rule as the precision mirror above).
    root = Path(__file__).resolve().parents[3]
    optimization = root / "4dTrajectory" / "optimization"
    assert f'TRAFFIC_RECORD_SCHEMA = "{comparison_builder.TRAFFIC_SCHEMA}"' in \
        (optimization / "traffic" / "loop.py").read_text(encoding="utf-8")
    assert f'TRAFFIC_SUFFIX = "{comparison_builder.TRAFFIC_SUFFIX}"' in \
        (optimization / "traffic_optimization.py").read_text(encoding="utf-8")
    assert f'M1_MODE, M2_MODE = "{comparison_builder.M1_MODE}", "{comparison_builder.M2_MODE}"' in \
        (optimization / "traffic" / "__init__.py").read_text(encoding="utf-8")
    assert f'BLOCK_RECORD_SCHEMA = "{comparison_builder.BLOCK_SCHEMA}"' in \
        (optimization / "traffic" / "block.py").read_text(encoding="utf-8")
    # the name rule the sidecar path is built by: the states name without its suffix, plus the sidecar's
    assert "states_name.removesuffix(_STATES_SUFFIX) + suffix" in \
        (optimization / "scenario_batch.py").read_text(encoding="utf-8")
    assert f'SCHEMA_VERSION = "{comparison_builder.ARRIVALS_SCHEMA_VERSION}"' in \
        (root / "trajectory_data_process" / "harvest" / "arrivals.py").read_text(encoding="utf-8")


def test_the_frontends_flight_key_request_limit_is_the_backends():
    # The frontend splits the recorded aircraft's keys into requests of at most this many; the backend
    # refuses a longer list outright.
    root = Path(__file__).resolve().parents[3]
    frontend = re.search(r"MAX_FLIGHT_KEYS_PER_REQUEST = (\d+);",
                         (root / "aeroviz-4d" / "src" / "data" / "observedTracks.ts").read_text(encoding="utf-8"))
    backend = re.search(r"MAX_TRAJECTORIES_PER_RESPONSE = (\d+)\n",
                        (root / "aeroviz_backend" / "observed_trajectories.py").read_text(encoding="utf-8"))
    assert frontend and backend and frontend.group(1) == backend.group(1)


def _run_cli(monkeypatch, *argv: str) -> None:
    import sys
    monkeypatch.setattr(sys, "argv", ["build_scenario_comparison_czml.py", *argv])
    comparison_builder.main()


def _publish(monkeypatch, directory: Path, results: list[dict], manifest: Path, *, mode: str | None) -> Path:
    """Run the CLI over a summary of ``results`` (a traffic summary names its roster); returns the category dir."""
    summary = {"total": len(results), "solved": len(results), "failed": 0, "results": results}
    if mode is not None:
        summary["mode"] = mode
        summary["optimization_config"] = {"traffic": {"selection": {"manifest": str(manifest)}}}
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "evaluation_report.json").write_text(json.dumps({
        "total": 1, "solved": 1, "successful": 1, "success_rate": 1.0, "lateral_m": None, "final_time_s": None,
        "trajectories": []}), encoding="utf-8")
    out = directory / "comparison" / "traffic_m1_runway"
    _run_cli(monkeypatch, "--summary", str(directory / "summary.json"), "--output-dir", str(out), "--airport", "KRDU",
             "--evaluation-report", str(directory / "evaluation_report.json"), "--category", "traffic_m1_runway",
             "--category-label", "Traffic M1 (runway)")
    return out


def test_a_traffic_summary_publishes_its_traffic_with_the_entry_fields_existing_categories_use(monkeypatch, tmp_path):
    out = _publish(monkeypatch, tmp_path, _write_traffic_window(tmp_path), _write_arrivals_manifest(tmp_path),
                   mode=comparison_builder.M1_MODE)

    index = json.loads((out / "comparison_index.json").read_text(encoding="utf-8"))
    assert index["groups"][0]["traffic"]["recorded"] == [f"ref-{EARLIER}", f"ref-{LATER}"]
    # ONLY the fields existing categories carry: `isComparisonCategory` rejects nothing here, and one
    # category it rejected would empty the whole airport's picker.
    categories = json.loads((tmp_path / "comparison" / "categories.json").read_text(encoding="utf-8"))["categories"]
    assert categories == [{"key": "traffic_m1_runway", "label": "Traffic M1 (runway)", "dir": "traffic_m1_runway",
                           "groups": 1, "constrained": False}]


def test_a_traffic_summary_requires_its_sidecars(monkeypatch, tmp_path):
    results = _write_traffic_window(tmp_path, write_sidecar=False)
    with pytest.raises(ValueError, match=f"{COMMANDED}: traffic sidecar {COMMANDED}_traffic.json is missing"):
        _publish(monkeypatch, tmp_path, results, _write_arrivals_manifest(tmp_path), mode=comparison_builder.M1_MODE)
    assert not (tmp_path / "comparison" / "traffic_m1_runway" / "comparison_index.json").exists()   # nothing committed


def test_a_summary_of_another_mode_is_published_as_before_sidecars_or_not(monkeypatch, tmp_path):
    out = _publish(monkeypatch, tmp_path, _write_traffic_window(tmp_path), _write_arrivals_manifest(tmp_path), mode=None)
    assert "traffic" not in json.loads((out / "comparison_index.json").read_text(encoding="utf-8"))["groups"][0]


@pytest.mark.parametrize("mode", ["traffic:m3", "traffic:"])
def test_a_traffic_summary_of_another_kind_is_refused_by_name_not_published_as_a_plain_category(monkeypatch, tmp_path, mode):
    with pytest.raises(ValueError, match=f"summary mode {mode!r}: this builder publishes 'traffic:m1' and 'traffic:m2' runs only"):
        _publish(monkeypatch, tmp_path, _write_traffic_window(tmp_path), _write_arrivals_manifest(tmp_path), mode=mode)
    assert not (tmp_path / "comparison" / "traffic_m1_runway" / "comparison_index.json").exists()


@pytest.mark.parametrize("schema_version", ["harvest-arrivals-v6-cifp-leg-vertical", None])
def test_an_arrivals_roster_of_another_schema_is_refused_by_name(monkeypatch, tmp_path, schema_version):
    manifest = _write_arrivals_manifest(tmp_path, schema_version=schema_version)
    if schema_version is None:                                  # a roster that names no schema at all
        document = json.loads(manifest.read_text(encoding="utf-8"))
        del document["schema_version"]
        manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=f"arrivals roster schema {schema_version!r}, this builder reads "
                                         f"'{comparison_builder.ARRIVALS_SCHEMA_VERSION}' only"):
        _publish(monkeypatch, tmp_path, _write_traffic_window(tmp_path), manifest, mode=comparison_builder.M1_MODE)


# ── An M2 run as one scene (4dTrajectory/optimization/traffic_optimization.py m2) ───────────────────────────

G1 = "AFR074_23L_a1b2c3_20260521T180900Z"     # solved, enters second
G2 = "DAL1312_23R_d4e5f6_20260521T181500Z"    # solved, enters last
G3 = "UPS22_23L_a7b8c9_20260521T180100Z"      # unsolved, enters FIRST: the scene starts at its entry
BG1 = "SWA1_23L_aaaaaa_20260521T181000Z"      # recorded aircraft that are not groups; BG1 enters before the scene
BG2 = "SWA2_32_bbbbbb_20260521T182000Z"
SCENE_ENTRY_UTC = {
    G3: "2026-05-21T17:50:00.250Z",
    G1: "2026-05-21T17:55:10.000Z",
    G2: "2026-05-21T18:00:30.500Z",
    BG1: "2026-05-21T17:49:00.000Z",
    BG2: "2026-05-21T18:10:00.000Z",
}
SCENE_START_UTC = "2026-05-21T17:50:00.250Z"
SCENE_STAGES = [(G1, "AFR074", "23L", "separated_at_baseline", 12.5, [G2, G3, BG2, BG1]),
                (G2, "DAL1312", "23R", "separated", 0.0, [G1, BG2, BG1])]      # BG1, BG2 recorded by both


def _write_scene(directory: Path, *, sidecar_schema: str = comparison_builder.BLOCK_SCHEMA,
                 write_sidecars: bool = True, reason: str | None = "BaselineFailed: slot solve (delay 180.5 s)",
                 entries: dict[str, str] = SCENE_ENTRY_UTC, eval_entry: str = SCENE_ENTRY_UTC[G3],
                 slots: list[dict] | None = None) -> dict:
    """An M2 summary in ``directory``: two solved groups with block sidecars, one unsolved group, the roster."""
    results = []
    for key, flight, runway, outcome, delay, recorded in SCENE_STAGES:
        state_data = {**STATE_DATA, "source": {**STATE_DATA["source"], "entry_time_utc": SCENE_ENTRY_UTC[key]}}
        (directory / f"{key}_states.json").write_text(json.dumps(state_data), encoding="utf-8")
        if write_sidecars:
            (directory / f"{key}_traffic.json").write_text(json.dumps({
                "schema": sidecar_schema, "flight_key": key, "outcome": outcome, "recorded": recorded,
                "slot": {"eta_utc_s": 1.0, "cta_utc_s": 2.0, "delay_s": delay}}), encoding="utf-8")
        results.append({"id": flight, "runway": runway, "status": "solved", "states_file": f"{key}_states.json",
                        "eval_file": f"{key}_eval.json"})
    (directory / f"{G3}_eval.json").write_text(json.dumps({"source": {"entry_time_utc": eval_entry}}), encoding="utf-8")
    results.append({"id": "UPS22", "runway": "23L", "status": "failed", "states_file": None,
                    "eval_file": f"{G3}_eval.json", "reason": reason})
    roster = _write_arrivals_manifest(directory, entries)
    return {
        "mode": comparison_builder.M2_MODE, "total": 3, "solved": 2, "failed": 1, "results": results,
        "optimization_config": {"traffic": {"selection": {"manifest": str(roster)}}},
        "blocks": [{"slots": slots if slots is not None else [
            {"flight_key": G1, "delay_s": 12.5}, {"flight_key": G2, "delay_s": 0.0}, {"flight_key": G3, "delay_s": 180.5}]}],
    }


def _publish_scene(monkeypatch, directory: Path, summary: dict) -> Path:
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "evaluation_report.json").write_text(json.dumps({
        "total": 3, "solved": 2, "successful": 2, "success_rate": 1.0, "lateral_m": None, "final_time_s": None,
        "trajectories": []}), encoding="utf-8")
    out = directory / "comparison" / "traffic_m2_runway"
    _run_cli(monkeypatch, "--summary", str(directory / "summary.json"), "--output-dir", str(out), "--airport", "KRDU",
             "--evaluation-report", str(directory / "evaluation_report.json"), "--category", "traffic_m2_runway",
             "--category-label", "Traffic M2 (runway)")
    return out


def _scene_index(monkeypatch, directory: Path, **kwargs) -> dict:
    out = _publish_scene(monkeypatch, directory, _write_scene(directory, **kwargs))
    return json.loads((out / "comparison_index.json").read_text(encoding="utf-8"))


def test_a_scene_starts_at_the_earliest_group_entry_solved_or_not_and_names_its_background(monkeypatch, tmp_path):
    index = _scene_index(monkeypatch, tmp_path)

    # The unsolved group enters first, so it is the scene start; the others are offset from it.
    assert index["scene"]["startUtc"] == SCENE_START_UTC
    # The background: the union of the solved sidecars' recorded aircraft minus every group (G2 and G3 are groups),
    # sorted, each once (BG1 and BG2 are recorded by both), offset from the scene start (BG1 enters before it).
    assert index["scene"]["background"] == {"recorded": [f"ref-{BG1}", f"ref-{BG2}"], "startOffsetsS": [-60.25, 1199.75]}
    groups = {group["group"]: group for group in index["groups"]}
    assert set(groups) == {G1, G2, G3}
    assert all("traffic" not in group for group in groups.values())          # the M1 field is the other mode's


def test_each_scene_group_carries_its_offset_outcome_and_delay(monkeypatch, tmp_path):
    groups = {group["group"]: group for group in _scene_index(monkeypatch, tmp_path)["groups"]}

    # solved: the sidecar's outcome and slot delay
    assert groups[G1]["scene"] == {"startOffsetS": 309.75, "outcome": "separated_at_baseline", "delayS": 12.5}
    assert groups[G2]["scene"] == {"startOffsetS": 630.25, "outcome": "separated", "delayS": 0.0}
    # unsolved: the row's reason, and its delay from the block's slots
    assert groups[G3]["scene"] == {"startOffsetS": 0.0, "delayS": 180.5,
                                   "outcome": "BaselineFailed: slot solve (delay 180.5 s)"}
    assert groups[G3]["status"] == "failed"


def test_an_unsolved_group_without_a_slot_has_no_delay(monkeypatch, tmp_path):
    slots = [{"flight_key": G1, "delay_s": 12.5}, {"flight_key": G2, "delay_s": 0.0}]
    groups = {group["group"]: group for group in _scene_index(monkeypatch, tmp_path, slots=slots)["groups"]}
    assert groups[G3]["scene"]["delayS"] is None


def test_the_result_paths_are_written_on_the_scene_clock(monkeypatch, tmp_path):
    out = _publish_scene(monkeypatch, tmp_path, _write_scene(tmp_path))
    index = json.loads((out / "comparison_index.json").read_text(encoding="utf-8"))
    packets = {}
    for name in {group["czml"] for group in index["groups"]}:
        packets.update({packet["id"]: packet for packet in json.loads((out / name).read_text(encoding="utf-8"))[1:]})

    # the records' own times are 0 s and 5 s; on the scene clock they are the group's offset later
    for key, offset in ((G1, 309.75), (G2, 630.25)):
        for kind in ("opt", "sim"):
            assert _offsets(packets[f"{kind}-{key}"]) == [offset, offset + 5.0]
    # and the group's own flight time stays its own
    assert {group["group"]: group["finalTimeS"] for group in index["groups"]}[G1] == STATE_DATA["final_time_s"]


def test_a_scene_category_is_listed_with_the_entry_fields_existing_categories_use(monkeypatch, tmp_path):
    _scene_index(monkeypatch, tmp_path)
    categories = json.loads((tmp_path / "comparison" / "categories.json").read_text(encoding="utf-8"))["categories"]
    assert categories == [{"key": "traffic_m2_runway", "label": "Traffic M2 (runway)", "dir": "traffic_m2_runway",
                           "groups": 3, "constrained": False}]


def test_a_solved_scene_row_without_its_sidecar_is_refused_by_name(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match=f"{G1}: traffic sidecar {G1}_traffic.json is missing"):
        _scene_index(monkeypatch, tmp_path, write_sidecars=False)


@pytest.mark.parametrize("schema", [comparison_builder.TRAFFIC_SCHEMA, "optimization-traffic-block-v1", None])
def test_a_scene_sidecar_of_another_schema_is_refused_by_name(monkeypatch, tmp_path, schema):
    with pytest.raises(ValueError, match=f"{G1}: traffic sidecar {G1}_traffic.json has schema {schema!r}, "
                                         f"this builder reads 'optimization-traffic-block-v2' only"):
        _scene_index(monkeypatch, tmp_path, sidecar_schema=schema)


def test_a_group_the_roster_lacks_is_refused_by_name(monkeypatch, tmp_path):
    entries = {key: entry for key, entry in SCENE_ENTRY_UTC.items() if key != G2}
    with pytest.raises(ValueError, match=f"1 group.*not in the arrivals manifest.*{G2}"):
        _scene_index(monkeypatch, tmp_path, entries=entries)


def test_a_background_flight_the_roster_lacks_is_refused_by_name(monkeypatch, tmp_path):
    entries = {key: entry for key, entry in SCENE_ENTRY_UTC.items() if key != BG2}
    with pytest.raises(ValueError, match=f"1 background flight_key.*not in the arrivals manifest.*{BG2}"):
        _scene_index(monkeypatch, tmp_path, entries=entries)


def test_a_roster_that_is_not_the_one_the_run_flew_is_refused_for_solved_and_unsolved_groups(monkeypatch, tmp_path):
    older = {**SCENE_ENTRY_UTC, G1: "2026-05-21T17:55:08.000Z"}               # a solved group's record says otherwise
    with pytest.raises(ValueError, match=f"{G1}: the arrivals manifest puts its entry at .*the manifest is not the one"):
        _scene_index(monkeypatch, tmp_path, entries=older)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match=f"{G3}: the arrivals manifest puts its entry at .*the manifest is not the one"):
        _scene_index(monkeypatch, other, eval_entry="2026-05-21T17:49:58.000Z")      # the unsolved group's eval record


def test_an_unsolved_scene_row_must_name_its_reason(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match=f"{G3}: an unsolved scene row names no reason"):
        _scene_index(monkeypatch, tmp_path, reason=None)


def test_a_scene_summary_is_not_read_as_an_m1_run(monkeypatch, tmp_path):
    # the mode decides which sidecar schema is read: M1 refuses the block sidecars of a scene
    summary = _write_scene(tmp_path)
    summary["mode"] = comparison_builder.M1_MODE
    with pytest.raises(ValueError, match=f"has schema 'optimization-traffic-block-v2', this builder reads "
                                         f"'optimization-traffic-v2' only"):
        _publish_scene(monkeypatch, tmp_path, summary)


def test_a_block_that_failed_publishes_its_rows_as_unsolved_groups_with_no_delay(monkeypatch, tmp_path):
    # `_fly_one_block`: a block that raises gets the summary shape of a normal one — no slots, every aircraft's
    # outcome "block failed: …", no final losses, plus its `error` — and a failed row with that reason for each aircraft.
    summary = _write_scene(tmp_path)
    reason = "block failed: ValueError: no schedule"
    for result, (key, *_rest) in zip(summary["results"], SCENE_STAGES):
        (tmp_path / f"{key}_eval.json").write_text(json.dumps({"source": {"entry_time_utc": SCENE_ENTRY_UTC[key]}}),
                                                   encoding="utf-8")
        result.update(status="failed", states_file=None, reason=reason)
    summary["results"][2]["reason"] = reason            # the unsolved third row (G3) already was failed
    summary.update(solved=0, failed=3)
    summary["blocks"] = [{"aircraft": 3, "scheduled": 0, "eta_failed": 0, "slot_failed": 0, "schedule_speed_mps": None,
                          "slots": [], "outcomes": {G1: reason, G2: reason, G3: reason}, "final_losses": {},
                          "error": "ValueError: no schedule"}]

    out = _publish_scene(monkeypatch, tmp_path, summary)

    index = json.loads((out / "comparison_index.json").read_text(encoding="utf-8"))
    assert {group["group"]: group["status"] for group in index["groups"]} == {G1: "failed", G2: "failed", G3: "failed"}
    assert {group["group"]: group["scene"] for group in index["groups"]} == {
        G1: {"startOffsetS": 309.75, "outcome": reason, "delayS": None},
        G2: {"startOffsetS": 630.25, "outcome": reason, "delayS": None},
        G3: {"startOffsetS": 0.0, "outcome": reason, "delayS": None},
    }
    # no solved group, so no sidecar was read and there is no background; the scene still starts at the earliest entry
    assert index["scene"] == {"startUtc": SCENE_START_UTC, "background": {"recorded": [], "startOffsetsS": []}}
