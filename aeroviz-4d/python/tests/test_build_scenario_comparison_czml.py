"""Tests for the scenario-comparison CZML builder (single + per-runway batch)."""

import json

from build_scenario_comparison_czml import (
    FAILED_COLOR,
    OFF_TARGET_COLOR,
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
    group_results_by_runway,
    load_verdicts,
    optimization_stats,
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
        # A short trailTime like the real trajectories.czml — the reference builder must OVERRIDE it.
        "path": {"leadTime": 0, "trailTime": 300, "material": {"solidColor": {"color": {"rgba": [255, 140, 0, 200]}}}},
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
    # The scenario map carries every flight's initial V + mass (built before optimization), so a
    # FAILED optimization still gets V + mass in the index.
    scenario_initial = {
        ("AFR074", "05L"): {"V": 130.0, "m": 78000.0},
        ("DAL1312", "05L"): {"V": 122.5, "m": 61000.0},
    }
    czml, index = build_runway_comparison(
        results, tmp_path, adsb, airport="KRDU", scenario_initial=scenario_initial
    )
    ids = [p["id"] for p in czml[1:]]

    # solved flight -> three entities, ids namespaced by {flightId}_{runway} (collision-free)
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
        "id": "N999XX", "name": "N999XX",
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
    miss_data = {**STATE_DATA, "source": {"id": "DAL1312"}}
    (tmp_path / "DAL1312_05L_states.json").write_text(json.dumps(miss_data), encoding="utf-8")
    results = [
        {"id": "AFR074", "runway": "05L", "status": "solved",
         "states_file": "AFR074_05L_states.json", "eval_file": "AFR074_05L_eval.json"},
        {"id": "DAL1312", "runway": "05L", "status": "solved",
         "states_file": "DAL1312_05L_states.json", "eval_file": "DAL1312_05L_eval.json"},
    ]
    adsb = [ADSB_CZML[0], ADSB_CZML[1], {**ADSB_CZML[1], "id": "DAL1312", "name": "DAL1312"}]
    verdicts = load_verdicts({"trajectories": [
        {"file": "AFR074_05L_eval.json", "solved": True, "success": True,
         "lateral_m": 2.8, "vertical_m": 0.1},
        {"file": "DAL1312_05L_eval.json", "solved": True, "success": False,
         "lateral_m": 179.5, "vertical_m": -25.4},
    ]})

    czml, index = build_runway_comparison(results, tmp_path, adsb, airport="KRDU",
                                          verdicts=verdicts)
    by_id = {p["id"]: p for p in czml[1:]}

    off_ref = by_id["ref-DAL1312_05L"]
    assert off_ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(OFF_TARGET_COLOR)
    assert off_ref["name"].endswith("(off target)")
    assert off_ref["properties"]["status"] == "offTarget"
    assert by_id["sim-DAL1312_05L"]["properties"]["status"] == "offTarget"
    # opt/sim keep their legend colours — the marking is the reference + status.
    assert by_id["sim-DAL1312_05L"]["path"]["material"]["solidColor"]["color"]["rgba"] \
        == list(SIMULATOR_COLOR)

    ok_ref = by_id["ref-AFR074_05L"]
    assert ok_ref["path"]["material"]["solidColor"]["color"]["rgba"] == list(REFERENCE_COLOR)
    assert ok_ref["properties"]["status"] == "solved"

    by_group = {r["group"]: r for r in index}
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


def test_publish_evaluation_report_copies_verbatim(tmp_path):
    # The frontend's evaluation view reads this published copy — every number in it comes
    # from `python -m evaluation` (the one backend exit), so the copy must be verbatim.
    from build_scenario_comparison_czml import publish_evaluation_report

    report = {"total": 3, "solve_rate": 0.5, "trajectories": [{"id": "X"}]}
    path = publish_evaluation_report(report, tmp_path)
    assert path == tmp_path / "evaluation_report.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_upsert_category_adds_and_replaces(tmp_path):
    manifest = tmp_path / "categories.json"
    _upsert_category(manifest, key="runway", label="Runway target", directory="runway", group_count=10)
    total = _upsert_category(manifest, key="asdb", label="ADS-B target", directory="asdb", group_count=20)
    assert total == 2
    cats = json.loads(manifest.read_text())["categories"]
    assert [c["key"] for c in cats] == ["asdb", "runway"]  # sorted by key
    # re-registering an existing key replaces it in place (no duplicate)
    _upsert_category(manifest, key="asdb", label="ADS-B target", directory="asdb", group_count=21)
    cats = json.loads(manifest.read_text())["categories"]
    assert len(cats) == 2
    assert next(c for c in cats if c["key"] == "asdb")["groups"] == 21
