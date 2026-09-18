"""A2's failure modes (`manoeuvre/failure_modes.py`, `experiments/executor_failure_modes.py`): the
course frame reads to-go / cross-track / heading against the target's course, the six modes are
taken in the module's order, the summary is per mode, and a per-mode record subset is a valid
record directory carrying its block."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from ts_transformer.experiments import executor_failure_modes as runner
from ts_transformer.inference.export import copy_record_subset
from ts_transformer.manoeuvre import failure_modes as fm

TARGET = {"t": 0.0, "lat": 35.8744, "lon": -78.8020, "alt": 129.3, "V": 74.6, "psi": math.pi / 4, "gamma": -0.05, "m": 66000.0}
COURSE = np.array([math.cos(TARGET["psi"]), math.sin(TARGET["psi"])])
RIGHT = np.array([math.sin(TARGET["psi"]), -math.cos(TARGET["psi"])])


def _state(t: float, to_go_m: float, cross_m: float, psi: float, *, V: float = 80.0, alt: float = 500.0) -> dict:
    east, north = -to_go_m * COURSE + cross_m * RIGHT
    return {"t": t, "lat": TARGET["lat"] + north / METRES_PER_DEG_LAT, "lon": TARGET["lon"] + east / metres_per_deg_lon(TARGET["lat"]),
            "alt": alt, "V": V, "psi": psi, "gamma": 0.0, "m": 66000.0}


def _path(points):
    return [_state(*p) for p in points]


INBOUND = TARGET["psi"]
TRUTH = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi / 2), (30.0, 5000.0, 3000.0, INBOUND + math.pi / 2),
               (60.0, 4000.0, 300.0, INBOUND), (90.0, 2000.0, 100.0, INBOUND), (120.0, 0.0, 0.0, INBOUND)])


def test_the_course_frame_reads_to_go_cross_track_and_heading_error():
    geometry = fm.course_geometry(TRUTH, TARGET)
    assert geometry["to_go_m"][0] == pytest.approx(8000.0, abs=1.0) and geometry["cross_m"][0] == pytest.approx(3000.0, abs=1.0)
    # the heading is the chart track's (the transport factors turn it by ≤ 0.2°), as approach_difficulty reads it
    assert geometry["heading_error_deg"][0] == pytest.approx(90.0, abs=0.2) and geometry["heading_error_deg"][-1] == pytest.approx(0.0, abs=0.2)
    assert geometry["established"].tolist() == [False, False, True, True, False]     # the last row is AT the threshold, not before it
    with pytest.raises(ValueError, match="at least one state"):
        fm.course_geometry([], TARGET)


def test_every_failure_mode_is_taken_in_order():
    # aligned and on the course, the budget ended 1500 m short
    short = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi / 2), (60.0, 4000.0, 200.0, INBOUND), (100.0, 1500.0, 50.0, INBOUND)])
    # ran past the threshold's abeam line 2 km right of the course, never on it
    abeam = _path([(0.0, 8000.0, 3000.0, INBOUND), (60.0, 2000.0, 2500.0, INBOUND), (120.0, -1000.0, 2000.0, INBOUND)])
    # cut through the extended centreline still turning, never aligned
    overshoot = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi / 2), (60.0, 5000.0, 500.0, INBOUND + 1.0), (100.0, 4000.0, -1500.0, INBOUND + 1.0)])
    # heading aligned, 1200 m right of the course, never captured
    offset = _path([(0.0, 8000.0, 1200.0, INBOUND), (60.0, 4000.0, 1200.0, INBOUND), (100.0, 2000.0, 1200.0, INBOUND)])
    # flew on downwind for the whole budget
    no_turn = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi), (60.0, 12000.0, 3000.0, INBOUND + math.pi), (100.0, 15000.0, 3000.0, INBOUND + math.pi)])
    # the turn under way at the budget: aligned briefly, off the course, not at the end
    other = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi / 2), (60.0, 5000.0, 2000.0, INBOUND), (100.0, 4000.0, 1500.0, INBOUND + 1.0)])
    expected = {"established-short": short, "passed-abeam": abeam, "overshoot": overshoot, "parallel-offset": offset,
                "no-turn": no_turn, "other": other}
    rows = {}
    for mode, flown in expected.items():
        row = fm.flight_failure(flown, TRUTH, TARGET)
        assert row["mode"] == mode, (mode, row)
        rows[mode] = row
    assert rows["established-short"]["end"]["to_go_m"] == pytest.approx(1500.0, abs=1.0) and rows["established-short"]["ever_established"]
    assert rows["established-short"]["turn_delay_s"] == pytest.approx(0.0)        # both first aligned at 60 s
    assert rows["no-turn"]["final_turn_s"] == {"flown": None, "truth": 60.0} and rows["no-turn"]["turn_delay_s"] is None
    assert rows["no-turn"]["first_aligned_s"] == {"flown": None, "truth": 60.0} and rows["no-turn"]["start"]["to_go_m"] == pytest.approx(8000.0, abs=1.0)
    assert rows["overshoot"]["first_aligned_s"]["flown"] is None and rows["overshoot"]["first_established_s"]["flown"] is None
    assert rows["passed-abeam"]["final_turn_s"]["flown"] == 0.0        # aligned from the first row: the onset is row 0
    # the final turn is the LAST onset: aligned at entry, vectored away, back onto the course at 100 s
    late = _path([(0.0, 9000.0, 200.0, INBOUND), (40.0, 7000.0, 2000.0, INBOUND + math.pi / 2), (100.0, 5000.0, 300.0, INBOUND),
                  (130.0, 3000.0, 100.0, INBOUND)])
    row = fm.flight_failure(late, TRUTH, TARGET)
    assert row["mode"] == "established-short" and row["final_turn_s"]["flown"] == 100.0 and row["turn_delay_s"] == pytest.approx(40.0)
    assert fm.alignment_onsets(np.array([True, True, False, True, False, False, True])).tolist() == [0, 3, 6]
    assert rows["overshoot"]["centreline_crossings"] == 1 and rows["passed-abeam"]["end"]["side"] == "right"
    # the truth's speed at the flown end is interpolated on the truth's clock (80 m/s everywhere here)
    assert rows["other"]["end_errors"]["speed_mps"] == pytest.approx(0.0)
    table = fm.summarise({f"KRDU:{mode}": row for mode, row in rows.items()})
    assert all(table[mode]["n"] == 1 and table[mode]["share"] == pytest.approx(1 / 6) for mode in fm.FAILURE_MODES)
    assert table["no-turn"]["turn_delay_p50_s"] is None and table["passed-abeam"]["right_side_share"] == 1.0
    assert table["overshoot"]["flights"] == ["KRDU:overshoot"]


RAW_KEYS = ("position_velocity_rmse_mps", "heading_consistency_p95_deg", "turn_rate_p95_deg_s", "acceleration_p95_mps2", "jerk_p95_mps3")


def _row(stem: str, ade: float, *, tortuosity: float = 1.5, established_at_anchor: bool = False) -> dict:
    """A `write_batch` results row (the metric columns `metrics_from_row` reads back)."""
    return {"states_file": f"{stem}_states.json", "eval_file": f"{stem}_eval.json", "ade_m": ade, "fde_m": ade * 2, "arrival_endpoint_error_m": ade,
            "cross_track_p95_m": ade / 2, "altitude_p95_m": 30.0, "metric_steps": 64, "final_time_error_s": 3.0, "true_final_time_s": 300.0,
            "raw_kinematics": {"predicted": {k: 1.0 for k in RAW_KEYS}, "observed_baseline": {**{k: 2.0 for k in RAW_KEYS}, RAW_KEYS[1]: None}},
            "route_tortuosity": tortuosity, "remaining_path_m": 20_000.0, "established_at_anchor": established_at_anchor, "arr_airport": "KRDU", "runway": "05L"}


def _record_dir(root, flights: dict[str, tuple[list, list]]) -> tuple:
    """A record directory of `write_batch`'s shape: ``flights`` maps a stem to (flown states, truth states)."""
    source = root / "records"
    (source / "references").mkdir(parents=True)
    rows = []
    for stem, (flown, truth) in flights.items():
        (source / f"{stem}_states.json").write_text(json.dumps({"predicted_states": flown, "observed_states": truth}), encoding="utf-8")
        def record(states, subject):
            return {"initial_state": {k: v for k, v in states[0].items() if k != "t"}, "target_state": {k: v for k, v in TARGET.items() if k != "t"},
                    "final_time_s": states[-1]["t"], "states": [], "controls": [],
                    "source": {"subject": subject, "arr_airport": "KRDU", "runway": "05L"}}
        (source / f"{stem}_eval.json").write_text(json.dumps({**record(flown, "predicted"),
                                                              "states_ref": {"file": f"{stem}_states.json", "key": "predicted_states"}}), encoding="utf-8")
        (source / "references" / f"{stem}_reference_eval.json").write_text(json.dumps({
            **record(truth, "observed"), "states_ref": {"file": f"../{stem}_states.json", "key": "observed_states", "start_index": 0}}), encoding="utf-8")
        rows.append(_row(stem, 100.0 * (len(rows) + 1)))
    (source / "summary.json").write_text(json.dumps({"mode": "tsTransformer:itransformer:normalized:control:val", "total": len(rows), "solved": len(rows),
                                                     "accuracy": {"flights": len(rows)}, "checkpoint": "x", "config": {},
                                                     "manoeuvre_lockstep": {"protocol": "none"}, "results": rows}), encoding="utf-8")
    return source, rows


def test_a_record_subset_keeps_the_three_files_of_each_flight_and_owns_its_accuracy(tmp_path):
    source, _rows = _record_dir(tmp_path, {stem: (TRUTH, TRUTH) for stem in ("A_05L", "B_05L", "C_05L")})
    out = copy_record_subset(source, tmp_path / "records_no-turn", ["B_05L", "C_05L"], extra_summary={"failure_modes": {"mode": "no-turn"}})
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == summary["solved"] == 2 and [r["states_file"] for r in summary["results"]] == ["B_05L_states.json", "C_05L_states.json"]
    # the subset's accuracy is its own, read back off the retained rows (review 2026-09-18: the picker needs it)
    assert summary["accuracy"]["flights"] == 2 and summary["accuracy"]["ade_m"]["mean"] == pytest.approx(250.0)
    assert summary["accuracy"]["raw_kinematics"]["observed_baseline"][RAW_KEYS[1]]["count"] == 0
    assert summary["subset"] == {"of": str(source), "of_total": 3, "stems": ["B_05L", "C_05L"]}
    assert summary["manoeuvre_lockstep"] == {"protocol": "none"} and summary["failure_modes"] == {"mode": "no-turn"}
    assert sorted(p.name for p in out.iterdir()) == ["B_05L_eval.json", "B_05L_states.json", "C_05L_eval.json", "C_05L_states.json", "references", "summary.json"]
    assert (out / "references" / "C_05L_reference_eval.json").is_file() and not (out / "A_05L_states.json").exists()
    with pytest.raises(FileExistsError):
        copy_record_subset(source, out, ["B_05L"], extra_summary={})
    with pytest.raises(ValueError, match="does not hold"):
        copy_record_subset(source, tmp_path / "other", ["Z_05L"], extra_summary={})
    with pytest.raises(ValueError, match="overwrite"):
        copy_record_subset(source, tmp_path / "other2", ["B_05L"], extra_summary={"total": 9})
    stale = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    del stale["results"][0]["cross_track_p95_m"]
    (source / "summary.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="predate"):
        copy_record_subset(source, tmp_path / "other3", ["A_05L"], extra_summary={})


def test_the_runner_reads_the_stratum_s_non_crossing_flights_off_the_records(tmp_path, capsys):
    """`executor_failure_modes` main(): the flights are the stratum's rows without the crossing
    verdict, each read from its eval record and its reference record (`records/` layout), and
    every mode with a flight gets a publishable subset with the `failure_modes` block."""
    no_turn = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi), (60.0, 12000.0, 3000.0, INBOUND + math.pi), (100.0, 15000.0, 3000.0, INBOUND + math.pi)])
    short = _path([(0.0, 8000.0, 3000.0, INBOUND + math.pi / 2), (60.0, 4000.0, 200.0, INBOUND), (100.0, 1500.0, 50.0, INBOUND)])
    lockstep = tmp_path / "lockstep"
    source, _rows = _record_dir(lockstep, {"A_05L": (no_turn, TRUTH), "B_05L": (short, TRUTH), "C_05L": (TRUTH, TRUTH), "D_05L": (no_turn, TRUTH)})
    covariates = {"route_tortuosity": 1.5, "established_at_anchor": False, "remaining_path_m": 20_000.0}
    rows = {
        "KRDU:A_05L": {"flight_id": "A_05L", "reference": {"established": False}, "ended": "horizon", "predictions": 5, **covariates},
        "KRDU:B_05L": {"flight_id": "B_05L", "reference": {"established": False}, "ended": "horizon", "predictions": 5, **covariates},
        "KRDU:C_05L": {"flight_id": "C_05L", "reference": {"established": True}, "ended": "crossed", "predictions": 3, **covariates},      # crossed: not read
        "KRDU:D_05L": {"flight_id": "D_05L", "reference": {"established": False}, "ended": "horizon", "predictions": 5, **covariates, "route_tortuosity": 1.0},  # straight-in
    }
    (lockstep / "manoeuvre_lockstep.json").write_text(json.dumps({"protocol": "none", "executor": "x", "first_prediction": {"rule": "fixed L-1 (row 7)"},
                                                                  "rows": rows}), encoding="utf-8")
    assert runner.main(["--lockstep", str(lockstep), "--out", str(tmp_path / "out"), "--per-mode", "1"]) == 0
    result = json.loads((tmp_path / "out" / "failure_modes.json").read_text(encoding="utf-8"))
    assert result["stratum"] == "vectored" and result["stratum_flights"] == 3 and result["flights"] == 2
    assert {key: row["mode"] for key, row in result["rows"].items()} == {"KRDU:A_05L": "no-turn", "KRDU:B_05L": "established-short"}
    assert result["modes"]["no-turn"]["flights"] == ["KRDU:A_05L"] and result["modes"]["established-short"]["ever_established_share"] == 1.0
    printed = capsys.readouterr().out
    assert "2 of 3 vectored flights did not cross the threshold" in printed and "records_no-turn: 1 of 1 flights" in printed
    subset = json.loads((tmp_path / "out" / "records_no-turn" / "summary.json").read_text(encoding="utf-8"))
    assert subset["failure_modes"]["mode"] == "no-turn" and subset["total"] == 1 and subset["manoeuvre_lockstep"] == {"protocol": "none"}
    assert not (tmp_path / "out" / "records_overshoot").exists()
    with pytest.raises(SystemExit):        # never overwritten
        runner.main(["--lockstep", str(lockstep), "--out", str(tmp_path / "out")])
