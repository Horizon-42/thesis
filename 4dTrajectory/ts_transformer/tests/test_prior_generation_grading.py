"""A generation-records directory graded beyond its pass rate (`experiments.prior_generation_grading`): a sentence ending on
another runway graded against that runway's own threshold point, and FDE split into its time and its place."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from evaluation.arrival import _require_target_agrees_with_runway_data
from evaluation.records import record_from_dict
from ts_transformer.experiments.prior_generation_grading import graded_copy, landed_target, summarise
from ts_transformer.tests.support import terminal_contexts

TARGET = {"lat": 35.0, "lon": -78.0, "alt": 150.0, "V": 70.0, "psi": 0.3, "gamma": -0.0524, "m": 60000.0}


def _context():
    (context,) = terminal_contexts().values()
    return context


def test_the_landed_target_is_the_evaluations_own_threshold_point_of_that_runway():
    context = _context()
    target = landed_target(TARGET, context)
    assert (target["lat"], target["lon"]) == (context.threshold_lat, context.threshold_lon)
    assert target["alt"] == pytest.approx(111.86 + 15.0)                    # the published LTP + TCH
    assert target["psi"] == pytest.approx(math.radians(45.0))               # course 045° → math-ENU 45°
    # a course where compass and math-ENU differ: 225° → math −135°
    assert landed_target(TARGET, replace(context, runway_course_deg=225.0))["psi"] == pytest.approx(math.radians(-135.0))
    assert {key: target[key] for key in ("V", "gamma", "m")} == {key: TARGET[key] for key in ("V", "gamma", "m")}
    # a runway publishing no threshold height keeps the target's (the evaluation then grades the vertical indeterminate)
    bare = replace(context, benchmark="rnp_apch_lnav_vnav_baro", threshold_crossing_height_m=None, threshold_elevation_msl_m=None,
                   threshold_elevation_hae_m=None, lpv_course_width_m=None)
    assert landed_target(TARGET, bare)["alt"] == TARGET["alt"]


def test_a_graded_copy_reads_the_original_states_names_its_runway_and_passes_the_target_check(tmp_path):
    context = _context()
    original = tmp_path / "records" / "sample_0" / "KRDU"
    states = original / "X_states.json"
    record = {"source": {"id": "X", "arr_airport": "KRDU", "runway": "05R", "target_source": "runway_threshold",
                         "subject": "predicted"},
              "initial_state": dict(TARGET), "target_state": dict(TARGET), "final_time_s": 10.0,
              "states": [], "controls": [], "reference_file": "references/X_reference_eval.json",
              "states_ref": {"file": "X_states.json", "key": "predicted_states"}}
    directory = tmp_path / "grading" / "landed_runway" / "sample_0" / "KRDU"
    copy = graded_copy(record, context.runway, context, states, directory)
    assert copy["source"]["runway"] == context.runway
    assert copy["source"]["landedRunwayGrading"] == {"observedRunway": "05R", "landedRunway": context.runway}
    assert "reference_file" not in copy                                 # no path deviation to another runway's flight
    assert (directory / copy["states_ref"]["file"]).resolve() == states.resolve()
    assert copy["states_ref"]["key"] == "predicted_states" and record["source"]["runway"] == "05R"   # the original kept
    graded = record_from_dict({**copy, "states": [{"t": 10.0, **TARGET}]})
    _require_target_agrees_with_runway_data(graded, context)            # the evaluation's own 1 cm check holds


def _row(verdict, on_landed, *, other=False, recorded=True, outcome="landed", dt=0.0, fde=50.0, endpoint=40.0):
    return {"airport": "KRDU", "stratum": "straight-in", "verdict": verdict, "verdict_landed_runway": on_landed,
            "observed_runway": 0, "last_runway": 1 if other else 0, "recorded": recorded, "outcome": outcome,
            "regraded": recorded and other and outcome != "crossed_other_runway",
            "final_time_error_s": dt, "fde_m": fde, "arrival_endpoint_error_m": endpoint}


def test_the_landed_runway_pass_rate_counts_every_sentence_and_fde_splits_late_from_early():
    rows = [_row("pass", "pass", dt=-5.0), _row("fail", "pass", other=True, dt=12.0, fde=900.0),
            _row("fail", "fail", other=True, dt=3.0, fde=200.0),
            # crossed another runway than it pointed at: keeps its verdict; a timeout is not an arrival
            _row("pass", "pass", other=True, outcome="crossed_other_runway", dt=1.0),
            _row("fail", "fail", outcome="timeout", dt=90.0, fde=7000.0),
            _row("not recorded", "not recorded", recorded=False, outcome="dynamics_failure")]
    for key in ("fde_m", "arrival_endpoint_error_m", "final_time_error_s"):
        rows[5].pop(key)
    out = summarise(rows, timing=True)
    assert out["pass_rate_observed_runway"] == pytest.approx(2 / 6) and out["pass_rate_landed_runway"] == 0.5
    assert out["ending_on_another_runway"] == 3
    assert out["graded_again"] == {"sentences": 2, "verdicts": {"fail": 1, "pass": 1}}
    assert out["landed"] == 3 and out["late_share"] == pytest.approx(2 / 3)       # the timeout is not counted late
    assert out["fde_late_m"]["median"] == 550.0 and out["fde_early_m"]["median"] == 50.0
    assert out["arrival_endpoint_error_m"]["count"] == 3
    assert "fde_m" not in summarise(rows, timing=False)
