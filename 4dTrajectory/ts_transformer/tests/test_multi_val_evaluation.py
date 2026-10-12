"""Stage D's val windows graded by the evaluation package (`multi_val_evaluation`): the records' states, target and
source, the readout's cells, and the selection readout's ends (`multi_train.selection_readout`'s ``window_ends``) — on
the stage D campaign of one round of `test_multi_compare`. Every write root under tmp."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.records import load_record
from ts_transformer.experiments import multi_train
from ts_transformer.experiments import multi_val_evaluation as graded
from ts_transformer.experiments.multi_train import stage_d
from ts_transformer.tests.test_multi_compare import campaign  # noqa: F401
from ts_transformer.tests.test_post_window_loop import setup  # noqa: F401


def _geometry():
    """An airport of two candidates, its frame a plain shift of degrees per metre."""
    frame = SimpleNamespace(latlon_from_horizontal=lambda e, n: (30.0 + np.asarray(n) * 1e-5,
                                                                 -90.0 + np.asarray(e) * 1e-5))
    candidates = (SimpleNamespace(ident="09", elevation_m=5.0,
                                  vertical_path=SimpleNamespace(crossing_height_m=15.0, glidepath_deg=3.0)),
                  SimpleNamespace(ident="27", elevation_m=6.0,
                                  vertical_path=SimpleNamespace(crossing_height_m=16.0, glidepath_deg=3.2)))
    return SimpleNamespace(code="KXXX", frame=frame, candidates=candidates,
                           candidate_index=lambda ident: [c.ident for c in candidates].index(ident))


def _context(runway, desired):
    return SimpleNamespace(runway=runway, threshold_lat=30.1, threshold_lon=-89.9, runway_course_deg=270.0,
                           desired_threshold_altitude_msl_m=desired)


def test_a_compass_track_is_the_records_math_heading():
    """North is π/2, east 0, south −π/2; a track of 226° is −136°."""
    assert [graded.math_heading_rad(t) for t in (0.0, 90.0, 180.0)] == pytest.approx([math.pi / 2, 0.0, -math.pi / 2])
    assert graded.math_heading_rad(226.0) == pytest.approx(math.radians(-136.0))


def test_the_rows_are_the_records_timed_states():
    """Each 2 s row: its time from row 0, its position through the airport frame, its MSL height, its airspeed and
    path angle from the ground speed and vertical rate (no wind), its heading from the track, the mass given."""
    rows = np.array([[100.0, 200.0, 900.0, 90.0, 80.0, 0.0], [260.0, 200.0, 896.0, 90.0, 80.0, -4.0]])
    states = graded.record_states(rows, _geometry(), 2.0, 60000.0)
    assert [s["t"] for s in states] == [0.0, 2.0]
    assert (states[1]["lat"], states[1]["lon"]) == pytest.approx((30.002, -89.9974))
    assert states[1]["alt"] == 896.0 and states[1]["m"] == 60000.0
    assert states[1]["V"] == pytest.approx(math.hypot(80.0, -4.0))
    assert states[1]["gamma"] == pytest.approx(math.atan2(-4.0, 80.0)) and states[1]["psi"] == pytest.approx(0.0)


def test_the_target_is_the_contexts_threshold():
    """The context's position, course and LTP + TCH; the candidate's TCH where none is published; the candidate's
    glidepath; the observed target's speed and mass."""
    observed = {"V": 70.0, "m": 60000.0}
    target = graded.target_state(_context("27", 21.5), _geometry(), observed)
    assert (target["lat"], target["lon"], target["alt"]) == (30.1, -89.9, 21.5)
    assert target["psi"] == pytest.approx(math.pi) or target["psi"] == pytest.approx(-math.pi)
    assert target["gamma"] == pytest.approx(-math.radians(3.2)) and (target["V"], target["m"]) == (70.0, 60000.0)
    assert graded.target_state(_context("27", None), _geometry(), observed)["alt"] == 22.0


def _observed():
    state = {"lat": 30.0, "lon": -90.0, "alt": 900.0, "V": 80.0, "psi": 0.0, "gamma": 0.0, "m": 60000.0}
    return {"source": {"subject": "observed", "arr_airport": "KXXX", "runway": "09", "flight_key": "F_09",
                       "crossing_span": {"kind": "measured_bracket"}, "observed_threshold_event": {},
                       "source_integrity": {}, "mass_source": "x", "aircraft_type": "B738", "tch_source": "x",
                       "threshold_crossing_height_m": 15.0, "published_glidepath_deg": 3.0},
            "initial_state": state, "target_state": {**state, "V": 70.0}}


def test_an_aircrafts_record_is_a_predicted_record_on_its_landing_runway(tmp_path):
    """A landed aircraft's runway is its crossing's, one that did not land keeps its observed runway; the source is
    the observed one's without the observed-only fields, the subject predicted, its type the dynamics type; no
    controls; the final time the last row's; the evaluation package reads it."""
    rows = np.array([[0.0, 0.0, 900.0, 90.0, 80.0, 0.0], [160.0, 0.0, 890.0, 90.0, 80.0, -5.0]])
    flight = {"airport": "KXXX", "runway": "09", "typecode": "B738"}
    contexts = {("KXXX", "09"): _context("09", 20.0), ("KXXX", "27"): _context("27", 21.0)}
    landed = SimpleNamespace(states=rows, crossing={"runway_index": 1}, outcome="landed")
    record = graded.aircraft_record(landed, flight, _observed(), _geometry(), contexts, 2.0, {"window_place": 3})
    source = record["source"]
    assert source["runway"] == "27" and record["target_state"]["alt"] == 21.0
    assert source["subject"] == "predicted" and source["dynamics_typecode"] == "B738"
    assert source["two_tier_outcome"] == "landed" and source["window_place"] == 3
    assert (source["threshold_crossing_height_m"], source["published_glidepath_deg"]) == (16.0, 3.2)
    assert not set(graded.OBSERVED_ONLY) & set(source)
    assert record["controls"] == [] and record["final_time_s"] == 2.0 and len(record["states"]) == 2
    timeout = SimpleNamespace(states=rows, crossing=None, outcome="timeout")
    assert graded.aircraft_record(timeout, flight, _observed(), _geometry(), contexts, 2.0, {})["source"]["runway"] == "09"
    path = tmp_path / "r_eval.json"
    path.write_text(json.dumps(record))
    loaded = load_record(path)
    assert loaded.source["subject"] == "predicted" and loaded.final_time_s == 2.0


def test_the_readout_counts_outcomes_verdicts_and_the_observed_pairing():
    """Pooled and by airport, each by span with its ``all``: the outcomes, the verdicts and their share, the gates of
    the landed aircraft, the verdicts by outcome, and of the aircraft whose observed track passes the share whose
    flown track passes."""
    def row(airport, span, outcome, verdict, seen):
        return {"airport": airport, "span": span, "outcome": outcome, "verdict": verdict, "observed_verdict": seen,
                "lateral_result": verdict, "vertical_result": "pass", "speed_result": "pass"}
    rows = [row("KA", "300", "landed", "pass", "pass"), row("KA", "1200", "landed", "fail", "pass"),
            row("KB", "300", "timeout", "fail", "fail")]
    table = graded.readout_table(rows)
    assert list(table) == ["all", "KA", "KB"] and list(table["KA"]) == ["all", "300", "1200"]
    whole = table["all"]["all"]
    assert whole["aircraft"] == 3 and whole["verdicts"] == {"pass": 1, "fail": 2, "indeterminate": 0}
    assert whole["pass_share"] == pytest.approx(1 / 3) and whole["outcomes"] == {"landed": 2, "timeout": 1}
    assert whole["verdict_by_outcome"]["landed"] == {"pass": 1, "fail": 1, "indeterminate": 0}
    assert whole["landed_gates"]["lateral_result"] == {"pass": 1, "fail": 1}
    assert whole["observed_pass_share"] == pytest.approx(2 / 3) and whole["flown_pass_of_observed_pass"] == 0.5
    assert table["KB"]["300"]["flown_pass_of_observed_pass"] is None


def test_the_selection_readout_hands_back_each_windows_ends(campaign):  # noqa: F811
    """``window_ends``: each window's place in order and its commanded aircraft's ends, the readout unchanged: their
    outcomes are the lines' and their count the readout's aircraft."""
    out, context, settings, _ = campaign
    model = multi_train.round_model(context, settings, out, 0)
    windows = stage_d().selection(context, settings, "select")
    lines, ends = [], []
    readout = multi_train.selection_readout(model, context, windows, settings, None, stage=stage_d(), lines=lines,
                                            window_ends=ends)
    held = json.loads((out / "round_0" / "round.json").read_text())["selection_readout"]
    assert json.loads(json.dumps(readout)) == held
    assert [place for place, _ in ends] == list(range(len(windows)))
    assert [[e.outcome for e in mine] for _, mine in ends] == [line["outcomes"] for line in lines]
    assert sum(len(mine) for _, mine in ends) == sum(a["all"]["all"]["aircraft"] for a in held.values())
