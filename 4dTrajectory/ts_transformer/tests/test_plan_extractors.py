"""The plan path's step 1: the skeleton in a flight's chart and the eight plan parameters.

The skeleton geometry is checked against a hand-built procedure (no document on disk
needed); the reader itself is checked on the KRDU documents where this machine has them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT, final_approach_fix
from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.dataset import build_series, truth_duration_s
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.outputs.plan.extractors import (
    DECEL_MARGIN_MPS,
    PLAN_PARAMETERS,
    extract_plan,
)
from ts_transformer.outputs.plan.skeleton import (
    RNP_HALF_WIDTH_M,
    ChartFix,
    RunwaySkeleton,
    runway_axes,
    runway_skeleton,
)
from ts_transformer.tests.support import AIRPORT, RUNWAY


def _series(n_flights: int = 3, **arrivals):
    config = TSConfig()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3, **arrivals)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def _fix(ident: str, role: str, d: float, xt: float, skeleton_axes, *, floor: float | None = None) -> ChartFix:
    """A fix placed by its runway-axes coordinates (the inverse of `runway_axes`)."""
    course, te, tn = skeleton_axes
    ue, un = math.cos(course), math.sin(course)
    e = te - d * ue + xt * un
    n = tn - d * un - xt * ue
    return ChartFix(ident=ident, role=role, e=e, n=n, d=d, xt=xt,
                    altitude_min_m=floor, altitude_max_m=None, speed_max_mps=None)


def _hand_skeleton(series) -> RunwaySkeleton:
    course = float(series.scenario.target.psi)
    te, tn = float(series.target_chart[0]), float(series.target_chart[1])
    axes = (course, te, tn)
    aim = float(series.frame.alt0)
    final = (
        _fix("IFX", "IF", 16_000.0, 0.0, axes, floor=aim + 900.0),
        _fix("FAF", "FAF", 10_000.0, 0.0, axes, floor=aim + 550.0),
        _fix("RWY", "MAPt", 0.0, 0.0, axes, floor=aim),
    )
    transition = (_fix("IAF", "IF", 27_000.0, 0.0, axes, floor=aim + 1500.0), final[0])
    return RunwaySkeleton(
        procedure_uid="TEST-R05LY-RW05L", course_rad=course,
        glidepath_tan=math.tan(math.radians(3.0)), aim_altitude_m=aim,
        target_e=te, target_n=tn, final=final, transitions=(transition,),
    )


def test_runway_axes_place_a_fix_where_it_was_put():
    series, _config = _series(1)
    skeleton = _hand_skeleton(series[0])
    d, xt = skeleton.axes(
        np.array([fix.e for fix in skeleton.final]), np.array([fix.n for fix in skeleton.final])
    )
    assert np.allclose(d, [16_000.0, 10_000.0, 0.0], atol=1e-6)
    assert np.allclose(xt, 0.0, atol=1e-6)
    # the runway axes are measured from the target, never the chart origin (C-11)
    d0, xt0 = runway_axes(
        np.array([skeleton.target_e]), np.array([skeleton.target_n]),
        course_rad=skeleton.course_rad, target_e=skeleton.target_e, target_n=skeleton.target_n,
    )
    assert float(d0[0]) == 0.0 and float(xt0[0]) == 0.0


def test_the_skeleton_answers_floor_speed_and_published_distance():
    series, _config = _series(1)
    skeleton = _hand_skeleton(series[0])
    aim = skeleton.aim_altitude_m
    # the floor that applies is the next fix AHEAD: inside the FAF it is the threshold's
    assert skeleton.floor_altitude_m(12_000.0) == pytest.approx(aim + 550.0)
    assert skeleton.floor_altitude_m(9_000.0) == pytest.approx(aim)
    assert skeleton.floor_altitude_m(20_000.0) == pytest.approx(aim + 900.0)
    assert skeleton.speed_limit_ahead_mps(20_000.0) is None
    # a point on the transition leg is at distance 0; one RNP box away is at the box edge
    iaf, ifx = skeleton.transitions[0]
    mid = np.array([(iaf.e + ifx.e) / 2]), np.array([(iaf.n + ifx.n) / 2])
    assert float(skeleton.distance_to_published_m(*mid)[0]) == pytest.approx(0.0, abs=1e-6)
    ue, un = math.cos(skeleton.course_rad), math.sin(skeleton.course_rad)
    beside = mid[0] + RNP_HALF_WIDTH_M * un, mid[1] - RNP_HALF_WIDTH_M * ue
    assert float(skeleton.distance_to_published_m(*beside)[0]) == pytest.approx(RNP_HALF_WIDTH_M, rel=1e-6)
    assert len(skeleton.published_legs()) == 2  # IAF→IF and the final's own IF→FAF; nothing past the FAF


def test_extractors_read_every_parameter_off_a_synthetic_track():
    series, config = _series(3)
    anchor = default_anchor(config)
    for item in series:
        skeleton = _hand_skeleton(item)
        labels = extract_plan(item, anchor, skeleton)
        values = labels.parameters()
        assert tuple(values) == PLAN_PARAMETERS
        assert values["T_s"] == pytest.approx(truth_duration_s(item, anchor))
        assert values["V_final_mps"] > 0.0
        assert values["side"] in (-1, 0, 1, None)
        assert labels.remaining_path_at_anchor_m > 0.0
        for name in ("d_decel_m", "d_join_m", "L_pre_m"):
            if values[name] is not None:
                assert 0.0 <= values[name] <= labels.remaining_path_at_anchor_m + 1e-6
        if values["d_join_m"] is not None and not labels.join_at_anchor:
            assert values["h_capture_m"] is not None and values["L_pre_m"] is not None
            assert labels.join_on_transition in (True, False)
        assert set(labels.ranges) == set(PLAN_PARAMETERS)
        low, _high = labels.ranges["V_final_mps"]
        assert 0.0 < low < item.scenario.aircraft.approach.reference_speed_ms  # the stall margin, under V_ref
        assert labels.ranges["T_s"][0] == pytest.approx(
            math.hypot(item.values[anchor, 0] - skeleton.target_e, item.values[anchor, 1] - skeleton.target_n)
            / item.scenario.aircraft.approach.max_speed_ms
        )
        # the deceleration point is where the speed first drops under V_final + the margin,
        # so the speed at the anchor is above it unless the point is the anchor itself
        if values["d_decel_m"] is not None and values["d_decel_m"] < labels.remaining_path_at_anchor_m:
            assert labels.ground_speed_at_anchor_mps >= values["V_final_mps"] + DECEL_MARGIN_MPS
        payload = labels.to_dict()
        assert payload["ranges"]["side"] == [-1.0, 1.0]


def test_a_join_before_the_anchor_censors_the_capture_side_and_pre_final_path():
    """59 % of KRDU val joins before the L−1 window: the extractor must say so rather than
    read the anchor height as the capture height and an empty base-leg window as
    straight-in (review of 2026-09-10)."""
    series, config = _series(3)
    anchor = default_anchor(config)
    skeleton = _hand_skeleton(series[0])
    labelled = [extract_plan(item, anchor, skeleton) for item in series]
    at_anchor = [labels for labels in labelled if labels.join_at_anchor]
    assert at_anchor, "the synthetic straight-in fixtures are established at L−1"
    for labels in at_anchor:
        assert labels.d_join_m == pytest.approx(labels.remaining_path_at_anchor_m)
        assert labels.side is None and labels.L_pre_m is None and labels.h_capture_m is None
        assert labels.join_on_transition is None
        assert labels.ranges["h_capture_m"] == (None, None)
    for labels in labelled:
        if not labels.join_at_anchor and labels.d_join_m is not None:
            assert labels.side in (-1, 0, 1) and labels.L_pre_m is not None


def test_a_track_that_never_slows_to_the_target_speed_has_no_deceleration_point():
    series, config = _series(2, entry_speed_ms=135.0, threshold_speed_ms=135.0)
    anchor = default_anchor(config)
    for item in series:
        labels = extract_plan(item, anchor, _hand_skeleton(item))
        assert labels.d_decel_m is None
        assert labels.V_final_mps > item.scenario.target.V + DECEL_MARGIN_MPS


@pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)
def test_the_real_krdu_skeleton_agrees_with_the_faf_read():
    series, _config = _series(1)
    skeleton = runway_skeleton(series[0])
    faf = final_approach_fix(AIRPORT, RUNWAY)
    assert skeleton.procedure_uid == faf.procedure_uid
    # the FAF's along-course distance from the vertical profile and from the fix geometry
    # agree to the document's coding precision
    assert skeleton.faf.d == pytest.approx(faf.distance_to_threshold_m, abs=25.0)
    assert abs(skeleton.faf.xt) < 50.0
    assert skeleton.final[-1].role == "MAPt"
    assert math.hypot(skeleton.final[-1].e - skeleton.target_e,
                      skeleton.final[-1].n - skeleton.target_n) < 15.0
    assert skeleton.transitions and all(branch[-1].ident == skeleton.final[0].ident
                                        for branch in skeleton.transitions)
