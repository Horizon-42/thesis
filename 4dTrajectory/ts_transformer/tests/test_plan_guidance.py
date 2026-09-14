"""The guidance layer's by-construction properties (design §4): every command inside the
envelope, the route laid as the plan asked, an established flight flown to the threshold.
Synthetic KRDU arrivals, the real 05L skeleton where this machine has it."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.config import CONTROL_THRUST_FRACTION, TSConfig, default_anchor
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.geometry.dubins import chart_from_axes_np, dubins_csc
from ts_transformer.geometry.flyability import flyability_summary, required_controls
from ts_transformer.inference.forecast import cut_at_threshold_crossing
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.dynamics.hooks import RolloutStateView
from ts_transformer.outputs.plan.extractors import extract_plan, extract_waypoints
from ts_transformer.outputs.plan.guidance.controller import (
    BANK_MAX_RAD,
    LOAD_FACTOR_MAX,
    LOAD_FACTOR_MIN,
    PlanGuidance,
    PlanToFly,
)
from ts_transformer.outputs.plan.forecast import fly_plans, guidance_config, n_segments_for
from ts_transformer.outputs.plan.guidance.route import (
    HOLD_TOLERANCE_M,
    LOOP_SWEEP_RAD,
    INTERCEPT_MAX_RAD,
    KIND_DIRECT,
    KIND_DOWNWIND,
    KIND_FINAL_ONLY,
    KIND_STRETCHED,
    KIND_WAYPOINTS,
    OVERRUN_M,
    PATH_STEP_M,
    STRETCH_TOLERANCE_M,
    build_route,
    route_time_s,
    speed_schedule_mps,
)
import ts_transformer.outputs.plan.guidance.route as route_module
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.tests.support import AIRPORT, RUNWAY

pytestmark = pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)


def _cohort(n_flights: int = 3):
    config = TSConfig()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    skeleton = runway_skeleton(series[0])
    anchor = default_anchor(config)
    labels = [extract_plan(item, anchor, skeleton) for item in series]
    return series, config, skeleton, anchor, labels


def test_the_route_lays_the_plans_length_and_ends_past_the_threshold():
    series, _config, skeleton, _anchor, _labels = _cohort(1)
    item = series[0]
    e0, n0 = float(item.values[0, IDX["e"]]), float(item.values[0, IDX["n"]])
    heading = math.atan2(item.values[0, IDX["ndot"]], item.values[0, IDX["edot"]])
    direct = build_route(e0, n0, heading, 80.0, d_join_m=9_000.0, side=0, pre_final_m=0.0, skeleton=skeleton)
    assert direct.kind == KIND_DIRECT
    assert direct.points[0].tolist() == pytest.approx([e0, n0])
    assert direct.threshold_arc_m == pytest.approx(direct.length_m - OVERRUN_M)
    d_end, xt_end = skeleton.axes(direct.points[-1:, 0], direct.points[-1:, 1])
    assert float(d_end[0]) == pytest.approx(-OVERRUN_M, abs=1.0) and abs(float(xt_end[0])) < 1.0
    # a longer pre-final path than the shortest turn-straight-turn is laid as a dog-leg
    stretched = build_route(
        e0, n0, heading, 80.0, d_join_m=9_000.0, side=1, pre_final_m=direct.pre_final_m + 8_000.0,
        skeleton=skeleton,
    )
    assert stretched.kind == KIND_STRETCHED and stretched.stretch_offset_m > 0.0
    assert stretched.pre_final_m == pytest.approx(direct.pre_final_m + 8_000.0, abs=100.0)
    assert abs(stretched.shortfall_m) <= 100.0
    # an established flight has no pre-final path: the final from its own distance to go
    on_e, on_n = chart_from_axes_np(np.array([12_000.0]), np.array([0.0]), skeleton.course_rad)
    final_only = build_route(
        float(on_e[0]) + skeleton.target_e, float(on_n[0]) + skeleton.target_n, skeleton.course_rad, 80.0,
        d_join_m=12_000.0, side=0, pre_final_m=0.0, skeleton=skeleton, join_at_anchor=True,
    )
    assert final_only.kind == KIND_FINAL_ONLY and final_only.pre_final_m == 0.0
    assert final_only.threshold_arc_m == pytest.approx(12_000.0, abs=51.0)


def test_the_speed_schedule_and_the_route_time_are_consistent():
    remaining = np.array([20_000.0, 10_000.0, 5_000.0, 0.0])
    schedule = speed_schedule_mps(remaining, v_mid=100.0, d_decel_m=10_000.0, v_final=70.0)
    # held to the deceleration point, then V² = V_mid² − 2·a·Δs down to V_final, held
    assert schedule.tolist() == pytest.approx([100.0, 100.0, math.sqrt(5000.0), 70.0])
    held = speed_schedule_mps(remaining, v_mid=100.0, d_decel_m=None, v_final=70.0)
    assert held.tolist() == pytest.approx([100.0] * 4)
    series, _config, skeleton, _anchor, _labels = _cohort(1)
    item = series[0]
    route = build_route(
        float(item.values[0, IDX["e"]]), float(item.values[0, IDX["n"]]),
        math.atan2(item.values[0, IDX["ndot"]], item.values[0, IDX["edot"]]), 80.0,
        d_join_m=9_000.0, side=0, pre_final_m=0.0, skeleton=skeleton,
    )
    fast = route_time_s(route, v_mid=100.0, d_decel_m=None, v_final=100.0)
    slow = route_time_s(route, v_mid=80.0, d_decel_m=None, v_final=80.0)
    assert fast == pytest.approx(route.threshold_arc_m / 100.0, rel=1e-6)
    assert slow > fast


def test_every_command_the_guidance_flies_is_inside_the_envelope():
    series, config, skeleton, anchor, labels = _cohort(3)
    forecasts, routes, times = fly_plans(series, anchor, labels, [skeleton] * 3, config)
    lag = guidance_config(config)
    for item, forecast, route, plan_time in zip(series, forecasts, routes, times, strict=True):
        max_thrust = float(item.scenario.aircraft.engine.max_thrust_total_n)
        thrust = forecast.controls[:, 0] / max_thrust
        n_segments = n_segments_for(max(lab.T_s for lab in labels))
        assert forecast.controls.shape == (n_segments, 3)
        assert np.all(thrust >= -0.2 - 1e-9) and np.all(thrust <= 1.0 + 1e-9)
        # the tracker banks to its own cap; the corridor barrier on the final may ask for
        # more, inside the grader's envelope
        assert np.all(np.abs(forecast.controls[:, 1]) <= math.pi / 4 + 1e-9)
        assert np.mean(np.abs(forecast.controls[:, 1]) > BANK_MAX_RAD + 1e-9) < 0.2
        assert np.all(forecast.controls[:, 2] >= LOAD_FACTOR_MIN - 1e-9)
        assert np.all(forecast.controls[:, 2] <= LOAD_FACTOR_MAX + 1e-9)
        assert forecast.command_hook == "plan-guidance"
        assert forecast.command_hook_diagnostics["steps"] == n_segments
        assert forecast.final_time_s == pytest.approx(forecast.segment_durations_s.sum())
        assert forecast.horizon_mode == lag.horizon_mode
        assert route.length_m > 0.0 and math.isfinite(plan_time)


def test_an_established_flight_is_flown_to_the_threshold_flyably():
    series, config, skeleton, anchor, labels = _cohort(3)
    established = [(item, lab) for item, lab in zip(series, labels, strict=True) if lab.join_at_anchor]
    assert established, "the synthetic fixtures include a flight established at L−1"
    items = [item for item, _lab in established]
    labs = [lab for _item, lab in established]
    forecasts, routes, _times = fly_plans(items, anchor, labs, [skeleton] * len(items), config)
    for item, forecast, route in zip(items, forecasts, routes, strict=True):
        assert route.kind == KIND_FINAL_ONLY
        cut = cut_at_threshold_crossing(forecast, item)
        assert cut.truncated_at_threshold, "the guidance must reach the threshold on the final"
        rows = [
            {"t": float(t), "lat": g[0], "lon": g[1], "alt": g[2], "V": g[3], "psi": g[4], "gamma": g[5], "m": g[6]}
            for t, g in zip(np.cumsum(cut.sample_durations_s), cut.geodetic_values, strict=True)
        ]
        controls = required_controls(rows, item.scenario.aircraft, aero=item.scenario.aero)
        summary = flyability_summary(controls, aircraft_code=str(item.scenario.aircraft.code))
        assert summary["violations"].get("stall", 0) == 0, summary["violations"]


def _pose(skeleton, d_m: float, xt_m: float) -> tuple[float, float]:
    e, n = chart_from_axes_np(np.array([d_m]), np.array([xt_m]), skeleton.course_rad)
    return float(e[0]) + skeleton.target_e, float(n[0]) + skeleton.target_n


def _chord_heading(points: np.ndarray, index: int) -> float:
    a, b = points[index - 1], points[index]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def test_the_join_is_aligned_with_the_course_from_every_anchor_pose():
    """Whatever the anchor's bearing and heading, and whether the route is the shortest path,
    a held heading or a dog-leg, the aircraft arrives at the join within the intercept
    steps of the runway course — the intercept is always taken from the COURSE, never
    from a heading a previous choice bent (review, 2026-09-10)."""
    _series, _config, skeleton, _anchor, _labels = _cohort(1)
    join_d = 9_000.0
    widest = INTERCEPT_MAX_RAD
    # the last chord of a sampled arc is off the arc's end tangent by half its subtended
    # angle: PATH_STEP_M over the 80 m/s turn radius
    chord_slack = 0.5 * PATH_STEP_M / (80.0 ** 2 / (9.81 * math.tan(math.radians(20.0)))) + 1e-6
    kinds = set()
    for bearing_deg in range(0, 360, 45):
        for heading_deg in range(0, 360, 90):
            bearing = math.radians(bearing_deg)
            e0, n0 = _pose(skeleton, join_d + 15_000.0 * math.cos(bearing), 15_000.0 * math.sin(bearing))
            heading = skeleton.course_rad + math.radians(heading_deg)
            direct = build_route(e0, n0, heading, 80.0, d_join_m=join_d, side=0, pre_final_m=0.0, skeleton=skeleton)
            for extra in (0.0, 6_000.0, 14_000.0):
                requested = direct.pre_final_m + extra
                route = build_route(
                    e0, n0, heading, 80.0, d_join_m=join_d, side=0, pre_final_m=requested, skeleton=skeleton,
                )
                kinds.add(route.kind)
                at_join = _chord_heading(route.points, route.join_index - 1)
                assert abs(route.intercept_rad) <= widest + 1e-9
                assert abs(_wrap(at_join - skeleton.course_rad)) <= widest + chord_slack, (
                    bearing_deg, heading_deg, extra, route.kind, math.degrees(_wrap(at_join - skeleton.course_rad)),
                )
                d_join, xt_join = skeleton.axes(route.points[route.join_index - 1 : route.join_index, 0],
                                                route.points[route.join_index - 1 : route.join_index, 1])
                assert float(d_join[0]) == pytest.approx(join_d, abs=1.0) and abs(float(xt_join[0])) < 1.0
                # the plan's length is laid as closely as a hold or a dog-leg can, never
                # exceeded by more than the tolerance, and the gap is reported — never a
                # silent extra
                assert route.shortfall_m == pytest.approx(requested - route.pre_final_m)
                assert route.shortfall_m >= -HOLD_TOLERANCE_M
                if route.kind in (KIND_DOWNWIND, KIND_STRETCHED):
                    assert abs(route.shortfall_m) < requested - direct.pre_final_m
                else:
                    # the shortest path, or the aligned join where the plan's length affords
                    # it — never longer than the plan asked, beyond the tolerance
                    assert route.kind == KIND_DIRECT
                    assert route.pre_final_m <= max(requested, direct.pre_final_m) + STRETCH_TOLERANCE_M
    assert {KIND_DIRECT, KIND_DOWNWIND} <= kinds, kinds


def test_a_dog_leg_that_overshoots_the_plans_length_is_not_taken(monkeypatch):
    """The dog-leg's laid length is not monotone in its offset (a Dubins type switch jumps
    it), so a via that lays MORE than the plan by over the tolerance is never taken —
    the shortest path is flown and the whole gap is reported (review, 2026-09-10);
    within reach, the dog-leg lays the plan inside the tolerance."""
    _series, _config, skeleton, _anchor, _labels = _cohort(1)
    e0, n0 = _pose(skeleton, 20_000.0, 6_000.0)
    direct = build_route(e0, n0, skeleton.course_rad, 80.0, d_join_m=9_000.0, side=0, pre_final_m=0.0, skeleton=skeleton)
    requested = direct.pre_final_m + 20_000.0
    via_path = route_module._via_path
    # no held heading lays it, and every via overshoots the plan by twice the tolerance
    monkeypatch.setattr(route_module, "_held_heading_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(route_module, "_via_path", lambda *args, **kwargs: (
        np.array([[0.0, 0.0], [requested + 2.0 * HOLD_TOLERANCE_M, 0.0]]), skeleton.course_rad,
    ))
    route = build_route(e0, n0, skeleton.course_rad, 80.0, d_join_m=9_000.0, side=1, pre_final_m=requested, skeleton=skeleton)
    assert route.kind == KIND_DIRECT and route.stretch_offset_m == 0.0
    assert direct.pre_final_m - STRETCH_TOLERANCE_M <= route.pre_final_m <= requested + STRETCH_TOLERANCE_M
    assert route.shortfall_m == pytest.approx(requested - route.pre_final_m)
    monkeypatch.setattr(route_module, "_via_path", via_path)
    laid = build_route(e0, n0, skeleton.course_rad, 80.0, d_join_m=9_000.0, side=1, pre_final_m=requested, skeleton=skeleton)
    assert laid.kind == KIND_STRETCHED and laid.stretch_offset_m > 0.0
    assert abs(laid.shortfall_m) <= HOLD_TOLERANCE_M


def test_the_bank_turns_the_aircraft_toward_its_route():
    """Positive bank is a CCW (left) turn in the math-ENU chart: an aircraft RIGHT of its
    route (+xt) banks positive, one LEFT of it banks negative, by the same amount."""
    series, config, skeleton, anchor, _labels = _cohort(1)
    row = dynamics_arrays(series[0], anchor, parameterization=CONTROL_THRUST_FRACTION)
    dynamics = {name: torch.from_numpy(np.stack([row[name]] * 2)) for name in row}
    course = skeleton.course_rad
    e_on, n_on = _pose(skeleton, 12_000.0, 0.0)
    route = build_route(
        e_on, n_on, course, 80.0, d_join_m=12_000.0, side=0, pre_final_m=0.0, skeleton=skeleton, join_at_anchor=True,
    )
    height = 12_000.0 * skeleton.glidepath_tan
    plan = PlanToFly(80.0, None, 75.0, height)
    hook = PlanGuidance(guidance_config(config), dynamics, [route, route], [plan, plan], np.array([height, height]))
    mass = float(row["initial_state"][6])
    chart = torch.tensor([
        [*_pose(skeleton, 12_000.0, xt), height, 80.0 * math.cos(course), 80.0 * math.sin(course),
         -80.0 * skeleton.glidepath_tan, mass]
        for xt in (400.0, -400.0)
    ], dtype=torch.float64)
    state = RolloutStateView(
        chart=chart, actuators=torch.tensor([[0.5, 0.0, 1.0]] * 2, dtype=torch.float64),
        duration_s=torch.full((2,), 3.0, dtype=torch.float64), remaining_s=torch.full((2,), 200.0, dtype=torch.float64),
    )
    command = hook(state, torch.zeros((2, 3), dtype=torch.float64), 0)
    right, left = float(command[0, 1]), float(command[1, 1])
    assert right > 0.0 and left < 0.0
    assert right == pytest.approx(-left, rel=1e-6)
    assert abs(right) <= BANK_MAX_RAD + 1e-9
    per_flight = hook.per_flight_diagnostics()
    assert per_flight["hook_plan_route_cross_track_m"].tolist() == pytest.approx([400.0, 400.0], abs=PATH_STEP_M)
    assert "hook_plan_thrust_idle_steps" in per_flight


def test_a_turn_straight_turn_path_with_two_radii_ends_on_its_pose_at_the_end_radius():
    """The route's last turn is flown at the join speed's radius, its first at the anchor's:
    the path still lands on the end pose, and its last arc has the end radius."""
    p0, h0 = np.array([0.0, 0.0]), math.radians(160.0)
    p1, h1 = np.array([9_000.0, -12_000.0]), 0.0
    for r0, r1 in ((3_600.0, 1_400.0), (1_400.0, 3_600.0), (2_000.0, 2_000.0)):
        path = dubins_csc(p0, h0, p1, h1, r0, end_radius=r1)
        assert path is not None
        assert path[0].tolist() == pytest.approx(p0.tolist()) and path[-1].tolist() == pytest.approx(p1.tolist(), abs=1e-6)
        assert _wrap(_chord_heading(path, 1) - h0) == pytest.approx(0.0, abs=0.5 * 50.0 / r0 + 1e-9)
        assert _wrap(_chord_heading(path, len(path) - 1) - h1) == pytest.approx(0.0, abs=0.5 * 50.0 / r1 + 1e-9)
        # the last arc's curvature: heading change per metre over its final chords
        a, b, c = path[-3], path[-2], path[-1]
        turn = abs(_wrap(math.atan2(*(c - b)[::-1]) - math.atan2(*(b - a)[::-1])))
        span = 0.5 * (math.hypot(*(b - a)) + math.hypot(*(c - b)))
        assert turn / span == pytest.approx(1.0 / r1, rel=0.05)
    assert dubins_csc(p0, h0, p1, h1, 2_000.0, end_radius=2_000.0).tolist() == dubins_csc(p0, h0, p1, h1, 2_000.0).tolist()


def test_a_pose_beside_the_centreline_heading_in_takes_the_chord_not_a_loop():
    """At the anchor's turn radius the aligned join from a pose 1 km beside the centreline is
    a full circle; the route is the straight chord onto the join instead, its intercept
    inside the alignment limit, and its length is the chord's (2026-09-10)."""
    _series, _config, skeleton, _anchor, _labels = _cohort(1)
    e0, n0 = _pose(skeleton, 11_000.0, 500.0)
    route = build_route(e0, n0, skeleton.course_rad, 100.0, d_join_m=9_000.0, side=0, pre_final_m=0.0, skeleton=skeleton)
    chord = math.hypot(2_000.0, 500.0)
    assert route.kind == KIND_DIRECT
    assert route.pre_final_m == pytest.approx(chord, abs=1.0)
    assert abs(route.intercept_rad) == pytest.approx(math.atan2(500.0, 2_000.0), abs=1e-6)
    # the primitive itself refuses the loop under the sweep limit and lays it without one
    p0, join = np.array([e0, n0]), np.array(_pose(skeleton, 9_000.0, 0.0))
    radius = 100.0 ** 2 / (9.81 * math.tan(math.radians(20.0)))
    looped = dubins_csc(p0, skeleton.course_rad, join, skeleton.course_rad, radius)
    assert looped is not None and float(np.sum(np.hypot(*np.diff(looped, axis=0).T))) > chord + 2.0 * math.pi * radius * 0.8
    assert dubins_csc(p0, skeleton.course_rad, join, skeleton.course_rad, radius, max_sweep_rad=LOOP_SWEEP_RAD) is None
    # a join a few hundred metres ahead from a pose beside the centreline heading 40° into
    # it: a teardrop at any radius — the route converges onto the centreline and flies
    # the final, the plan's pre-final length reported whole
    e1, n1 = _pose(skeleton, 14_500.0, 500.0)
    near = build_route(
        e1, n1, skeleton.course_rad + math.radians(40.0), 90.0, d_join_m=14_300.0, side=0, pre_final_m=300.0, skeleton=skeleton,
    )
    assert near.kind == KIND_FINAL_ONLY and near.pre_final_m == 0.0 and near.shortfall_m == 300.0
    assert near.length_m < 14_500.0 + OVERRUN_M + 600.0
    assert abs(near.intercept_rad) <= INTERCEPT_MAX_RAD


def test_a_waypoints_route_rounds_the_fixes_and_reads_back_as_them():
    """The fixed-K representation round-trips: a route laid through two fly-by fixes runs
    the legs between them (rounding each corner), the extractor reads the same two fixes
    back, and the route still ends on the join pose within the alignment limit and runs
    down the final."""
    _series, _config, skeleton, _anchor, _labels = _cohort(1)
    join_d = 9_000.0
    e0, n0 = _pose(skeleton, 30_000.0, -8_000.0)
    # a 116° turn at the first fix, 9° at the second (onto the course), the anchor 30° off
    # the first leg — no reversal, which the extractor would split into two fixes
    fixes = (_pose(skeleton, 33_000.0, 2_000.0), _pose(skeleton, 20_000.0, 0.0))
    heading = math.atan2(fixes[0][1] - n0, fixes[0][0] - e0) + math.radians(30.0)
    route = build_route(
        e0, n0, heading, 90.0, d_join_m=join_d, side=0, pre_final_m=30_000.0, skeleton=skeleton, waypoints=fixes,
    )
    assert route.kind == KIND_WAYPOINTS
    # the leg between the fixes is flown: its midpoint lies on the route
    mid = 0.5 * (np.array(fixes[0]) + np.array(fixes[1]))
    assert float(np.min(np.hypot(route.points[:, 0] - mid[0], route.points[:, 1] - mid[1]))) < PATH_STEP_M
    d_join, xt_join = skeleton.axes(route.points[route.join_index - 1 : route.join_index, 0],
                                    route.points[route.join_index - 1 : route.join_index, 1])
    assert float(d_join[0]) == pytest.approx(join_d, abs=1.0) and abs(float(xt_join[0])) < 1.0
    assert abs(route.intercept_rad) <= INTERCEPT_MAX_RAD + 1e-9
    assert route.shortfall_m == pytest.approx(30_000.0 - route.pre_final_m)
    # read back at 50 m rows one second apart (the 90 m/s radius turns at 1.6°/s): the
    # fixes laid are among the route's turns (the anchor's own turn onto the first leg
    # and the turn onto the join are turns of the route too)
    pre = route.points[: route.join_index]
    found, _dropped = extract_waypoints(pre[:, 0], pre[:, 1], 1.0, max_waypoints=4)
    for truth in fixes:
        assert min(math.hypot(fix[0] - truth[0], fix[1] - truth[1]) for fix in found) < 400.0
    # no fixes: the plan route as before
    plain = build_route(e0, n0, heading, 90.0, d_join_m=join_d, side=0, pre_final_m=30_000.0, skeleton=skeleton, waypoints=())
    assert plain.kind != KIND_WAYPOINTS


def test_the_speed_schedule_runs_through_the_fix_speeds():
    """Under the waypoints route the schedule interpolates the anchor's and the fixes'
    speeds, holds the first before it, and past the last fix decelerates at the plan's rate
    to V_final (from the deceleration point where that is earlier)."""
    points = ((30_000.0, 100.0), (20_000.0, 90.0), (15_000.0, 80.0))
    remaining = np.array([35_000.0, 30_000.0, 25_000.0, 20_000.0, 15_000.0, 10_000.0, 5_000.0, 0.0])
    schedule = speed_schedule_mps(remaining, v_mid=100.0, d_decel_m=8_000.0, v_final=70.0, points=points)
    # held before the first point; through the points; 80 held to the deceleration point at
    # 8 km, then V² = 80² − 2·0.5·Δs floored at 70
    assert schedule.tolist() == pytest.approx([100.0, 100.0, 95.0, 90.0, 80.0, 80.0, 70.0, 70.0])
    # a last fix already below V_final holds its own speed
    low = speed_schedule_mps(remaining, v_mid=100.0, d_decel_m=8_000.0, v_final=70.0, points=((30_000.0, 100.0), (15_000.0, 65.0)))
    assert low[-1] == pytest.approx(65.0)
    # no points: the plan's own law, unchanged
    plain = speed_schedule_mps(remaining, v_mid=100.0, d_decel_m=8_000.0, v_final=70.0)
    assert plain[0] == 100.0 and plain[-1] == 70.0


def test_a_fix_is_rounded_at_its_own_speed():
    """The fly-by radius at a fix is the speed the plan carries there: a slow fix is
    rounded tighter and the route passes closer to it."""
    _series, _config, skeleton, _anchor, _labels = _cohort(1)
    e0, n0 = _pose(skeleton, 30_000.0, -8_000.0)
    fixes = (_pose(skeleton, 33_000.0, 2_000.0), _pose(skeleton, 20_000.0, 0.0))
    heading = math.atan2(fixes[0][1] - n0, fixes[0][0] - e0) + math.radians(30.0)
    closest = {}
    for speed in (60.0, 120.0):
        route = build_route(
            e0, n0, heading, 90.0, d_join_m=9_000.0, side=0, pre_final_m=30_000.0, skeleton=skeleton,
            waypoints=fixes, waypoint_speeds=((25_000.0, speed), (10_000.0, speed)),
        )
        assert route.kind == KIND_WAYPOINTS
        closest[speed] = float(np.min(np.hypot(route.points[:, 0] - fixes[0][0], route.points[:, 1] - fixes[0][1])))
    assert closest[60.0] < closest[120.0]
