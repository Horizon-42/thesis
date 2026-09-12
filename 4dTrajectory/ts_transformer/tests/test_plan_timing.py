"""The time closure (design §4.6; v5.4, §9 step 4): the speed lever bisected between the
floor and the maximum, the path lever where the floor cannot absorb, X where neither can —
and the assignment flown by the head's lockstep on the synthetic KRDU cohort."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.config import CHECKPOINT_SELECTION_OBJECTIVE, PREDICTION_PLAN, TSConfig, default_anchor
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import cut_at_threshold_crossing
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import (
    LegOrder,
    Instruction,
    leg_route,
    lockstep_states,
    plan_to_fly,
    ROUTE_WAYPOINTS,
)
from ts_transformer.outputs.plan.guidance.controller import PlanToFly
from ts_transformer.outputs.plan.guidance.route import OVERRUN_M, Route
from ts_transformer.outputs.plan.guidance.timing import (
    MAX_STRETCH_M,
    MIN_STRETCH_LAID_M,
    STRETCH_PROBES,
    TIME_TOLERANCE_S,
    close_speed,
    close_time,
    held_speed_mps,
    plan_time_s,
    reanchored,
    scaled,
    stall_floor_mps,
)
from ts_transformer.outputs.plan.labels import SPEED_MAX_MPS, PlanOrder
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.outputs.plan.strategy import NO_ASSIGNMENT, Assignment, assigned_order, rolled_predictions_lockstep
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance
from ts_transformer.training.train import load_checkpoint, train

pytestmark = pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)

TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=2.0,
            device="cpu", horizon_mode="normalized", checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE,
            epochs=1, patience=1, batch_size=8)


def _cohort(n_flights: int = 3):
    config = TSConfig()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config, runway_skeleton(series[0]), default_anchor(config)


def _first_route(series, config, skeleton, anchor):
    labels = extract_plan(series, anchor, skeleton)
    plan, _clamped = plan_to_fly(labels, float(series.values[anchor, 2]), skeleton, route=ROUTE_WAYPOINTS)
    states = lockstep_states([series], [anchor], [skeleton], [labels.remaining_path_at_anchor_m])
    state = states[0]
    route, leg_plan, _join = leg_route(state.current, None, labels.d_join_m, skeleton, plan)
    return state, labels, plan, route, leg_plan


def test_the_speed_lever_meets_the_time_between_the_floor_and_the_maximum():
    series, config, skeleton, anchor = _cohort(1)
    state, labels, plan, route, leg_plan = _first_route(series[0], config, skeleton, anchor)
    floor = stall_floor_mps(state.current.row)
    assert 40.0 < floor < held_speed_mps(leg_plan) < SPEED_MAX_MPS
    t0 = plan_time_s(route, leg_plan)
    # from a point the aircraft has reached, the time left is less
    assert plan_time_s(route, leg_plan, start=len(route.points) // 2) < t0
    assert plan_time_s(route, leg_plan, start=0) == t0
    # scaled: the anchor's own speed point and V_final untouched; on a closing (one point,
    # the anchor's) a new held speed is added, reached at the deceleration rate
    half = scaled(leg_plan, 0.5)
    assert half.v_mid_mps == pytest.approx(0.5 * leg_plan.v_mid_mps) and half.v_final_mps == leg_plan.v_final_mps
    assert half.speed_points[0] == leg_plan.speed_points[0]
    assert len(leg_plan.speed_points) == 1 and len(half.speed_points) == 2
    assert half.speed_points[1][1] == pytest.approx(0.5 * leg_plan.speed_points[0][1])
    assert 0.0 <= half.speed_points[1][0] < half.speed_points[0][0]
    with_fix = scaled(PlanToFly(90.0, 8_000.0, 70.0, 600.0, ((20_000.0, 110.0), (12_000.0, 100.0))), 0.8)
    assert with_fix.speed_points == ((20_000.0, 110.0), (12_000.0, 80.0))
    # a time inside the lever's reach is met to the tolerance, slower and faster alike
    for target in (t0 + 20.0, t0 - 10.0):
        closed, factor = close_speed(route, leg_plan, target, v_floor_mps=floor)
        assert abs(plan_time_s(route, closed) - target) <= TIME_TOLERANCE_S
        assert (factor < 1.0) == (target > t0)
    # beyond the floor the lever stops AT the floor, beyond the maximum at the maximum
    slowest, f_lo = close_speed(route, leg_plan, t0 + 10_000.0, v_floor_mps=floor)
    assert held_speed_mps(slowest) == pytest.approx(max(floor, leg_plan.v_final_mps)) and plan_time_s(route, slowest) < t0 + 10_000.0
    fastest, f_hi = close_speed(route, leg_plan, 1.0, v_floor_mps=floor)
    assert held_speed_mps(fastest) == pytest.approx(SPEED_MAX_MPS) and f_hi > 1.0 > f_lo


def test_the_path_lever_absorbs_what_the_floor_cannot_and_x_is_what_neither_can():
    series, config, skeleton, anchor = _cohort(1)
    # an EARLY anchor: at L-1 the synthetic flight is ~5 km from its join at 113 m/s (a 3.6 km
    # turn radius), where no hold or dog-leg fits and the lever is rightly exhausted
    anchor = min(anchor, 12)
    state, labels, plan, route, leg_plan = _first_route(series[0], config, skeleton, anchor)
    floor = stall_floor_mps(state.current.row)
    t0 = plan_time_s(route, leg_plan)
    slowest = plan_time_s(route, close_speed(route, leg_plan, t0 + 10_000.0, v_floor_mps=floor)[0])

    def lay(extra_m: float):
        r, p, _j = leg_route(state.current, None, labels.d_join_m, skeleton, plan, stretch_m=extra_m)
        return r, p

    # the lever itself: a closing asked for more path gets a hold or a dog-leg from the plain builder
    longer, _p, _j = leg_route(state.current, None, labels.d_join_m, skeleton, plan, stretch_m=6_000.0)
    assert longer.length_m > route.length_m + 1_000.0 and longer.kind != "waypoints"

    # a delay the floor can hold: no stretch
    closure, r1, p1, base1 = close_time(route, leg_plan, t0 + 15.0, v_floor_mps=floor, lay=lay)
    assert closure.stretch_m == 0.0 and r1 is route and abs(closure.unabsorbed_s) <= TIME_TOLERANCE_S and base1 is leg_plan
    # a delay past the floor: path added, the closed time nearer the target than the floor alone
    target = slowest + 120.0
    closure, r2, p2, base2 = close_time(route, leg_plan, target, v_floor_mps=floor, lay=lay)
    assert closure.stretch_m > 0.0 and r2 is not route and r2.length_m > route.length_m
    assert closure.stretch_m == pytest.approx(r2.length_m - route.length_m) and closure.stretch_requested_m > 0.0
    assert abs(closure.unabsorbed_s) < target - slowest
    assert base2 is not leg_plan and held_speed_mps(p2) <= held_speed_mps(base2) + 1e-9
    # an advance the maximum cannot make: X negative, nothing stretched
    closure, r3, _p3, _b3 = close_time(route, leg_plan, max(t0 - 600.0, 1.0), v_floor_mps=floor, lay=lay)
    assert closure.unabsorbed_s < -TIME_TOLERANCE_S and closure.stretch_m == 0.0 and r3 is route
    assert MAX_STRETCH_M > 0.0


def test_an_assigned_order_replaces_the_time_and_the_join_and_nothing_else():
    series, config, skeleton, anchor = _cohort(1)
    state, labels, plan, route, leg_plan = _first_route(series[0], config, skeleton, anchor)
    fix = Instruction(1_000.0, 2_000.0, 0.3, 80.0, 10_000.0, 500.0)
    order = PlanOrder(300.0, 90.0, 8_000.0, 70.0, 600.0, 10_000.0, 25_000.0, fix, 0.1)
    same = assigned_order(order, Assignment(), state)
    assert same is order
    timed = assigned_order(order, Assignment(arrival_time_s=state.current.time_s + 410.0), state)
    assert timed.T_s == pytest.approx(410.0) and timed.d_join_m == order.d_join_m and timed.instruction is fix
    joined = assigned_order(order, Assignment(d_join_m=12_000.0), state)
    assert joined.d_join_m == 12_000.0 and joined.T_s == order.T_s
    clamped = assigned_order(order, Assignment(arrival_time_s=state.current.time_s + 1.0, d_join_m=40_000.0), state)
    assert clamped.T_s == 30.0 and clamped.d_join_m == order.remaining_m
    assert set(clamped.clamped) == {"assigned_T_s", "assigned_d_join_m"}


def test_the_lockstep_flies_an_assigned_time_and_reports_x(tmp_path):
    """A plan head asked to arrive at the truth's own time arrives nearer it than unassigned;
    asked far later it stretches or reports X > 0; asked far earlier it reports X < 0."""
    config = TSConfig(prediction_output=PREDICTION_PLAN, **TINY)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == 4, report.format()
    skeleton = runway_skeleton(series[0])
    train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    anchor, device = default_anchor(loaded), torch.device("cpu")
    item = series[0]
    labels = extract_plan(item, anchor, skeleton)
    truth_arrival = float(item.times[anchor]) + labels.T_s
    common = (model, [item], loaded, normalizer, [anchor], device, [skeleton])
    free, = rolled_predictions_lockstep(*common)
    on_time, = rolled_predictions_lockstep(*common, assignments=[Assignment(arrival_time_s=truth_arrival)])
    late, = rolled_predictions_lockstep(*common, assignments=[Assignment(arrival_time_s=truth_arrival + 400.0)])
    early, = rolled_predictions_lockstep(*common, assignments=[Assignment(arrival_time_s=truth_arrival - 200.0)])
    d_free, d_on = free.forecast.command_hook_diagnostics, on_time.forecast.command_hook_diagnostics
    assert d_free["planAssigned"] == 0.0 and "planUnabsorbedFirstS" not in d_free
    assert d_on["planAssigned"] == 1.0 and all(np.isfinite(on_time.forecast.values).ravel())
    assert "closure" in on_time.orders[0] and "closure" not in free.orders[0]
    # the flown arrival against the assigned time, where the closure said it could meet it
    cut = cut_at_threshold_crossing(on_time.forecast, item)
    if abs(d_on["planUnabsorbedFirstS"]) <= TIME_TOLERANCE_S and cut.truncated_at_threshold:
        flown = float(cut.times[-1]) - float(item.times[anchor])
        assert abs(flown - labels.T_s) < 60.0
    # the closures at the first ask are monotone in the assignment (a one-epoch head's own
    # route may already take longer than the truth + 400 s, so no sign of X is assumed):
    # more time → a smaller speed factor, a longer closed time, more path, a larger X
    first = [flight.orders[0]["closure"] for flight in (early, on_time, late)]
    factors = [c["speed_factor"] for c in first]
    times = [c["plan_time_s"] for c in first]
    assert factors[0] >= factors[1] >= factors[2] and times[0] == times[1] == times[2]
    closed = [c["assigned_remaining_s"] - c["unabsorbed_s"] for c in first]
    assert closed[0] <= closed[1] <= closed[2]
    assert first[0]["unabsorbed_s"] <= first[1]["unabsorbed_s"] <= first[2]["unabsorbed_s"]
    assert first[2]["stretch_m"] >= first[1]["stretch_m"] >= first[0]["stretch_m"] == 0.0
    d_late, d_early = late.forecast.command_hook_diagnostics, early.forecast.command_hook_diagnostics
    assert d_early["planSpeedFactorFirst"] >= d_on["planSpeedFactorFirst"] >= d_late["planSpeedFactorFirst"]
    assert d_early["planStretchM"] == 0.0 and math.isfinite(d_late["planUnabsorbedLastS"])
    assert "planUnabsorbedStepP50S" in d_late and "planStretchDrops" in d_late
    # the assigned time is a target, not the budget: an advance the speed cannot make lands
    # LATE and says so, it is not cut short of the final
    assert early.capped_by is None
    assert cut_at_threshold_crossing(early.forecast, item).truncated_at_threshold == cut_at_threshold_crossing(free.forecast, item).truncated_at_threshold
    # the head's own ETA stays the prediction the rolled flight reports
    assert early.forecast.predicted_final_time_s == pytest.approx(free.forecast.predicted_final_time_s)
    # no assignment given two ways is one flight
    none_given, = rolled_predictions_lockstep(*common, assignments=[NO_ASSIGNMENT])
    assert np.array_equal(none_given.forecast.values, free.forecast.values)
    leg = LegOrder(plan=PlanToFly(90.0, 8_000.0, 70.0, 600.0), d_join_m=10_000.0, instruction=None, budget_s=100.0)
    assert leg.assigned_arrival_s is None



def _straight_route(length_m: float) -> Route:
    """A straight route of ``length_m`` to the threshold (plus the overrun), 50 m steps."""
    n = int((length_m + OVERRUN_M) / 50.0) + 1
    x = np.linspace(0.0, length_m + OVERRUN_M, n)
    points = np.stack([x, np.zeros_like(x)], axis=1)
    return Route(points=points, arc_m=x.copy(), join_index=n - 11, pre_final_m=length_m, requested_pre_final_m=length_m,
                 shortfall_m=0.0, stretch_offset_m=0.0, intercept_rad=0.0, kind="test")


def test_the_path_lever_brackets_a_builder_that_lays_in_quanta_and_never_takes_a_late_lay():
    """The route builder lays path in quanta (nothing below a threshold, then a large block):
    the lever doubles its request until something is laid, refuses a lay that would make the
    flight late, and halves between an early and a late lay."""
    base = _straight_route(20_000.0)
    plan = PlanToFly(100.0, None, 70.0, 500.0, ((20_000.0, 100.0),))
    floor = 70.0
    t_floor = plan_time_s(base, scaled(plan, floor / 100.0))
    requests: list[float] = []

    def quantised(extra_m: float):
        # below 3 km of request nothing is laid; then the laid length is the request rounded
        # DOWN to 5 km blocks (the dog-leg's quantum)
        requests.append(extra_m)
        laid = 0.0 if extra_m < 3_000.0 else 5_000.0 * math.floor(extra_m / 5_000.0)
        r = _straight_route(20_000.0 + laid)
        return r, plan

    # a delay the floor cannot hold that needs ~one block: the first request (residual × floor
    # speed) is under the quantum, the doubling finds it, the lay is early-or-on, never late
    need_s = t_floor + 5_000.0 / floor - 1.0
    closure, route, closed, laid = close_time(base, plan, need_s, v_floor_mps=floor, lay=quantised)
    assert closure.stretch_m == pytest.approx(5_000.0) and closure.unabsorbed_s >= -TIME_TOLERANCE_S
    assert len(requests) <= STRETCH_PROBES and route.length_m == pytest.approx(base.length_m + 5_000.0)
    assert laid.speed_points[0][0] == pytest.approx(route.remaining_m(0))    # re-anchored to the laid path
    # a delay between one and two blocks: the lever takes ONE block (early) and reports the
    # residual as X, never the two-block lay that would be late
    requests.clear()
    need_s = t_floor + 7_500.0 / floor
    closure, route, _closed, _laid = close_time(base, plan, need_s, v_floor_mps=floor, lay=quantised)
    assert closure.stretch_m == pytest.approx(5_000.0) and closure.unabsorbed_s > TIME_TOLERANCE_S
    assert route.length_m == pytest.approx(base.length_m + 5_000.0)
    # a builder that lays nothing at any request: the lever is exhausted, X is the residual
    closure, route, _c, _l = close_time(base, plan, t_floor + 60.0, v_floor_mps=floor, lay=lambda e: (_straight_route(20_000.0), plan))
    assert closure.stretch_m == 0.0 and route is base and closure.unabsorbed_s == pytest.approx(60.0, abs=TIME_TOLERANCE_S)
    assert MAX_STRETCH_M >= MIN_STRETCH_LAID_M
    # at the threshold there is nothing to close
    closure, route, c, l = close_time(base, plan, 100.0, v_floor_mps=floor, lay=quantised, start=len(base.points) - 1)
    assert closure.unabsorbed_s == 100.0 and c is plan and l is plan


def test_a_faster_held_speed_moves_the_deceleration_point_out_to_reach_v_final():
    """The deceleration point follows the held speed: what the new speed needs to bleed off
    to `V_final` at the deceleration rate, or the plan's own where that is farther out; a
    plan that never decelerates keeps none."""
    base = _straight_route(20_000.0)
    plan = PlanToFly(100.0, 3_000.0, 70.0, 500.0, ((20_000.0, 100.0),))
    fastest, factor = close_speed(base, plan, 1.0, v_floor_mps=60.0)
    assert held_speed_mps(fastest) == pytest.approx(SPEED_MAX_MPS) and factor == pytest.approx(SPEED_MAX_MPS / 100.0)
    assert fastest.d_decel_m == pytest.approx((SPEED_MAX_MPS ** 2 - 70.0 ** 2) / (2.0 * 0.5))
    slow = scaled(plan, 0.72)                       # 72 m/s needs 284 m: the plan's 3 km stands
    assert slow.d_decel_m == pytest.approx(3_000.0)
    assert scaled(PlanToFly(100.0, None, 70.0, 500.0), 1.3).d_decel_m is None
    assert scaled(PlanToFly(100.0, 1_000.0, 70.0, 500.0), 1.0).d_decel_m == pytest.approx((100.0 ** 2 - 70.0 ** 2) / (2.0 * 0.5))
    # re-anchoring moves only the first point's coordinate
    moved = reanchored(plan, _straight_route(25_000.0))
    assert moved.speed_points[0] == (pytest.approx(25_000.0), 100.0) and moved.v_mid_mps == plan.v_mid_mps
