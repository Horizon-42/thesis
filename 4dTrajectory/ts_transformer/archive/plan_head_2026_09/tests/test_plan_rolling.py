"""The rolled flight (design v5 §4.2): a plan flown one instruction at a time on the
synthetic KRDU cohort — the truth's own instructions reach the threshold on the final, the
legs concatenate into one forecast with one clock, and a stale instruction is skipped; and
the lockstep's order hold (v5.3): a jittering order is held, a persistent change adopted."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import cut_at_threshold_crossing
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import (
    KIND_ROLLED,
    ORDER_HOLD_ASKS,
    RELAY_FIX_M,
    RELAY_HEADING_RAD,
    ROUTE_WAYPOINTS,
    Instruction,
    LegOrder,
    fly_lockstep,
    fly_lockstep_truth,
    fly_plans,
    fly_rolling,
    lockstep_states,
    orders_differ,
    plan_to_fly,
)
from ts_transformer.outputs.plan.labels import truth_instructions
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.tests.support import AIRPORT, RUNWAY

pytestmark = pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)


def _cohort(n_flights: int = 3, **arrivals):
    config = TSConfig()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3, **arrivals)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    skeleton = runway_skeleton(series[0])
    anchor = default_anchor(config)
    return series, config, skeleton, anchor


def test_the_truths_instructions_rolled_reach_the_threshold_as_one_forecast():
    series, config, skeleton, anchor = _cohort(3)
    # an EARLY anchor, so the synthetic turn onto the final lies ahead as an instruction
    anchor = min(anchor, 12)
    rolled_any = False
    for item in series:
        labels = extract_plan(item, anchor, skeleton)
        instructions = truth_instructions(labels, skeleton)
        flight = fly_rolling(item, anchor, labels, skeleton, config, instructions=instructions)
        forecast = flight.forecast
        assert len(flight.routes) == flight.instructions_flown + 1
        assert flight.routes[-1].kind != KIND_ROLLED
        # one clock: strictly increasing times from the anchor, the samples' durations
        # adding up to the total, one control row per segment
        assert 0.0 < forecast.times[0] - float(item.times[anchor]) <= 3.0
        assert np.all(np.diff(forecast.times) > 0.0)
        assert forecast.sample_durations_s.sum() == pytest.approx(forecast.final_time_s)
        assert forecast.controls.shape[0] == forecast.segment_durations_s.shape[0]
        assert forecast.command_hook_diagnostics["steps"] == pytest.approx(forecast.controls.shape[0])
        assert forecast.command_hook_diagnostics["rolledLegs"] == len(flight.routes)
        cut = cut_at_threshold_crossing(forecast, item)
        assert cut.truncated_at_threshold, "the rolled flight must reach the threshold on the final"
        rolled_any = rolled_any or flight.instructions_flown > 0
    assert rolled_any, "the synthetic cohort holds a flight with an instruction ahead of the early anchor"


def test_a_rolled_flight_with_no_instruction_ahead_is_one_closing_leg_onto_the_final():
    """Without an instruction the rolled flight is the closing flown once — onto the
    plan's join and down the final, on the same horizon as the whole-path flight."""
    series, config, skeleton, anchor = _cohort(2)
    item = series[0]
    labels = extract_plan(item, anchor, skeleton)
    whole, _routes, _times = fly_plans([item], anchor, [labels], [skeleton], config)
    flight = fly_rolling(item, anchor, labels, skeleton, config, instructions=[], horizon_s=labels.T_s)
    assert flight.instructions_flown == 0 and len(flight.routes) == 1 and flight.capped_by is None
    assert flight.forecast.final_time_s == pytest.approx(whole[0].final_time_s, rel=1e-6)
    last = flight.forecast.values[-1]
    assert math.hypot(last[IDX["e"]] - skeleton.target_e, last[IDX["n"]] - skeleton.target_n) < 1_500.0


def test_an_instruction_on_top_of_the_aircraft_is_skipped_and_a_turn_back_is_flown():
    series, config, skeleton, anchor = _cohort(1)
    item = series[0]
    labels = extract_plan(item, anchor, skeleton)
    e, n = float(item.values[anchor, IDX["e"]]), float(item.values[anchor, IDX["n"]])
    heading = math.atan2(item.values[anchor, IDX["ndot"]], item.values[anchor, IDX["edot"]])
    on_top = Instruction(e + 50.0 * math.cos(heading), n + 50.0 * math.sin(heading), heading, 80.0,
                         labels.remaining_path_at_anchor_m - 50.0, float(item.values[anchor, IDX["u"]]))
    flight = fly_rolling(item, anchor, labels, skeleton, config, instructions=[on_top])
    assert flight.instructions_skipped == 1 and flight.instructions_flown == 0
    assert len(flight.routes) == 1
    # a fix 4 km off to the side, heading back toward the runway: a turn, flown as a leg
    beside = Instruction(e + 4_000.0 * math.cos(heading + math.pi / 2), n + 4_000.0 * math.sin(heading + math.pi / 2),
                         skeleton.course_rad, 80.0, labels.remaining_path_at_anchor_m, float(item.values[anchor, IDX["u"]]))
    flight = fly_rolling(item, anchor, labels, skeleton, config, instructions=[beside])
    assert flight.instructions_flown == 1 and flight.instructions_skipped == 0 and len(flight.routes) == 2


def test_the_lockstep_oracle_reaches_the_threshold_and_a_group_flies_as_its_members_alone():
    """v5.1: the truth's instructions re-laid every step, the group stepped together — each
    flight closes on the final, executes its instructions, and flies exactly as it does in
    a group of one (the batch is a batch, not a coupling)."""
    series, config, skeleton, anchor = _cohort(3)
    anchor = min(anchor, 12)
    labels = [extract_plan(item, anchor, skeleton) for item in series]
    horizons = [lab.T_s + 30.0 for lab in labels]
    together = fly_lockstep_truth(series, [anchor] * 3, labels, [skeleton] * 3, config, horizons_s=horizons)
    assert any(flight.instructions_flown > 0 for flight in together)
    for item, flight, lab, horizon in zip(series, together, labels, horizons, strict=True):
        assert flight.capped_by is None and flight.turns_incomplete == 0
        assert flight.forecast.command_hook_diagnostics["planSteps"] >= 2
        assert cut_at_threshold_crossing(flight.forecast, item).truncated_at_threshold
        alone, = fly_lockstep_truth([item], [anchor], [lab], [skeleton], config, horizons_s=[horizon])
        assert alone.forecast.values.shape == flight.forecast.values.shape
        assert np.allclose(alone.forecast.values, flight.forecast.values, atol=1e-3)
        # v5.3: the order hold is a no-op on the oracle's path — its orders change only at
        # execution, a phase the hold never gates
        held, = fly_lockstep_truth([item], [anchor], [lab], [skeleton], config, horizons_s=[horizon], hold_asks=2)
        assert held.forecast.values.shape == flight.forecast.values.shape
        assert np.array_equal(held.forecast.values, flight.forecast.values)
        assert held.forecast.command_hook_diagnostics["planHeldSteps"] == 0.0


def _scripted_policy(script: list[Instruction | None], plan, d_join_m):
    """A policy answering, per flight, the instruction ``script[step]`` (the last past the end)."""

    def policy(active):
        return [
            LegOrder(plan=plan, d_join_m=d_join_m, instruction=script[min(s.steps, len(script) - 1)], budget_s=900.0)
            for s in active
        ]

    return policy


def test_a_jittering_order_is_held_and_a_persistent_change_is_adopted():
    """v5.3 (`forecast.held_order`): under `hold_asks=2` an order that differs materially
    from the one in force is adopted only once the policy has given it on two consecutive
    asks; a one-ask change is held (the instruction in force flown on); `hold_asks=1` — the
    default, the v5.1/v5.2 behaviour — adopts at once; `hold_flips_only` lets a moved fix
    through and holds a fix ↔ none flip. The record says which steps were held and flown."""
    assert ORDER_HOLD_ASKS == 1
    series, config, skeleton, anchor = _cohort(3)
    anchor = min(anchor, 12)
    # a flight with a turn ahead of the early anchor: not on the final there, so the
    # lockstep's own on-final rule stays out of the hold's way
    item, labels = next(
        (item, lab) for item in series if (lab := extract_plan(item, anchor, skeleton)).waypoints
    )
    plan, _clamped = plan_to_fly(labels, float(item.values[anchor, IDX["u"]]), skeleton, route=ROUTE_WAYPOINTS)
    e, n = float(item.values[anchor, IDX["e"]]), float(item.values[anchor, IDX["n"]])
    heading = math.atan2(item.values[anchor, IDX["ndot"]], item.values[anchor, IDX["edot"]])
    height = float(item.values[anchor, IDX["u"]])
    # two fixes 30 km ahead (never executed inside the cap), 3 km apart: a material change
    a = Instruction(e + 30_000.0 * math.cos(heading), n + 30_000.0 * math.sin(heading), heading, 80.0, 20_000.0, height)
    b = Instruction(
        a.fix_e + 3_000.0 * math.cos(heading + math.pi / 2), a.fix_n + 3_000.0 * math.sin(heading + math.pi / 2),
        heading, 80.0, 20_000.0, height,
    )
    script = [a, b, a, b, b]
    fix_a, fix_b = [a.fix_e, a.fix_n], [b.fix_e, b.fix_n]

    def fly(hold_asks: int, script=script, flips_only: bool = False):
        states = lockstep_states([item], [anchor], [skeleton], [labels.remaining_path_at_anchor_m])
        flight, = fly_lockstep(
            states, config, policy=_scripted_policy(script, plan, labels.d_join_m), time_caps_s=[150.0],
            hold_asks=hold_asks, hold_flips_only=flips_only,
        )
        return flight

    held = fly(2)
    assert len(held.orders) == 5
    assert [order["held"] for order in held.orders] == [False, True, False, True, False]
    assert [order["flown_fix"] for order in held.orders] == [fix_a, fix_a, fix_a, fix_a, fix_b]
    assert held.forecast.command_hook_diagnostics["planHeldSteps"] == 2.0
    assert held.forecast.command_hook_diagnostics["planOrderChanges"] == 1.0
    at_once = fly(1)
    assert [order["held"] for order in at_once.orders] == [False] * 5
    assert [order["flown_fix"] for order in at_once.orders] == [fix_a, fix_b, fix_a, fix_b, fix_b]
    assert at_once.forecast.command_hook_diagnostics["planHeldSteps"] == 0.0
    assert at_once.forecast.command_hook_diagnostics["planOrderChanges"] == 3.0
    with pytest.raises(ValueError, match="hold_asks"):
        fly(0)
    # flips only: the moved fix passes at once, a fix ↔ none flip is held
    moved = fly(2, flips_only=True)
    assert [order["held"] for order in moved.orders] == [False] * 5
    assert [order["flown_fix"] for order in moved.orders] == [fix_a, fix_b, fix_a, fix_b, fix_b]
    flipped = fly(2, script=[a, None, a, None, None], flips_only=True)
    assert [order["held"] for order in flipped.orders] == [False, True, False, True, False]
    assert [order["flown_fix"] for order in flipped.orders] == [fix_a, fix_a, fix_a, fix_a, None]
    with pytest.raises(ValueError, match="hold_flips_only"):
        fly(1, flips_only=True)
    # three or more distinct orders never agree twice: the first is flown to the cap
    c = Instruction(b.fix_e + 3_000.0 * math.cos(heading + math.pi / 2), b.fix_n + 3_000.0 * math.sin(heading + math.pi / 2),
                    heading, 80.0, 20_000.0, height)
    walking = fly(2, script=[a, b, c, b, c])
    assert [order["flown_fix"] for order in walking.orders] == [fix_a] * 5
    assert walking.forecast.command_hook_diagnostics["planOrderChanges"] == 0.0
    assert walking.forecast.command_hook_diagnostics["planHoldAsks"] == 2.0


def test_orders_differ_is_the_re_lays_material_test():
    """`forecast.orders_differ`: fix ↔ none, a join moved over `RELAY_FIX_M` (with or
    without an instruction), a fix moved that far, a heading over `RELAY_HEADING_RAD`."""
    fix = Instruction(0.0, 0.0, 0.0, 80.0, 10_000.0, 500.0)
    assert orders_differ(None, 5_000.0, fix, 5_000.0) and orders_differ(fix, 5_000.0, None, 5_000.0)
    assert not orders_differ(None, 5_000.0, None, 5_000.0 + RELAY_FIX_M)
    assert orders_differ(None, 5_000.0, None, 5_000.0 + RELAY_FIX_M + 1.0)
    assert orders_differ(None, None, None, 5_000.0) and not orders_differ(None, None, None, None)
    assert not orders_differ(fix, 5_000.0, fix, 5_000.0)
    assert orders_differ(fix, 5_000.0, fix, 5_000.0 + RELAY_FIX_M + 1.0)
    moved = Instruction(RELAY_FIX_M + 1.0, 0.0, 0.0, 80.0, 10_000.0, 500.0)
    assert orders_differ(fix, 5_000.0, moved, 5_000.0)
    turned = Instruction(0.0, 0.0, RELAY_HEADING_RAD + 0.01, 80.0, 10_000.0, 500.0)
    assert orders_differ(fix, 5_000.0, turned, 5_000.0)
    assert not orders_differ(fix, 5_000.0, Instruction(0.0, 0.0, RELAY_HEADING_RAD - 0.01, 90.0, 1.0, 1.0), 5_000.0)
