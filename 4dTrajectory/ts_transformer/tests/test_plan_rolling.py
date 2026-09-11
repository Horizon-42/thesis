"""The rolled flight (design v5 §4.2): a plan flown one instruction at a time on the
synthetic KRDU cohort — the truth's own instructions reach the threshold on the final, the
legs concatenate into one forecast with one clock, and a stale instruction is skipped."""

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
    Instruction,
    fly_lockstep_truth,
    fly_plans,
    fly_rolling,
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
