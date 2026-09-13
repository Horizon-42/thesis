"""Runway-intent R2a's evaluation roster and its reading of the heads (`experiments.runway_intent_r2`)."""

from types import SimpleNamespace

import pytest

from ts_transformer.config import TSConfig
from ts_transformer.data.runway_context import operational_day, parse_utc
from ts_transformer.experiments.runway_intent_r1 import day_folds
from ts_transformer.experiments.runway_intent_r2 import (
    _expected,
    _top,
    anchor_waypoint,
    day_validation_flights,
    evaluation_flights,
)
from ts_transformer.experiments.runway_intent_r2b_cohort import training_day_flights


def test_the_head_reads_the_last_track_point_at_or_before_the_experts_anchor():
    flight = {"waypoints": [[3.0, 0, 0, 0], [5.0, 0, 0, 0], [8.0, 0, 0, 0], [12.0, 0, 0, 0], [15.0, 0, 0, 0]]}
    assert anchor_waypoint(flight, 9.0) == 3          # 12 - 3 = 9 s after the first point
    assert anchor_waypoint(flight, 8.9) == 2          # nothing after the anchor
    assert anchor_waypoint(flight, 0.0) == 0


def test_the_belief_weighted_error_renormalises_over_the_candidates_with_a_forecast():
    prob = {"05L": 0.6, "05R": 0.3, "23R": 0.1}
    rows = {"05L": {"fde_m": 100.0, "ade_m": 50.0}, "05R": {"fde_m": 1000.0, "ade_m": 500.0}}
    got = _expected(prob, rows)
    assert got["fde_m"] == pytest.approx((0.6 * 100 + 0.3 * 1000) / 0.9)
    assert got["ade_m"] == pytest.approx((0.6 * 50 + 0.3 * 500) / 0.9)
    assert _top(prob, set(rows)) == "05L" and _top(prob, {"05R", "23R"}) == "05R"


def test_an_evaluation_flight_is_expert_validation_on_a_partitions_validation_day():
    config = TSConfig()
    days = [f"2026-06-{d:02d}" for d in range(1, 31)]
    usable, expected = {}, {}
    for i, day in enumerate(days):
        key = f"f{i}"
        usable[key] = {"landing_time_utc": f"{day}T14:00:00Z"}
        folds = day_folds(operational_day(parse_utc(usable[key]["landing_time_utc"])), config)
        if folds["a"] == "val":
            expected[key] = "day_a"
        elif folds["b"] == "val":
            expected[key] = "day_b"
    s = SimpleNamespace(usable=usable, config=config)
    everyone = evaluation_flights(s, set(usable))
    assert everyone == expected and set(everyone.values()) == {"day_a", "day_b"}
    # a flight outside the expert's validation split is out, whatever its day
    held_back = evaluation_flights(s, set(usable) - {next(iter(expected))})
    assert next(iter(expected)) not in held_back


def test_the_r2b_cohort_keeps_training_days_and_the_r2b_roster_takes_the_validation_days():
    config = TSConfig()
    days = [f"2026-06-{d:02d}" for d in range(1, 31)]
    landing = {f"KSMF:f{i}": f"{day}T14:00:00Z" for i, day in enumerate(days)}
    fold = {fid: day_folds(operational_day(parse_utc(t)), config)["a"] for fid, t in landing.items()}
    kept = training_day_flights(tuple(landing), landing, "day_a", config)
    assert kept == [fid for fid in landing if fold[fid] == "train"]
    s = SimpleNamespace(usable={fid.split(":")[1]: {"landing_time_utc": t} for fid, t in landing.items()}, config=config)
    roster = day_validation_flights(s, "day_a")
    assert set(roster) == {fid.split(":")[1] for fid in landing if fold[fid] == "val"}
    assert not set(roster) & {fid.split(":")[1] for fid in kept}      # disjoint by construction

