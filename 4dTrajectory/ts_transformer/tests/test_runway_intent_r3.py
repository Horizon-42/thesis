"""Runway-intent R3's readings (`experiments.runway_intent_r3`) and gates (`experiments.runway_intent_r3_readout`)."""

import math

import pytest
from geokit import METRES_PER_DEG_LAT, NM_M

from ts_transformer.experiments.runway_intent_r3 import (
    approach_speed_mps,
    kept_share,
    one_runway_pairs,
    order_agreement,
)
from ts_transformer.experiments.runway_intent_r3_readout import gates
from ts_transformer.inference.runway_schedule import INDEPENDENT, SINGLE, Separation, Slot

SPEED = 70.0
GAP = 3.0 * NM_M / SPEED
# 12L/12R one runway, 12L's threshold 0.5 NM further along; 17 independent of both
RULES = Separation(same_nm=3.0, speed_mps=SPEED,
                   relations={frozenset(("12L", "12R")): SINGLE, frozenset(("12L", "17")): INDEPENDENT,
                              frozenset(("12R", "17")): INDEPENDENT},
                   along_nm={"12L": 0.5, "12R": 0.0, "17": 0.0})
LEAD = 0.5 * NM_M / SPEED        # 12L's threshold time runs this far behind its approach clock


def test_pairs_follow_the_approach_clock_across_a_group_and_skip_other_runways():
    # a 12L landing 10 s after a 12R one passed abeam it 16 s BEFORE: on the approach it led
    slots = [Slot("r", "12R", 100.0, 100.0, "F"), Slot("l", "12L", 110.0, 110.0, "F"), Slot("x", "17", 105.0, 105.0, "F"),
             Slot("r2", "12R", 400.0, 400.0, "F")]
    assert [(a.key, b.key) for a, b in one_runway_pairs(slots, RULES)] == [("l", "r"), ("r", "r2")]


def test_the_kept_share_reads_every_pair_or_only_the_close_ones():
    slots = [Slot("a", "12R", 0.0, 0.0, "F"), Slot("b", "12R", GAP - 20.0, 0.0, "F"),     # 20 s short
             Slot("c", "12R", 1000.0, 0.0, "F"), Slot("d", "12R", 2000.0, 0.0, "F")]      # far apart: kept
    assert kept_share(slots, RULES, 10.0) == (pytest.approx(2 / 3), 3)
    assert kept_share(slots, RULES, 10.0, close_only=True) == (0.0, 1)
    assert kept_share(slots, RULES, 25.0, close_only=True) == (1.0, 1)


def _row(key: str, runway: str, truth_s: float, independent_s: float) -> dict:
    return {"flight_key": key, "category": "F", "truth": {"runway": runway, "time_s": truth_s},
            "independent": {"runway": runway, "time_s": independent_s}}


def test_the_order_agreement_reads_every_true_pair_or_only_the_contestable_ones():
    rows = [_row("a", "12R", 0.0, 30.0), _row("b", "12R", 90.0, 20.0),                  # close, and swapped
            _row("c", "12R", 1000.0, 1000.0), _row("d", "12R", 2000.0, 2000.0)]         # far apart, in order
    assert order_agreement(rows, RULES, "independent") == (pytest.approx(2 / 3), 3)
    assert order_agreement(rows, RULES, "independent", close_only=True) == (0.0, 1)
    # staggered: a 12L truth landing 10 s after a 12R one led it, so a forecast that keeps the times keeps the order
    staggered = [_row("r", "12R", 0.0, 0.0), _row("l", "12L", 10.0, 10.0)]
    assert order_agreement(staggered, RULES, "independent") == (1.0, 1)
    assert LEAD > 10.0


def _track(start_m: float, speed_mps: float, rollout_rows: int) -> dict:
    # north-bound at ``speed_mps`` from ``start_m`` short of a threshold at (35, -80), then down the runway at 20 m/s
    rows, t, d = [], 0.0, -start_m
    while d < 0.0:
        rows.append([t, -80.0, 35.0 + d / METRES_PER_DEG_LAT, 100.0])
        t, d = t + 2.0, d + 2.0 * speed_mps
    for _ in range(rollout_rows):
        rows.append([t, -80.0, 35.0 + d / METRES_PER_DEG_LAT, 0.0])
        t, d = t + 2.0, d + 40.0
    return {"runway": "36", "waypoints": rows}


def test_the_approach_speed_stops_at_the_threshold_and_skips_a_track_that_starts_inside():
    targets = {"36": {"lat": 35.0, "lon": -80.0, "course_deg": 0.0}}
    speed, n = approach_speed_mps([_track(8000.0, 70.0, 20), _track(8000.0, 60.0, 0), _track(2000.0, 10.0, 0)], targets)
    assert n == 2                                     # the third starts inside the 2 NM window
    assert speed == pytest.approx(65.0, abs=0.5)      # the rollout rows past the threshold are never read


def _artifact(*, flown_kept: float | None, busy: tuple[float, float] | None, all_hours: tuple[float, float] = (12.0, 12.0),
              runway: tuple[float, float] = (0.97, 0.97)) -> dict:
    def cell(sched_dt: float, indep_dt: float) -> dict:
        return {"scheduled": {"time_error_s": {"p50": sched_dt}, "runway_accuracy": runway[0]},
                "independent": {"time_error_s": {"p50": indep_dt}, "runway_accuracy": runway[1]}}

    summary = {"all": cell(*all_hours), "quiet": cell(*all_hours)}
    if busy is not None:
        summary["busy"] = cell(*busy)
    checks = {"plan_violations": 0}
    if flown_kept is not None:
        checks["flown_one_runway_kept_share"] = [flown_kept, 100]
    return {"checks": checks, "summary": summary}


def test_a_gate_not_measured_leaves_the_verdict_open():
    verdicts = gates({
        "PASS": _artifact(flown_kept=0.99, busy=(10.0, 12.0)),
        "NOFLY": _artifact(flown_kept=None, busy=(10.0, 12.0)),
        "NOBUSY": _artifact(flown_kept=0.99, busy=None),
        "FAIL": _artifact(flown_kept=0.90, busy=(10.0, 12.0)),
        "RUNWAY": _artifact(flown_kept=0.99, busy=(10.0, 12.0), runway=(0.95, 0.97)),
    })
    assert verdicts["PASS"]["passed"] is True
    assert verdicts["NOFLY"]["passed"] is None and verdicts["NOBUSY"]["passed"] is None
    assert verdicts["FAIL"]["passed"] is False and verdicts["RUNWAY"]["passed"] is False
    assert verdicts["PASS"]["r4_timing_ratio_busy_over_quiet"] == pytest.approx(10.0 / 12.0)
    assert math.isclose(verdicts["RUNWAY"]["r4_runway_gap"], 0.0)
