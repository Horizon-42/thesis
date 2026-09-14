"""Runway-intent R3's readings (`experiments.runway_intent_r3`) and gates (`experiments.runway_intent_r3_readout`)."""

import json
import math

import numpy as np
import pytest
import torch
from geokit import METRES_PER_DEG_LAT, NM_M

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import TSConfig
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.experiments import runway_intent_r3
from ts_transformer.experiments.runway_intent_r3 import (
    Flown,
    approach_speed_mps,
    endpoint_error_m,
    kept_share,
    one_runway_pairs,
    order_agreement,
    write_schedule_records,
)
from ts_transformer.inference.forecast import forecast_approach
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


def test_the_end_point_error_is_where_the_forecast_ends_against_the_true_threshold():
    assert endpoint_error_m({"endpoint_along_track_true_m": -3.0, "endpoint_cross_track_true_m": 4.0, "fde_m": 900.0}) == 5.0


def test_the_schedule_records_carry_the_scheduled_time_as_the_prediction(tmp_path):
    """R3.3: a record's prediction is its SCHEDULED landing time (as a CTA arm's is its CTA), graded against
    the observed track; the head's ETA error and the flown one ride beside it; an unlanded flight has no
    delivery; the summary block says what the numbers measure and under which settings."""
    config = TSConfig()
    series, _report = build_series(synthetic_arrivals("KRDU", "05L", n_flights=2, seed=3), config, airport="KRDU")
    model, normalizer = build_model(config).eval(), Normalizer.fit(series)
    flown, rows = {}, []
    for number, (x, landed) in enumerate(zip(series, (True, False), strict=True)):
        forecast = forecast_approach(model, x, config, normalizer, device=torch.device("cpu"))
        anchor_s = 1000.0 * (number + 1)
        truth_s = anchor_s + float(x.supervision_times[-1] - x.times[forecast.anchor])
        key = f"F{number}"
        rows.append({
            "flight_key": key, "category": "D", "anchor_s": anchor_s, "probabilities": {"05L": 0.9, "05R": 0.1},
            "truth": {"runway": "05L", "time_s": truth_s}, "independent": {"runway": "05L"},
            "scheduled": {"runway": "05L" if landed else "05R", "time_s": truth_s + 30.0, "eta_s": truth_s - 12.0,
                          "delay_s": 42.0},
        })
        flown[key] = Flown(
            landed=landed, time_s=truth_s + 35.0, metrics={}, wall_s=np.zeros(1), east_m=np.zeros(1),
            north_m=np.zeros(1), on_final=np.zeros(1, dtype=bool), shown=forecast, truth=x,
            closure={"planUnabsorbedFirstS": 1.0, "planUnabsorbedLastS": 0.5},
        )
    checkpoint = tmp_path / "expert" / "checkpoint.pt"
    settings = {"r2": "r2.json", "delay_weight_per_s": 1 / 60, "min_probability": 0.01, "approach_speed_mps": 70.0}

    out = write_schedule_records(tmp_path / "records", flown, rows, config=config, checkpoint=checkpoint,
                                 airport="KRDU", settings=settings)

    summary = json.loads((tmp_path / "records" / "summary.json").read_text())
    assert out["records"] == 2 and summary["split"] == runway_intent_r3.RECORDS_SPLIT == "dayval"
    assert summary["checkpoint"] == str(checkpoint)
    block = summary["runway_schedule"]
    assert (block["records"], block["landed"], block["moved"]) == (2, 1, 1)
    assert {name: block[name] for name in settings} == settings
    assert "scheduled landing time" in block["timing"] and "TRUE runway" in block["runway_frame"]
    landed_row, unlanded_row = summary["results"]
    assert landed_row["final_time_error_s"] == pytest.approx(30.0, abs=1e-6)   # scheduled - truth
    schedule = [json.loads((tmp_path / "records" / row["eval_file"]).read_text())["source"]["runwaySchedule"]
                for row in summary["results"]]
    assert schedule[0]["etaErrorS"] == pytest.approx(-12.0) and schedule[0]["scheduledTimeErrorS"] == pytest.approx(30.0)
    assert schedule[0]["flownTimeErrorS"] == pytest.approx(35.0) and schedule[0]["deliveryS"] == pytest.approx(5.0)
    assert schedule[1]["landed"] is False and schedule[1]["deliveryS"] is None and schedule[1]["flownTimeErrorS"] is None
    assert (schedule[1]["scheduledRunway"], schedule[1]["trueRunway"]) == ("05R", "05L")


def test_records_are_written_only_for_a_flown_schedule(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        runway_intent_r3.main(["--airport", "KRDU", "--r2", "r2.json", "--checkpoint", "c.pt",
                               "--output-dir", str(tmp_path / "out"), "--no-fly", "--write-records", str(tmp_path / "rec")])
    assert exit_info.value.code == 2
    assert not (tmp_path / "rec").exists() and not (tmp_path / "out").exists()
