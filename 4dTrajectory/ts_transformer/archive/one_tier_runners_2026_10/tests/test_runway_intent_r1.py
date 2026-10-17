"""Runway-intent R1's split and scoring logic (`experiments.runway_intent_r1`)."""

from collections import Counter

import numpy as np

from ts_transformer.config import TSConfig
from ts_transformer.experiments.runway_intent_r1 import (
    by_day,
    day_folds,
    expected_calibration_error,
    level_accuracy,
    minority_runways,
)


def test_both_day_partitions_share_one_test_fold_and_split_the_rest():
    config = TSConfig()
    days = [f"2026-06-{d:02d}" for d in range(1, 31)] + [f"2026-07-{d:02d}" for d in range(1, 32)]
    folds = [day_folds(day, config) for day in days]
    assert all((f["a"] == "test") == (f["b"] == "test") for f in folds)
    assert {f["a"] for f in folds} == {"train", "val", "test"}
    assert {f["b"] for f in folds} == {"train", "val", "test"}
    # the two partitions are different partitions of the non-test days
    assert any(f["a"] != f["b"] for f in folds if f["a"] != "test")
    # deterministic: a day's fold never depends on the others
    assert day_folds("2026-06-05", config) == day_folds("2026-06-05", config)


def test_the_minority_runways_are_all_but_the_busiest_of_each_parallel_group():
    groups = {"05L": 0, "05R": 0, "23L": 1, "23R": 1, "14": 2}
    majority = Counter({"05L": 300, "05R": 150, "23R": 600, "23L": 300, "14": 5})
    assert minority_runways(groups, majority) == ["05R", "23L"]


def test_side_accuracy_is_read_only_where_the_direction_is_right_and_has_a_choice():
    group_of = np.array([0, 0, 1])          # 05L, 05R, 14
    multi = np.array([True, True, False])
    truth = np.array([0, 0, 1, 2, 2])
    picks = np.array([0, 1, 1, 2, 0])      # right, wrong side, right, right single, wrong direction
    cell = level_accuracy(picks, truth, group_of, multi)
    assert cell["exact"] == 3 / 5
    assert cell["direction"] == 4 / 5
    assert cell["side_samples"] == 3 and cell["side_given_direction"] == 2 / 3


def test_a_calibrated_forecaster_has_no_calibration_error():
    prob = np.array([[0.75, 0.25]] * 4)
    truth = np.array([0, 0, 0, 1])           # right 3 of 4 at 75 % confidence
    assert abs(expected_calibration_error(prob, truth)) < 1e-12
    overconfident = np.array([[0.99, 0.01]] * 4)
    assert expected_calibration_error(overconfident, truth) > 0.2


def test_the_day_blocks_count_flights_at_the_entry_and_score_every_pick_array():
    days = np.array(["d1", "d1", "d1", "d2", "d2", "d3"])
    anchors = np.array(["entry", "r10km", "entry", "entry", "r10km", "entry"])
    truth = np.array([1, 1, 0, 1, 1, 0])
    picks = {"model": np.array([1, 0, 0, 0, 0, 0]), "rule": np.array([1, 1, 1, 1, 1, 1])}
    mask = np.array([True, True, True, True, True, False])          # d3 is not validation
    cells = by_day(mask, days, truth, anchors, picks, ["30L", "30R"])
    assert list(cells) == ["d1", "d2"]
    assert cells["d1"]["flights"] == 2 and cells["d1"]["samples"] == 3
    assert cells["d1"]["top_share"] == 0.5 and cells["d1"]["top_runway"] == "30L"   # a tie goes to the lower index
    assert cells["d1"]["exact"] == {"model": 2 / 3, "rule": 2 / 3}
    assert cells["d2"]["top_runway"] == "30R" and cells["d2"]["top_share"] == 1.0
    assert cells["d2"]["exact"] == {"model": 0.0, "rule": 1.0}
