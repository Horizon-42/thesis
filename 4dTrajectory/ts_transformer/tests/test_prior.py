"""The prior (`prior/`, prior design): the steps it is trained on, its batches, the network's causality and
runway mask, and the baselines' arithmetic, on synthetic flights."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import (
    ALTITUDE, APPROACH, APPROACH_CLEARED, APPROACH_NOT_CLEARED, HEADING, RUNWAY, SPEED, UNCHANGED, Words,
)
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Flight, Split, batches, column_classes, flight_steps
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.readout import Baselines
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec


def _grid(words):
    grid = np.full((6, 6), UNCHANGED, dtype=np.int16)
    grid[0] = [0, APPROACH_NOT_CLEARED, words.heading_index(270.0), words.altitude_index(1200.0), 0,
               words.speed_index(100.0)]
    grid[3, HEADING] = words.heading_index(180.0)
    grid[3, APPROACH] = APPROACH_CLEARED
    grid[5, SPEED] = words.speed_unspecified
    return grid


def test_a_step_sees_the_words_in_force_before_it_and_is_asked_for_its_own():
    words = Words(spec())
    grid = _grid(words)
    signals = instruction_flight(*fly_legs([(10, 0.0, 90.0, 0.0)], 270.0, 900.0, -5000.0, 300.0))
    features, in_force, since, targets = flight_steps(signals, grid, instruction_airport())
    assert (in_force[0] == 0).all()                                      # nothing said before step 0
    assert (in_force[1] == grid[0] + 1).all() and (in_force[4, HEADING] == grid[3, HEADING] + 1)
    assert in_force[3, HEADING] == grid[0, HEADING] + 1                  # step 3's own word is its target, not input
    assert (targets[0] == grid[0] + 1).all() and targets[1].sum() == 0
    assert targets[3, HEADING] == grid[3, HEADING] + 1 and targets[3, ALTITUDE] == 0
    assert since[4, HEADING] == pytest.approx(math.log1p(1) / prior_data.SINCE_SCALE)
    assert since[3, HEADING] == pytest.approx(math.log1p(3) / prior_data.SINCE_SCALE)
    # the runway features: zero, flag down, at step 0 (the pointer is still to be said); known from step 1
    runway = features[:, len(prior_data.STATE_FEATURES):]
    assert (runway[0] == 0).all() and (runway[1:, -1] == 1).all()
    # westbound 300 m north of the extended centreline, east of the threshold: before it by −e, the track reversed
    assert runway[1, 0] == pytest.approx(-signals.e_m[1] / prior_data.POSITION_SCALE_M)
    assert runway[1, 4] == pytest.approx(-1.0)


def test_batches_hold_every_flight_once_under_the_token_budget():
    flights = [Flight(f"F{i}", 0, np.zeros((n, prior_data.N_FEATURES), np.float32), np.zeros((n, 6), np.int16),
                      np.zeros((n, 6), np.float32), np.zeros((n, 6), np.int16)) for i, n in enumerate([5, 50, 7, 300, 40, 41])]
    groups = list(batches(flights, 100, np.random.default_rng(0)))
    assert sorted(i for g in groups for i in g) == list(range(6))
    for g in groups:
        assert len(g) == 1 or max(flights[i].rows for i in g) * len(g) <= 100


def _model(classes=None):
    words = Words(spec())
    classes = classes or column_classes(words, 2)
    candidates = torch.zeros(1, 2, len(prior_data.CANDIDATE_FEATURES))
    candidates[0, 0, -1] = 1.0                                           # one real candidate, one empty slot
    torch.manual_seed(0)
    return Prior(PriorConfig(classes=classes, airports=("KXXX",), candidate_slots=2, d_model=32, layers=2, heads=4,
                             feedforward=64, dropout=0.0), candidates).eval()


def test_a_step_sees_only_itself_and_the_steps_before_it_and_the_runway_head_only_real_candidates():
    model = _model()
    rows = 7
    features = torch.randn(1, rows, prior_data.N_FEATURES)
    in_force = torch.zeros(1, rows, 6, dtype=torch.long)
    since = torch.zeros(1, rows, 6)
    padding = torch.zeros(1, rows, dtype=torch.bool)
    airport = torch.zeros(1, dtype=torch.long)
    before = model(features, in_force, since, airport, padding)
    changed = features.clone()
    changed[0, 5:] += 3.0
    after = model(changed, in_force, since, airport, padding)
    for a, b in zip(before, after):
        assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5) and not torch.allclose(a[:, 5:], b[:, 5:])
    runway = before[RUNWAY][0]
    assert torch.isfinite(runway[:, :2]).all() and torch.isinf(runway[:, 2]).all()   # the empty slot is masked


def test_the_baselines_count_changes_and_values_from_train():
    classes = (3, 3, 4, 3, 3, 3)
    targets = np.zeros((4, 6), dtype=np.int16)
    targets[0] = [1, 1, 1, 1, 1, 1]
    targets[2, HEADING] = 3                                              # heading changes once, to value 2
    flight = Flight("F", 0, np.zeros((4, prior_data.N_FEATURES), np.float32),
                    np.vstack(([0] * 6, [1] * 6, [1] * 6, [1, 1, 3, 1, 1, 1])).astype(np.int16),
                    np.zeros((4, 6), np.float32), targets)
    split = Split([flight], ("KXXX",), np.zeros((1, 2, len(prior_data.CANDIDATE_FEATURES)), np.float32), classes)
    base = Baselines.count(split)
    assert base.change[HEADING] == pytest.approx((1 + 1) / (3 + 2))
    assert base.unigram[HEADING].tolist() == pytest.approx([1 / 4, 1 / 4, 2 / 4])
    assert base.bigram[HEADING][1].tolist() == pytest.approx([1 / 4, 1 / 4, 2 / 4])   # after value 0 (class 1)
    nll = base.nll_per_step(split)
    first = -math.log(base.first[HEADING][0])
    expected = (first - 2 * math.log(1 - base.change[HEADING]) - math.log(base.change[HEADING] * 2 / 4)) / 4
    assert nll["repeat"]["heading"] == pytest.approx(expected)
