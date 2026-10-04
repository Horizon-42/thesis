"""Runway-intent R1.1's head and its airline share (`experiments.runway_intent_r11`)."""

from collections import Counter

import numpy as np

from ts_transformer.data.runway_features import CANDIDATE_ROW_NAMES
from ts_transformer.experiments.runway_intent_r11 import (
    AIRLINE_PRIOR_WEIGHT,
    IDENTITY_COLUMNS,
    ListwiseBooster,
    airline_shares,
    head_columns,
    head_table,
    prior_share,
)

SMALL = dict(n_iter=60, learning_rate=0.3, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=5)


def _table(n: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Column 0 decides: the answer is the candidate with the largest value; column 1 is noise."""
    table = rng.normal(size=(n, count, 2))
    return table, table[:, :, 0].argmax(axis=1)


def test_a_rule_learned_on_some_runways_applies_to_one_never_seen_as_the_answer():
    rng = np.random.default_rng(0)
    table, truth = _table(3000, 3, rng)
    # training never has candidate 2 as the answer: its column-0 value is pushed below the others
    train = table.copy()
    train[:, 2, 0] = np.minimum(train[:, 2, 0], train[:, :2, 0].max(axis=1) - 0.5)
    head = ListwiseBooster(**SMALL).fit(train, train[:, :, 0].argmax(axis=1))
    test, test_truth = _table(2000, 3, rng)
    picked = head.predict_proba(test).argmax(axis=1)
    on_two = test_truth == 2
    assert on_two.sum() > 500
    assert (picked[on_two] == 2).mean() > 0.9           # a per-runway head could not know this
    assert (picked == test_truth).mean() > 0.9


def test_reordering_the_candidates_reorders_the_probabilities_the_same_way():
    rng = np.random.default_rng(1)
    table, truth = _table(1500, 4, rng)
    head = ListwiseBooster(**SMALL).fit(table, truth)
    order = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(head.predict_proba(table[:, order]), head.predict_proba(table)[:, order], atol=1e-12)
    prob = head.predict_proba(table)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)


def test_the_prior_is_each_runways_share_of_the_training_landings():
    np.testing.assert_allclose(prior_share(Counter({"30L": 3, "30R": 1}), ["12L", "30L", "30R"]), [0.0, 0.75, 0.25])


def _one_flight_per_day(labels: list[int], operator: str = "AAL") -> tuple[np.ndarray, ...]:
    """Flight k lands on day k (two samples each, as R1's anchors give)."""
    days = np.repeat(np.array([f"2026-06-{d + 1:02d}" for d in range(len(labels))]), 2)
    keys = np.repeat(np.array([f"f{d}" for d in range(len(labels))]), 2)
    return (np.array([operator] * len(days)), np.repeat(np.array(labels), 2), days, keys)


def test_a_samples_airline_share_counts_only_training_days_before_its_own():
    operators, labels, days, keys = _one_flight_per_day([0, 0, 1, 1, 0, 1])
    train = np.array([True] * 8 + [False] * 4)          # the last two days are validation
    share, base = airline_shares(operators, labels, days, keys, train, 2)
    for flipped_day in (3, 4, 5):                          # its own day, a later day, a validation day
        changed = labels.copy()
        changed[2 * flipped_day: 2 * flipped_day + 2] ^= 1
        again, _ = airline_shares(operators, changed, days, keys, train, 2)
        np.testing.assert_array_equal(again[6:8], share[6:8])        # day 3's samples do not move
    earlier = labels.copy()
    earlier[0:2] = 1
    moved, _ = airline_shares(operators, earlier, days, keys, train, 2)
    assert not np.array_equal(moved[6:8], share[6:8])               # an earlier training day does
    # day 3 has seen days 0-2 (labels 0, 0, 1), each flight once: base = (2+1, 1+1) / (3+2)
    np.testing.assert_allclose(base[6], [3 / 5, 2 / 5])
    np.testing.assert_allclose(share[6], (np.array([2.0, 1.0]) + AIRLINE_PRIOR_WEIGHT * base[6]) / (3 + AIRLINE_PRIOR_WEIGHT))
    np.testing.assert_allclose(share[0], base[0])                    # the first day has no history


def test_no_callsign_reads_the_base_rate_and_an_unseen_operator_does_too():
    operators, labels, days, keys = _one_flight_per_day([0, 1, 0])
    operators = operators.copy()
    operators[4:6] = ""
    share, base = airline_shares(operators, labels, days, keys, np.ones(6, dtype=bool), 2)
    np.testing.assert_allclose(share[4:6], base[4:6])
    other, _ = airline_shares(np.array(["UAL"] * 6), labels, days, keys, np.array([False] * 4 + [True] * 2), 2)
    np.testing.assert_allclose(other[4], [0.5, 0.5])   # no training day before day 2: add-one base


def test_the_predict_path_reproduces_the_scores_the_fit_accumulated():
    rng = np.random.default_rng(2)
    table, truth = _table(800, 3, rng)
    head = ListwiseBooster(**SMALL).fit(table, truth)
    np.testing.assert_allclose(head.decision_function(table), head.train_scores_, atol=1e-9)


def test_a_newton_leaf_is_minus_g_over_h_plus_lambda():
    """One step from zero scores on two candidates: every row's gradient is p - y = +-1/2 and its
    hessian 1/4; a stump on the one informative column separates them exactly."""
    table = np.zeros((40, 2, 1))
    table[:, 0, 0] = 1.0                                  # candidate 0 always marked; it is the answer
    truth = np.zeros(40, dtype=int)
    head = ListwiseBooster(n_iter=1, learning_rate=1.0, max_leaf_nodes=2, l2_regularization=1.0,
                           min_samples_leaf=5).fit(table, truth)
    lam = 1.0
    # the answers' leaf: 40 rows of g = -1/2, h = 1/4; the others' leaf: g = +1/2
    np.testing.assert_allclose(head.train_scores_[:, 0], 20.0 / (10.0 + lam))
    np.testing.assert_allclose(head.train_scores_[:, 1], -20.0 / (10.0 + lam))


def test_the_identity_free_heads_drop_the_base_rate_and_keep_only_the_operators_deviation():
    rng = np.random.default_rng(3)
    table = rng.random((5, 3, len(CANDIDATE_ROW_NAMES)))
    index = {name: i for i, name in enumerate(CANDIDATE_ROW_NAMES)}
    base = rng.random((5, 3))
    np.testing.assert_array_equal(head_table(table, "r11", base), table)
    noid, lift = head_table(table, "r11_noid", base), head_table(table, "r11_lift", base)
    assert noid.shape[2] == len(head_columns("r11_noid")) == len(CANDIDATE_ROW_NAMES) - len(IDENTITY_COLUMNS)
    assert not set(IDENTITY_COLUMNS) & set(head_columns("r11_noid"))
    assert head_columns("r11_lift")[-1] == "airline_lift" and "prior_share" not in head_columns("r11_lift")
    np.testing.assert_array_equal(noid[:, :, 0], table[:, :, index[head_columns("r11_noid")[0]]])
    np.testing.assert_allclose(lift[:, :, -1], table[:, :, index["airline_share"]] - base)

