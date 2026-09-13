"""Runway-intent R1.1's head and its airline share (`experiments.runway_intent_r11`)."""

from collections import Counter

import numpy as np

from ts_transformer.data.runway_features import CANDIDATE_ROW_NAMES
from ts_transformer.experiments.runway_intent_r11 import (
    AIRLINE_PRIOR_WEIGHT,
    IDENTITY_COLUMNS,
    ListwiseBooster,
    airline_block,
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


def _flights_on(days: list[str], per_day: int) -> tuple[np.ndarray, ...]:
    keys, day_of = [], []
    for d in days:
        for j in range(per_day):
            keys.append(f"{d}-{j}")
            day_of.append(d)
    return np.array(keys), np.array(day_of)


def test_a_training_samples_airline_share_never_counts_its_own_label():
    days = [f"2026-06-{d:02d}" for d in range(1, 21)]
    keys, day_of = _flights_on(days, 3)
    # every flight is sampled twice (two anchors), as R1's samples are
    keys, day_of = np.repeat(keys, 2), np.repeat(day_of, 2)
    n = len(keys)
    operators = np.array(["AAL"] * n)
    labels = np.zeros(n, dtype=int)
    train = np.ones(n, dtype=bool)
    prior = np.array([0.5, 0.5])
    base = airline_shares(operators, labels, day_of, keys, train, prior)
    flipped_labels = labels.copy()
    flipped_labels[:2] = 1                               # the first flight's own label changes
    flipped = airline_shares(operators, flipped_labels, day_of, keys, train, prior)
    np.testing.assert_array_equal(flipped[:2], base[:2])  # its own share does not move
    same_block = np.array([airline_block(d) == airline_block(day_of[0]) for d in day_of])
    np.testing.assert_array_equal(flipped[same_block], base[same_block])  # nor its block's
    assert not np.array_equal(flipped[~same_block], base[~same_block])    # the other blocks see it


def test_validation_reads_every_training_flight_once_and_an_unknown_operator_reads_the_prior():
    keys = np.array(["a", "a", "b", "c", "v", "w"])
    days = np.array(["d1", "d1", "d2", "d3", "d4", "d4"])
    operators = np.array(["AAL", "AAL", "AAL", "AAL", "AAL", ""])
    labels = np.array([0, 0, 0, 1, 1, 1])
    train = np.array([True, True, True, True, False, False])
    prior = np.array([0.6, 0.4])
    shares = airline_shares(operators, labels, days, keys, train, prior)
    # three training flights (a counted once): two on candidate 0, one on candidate 1
    expected = (np.array([2.0, 1.0]) + AIRLINE_PRIOR_WEIGHT * prior) / (3.0 + AIRLINE_PRIOR_WEIGHT)
    np.testing.assert_allclose(shares[4], expected)
    np.testing.assert_allclose(shares[5], prior)


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
    np.testing.assert_array_equal(head_table(table, "r11"), table)
    noid, lift = head_table(table, "r11_noid"), head_table(table, "r11_lift")
    assert noid.shape[2] == len(head_columns("r11_noid")) == len(CANDIDATE_ROW_NAMES) - len(IDENTITY_COLUMNS)
    assert not set(IDENTITY_COLUMNS) & set(head_columns("r11_noid"))
    assert head_columns("r11_lift")[-1] == "airline_lift" and "prior_share" not in head_columns("r11_lift")
    np.testing.assert_array_equal(noid[:, :, 0], table[:, :, index[head_columns("r11_noid")[0]]])
    np.testing.assert_allclose(lift[:, :, -1], table[:, :, index["airline_share"]] - table[:, :, index["prior_share"]])

