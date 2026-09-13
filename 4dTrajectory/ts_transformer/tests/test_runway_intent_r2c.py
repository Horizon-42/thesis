"""Runway-intent R2c's ask grid and lock rules (`experiments.runway_intent_r2c`)."""

from ts_transformer.experiments.runway_intent_r2c import ASK_S, asks, bin_of, flips, lock_picks


def test_the_head_is_asked_at_the_entry_and_every_30_s_until_the_landing():
    assert ASK_S == 30.0
    waypoints = [[t, 0.0, 0.0, 0.0] for t in (5.0, 15.0, 25.0, 35.0, 45.0, 66.0, 75.0)]
    got = asks({"waypoints": waypoints})
    assert [(index, t) for index, t, _ in got] == [(0, 0.0), (3, 30.0), (4, 60.0)]   # nothing after an ask is read
    assert got[0][2] >= got[1][2] >= got[2][2] >= 0.0                                   # the path left only shrinks


def test_each_lock_rule_holds_the_runway_it_names():
    rules = lock_picks([0, 1, 1, 0], [0.5, 0.95, 0.7, 0.99])
    assert rules["never"] == [0, 1, 1, 0]
    assert rules["first"] == [0, 0, 0, 0]
    assert rules["p0.9"] == rules["p0.95"] == [0, 1, 1, 1]     # locked at the first confident ask
    assert rules["p0.8"] == [0, 1, 1, 1]
    assert lock_picks([2, 2], [0.1, 0.2])["p0.95"] == [2, 2]  # never confident: follows the top pick
    assert flips(rules["never"]) == 2 and flips(rules["first"]) == 0


def test_the_anytime_bins_run_far_to_near():
    assert bin_of(45_000.0) == ">40km"
    assert bin_of(35_000.0) == "30-40km"
    assert bin_of(12_000.0) == "10-15km"
    assert bin_of(2_000.0) == "0-3km"
