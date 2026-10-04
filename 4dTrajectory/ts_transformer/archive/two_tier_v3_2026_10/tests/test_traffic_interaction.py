"""Multi-aircraft M1's readout (`experiments/traffic_interaction`): the step flags, the phase strata and the matched
difference on hand-built flights."""

import numpy as np
import pytest

from ts_transformer.prior.scene import Presence


def _presence(key: str, start_s: float, to_threshold_m, *, runway: str = "R", landing_s: float) -> Presence:
    to_go = np.asarray(to_threshold_m, dtype=float)
    return Presence(key, "KXXX", runway, landing_s, True, start_s + 2.0 * np.arange(len(to_go)), to_go)


def test_a_step_has_a_leader_within_10_km_ahead_on_its_runway_and_is_busy_with_three_aircraft():
    from ts_transformer.experiments.traffic_interaction import step_flags

    ego = _presence("E", 0.0, np.linspace(20_000.0, 10_000.0, 11), landing_s=500.0)
    # ahead of E by 12 km, then 8 km (it slows less than E closes): a leader from where it is 10 km or less ahead
    ahead = _presence("A", 0.0, [8_000.0, 7_000.0, 6_000.0, 5_000.0, 4_000.0, 3_000.0, 2_500.0, 2_000.0, 1_500.0,
                                 1_000.0, 500.0], landing_s=300.0)
    other_runway = _presence("B", 4.0, np.full(3, 15_000.0), runway="L", landing_s=200.0)
    flags = step_flags([ego], [ahead, other_runway])[0]
    gap = ego.to_threshold_m - ahead.to_threshold_m
    assert flags["leader"].tolist() == ((gap > 0) & (gap <= 10_000.0)).tolist()
    assert flags["busy"].tolist() == [False, False, True, True, True] + [False] * 6      # B present at 4–8 s


def test_the_phase_stratum_is_the_capture_times_the_distance_bin():
    from ts_transformer.experiments.traffic_interaction import STRATA, phase_strata

    ego = _presence("E", 0.0, [26_000.0, 19_000.0, 12_000.0, 7_000.0, 4_000.0], landing_s=100.0)
    bins = STRATA // 2
    assert phase_strata(ego, 3).tolist() == [4, 3, 2, bins + 1, bins + 0]


def test_the_matched_difference_weights_the_flagged_steps_and_skips_strata_without_others():
    from ts_transformer.experiments.traffic_interaction import _matched

    # stratum 0: flagged 2 steps mean 1.0, others 1 step 0.5; stratum 1: flagged 1 step 3.0, others 3 steps mean 2.0;
    # stratum 2: flagged only (not comparable)
    sums = np.array([[0.5, 2.0], [6.0, 3.0], [0.0, 4.0]])
    counts = np.array([[1, 2], [3, 1], [0, 2]], dtype=float)
    read = _matched(sums, counts, 5)
    assert read["difference"] == pytest.approx((2 * 0.5 + 1 * 1.0) / 3)
    assert read["flagged_steps_covered"] == pytest.approx(3 / 5)


def test_a_group_reads_the_raw_means_the_matched_difference_and_its_interval():
    from ts_transformer.experiments.traffic_interaction import READ, read_group

    rng = np.random.default_rng(0)
    flights = []
    for n in range(40):
        rows = 20
        stratum = np.repeat([0, 1], 10)
        leader = (np.arange(rows) % 2 == 0) & (n % 2 == 0)
        base = np.where(stratum == 1, 1.0, 0.2)                # stratum 1 is harder, whatever the traffic
        nll = np.column_stack([base + 0.1 * leader] * len(READ))   # a leader adds 0.1 inside each stratum
        nll[:8] = np.nan                                        # the observed rows: not counted
        flights.append({"nll": nll, "p_change": np.full((rows, len(READ)), 0.1),
                        "changed": np.zeros((rows, len(READ)), dtype=bool), "leader": leader,
                        "busy": np.zeros(rows, dtype=bool), "stratum": stratum, "cluster": ("KXXX", n // 4)})
    read = read_group(flights, 2, rng)
    assert read["clusters"] == 10
    speed = read["speed"]["leader"]
    assert speed["steps"] == 20 * 6 and speed["nll"]["difference"] == pytest.approx(0.1)
    # every resample holds only flights whose leader steps are 0.1 worse in each stratum: the interval is that point
    assert speed["nll"]["difference_95"] == pytest.approx([0.1, 0.1])
    assert speed["nll"]["resamples_without_a_comparison"] == 0
    assert read["speed"]["busy"]["steps"] == 0 and read["speed"]["busy"]["nll"]["flagged_mean"] is None


def test_the_pool_compares_within_airports_so_a_busier_harder_airport_is_not_read_as_traffic():
    """Airport 1 is harder (NLL 1.0 against 0.2) and busy on every step; airport 0 is busy on half: inside each airport
    busy steps are no harder, so with airport × phase strata the pool's busy difference is 0 (one stratum per airport
    would read 0.64)."""
    from ts_transformer.experiments.traffic_interaction import READ, STRATA, pooled_flights, read_group

    by_airport = {}
    for airport, level, busy_every in (("KAAA", 0.2, 2), ("KBBB", 1.0, 1)):
        by_airport[airport] = []
        for n in range(10):
            rows = 12
            nll = np.full((rows, len(READ)), level)
            by_airport[airport].append({"nll": nll, "p_change": np.zeros((rows, len(READ))),
                                        "changed": np.zeros((rows, len(READ)), dtype=bool),
                                        "leader": np.zeros(rows, dtype=bool), "busy": np.arange(rows) % busy_every == 0,
                                        "stratum": np.zeros(rows, dtype=np.int64), "cluster": (airport, n)})
    read = read_group(pooled_flights(by_airport), 2 * STRATA, np.random.default_rng(0))
    assert read["speed"]["busy"]["nll"]["difference"] == pytest.approx(0.0)
    assert read["speed"]["busy"]["nll"]["flagged_mean"] > read["speed"]["busy"]["nll"]["other_mean"]


def test_the_first_predicted_row_is_not_read(monkeypatch):
    """At row N_LOOK the prior says every column from scratch: its reading starts at the next row."""
    from types import SimpleNamespace

    import torch

    from ts_transformer.experiments import traffic_interaction as ti
    from ts_transformer.prior.scene import N_LOOK

    rows, classes = N_LOOK + 4, 3
    targets = np.ones((rows, 6), dtype=np.int64)
    flight = SimpleNamespace(rows=rows, asked=np.arange(rows)[:, None].repeat(6, 1) >= N_LOOK, targets=targets)
    monkeypatch.setattr(ti, "to_batch", lambda split, indices, device: {
        "targets": torch.as_tensor(targets[None, None])})
    monkeypatch.setattr(ti, "batch_logits", lambda model, batch: [torch.zeros(1, 1, rows, classes)] * 6)
    read = ti.step_readings(None, SimpleNamespace(flights=[flight]), 10_000, torch.device("cpu"))[0]
    counted = ~np.isnan(read["nll"][:, 0])
    assert counted.tolist() == [False] * (N_LOOK + 1) + [True] * 3
    assert read["nll"][N_LOOK + 1, 0] == pytest.approx(np.log(classes))
