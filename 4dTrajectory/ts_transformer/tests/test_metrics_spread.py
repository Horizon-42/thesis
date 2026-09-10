"""The spread metric.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math

import numpy as np
import pytest

import ts_transformer.channels as ch
from ts_transformer.metrics import raw_kinematic_metrics


# ── Metrics ──────────────────────────────────────────────────────────────────


def test_spread_matches_the_gate_side_signed_spread():
    # metrics.signed_spread is the VECTORISED twin of evaluation/stats.signed_spread (that one
    # is stdlib-only by design and would sort millions of boxed floats here). This seam
    # test is what makes "same statistic" a checked property instead of a mirror comment:
    # if either side changes its percentile method or keys, this fails.
    from evaluation.stats import signed_spread
    from ts_transformer.metrics import signed_spread as vectorised_spread

    values = np.array([3.0, -1.5, 0.25, -7.0, 4.0, 2.5, -0.75])
    ours, theirs = vectorised_spread(values), signed_spread(values.tolist())
    assert set(ours) == set(theirs)
    for key in ours:
        assert ours[key] == pytest.approx(theirs[key])


def test_raw_kinematic_metrics_use_nonuniform_segment_durations():
    # Constant 1 m/s² eastward acceleration sampled at deliberately nonuniform node times.
    # Trapezoidal velocity integration is exact here, acceleration is constant, and jerk is
    # zero. A metric that silently assumes uniform N spacing fails this test.
    node_times = np.array([0.0, 1.0, 3.0, 6.0])
    nodes = np.zeros((1, len(node_times), len(ch.CHANNELS)), dtype=np.float64)
    nodes[0, :, ch.IDX["e"]] = 0.5 * node_times**2
    nodes[0, :, ch.IDX["edot"]] = node_times

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.diff(node_times)[None, :]
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)
    assert block["heading_consistency_p95_deg"] == pytest.approx(0.0, abs=1e-12)
    assert block["turn_rate_p95_deg_s"] == pytest.approx(0.0, abs=1e-12)
    assert block["acceleration_p95_mps2"] == pytest.approx(1.0)
    assert block["jerk_p95_mps3"] == pytest.approx(0.0, abs=1e-12)


def test_raw_kinematic_metrics_measure_geometric_turn_acceleration_and_jerk():
    # Three one-second position segments turn east -> north -> west. Node velocities are
    # chosen so their trapezoidal midpoint exactly matches each geometric segment velocity;
    # consistency and heading error must therefore be zero while the path still has a
    # 90 deg/s turn rate, sqrt(200) m/s² acceleration and 20 m/s³ jerk.
    nodes = np.zeros((1, 4, len(ch.CHANNELS)), dtype=np.float64)
    nodes[0][:, [ch.IDX["e"], ch.IDX["n"]]] = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
    ])
    nodes[0][:, [ch.IDX["edot"], ch.IDX["ndot"]]] = np.array([
        [10.0, 0.0], [10.0, 0.0], [-10.0, 20.0], [-10.0, -20.0],
    ])

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.ones((1, 3), dtype=np.float64)
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(0.0, abs=1e-12)
    assert block["heading_consistency_p95_deg"] == pytest.approx(0.0, abs=1e-12)
    assert block["turn_rate_p95_deg_s"] == pytest.approx(90.0)
    assert block["acceleration_p95_mps2"] == pytest.approx(math.sqrt(200.0))
    assert block["jerk_p95_mps3"] == pytest.approx(20.0)


def test_raw_kinematic_metrics_detect_velocity_heading_disagreement():
    nodes = np.zeros((1, 4, len(ch.CHANNELS)), dtype=np.float64)
    nodes[0, :, ch.IDX["e"]] = np.arange(4) * 10.0
    nodes[0, :, ch.IDX["ndot"]] = 10.0

    block = raw_kinematic_metrics(
        nodes[:, 0], nodes[:, 1:], np.ones((1, 3), dtype=np.float64)
    )

    assert block["position_velocity_rmse_mps"] == pytest.approx(math.sqrt(200.0 / 3.0))
    assert block["heading_consistency_p95_deg"] == pytest.approx(90.0)
