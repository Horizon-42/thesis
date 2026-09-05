"""The time-free path metrics, on paths whose distances are known before any model runs.

Each metric has one property the others lack: chamfer is order-blind, Fréchet is
order-preserving, the arc-aligned ADE is speed-free, the along-path lag is the speed —
and the test for each is the case that separates it from its neighbours."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import geometric_metrics as gm  # noqa: E402


def _line(length_m: float, *, offset_n: float = 0.0, speed_mps: float = 100.0, alt: float = 0.0, n: int = 51):
    s = np.linspace(0.0, length_m, n)
    return np.stack([s, np.full(n, offset_n), np.full(n, alt), s / speed_mps], 1)


def test_identical_paths_score_zero():
    path = _line(10_000.0)
    metrics = gm.path_metrics(path, path)
    assert metrics["chamfer_m"] == 0.0 and metrics["frechet_m"] == 0.0 and metrics["arc_ade_m"] == 0.0
    assert metrics["along_path_lag_median_s"] == 0.0 and metrics["duration_error_s"] == 0.0
    assert metrics["path_length_ratio"] == pytest.approx(1.0)


def test_parallel_offset_lines_measure_the_offset_in_every_metric():
    truth, pred = _line(10_000.0), _line(10_000.0, offset_n=300.0, alt=50.0)
    metrics = gm.path_metrics(pred, truth)
    assert metrics["chamfer_m"] == pytest.approx(300.0, abs=1e-6)
    assert metrics["frechet_m"] == pytest.approx(300.0, abs=1e-6)
    # 3D at the same arc fraction: the 50 m height offset joins the 300 m lateral one.
    assert metrics["arc_ade_m"] == pytest.approx(np.hypot(300.0, 50.0), abs=1e-6)


def test_resampling_keeps_both_endpoints_without_a_near_duplicate():
    # 1050 m: the end is 50 m past the last grid point — appended as its own point.
    pts = gm.resample_by_step(_line(1_050.0)[:, :2], 100.0)
    assert pts[0, 0] == 0.0 and pts[-1, 0] == pytest.approx(1_050.0)
    assert np.allclose(np.diff(pts[:-1, 0]), 100.0) and len(pts) == 12
    # 1010 m: within half a step of the last grid point — that point snaps onto the end.
    pts = gm.resample_by_step(_line(1_010.0)[:, :2], 100.0)
    assert pts[-1, 0] == pytest.approx(1_010.0) and len(pts) == 11
    assert 50.0 < np.diff(pts[:, 0]).min() and np.diff(pts[:, 0]).max() <= 150.0
    assert len(gm.resample_by_step(_line(149.0)[:, :2], 100.0)) == 2
    short = np.array([[0.0, 0.0], [30.0, 0.0]])
    assert np.array_equal(gm.resample_by_step(short, 100.0), short)


def test_a_doubled_back_path_fools_chamfer_but_not_frechet():
    """Chamfer is order-blind: A→B→A→B covers the same points as A→B. Fréchet must walk
    both paths monotonically, so the excursion costs half the segment length."""
    truth = _line(4_000.0)
    forth = _line(4_000.0)[:, :2]
    back = forth[::-1][1:]
    doubled = np.concatenate([forth, back, forth[1:]], axis=0)
    assert gm.chamfer_m(doubled, truth) < 1.0
    assert gm.discrete_frechet_m(doubled, truth) == pytest.approx(2_000.0, abs=1.0)


def _frechet_reference(a: np.ndarray, b: np.ndarray) -> float:
    @lru_cache(maxsize=None)
    def c(i: int, j: int) -> float:
        d = float(np.linalg.norm(a[i] - b[j]))
        if i == 0 and j == 0:
            return d
        if i == 0:
            return max(c(0, j - 1), d)
        if j == 0:
            return max(c(i - 1, 0), d)
        return max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
    return c(len(a) - 1, len(b) - 1)


@pytest.mark.parametrize("seed", range(5))
def test_anti_diagonal_frechet_matches_the_textbook_recursion(seed):
    rng = np.random.default_rng(seed)
    a = np.cumsum(rng.normal(size=(rng.integers(1, 25), 2)), axis=0)
    b = np.cumsum(rng.normal(size=(rng.integers(1, 25), 2)), axis=0)
    # A step longer than either path leaves the raw vertices in place.
    assert gm.discrete_frechet_m(a, b, step=1e9) == pytest.approx(_frechet_reference(a, b))


def test_frechet_degenerate_shapes():
    """One path a single point: the distance is the farthest point of the other."""
    point = np.array([[0.0, 0.0]])
    line = np.array([[0.0, 0.0], [300.0, 0.0], [500.0, 0.0]])
    assert gm.discrete_frechet_m(point, line, step=1e9) == 500.0
    assert gm.discrete_frechet_m(line, point, step=1e9) == 500.0


def test_arc_aligned_ade_ignores_speed_and_the_lag_reports_it():
    """The same path flown at half speed: no geometric error, a lag of t_truth(f) at every
    fraction f, and a duration error of the whole truth duration."""
    truth = _line(10_000.0, speed_mps=100.0)          # 100 s
    slow = _line(10_000.0, speed_mps=50.0, n=23)      # 200 s, different sampling too
    metrics = gm.path_metrics(slow, truth)
    assert metrics["arc_ade_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["chamfer_m"] == pytest.approx(0.0, abs=1e-6)
    lag = gm.along_path_lag_s(slow[:, [0, 1, 3]], truth[:, [0, 1, 3]])
    fractions = np.arange(1, gm.ARC_POINTS + 1) / gm.ARC_POINTS
    assert np.allclose(lag, 100.0 * fractions)
    assert lag[-1] == pytest.approx(100.0) == pytest.approx(metrics["duration_error_s"])
    assert metrics["along_path_lag_median_s"] == pytest.approx(np.median(100.0 * fractions))


def test_arc_family_compares_fractions_of_each_paths_own_length():
    """A prediction twice as long as the truth, both at 100 m/s: at fraction f the
    prediction is at 2·f·L (time 2·f·T), the truth at f·L (time f·T) — the position error
    is f·L and the lag is f·T, i.e. the arc family reads the LENGTH mismatch as both."""
    truth, pred = _line(10_000.0), _line(20_000.0)
    fractions = np.arange(1, gm.ARC_POINTS + 1) / gm.ARC_POINTS
    assert gm.arc_aligned_ade_m(pred, truth) == pytest.approx(10_000.0 * fractions.mean())
    lag = gm.along_path_lag_s(pred[:, [0, 1, 3]], truth[:, [0, 1, 3]])
    assert np.allclose(lag, 100.0 * fractions)
    metrics = gm.path_metrics(pred, truth)
    assert metrics["path_length_ratio"] == pytest.approx(2.0)
    assert metrics["duration_error_s"] == pytest.approx(100.0)
    # A long-but-smooth prediction is a route: the length mismatch is an error, not an artifact.
    assert metrics["reversal_share"] == 0.0 and metrics["arc_family_valid"] is True


def test_a_saw_tooth_polyline_is_not_a_route():
    """The state output's exported polyline reverses heading at every other node; its arc
    length is not the route's, so the flight leaves the arc family — chamfer stays."""
    truth = _line(10_000.0)
    saw = _line(10_000.0, n=101)
    saw[1::2, 1] = 150.0                                  # every other node 150 m to the side
    metrics = gm.path_metrics(saw, truth)
    assert metrics["reversal_share"] > 0.9 and metrics["arc_family_valid"] is False
    assert metrics["path_length_ratio"] > 1.5
    assert metrics["chamfer_m"] < 100.0
    assert gm.reversal_share(truth) == 0.0
    # A saw-tooth TRUTH disqualifies the flight just the same (both paths are arc-parametrised).
    assert gm.path_metrics(truth, saw)["arc_family_valid"] is False


def test_reversal_share_ignores_repeated_nodes_and_wraps_at_180_degrees():
    # A repeated node is a zero-length step: arctan2(0, 0) would inject a due-east heading.
    assert gm.reversal_share(np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])) == 0.0
    # A near-straight westbound path: headings straddle ±180° and must not read as reversals.
    west = np.array([[0.0, 0.0], [-100.0, 0.5], [-200.0, -0.5], [-300.0, 0.5], [-400.0, -0.5]])
    assert gm.reversal_share(west) == 0.0
    assert gm.reversal_share(np.array([[0.0, 0.0], [100.0, 0.0]])) == 0.0


def _record(*, observed_short_m: float, pred_to_threshold: bool):
    """One exported flight: a straight-in truth observed to ``observed_short_m`` before the
    threshold (the chart origin), a prediction on the same line, both at 100 m/s."""
    lat0, lon0 = 36.0, -78.8
    target = {"lat": lat0, "lon": lon0, "alt": 100.0, "psi": 0.0, "gamma": -0.05}
    m_per_deg_lon = gm.metres_per_deg_lon(lat0)

    def rows(e_start: float, e_end: float, t0: float, n: int):
        e = np.linspace(e_start, e_end, n)
        t = t0 + (e - e_start) / 100.0
        return [{"t": float(ti), "lat": lat0, "lon": lon0 + ei / m_per_deg_lon, "alt": 100.0}
                for ei, ti in zip(e, t)]

    observed = rows(-10_000.0, -observed_short_m, 0.0, 41)
    true_final_time_s = 100.0                     # the fitted tail reaches the threshold at 100 s
    pred_end = 0.0 if pred_to_threshold else -observed_short_m
    predicted = rows(-10_000.0, pred_end, 0.0, 81)
    states = {"observed_states": [{"t": -10.0, "lat": lat0, "lon": lon0 - 0.12, "alt": 100.0}] + observed,
              "predicted_states": predicted}
    eval_record = {"target_state": target}
    row = {"true_final_time_s": true_final_time_s}
    return eval_record, states, row


def test_closed_truth_ends_at_the_threshold_and_observed_truth_carries_the_gap():
    eval_record, states, row = _record(observed_short_m=400.0, pred_to_threshold=True)
    closed = gm.record_geometry(eval_record, states, row, geometry_truth=gm.GEOMETRY_TRUTH_CLOSED)
    observed = gm.record_geometry(eval_record, states, row, geometry_truth=gm.GEOMETRY_TRUTH_OBSERVED)
    # Closed: the truth is the whole line to the threshold, the prediction flies exactly it.
    for key in ("chamfer_m", "frechet_m", "arc_ade_m", "duration_error_s"):
        assert closed[key] == pytest.approx(0.0, abs=1e-6), key
    assert closed["path_length_ratio"] == pytest.approx(1.0)
    assert closed["truth_closure_m"] == pytest.approx(400.0, abs=1e-6)
    assert closed["truth_closure_s"] == pytest.approx(4.0)
    # Observed: the truth stops 400 m / 4 s short; the endpoint-sensitive metrics see it,
    # and the duration error is read against the last observed time.
    assert observed["frechet_m"] == pytest.approx(400.0, abs=1e-6)
    assert observed["arc_ade_m"] > 100.0
    assert observed["duration_error_s"] == pytest.approx(4.0)
    assert observed["truth_closure_m"] == 0.0 and observed["truth_closure_s"] == 0.0
    assert 0.0 < observed["chamfer_m"] < 50.0


def test_record_geometry_refuses_an_unknown_truth():
    eval_record, states, row = _record(observed_short_m=400.0, pred_to_threshold=True)
    with pytest.raises(ValueError, match="geometry_truth"):
        gm.record_geometry(eval_record, states, row, geometry_truth="supervision")


def _rows(valid_flags, durations):
    return [{"chamfer_m": 100.0 * (i + 1), "frechet_m": 200.0 * (i + 1), "arc_ade_m": 300.0 * (i + 1),
             "path_length_ratio": 1.0 if valid else 2.0, "reversal_share": 0.0 if valid else 0.5,
             "truth_reversal_share": 0.0, "arc_family_valid": valid,
             "along_path_lag_median_s": -5.0 * (i + 1),
             "along_path_lag_abs_mean_s": 6.0, "duration_error_s": duration,
             "truth_closure_m": 380.0, "truth_closure_s": 6.0}
            for i, (valid, duration) in enumerate(zip(valid_flags, durations))]


def test_summary_block_and_table_cells_agree():
    block = gm.summarize(_rows((True, True, True), (-30.0, 20.0, 10.0)))
    assert block["chamfer_median_m"] == 200.0 and block["frechet_median_m"] == 400.0
    assert block["arc_ade_mean_m"] == 600.0 and block["along_path_lag_flight_median_s"] == -10.0
    # Signed and absolute medians differ: |−30, 20, 10| → 20; signed → 10.
    assert block["duration_error_abs_median_s"] == 20.0 and block["duration_error_median_s"] == 10.0
    assert block["arc_family_flights"] == 3 and block["arc_family_share"] == 1.0 and block["arc_family_printed"]
    assert gm.geometry_table_cells(block) == ["200", "400", "600", "1.00", "20.0", "-10.0"]
    assert len(gm.GEOMETRY_TABLE_HEADER) == 6 and not any("|" in h for h in gm.GEOMETRY_TABLE_HEADER)
    notice = gm.geometry_truth_notice(gm.GEOMETRY_TRUTH_CLOSED, block, 3)
    assert "CLOSED" in notice and "380 m / 6 s" in notice and "100 m steps" in notice
    observed = gm.geometry_truth_notice(gm.GEOMETRY_TRUTH_OBSERVED, block, 3)
    assert "OBSERVED" in observed and "100 m steps" in observed and "64 fractions" in observed


def test_arc_family_aggregates_over_route_polylines_only_and_states_the_share():
    """A block mixing routes and saw-teeth: the arc family is the routes' number with the
    exact share printed beside it; below the printing share it is n/a in the table AND
    null in the JSON (a two-flight mean must never be quotable as the arm's arc-ADE).
    Chamfer / Fréchet / Δdur always cover the whole block."""
    rows = _rows((True,) * 19 + (False,), (0.0,) * 20)
    block = gm.summarize(rows)
    assert block["arc_family_flights"] == 19 and block["arc_family_share"] == pytest.approx(0.95)
    assert block["arc_ade_mean_m"] == pytest.approx(np.mean([300.0 * (i + 1) for i in range(19)]))
    assert block["chamfer_median_m"] == np.median([100.0 * (i + 1) for i in range(20)])
    cells = gm.geometry_table_cells(block)
    assert cells[2] == "3000 (95.0%)" and cells[5] == "-50.0 (95.0%)"
    # 1499 of 1501: the share is printed exactly, never rounded up to 100 %.
    nearly = gm.summarize(_rows((True,) * 1499 + (False,) * 2, (0.0,) * 1501))
    assert gm.geometry_table_cells(nearly)[2].endswith("(99.9%)")
    mostly_saw = gm.summarize(_rows((True, False, False, False), (0.0,) * 4))
    assert mostly_saw["arc_family_printed"] is False and mostly_saw["arc_family_flights"] == 1
    for key in ("arc_ade_mean_m", "arc_ade_median_m", "along_path_lag_flight_median_s", "along_path_lag_abs_mean_s"):
        assert mostly_saw[key] is None, key
    assert json.loads(json.dumps(mostly_saw, allow_nan=False))["arc_ade_mean_m"] is None
    cells = gm.geometry_table_cells(mostly_saw)
    assert cells[2] == "n/a" and cells[5] == "n/a" and cells[0] == "250" and cells[3] == "2.00"
    none = gm.summarize(_rows((False, False), (0.0, 0.0)))
    assert none["arc_ade_mean_m"] is None and none["arc_family_flights"] == 0
    assert gm.geometry_table_cells(none)[2] == "n/a"
    with pytest.raises(ValueError, match="at least one flight"):
        gm.summarize([])
