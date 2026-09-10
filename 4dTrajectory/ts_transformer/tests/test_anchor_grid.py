"""The anchor grid has ONE definition, and the A0 runner uses it.

`experiments/anytime_curve.py` draws the curve; the `anchor-grid-common-grid-ade` selection
metric picks the epoch on four of the same bins. Two grids that merely agreed today would
make "the curve improved" and "the epoch was selected on the curve" statements about
different anchors, so these tests pin that they are literally the same objects, and that
the two rules the grid is made of (per-flight admissibility, strata fixed at L−1) kept
their behaviour when they moved out of the runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ts_transformer.data.anchor_grid as anchor_grid
import ts_transformer.experiments.anytime_curve as runner
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    remaining_path_profile_m,
    strata_masks,
)
from ts_transformer.config import TSConfig
from ts_transformer.data.dataset import FlightSeries, build_series
from ts_transformer.data.synthetic import synthetic_arrivals

AIRPORT, RUNWAY = "KRDU", "05L"


def _bare_series(horizontal: list[tuple[float, float]], *, dt_s: float = 2.0) -> FlightSeries:
    """A flight that is nothing but a horizontal path — all the anchor rules read."""
    values = np.zeros((len(horizontal), 6), dtype=np.float64)
    values[:, :2] = np.array(horizontal, dtype=np.float64)
    return FlightSeries(
        flight_id="BARE1_05L_abc123_20260101T000000Z",
        scenario=SimpleNamespace(target=None),
        frame=None,
        times=np.arange(len(horizontal), dtype=np.float64) * dt_s,
        values=values,
    )


@pytest.fixture(scope="module")
def cohort() -> list[FlightSeries]:
    config = TSConfig(
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8,
        val_fraction=0.25, test_fraction=0.25,
    )
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    return series


# ── one definition, two consumers ───────────────────────────────────────────

def test_the_runner_grid_is_the_packages() -> None:
    """Not "equal today": the same objects, so a change here cannot miss the runner."""
    assert runner.bin_anchor is anchor_grid.bin_anchor
    assert runner.anchors_for_bin is anchor_grid.anchors_for_bin
    assert runner.strata_fixed_at_l1 is anchor_grid.strata_fixed_at_l1
    assert runner.DEFAULT_MIN_FUTURE_S == anchor_grid.DEFAULT_GRID_MIN_FUTURE_S
    # The `partial` threshold is one number: the curve refuses to READ such a bin and the
    # selection metric refuses to AVERAGE it, off the same constant.
    assert runner.PARTIAL_COVERAGE is anchor_grid.PARTIAL_COVERAGE
    assert runner.DEFAULT_BINS_KM == "20,16,12,8,6,4,2"
    assert tuple(
        float(token) for token in runner.DEFAULT_BINS_KM.split(",")
    ) == tuple(float(value) for value in anchor_grid.DEFAULT_ANCHOR_GRID_KM)


def test_the_selection_bins_are_bins_of_the_measurement_grid() -> None:
    """A1's CANDIDATES are a subset of A0's grid — never a bin nobody plotted.

    Which candidates a run actually selects on is decided per cohort against
    `PARTIAL_COVERAGE` (`validation.build_anchor_grid_validation_plans`); on the real KRDU
    validation cohort 16 km is dropped at 37 %.
    """
    assert set(anchor_grid.VALIDATION_ANCHOR_GRID_KM) <= set(
        anchor_grid.DEFAULT_ANCHOR_GRID_KM
    )
    assert anchor_grid.VALIDATION_ANCHOR_GRID_KM == (16, 12, 8, 6)
    assert anchor_grid.VALIDATION_ANCHOR_GRID_M == (16000.0, 12000.0, 8000.0, 6000.0)
    # The three bins A0 measures and A1 will not even consider, with their reasons in the
    # module docstring: 20 km is under half coverage on every measured cohort, and
    # 4 / 2 km are under the future floor by construction.
    assert set(anchor_grid.DEFAULT_ANCHOR_GRID_KM) - set(
        anchor_grid.VALIDATION_ANCHOR_GRID_KM
    ) == {20, 4, 2}


# ── the per-flight anchor rule ──────────────────────────────────────────────

def test_the_bin_takes_the_closest_sample() -> None:
    series = _bare_series([(-3_000.0, 0.0), (-2_000.0, 0.0), (-1_000.0, 0.0), (0.0, 0.0)])
    profile = remaining_path_profile_m(series)
    for target, expected in ((3_000.0, 0), (1_800.0, 1), (1_200.0, 2)):
        assert anchor_grid.bin_anchor(
            series, profile, target, seq_len=1, min_future_s=0.0
        ) == expected


def test_an_inadmissible_closest_sample_empties_the_bin() -> None:
    """Both halves of admissibility: the lookback before it, the truth after it."""
    series = _bare_series([(-3_000.0, 0.0), (-2_000.0, 0.0), (-1_000.0, 0.0), (0.0, 0.0)])
    profile = remaining_path_profile_m(series)
    assert anchor_grid.bin_anchor(series, profile, 3_000.0, seq_len=3, min_future_s=0.0) is None
    assert anchor_grid.bin_anchor(series, profile, 1_000.0, seq_len=3, min_future_s=0.0) == 2
    assert anchor_grid.bin_anchor(series, profile, 1_000.0, seq_len=1, min_future_s=2.0) == 2
    assert anchor_grid.bin_anchor(series, profile, 1_000.0, seq_len=1, min_future_s=3.0) is None


def test_the_bin_map_is_the_per_flight_rule_applied_once_per_flight(cohort) -> None:
    """`anchors_for_bin` is the loop, not a second rule: absent = no reading here."""
    profiles = anchor_grid.remaining_path_profiles(cohort)
    for target_m in anchor_grid.VALIDATION_ANCHOR_GRID_M + (4_000.0,):
        anchors = anchor_grid.anchors_for_bin(
            cohort, profiles, target_m,
            seq_len=8, min_future_s=anchor_grid.DEFAULT_GRID_MIN_FUTURE_S,
        )
        expected = {
            index: anchor_grid.bin_anchor(
                item, profiles[index], target_m,
                seq_len=8, min_future_s=anchor_grid.DEFAULT_GRID_MIN_FUTURE_S,
            )
            for index, item in enumerate(cohort)
        }
        assert anchors == {
            index: anchor for index, anchor in expected.items() if anchor is not None
        }


def test_the_four_candidate_bins_are_populated_and_four_km_is_not(cohort) -> None:
    """The coverage claim behind `VALIDATION_ANCHOR_GRID_KM`, on a real anchor rule.

    4 km leaves ≈ 45-50 s of truth on this cohort, under the 60 s floor — which is exactly
    why it is measured by A0 and not a candidate for A1. The four candidates are fully
    covered HERE; on the real KRDU cohort 16 km is not, and the coverage gate drops it.
    """
    profiles = anchor_grid.remaining_path_profiles(cohort)
    populated = {
        target_m: len(anchor_grid.anchors_for_bin(
            cohort, profiles, target_m,
            seq_len=8, min_future_s=anchor_grid.DEFAULT_GRID_MIN_FUTURE_S,
        ))
        for target_m in anchor_grid.VALIDATION_ANCHOR_GRID_M + (4_000.0,)
    }
    assert all(populated[target_m] == len(cohort)
               for target_m in anchor_grid.VALIDATION_ANCHOR_GRID_M)
    assert populated[4_000.0] == 0


# ── the stratum rule ────────────────────────────────────────────────────────

def test_the_strata_are_the_covariates_read_once_at_l_minus_one(cohort) -> None:
    keys = [item.dataset_id for item in cohort]
    masks = anchor_grid.strata_fixed_at_l1(cohort, keys, seq_len=8)
    expected = strata_masks(
        {key: approach_difficulty(item, 7).to_dict() for key, item in zip(keys, cohort)},
        keys,
    )
    assert set(masks) == set(expected)
    for stratum, mask in masks.items():
        assert np.array_equal(mask, expected[stratum])
    assert masks[STRATUM_ALL].all()
    # The label does NOT follow a later anchor: every flight is established close in, and
    # a per-bin relabel would move the whole cohort into one stratum.
    assert masks[STRATUM_STRAIGHT_IN].sum() + masks[STRATUM_VECTORED].sum() \
        + masks[STRATUM_ESTABLISHED].sum() == len(cohort)
