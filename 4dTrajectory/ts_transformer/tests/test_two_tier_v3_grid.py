"""`dataset.series_from_row`: a series cut at a later row keeps the clock and cuts the truth alike (two-tier plan v3
stage A's bin reading, `docs/history/2026-09_two_tier_v3/2026-09-18_two_tier_plan_v3.zh.md` §5–§6). The tests of that
plan's grid declaration and cohorts are archived in `archive/manoeuvre_lockstep_2026_10/tests/`."""

from __future__ import annotations


import numpy as np
import pytest

from ts_transformer.config import (
    CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, TSConfig, recipe_settings,
)
from ts_transformer.data.dataset import build_series, series_from_row
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.tests.support import AIRPORT, RUNWAY

def _cell_config(seq_len: int, horizon_s: float) -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, control_horizon_s=horizon_s, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=seq_len, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, device="cpu", epochs=1, patience=1, random_train_anchor=True,
        random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM, random_train_anchor_min_future_s=horizon_s,
        split_seed=1337,
    ))
    return TSConfig(**settings)


def test_series_from_row_keeps_the_clock_and_cuts_the_truth_alike():
    config = _cell_config(seq_len=4, horizon_s=20.0)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=3), config, airport=AIRPORT)
    whole = series[0]
    later = series_from_row(whole, 10)
    assert later.n_samples == whole.n_samples - 10 and later.times[0] == whole.times[10]
    np.testing.assert_array_equal(later.values, whole.values[10:])
    assert later.supervision_times[0] == pytest.approx(whole.times[10]) and later.supervision_times[-1] == whole.supervision_times[-1]
    assert later.n_supervision_samples == whole.n_supervision_samples - 10
    assert later.flight_id == whole.flight_id and later.scenario is whole.scenario
    assert series_from_row(whole, 0) is not whole and series_from_row(whole, 0).n_samples == whole.n_samples
    with pytest.raises(ValueError, match="first_row"):
        series_from_row(whole, whole.n_samples)


