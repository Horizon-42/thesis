"""`control_horizon_s` — a FIXED rollout horizon (two-tier L1, `docs/2026-09-17_two_tier_plan_v2.zh.md` §3).

The contract, each piece pinned here: every window set admits only anchors with Δ of truth
after them; a sample's targets cover [0, Δ] at the segment ends; the model rolls exactly Δ
and builds no duration head; the selection truth spans Δ; the axis is named and pinned by
the recipes; and every axis that would decide a duration is refused, not left inert.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_RECIPE_SIMPLE_V2,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    TSConfig,
    absent_field_defaults,
    control_recipe,
    default_anchor,
    recipe_settings,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.dataset import (
    FixedAnchorTrajectoryWindows,
    Normalizer,
    build_series,
    target_horizon_s,
    truth_duration_s,
    window_anchors,
)
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import forecast_approaches
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.fixed_anchor_validation import common_truth_at_anchors
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import fake_data_provenance

AIRPORT, RUNWAY = "KRDU", "05L"
HORIZON_S = 16.0
TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=100.0,
            device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8, dropout=0.0)


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_horizon_s=HORIZON_S,
        final_time_loss_weight=0.0,
        **TINY,
    )
    settings.update(overrides)
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def cohort():
    config = _config()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == 8, report.format()
    return flights, series, config


# ── the config ────────────────────────────────────────────────────────────────

def test_the_axis_is_named_pinned_by_the_recipes_and_absent_as_zero() -> None:
    config = _config()
    assert "horizon=16" in run_display_name(config.to_dict())
    assert control_recipe(config)["horizon_s"] == HORIZON_S
    whole = _config(control_horizon_s=0.0, final_time_loss_weight=1.0)
    assert "horizon_s" not in control_recipe(whole)      # every stored checkpoint's metadata
    assert absent_field_defaults({"prediction_output": PREDICTION_CONTROL})["control_horizon_s"] == 0.0
    # a named recipe predicts the whole approach: the pin refuses a horizon under its name
    # (simple-v2 rather than v3, whose imitation term the horizon refuses before the pin is read)
    with pytest.raises(ValueError, match="frozen.*control_horizon_s"):
        TSConfig(**{**recipe_settings(CONTROL_RECIPE_SIMPLE_V2, keep_name=True), **TINY,
                    "prediction_output": PREDICTION_CONTROL, "control_horizon_s": HORIZON_S,
                    "final_time_loss_weight": 0.0})


def test_every_axis_that_would_decide_a_duration_is_refused() -> None:
    with pytest.raises(ValueError, match="fixes the rollout duration"):
        _config(cta_conditioning="given")
    with pytest.raises(ValueError, match="predicts no duration"):
        _config(duration_head="quantile")
    with pytest.raises(ValueError, match="identically zero"):
        _config(final_time_loss_weight=1.0)
    with pytest.raises(ValueError, match="latent"):
        _config(latent_dim=8)
    with pytest.raises(ValueError, match="imitation"):
        _config(control_imitation_loss_weight=1.0)
    with pytest.raises(ValueError, match="seconds"):
        _config(control_horizon_s=-1.0)
    with pytest.raises(ValueError, match="below control_horizon_s"):
        _config(random_train_anchor=True, random_train_anchor_min_future_s=HORIZON_S / 2)
    _config(random_train_anchor=True, random_train_anchor_min_future_s=HORIZON_S)


# ── the windows ───────────────────────────────────────────────────────────────

def test_every_window_set_admits_only_anchors_with_the_horizon_of_truth_after_them(cohort) -> None:
    _flights, series, config = cohort
    whole = _config(control_horizon_s=0.0, final_time_loss_weight=1.0)
    for item in series:
        horizon = window_anchors(item, config)
        unbounded = window_anchors(item, whole)
        assert horizon.start == unbounded.start == default_anchor(config)
        assert horizon.stop < unbounded.stop
        assert all(truth_duration_s(item, anchor) >= HORIZON_S - 1e-9 for anchor in horizon)
        assert truth_duration_s(item, horizon.stop) < HORIZON_S
        assert target_horizon_s(item, horizon.start, config) == HORIZON_S
        assert target_horizon_s(item, horizon.start, whole) == truth_duration_s(item, horizon.start)


def test_the_targets_cover_the_horizon_at_the_segment_ends(cohort) -> None:
    _flights, series, config = cohort
    normalizer = Normalizer.fit(series)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    _x, y, weights, final_time_s, _flight_weights, _context, _dense = unpack_batch(windows.batch([0, 1]))
    assert final_time_s.tolist() == [HORIZON_S, HORIZON_S]
    for row, index in enumerate((0, 1)):
        s_idx, anchor = windows.index[index]
        item = series[s_idx]
        ends = float(item.times[anchor]) + (np.arange(1, config.n_segments + 1) * HORIZON_S / config.n_segments)
        expected = np.column_stack([
            np.interp(ends, item.supervision_times, item.supervision_values[:, c]) for c in range(len(config.channels))
        ])
        assert y[row].numpy() == pytest.approx(normalizer.encode(expected).astype(np.float32), abs=1e-5)
        assert bool((weights[row] > 0.0).all())              # inside the truth: every node supervised
    # the heading-rate reference reads the same span: its endpoints are the segment ends of Δ
    heading = _config(control_heading_rate_loss_weight=1.0)
    rows = FixedAnchorTrajectoryWindows(series, heading, normalizer).context.row(0)
    assert rows["reference_heading_rate_weight"].tolist() == [1.0] * heading.n_segments


# ── the model ─────────────────────────────────────────────────────────────────

def test_the_model_rolls_exactly_the_horizon_and_builds_no_duration_head(cohort) -> None:
    _flights, series, config = cohort
    normalizer = Normalizer.fit(series)
    torch.manual_seed(0)
    model = build_model(config, normalizer)
    assert model.final_time_head is None and model.duration_quantile_head is None
    assert not any(name.startswith("final_time_head") for name, _p in model.named_parameters())
    dynamics = probe_dynamics(3, torch.device("cpu"), config)
    history = torch.zeros((3, config.seq_len, config.enc_in), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(history, dynamics)
    assert prediction.final_time_s.tolist() == [HORIZON_S] * 3
    assert prediction.segment_durations.tolist() == [[HORIZON_S / config.n_segments] * config.n_segments] * 3
    assert prediction.duration_quantiles_s is None
    forecasts = forecast_approaches(model, series[:2], config, normalizer, device=torch.device("cpu"))
    for item, forecast in zip(series[:2], forecasts, strict=True):
        assert forecast.final_time_s == pytest.approx(HORIZON_S)
        assert forecast.predicted_final_time_s == pytest.approx(HORIZON_S)
        assert float(forecast.times[-1] - item.times[forecast.anchor]) == pytest.approx(HORIZON_S)
        assert np.isfinite(forecast.values).all()


def test_the_selection_truth_spans_the_horizon(cohort) -> None:
    _flights, series, config = cohort
    anchor = default_anchor(config)
    truth, durations, progress = common_truth_at_anchors(series, config, 8, [anchor] * len(series))
    assert durations.tolist() == [HORIZON_S] * len(series)
    item = series[0]
    times = float(item.times[anchor]) + progress * HORIZON_S
    expected = np.column_stack([np.interp(times, item.supervision_times, item.supervision_values[:, c]) for c in range(3)])
    assert truth[0, :, :3] == pytest.approx(expected, abs=1e-3)


def test_a_fixed_horizon_run_trains_loads_and_predicts(tmp_path, cohort) -> None:
    _flights, series, config = cohort
    torch.manual_seed(0)
    train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.control_horizon_s == HORIZON_S and model.final_time_head is None
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert np.isfinite(history["history"][-1]["val_loss"])
    assert np.isfinite(history["history"][-1]["validation_selection_value"])
    forecasts = forecast_approaches(model, series[:2], loaded, normalizer, device=torch.device("cpu"))
    assert all(f.final_time_s == pytest.approx(HORIZON_S) for f in forecasts)
