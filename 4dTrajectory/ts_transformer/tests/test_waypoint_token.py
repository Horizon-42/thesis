"""`plan_conditioning=waypoints` — the coarse plan's next waypoints as the control decoder's plan
token (two-tier v2, `docs/2026-09-17_two_tier_plan_v2.zh.md` §3).

Pinned: the token's layout and width; the training row is the TRUTH's position at each coarse
segment after the anchor, relative to the anchor's own row; a waypoint the truth never reaches
is invalid and carries nothing; every batch surface carries the same width; a waypoint run
trains, predicts and names itself; the axis is refused where its token is undefined.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PLAN_CONDITIONING_TRUTH_NEXT,
    PLAN_CONDITIONING_WAYPOINTS,
    PLAN_WAYPOINT_SEGMENT_S,
    PREDICTION_CONTROL,
    TSConfig,
    default_anchor,
    plan_waypoint_count,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series, truth_duration_s
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import forecast_approaches
from ts_transformer.outputs.control.plan_token import (
    PLAN_TOKEN_KEY,
    PLAN_TOKEN_WIDTH,
    WAYPOINT_DELTA_SCALE_M,
    WAYPOINT_LEAD_SCALE_S,
    Waypoints,
    plan_token_width,
    probe_plan_token,
    truth_waypoint_token,
    truth_waypoints,
    waypoint_token,
)
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import fake_data_provenance

AIRPORT, RUNWAY = "KRDU", "05L"
HORIZON_S = 2 * PLAN_WAYPOINT_SEGMENT_S
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
        plan_conditioning=PLAN_CONDITIONING_WAYPOINTS,
        plan_conditioning_dropout=0.5,
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


# ── the token ─────────────────────────────────────────────────────────────────

def test_the_token_is_scaled_deltas_leads_valid_bits_and_a_present_flag() -> None:
    config = _config()
    count = plan_waypoint_count(config)
    assert count == 2 and plan_token_width(config) == 5 * count + 1 == 11
    assert plan_token_width(_config(plan_conditioning=PLAN_CONDITIONING_TRUTH_NEXT, control_horizon_s=HORIZON_S)) == PLAN_TOKEN_WIDTH
    deltas = np.array([[1000.0, -2000.0, -30.0], [2500.0, -4000.0, -60.0]])
    leads = np.array([30.0, 60.0])
    token = waypoint_token(Waypoints(deltas=deltas, lead_s=leads, valid=np.array([1.0, 0.0])), count)
    assert token[:3] == pytest.approx(deltas[0] / WAYPOINT_DELTA_SCALE_M)
    assert np.all(token[3:6] == 0.0)                       # an invalid waypoint carries nothing
    assert token[6:8] == pytest.approx([leads[0] / WAYPOINT_LEAD_SCALE_S, 0.0])
    assert token[8:10].tolist() == [1.0, 0.0] and token[10] == 1.0
    assert np.all(waypoint_token(None, count) == 0.0)      # absent: the present bit is 0 too
    with pytest.raises(ValueError, match="carries 2 waypoints"):
        waypoint_token(Waypoints(deltas=deltas[:1], lead_s=leads[:1], valid=np.ones(1)), count)


def test_the_training_token_is_the_truths_position_at_each_coarse_segment_after_the_anchor(cohort) -> None:
    _flights, series, config = cohort
    normalizer = Normalizer.fit(series)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    count = plan_waypoint_count(config)
    for i in range(len(series)):
        s_idx, anchor = windows.index[i]
        item = series[s_idx]
        origin = np.asarray(item.values[anchor], dtype=np.float64)[list(POSITION_IDX)]
        times = float(item.times[anchor]) + PLAN_WAYPOINT_SEGMENT_S * np.arange(1, count + 1)
        expected = np.column_stack([
            np.interp(times, item.supervision_times, item.supervision_values[:, c]) for c in POSITION_IDX
        ]) - origin
        token = windows.context.row(i)[PLAN_TOKEN_KEY]
        assert token == pytest.approx(truth_waypoint_token(item, anchor, config))
        assert token[: 3 * count].reshape(count, 3) == pytest.approx(expected / WAYPOINT_DELTA_SCALE_M, abs=1e-5)
        assert token[3 * count : 4 * count] == pytest.approx(PLAN_WAYPOINT_SEGMENT_S * np.arange(1, count + 1) / WAYPOINT_LEAD_SCALE_S)
        assert token[4 * count : 5 * count].tolist() == [1.0] * count and token[5 * count] == 1.0


def test_a_waypoint_the_truth_never_reaches_is_invalid_and_carries_nothing(cohort) -> None:
    _flights, series, _config = cohort
    item = series[0]
    # an origin with between one and two coarse segments of truth left
    end = float(item.supervision_times[-1])
    origin_time = end - 1.5 * PLAN_WAYPOINT_SEGMENT_S
    origin = np.array([100.0, 200.0, 300.0])
    waypoints = truth_waypoints(item, origin_time, origin, 2)
    assert waypoints.valid.tolist() == [1.0, 0.0]
    assert np.all(waypoints.deltas[1] == 0.0)
    first = np.array([np.interp(origin_time + PLAN_WAYPOINT_SEGMENT_S, item.supervision_times, item.supervision_values[:, c]) for c in POSITION_IDX])
    assert waypoints.deltas[0] == pytest.approx(first - origin)
    # the fixed horizon keeps every TRAINING anchor at least two segments from the end, so a
    # training row never carries an invalid waypoint
    assert truth_duration_s(item, default_anchor(_config)) >= HORIZON_S


# ── every batch surface carries it ────────────────────────────────────────────

def test_the_probe_and_the_real_batch_carry_the_same_token_width(cohort) -> None:
    _flights, series, config = cohort
    real = unpack_batch(FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series)).batch([0]))[5]
    probe = probe_dynamics(1, torch.device("cpu"), config)
    assert set(probe) == set(real)
    assert tuple(probe[PLAN_TOKEN_KEY].shape) == tuple(real[PLAN_TOKEN_KEY].shape) == (1, plan_token_width(config))
    assert probe_plan_token(config)[-1] == 1.0 and np.all(probe_plan_token(config)[:-1] == 0.0)
    model = build_model(config, Normalizer.fit(series))
    assert model.plan_encoder[0].in_features == plan_token_width(config)


def test_a_waypoint_checkpoint_trains_predicts_and_names_itself(tmp_path, cohort) -> None:
    _flights, series, config = cohort
    torch.manual_seed(0)
    train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.plan_conditioning == PLAN_CONDITIONING_WAYPOINTS
    name = run_display_name(loaded.to_dict())
    assert "plan=waypoints" in name and "ctrl-horizon=60" in name
    forecasts = forecast_approaches(model, series[:2], loaded, normalizer, device=torch.device("cpu"))
    assert all(f.final_time_s == pytest.approx(HORIZON_S) and np.isfinite(f.values).all() for f in forecasts)


# ── refusals ──────────────────────────────────────────────────────────────────

def test_the_waypoint_axis_is_refused_where_its_token_is_undefined() -> None:
    with pytest.raises(ValueError, match="whole number"):
        _config(control_horizon_s=0.0, final_time_loss_weight=1.0)
    with pytest.raises(ValueError, match="whole number"):
        _config(control_horizon_s=1.5 * PLAN_WAYPOINT_SEGMENT_S)
    with pytest.raises(ValueError, match="threshold-anchored ENU"):
        _config(coordinate_frame="airport-enu")
    with pytest.raises(ValueError, match="reads the truth's plan"):
        _config(checkpoint_selection_metric=CHECKPOINT_SELECTION_ANCHOR_GRID_ADE)
