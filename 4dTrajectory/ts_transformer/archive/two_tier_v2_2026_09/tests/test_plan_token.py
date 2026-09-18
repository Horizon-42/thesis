"""The plan token (two-tier T1, `docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §10.3).

The token is the plan head's own label fused into the control decoder, so the load-bearing
checks are: it IS the label the plan path trains on at that anchor, it reaches the decoder,
a dropped token is exactly the absent one, and every surface that builds a control batch
(training rows, the forecast, the batch-size probe) carries it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import ts_transformer.experiments.anytime_curve as anytime
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CHECKPOINT_SELECTION_OBJECTIVE,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CTA_CONDITIONING_GIVEN,
    PLAN_CONDITIONING_TRUTH_NEXT,
    PREDICTION_CONTROL,
    PREDICTION_PLAN,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.forecast import forecast_approaches
from ts_transformer.outputs.control.plan_token import PLAN_TOKEN_KEY, PLAN_TOKEN_WIDTH, plan_token
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    SCALE_VECTOR,
    TARGETS,
    PlanTargets,
)
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance

pytestmark = pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)

TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=2.0,
            device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8, dropout=0.0)


def _control_config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        cta_conditioning=CTA_CONDITIONING_GIVEN,
        plan_conditioning=PLAN_CONDITIONING_TRUTH_NEXT,
        **TINY,
    )
    settings.update(overrides)
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def cohort():
    config = _control_config()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == 6, report.format()
    return flights, series, config


# ── the token ─────────────────────────────────────────────────────────────────

def test_the_token_is_the_scaled_target_vector_its_valid_bits_and_two_flags() -> None:
    width = len(TARGETS)
    values = np.arange(1, width + 1, dtype=np.float32) * 100.0
    valid = np.ones(width, dtype=np.float32)
    valid[-3:] = 0.0
    token = plan_token(PlanTargets(values=values, valid=valid, next_is_join=True))
    assert token.shape == (PLAN_TOKEN_WIDTH,) == (2 * width + 2,)
    assert token[:width] == pytest.approx(values / SCALE_VECTOR * valid)
    assert np.all(token[width - 3 : width] == 0.0)          # an undefined entry carries nothing
    assert token[width : 2 * width] == pytest.approx(valid)
    assert token[2 * width] == 1.0 and token[2 * width + 1] == 1.0
    assert np.all(plan_token(None) == 0.0)                  # absent: the present bit is 0 too


def test_the_training_token_is_the_plan_heads_own_label_at_the_anchor(cohort) -> None:
    """Not re-derived here: the plan path's window set builds its label row at the same anchor,
    and the control row's token must be that row, scaled."""
    _flights, series, config = cohort
    normalizer = Normalizer.fit(series)
    control = FixedAnchorTrajectoryWindows(series, config, normalizer)
    plan_config = TSConfig(prediction_output=PREDICTION_PLAN,
                           **{**TINY, "checkpoint_selection_metric": CHECKPOINT_SELECTION_OBJECTIVE})
    plan = FixedAnchorTrajectoryWindows(series, plan_config, normalizer)
    assert control.anchor == plan.anchor == default_anchor(config)
    for i in range(len(series)):
        label = plan.context.row(i)
        expected = plan_token(PlanTargets(
            values=label[CONTEXT_TARGETS], valid=label[CONTEXT_VALID],
            next_is_join=bool(float(label[CONTEXT_NEXT_IS_JOIN])),
        ))
        assert control.context.row(i)[PLAN_TOKEN_KEY] == pytest.approx(expected)


# ── the decoder ───────────────────────────────────────────────────────────────

def _fused(model, config, token: torch.Tensor) -> torch.Tensor:
    batch = token.shape[0]
    dynamics = probe_dynamics(batch, torch.device("cpu"), config)
    dynamics[PLAN_TOKEN_KEY] = token
    history = torch.zeros((batch, config.seq_len, config.enc_in), dtype=torch.float32)
    return model.fused_features(history, dynamics)


def test_the_token_reaches_the_decoder_and_a_dropped_token_is_the_absent_one(cohort) -> None:
    _flights, series, _config = cohort
    config = _control_config(plan_conditioning_dropout=0.5)
    torch.manual_seed(0)
    model = build_model(config, Normalizer.fit(series))
    present = torch.zeros((1, PLAN_TOKEN_WIDTH))
    present[0, 0], present[0, -1] = 0.7, 1.0
    absent = torch.zeros((1, PLAN_TOKEN_WIDTH))
    model.eval()
    with torch.no_grad():
        with_plan, without = _fused(model, config, present), _fused(model, config, absent)
        assert not torch.allclose(with_plan, without)
        # eval never drops
        assert torch.allclose(_fused(model, config, present.expand(64, -1)), with_plan.expand(64, -1), atol=1e-6)
        model.train()
        torch.manual_seed(1)
        rows = _fused(model, config, present.expand(400, -1))
    dropped = torch.isclose(rows, without.expand(400, -1), atol=1e-6).all(dim=1)
    kept = torch.isclose(rows, with_plan.expand(400, -1), atol=1e-6).all(dim=1)
    assert bool((dropped | kept).all())                     # every row is one or the other
    assert 0.4 < float(dropped.float().mean()) < 0.6


def test_a_plan_free_control_model_builds_no_plan_encoder(cohort) -> None:
    _flights, series, _config = cohort
    model = build_model(_control_config(plan_conditioning="off"), Normalizer.fit(series))
    assert model.plan_encoder is None and not model.plan_given


# ── every batch surface carries it ────────────────────────────────────────────

def test_the_probe_carries_the_token_the_real_batch_carries(cohort) -> None:
    _flights, series, config = cohort
    real = unpack_batch(FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series)).batch([0]))[5]
    probe = probe_dynamics(1, torch.device("cpu"), config)
    assert set(probe) == set(real)
    assert tuple(probe[PLAN_TOKEN_KEY].shape) == tuple(real[PLAN_TOKEN_KEY].shape) == (1, PLAN_TOKEN_WIDTH)


def test_a_plan_conditioned_checkpoint_trains_predicts_and_names_itself(monkeypatch, tmp_path, cohort) -> None:
    flights, series, config = cohort
    torch.manual_seed(0)
    train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.plan_conditioning == PLAN_CONDITIONING_TRUTH_NEXT
    assert "plan=truth-next" in run_display_name(loaded.to_dict())
    forecasts = forecast_approaches(model, series[:2], loaded, normalizer, device=torch.device("cpu"))
    assert all(forecast.anchor == default_anchor(loaded) and np.isfinite(forecast.values).all()
               for forecast in forecasts)
    # the replay runners refuse the oracle
    grid = anytime.Grid(split="val", bins_m=(), min_future_s=0.0, batch_size=None, limit=0)
    monkeypatch.setattr(anytime, "load_checkpoint", lambda _path: (model, replace(loaded, cta_conditioning="off"), normalizer, _payload))
    with pytest.raises(SystemExit, match="plan_conditioning"):
        anytime.load_arm("t1", tmp_path / "run" / "checkpoint.pt", grid, torch.device("cpu"))


def test_the_plan_axis_is_refused_where_its_token_is_undefined() -> None:
    with pytest.raises(ValueError, match="threshold-anchored ENU"):
        _control_config(coordinate_frame="airport-enu")
    with pytest.raises(ValueError, match="drops a plan token"):
        _control_config(plan_conditioning="off", plan_conditioning_dropout=0.5)
    with pytest.raises(ValueError, match="in \\[0, 1\\)"):
        _control_config(plan_conditioning_dropout=1.0)
    with pytest.raises(ValueError, match="latent"):
        _control_config(latent_dim=8)
