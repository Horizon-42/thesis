"""The quantile duration head (anytime / calibrated-ETA design §三 3.1, B1).

``duration_head='quantile'`` replaces the scalar ``FinalTimeHead`` with five monotone
quantiles of the SAME quantity. Its median walks the existing ``final_time_s`` contract —
the duration the rollout flies — so nothing downstream changes; the other four ride out to
the record as ``source.durationQuantilesS``. The loss keeps the component name
``final_time`` and swaps the squared residual for the sum of the five pinball losses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ts_transformer.batch_contract import model_forward
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CTA_CONDITIONING_GIVEN,
    DURATION_HEAD_POINT,
    DURATION_HEAD_QUANTILE,
    DURATION_MEDIAN_INDEX,
    DURATION_QUANTILES,
    PREDICTION_CLOSURE,
    PREDICTION_STATE,
    TSConfig,
)
from ts_transformer.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.dataset import Normalizer, build_series
from ts_transformer.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.forecast import forecast_approaches
from ts_transformer.models import build_model
from ts_transformer.objective import loss_component_names
from ts_transformer.outputs.duration_heads import QuantileFinalTimeHead, pinball_duration_loss
from ts_transformer.run_naming import run_display_name, run_slug
from ts_transformer.synthetic import synthetic_arrivals
from ts_transformer.train import load_checkpoint, train
from ts_transformer.tests.support import dynamics_context

AIRPORT, RUNWAY = "KRDU", "05L"


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output="control",
        duration_head=DURATION_HEAD_QUANTILE,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
    )
    settings.update(overrides)
    return TSConfig(**settings)


# ── the head ────────────────────────────────────────────────────────────────

def test_the_head_is_monotone_for_any_input():
    """Cumulative softplus: no input can produce a crossing pair, so the loss never has
    to price one and a calibrated interval can never come out inverted."""
    torch.manual_seed(0)
    config = _config()
    head = QuantileFinalTimeHead(config)
    with torch.no_grad():   # a zeroed last layer would make every row identical
        head.network[-1].weight.normal_(std=3.0)
        head.network[-1].bias.normal_(std=3.0)
    for scale in (0.0, 1.0, 50.0):
        quantiles = head(torch.randn(64, config.seq_len, config.enc_in) * scale)
        assert quantiles.shape == (64, len(DURATION_QUANTILES))
        # Never a CROSSING pair, at any input scale: that is the property the calibration
        # depends on. Strict positivity of every increment is float32's to lose — at a very
        # negative logit `softplus` underflows to zero and two levels tie — so the invariant
        # is non-decreasing, and the strict version is asserted on the trained-scale head
        # below.
        assert torch.all(quantiles >= 0.0)
        assert torch.all(torch.diff(quantiles, dim=-1) >= 0.0)
    modest = head(torch.randn(64, config.seq_len, config.enc_in) * 0.1)
    assert torch.all(torch.diff(modest, dim=-1) > 0.0)


def test_the_median_starts_where_the_point_head_starts():
    """Initialization: an arm that differs only in `duration_head` differs only in what it
    LEARNS — on the first batch the two heads predict the same duration."""
    torch.manual_seed(0)
    point = build_model(_config(duration_head=DURATION_HEAD_POINT)).eval()
    torch.manual_seed(0)
    quantile = build_model(_config()).eval()
    history = torch.randn(3, point.final_time_head.network[1].in_features // 6, 6)
    with torch.no_grad():
        assert torch.allclose(
            quantile.final_time_head(history)[:, DURATION_MEDIAN_INDEX],
            point.final_time_head(history),
            atol=1e-6,
        )


def test_the_median_is_the_duration_the_rollout_flies():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).eval()
    with torch.no_grad():
        model.final_time_head.network[-1].weight.normal_(std=0.5)
    history = torch.randn(4, config.seq_len, config.enc_in)
    prediction = model_forward(model, history, dynamics_context(4))
    assert prediction.duration_quantiles_s.shape == (4, len(DURATION_QUANTILES))
    assert torch.allclose(
        prediction.final_time_s,
        prediction.duration_quantiles_s[:, DURATION_MEDIAN_INDEX],
    )
    assert torch.allclose(prediction.segment_durations.sum(dim=1), prediction.final_time_s)


def test_the_point_head_carries_no_quantiles():
    torch.manual_seed(0)
    config = _config(duration_head=DURATION_HEAD_POINT)
    model = build_model(config).eval()
    prediction = model_forward(model, torch.randn(2, config.seq_len, config.enc_in), dynamics_context(2))
    assert prediction.duration_quantiles_s is None


def test_a_given_cta_still_trains_the_quantile_head_but_not_the_point_head():
    """B3 decodes a `given` checkpoint at its OWN quantiles, so the head has to be trained
    there; the point head stays inert under a CTA, exactly as L3 built it."""
    torch.manual_seed(0)
    cta = torch.tensor([200.0, 250.0])
    quantile = build_model(_config(cta_conditioning=CTA_CONDITIONING_GIVEN)).eval()
    prediction = model_forward(
        quantile, torch.randn(2, 8, quantile.final_time_head.network[1].in_features // 8),
        dynamics_context(2, cta),
    )
    assert torch.allclose(prediction.final_time_s, cta)             # the CTA is the duration
    assert prediction.duration_quantiles_s is not None              # ...and the head is read
    point = build_model(
        _config(cta_conditioning=CTA_CONDITIONING_GIVEN, duration_head=DURATION_HEAD_POINT)
    ).eval()
    inert = model_forward(
        point, torch.randn(2, 8, point.final_time_head.network[1].in_features // 8),
        dynamics_context(2, cta),
    )
    assert inert.duration_quantiles_s is None


# ── the loss ────────────────────────────────────────────────────────────────

def test_the_pinball_loss_is_minimized_at_the_empirical_quantiles():
    """The check function's optimum IS the quantile: perturbing any level away from the
    sample quantile of a fixed target population can only raise the loss."""
    rng = np.random.default_rng(0)
    target = torch.tensor(rng.normal(400.0, 90.0, size=4096), dtype=torch.float64)
    optimum = torch.tensor(
        np.quantile(target.numpy(), DURATION_QUANTILES), dtype=torch.float64
    ).expand(len(target), -1).contiguous()
    best = pinball_duration_loss(optimum, target, 600.0).mean()
    for level in range(len(DURATION_QUANTILES)):
        for step in (-25.0, +25.0):
            moved = optimum.clone()
            moved[:, level] += step
            assert pinball_duration_loss(moved, target, 600.0).mean() > best


def test_the_pinball_loss_is_in_the_point_head_s_units():
    """One target, one level, both heads at the same prediction: the residual both price
    is `(T - q) / final_time_scale_s`."""
    quantiles = torch.tensor([[100.0, 200.0, 300.0, 400.0, 500.0]])
    target = torch.tensor([300.0])
    scale = 600.0
    expected = sum(
        max(tau * (300.0 - q) / scale, (tau - 1.0) * (300.0 - q) / scale)
        for tau, q in zip(DURATION_QUANTILES, quantiles[0].tolist())
    )
    assert pinball_duration_loss(quantiles, target, scale).item() == pytest.approx(expected)


def test_the_loss_component_names_do_not_change():
    """§三 3.1: the quantile loss REPLACES the `final_time` component under the same name,
    so no history row, readout or `extras` key moves."""
    assert (
        loss_component_names(_config())
        == loss_component_names(_config(duration_head=DURATION_HEAD_POINT))
    )
    assert "final_time" in loss_component_names(_config())


# ── refusals ────────────────────────────────────────────────────────────────

def test_the_quantile_head_is_refused_off_the_control_output():
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(prediction_output=PREDICTION_STATE, duration_head=DURATION_HEAD_QUANTILE)
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(
            prediction_output=PREDICTION_CLOSURE,
            horizon_mode="normalized",
            checkpoint_selection_metric="fixed-anchor-objective",
            closure_labels_path="labels.json",
            duration_head=DURATION_HEAD_QUANTILE,
        )


def test_the_quantile_head_is_refused_with_a_latent():
    with pytest.raises(ValueError, match="refused together"):
        _config(latent_dim=4)


def test_an_unknown_duration_head_is_refused():
    with pytest.raises(ValueError, match="unknown duration_head"):
        _config(duration_head="deciles")


def test_a_named_recipe_cannot_wear_a_quantile_head():
    """The recipes pin `duration_head='point'`, so a quantile run is `custom` — and its
    name therefore carries `T=q5` instead of hiding behind a recipe name."""
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        TSConfig(**{
            **dict(_config().to_dict()),
            "control_recipe_name": "simple-v3",
        })


# ── the name ────────────────────────────────────────────────────────────────

def test_the_run_name_carries_the_head():
    quantile = _config().to_dict()
    assert f"T=q{len(DURATION_QUANTILES)}" in run_display_name(quantile)
    assert f"t-q{len(DURATION_QUANTILES)}" in run_slug(quantile)
    assert "T=" not in run_display_name(_config(duration_head=DURATION_HEAD_POINT).to_dict())


# ── the record ──────────────────────────────────────────────────────────────

def test_the_record_and_summary_carry_the_five_quantiles(tmp_path: Path):
    config = _config()
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                                 "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.duration_head == DURATION_HEAD_QUANTILE

    forecasts = forecast_approaches(
        model, series[:3], loaded, normalizer, device=torch.device("cpu")
    )
    out = tmp_path / "pred"
    write_batch(
        [build_prediction_record(item, f, index=i, model_name=loaded.model,
                                 horizon_mode=loaded.horizon_mode)
         for i, (item, f) in enumerate(zip(series[:3], forecasts))],
        output_dir=out, config_dict=loaded.to_dict(),
        flight_metrics=[observed_series_metrics(item, f)
                        for item, f in zip(series[:3], forecasts)],
    )
    summary = json.loads((out / "summary.json").read_text())
    row = summary["results"][0]
    states = json.loads((out / row["states_file"]).read_text())
    published = states["source"]["durationQuantilesS"]
    assert len(published) == len(DURATION_QUANTILES)
    assert published == sorted(published)
    assert row["duration_quantiles_s"] == published
    # The median IS the duration the record's states were rolled over.
    assert published[DURATION_MEDIAN_INDEX] == pytest.approx(
        states["source"]["durationHeadFinalTimeS"], rel=1e-9
    )


def test_a_point_head_record_has_no_quantile_fields(tmp_path: Path):
    config = _config(duration_head=DURATION_HEAD_POINT)
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3), config, airport=AIRPORT
    )
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    forecasts = forecast_approaches(
        model, series[:2], config, normalizer, device=torch.device("cpu")
    )
    out = tmp_path / "pred"
    write_batch(
        [build_prediction_record(item, f, index=i, model_name=config.model,
                                 horizon_mode=config.horizon_mode)
         for i, (item, f) in enumerate(zip(series[:2], forecasts))],
        output_dir=out, config_dict=config.to_dict(),
        flight_metrics=[observed_series_metrics(item, f)
                        for item, f in zip(series[:2], forecasts)],
    )
    summary = json.loads((out / "summary.json").read_text())
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert "durationQuantilesS" not in states["source"]
    assert summary["results"][0]["duration_quantiles_s"] is None
