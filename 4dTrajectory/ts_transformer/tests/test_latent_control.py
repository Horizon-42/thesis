"""The latent intent on the control output: contracts that would fail silently otherwise.

The posterior must be reachable only through the training loop's ``future``; the top-1
inference path must be deterministic and prior-only; the KL must be the closed form when
the prior is a single Gaussian; the latent must reach the DURATION as well as the controls;
and a run with a latent must be named as a different model, not as a loss edit.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ts_transformer.data.batch_contract import model_forward
from ts_transformer.config import (
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from ts_transformer.outputs.control.latent import (
    ACTIVE_UNIT_KL_NATS,
    LATENT_AUX_COMPONENT,
    LATENT_COMPONENT_KL_PER_DIM_PREFIX,
    PRIOR_MEAN_INIT_STD,
    PriorNetwork,
    LATENT_KL_COMPONENT,
    LatentControlModel,
    LatentControlPrediction,
    effective_latent_beta,
    latent_epoch_record,
    latent_kl,
    per_dimension_kl,
    per_dimension_kl_mean_term,
    sigma_from_logvar,
    with_latent_aux_duration,
    with_latent_kl,
)
from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.backbone.adapters import build_model
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.run_naming import output_name, run_display_name
from ts_transformer.training.objective import loss_component_names
from ts_transformer.training.train import load_checkpoint, train

_CLI_SPEC = importlib.util.spec_from_file_location("ts_transformer_cli_latent_test", Path(__file__).resolve().parents[1] / "__main__.py")
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

import ts_transformer.cli.predict as cli_predict
from ts_transformer.config import CONTROL_DURATION_UNIFORM
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import forecast_approach
from ts_transformer.outputs.control.forecast import (
    latent_derangement,
    latent_mode_forecasts,
    posterior_latent_forecasts,
    random_latent_forecasts,
    shuffled_latent_forecasts,
)
from ts_transformer.outputs.control.latent import displacement_verdict
from ts_transformer.experiments.latent_readout import kept_epoch_latent, readout, render_latent
from ts_transformer.experiments.latent_readout import main as readout_main
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.tests.support import dynamics_context


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        n_segments=8,
        d_model=16,
        d_ff=32,
        e_layers=1,
        n_heads=2,
        dropout=0.0,
        latent_dim=4,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _history(config: TSConfig, batch: int) -> torch.Tensor:
    return torch.randn(batch, config.seq_len, config.enc_in)


def _future(config: TSConfig, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch, config.pred_len, len(config.channels)), torch.full((batch,), 300.0)


def test_build_model_picks_the_latent_head_only_when_latent_dim_is_positive():
    assert isinstance(build_model(_config()), LatentControlModel)
    assert not isinstance(build_model(_config(latent_dim=0)), LatentControlModel)


def test_config_refuses_a_latent_on_a_non_control_output_and_bad_values():
    with pytest.raises(ValueError, match="control output"):
        TSConfig(prediction_output=PREDICTION_STATE, latent_dim=4)
    with pytest.raises(ValueError, match="latent_dim"):
        _config(latent_dim=-1)
    with pytest.raises(ValueError, match="non-negative"):
        _config(latent_beta=-0.1)


def test_inference_is_prior_only_and_deterministic():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).eval()
    history, dynamics = _history(config, 3), dynamics_context(3)
    first = model_forward(model, history, dynamics)
    second = model_forward(model, history, dynamics)
    assert isinstance(first, LatentControlPrediction)
    assert first.posterior_mean is None and first.posterior_logvar is None
    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.controls, second.controls)
    assert torch.equal(first.final_time_s, second.final_time_s)
    # The top-1 latent is the most likely component's mean.
    top = first.prior_logits.argmax(dim=-1)
    assert torch.allclose(first.latent, first.prior_mean[torch.arange(3), top])
    assert first.controls.shape == (3, config.n_segments, 3)
    assert first.segment_durations.shape == (3, config.n_segments)


def test_training_decodes_a_posterior_sample_from_the_future():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).train()
    history, dynamics, future = _history(config, 3), dynamics_context(3), _future(config, 3)
    prediction = model_forward(model, history, dynamics, future=future)
    assert prediction.posterior_mean is not None and prediction.posterior_mean.shape == (3, 4)
    # A posterior sample, not the prior's top-1.
    top = prediction.prior_logits.argmax(dim=-1)
    assert not torch.allclose(prediction.latent, prediction.prior_mean[torch.arange(3), top])


def test_model_forward_never_hands_the_future_to_a_plain_control_model():
    config = _config(latent_dim=0)
    model = build_model(config).eval()
    history, dynamics, future = _history(config, 2), dynamics_context(2), _future(config, 2)
    prediction = model_forward(model, history, dynamics, future=future)
    assert type(prediction) is ControlPrediction


def test_the_latent_reaches_the_duration():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).eval()
    with torch.no_grad():
        model.latent_duration.weight.fill_(1.0)
        # The control head starts at the neutral schedule with ZERO projection weights (its
        # deliberate initialization), so give the projection something to carry.
        model.control_head.control_projection.weight.normal_(std=0.1)
    history, dynamics = _history(config, 2), dynamics_context(2)
    base = model(history, dynamics, latent=torch.zeros(2, 4))
    moved = model(history, dynamics, latent=torch.full((2, 4), 0.5))
    assert torch.all(moved.final_time_s > base.final_time_s)
    assert not torch.allclose(moved.controls, base.controls)


def test_sigma_from_logvar_is_the_one_spelling():
    logvar = torch.tensor([[-2.0, 0.0, 2.0]])
    assert torch.equal(sigma_from_logvar(logvar), (0.5 * logvar).exp())


def test_per_dimension_kl_matches_the_closed_form():
    zeros = torch.zeros(1, 3)
    assert torch.allclose(per_dimension_kl(zeros, zeros, zeros, zeros), zeros)
    # q = N(1, 1) against p = N(0, 1): KL = 0.5 per dimension.
    assert torch.allclose(per_dimension_kl(torch.ones(1, 3), zeros, zeros, zeros), torch.full((1, 3), 0.5))
    # q = N(0, e) against p = N(0, 1): KL = 0.5 (e − 1 − 1) per dimension.
    expected = 0.5 * (math.e - 2.0)
    assert torch.allclose(
        per_dimension_kl(zeros, torch.ones(1, 3), zeros, zeros), torch.full((1, 3), expected)
    )


def _prediction(*, components: int, latent_dim: int = 3, batch: int = 2) -> LatentControlPrediction:
    return LatentControlPrediction(
        controls=torch.zeros(batch, 2, 3),
        segment_durations=torch.ones(batch, 2),
        final_time_s=torch.full((batch,), 2.0),
        latent=torch.ones(batch, latent_dim),
        prior_logits=torch.zeros(batch, components),
        prior_mean=torch.zeros(batch, components, latent_dim),
        prior_logvar=torch.zeros(batch, components, latent_dim),
        posterior_mean=torch.ones(batch, latent_dim),
        posterior_logvar=torch.zeros(batch, latent_dim),
    )


def test_single_gaussian_kl_is_analytic_and_free_bits_are_per_dimension():
    kl = latent_kl(_prediction(components=1), free_bits_nats=0.0)
    assert torch.allclose(kl.charged, torch.full((2,), 1.5))
    assert torch.allclose(kl.per_dimension, torch.full((2, 3), 0.5))
    charged = latent_kl(_prediction(components=1), free_bits_nats=0.4).charged
    assert torch.allclose(charged, torch.full((2,), 0.3))         # (0.5 − 0.4) × 3
    uncharged = latent_kl(_prediction(components=1), free_bits_nats=0.5).charged
    assert torch.allclose(uncharged, torch.zeros(2))


def test_the_kl_splits_into_a_mean_term_and_a_variance_term_that_sum_back():
    """q = N(1, 1) against p = N(0, 1): all 0.5 nats/dim are the MEAN term. The variance
    term is the remainder of the same expression, never a second closed form."""
    kl = latent_kl(_prediction(components=1), free_bits_nats=0.0)
    assert torch.allclose(kl.mean_term_per_dimension, torch.full((2, 3), 0.5))
    assert torch.allclose(kl.variance_term_per_dimension, torch.zeros(2, 3), atol=1e-7)
    assert torch.allclose(kl.displacement_sigma, torch.ones(2, 3))   # |1 − 0| / 1
    # a wider posterior on the prior's mean: nothing in the mean term, all in the width
    q_mean, p = torch.zeros(2, 3), torch.zeros(2, 3)
    wide = per_dimension_kl(q_mean, torch.ones(2, 3), p, p)
    assert torch.allclose(
        per_dimension_kl_mean_term(q_mean, p, p), torch.zeros(2, 3)
    ) and torch.all(wide > 0.0)
    # and the two halves add back up to the number the objective charges
    mixed = latent_kl(_prediction(components=1, latent_dim=3), free_bits_nats=0.0)
    assert torch.allclose(
        mixed.mean_term_per_dimension + mixed.variance_term_per_dimension, mixed.per_dimension
    )


def test_mixture_kl_is_finite_and_keeps_a_per_dimension_diagnostic():
    torch.manual_seed(0)
    kl = latent_kl(_prediction(components=4), free_bits_nats=0.0)
    assert kl.charged.shape == (2,) and torch.all(torch.isfinite(kl.charged))
    assert torch.all(kl.charged >= 0.0)
    assert kl.per_dimension.shape == (2, 3)
    # the diagnostics are taken against ONE component, and the record says which one's
    assert kl.prior_mean.shape == (2, 3) and kl.prior_logvar.shape == (2, 3)


def test_a_prediction_from_the_prior_has_no_kl():
    prediction = LatentControlPrediction(
        controls=torch.zeros(1, 2, 3), segment_durations=torch.ones(1, 2), final_time_s=torch.ones(1),
        latent=torch.zeros(1, 3), prior_logits=torch.zeros(1, 1),
        prior_mean=torch.zeros(1, 1, 3), prior_logvar=torch.zeros(1, 1, 3),
    )
    with pytest.raises(ValueError, match="decoded from the prior"):
        latent_kl(prediction, free_bits_nats=0.0)


def test_with_latent_kl_adds_the_weighted_term_and_the_collapse_diagnostics():
    config = _config(latent_dim=3, latent_beta=2.0)
    zero = torch.zeros(())
    components = LossComponents(state=zero, final_time=zero, kinematic=zero, terminal=zero)
    out = with_latent_kl(components, _prediction(components=1, latent_dim=3), config, torch.ones(2))
    assert out.extras[LATENT_KL_COMPONENT] == pytest.approx(2.0 * 1.5)
    assert out.diagnostics["latent_flights"] == pytest.approx(2)
    # Flight weights scale the TERM, never the diagnostics.
    weighted = with_latent_kl(components, _prediction(components=1, latent_dim=3), config, torch.tensor([2.0, 0.0]))
    assert weighted.extras[LATENT_KL_COMPONENT] == pytest.approx(2.0 * 1.5)
    assert weighted.diagnostics["latent_kl_nats"] == pytest.approx(out.diagnostics["latent_kl_nats"])
    assert out.total == pytest.approx(3.0)
    assert out.diagnostics["latent_kl_nats"] == pytest.approx(0.5 * 3 * 2)
    # every dimension carries 0.5 nats > ACTIVE_UNIT_KL_NATS -> 3 active units, batch-summed
    assert ACTIVE_UNIT_KL_NATS < 0.5
    assert out.diagnostics["latent_active_units"] == pytest.approx(3 * 2)
    assert out.diagnostics["latent_active_units_0p05"] == pytest.approx(3 * 2)
    # the split, the displacement and the per-dimension vector, all batch-summed alike
    assert out.diagnostics["latent_component_kl_mean_term_nats"] == pytest.approx(0.5 * 3 * 2)
    assert out.diagnostics["latent_component_kl_variance_term_nats"] == pytest.approx(0.0, abs=1e-6)
    assert out.diagnostics["latent_mean_displacement_sigma"] == pytest.approx(1.0 * 2)
    for index in range(3):
        key = f"{LATENT_COMPONENT_KL_PER_DIM_PREFIX}{index:02d}"
        assert out.diagnostics[key] == pytest.approx(0.5 * 2)


def test_the_active_unit_count_on_the_fixed_ruler_ignores_the_free_bits_budget():
    """`active_units` moves with the budget (that is its definition); the 0.05 count is the
    ruler that stays comparable between two runs on different budgets."""
    zero = torch.zeros(())
    components = LossComponents(state=zero, final_time=zero, kinematic=zero, terminal=zero)
    prediction = _prediction(components=1, latent_dim=3)     # 0.5 nats in every dimension
    budgeted = with_latent_kl(components, prediction, _config(latent_dim=3, latent_free_bits_nats=1.0),
                              torch.ones(2))
    assert budgeted.diagnostics["latent_active_units"] == pytest.approx(0.0)
    assert budgeted.diagnostics["latent_active_units_0p05"] == pytest.approx(3 * 2)


def test_the_epoch_record_divides_the_summed_diagnostics_by_the_flight_count():
    zero = torch.zeros(())
    components = LossComponents(state=zero, final_time=zero, kinematic=zero, terminal=zero)
    config = _config(latent_dim=3)
    out = with_latent_kl(components, _prediction(components=1, latent_dim=3), config, torch.ones(2))
    totals = {name: float(value) for name, value in out.diagnostics.items()}
    record = latent_epoch_record(totals, config, beta_effective=config.latent_beta)
    assert record["component_kl_nats_per_flight"] == pytest.approx(1.5)
    assert record["component_kl_mean_term_nats"] == pytest.approx(1.5)
    assert record["component_kl_variance_term_nats"] == pytest.approx(0.0, abs=1e-6)
    assert record["component_kl_per_dim"] == pytest.approx([0.5, 0.5, 0.5])
    # the two halves sum to the COMPONENT KL, not to the charged one
    assert record["component_kl_mean_term_nats"] + record["component_kl_variance_term_nats"] == (
        pytest.approx(record["component_kl_nats_per_flight"])
    )
    assert record["mean_displacement_sigma"] == pytest.approx(1.0)
    assert record["active_units"] == pytest.approx(3.0)
    assert record["active_units_0p05"] == pytest.approx(3.0)


def test_the_kl_component_is_registered_only_with_a_latent():
    assert LATENT_KL_COMPONENT in loss_component_names(_config())
    assert LATENT_KL_COMPONENT not in loss_component_names(_config(latent_dim=0))


def test_sample_latents_draws_from_the_chosen_component():
    torch.manual_seed(0)
    config = _config(latent_prior_components=3)
    model = build_model(config)
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    mean = torch.arange(3, dtype=torch.float32).view(1, 3, 1).expand(1, 3, 4).clone()
    logvar = torch.full((1, 3, 4), -20.0)
    latents, components = model.sample_latents(logits, mean, logvar, samples=5)
    assert latents.shape == (5, 1, 4) and components.shape == (5, 1)
    assert torch.all(components == 0)
    assert torch.allclose(latents, torch.zeros(5, 1, 4), atol=1e-3)


def test_a_latent_run_is_named_as_a_different_model():
    plain = _config(latent_dim=0).to_dict()
    latent = _config(latent_dim=8).to_dict()
    mixture = _config(latent_dim=8, latent_prior_components=4).to_dict()
    assert output_name(plain) == "control"
    assert output_name(latent) == "control+z8"
    assert output_name(mixture) == "control+z8k4"
    assert run_display_name(latent).startswith("control+z8 ·")


def test_state_dict_round_trips_into_a_fresh_build():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config)
    history, dynamics = _history(config, 2), dynamics_context(2)
    expected = model.eval()(history, dynamics)
    rebuilt = build_model(config)
    rebuilt.load_state_dict(model.state_dict())
    actual = rebuilt.eval()(history, dynamics)
    assert torch.allclose(expected.controls, actual.controls)
    assert torch.allclose(expected.final_time_s, actual.final_time_s)


AIRPORT, RUNWAY = "KRDU", "05L"


def test_train_checkpoint_forecast_and_export_one_latent_run(tmp_path: Path):
    """The whole chain on synthetic arrivals: the loop hands the future to the posterior,
    the KL term and the collapse record reach history.json, the checkpoint rebuilds the
    latent model, and the forecast is a plain control forecast — no latent in any record."""
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        # the objective metric is refused for a latent run (it reads the posterior)
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        latent_dim=3, latent_beta=0.5, latent_free_bits_nats=0.01,
    )
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    assert report.built == 8, report.format()
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)

    first = json.loads((tmp_path / "run" / "history.json").read_text())["history"][0]
    assert LATENT_KL_COMPONENT in first["train_components"]
    assert set(first["latent"]) == {
        "kl_nats_per_flight", "component_kl_nats_per_flight", "component_kl_mean_term_nats",
        "component_kl_variance_term_nats", "component_kl_per_dim", "mean_displacement_sigma",
        "active_units", "active_units_0p05", "beta_effective",
    }
    assert first["latent"]["beta_effective"] == config.latent_beta   # no warm-up
    assert 0.0 <= first["latent"]["active_units"] <= 3.0
    assert 0.0 <= first["latent"]["active_units_0p05"] <= 3.0
    # the vector is one entry per latent dimension and adds up to the analytic total
    assert len(first["latent"]["component_kl_per_dim"]) == 3
    assert sum(first["latent"]["component_kl_per_dim"]) == pytest.approx(
        first["latent"]["component_kl_nats_per_flight"], rel=1e-6
    )
    assert (
        first["latent"]["component_kl_mean_term_nats"]
        + first["latent"]["component_kl_variance_term_nats"]
    ) == pytest.approx(first["latent"]["component_kl_nats_per_flight"], rel=1e-6)
    assert first["latent"]["mean_displacement_sigma"] > 0.0

    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert isinstance(model, LatentControlModel) and loaded.latent_dim == 3
    assert "control+z3" in run_display_name(loaded.to_dict())

    records, metrics = [], []
    for index, item in enumerate(series[:3]):
        forecast = forecast_approach(model, item, loaded, normalizer, device=torch.device("cpu"))
        assert forecast.prediction_output == PREDICTION_CONTROL and forecast.controls is not None
        records.append(build_prediction_record(
            item, forecast, index=index, model_name=loaded.model, horizon_mode=loaded.horizon_mode
        ))
        metrics.append(observed_series_metrics(item, forecast))
    out = tmp_path / "pred"
    write_batch(records, output_dir=out, config_dict=loaded.to_dict(), flight_metrics=metrics)
    summary = json.loads((out / "summary.json").read_text())
    assert all(row["ade_m"] is not None for row in summary["results"])
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert len(states["control_segments"]) == config.n_segments
    # z is not an output: nothing latent-shaped ANYWHERE in the top-1 record or its row.
    assert "latent" not in json.dumps(states).lower()
    assert "latent" not in json.dumps(summary["results"][0]).lower()

    # K prior samples: [sample][flight], stamped with their index and probability; the
    # single-Gaussian prior has no discrete weight, so every mode reports 1/K.
    modes = latent_mode_forecasts(model, series[:3], loaded, normalizer, samples=2, seed=0,
                                  device=torch.device("cpu"))
    assert len(modes) == 2 and all(len(mode) == 3 for mode in modes)
    assert [f.mode_index for f in modes[1]] == [1, 1, 1]
    assert all(f.mode_probability == pytest.approx(0.5) for mode in modes for f in mode)
    assert not np.allclose(modes[0][0].controls, modes[1][0].controls)
    for index, mode in enumerate(modes):
        mode_records = [build_prediction_record(item, f, index=i, model_name=loaded.model,
                                                horizon_mode=loaded.horizon_mode)
                        for i, (item, f) in enumerate(zip(series[:3], mode))]
        write_batch(mode_records, output_dir=out / "modes" / f"mode{index:02d}",
                    config_dict=loaded.to_dict(),
                    flight_metrics=[observed_series_metrics(item, f) for item, f in zip(series[:3], mode)])
    mode_states = json.loads((out / "modes" / "mode01" / summary["results"][0]["states_file"]).read_text())
    assert mode_states["source"]["modeIndex"] == 1 and mode_states["source"]["modeProbability"] == pytest.approx(0.5)

    # The same-K control: latents from N(0, I), stamped like modes, written under random/.
    randoms = random_latent_forecasts(model, series[:3], loaded, normalizer, samples=2, seed=1,
                                      device=torch.device("cpu"))
    assert len(randoms) == 2 and all(f.mode_probability == pytest.approx(0.5) for r in randoms for f in r)
    for index, mode in enumerate(randoms):
        write_batch([build_prediction_record(item, f, index=i, model_name=loaded.model, horizon_mode=loaded.horizon_mode)
                     for i, (item, f) in enumerate(zip(series[:3], mode))],
                    output_dir=out / "random" / f"mode{index:02d}", config_dict=loaded.to_dict(),
                    flight_metrics=[observed_series_metrics(item, f) for item, f in zip(series[:3], mode)])

    # The collapse diagnostic: every flight decoded from another flight's latent.
    shuffled = shuffled_latent_forecasts(model, series[:3], loaded, normalizer, seed=0,
                                         device=torch.device("cpu"))
    assert all(f.latent_shuffled for f in shuffled) and all(f.mode_index is None for f in shuffled)
    write_batch([build_prediction_record(item, f, index=i, model_name=loaded.model, horizon_mode=loaded.horizon_mode)
                 for i, (item, f) in enumerate(zip(series[:3], shuffled))],
                output_dir=out / "shuffled", config_dict=loaded.to_dict(),
                flight_metrics=[observed_series_metrics(item, f) for item, f in zip(series[:3], shuffled)])
    shuffled_states = json.loads((out / "shuffled" / summary["results"][0]["states_file"]).read_text())
    assert shuffled_states["source"]["latentShuffled"] is True

    # The readout joins the three by flight and takes the per-flight best.
    result = readout(out, None)
    everything = result["strata"]["all"]
    assert result["modes"] == 2 and result["random_modes"] == 2 and result["shuffled"] and everything["n"] == 3
    assert result["control"].endswith("/random")
    assert everything["min_ade_mean_m"] <= everything["top1_ade_mean_m"] + 1e-9
    assert everything["control_min_ade_mean_m"] <= everything["top1_ade_mean_m"] + 1e-9
    assert everything["mode_fde_spread_m"] is not None and everything["shuffled_delta_ade_m"] is not None

    # ...and the kept epoch's own latent diagnostics, read off the artifact's best_epoch.
    diagnostics = kept_epoch_latent(tmp_path / "run" / "history.json")
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert diagnostics["epoch"] == history["fit_diagnostics"]["training_objective"]["best_epoch"]
    assert len(diagnostics["component_kl_per_dim"]) == 3
    text = render_latent(diagnostics)
    assert "per-dimension component KL" in text and "p sigma" in text
    # the verdict is the shared sentence, and it is printed because the key is there
    assert displacement_verdict(diagnostics["mean_displacement_sigma"]) in text
    # ...and a pre-L2.f run prints the keys it has, without a verdict it cannot compute
    older = {key: value for key, value in diagnostics.items()
             if key not in ("mean_displacement_sigma", "component_kl_per_dim")}
    assert "displacement median" not in render_latent(older)

    # The documented epoch-1 command is `--history` ALONE: it must parse and print.
    assert readout_main(["--history", str(tmp_path / "run" / "history.json")]) == 0
    assert readout_main(["--arm", str(out), "--history", str(tmp_path / "run" / "history.json")]) == 0
    with pytest.raises(SystemExit):
        readout_main([])                      # neither an arm nor a history
    with pytest.raises(SystemExit):
        readout_main(["--control", str(out)])  # a control with nothing to compare


def test_every_latent_forecast_entry_refuses_a_non_latent_checkpoint():
    """One refusal, in the shared fan-out, for all four entries.

    Each of the four used to carry its own copy; T3-22 dropped them because `__main__`
    refuses the flags. That left the LIBRARY with none, and its callers are not all the
    CLI — a non-latent model has no `sample_latents`, no `posterior`, and would have died
    on an `AttributeError` naming a method instead of the contract.
    """
    torch.manual_seed(0)
    config = _config(latent_dim=0)
    model = build_model(config).eval()
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT
    )
    normalizer = Normalizer.fit(series)
    device = torch.device("cpu")
    calls = (
        lambda: latent_mode_forecasts(model, series, config, normalizer, samples=2, seed=0, device=device),
        lambda: random_latent_forecasts(model, series, config, normalizer, samples=2, seed=0, device=device),
        lambda: posterior_latent_forecasts(model, series, config, normalizer, device=device),
        lambda: shuffled_latent_forecasts(model, series, config, normalizer, seed=0, device=device),
    )
    for call in calls:
        with pytest.raises(ValueError, match="latent_dim > 0"):
            call()


def test_shuffled_latents_never_hand_a_flight_its_own_and_need_two_flights():
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).eval()
    with pytest.raises(ValueError, match="at least two"):
        shuffled_latent_forecasts(model, [], config, normalizer=None, seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("count", [2, 3, 7, 50])
def test_the_latent_derangement_has_no_fixed_point_and_is_a_permutation(count):
    for seed in range(5):
        source = latent_derangement(count, seed)
        assert sorted(source.tolist()) == list(range(count))
        assert not np.any(source == np.arange(count))
    with pytest.raises(ValueError, match="at least two"):
        latent_derangement(1, 0)


def test_the_deployable_replay_is_decoded_from_the_prior_not_the_posterior(tmp_path: Path, monkeypatch):
    """The objective forward reads the future; the replay that selects the checkpoint must
    not. Intercept every prediction the replay is built from and require it to be a
    prior decode (no posterior) — counting forwards would pass even if the replay still
    reused the posterior object, because the end-of-training cohort evaluation also runs
    prior-only forwards."""
    import ts_transformer.training.validation as validation

    replayed: list[object] = []
    original_replay = validation._prediction_batch_replay

    def intercepting_replay(output, *args, **kwargs):
        replayed.append(output)
        return original_replay(output, *args, **kwargs)

    monkeypatch.setattr(validation, "_prediction_batch_replay", intercepting_replay)
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0, latent_dim=3,
    )
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT)
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    assert replayed, "the validation replay never ran"
    assert all(isinstance(output, LatentControlPrediction) for output in replayed)
    assert all(output.posterior_mean is None for output in replayed)


def test_config_refuses_objective_checkpoint_selection_with_a_latent():
    from ts_transformer.config import CHECKPOINT_SELECTION_OBJECTIVE
    with pytest.raises(ValueError, match="cannot select its checkpoint on the validation objective"):
        _config(checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE)


def test_mode_probability_is_the_sampled_component_s_mixture_weight():
    """K=3 with asymmetric logits: every mode's probability must be softmax(logits) at the
    component it was drawn from (the gather/transpose in latent_mode_forecasts)."""
    torch.manual_seed(0)
    config = _config(latent_prior_components=3)
    model = build_model(config).eval()
    with torch.no_grad():
        model.prior.network.bias[:3] = torch.tensor([2.0, 0.0, -2.0])
        logits, mean, logvar = model.prior(torch.zeros(2, config.d_model))
        weights = torch.softmax(logits, dim=-1)
        latents, components = model.sample_latents(logits, mean, logvar, samples=6,
                                                   generator=torch.Generator().manual_seed(1))
    assert latents.shape == (6, 2, 4) and components.shape == (6, 2)
    gathered = weights.gather(1, components.transpose(0, 1)).transpose(0, 1)
    for s in range(6):
        for b in range(2):
            assert gathered[s, b] == pytest.approx(float(weights[b, components[s, b]]))
    # and the component that dominates the logits is drawn most often
    assert (components == 0).float().mean() > 0.6


def test_the_mixture_kl_estimator_is_unbiased_without_a_budget():
    """Two identical components ARE one Gaussian, so the one-sample estimate must average
    to the analytic KL — a clamp at zero would overcharge it near KL ≈ 0."""
    torch.manual_seed(0)
    q_mean, q_logvar = torch.full((1, 4), 0.3), torch.zeros(1, 4)
    analytic = per_dimension_kl(q_mean, q_logvar, torch.zeros(1, 4), torch.zeros(1, 4)).sum()
    draws = 20_000
    latents = q_mean + torch.randn(draws, 4) * (0.5 * q_logvar).exp()
    prediction = LatentControlPrediction(
        controls=torch.zeros(draws, 2, 3), segment_durations=torch.ones(draws, 2),
        final_time_s=torch.full((draws,), 2.0), latent=latents,
        prior_logits=torch.zeros(draws, 2), prior_mean=torch.zeros(draws, 2, 4),
        prior_logvar=torch.zeros(draws, 2, 4),
        posterior_mean=q_mean.expand(draws, -1), posterior_logvar=q_logvar.expand(draws, -1),
    )
    total = latent_kl(prediction, free_bits_nats=0.0).charged
    assert (total < 0).float().mean() > 0.2            # a signed estimator, as it must be
    assert total.mean() == pytest.approx(float(analytic), abs=0.02)


def test_a_mixture_prior_starts_with_distinguishable_components():
    torch.manual_seed(0)
    prior = PriorNetwork(_config(latent_dim=4, latent_prior_components=4))
    logits, mean, logvar = prior(torch.zeros(1, 16))
    assert torch.allclose(logits, torch.zeros(1, 4)) and torch.allclose(logvar, torch.zeros(1, 4, 4))
    spread = mean[0].std(dim=0).mean()
    assert spread > 0.1 * PRIOR_MEAN_INIT_STD          # not one Gaussian wearing four labels
    single = PriorNetwork(_config(latent_dim=4, latent_prior_components=1))
    assert torch.allclose(single(torch.zeros(1, 16))[1], torch.zeros(1, 1, 4))


def test_config_refuses_latent_knobs_without_a_latent():
    with pytest.raises(ValueError, match="mean nothing without a latent"):
        _config(latent_dim=0, latent_beta=0.5)
    with pytest.raises(ValueError, match="mean nothing without a latent"):
        _config(latent_dim=0, latent_posterior_init_std=0.1)
    with pytest.raises(ValueError, match="mean nothing without a latent"):
        _config(latent_dim=0, latent_beta_warmup_epochs=40)
    with pytest.raises(ValueError, match="latent_posterior_init_std must be positive"):
        _config(latent_posterior_init_std=0.0)
    with pytest.raises(ValueError, match="latent_beta_warmup_epochs must be >= 0"):
        _config(latent_beta_warmup_epochs=-1)


def test_the_beta_warmup_is_linear_from_zero_and_flat_without_one():
    config = _config(latent_beta=0.02, latent_beta_warmup_epochs=4)
    assert [effective_latent_beta(config, epoch) for epoch in (1, 2, 3, 4, 5, 200)] == (
        pytest.approx([0.005, 0.01, 0.015, 0.02, 0.02, 0.02])
    )
    flat = _config(latent_beta=0.02)
    assert {effective_latent_beta(flat, epoch) for epoch in range(1, 200)} == {0.02}
    # ...and the epoch record says which weight the epoch was charged at
    zero = torch.zeros(())
    components = LossComponents(state=zero, final_time=zero, kinematic=zero, terminal=zero)
    totals = {
        name: float(value)
        for name, value in with_latent_kl(
            components, _prediction(components=1, latent_dim=config.latent_dim),
            config, torch.ones(2),
        ).diagnostics.items()
    }
    assert latent_epoch_record(totals, config, beta_effective=0.005)["beta_effective"] == 0.005
    # a renamed diagnostic is a KeyError here, never a silent zero
    with pytest.raises(KeyError):
        latent_epoch_record({}, config, beta_effective=0.005)


def test_the_warmup_names_the_run_only_when_it_is_set():
    assert "beta-warmup" not in run_display_name(_config(latent_dim=8).to_dict())
    assert "beta-warmup=40" in run_display_name(
        _config(latent_dim=8, latent_beta_warmup_epochs=40).to_dict()
    )


def test_the_named_recipes_are_non_latent_by_definition():
    """A named recipe is a published DETERMINISTIC comparison arm: the whole latent axis is
    pinned at its default, so a latent run is `custom` (which every latent arm file says)."""
    from ts_transformer.config import CONTROL_RECIPE_NAMES, CONTROL_RECIPE_CUSTOM, control_recipe_overrides
    defaults = TSConfig().to_dict()
    latent_fields = [name for name in defaults if name.startswith("latent_")]
    assert len(latent_fields) == 7
    for name in CONTROL_RECIPE_NAMES:
        if name == CONTROL_RECIPE_CUSTOM:
            continue
        overrides = control_recipe_overrides(name)
        for field in latent_fields:
            assert overrides[field] == defaults[field], (name, field)
    # ...and a recipe-named run may therefore not carry one
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        _config(control_recipe_name="simple-v3", latent_dim=8)


def test_config_refuses_the_aux_target_without_a_latent_and_under_a_given_cta():
    with pytest.raises(ValueError, match="mean nothing without a latent"):
        _config(latent_dim=0, latent_aux_duration_weight=1.0)
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _config(latent_aux_duration_weight=-1.0)
    with pytest.raises(ValueError, match="already hands that duration to the decoder"):
        _config(cta_conditioning="given", latent_aux_duration_weight=1.0)
    # the CTA alone is fine, and so is the aux target alone
    assert _config(cta_conditioning="given").latent_aux_duration_weight == 0.0
    assert _config(latent_aux_duration_weight=1.0).cta_conditioning == "off"


def test_the_aux_head_exists_only_when_weighted_and_reads_only_a_posterior_sample():
    torch.manual_seed(0)
    plain = build_model(_config())
    assert plain.aux_duration is None
    assert not any("aux_duration" in key for key in plain.state_dict())
    config = _config(latent_aux_duration_weight=1.0)
    model = build_model(config)
    assert isinstance(model.aux_duration, torch.nn.Linear)
    assert model.aux_duration.in_features == config.latent_dim
    history, dynamics, future = _history(config, 3), dynamics_context(3), _future(config, 3)
    trained = model_forward(model, history, dynamics, future=future)
    assert trained.aux_normalized_duration is not None
    assert trained.aux_normalized_duration.shape == (3,)
    assert torch.all(torch.isfinite(trained.aux_normalized_duration))
    # inference: the prior's top-1, and an explicit latent (modes, shuffle, z-oracle)
    assert model_forward(model, history, dynamics).aux_normalized_duration is None
    assert model(history, dynamics, latent=torch.zeros(3, config.latent_dim)).aux_normalized_duration is None
    # and the head is not in the decode path: the same latent decodes the same controls
    # whether the head is there or not
    rebuilt = build_model(_config())
    rebuilt.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "aux_duration" not in k}
    )
    latent = torch.randn(3, config.latent_dim)
    assert torch.allclose(
        model.eval()(history, dynamics, latent=latent).controls,
        rebuilt.eval()(history, dynamics, latent=latent).controls,
    )


def test_the_aux_component_is_registered_only_when_it_is_charged():
    assert LATENT_AUX_COMPONENT not in loss_component_names(_config())
    assert LATENT_AUX_COMPONENT in loss_component_names(_config(latent_aux_duration_weight=1.0))


def test_the_aux_term_is_the_weighted_mse_against_the_normalized_duration():
    config = _config(latent_dim=3, latent_aux_duration_weight=2.0, final_time_scale_s=100.0)
    zero = torch.zeros(())
    components = LossComponents(state=zero, final_time=zero, kinematic=zero, terminal=zero)
    prediction = replace(
        _prediction(components=1, latent_dim=3),
        aux_normalized_duration=torch.tensor([1.0, 2.0]),
    )
    out = with_latent_aux_duration(
        components, prediction, config, torch.tensor([300.0, 100.0]), torch.ones(2)
    )
    # targets 3.0 and 1.0 -> squared errors 4.0 and 1.0 -> weight 2 x mean 2.5
    assert out.extras[LATENT_AUX_COMPONENT] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="decoded from the prior"):
        with_latent_aux_duration(components, _prediction(components=1, latent_dim=3), config,
                                 torch.tensor([300.0, 100.0]), torch.ones(2))


def test_the_aux_target_names_the_run_only_when_it_is_set():
    assert "aux-T" not in run_display_name(_config(latent_dim=8).to_dict())
    assert "aux-T=1" in run_display_name(
        _config(latent_dim=8, latent_aux_duration_weight=1.0).to_dict()
    )


@pytest.mark.parametrize("init_std", [1.0, 0.1])
def test_the_posterior_opens_at_the_configured_std_and_its_mean_reads_the_future(init_std):
    """The 2026-09-07 collapse mechanism: a posterior that opens at unit variance around a
    mean of ~0.2 hands the decoder noise. The log-variance half starts as the configured
    constant for EVERY flight; the mean half is a function of the future from step 0."""
    torch.manual_seed(0)
    config = _config(latent_posterior_init_std=init_std)
    model = build_model(config)
    future = torch.randn(4, config.pred_len, len(config.channels))
    mean, logvar = model.posterior(future, torch.tensor([200.0, 250.0, 300.0, 350.0]))
    assert torch.allclose(logvar, torch.full_like(logvar, 2.0 * math.log(init_std)))
    assert not torch.allclose(mean[0], mean[1])           # not a constant: it reads the future
    assert mean.abs().mean() > 0.01
    # the axis names the run only when it leaves the default
    name = run_display_name(config.to_dict())
    assert ("q-std=0.1" in name) == (init_std != 1.0)


def test_the_latent_model_trains_under_simple_v3_s_own_supervision(tmp_path: Path):
    """The likely L2 base is the native endpoint grid + true-time-position + the imitation
    teacher (simple-v3's objective). The latent's KL must ride beside those terms, and the
    imitation teacher's per-flight inverse must be built for a latent run too."""
    from ts_transformer.config import (
        CONTROL_STATE_LOSS_GRID_NATIVE,
        CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    )
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_imitation_loss_weight=4.0,
        control_velocity_loss_weight=0.003,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
        latent_dim=3, latent_free_bits_nats=0.01,
    )
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    assert report.built == 8, report.format()
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    first = json.loads((tmp_path / "run" / "history.json").read_text())["history"][0]
    for name in ("state", "velocity", "imitation", LATENT_KL_COMPONENT):
        assert name in first["train_components"], name
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert isinstance(model, LatentControlModel)
    forecast = forecast_approach(model, series[0], loaded, normalizer, device=torch.device("cpu"))
    assert forecast.controls is not None and forecast.controls.shape[0] == config.n_segments


def test_the_annealed_beta_is_what_the_epoch_charges_and_the_record_says_so(tmp_path: Path):
    """β warm-up end to end: the epoch record carries the effective β, and the KL loss
    COMPONENT is that β times the unscaled KL the same record reports (one airport, so the
    macro weights are all 1 and the two averages are the same average). This is the
    contract a reader of history.json depends on: `latent.kl_nats_per_flight` is nats,
    `train_components.latent_kl` is what those nats cost."""
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=2, patience=2, batch_size=8, dropout=0.0,
        latent_dim=3, latent_beta=0.5, latent_beta_warmup_epochs=4,
    )
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    history = json.loads((tmp_path / "run" / "history.json").read_text())["history"]
    assert [row["latent"]["beta_effective"] for row in history] == pytest.approx([0.125, 0.25])
    for row in history:
        assert row["train_components"][LATENT_KL_COMPONENT] == pytest.approx(
            row["latent"]["beta_effective"] * row["latent"]["kl_nats_per_flight"], rel=1e-5
        )
    # the checkpoint keeps the run's own beta, not the epoch's
    _model, loaded, _normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.latent_beta == 0.5 and loaded.latent_beta_warmup_epochs == 4


def _aux_arm(tmp_path: Path, weight: float, seed: int, *, epochs: int = 2, flights: int = 12):
    """One tiny latent run at the given auxiliary weight; returns its epoch records.

    The two weights are PAIRED: `aux_duration` is the last module built, so every other
    module draws the same initialization, and the data, split and torch seed are identical.
    The only difference between the arms is the auxiliary gradient.
    """
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=epochs, patience=epochs, batch_size=8, dropout=0.0, seed=seed,
        val_fraction=0.25, test_fraction=0.25,
        latent_dim=3, latent_beta=0.5, latent_free_bits_nats=0.01,
        latent_aux_duration_weight=weight,
    )
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=flights, seed=3), config, airport=AIRPORT
    )
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    torch.manual_seed(0)
    out = tmp_path / f"w{weight:g}_s{seed}"
    train(series, config, output_dir=out, data_provenance=provenance, verbose=False)
    return json.loads((out / "history.json").read_text())["history"]


def test_the_aux_target_keeps_information_in_the_posterior_mean(tmp_path: Path):
    """The designed effect, measured: the KL's MEAN term — the half that carries per-flight
    information — is larger with the auxiliary target than without it, on the same seed.

    ONE seed is asserted (7, the widest margin, ~11 s); the other two were measured the same
    way and are recorded here rather than run on every suite. Synthetic arrivals, 12 flights,
    2 epochs, paired init, epoch 2, component KL mean term without -> with:
    seed 7 105.46 -> 107.92, seed 11 26.146 -> 26.207, seed 13 15.96 -> 16.17 — three of
    three; at 16 flights the same three are 102.60 -> 108.06, 26.854 -> 26.917,
    15.48 -> 15.69, and the gap widens with epochs (seed 7, 16 flights, 6 epochs: 58.8 -> 94.0).

    The `mean_displacement_sigma` MEDIAN is NOT monotone here and this test deliberately
    does not assert it: 9.45 -> 9.58 (seed 7) and 2.03 -> 2.14 (seed 13) rise, 2.455 -> 2.441
    (seed 11) falls. The explanation is the shape of the target — a scalar read-out needs
    ONE latent direction, so it can raise the summed mean term while leaving a median over
    three dimensions flat or lower. Read `component_kl_per_dim` beside the median on this arm.

    And the target is literally an INPUT of the posterior encoder (the true duration), so a
    small `latent_aux` on its own proves only that one latent coordinate can copy one input.
    What is asserted here is the KL's mean term MOVING; the campaign gate needs a downstream
    metric to move as well.
    """
    seed = 7
    without = _aux_arm(tmp_path, 0.0, seed)
    with_target = _aux_arm(tmp_path, 1.0, seed)
    assert all(row["train_components"].get(LATENT_AUX_COMPONENT) is None for row in without)
    assert all(
        row["train_components"][LATENT_AUX_COMPONENT] > 0.0 and
        math.isfinite(row["train_components"][LATENT_AUX_COMPONENT])
        for row in with_target
    )
    assert (
        with_target[-1]["latent"]["component_kl_mean_term_nats"]
        > without[-1]["latent"]["component_kl_mean_term_nats"]
    )


def test_an_aux_trained_checkpoint_forecasts_and_writes_no_aux_output(tmp_path: Path):
    """The head is training-only: the checkpoint rebuilds with it, the forecast never calls
    it, and nothing about it (or about z) reaches a record."""
    _aux_arm(tmp_path, 1.0, seed=7, epochs=1)
    run = tmp_path / "w1_s7"
    model, loaded, normalizer, _payload = load_checkpoint(run / "checkpoint.pt")
    assert isinstance(model, LatentControlModel) and model.aux_duration is not None
    assert loaded.latent_aux_duration_weight == 1.0
    series, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), loaded, airport=AIRPORT
    )
    forecast = forecast_approach(model, series[0], loaded, normalizer, device=torch.device("cpu"))
    assert forecast.controls is not None
    out = tmp_path / "pred"
    write_batch(
        [build_prediction_record(series[0], forecast, index=0, model_name=loaded.model,
                                 horizon_mode=loaded.horizon_mode)],
        output_dir=out, config_dict=loaded.to_dict(),
        flight_metrics=[observed_series_metrics(series[0], forecast)],
    )
    summary = json.loads((out / "summary.json").read_text())
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert "aux" not in json.dumps(states).lower()
    assert "aux" not in json.dumps(summary["results"][0]).lower()
    # state dict round trip into a fresh build of the same config
    rebuilt = build_model(loaded)
    rebuilt.load_state_dict(model.state_dict())
    history, dynamics = _history(loaded, 2), dynamics_context(2)
    assert torch.allclose(
        model.eval()(history, dynamics).controls, rebuilt.eval()(history, dynamics).controls
    )


def test_the_z_oracle_decodes_the_posterior_mean_of_each_flight_s_own_future(tmp_path: Path):
    """Gate 3's instrument: the decode must equal a forward from the training-side posterior
    mean (the dataset's own target rows and duration), and the record must say it read
    the future."""
    torch.manual_seed(0)
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0, latent_dim=3,
    )
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3), config, airport=AIRPORT)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.1)
    oracle = posterior_latent_forecasts(model, series, config, normalizer, device=torch.device("cpu"))
    top1 = forecast_approach(model, series[0], config, normalizer, device=torch.device("cpu"))
    assert all(f.z_from_posterior for f in oracle) and not top1.z_from_posterior
    assert not np.allclose(oracle[0].controls, top1.controls)     # a different latent decoded
    # the same decode as the training-side posterior mean
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, y, _w, final_time, _fw, dynamics, _s = windows.batch(np.arange(4))
    with torch.no_grad():
        mean, _ = model.posterior(y, final_time)
        direct = model(x, dynamics, latent=mean)
    assert np.allclose(oracle[1].segment_durations_s.sum(), float(direct.final_time_s[1]), atol=1e-4)
    out = tmp_path / "oracle"
    write_batch([build_prediction_record(item, f, index=i, model_name=config.model, horizon_mode=config.horizon_mode)
                 for i, (item, f) in enumerate(zip(series, oracle))],
                output_dir=out, config_dict=config.to_dict(),
                flight_metrics=[observed_series_metrics(item, f) for item, f in zip(series, oracle)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mode"].endswith(":z-posterior") and summary["results"][0]["z_from_posterior"] is True
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert states["source"]["zFromPosterior"] is True


def test_the_predict_cli_writes_the_latent_decodes_beside_the_top1(tmp_path: Path, monkeypatch):
    """Through the real predict command, the way the campaign runner calls it (--output-dir
    as a string): the modes/, random/ and shuffled/ record sets must cross the same path
    boundary as the top-1 records. The L2 campaign died on `str / "modes"` after a full
    54-minute train because the chain test above mirrored the CLI instead of walking it."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    config = _config(
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5, seq_len=8, n_segments=4, n_heads=4,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, latent_dim=3,
    )
    series, _report = build_series(flights, config, airport=AIRPORT)
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    _model, _loaded, _normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert len(payload["split"]["val"]) >= 2   # the shuffle needs another flight in the batch
    monkeypatch.setattr(cli_predict, "provenance_from_args", lambda _args: provenance)
    monkeypatch.setattr(cli_predict, "load_flight_dicts", lambda _path, include_flight_keys=None: flights)

    out = tmp_path / "pred"
    assert ts_cli.main([
        "predict", "--checkpoint", str(tmp_path / "run" / "checkpoint.pt"),
        "--data", str(tmp_path / "manifest.json"), "--airport", AIRPORT,
        "--output-dir", str(out), "--split", "val", "--device", "cpu",
        "--latent-samples", "2", "--latent-random", "2", "--latent-shuffle",
    ]) == 0
    top1 = json.loads((out / "summary.json").read_text())["results"]
    assert len(top1) == len(payload["split"]["val"])
    for sub in ("modes/mode00", "modes/mode01", "random/mode00", "random/mode01", "shuffled"):
        rows = json.loads((out / sub / "summary.json").read_text())["results"]
        assert len(rows) == len(top1), sub
    shuffled = json.loads((out / "shuffled" / top1[0]["states_file"]).read_text())
    assert shuffled["source"]["latentShuffled"] is True
