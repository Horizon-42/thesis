"""The latent intent on the control output: contracts that would fail silently otherwise.

The posterior must be reachable only through the training loop's ``future``; the top-1
inference path must be deterministic and prior-only; the KL must be the closed form when
the prior is a single Gaussian; the latent must reach the DURATION as well as the controls;
and a run with a latent must be named as a different model, not as a loss edit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from batch_contract import model_forward
from config import (
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from control.conditioning import DYNAMICS_CONDITION_NAMES
from control.envelope import CONTROL_LOWER, CONTROL_UPPER
from control.latent import (
    ACTIVE_UNIT_KL_NATS,
    PRIOR_MEAN_INIT_STD,
    PriorNetwork,
    LATENT_KL_COMPONENT,
    LatentControlModel,
    LatentControlPrediction,
    latent_kl,
    per_dimension_kl,
    with_latent_kl,
)
from batch_contract import LossComponents
from models import build_model
from prediction_outputs import ControlPrediction
from run_naming import output_name, run_display_name
from train import load_checkpoint, loss_component_names, train

from config import CHECKPOINT_SELECTION_OBJECTIVE, CONTROL_DURATION_UNIFORM
from dataset import ARRIVAL_DATA_PROVENANCE_SCHEMA, build_series
from export import build_prediction_record, observed_series_metrics, write_batch
from forecast import (
    forecast_approach,
    latent_derangement,
    latent_mode_forecasts,
    random_latent_forecasts,
    shuffled_latent_forecasts,
)
from run_ts_latent_readout import readout
from synthetic import synthetic_arrivals


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


def _dynamics(batch: int) -> dict[str, torch.Tensor]:
    return {
        "condition": torch.randn(batch, len(DYNAMICS_CONDITION_NAMES)),
        "control_lower": torch.tensor(CONTROL_LOWER, dtype=torch.float32).expand(batch, -1).clone(),
        "control_upper": torch.tensor(CONTROL_UPPER, dtype=torch.float32).expand(batch, -1).clone(),
    }


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
    history, dynamics = _history(config, 3), _dynamics(3)
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
    history, dynamics, future = _history(config, 3), _dynamics(3), _future(config, 3)
    prediction = model_forward(model, history, dynamics, future=future)
    assert prediction.posterior_mean is not None and prediction.posterior_mean.shape == (3, 4)
    # A posterior sample, not the prior's top-1.
    top = prediction.prior_logits.argmax(dim=-1)
    assert not torch.allclose(prediction.latent, prediction.prior_mean[torch.arange(3), top])


def test_model_forward_never_hands_the_future_to_a_plain_control_model():
    config = _config(latent_dim=0)
    model = build_model(config).eval()
    history, dynamics, future = _history(config, 2), _dynamics(2), _future(config, 2)
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
    history, dynamics = _history(config, 2), _dynamics(2)
    base = model(history, dynamics, latent=torch.zeros(2, 4))
    moved = model(history, dynamics, latent=torch.full((2, 4), 0.5))
    assert torch.all(moved.final_time_s > base.final_time_s)
    assert not torch.allclose(moved.controls, base.controls)


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
    kl, kl_dim = latent_kl(_prediction(components=1), free_bits_nats=0.0)
    assert torch.allclose(kl, torch.full((2,), 1.5))
    assert torch.allclose(kl_dim, torch.full((2, 3), 0.5))
    charged, _ = latent_kl(_prediction(components=1), free_bits_nats=0.4)
    assert torch.allclose(charged, torch.full((2,), 0.3))         # (0.5 − 0.4) × 3
    uncharged, _ = latent_kl(_prediction(components=1), free_bits_nats=0.5)
    assert torch.allclose(uncharged, torch.zeros(2))


def test_mixture_kl_is_finite_and_keeps_a_per_dimension_diagnostic():
    torch.manual_seed(0)
    kl, kl_dim = latent_kl(_prediction(components=4), free_bits_nats=0.0)
    assert kl.shape == (2,) and torch.all(torch.isfinite(kl)) and torch.all(kl >= 0.0)
    assert kl_dim.shape == (2, 3)


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
    history, dynamics = _history(config, 2), _dynamics(2)
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
        checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE,
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
    assert set(first["latent"]) == {"kl_nats_per_flight", "active_units"}
    assert 0.0 <= first["latent"]["active_units"] <= 3.0

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
    not. Count both kinds of forward during one training epoch."""
    calls = {"posterior": 0, "prior": 0}
    original = LatentControlModel.forward

    def counting_forward(self, history, dynamics, future=None, latent=None):
        calls["posterior" if future is not None else "prior"] += 1
        return original(self, history, dynamics, future=future, latent=latent)

    monkeypatch.setattr(LatentControlModel, "forward", counting_forward)
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
    # Training + validation objective decode the posterior; the deployable replay decodes
    # the prior. Both must have happened.
    assert calls["posterior"] > 0 and calls["prior"] > 0


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
    total, _ = latent_kl(prediction, free_bits_nats=0.0)
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
