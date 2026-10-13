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

from ts_transformer.tests.support_prediction import latent_config as _config

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
