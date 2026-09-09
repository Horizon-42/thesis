"""`run_ts_latent_probe.py`: the direct measurement of a checkpoint's two densities.

What must not drift, because each would make the table read like a working latent:

* the posterior comes from the SAME entry the training loop uses (`model_forward` with the
  truth's future), not a second call into the encoder;
* the KL's split is the one `control/latent.py` charges — mean term plus variance term add
  back up to the per-dimension COMPONENT KL, per dimension;
* the prior's total std is the MIXTURE's own moments, so a K>1 prior's range (which lives
  between its components) is not read off one of them;
* the cohort is the checkpoint's own split, rebuilt through the shared replay loader, so
  the roster rule has one owner;
* a checkpoint with no posterior (no latent) or one that reads the future a second way
  (`cta_conditioning=given`) is refused before any track is read;
* the artifact is immutable and appears whole.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import run_ts_anytime_curve as replay
import run_ts_latent_probe as probe
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CTA_CONDITIONING_GIVEN,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.outputs.control.latent import displacement_verdict
from ts_transformer.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.dataset import build_series, dataset_flight_key
from ts_transformer.synthetic import synthetic_arrivals
from ts_transformer.train import train

AIRPORT, RUNWAY = "KRDU", "05L"
LATENT_DIM = 3


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
        val_fraction=0.25, test_fraction=0.25,
        latent_dim=LATENT_DIM, latent_posterior_init_std=0.1, latent_beta=0.01,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _provenance() -> dict:
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                       "source_records": []}],
    }


def _train(config: TSConfig, out: Path, flights) -> Path:
    torch.manual_seed(0)
    series, _report = build_series(flights, config, airport=AIRPORT)
    train(series, config, output_dir=out, data_provenance=_provenance(), verbose=False)
    return out / "checkpoint.pt"


@pytest.fixture(scope="module")
def latent_checkpoint(tmp_path_factory):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=16, seed=3)
    return flights, _train(_config(), tmp_path_factory.mktemp("latent_run"), flights)


def _patch_data_plane(monkeypatch, flights, tmp_path):
    """The shared replay loader's three data-plane seams, on the synthetic flights.

    Patched on `run_ts_anytime_curve` because that is where `load_arm` / `cohort_series`
    resolve them — the probe reuses those rather than owning a second rebuild.
    """
    manifest = tmp_path / "manifest.json"
    indexed = {dataset_flight_key(flight, index): flight
               for index, flight in enumerate(flights)}
    monkeypatch.setattr(replay.pipeline, "arrival_manifest_path", lambda _airport: manifest)
    monkeypatch.setattr(
        replay, "checkpoint_data_provenance", lambda _payload, _manifests: _provenance()
    )
    monkeypatch.setattr(
        replay, "load_flight_dicts",
        lambda _paths, include_flight_keys, verbose=True: [
            flight for key, flight in indexed.items() if key in include_flight_keys
        ],
    )


def _run(monkeypatch, flights, tmp_path, checkpoint: Path, out: Path, *extra: str) -> dict:
    _patch_data_plane(monkeypatch, flights, tmp_path)
    assert probe.main([
        "--checkpoint", f"arm={checkpoint}", "--split", "val", "--device", "cpu",
        "--out", str(out), *extra,
    ]) == 0
    return json.loads((out / "latent_probe.json").read_text())


def test_the_probe_reports_both_densities_and_a_kl_split_that_adds_up(
    monkeypatch, tmp_path, latent_checkpoint
):
    flights, checkpoint = latent_checkpoint
    payload = _run(monkeypatch, flights, tmp_path, checkpoint, tmp_path / "probe")
    assert payload["schema"] == probe.RESULT_SCHEMA and payload["split"] == "val"
    block = payload["checkpoints"]["arm"]
    assert block["latent_dim"] == LATENT_DIM and block["flights"] >= 2
    for key in ("prior_mean_across_flight_std", "prior_sigma_median",
                "posterior_mean_across_flight_std", "posterior_sigma_median",
                "displacement_sigma_median_per_dim", "component_kl_per_dim_nats",
                "component_kl_mean_term_per_dim_nats",
                "component_kl_variance_term_per_dim_nats"):
        assert len(block[key]) == LATENT_DIM, key
        assert all(math.isfinite(value) for value in block[key]), key
    # The split is the one the objective charges: the variance term is the REMAINDER of
    # the per-dimension KL, so the two add back up to within the model dtype's rounding
    # (float32; the probe's own reductions are float64).
    for total, mean_term, variance_term in zip(
        block["component_kl_per_dim_nats"], block["component_kl_mean_term_per_dim_nats"],
        block["component_kl_variance_term_per_dim_nats"],
    ):
        assert mean_term + variance_term == pytest.approx(total, rel=1e-6)
    assert (
        block["component_kl_mean_term_nats_per_flight"]
        + block["component_kl_variance_term_nats_per_flight"]
        == pytest.approx(block["component_kl_nats_per_flight"], rel=1e-6)
    )
    assert block["component_kl_nats_per_flight"] == pytest.approx(
        sum(block["component_kl_per_dim_nats"]), rel=1e-9
    )
    # the warm posterior opened at 0.1 and one epoch cannot widen it far
    assert all(0.0 < value < 1.0 for value in block["posterior_sigma_median"])
    assert block["prior_total_std"] > 0.0
    text = (tmp_path / "probe" / "latent_probe.txt").read_text()
    assert "|q mean − p mean| / p sigma" in text and "MEAN term" in text
    assert "never a prediction result" in text
    # the verdict is the shared sentence, on the shared ruler
    assert displacement_verdict(block["displacement_sigma_median"]) in text


def test_the_probe_artifact_is_immutable_and_the_limit_narrows_the_cohort(
    monkeypatch, tmp_path, latent_checkpoint
):
    flights, checkpoint = latent_checkpoint
    payload = _run(monkeypatch, flights, tmp_path, checkpoint, tmp_path / "one", "--limit", "1")
    assert payload["limit"] == 1 and payload["checkpoints"]["arm"]["flights"] == 1
    # a limit is a PREFIX of the split, and the artifact says so where a reader will see it
    assert "prefix, not a random sample" in payload["limit_meaning"]
    assert payload["limit_meaning"] in (tmp_path / "one" / "latent_probe.txt").read_text()
    with pytest.raises(FileExistsError, match="immutable"):
        _run(monkeypatch, flights, tmp_path, checkpoint, tmp_path / "one", "--limit", "1")


def test_the_probe_refuses_a_checkpoint_with_no_posterior(monkeypatch, tmp_path):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=16, seed=3)
    checkpoint = _train(_config(latent_dim=0, latent_posterior_init_std=1.0, latent_beta=1.0),
                        tmp_path / "plain", flights)
    _patch_data_plane(monkeypatch, flights, tmp_path)
    with pytest.raises(SystemExit, match="no posterior"):
        probe.main(["--checkpoint", f"arm={checkpoint}", "--device", "cpu",
                    "--out", str(tmp_path / "probe")])
    assert not (tmp_path / "probe").exists()


def test_the_probe_refuses_a_given_cta_checkpoint(monkeypatch, tmp_path):
    """Inherited from the shared loader: a given CTA is the truth duration, a second way of
    reading the future."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=16, seed=3)
    checkpoint = _train(_config(cta_conditioning=CTA_CONDITIONING_GIVEN), tmp_path / "cta", flights)
    _patch_data_plane(monkeypatch, flights, tmp_path)
    with pytest.raises(SystemExit, match="reads the future"):
        probe.main(["--checkpoint", f"arm={checkpoint}", "--device", "cpu",
                    "--out", str(tmp_path / "probe")])
    assert not (tmp_path / "probe").exists()


def test_the_probe_refuses_the_sealed_split(monkeypatch, tmp_path, latent_checkpoint):
    _flights, checkpoint = latent_checkpoint
    with pytest.raises(SystemExit):
        probe.main(["--checkpoint", f"arm={checkpoint}", "--split", "test",
                    "--out", str(tmp_path / "probe")])


def test_the_prior_total_std_is_the_mixture_s_own_moments():
    """K>1 keeps most of its range BETWEEN the components, so the total must come from the
    mixture's moments and not from one of them."""
    # two components at ±1 with negligible width and equal weights: E[z] = 0, Var = 1
    logits = torch.zeros(1, 2)
    mean = torch.tensor([[[-1.0], [1.0]]])
    logvar = torch.full((1, 2, 1), -40.0)
    expectation, variance = probe.mixture_moments(logits, mean, logvar)
    assert expectation == pytest.approx(0.0, abs=1e-6)
    assert float(variance) == pytest.approx(1.0, rel=1e-5)
    # K = 1 reduces to that component's own moments
    single_mean, single_logvar = torch.tensor([[[0.3, -0.2]]]), torch.tensor([[[0.0, 1.0]]])
    one_expectation, one_variance = probe.mixture_moments(
        torch.zeros(1, 1), single_mean, single_logvar
    )
    assert torch.allclose(one_expectation, single_mean[:, 0])
    assert torch.allclose(one_variance, single_logvar[:, 0].exp(), atol=1e-6)


def test_the_probe_runs_on_a_mixture_prior(monkeypatch, tmp_path):
    """A K=4 checkpoint end to end: the diagnostics are taken against the most responsible
    component per flight (the artifact says so) and the prior's total std is the mixture's."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=16, seed=3)
    checkpoint = _train(_config(latent_prior_components=4), tmp_path / "mixture", flights)
    payload = _run(monkeypatch, flights, tmp_path, checkpoint, tmp_path / "probe")
    block = payload["checkpoints"]["arm"]
    assert block["prior_components"] == 4
    assert "most responsible" in block["component"]
    assert math.isfinite(block["prior_total_std"]) and block["prior_total_std"] > 0.0
    assert len(block["component_kl_per_dim_nats"]) == LATENT_DIM
