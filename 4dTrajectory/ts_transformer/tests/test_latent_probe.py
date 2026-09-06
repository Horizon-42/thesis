"""`run_ts_latent_probe.py`: the direct measurement of a checkpoint's two densities.

What must not drift, because each would make the table read like a working latent:

* the posterior comes from the SAME entry the training loop uses (`model_forward` with the
  truth's future), not a second call into the encoder;
* the KL's split is the one `control/latent.py` charges — mean term plus variance term add
  back up to the per-dimension KL, per dimension;
* the cohort is the checkpoint's own split, rebuilt through the shared replay loader, so
  the roster rule has one owner;
* a checkpoint with no posterior (no latent) or one that reads the future a second way
  (`cta_conditioning=given`) is refused before any track is read;
* the artifact is immutable and appears whole.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import run_ts_anytime_curve as replay
import run_ts_latent_probe as probe
from config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CTA_CONDITIONING_GIVEN,
    PREDICTION_CONTROL,
    TSConfig,
)
from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from dataset import build_series, dataset_flight_key
from synthetic import synthetic_arrivals
from train import train

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
                "displacement_sigma_median_per_dim", "kl_per_dim_nats",
                "kl_mean_term_per_dim_nats", "kl_variance_term_per_dim_nats"):
        assert len(block[key]) == LATENT_DIM, key
        assert all(value == pytest.approx(value) for value in block[key]), key   # finite
    # The split is the one the objective charges: the variance term is the REMAINDER of
    # the per-dimension KL, so the two add back up to within the model dtype's rounding
    # (float32; the probe's own reductions are float64).
    for total, mean_term, variance_term in zip(
        block["kl_per_dim_nats"], block["kl_mean_term_per_dim_nats"],
        block["kl_variance_term_per_dim_nats"],
    ):
        assert mean_term + variance_term == pytest.approx(total, rel=1e-6)
    assert (
        block["kl_mean_term_nats_per_flight"] + block["kl_variance_term_nats_per_flight"]
        == pytest.approx(block["kl_nats_per_flight"], rel=1e-6)
    )
    assert block["kl_nats_per_flight"] == pytest.approx(sum(block["kl_per_dim_nats"]), rel=1e-9)
    # the warm posterior opened at 0.1 and one epoch cannot widen it far
    assert all(0.0 < value < 1.0 for value in block["posterior_sigma_median"])
    assert block["prior_total_std"] > 0.0
    text = (tmp_path / "probe" / "latent_probe.txt").read_text()
    assert "|q mean − p mean| / p sigma" in text and "MEAN term" in text
    assert "never a prediction result" in text


def test_the_probe_artifact_is_immutable_and_the_limit_narrows_the_cohort(
    monkeypatch, tmp_path, latent_checkpoint
):
    flights, checkpoint = latent_checkpoint
    payload = _run(monkeypatch, flights, tmp_path, checkpoint, tmp_path / "one", "--limit", "1")
    assert payload["limit"] == 1 and payload["checkpoints"]["arm"]["flights"] == 1
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
