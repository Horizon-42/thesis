"""The controlled time of arrival as a decoder input (latent-intent design §六 L3).

Under ``cta_conditioning=given`` the flight's duration IS the CTA — the duration head is
bypassed, not regressed toward it — and the network decides only the path that arrives
then. A given-CTA run reads the future, so its records say what they were given and its
run name wears ``cta=given``.
"""

from __future__ import annotations

import importlib.util
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
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from ts_transformer.control.conditioning import DYNAMICS_CONDITION_NAMES
from ts_transformer.control.envelope import CONTROL_LOWER, CONTROL_UPPER
from ts_transformer.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series, truth_duration_s
from ts_transformer.export import build_prediction_record, observed_series_metrics, write_batch
import ts_transformer.batching as batching
from ts_transformer.forecast import forecast_approaches, latent_mode_forecasts, shuffled_latent_forecasts
from ts_transformer.models import build_model
from ts_transformer.run_naming import run_display_name
from ts_transformer.synthetic import synthetic_arrivals
from ts_transformer.train import load_checkpoint, train

_CLI_SPEC = importlib.util.spec_from_file_location("ts_transformer_cli_cta_test", Path(__file__).resolve().parents[1] / "__main__.py")
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

AIRPORT, RUNWAY = "KRDU", "05L"


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
        cta_conditioning=CTA_CONDITIONING_GIVEN,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _dynamics(batch: int, cta_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    rows = {
        "condition": torch.randn(batch, len(DYNAMICS_CONDITION_NAMES)),
        "control_lower": torch.tensor(CONTROL_LOWER, dtype=torch.float32).expand(batch, -1).clone(),
        "control_upper": torch.tensor(CONTROL_UPPER, dtype=torch.float32).expand(batch, -1).clone(),
    }
    if cta_s is not None:
        rows["cta_s"] = cta_s
    return rows


def test_config_refuses_a_cta_off_the_control_path_and_unknown_values():
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(prediction_output=PREDICTION_STATE, cta_conditioning=CTA_CONDITIONING_GIVEN)
    with pytest.raises(ValueError, match="unknown cta_conditioning"):
        _config(cta_conditioning="truth")


@pytest.mark.parametrize("latent_dim", [0, 3])
def test_the_given_cta_is_the_duration_and_reaches_the_controls(latent_dim):
    torch.manual_seed(0)
    config = _config(latent_dim=latent_dim)
    model = build_model(config).eval()
    with torch.no_grad():   # the control projection starts at zero: give it something to carry
        model.control_head.control_projection.weight.normal_(std=0.1)
    history = torch.randn(2, config.seq_len, config.enc_in)
    early = model_forward(model, history, _dynamics(2, torch.tensor([200.0, 250.0])))
    late = model_forward(model, history, _dynamics(2, torch.tensor([260.0, 310.0])))
    assert torch.allclose(early.final_time_s, torch.tensor([200.0, 250.0]))
    assert torch.allclose(late.final_time_s, torch.tensor([260.0, 310.0]))
    assert torch.allclose(early.segment_durations.sum(dim=1), early.final_time_s)
    assert not torch.allclose(early.controls, late.controls)      # the path knows when to arrive


def test_a_plain_run_has_no_cta_token_and_ignores_a_cta_in_the_context():
    torch.manual_seed(0)
    config = _config(cta_conditioning="off")
    model = build_model(config).eval()
    assert model.cta_encoder is None
    history = torch.randn(2, config.seq_len, config.enc_in)
    without = model_forward(model, history, _dynamics(2))
    with_cta = model_forward(model, history, _dynamics(2, torch.tensor([200.0, 250.0])))
    assert torch.equal(without.final_time_s, with_cta.final_time_s)


def test_the_dataset_feeds_the_truth_duration_as_the_cta():
    config = _config()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT)
    windows = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    _x, _y, _w, final_time, _fw, dynamics, _sup = windows.batch(np.array([0, 1, 2]))
    assert torch.equal(dynamics["cta_s"].to(torch.float32), final_time)   # IS the duration, not near it
    anchor = windows.index[0][1]
    assert dynamics["cta_s"][0] == pytest.approx(truth_duration_s(series[0], anchor))


def test_train_then_forecast_at_the_truth_cta_and_at_a_counterfactual(tmp_path: Path):
    config = _config()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT)
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    history = json.loads((tmp_path / "run" / "history.json").read_text())["history"]
    assert all(row["train_components"]["final_time"] == 0.0 for row in history)   # an identity
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.cta_conditioning == CTA_CONDITIONING_GIVEN
    assert "cta=given" in run_display_name(loaded.to_dict())
    # the duration head was never trained: still at its initialization
    fresh = build_model(loaded)
    assert torch.equal(model.final_time_head.network[-1].weight, fresh.final_time_head.network[-1].weight)

    anchor = loaded.seq_len - 1
    identity = forecast_approaches(model, series[:3], loaded, normalizer, device=torch.device("cpu"))
    shifted = forecast_approaches(model, series[:3], loaded, normalizer, device=torch.device("cpu"), cta_offset_s=60.0)
    for item, same, later in zip(series[:3], identity, shifted):
        truth = truth_duration_s(item, anchor)
        assert same.cta_s == pytest.approx(truth) and same.cta_offset_s == 0.0
        assert same.predicted_final_time_s == pytest.approx(truth)          # the identity check
        assert later.cta_s == pytest.approx(truth + 60.0) and later.cta_offset_s == 60.0
        assert later.predicted_final_time_s == pytest.approx(truth + 60.0)  # follows the CTA exactly

    out = tmp_path / "pred"
    write_batch(
        [build_prediction_record(item, f, index=i, model_name=loaded.model, horizon_mode=loaded.horizon_mode)
         for i, (item, f) in enumerate(zip(series[:3], shifted))],
        output_dir=out, config_dict=loaded.to_dict(),
        flight_metrics=[observed_series_metrics(item, f) for item, f in zip(series[:3], shifted)],
    )
    summary = json.loads((out / "summary.json").read_text())
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert states["source"]["ctaOffsetS"] == 60.0
    assert states["source"]["ctaS"] == pytest.approx(truth_duration_s(series[0], anchor) + 60.0)
    assert summary["results"][0]["cta_offset_s"] == 60.0 and summary["mode"].endswith(":cta+60s")
    assert summary["results"][0]["final_time_error_s"] == pytest.approx(60.0)   # by construction


def test_the_offset_is_refused_off_the_cta_path():
    config = _config(cta_conditioning="off")
    model = build_model(config).eval()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3), config, airport=AIRPORT)
    with pytest.raises(ValueError, match="cta_conditioning=given only"):
        forecast_approaches(model, series, config, Normalizer.fit(series), device=torch.device("cpu"), cta_offset_s=30.0)


def test_the_latent_decodes_carry_the_same_offset_as_the_top1():
    torch.manual_seed(0)
    config = _config(latent_dim=3)
    model = build_model(config).eval()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT)
    normalizer = Normalizer.fit(series)
    anchor = config.seq_len - 1
    modes = latent_mode_forecasts(model, series, config, normalizer, samples=2, seed=0,
                                  device=torch.device("cpu"), cta_offset_s=45.0)
    shuffled = shuffled_latent_forecasts(model, series, config, normalizer, seed=0,
                                         device=torch.device("cpu"), cta_offset_s=45.0)
    for item, mode0, mode1, shuf in zip(series, modes[0], modes[1], shuffled):
        expected = truth_duration_s(item, anchor) + 45.0
        for f in (mode0, mode1, shuf):
            assert f.cta_offset_s == 45.0 and f.cta_s == pytest.approx(expected)
            assert f.predicted_final_time_s == pytest.approx(expected)


def test_the_auto_batch_probe_carries_the_cta(monkeypatch):
    """`--batch-size auto` runs the real training step on a probe batch; under `given` that
    step reads dynamics["cta_s"], so the probe must carry one — a finite CTA per row."""
    seen: list[torch.Tensor] = []
    original = batching.model_forward

    def recording_forward(model, history, dynamics, future=None):
        seen.append(dynamics["cta_s"])
        return original(model, history, dynamics, future=future)

    monkeypatch.setattr(batching, "model_forward", recording_forward)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    batching._probe_training_step(_config(), 2, torch.device("cpu"))
    assert seen and seen[0].shape == (2,) and torch.all(torch.isfinite(seen[0])) and torch.all(seen[0] > 0)


def test_a_counterfactual_cta_skips_the_flights_it_cannot_be_asked_of(tmp_path: Path, monkeypatch):
    """The L3 scan died at −90 s: truth durations start at ~21 s, so the shifted CTA went
    below zero and the rollout refused. Flights whose CTA would leave less than the
    package's minimum remaining future are skipped and COUNTED in summary.json — never
    clamped, which would silently change the offset the scan is read against — and an
    offset that leaves no flight is refused."""
    from ts_transformer.config import DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S
    import ts_transformer.cli.predict as predict_module
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    config = _config()
    series, _report = build_series(flights, config, airport=AIRPORT)
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    _model, loaded, _normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: provenance)
    monkeypatch.setattr(predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights)
    anchor = loaded.seq_len - 1
    val_ids = set(payload["split"]["val"])
    durations = sorted(truth_duration_s(item, anchor) for item in series if item.dataset_id in val_ids)
    assert len(durations) >= 3
    # an offset that leaves the longest flight flyable and the shortest not
    offset = DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S - (durations[0] + durations[-1]) / 2.0
    expected_skipped = sum(1 for d in durations if d + offset < DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S)
    assert 0 < expected_skipped < len(durations)

    def run(out: Path, cta_offset: float) -> int:
        return ts_cli.main([
            "predict", "--checkpoint", str(tmp_path / "run" / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"), "--airport", AIRPORT,
            "--output-dir", str(out), "--split", "val", "--device", "cpu",
            "--cta-offset-s", str(cta_offset),
        ])

    assert run(tmp_path / "scan", offset) == 0
    summary = json.loads((tmp_path / "scan" / "summary.json").read_text())
    assert summary["skipped"] == {"cta_below_min_future": expected_skipped}
    assert len(summary["results"]) == len(durations) - expected_skipped
    assert all(row["final_time_error_s"] == pytest.approx(offset, abs=1e-3) for row in summary["results"])
    with pytest.raises(SystemExit):
        run(tmp_path / "none", -(durations[-1] + 1.0))


# ── review 2026-09-09: predict's contract holes ──────────────────────────────

def test_predict_writes_every_directory_through_one_emitter():
    """Review C-6: the `modes/`, `random/`, `quantiles/` and `shuffled/` write_batch calls
    omitted `skipped`, so under `--cta-offset-s` they published a subset as the split. One
    emitter carries it to every directory — pinned on the source, since exercising every
    arm needs a latent AND a quantile checkpoint."""
    import inspect
    import ts_transformer.cli.predict as predict_module
    source = inspect.getsource(predict_module.run_cli)
    assert source.count("write_batch(") == 1
    assert "skipped=skipped" in source


def test_predict_records_the_aircraft_type_it_built_the_series_under_and_refuses_a_pooled_airport(tmp_path: Path, monkeypatch):
    """Review C-5: `--aircraft-type X` built the series under X but the config written
    beside the records (and the run name) recorded the checkpoint's type. Review C-19:
    repeated `--data` with `--airport` re-homes another airport's flights; train refused
    it, predict did not."""
    import ts_transformer.cli.predict as predict_module
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3)
    config = _config()
    series, _report = build_series(flights, config, airport=AIRPORT)
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}]}
    train(series, config, output_dir=tmp_path / "run", data_provenance=provenance, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: provenance)
    monkeypatch.setattr(predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights)
    assert config.aircraft_type != "B738"
    assert ts_cli.main([
        "predict", "--checkpoint", str(tmp_path / "run" / "checkpoint.pt"),
        "--data", str(tmp_path / "manifest.json"), "--airport", AIRPORT,
        "--output-dir", str(tmp_path / "pred"), "--split", "val", "--device", "cpu",
        "--aircraft-type", "B738",
    ]) == 0
    summary = json.loads((tmp_path / "pred" / "summary.json").read_text())
    assert summary["config"]["aircraft_type"] == "B738"
    with pytest.raises(SystemExit):
        ts_cli.main([
            "predict", "--checkpoint", str(tmp_path / "run" / "checkpoint.pt"),
            "--data", str(tmp_path / "a.json"), "--data", str(tmp_path / "b.json"),
            "--airport", AIRPORT, "--output-dir", str(tmp_path / "pooled"), "--split", "val",
        ])
