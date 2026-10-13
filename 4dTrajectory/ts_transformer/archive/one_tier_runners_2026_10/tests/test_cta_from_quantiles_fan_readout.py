"""`predict --cta-from-quantiles`: the CTA is the model's OWN duration quantile (B3).

Design §三 3.3. The decoder is the one L3 built — ``cta_conditioning=given`` makes the given
arrival time the duration and the network draws the path that arrives then — but the arrival
time no longer comes from the truth. It comes from the flight's own ``q_τ``, so this is the
first CTA arm that reads no future: the run wears ``cta=self-q`` and never ``cta=given``
(design §六 5), and its top-1 (the q50 decode) is quotable as a prediction.
"""

from __future__ import annotations

import importlib.util

import json

from pathlib import Path

import pytest

from ts_transformer.inference.calibration import (
    FAN_INTERVAL_ALPHA,
    QUANTILE_DIR_NAME,
    calibrate,
    interval_directory_name,
    quantile_directory_name,
    write_conformal_table,
)

from ts_transformer.config import (
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_SELF_QUANTILE,
    DURATION_HEAD_POINT,
    DURATION_MEDIAN_INDEX,
    DURATION_QUANTILES,
)

from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA

from flight_scenarios.identity import summary_row_key

from ts_transformer.data.dataset import build_series

from ts_transformer.experiments import quantile_fan_readout

from ts_transformer.io_utils import file_sha256

from ts_transformer.run_naming import run_display_name

from ts_transformer.data.synthetic import synthetic_arrivals

from ts_transformer.training.train import load_checkpoint, train

from ts_transformer.tests.support_prediction import quantile_config as _config

from ts_transformer.tests.support_prediction import eta_cohort as _cohort

AIRPORT, RUNWAY = "KRDU", "05L"

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_fan_test", Path(__file__).resolve().parents[1] / "__main__.py"
)

ts_cli = importlib.util.module_from_spec(_CLI_SPEC)

PROVENANCE = {
    "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
    "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                   "source_records": []}],
}

def _trained(tmp_path: Path, monkeypatch, **overrides) -> tuple[Path, list]:
    """A tiny checkpoint plus the flights `predict` will be handed for it."""
    import ts_transformer.cli.predict as predict_module

    config = _config(**{"cta_conditioning": CTA_CONDITIONING_GIVEN, **overrides})
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=PROVENANCE, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: PROVENANCE)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )
    return run / "checkpoint.pt", series

def _predict(checkpoint: Path, out: Path, tmp_path: Path, *extra: str) -> int:
    return ts_cli.main([
        "predict", "--checkpoint", str(checkpoint),
        "--data", str(tmp_path / "manifest.json"), "--airport", AIRPORT,
        "--output-dir", str(out), "--split", "val", "--device", "cpu", *extra,
    ])

def test_the_fan_readout_runs_on_a_predicted_directory(tmp_path: Path, monkeypatch, capsys):
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    out = tmp_path / "fan"
    assert _predict(checkpoint, out, tmp_path, "--cta-from-quantiles") == 0

    assert quantile_fan_readout.main(["--arm", str(out), "--json", str(tmp_path / "fan.json")]) == 0
    payload = json.loads((tmp_path / "fan.json").read_text())
    assert payload["calibrated"] is False
    assert payload["quantiles"] == list(DURATION_QUANTILES)
    block = payload["strata"]["all"]
    assert 0.0 <= block["truth_in_fan_share"] <= 1.0
    assert block["fan_width_p50_s"] >= 0.0
    geometry = block["geometry"]["all"]
    # The nearest of the five can never be further than q50, which is one of the five.
    assert geometry["chamfer_nearest_p50_m"] <= geometry["chamfer_q50_p50_m"] + 1e-9
    text = capsys.readouterr().out
    assert "readout, not a coverage guarantee" in text

def test_the_readout_refuses_a_directory_without_a_fan(tmp_path: Path, monkeypatch):
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    out = tmp_path / "plain"
    assert _predict(checkpoint, out, tmp_path) == 0          # cta=given, no fan
    with pytest.raises(SystemExit, match="no complete quantile fan"):
        quantile_fan_readout.main(["--arm", str(out)])
