"""The two-head duration (anytime / calibrated-ETA design §三 3.1b, B1.b).

``duration_head='two-head'`` carries BOTH heads at once. The POINT head drives the rollout
duration, exactly as under ``point`` — that is where B1_point_matched's path gain came from
(the duration term's weight, not the quantile head) — and the QUANTILE head emits nothing
but the published ETA distribution B2 calibrates and B3 decodes, which is where B1's
arrival-time gain came from. The two gains do not overlap, and this is the head that takes
both without the quantile head's path cost being forced onto the trajectory.

What is asserted here: the point head is what the rollout flies (never q50); both weights
bind and price their own term; every refusal names its reason; and the record, the
calibration runner, ``predict --cta-from-quantiles`` and the ETA-error readout all work on a
two-head checkpoint. `point` / `quantile` are unchanged — that is a bit-exactness harness
run, reported in the commit, not something a unit test can see.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.data.batch_contract import model_forward
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CTA_CONDITIONING_GIVEN,
    DEFAULT_DURATION_QUANTILE_LOSS_WEIGHT,
    DURATION_HEAD_POINT,
    DURATION_HEAD_QUANTILE,
    DURATION_HEAD_TWO_HEAD,
    DURATION_MEDIAN_INDEX,
    DURATION_QUANTILES,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from ts_transformer.outputs.envelope import CONTROL_LOWER, CONTROL_UPPER
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.data.dataset import build_series
from ts_transformer.outputs.control.forecast import duration_quantile_predictions
from ts_transformer.io_utils import file_sha256
from ts_transformer.backbone.adapters import build_model
import ts_transformer.outputs.control.loss.objective as control_objective
from ts_transformer.outputs.control.loss.objective import DURATION_QUANTILE_COMPONENT
from ts_transformer.training.objective import loss_component_names
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.duration_heads import QuantileFinalTimeHead, pinball_duration_loss
from ts_transformer.run_naming import run_display_name, run_slug
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.training.train import load_checkpoint, train

from ts_transformer.tests.support import dynamics_context
from ts_transformer.tests.support_prediction import quantile_config as _config

AIRPORT, RUNWAY = "KRDU", "05L"

PROVENANCE = {
    "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
    "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                   "source_records": []}],
}


def _two_head(**overrides) -> TSConfig:
    return _config(duration_head=DURATION_HEAD_TWO_HEAD, **overrides)


def _trained_two_head(tmp_path: Path, monkeypatch, **overrides):
    import ts_transformer.cli.predict as predict_module

    config = _two_head(final_time_loss_weight=26.0, **overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=PROVENANCE, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: PROVENANCE)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )
    return run / "checkpoint.pt", series


def _cli():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("ts_cli_two_head_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict(checkpoint: Path, out: Path, tmp_path: Path, *extra: str) -> int:
    return _cli().main([
        "predict", "--checkpoint", str(checkpoint), "--data", str(tmp_path / "manifest.json"),
        "--airport", AIRPORT, "--output-dir", str(out), "--split", "val", "--device", "cpu",
        *extra,
    ])


def test_the_eta_error_readout_reports_the_quantile_head_beside_the_rollout(
    tmp_path: Path, monkeypatch, capsys
):
    """Gate 1 of §三 3.1b is read on two different heads: ADE and the rollout's own duration
    on the point head, the arrival-time MAE on q50. The readout prints both blocks and
    derives q50's error from the row's own published quantiles."""
    import importlib.util

    checkpoint, _series = _trained_two_head(tmp_path, monkeypatch)
    out = tmp_path / "pred"
    assert _predict(checkpoint, out, tmp_path) == 0

    readout = importlib.import_module("ts_transformer.experiments.eta_error_readout")

    payload = readout.readout("B1b", out)
    assert payload["duration_head"] == DURATION_HEAD_TWO_HEAD
    assert readout.Q50_METRIC in payload["metrics"]
    rows = json.loads((out / "summary.json").read_text())["results"]
    expected = np.abs(np.array(
        [row["duration_quantiles_s"][DURATION_MEDIAN_INDEX] - row["true_final_time_s"]
         for row in rows], dtype=np.float64
    )).mean()
    pooled = next(iter(payload["strata"].values()))
    assert pooled["n"] == len(rows)
    assert pooled[readout.Q50_METRIC]["mae"] == pytest.approx(expected)
    # ...and the point head's own MAE is the block that was already there, on the duration
    # the rollout actually flew.
    assert pooled["final_time_error_s"]["mae"] == pytest.approx(
        np.abs(np.array([row["final_time_error_s"] for row in rows])).mean()
    )
    text = readout.render({"schema": readout.RESULT_SCHEMA, "arms": [payload]})
    assert "GATE 1" in text and "POINT head" in text
    assert readout.DURATION_MEDIAN_INDEX == DURATION_MEDIAN_INDEX


def test_a_point_head_directory_has_no_q50_block(tmp_path: Path, monkeypatch):
    """The block exists only where a quantile head wrote the rows, so a point-head arm's
    readout is the one B0 published."""
    import importlib.util

    import ts_transformer.cli.predict as predict_module

    config = _config(duration_head=DURATION_HEAD_POINT)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=PROVENANCE, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: PROVENANCE)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )
    out = tmp_path / "pred"
    assert _predict(run / "checkpoint.pt", out, tmp_path) == 0

    readout = importlib.import_module("ts_transformer.experiments.eta_error_readout")
    payload = readout.readout("point", out)
    assert readout.Q50_METRIC not in payload["metrics"]
    assert "GATE 1" not in readout.render(
        {"schema": readout.RESULT_SCHEMA, "arms": [payload]}
    )
