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

from ts_transformer.calibration import (
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
from ts_transformer.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from flight_scenarios.identity import summary_row_key
from ts_transformer.dataset import build_series
from ts_transformer.io_utils import file_sha256
from ts_transformer.run_naming import run_display_name
from ts_transformer.synthetic import synthetic_arrivals
from ts_transformer.train import load_checkpoint, train

from ts_transformer.tests.test_duration_quantiles import _config
from ts_transformer.tests.test_eta_calibration import _cohort

AIRPORT, RUNWAY = "KRDU", "05L"

_CLI_SPEC = importlib.util.spec_from_file_location(
    "ts_transformer_cli_fan_test", Path(__file__).resolve().parents[1] / "__main__.py"
)
ts_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(ts_cli)

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


# ── refusals ────────────────────────────────────────────────────────────────

def test_the_fan_needs_a_cta_conditioned_checkpoint(tmp_path: Path, monkeypatch):
    checkpoint, _series = _trained(tmp_path, monkeypatch, cta_conditioning="off")
    with pytest.raises(SystemExit):
        _predict(checkpoint, tmp_path / "out", tmp_path, "--cta-from-quantiles")


def test_the_fan_needs_a_quantile_duration_head(tmp_path: Path, monkeypatch):
    checkpoint, _series = _trained(tmp_path, monkeypatch, duration_head=DURATION_HEAD_POINT)
    with pytest.raises(SystemExit):
        _predict(checkpoint, tmp_path / "out", tmp_path, "--cta-from-quantiles")


def test_the_fan_is_refused_with_the_counterfactual_offset_and_the_z_oracle(
    tmp_path: Path, monkeypatch
):
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _predict(checkpoint, tmp_path / "a", tmp_path, "--cta-from-quantiles",
                 "--cta-offset-s", "30")
    with pytest.raises(SystemExit):
        _predict(checkpoint, tmp_path / "b", tmp_path, "--cta-from-quantiles",
                 "--z-from-posterior")


# ── the fan ─────────────────────────────────────────────────────────────────

def test_the_fan_decodes_every_flight_at_its_own_quantiles(tmp_path: Path, monkeypatch):
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    out = tmp_path / "fan"
    assert _predict(checkpoint, out, tmp_path, "--cta-from-quantiles") == 0

    top1 = json.loads((out / "summary.json").read_text())
    # The name says which arm this is, and it is NOT the oracle.
    assert top1["config"]["cta_conditioning"] == CTA_CONDITIONING_SELF_QUANTILE
    name = run_display_name(top1["config"])
    assert "cta=self-q" in name and "cta=given" not in name
    assert top1["mode"].endswith(":cta-self-q0.5")

    median_name = quantile_directory_name(DURATION_QUANTILES[DURATION_MEDIAN_INDEX])
    for row in top1["results"]:
        assert row["cta_from_quantiles"] is True
        assert row["cta_quantile"] == 0.5
        assert row["cta_offset_s"] is None                       # there is no truth to offset
        # The top-1 IS the q50 decode: the CTA the rollout flew is the flight's own median.
        assert row["cta_s"] == pytest.approx(row["duration_quantiles_s"][DURATION_MEDIAN_INDEX])
        assert row["predicted_final_time_s"] == pytest.approx(row["cta_s"])

    by_key = {}
    for tau in DURATION_QUANTILES:
        directory = out / QUANTILE_DIR_NAME / quantile_directory_name(tau)
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["mode"].endswith(f":cta-self-q{tau:g}")
        assert len(summary["results"]) == len(top1["results"])
        for row in summary["results"]:
            assert row["cta_quantile"] == tau
            by_key.setdefault(summary_row_key(row), []).append(row["cta_s"])
    # Every flight's fan is ordered, because its quantiles are.
    for durations in by_key.values():
        assert durations == sorted(durations)

    # The q50 leaf and the top-1 are the same decode, not two.
    median = json.loads((out / QUANTILE_DIR_NAME / median_name / "summary.json").read_text())
    assert [row["cta_s"] for row in median["results"]] == [
        row["cta_s"] for row in top1["results"]
    ]
    # An uncalibrated fan has no interval endpoints to decode.
    assert not (out / QUANTILE_DIR_NAME / interval_directory_name(FAN_INTERVAL_ALPHA, "lo")).exists()


def _write_table(checkpoint: Path, *, delta_s: float = 0.0):
    """A conformal table for THIS checkpoint, with every delta planted at ``delta_s``.

    The synthetic calibration cohort lives on the real duration scale (hundreds of seconds)
    while this tiny model emits durations of about two, so its own fitted delta would swamp
    the model's whole interval. δ recovery is measured in `test_eta_calibration`; what these
    tests need is a VALID table whose deltas are on the model's scale.
    """
    _model, loaded, _normalizer, _payload = load_checkpoint(checkpoint)
    table = calibrate(
        _cohort(400, narrow_s=0.0), split_seed=loaded.resolved_split_seed, split="val",
        checkpoint_sha256=file_sha256(checkpoint), airports=("KRDU",),
    )
    for block in table["alphas"].values():
        for cell in block["strata"].values():
            cell["delta_s"] = delta_s
    write_conformal_table(checkpoint.parent / "checkpoint_metadata.json", table)
    return table


def test_the_endpoints_are_opt_in(tmp_path: Path, monkeypatch):
    """The fan readout scores the five quantile leaves; the calibrated endpoints are an
    extra arm, so a calibrated run without the flag writes exactly the five."""
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    _write_table(checkpoint)
    out = tmp_path / "five"
    assert _predict(checkpoint, out, tmp_path, "--cta-from-quantiles") == 0
    leaves = sorted(path.name for path in (out / QUANTILE_DIR_NAME).iterdir())
    assert leaves == sorted(quantile_directory_name(tau) for tau in DURATION_QUANTILES)
    # ...and the flag needs the fan it is part of.
    with pytest.raises(SystemExit):
        _predict(checkpoint, tmp_path / "lonely", tmp_path, "--interval-endpoints")


def test_a_fan_cta_that_the_rollout_cannot_fly_is_refused(tmp_path: Path, monkeypatch, capsys):
    """A delta wider than q10 drives the lo endpoint to zero or below. The batch fails
    loudly with that diagnosis — never clamped, never skipped, because leaves holding
    different flights are not a fan."""
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    _write_table(checkpoint, delta_s=1_000.0)      # far wider than this model's q10
    with pytest.raises(SystemExit):
        _predict(checkpoint, tmp_path / "broken", tmp_path,
                 "--cta-from-quantiles", "--interval-endpoints")
    # argparse exits with a status; the diagnosis is on stderr and names the cause.
    message = capsys.readouterr().err
    assert "not positive" in message and "conformal delta wider" in message


def test_a_calibrated_fan_also_decodes_the_interval_endpoints(tmp_path: Path, monkeypatch):
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    table = _write_table(checkpoint)
    out = tmp_path / "fan"
    assert _predict(
        checkpoint, out, tmp_path, "--cta-from-quantiles", "--interval-endpoints"
    ) == 0

    for end in ("lo", "hi"):
        directory = out / QUANTILE_DIR_NAME / interval_directory_name(FAN_INTERVAL_ALPHA, end)
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["mode"].endswith(f":cta-self-q-a{FAN_INTERVAL_ALPHA:g}{end}")
        for row in summary["results"]:
            assert row["cta_quantile"] is None
            assert row["cta_from_quantiles"] is True
            states = json.loads((directory / row["states_file"]).read_text())
            assert states["source"]["ctaInterval"] == {"alpha": FAN_INTERVAL_ALPHA, "end": end}
            # Every calibrated record names the table it came from, so a narrowed-airport
            # run deployed on a pooled delta is visible without opening the checkpoint.
            assert states["source"]["durationIntervalCohort"] == {
                "split": table["split"],
                "airports": table["airports"],
                "smokeTest": table["smoke_test"],
            }
            # ...and the endpoint IS the calibrated interval the record itself publishes,
            # looked up by its own alpha rather than by position in the list.
            published = next(
                entry for entry in row["duration_interval_s"]
                if entry["alpha"] == FAN_INTERVAL_ALPHA
            )
            assert row["cta_s"] == pytest.approx(published[end])


# ── the readout ─────────────────────────────────────────────────────────────

def test_the_fan_readout_runs_on_a_predicted_directory(tmp_path: Path, monkeypatch, capsys):
    checkpoint, _series = _trained(tmp_path, monkeypatch)
    out = tmp_path / "fan"
    assert _predict(checkpoint, out, tmp_path, "--cta-from-quantiles") == 0

    spec = importlib.util.spec_from_file_location(
        "run_ts_quantile_fan_readout_test",
        Path(__file__).resolve().parents[3] / "run_ts_quantile_fan_readout.py",
    )
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
    assert readout.main(["--arm", str(out), "--json", str(tmp_path / "fan.json")]) == 0
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
    spec = importlib.util.spec_from_file_location(
        "run_ts_quantile_fan_readout_missing",
        Path(__file__).resolve().parents[3] / "run_ts_quantile_fan_readout.py",
    )
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
    with pytest.raises(SystemExit, match="no complete quantile fan"):
        readout.main(["--arm", str(out)])
