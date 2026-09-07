"""Split-conformal calibration of the ETA interval (anytime design §三 3.2, B2).

CQR on the VALIDATION split, halved: δ from one half, coverage measured on the other, then
swapped. The table is a sidecar of ``checkpoint_metadata.json`` bound to the checkpoint's
digest, and `predict` stamps the interval on every record — or says ``calibrated: false``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import calibration
from approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
)
from calibration import (
    CONFORMAL_ALPHAS,
    CONFORMAL_METADATA_KEY,
    CONFORMAL_SCHEMA,
    CalibrationSample,
    calibrate,
    calibration_halves,
    conformal_delta,
    conformity_scores,
    interval_stratum,
    load_conformal_table,
    quantile_pair_indices,
    render,
    write_conformal_table,
)
from config import DURATION_HEAD_POINT, DURATION_QUANTILES

from tests.test_duration_quantiles import _config  # the tiny quantile-head config

CHECKPOINT_SHA = "b" * 64


def _covariates(tortuosity: float = 1.0, established: bool = False) -> dict:
    return {
        "route_tortuosity": tortuosity,
        "established_at_anchor": established,
        "remaining_path_m": 12_000.0,
        "anchor_range_m": 12_000.0,
        "anchor_cross_track_m": 0.0,
    }


def _cohort(
    count: int, *, narrow_s: float, seed: int = 0, tortuosity: float = 1.0
) -> list[CalibrationSample]:
    """Flights whose head emits the TRUE quantiles of the truth, narrowed by ``narrow_s``.

    The score of a narrowed interval is the true-interval score plus the narrowing, and the
    true interval's (1 − α) score quantile is 0 by definition — so δ must come back as
    ``narrow_s``, whatever the distribution. That is the known miscalibration.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(400.0, 60.0, size=count)
    exact = np.array([400.0 + 60.0 * _z(tau) for tau in DURATION_QUANTILES])
    narrowing = np.array([+narrow_s, +narrow_s, 0.0, -narrow_s, -narrow_s])
    return [
        CalibrationSample(
            key=f"KRDU:flight{index:04d}",
            quantiles_s=exact + narrowing,
            truth_final_time_s=float(value),
            covariates=_covariates(tortuosity),
        )
        for index, value in enumerate(truth)
    ]


def _z(tau: float) -> float:
    """The standard normal quantile, without scipy."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(tau)


# ── the arithmetic ──────────────────────────────────────────────────────────

def test_the_conformity_score_is_the_distance_outside_the_interval():
    lo = np.array([100.0, 100.0, 100.0])
    hi = np.array([200.0, 200.0, 200.0])
    truth = np.array([150.0, 80.0, 260.0])
    assert list(conformity_scores(lo, hi, truth)) == [-50.0, 20.0, 60.0]


def test_the_conformal_quantile_uses_the_finite_sample_level():
    scores = np.arange(100, dtype=np.float64)          # 0 … 99
    # ceil((100 + 1) * 0.8) / 100 = 0.81 -> the 81st order statistic, not the 80th.
    assert conformal_delta(scores, 0.2) == 81.0
    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_delta(scores, 1.0)


def test_the_halves_are_deterministic_equal_and_disjoint():
    keys = [f"KRDU:f{i}" for i in range(101)]
    first, second = calibration_halves(keys, 7)
    assert sorted([*first, *second]) == list(range(101))
    assert abs(len(first) - len(second)) <= 1
    assert (calibration_halves(keys, 7)[0] == first).all()
    assert not (calibration_halves(keys, 8)[0] == first).all()   # the seed decides


# ── the table ───────────────────────────────────────────────────────────────

def test_delta_recovers_a_known_miscalibration():
    """The head is narrowed by exactly 25 s on each side; δ must be 25 s."""
    table = calibrate(
        _cohort(1600, narrow_s=25.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    block = table["alphas"]["0.2"]["strata"][STRATUM_ALL]
    assert block["delta_s"] == pytest.approx(25.0, abs=6.0)
    # ...and the interval it produces covers what it says it does, on the OTHER half.
    assert block["coverage"] == pytest.approx(0.8, abs=0.04)
    assert block["coverage_halves"] == [pytest.approx(0.8, abs=0.06)] * 2
    half = table["alphas"]["0.5"]["strata"][STRATUM_ALL]
    assert half["coverage"] == pytest.approx(0.5, abs=0.05)


def test_a_head_that_is_already_calibrated_needs_no_widening():
    table = calibrate(
        _cohort(1600, narrow_s=0.0, seed=5), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert table["alphas"]["0.2"]["strata"][STRATUM_ALL]["delta_s"] == pytest.approx(0.0, abs=6.0)


def test_a_thin_stratum_refuses_and_the_flight_falls_through_to_the_pooled_one():
    # Every flight straight-in: the vectored and established strata are empty.
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=3, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    block = table["alphas"]["0.2"]
    assert STRATUM_STRAIGHT_IN in block["strata"] and STRATUM_ALL in block["strata"]
    assert block["refused_strata"][STRATUM_VECTORED] == [0, 0]
    assert STRATUM_ESTABLISHED in block["refused_strata"]
    # A vectored flight has no delta of its own, so it reads the pooled one.
    assert interval_stratum(table, _covariates(tortuosity=2.4)) == STRATUM_ALL
    assert interval_stratum(table, _covariates(tortuosity=1.0)) == STRATUM_STRAIGHT_IN


def test_a_cohort_too_small_to_calibrate_at_all_is_refused():
    with pytest.raises(ValueError, match="nothing to calibrate"):
        calibrate(
            _cohort(20, narrow_s=0.0), split_seed=1, split="val",
            checkpoint_sha256=CHECKPOINT_SHA,
        )
    with pytest.raises(ValueError, match="at least one flight"):
        calibrate([], split_seed=1, split="val", checkpoint_sha256=CHECKPOINT_SHA)


def test_the_interval_is_the_quantile_pair_widened_by_its_own_delta():
    table = calibrate(
        _cohort(800, narrow_s=15.0, seed=2), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    quantiles = [200.0, 300.0, 400.0, 500.0, 600.0]
    intervals = calibration.conformal_intervals(quantiles, table, STRATUM_ALL)
    assert [entry["alpha"] for entry in intervals] == list(CONFORMAL_ALPHAS)
    for entry in intervals:
        low_index, high_index = quantile_pair_indices(entry["alpha"])
        delta = table["alphas"][f"{entry['alpha']:g}"]["strata"][STRATUM_ALL]["delta_s"]
        assert entry["lo"] == pytest.approx(quantiles[low_index] - delta)
        assert entry["hi"] == pytest.approx(quantiles[high_index] + delta)
        assert entry["lo"] <= entry["hi"]


def test_the_readout_prints_both_halves_coverage():
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    text = render(table)
    assert "dA->B" in text and "dB->A" in text and "refused" in text
    assert "alpha 0.2" in text and "alpha 0.5" in text


# ── the sidecar ─────────────────────────────────────────────────────────────

def _metadata(tmp_path: Path, sha: str = CHECKPOINT_SHA) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "checkpoint_metadata.json"
    path.write_text(json.dumps({"checkpoint_sha256": sha, "schema_version": "x"}))
    return path


def test_the_table_round_trips_through_the_checkpoint_metadata(tmp_path: Path):
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    metadata_path = _metadata(tmp_path)
    write_conformal_table(metadata_path, table)
    stored = json.loads(metadata_path.read_text())
    assert stored["schema_version"] == "x"          # merged, not replaced
    assert stored[CONFORMAL_METADATA_KEY]["schema"] == CONFORMAL_SCHEMA
    loaded = load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA)
    assert loaded == table


def test_a_table_from_another_checkpoint_is_refused_not_applied(tmp_path: Path):
    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    metadata_path = _metadata(tmp_path)
    write_conformal_table(metadata_path, table)
    with pytest.raises(ValueError, match="was fitted on checkpoint"):
        load_conformal_table(tmp_path / "checkpoint.pt", "c" * 64)
    with pytest.raises(ValueError, match="belongs to checkpoint"):
        write_conformal_table(_metadata(tmp_path / "other", sha="d" * 64), table)


def test_an_unknown_schema_is_refused(tmp_path: Path):
    metadata_path = _metadata(tmp_path)
    metadata_path.write_text(json.dumps({
        "checkpoint_sha256": CHECKPOINT_SHA,
        CONFORMAL_METADATA_KEY: {"schema": "ts-conformal-cqr-v0",
                                 "checkpoint_sha256": CHECKPOINT_SHA},
    }))
    with pytest.raises(ValueError, match="conformal schema"):
        load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA)


def test_no_metadata_and_no_table_both_read_as_uncalibrated(tmp_path: Path):
    assert load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA) is None
    _metadata(tmp_path)
    assert load_conformal_table(tmp_path / "checkpoint.pt", CHECKPOINT_SHA) is None


# ── the runner's refusals ───────────────────────────────────────────────────

def _runner():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "run_ts_eta_calibration.py"
    spec = importlib.util.spec_from_file_location("run_ts_eta_calibration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_refuses_every_split_but_val(tmp_path: Path):
    runner = _runner()
    for split in ("test", "train"):
        with pytest.raises(SystemExit):
            runner.main(["--checkpoint", str(tmp_path / "checkpoint.pt"),
                         "--out", str(tmp_path / "out"), "--split", split])


# ── predict reads it ────────────────────────────────────────────────────────

def test_predict_stamps_the_calibrated_interval_on_every_record(tmp_path: Path, monkeypatch):
    """End to end on a tiny checkpoint: uncalibrated records say so, and after a table is
    written beside the checkpoint the same command writes an interval per alpha."""
    import importlib.util

    import cli.predict as predict_module
    from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
    from dataset import build_series
    from synthetic import synthetic_arrivals
    from train import load_checkpoint, train

    cli_path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("ts_cli_calibration_test", cli_path)
    ts_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts_cli)

    config = _config()
    flights = synthetic_arrivals("KRDU", "05L", n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport="KRDU")
    provenance = {"schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
                  "manifests": [{"airport": "KRDU", "arrival_manifest_sha256": "a" * 64,
                                 "source_records": []}]}
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=provenance, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: provenance)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )

    def predict(out: Path) -> int:
        return ts_cli.main([
            "predict", "--checkpoint", str(run / "checkpoint.pt"),
            "--data", str(tmp_path / "manifest.json"), "--airport", "KRDU",
            "--output-dir", str(out), "--split", "val", "--device", "cpu",
        ])

    assert predict(tmp_path / "raw") == 0
    raw = json.loads((tmp_path / "raw" / "summary.json").read_text())["results"][0]
    assert raw["calibrated"] is False and raw["duration_interval_s"] is None
    assert len(raw["duration_quantiles_s"]) == len(DURATION_QUANTILES)

    # A table for THIS checkpoint, fitted on a synthetic cohort (the runner's own path
    # needs the real arrival manifests; the table it writes is this object).
    _model, loaded, _normalizer, _payload = load_checkpoint(run / "checkpoint.pt")
    from io_utils import file_sha256

    table = calibrate(
        _cohort(400, narrow_s=12.0), split_seed=loaded.resolved_split_seed, split="val",
        checkpoint_sha256=file_sha256(run / "checkpoint.pt"),
    )
    write_conformal_table(run / "checkpoint_metadata.json", table)

    assert predict(tmp_path / "calibrated") == 0
    row = json.loads((tmp_path / "calibrated" / "summary.json").read_text())["results"][0]
    assert row["calibrated"] is True
    assert [entry["alpha"] for entry in row["duration_interval_s"]] == list(CONFORMAL_ALPHAS)
    assert row["duration_interval_stratum"] in (
        STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
    )
    states = json.loads((tmp_path / "calibrated" / row["states_file"]).read_text())
    assert states["source"]["durationIntervalS"] == row["duration_interval_s"]
    # The interval brackets the quantile pair it was built from.
    quantiles = row["duration_quantiles_s"]
    for entry in row["duration_interval_s"]:
        low_index, high_index = quantile_pair_indices(entry["alpha"])
        assert entry["lo"] <= entry["hi"]
        assert entry["hi"] - entry["lo"] >= 0.0
        assert entry["lo"] == pytest.approx(
            quantiles[low_index]
            - table["alphas"][f"{entry['alpha']:g}"]["strata"][
                row["duration_interval_stratum"]]["delta_s"]
        )


def test_neither_half_of_the_pair_alone_produces_an_interval():
    """The interval needs BOTH a quantile forecast and a table: a point-head forecast has
    nothing to widen, and an uncalibrated quantile forecast has nothing to widen it by."""
    from forecast import _calibrated_interval_fields

    table = calibrate(
        _cohort(400, narrow_s=10.0), split_seed=1, split="val",
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert _config(duration_head=DURATION_HEAD_POINT).duration_head == DURATION_HEAD_POINT
    assert _calibrated_interval_fields(_UNUSED_SERIES, 0, None, table) == {}
    assert _calibrated_interval_fields(_UNUSED_SERIES, 0, np.zeros(5), None) == {}


#: Neither branch above reaches the series, and building one would hide that.
_UNUSED_SERIES = object()
