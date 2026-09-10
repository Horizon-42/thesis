"""Cross-validation: outer-train only, the atomic resume, the contract check, the default grid.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import itertools
import json
from types import SimpleNamespace

import pytest
import torch

import ts_transformer.cross_validation as cv
from ts_transformer.config import (
    HORIZON_FULL,
    HORIZON_WINDOW,
    CHECKPOINT_SELECTION_METRICS,
    TSConfig,
)
from ts_transformer.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.dataset import Normalizer, build_series
from ts_transformer.splits import split_by_flight
from ts_transformer.synthetic import synthetic_arrivals
# Imported, never restated: a schema version pinned by hand in a fixture is a version
# the fixture cannot check, and this one gates every loader that reads the roster.

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def test_cross_validation_never_passes_outer_val_or_test_to_fit(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=36, device="cpu", epochs=1, patience=1,
        d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    outer_train, outer_val, outer_test = split_by_flight(series, config)
    forbidden = {item.dataset_id for item in [*outer_val, *outer_test]}
    observed: list[set[str]] = []

    def fake_fit(train_series, val_series, fold_config, **_kwargs):
        identities = {item.dataset_id for item in [*train_series, *val_series]}
        observed.append(identities)
        score = float(fold_config.learning_rate + fold_config.d_model * 1e-8)
        row = SimpleNamespace(
            epoch=1, val_loss=score, val_by_airport={AIRPORT: score}
        )
        return SimpleNamespace(
            history=[row], best_val_loss=score, config=fold_config,
            model=object(), normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", fake_fit)
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=3,
        cv_parameters=("learning_rate",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    assert observed
    assert all(not (identities & forbidden) for identities in observed)
    assert set.union(*observed) == {item.dataset_id for item in outer_train}
    assert result["leakage_guard"]["outer_test_used"] is False
    assert result["schema_version"] == cv.RESULTS_SCHEMA
    assert result["selection_metric"] == (
        "mean outer-train-fold airport-macro fixed-anchor common physical-time ADE"
    )
    assert (tmp_path / "best_config.json").is_file()


def test_cross_validation_resumes_from_atomic_candidate_checkpoint(
    tmp_path, monkeypatch
):
    series, config = _series(
        n_flights=24,
        device="cpu",
        epochs=1,
        patience=1,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    first_run_calls = 0

    def interrupted_fit(train_series, _val_series, fold_config, **_kwargs):
        nonlocal first_run_calls
        first_run_calls += 1
        if first_run_calls == 3:
            raise KeyboardInterrupt
        score = float(fold_config.weight_decay + fold_config.seed * 1e-9)
        row = SimpleNamespace(
            epoch=1,
            val_loss=score,
            val_by_airport={AIRPORT: score},
        )
        return SimpleNamespace(
            history=[row],
            best_val_loss=score,
            config=fold_config,
            model=object(),
            normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", interrupted_fit)
    with pytest.raises(KeyboardInterrupt):
        cv.cross_validate(
            series,
            config,
            output_dir=tmp_path,
            data_provenance=provenance,
            n_splits=2,
            cv_parameters=("weight_decay",),
            cv_epochs=1,
            cv_patience=1,
            verbose=False,
        )

    progress_path = tmp_path / cv.PROGRESS_NAME
    progress = json.loads(progress_path.read_text())
    assert progress["schema_version"] == cv.PROGRESS_SCHEMA
    assert progress["completed_candidates"] == 1
    assert [row["candidate"] for row in progress["candidates"]] == [0]
    assert not progress_path.with_suffix(progress_path.suffix + ".tmp").exists()

    resumed_calls = 0

    def resumed_fit(train_series, _val_series, fold_config, **_kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        score = float(fold_config.weight_decay + fold_config.seed * 1e-9)
        row = SimpleNamespace(
            epoch=1,
            val_loss=score,
            val_by_airport={AIRPORT: score},
        )
        return SimpleNamespace(
            history=[row],
            best_val_loss=score,
            config=fold_config,
            model=object(),
            normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", resumed_fit)
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )

    assert resumed_calls == 2
    assert [row["candidate"] for row in result["candidates"]] == [0, 1]
    completed = json.loads(progress_path.read_text())
    assert completed["completed_candidates"] == 2
    assert completed["run_contract_sha256"] == result["run_contract_sha256"]


def test_cross_validation_rejects_candidate_checkpoint_from_another_contract(
    tmp_path, monkeypatch
):
    series, config = _series(
        n_flights=24,
        device="cpu",
        epochs=1,
        patience=1,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }

    def fake_fit(train_series, _val_series, fold_config, **_kwargs):
        score = float(fold_config.weight_decay + fold_config.seed * 1e-9)
        row = SimpleNamespace(
            epoch=1,
            val_loss=score,
            val_by_airport={AIRPORT: score},
        )
        return SimpleNamespace(
            history=[row],
            best_val_loss=score,
            config=fold_config,
            model=object(),
            normalizer=Normalizer.fit(train_series),
            device=torch.device("cpu"),
        )

    monkeypatch.setattr(cv, "fit_model", fake_fit)
    cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    progress_path = tmp_path / cv.PROGRESS_NAME
    progress = json.loads(progress_path.read_text())
    progress["run_contract_sha256"] = "0" * 64
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    # The refusal must name the FILE and the remedy: every progress file written before the
    # 2026-09-08 eligible-set rename lands here, and deleting it is the only way forward.
    with pytest.raises(
        ValueError,
        match=rf"{cv.PROGRESS_NAME}.*current CV run contract.*Delete the file",
    ):
        cv.cross_validate(
            series,
            config,
            output_dir=tmp_path,
            data_provenance=provenance,
            n_splits=2,
            cv_parameters=("weight_decay",),
            cv_epochs=1,
            cv_patience=1,
            verbose=False,
        )


def test_cross_validation_describes_every_checkpoint_selection_metric():
    assert set(cv.SELECTION_METRIC_DESCRIPTIONS) == set(CHECKPOINT_SELECTION_METRICS)


def test_cross_validation_runs_real_two_fold_search(tmp_path):
    series, config = _series(
        n_flights=20, device="cpu", epochs=1, patience=1,
        d_model=16, d_ff=32, n_heads=4, e_layers=1,
        seq_len=20, n_segments=10, batch_size=32,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }
    result = cv.cross_validate(
        series,
        config,
        output_dir=tmp_path,
        data_provenance=provenance,
        n_splits=2,
        cv_parameters=("weight_decay",),
        cv_epochs=1,
        cv_patience=1,
        verbose=False,
    )
    assert len(result["candidates"]) == 2
    assert all(len(candidate["folds"]) == 2 for candidate in result["candidates"])
    assert all(
        "validation_selection_by_airport" in fold
        and "validation_metrics" not in fold
        for candidate in result["candidates"]
        for fold in candidate["folds"]
    )
    assert result["base_config"]["n_segments"] == config.n_segments
    assert "n_segments" not in result["best_overrides"]
    assert json.loads((tmp_path / "best_config.json").read_text()) == result["best_overrides"]


def test_cross_validation_exhausts_the_default_three_parameter_grid():
    config = TSConfig(n_segments=128)
    candidates = cv._candidate_overrides(config)

    assert len(candidates) == 45
    assert {
        (candidate["n_segments"], candidate["learning_rate"], candidate["d_model"])
        for candidate in candidates
    } == set(itertools.product(
        (16, 32, 64, 128, 256),
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
    ))
    assert all(candidate["d_ff"] == 2 * candidate["d_model"] for candidate in candidates)


@pytest.mark.parametrize("horizon_mode", [HORIZON_FULL, HORIZON_WINDOW])
def test_fixed_horizon_cv_does_not_repeat_inert_n_segment_candidates(horizon_mode):
    config = TSConfig(horizon_mode=horizon_mode)
    candidates = cv._candidate_overrides(config)

    assert len(candidates) == 9
    assert all("n_segments" not in candidate for candidate in candidates)
    assert {
        (candidate["learning_rate"], candidate["d_model"])
        for candidate in candidates
    } == set(itertools.product(
        (1e-4, 3e-4, 5e-4),
        (64, 128, 256),
    ))
