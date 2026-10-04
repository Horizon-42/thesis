"""The fixtures the one-tier prediction tests share (the state / control / quantile-head / latent / ETA-calibration paths).

`support.py` holds the two-tier line's fixtures (instruction artefacts, the executor, the closed loop); the helpers here
were defined again in each test file, or imported from another test file — a test module imported by a test module is
loaded twice under `--import-mode=importlib`. A test file imports what it needs from here, as the name it always used.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ts_transformer.data.channels as ch
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CTA_CONDITIONING_GIVEN,
    DURATION_HEAD_QUANTILE,
    DURATION_QUANTILES,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.calibration import CalibrationSample

AIRPORT, RUNWAY = "KRDU", "05L"
ETA_CHECKPOINT_SHA = "b" * 64


def series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def series_for(config: TSConfig, n_flights: int = 2):
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3), config, airport=AIRPORT
    )
    assert report.built == n_flights, report.format()
    return series


def control_config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        n_segments=8, seq_len=16, d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def identity_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(len(ch.CHANNELS), dtype=np.float64),
        std=np.ones(len(ch.CHANNELS), dtype=np.float64),
    )


def quantile_config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output="control",
        duration_head=DURATION_HEAD_QUANTILE,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        control_rollout_integrator_dt_s=0.5,
        seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu", horizon_mode="normalized",
        epochs=1, patience=1, batch_size=8, dropout=0.0,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def latent_config(**overrides) -> TSConfig:
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


def given_cta_config(**overrides) -> TSConfig:
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


def eta_covariates(tortuosity: float = 1.0, established: bool = False) -> dict:  # noqa: D401
    return {
        "route_tortuosity": tortuosity,
        "established_at_anchor": established,
        "remaining_path_m": 12_000.0,
        "anchor_range_m": 12_000.0,
        "anchor_cross_track_m": 0.0,
    }


def eta_z(tau: float) -> float:
    """The standard normal quantile, without scipy."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(tau)


def eta_cohort(
    count: int, *, narrow_s: float, seed: int = 0, tortuosity: float = 1.0,
    spread: float = 60.0, key_prefix: str = "flight",
) -> list[CalibrationSample]:
    """Flights whose head emits the TRUE quantiles of a N(400, 60) truth, narrowed by
    ``narrow_s``, while their OWN truth is drawn with standard deviation ``spread``.

    With ``spread == 60`` the head is right and the score of a narrowed interval is the
    true-interval score plus the narrowing, whose (1 − α) quantile is 0 by definition — so δ
    must come back as ``narrow_s``, whatever else changes. A larger ``spread`` makes the
    same head wrong for those flights, which is how a stratum that needs a much wider
    interval than the pooled one is built.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(400.0, spread, size=count)
    exact = np.array([400.0 + 60.0 * eta_z(tau) for tau in DURATION_QUANTILES])
    narrowing = np.array([+narrow_s, +narrow_s, 0.0, -narrow_s, -narrow_s])
    return [
        CalibrationSample(
            key=f"KRDU:{key_prefix}{index:04d}",
            quantiles_s=exact + narrowing,
            truth_final_time_s=float(value),
            covariates=eta_covariates(tortuosity),
        )
        for index, value in enumerate(truth)
    ]


def eta_metadata(tmp_path: Path, sha: str = ETA_CHECKPOINT_SHA) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "checkpoint_metadata.json"
    path.write_text(json.dumps({"checkpoint_sha256": sha, "schema_version": "x"}))
    return path


def eta_runner():
    import importlib.util

    return importlib.import_module("ts_transformer.experiments.eta_calibration")


def eta_stubbed_runner(monkeypatch, samples: list[CalibrationSample], split_seed: int,
                    duration_head: str = DURATION_HEAD_QUANTILE):
    """The runner with its checkpoint load, cohort rebuild and forward pass replaced.

    Those three are what make the real thing need a trained model and the arrival manifests,
    and they decide none of what is under test here: which seed cuts the halves, and whether
    the checkpoint's sidecar is written, are `main`'s own control flow. ``duration_head`` is
    the one config field it reads (B1.b's `two-head` calibrates the same way).
    """
    runner = eta_runner()
    truths = {sample.key: sample.truth_final_time_s for sample in samples}
    covariates = {sample.key: sample.covariates for sample in samples}
    arm = SimpleNamespace(
        config=SimpleNamespace(duration_head=duration_head,
                               resolved_split_seed=split_seed),
        airports=("KRDU",), model=object(), normalizer=object(),
    )
    monkeypatch.setattr(runner, "resolve_device", lambda _name: "cpu")
    monkeypatch.setattr(runner, "load_arm", lambda *args, **kwargs: arm)
    monkeypatch.setattr(runner, "cohort_series", lambda _arm, _grid: [
        SimpleNamespace(dataset_id=sample.key) for sample in samples
    ])
    monkeypatch.setattr(runner, "default_anchor", lambda _config: 0)
    monkeypatch.setattr(runner, "duration_quantile_predictions", lambda *args, **kwargs:
                        np.stack([sample.quantiles_s for sample in samples]))
    monkeypatch.setattr(runner, "truth_duration_s",
                        lambda item, _anchor: truths[item.dataset_id])
    monkeypatch.setattr(runner, "approach_difficulty", lambda item, _anchor:
                        SimpleNamespace(to_dict=lambda: covariates[item.dataset_id]))
    return runner
