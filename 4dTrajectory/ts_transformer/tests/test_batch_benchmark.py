"""Focused tests for the standalone TS batch-throughput benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


TS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(TS_ROOT), str(REPO_ROOT)]

import batch_benchmark  # noqa: E402
import channels  # noqa: E402
from config import PREDICTION_CONTROL, TSConfig  # noqa: E402
from dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402


def test_candidate_grid_and_throughput_selection() -> None:
    assert batch_benchmark.candidate_batch_sizes(32, 256) == [32, 64, 128, 256]
    best = batch_benchmark.select_best_batch([
        {"batch_size": 64, "status": "ok", "median_samples_per_second": 1000.0},
        {"batch_size": 128, "status": "ok", "median_samples_per_second": 1400.0},
        {"batch_size": 256, "status": "oom"},
    ])
    assert best["batch_size"] == 128


def test_benchmark_candidate_executes_current_training_loss(monkeypatch) -> None:
    config = TSConfig(
        device="cpu",
        seq_len=20,
        n_segments=8,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
    )
    flights = synthetic_arrivals("KRDU", "05L", n_flights=4, seed=3)
    series, report = build_series(flights, config, airport="KRDU")
    assert report.built == 4, report.format()
    normalizer = Normalizer(
        mean=np.zeros(len(channels.CHANNELS), dtype=np.float64),
        std=np.ones(len(channels.CHANNELS), dtype=np.float64),
    )
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)

    result = batch_benchmark.benchmark_candidate(
        windows,
        config,
        torch.device("cpu"),
        batch_size=2,
        warmup_steps=1,
        measure_steps=1,
        repeats=1,
    )

    assert result["status"] == "ok"


def test_benchmark_candidate_executes_control_rollout_loss(monkeypatch) -> None:
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        device="cpu",
        seq_len=20,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
    )
    flights = synthetic_arrivals("KRDU", "05L", n_flights=4, seed=5)
    series, report = build_series(flights, config, airport="KRDU")
    assert report.built == 4, report.format()
    normalizer = Normalizer.fit(series)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 0)

    result = batch_benchmark.benchmark_candidate(
        windows,
        config,
        torch.device("cpu"),
        batch_size=2,
        warmup_steps=1,
        measure_steps=1,
        repeats=1,
    )

    assert result["status"] == "ok"
