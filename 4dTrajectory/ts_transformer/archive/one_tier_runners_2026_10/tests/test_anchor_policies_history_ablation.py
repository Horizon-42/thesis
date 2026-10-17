"""The anchor policies: fixed, common, random and remaining-path-uniform, and the cohort floor.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import math

from dataclasses import replace

from types import SimpleNamespace

import numpy as np

import pytest

import torch

import ts_transformer.data.channels as ch

import ts_transformer.experiments.history_ablation as history_ablation

import ts_transformer.training.train as train_module

from ts_transformer.outputs.control.strategy import CONTROL_ANCHOR_STALL_MARGIN

from ts_transformer.outputs import strategy as output_strategy

from ts_transformer.config import PREDICTION_CONTROL, TSConfig

from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA

from ts_transformer.data.dataset import (
    FixedAnchorTrajectoryWindows,
    FlightEpochSampler,
    Normalizer,
    RandomAnchorTrajectoryWindows,
    build_series,
)

from ts_transformer.data.splits import split_by_flight

from ts_transformer.data.synthetic import synthetic_arrivals

from ts_transformer.tests.support_prediction import series as _series

AIRPORT, RUNWAY = "KRDU", "05L"

def test_history_ablation_runs_on_common_outer_train_anchors(tmp_path):
    series, config = _series(
        n_flights=20,
        device="cpu",
        seq_len=20,
        n_segments=8,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        batch_size=32,
        epochs=1,
        patience=1,
    )
    provenance = {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": AIRPORT,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }

    result = history_ablation.run_history_ablation(
        series,
        config,
        seq_lens=(10, 20),
        data_provenance=provenance,
        output_dir=tmp_path,
        n_splits=2,
        epochs=1,
        patience=1,
        auto_batch_size=False,
        verbose=False,
    )

    assert result["selected_seq_len"] in (10, 20)
    assert result["common_anchor"]["index"] == 19
    assert result["leakage_guard"]["outer_test_used"] is False
    signatures = [
        history_ablation.anchor_signature(
            split_by_flight(series, replace(config, seq_len=20))[0],
            replace(config, seq_len=seq_len),
            19,
        )
        for seq_len in (10, 20)
    ]
    assert signatures[0] == signatures[1]
    for name in (
        "history_length_ablation.json",
        "best_history_length.json",
        "history_length_candidates.csv",
        "history_length_folds.csv",
        "plots/history_length_cv_loss.png",
        "plots/history_length_metrics.svg",
        "plots/index.md",
    ):
        assert (tmp_path / name).is_file(), name
