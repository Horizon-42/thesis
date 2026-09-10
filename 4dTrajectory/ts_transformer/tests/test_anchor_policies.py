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


def test_fixed_anchor_dataset_keeps_one_window_per_flight():
    series, config = _series(n_flights=4)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    assert len(dataset) == len(series)
    assert all(anchor == config.seq_len - 1 for _series_index, anchor in dataset.index)


def test_vectorized_batch_matches_individual_random_anchor_samples():
    series, config = _series(n_flights=3, n_segments=16)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    indices = np.array([0, len(dataset) // 2, len(dataset) - 1])

    batch = dataset.batch(indices)
    individual = [dataset[int(index)] for index in indices]
    for field, values in enumerate(batch):
        assert torch.equal(values, torch.stack([sample[field] for sample in individual]))


def test_common_anchor_is_independent_of_history_length():
    series, base = _series(n_flights=2)
    normalizer = Normalizer.fit(series)
    common_anchor = 89
    datasets = [
        FixedAnchorTrajectoryWindows(
            series,
            replace(base, seq_len=seq_len),
            normalizer,
            minimum_anchor_index=common_anchor,
        )
        for seq_len in (30, 60, 90)
    ]

    assert all(dataset.index[0] == (0, common_anchor) for dataset in datasets)
    samples = [tuple(t[0] for t in dataset.batch([0])[:5]) for dataset in datasets]
    assert [sample[0].shape[0] for sample in samples] == [30, 60, 90]
    assert all(torch.equal(samples[0][1], sample[1]) for sample in samples[1:])
    assert all(float(samples[0][3]) == pytest.approx(float(sample[3])) for sample in samples[1:])


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


def test_fixed_epoch_sampler_uses_every_flight_once_and_reshuffles():
    series, config = _series(n_flights=8)
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    first = list(FlightEpochSampler(dataset, seed=7))
    repeated = list(FlightEpochSampler(dataset, seed=7))
    reshuffled = list(FlightEpochSampler(dataset, seed=8))

    assert first == repeated
    assert first != reshuffled
    assert len(first) == len(series)
    assert {dataset.index[index][0] for index in first} == set(range(len(series)))


def test_random_epoch_sampler_selects_one_valid_anchor_per_flight():
    series, config = _series(n_flights=8)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))
    first = list(FlightEpochSampler(dataset, seed=7))
    second = list(FlightEpochSampler(dataset, seed=8))

    for epoch in (first, second):
        assert len(epoch) == len(series)
        assert {dataset.index[index][0] for index in epoch} == set(range(len(series)))
    assert {dataset.index[index] for index in first} != {
        dataset.index[index] for index in second
    }


def test_random_anchor_requires_sixty_seconds_of_future_by_default():
    series, config = _series(n_flights=4)
    dataset = RandomAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    assert dataset.minimum_future_s == pytest.approx(60.0)
    assert dataset.index
    for series_index, anchor in dataset.index:
        remaining = (
            series[series_index].supervision_times[-1]
            - series[series_index].times[anchor]
        )
        assert remaining >= 60.0 - 1e-9


def test_random_anchor_choice_is_stable_per_flight_when_roster_order_changes():
    series, config = _series(n_flights=8)

    def selected(group):
        dataset = RandomAnchorTrajectoryWindows(group, config, Normalizer.fit(group))
        return {
            dataset.series[dataset.index[index][0]].dataset_id: dataset.index[index][1]
            for index in dataset.epoch_indices(seed=17)
        }

    assert selected(series) == selected(list(reversed(series)))


def test_control_random_anchor_candidates_require_airborne_stall_margin():
    config = TSConfig(prediction_output=PREDICTION_CONTROL)
    stall_speed = math.sqrt(
        2.0 * 60_000.0 * 9.81 / (1.225 * 100.0 * 2.0)
    )
    values = np.zeros((4, len(ch.CHANNELS)))
    values[:, ch.IDX["edot"]] = [
        0.0, stall_speed, 1.11 * stall_speed, 2.0 * stall_speed,
    ]
    series = SimpleNamespace(
        values=values,
        scenario=SimpleNamespace(
            initial=SimpleNamespace(m=60_000.0),
            aero=SimpleNamespace(S=100.0, Cl_max=2.0),
        ),
    )

    assert CONTROL_ANCHOR_STALL_MARGIN == pytest.approx(1.10)
    assert output_strategy(config).eligible_anchors(series, range(4)) == [2, 3]
    assert output_strategy(
        replace(config, prediction_output="state")
    ).eligible_anchors(series, range(4)) == [0, 1, 2, 3]


def test_training_cohort_floor_filters_only_the_supplied_train_roster():
    config = TSConfig(seq_len=3, training_cohort_min_future_s=60.0)
    short = SimpleNamespace(
        dataset_id="KAAA:SHORT",
        times=np.array([0.0, 2.0, 4.0]),
        supervision_times=np.array([0.0, 2.0, 4.0, 54.0]),
    )
    eligible = SimpleNamespace(
        dataset_id="KAAA:ELIGIBLE",
        times=np.array([0.0, 2.0, 4.0]),
        supervision_times=np.array([0.0, 2.0, 4.0, 64.0]),
    )

    retained, audit = train_module.filter_training_cohort(
        [short, eligible], config, verbose=False
    )

    assert retained == [eligible]
    assert audit == {
        "scope": "train only after by-flight split",
        "anchor": "fixed L-1",
        "minimum_future_s": 60.0,
        "input_flights": 2,
        "retained_flights": 1,
        "excluded_flights": 1,
        "excluded": [{
            "dataset_id": "KAAA:SHORT",
            "fixed_anchor_remaining_s": 50.0,
        }],
    }
