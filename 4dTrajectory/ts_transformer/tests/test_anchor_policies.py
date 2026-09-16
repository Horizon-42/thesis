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


def test_the_run_anchor_floor_is_the_ablation_floor_carried_by_the_config():
    """`anchor_floor_index` (two-tier T0(c), 2026-09-16) is the common anchor the history
    ablation passes at call time, carried by the run itself: every lookback's window set is
    identical to the call-time floor's, anchor and targets alike."""
    series, base = _series(n_flights=2)
    normalizer = Normalizer.fit(series)
    floor = 89
    for seq_len in (30, 60, 90):
        by_config = FixedAnchorTrajectoryWindows(
            series, replace(base, seq_len=seq_len, anchor_floor_index=floor), normalizer
        )
        by_call = FixedAnchorTrajectoryWindows(
            series, replace(base, seq_len=seq_len), normalizer, minimum_anchor_index=floor
        )
        assert by_config.anchor == by_call.anchor == floor
        assert by_config.index == by_call.index
        for ours, theirs in zip(by_config.batch([0, 1])[:5], by_call.batch([0, 1])[:5]):
            assert torch.equal(ours, theirs)


def test_default_anchor_is_the_later_of_the_lookback_and_the_floor():
    from ts_transformer.config import default_anchor, lookback_anchor

    assert default_anchor(TSConfig(seq_len=60)) == lookback_anchor(TSConfig(seq_len=60)) == 59
    assert default_anchor(TSConfig(seq_len=60, anchor_floor_index=119)) == 119
    # a floor inside the lookback moves nothing
    assert default_anchor(TSConfig(seq_len=120, anchor_floor_index=100)) == 119
    with pytest.raises(ValueError, match="anchor_floor_index"):
        TSConfig(anchor_floor_index=-1)
    # every stored config predates the field and reads as L-1
    stored = TSConfig(seq_len=60).to_dict()
    del stored["anchor_floor_index"]
    assert default_anchor(TSConfig.from_dict(stored)) == 59


def test_a_floored_run_says_which_anchor_its_cohort_was_filtered_at():
    """The audit label names L-1 only where the anchor IS L-1: at L=90 a floor of 89 is L-1,
    at L=30 it is not."""
    series, base = _series(n_flights=2)
    _kept, at_l1 = train_module.filter_training_cohort(
        series, replace(base, seq_len=90, anchor_floor_index=89), verbose=False
    )
    _kept, floored = train_module.filter_training_cohort(
        series, replace(base, seq_len=30, anchor_floor_index=89), verbose=False
    )
    assert at_l1["anchor"] == "fixed L-1"
    assert floored["anchor"] == "fixed index 89"


def test_predict_defaults_to_the_floor():
    from ts_transformer.backbone.adapters import build_model
    from ts_transformer.inference.forecast import forecast_approaches

    series, base = _series(n_flights=2, device="cpu", d_model=16, d_ff=32, n_heads=4, e_layers=1)
    config = replace(base, seq_len=30, anchor_floor_index=89)
    normalizer = Normalizer.fit(series)
    model = build_model(config, normalizer)
    forecasts = forecast_approaches(model, series, config, normalizer, device=torch.device("cpu"))
    assert [forecast.anchor for forecast in forecasts] == [89, 89]
    assert all(forecast.times[0] > series[i].times[89] for i, forecast in enumerate(forecasts))


def test_a_floored_build_keeps_the_same_flights_for_every_lookback():
    """Review 2026-09-16: `build_series` dropped short tracks by the LOOKBACK (``seq_len + 1``
    samples), so under a common floor the L=120 arm lost flights the L=60 arm kept (11 of
    1382 on KRDU val). The rule is the fixed anchor's: one cohort for every L."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3)
    trimmed = dict(flights[0])
    t0 = float(trimmed["waypoints"][0][0])
    # 150 s of track: one window at L=30 (60 s), none at the floor 89 (178 s)
    trimmed["waypoints"] = [w for w in trimmed["waypoints"] if float(w[0]) - t0 <= 150.0]
    cohort = [trimmed, *flights[1:]]
    built = {}
    for seq_len in (30, 60, 90):
        config = TSConfig(seq_len=seq_len, anchor_floor_index=89)
        series, report = build_series(cohort, config, airport=AIRPORT)
        built[seq_len] = (
            [item.dataset_id for item in train_module.usable_series(series, config, verbose=False)],
            [item.dataset_id for item in series],
        )
    assert built[30] == built[60] == built[90]
    assert len(built[30][1]) == 3            # the trimmed flight is out at build, for every L
