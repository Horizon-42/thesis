"""The validation replay: one forward per split, the common-grid truth cache, the fixed-anchor caches.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""


import numpy as np
import pytest
import torch

import ts_transformer.data.batch_contract as batch_contract
from ts_transformer.data.batch_contract import anchor_state, model_forward, unpack_batch
import ts_transformer.data.dataset as dataset_module
import ts_transformer.training.train as train_module
import ts_transformer.training.validation as validation
from ts_transformer.config import (
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.data.dataset import iter_batches
from ts_transformer.data.dataset import (
    FixedAnchorTrajectoryWindows,
    FlightEpochSampler,
    Normalizer,
    build_series,
)
from ts_transformer.data.splits import split_by_flight
from ts_transformer.data.fixed_dt_supervision import build_fixed_dt_supervision
from ts_transformer.backbone.adapters import build_model
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.training.objective import (
    loss_component_names,
    move_dynamics,
    move_fixed_dt_supervision,
    prediction_loss_components,
)
from ts_transformer.tests.support import fake_data_provenance
from ts_transformer.training.train import FIT_EVALUATION_NAME, evaluate_fit_splits, train

AIRPORT, RUNWAY = "KRDU", "05L"


def _series(n_flights=8, **config_overrides):
    """Synthetic KRDU arrivals, built into FlightSeries. Returns ``(series, config)``."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


def _two_pass_loss_components(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """The two-pass reference the shared validation forward is checked against.

    It re-runs the objective on its own, so `evaluate_validation_airport`'s single forward
    has something independent to be equal to. It lived in `validation.py` until the T3
    review pointed out it has no production caller — a test oracle that ships inside the
    module it is testing is one edit away from being the thing it checks.
    """
    names = loss_component_names(dataset.config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            (
                x,
                y,
                mask,
                final_time_s,
                flight_weights,
                dynamics,
                dense_supervision,
            ) = unpack_batch(raw_batch)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            dense_supervision = move_fixed_dt_supervision(dense_supervision, device)
            prediction = model_forward(model, x, dynamics, future=(y, final_time_s))
            components = prediction_loss_components(
                prediction,
                anchor_state(x, len(dataset.config.channels)),
                y,
                mask,
                final_time_s,
                flight_weights,
                dataset.config,
                dataset.normalizer,
                dynamics,
                dense_supervision,
            )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
    denominator = max(flight_weight_total, 1.0)
    return {name: value / denominator for name, value in component_totals.items()}


def test_common_grid_checkpoint_selection_still_validates_fixed_anchor():
    series, config = _series(
        n_flights=12,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    fit = train_module.fit_model(
        train_series, val_series, config, verbose=False
    )

    assert fit.best_validation_selection > 0.0
    assert fit.history[0].validation_selection_metric == (
        CHECKPOINT_SELECTION_COMMON_GRID_ADE
    )
    assert fit.history[0].validation_selection_value == pytest.approx(
        fit.best_validation_selection
    )
    assert set(fit.history[0].validation_selection_by_airport) == {AIRPORT}
    timing = fit.history[0].timing
    assert timing["epoch_total_s"] > 0.0
    assert timing["train_forward_s"] > 0.0
    assert timing["train_rollout_loss_s"] > 0.0
    assert timing["train_backward_step_s"] > 0.0
    assert timing["val_objective_s"] > 0.0
    assert timing["val_checkpoint_selection_s"] > 0.0
    assert timing["optimizer_updates_per_s"] > 0.0
    profile = fit.history[0].validation_profile_by_airport[AIRPORT]
    assert profile["flights"] == len(val_series)
    assert profile["query_points"] > 0
    assert sum(profile["duration_bucket_flights"].values()) == len(val_series)


def test_common_grid_checkpoint_selection_reuses_one_truth_cache(monkeypatch):
    series, config = _series(
        n_flights=12,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        validation_common_grid_points=7,
        epochs=2,
        patience=2,
        batch_size=32,
        d_model=16,
        d_ff=32,
        n_heads=4,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    cached_truth_ids: list[int] = []
    original = validation.fixed_anchor_common_grid_ade_metrics

    def record_cache(*args, **kwargs):
        cached_truth = kwargs.get("common_truth")
        assert cached_truth is not None
        cached_truth_ids.append(id(cached_truth))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        validation,
        "fixed_anchor_common_grid_ade_metrics",
        record_cache,
    )
    train_module.fit_model(train_series, val_series, config, verbose=False)

    assert len(cached_truth_ids) == config.epochs
    assert len(set(cached_truth_ids)) == 1


def test_shared_validation_forward_matches_two_pass_control_metrics(monkeypatch):
    series, config = _series(
        n_flights=4,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
        batch_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        device="cpu",
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    model = build_model(config).eval()
    legacy_components = _two_pass_loss_components(
        model, dataset, torch.device("cpu"), config.batch_size
    )
    legacy_common = validation.evaluate_fixed_anchor_common_grid(
        model, dataset, normalizer, config, torch.device("cpu")
    )

    calls = 0
    original_forward = validation.model_forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(validation, "model_forward", counted_forward)
    plan = validation.build_validation_batch_plan(dataset, config.batch_size)
    shared = validation.evaluate_validation_airport(
        model,
        plan,
        torch.device("cpu"),
        config=config,
    )
    shared_common = validation.evaluate_fixed_anchor_common_grid(
        model,
        dataset,
        normalizer,
        config,
        torch.device("cpu"),
        replay=shared.replay,
    )

    assert calls == len(plan.batches)
    assert shared.components == pytest.approx(legacy_components, rel=1e-6, abs=1e-7)
    for key in (
        "ade_m",
        "fde_m",
        "final_time_mae_s",
        "terminal_velocity_error_mps",
        "arc_length_geometry_loss",
    ):
        assert shared_common[key] == pytest.approx(
            legacy_common[key], rel=1e-6, abs=1e-6
        )


def test_control_validation_replay_uses_dense_dynamics_queries():
    series, config = _series(
        n_flights=1,
        prediction_output=PREDICTION_CONTROL,
        seq_len=8,
        n_segments=2,
        validation_common_grid_points=5,
        control_rollout_integrator_dt_s=0.5,
    )
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    raw_batch = next(dataset_module.iter_batches(dataset, 1, shuffle=False, seed=0))
    x, y, mask, final_time_s, _weights, dynamics, _dense = (
        batch_contract.unpack_batch(raw_batch)
    )
    assert dynamics is not None
    midpoint = 0.5 * (dynamics["control_lower"] + dynamics["control_upper"])
    prediction = ControlPrediction(
        controls=midpoint[:, None, :].expand(-1, 2, -1).contiguous(),
        segment_durations=torch.tensor([[0.75, 1.25]], dtype=torch.float32),
        final_time_s=torch.tensor([2.0], dtype=torch.float32),
    )

    replay = validation._prediction_batch_replay(
        prediction, x, y, mask, final_time_s, dynamics, dataset
    )

    assert replay.predicted.shape == (1, 5, config.enc_in)
    assert replay.segment_durations_s.shape == (1, 5)
    np.testing.assert_allclose(replay.segment_durations_s, 0.4)
    assert replay.predicted_time_s.tolist() == pytest.approx([2.0])


def test_fixed_anchor_cache_is_bitwise_identical_to_uncached_builders():
    series, config = _series(
        n_flights=3,
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        seq_len=8,
        n_segments=2,
    )
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    for index in range(len(dataset)):
        cached = dataset._sample_arrays(index)
        uncached = dataset_module.TrajectoryWindows._sample_arrays(dataset, index)
        for cached_array, uncached_array in zip(cached, uncached):
            assert np.array_equal(cached_array, uncached_array)
        cached_dynamics = dataset.context.row(index)
        uncached_dynamics = dataset.context._build_row(index)
        for key in cached_dynamics:
            assert np.array_equal(cached_dynamics[key], uncached_dynamics[key])

    indices = np.arange(len(dataset), dtype=np.int64)
    cached_dense = dataset.context.dense(indices)
    uncached_dense = build_fixed_dt_supervision(
        dataset.series,
        dataset.encoded,
        dataset.index,
        dt_s=config.dt_s,
    )
    for field in ("query_offsets_s", "states", "weights", "valid"):
        assert torch.equal(getattr(cached_dense, field), getattr(uncached_dense, field))


def test_fit_evaluation_reuses_one_prediction_pass_per_split(monkeypatch):
    series, config = _series(
        n_flights=12,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )
    train_series, val_series, _test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)
    model = build_model(config)
    calls = 0
    original_forward = validation.model_forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(validation, "model_forward", counted_forward)
    evaluate_fit_splits(
        model, train_series, val_series, normalizer, config, torch.device("cpu")
    )

    assert calls == 2


def test_training_saves_checkpoint_before_derived_fit_replay(tmp_path, monkeypatch):
    series, config = _series(
        n_flights=12,
        epochs=1,
        patience=1,
        batch_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        e_layers=1,
        seq_len=20,
        n_segments=8,
        device="cpu",
    )

    def fail_replay(*_args, **_kwargs):
        raise RuntimeError("derived report failed")

    monkeypatch.setattr(train_module, "evaluate_fit_splits", fail_replay)
    with pytest.raises(RuntimeError, match="derived report failed"):
        train(
            series,
            config,
            output_dir=tmp_path,
            data_provenance=fake_data_provenance(),
            verbose=False,
        )

    assert (tmp_path / "checkpoint.pt").is_file()
    assert not (tmp_path / FIT_EVALUATION_NAME).exists()


def test_flight_loss_weights_give_every_airport_equal_epoch_weight():
    series, config = _series(n_flights=8)
    series[0].scenario.source["arr_airport"] = "KAAA"
    for item in series[1:]:
        item.scenario.source["arr_airport"] = "KBBB"
    dataset = FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series))

    totals = {"KAAA": 0.0, "KBBB": 0.0}
    for index in FlightEpochSampler(dataset, seed=7):
        series_index, _anchor = dataset.index[index]
        totals[series[series_index].airport] += float(dataset[index][4])

    assert totals["KAAA"] == pytest.approx(totals["KBBB"])
    assert sum(totals.values()) == pytest.approx(len(series))
