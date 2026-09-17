"""The segment-plan output (two-tier v2 §4): the config contract, the truth's coarse plan
as labels, the segment features' runway-axis invariance, the head under both token axes,
the loss on the labels, the decode into rows, and one whole chain — train, checkpoint,
forecast, export, evaluation — on synthetic arrivals."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from evaluation.metrics import evaluate_batch
from evaluation.records import load_records
from ts_transformer.config import (
    CHECKPOINT_SELECTION_OBJECTIVE,
    PLAN_WAYPOINT_SEGMENT_S,
    PREDICTION_SEGMENT_PLAN,
    PREDICTION_STATE,
    SEGMENT_PLAN_ATTENTION_CHANNELS,
    SEGMENT_PLAN_ATTENTION_SEGMENTS,
    SEGMENT_PLAN_ATTENTIONS,
    TSConfig,
    default_anchor,
    segment_plan_input_segments,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.channels import IDX, POSITION_IDX
from ts_transformer.data.dataset import Normalizer, build_series, training_window_class, truth_duration_s
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import forecast_approach
from ts_transformer.outputs import strategy
from ts_transformer.outputs.segment_plan.decode import (
    ARRIVAL_PROBABILITY,
    MIN_ARRIVAL_S,
    decode_plan,
    plan_rows,
    replay_rows,
)
from ts_transformer.outputs.segment_plan.features import SEGMENT_FEATURES, segment_features
from ts_transformer.outputs.segment_plan.labels import (
    CONTEXT_ARRIVAL_FRACTION,
    CONTEXT_ARRIVAL_VALID,
    CONTEXT_ARRIVED,
    CONTEXT_KEYS,
    CONTEXT_RUNWAY_HEADING,
    CONTEXT_TARGET_CHART,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    SEGMENT_TARGET_CONTRACT,
    chart_deltas,
    runway_deltas,
    runway_heading_rad,
    segment_labels,
)
from ts_transformer.outputs.segment_plan.model import (
    SEGMENT_PLAN_LOSS_COMPONENT_NAMES,
    SegmentPlanModel,
    SegmentPlanPrediction,
    probe_segment_plan_context,
    segment_plan_loss_components,
)
from ts_transformer.outputs.segment_plan.readout import plan_reading, summarize_readings
from ts_transformer.outputs.segment_plan.strategy import (
    ARRIVAL_PROBABILITY_KEY,
    ARRIVAL_SEGMENT_KEY,
    SegmentPlanContext,
    decode_series,
)
from ts_transformer.backbone.adapters import parameter_count
from ts_transformer.run_naming import run_display_name, run_slug
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance, terminal_contexts

# seq_len 31 = two coarse segments of 15 intervals + the anchor; dt 2 s
TINY = dict(seq_len=31, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, device="cpu",
            horizon_mode="normalized", checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE,
            epochs=1, patience=1, batch_size=8, segment_plan_segments=6)


def _config(**overrides) -> TSConfig:
    return TSConfig(prediction_output=PREDICTION_SEGMENT_PLAN, **{**TINY, **overrides})


def _cohort(n_flights: int = 4, **overrides):
    config = _config(**overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config


# ── the config contract ──────────────────────────────────────────────────────


def test_config_contract_and_naming():
    config = _config(segment_plan_attention=SEGMENT_PLAN_ATTENTION_SEGMENTS, segment_plan_arrival_loss_weight=2.0)
    assert config.output.segment_plan_segments == 6 and config.output.segment_plan_attention == "segments"
    assert segment_plan_input_segments(config) == 2
    name = run_display_name(config.to_dict())
    assert name.startswith("segment-plan · iTransformer · waypoints · segment-plan-v1(seg-arrival=2)"), name
    assert "attend=segments" in name and "M=6" in name
    assert run_slug(config.to_dict()) != run_slug(_config().to_dict())
    with pytest.raises(ValueError, match="belongs to the segment-plan output"):
        TSConfig(prediction_output=PREDICTION_STATE, segment_plan_segments=4, **{k: v for k, v in TINY.items() if k != "segment_plan_segments"})
    with pytest.raises(ValueError, match="whole number of 15-sample coarse segments"):
        _config(seq_len=30)
    with pytest.raises(ValueError, match="iTransformer"):
        _config(model="patchtst", patch_len=4, stride=2)
    with pytest.raises(ValueError, match="selected on its objective"):
        _config(checkpoint_selection_metric="fixed-anchor-common-grid-ade")
    with pytest.raises(ValueError, match="segment_plan_attention"):
        _config(segment_plan_attention="rows")
    with pytest.raises(ValueError, match="normalized horizon"):
        _config(horizon_mode="window")
    with pytest.raises(ValueError, match="use_norm"):
        _config(use_norm=True)
    stored = _config(segment_plan_segments=8).to_dict()
    assert TSConfig.from_dict(stored).segment_plan_segments == 8
    del stored["segment_plan_segments"]
    with pytest.raises(ValueError, match="segment_plan_segments"):
        TSConfig.from_dict(stored)
    assert strategy(config).target_contract(config) == SEGMENT_TARGET_CONTRACT
    assert strategy(config).loss_component_names(config) == SEGMENT_PLAN_LOSS_COMPONENT_NAMES
    assert set(SEGMENT_PLAN_ATTENTIONS) == {SEGMENT_PLAN_ATTENTION_CHANNELS, SEGMENT_PLAN_ATTENTION_SEGMENTS}


# ── the labels ───────────────────────────────────────────────────────────────


def test_runway_axes_round_trip():
    psi = 0.7
    de, dn = np.array([1000.0, -200.0]), np.array([300.0, 50.0])
    along, across = runway_deltas(de, dn, psi)
    back = chart_deltas(along, across, psi)
    np.testing.assert_allclose(back[0], de, atol=1e-9)
    np.testing.assert_allclose(back[1], dn, atol=1e-9)
    # a step straight along the course reads as pure along
    along, across = runway_deltas(np.array([math.cos(psi)]), np.array([math.sin(psi)]), psi)
    assert along[0] == pytest.approx(1.0) and across[0] == pytest.approx(0.0, abs=1e-12)


def test_labels_are_the_truth_at_the_segment_ends_in_runway_axes():
    series, config = _cohort()
    anchor = default_anchor(config)
    for item in series:
        row = segment_labels(item, anchor, config)
        assert set(row) == set(CONTEXT_KEYS)
        m = config.segment_plan_segments
        assert row[CONTEXT_TARGETS].shape == (m, 3) and row[CONTEXT_VALID].shape == (m,)
        psi = runway_heading_rad(item)
        assert float(row[CONTEXT_RUNWAY_HEADING]) == pytest.approx(psi)
        np.testing.assert_allclose(row[CONTEXT_TARGET_CHART], item.target_chart, atol=1e-6)
        arrival = truth_duration_s(item, anchor)
        ends = PLAN_WAYPOINT_SEGMENT_S * np.arange(1, m + 1)
        np.testing.assert_array_equal(row[CONTEXT_VALID], (ends <= arrival + 1e-9).astype(np.float32))
        np.testing.assert_array_equal(row[CONTEXT_ARRIVED], (ends >= arrival - 1e-9).astype(np.float32))
        anchor_position = item.values[anchor, list(POSITION_IDX)]
        for k in range(m):
            if not row[CONTEXT_VALID][k]:
                np.testing.assert_array_equal(row[CONTEXT_TARGETS][k], 0.0)
                continue
            truth = np.array([
                np.interp(item.times[anchor] + ends[k], item.supervision_times, item.supervision_values[:, c])
                for c in POSITION_IDX
            ]) - anchor_position
            de, dn = chart_deltas(row[CONTEXT_TARGETS][k, 0], row[CONTEXT_TARGETS][k, 1], psi)
            np.testing.assert_allclose([de, dn, row[CONTEXT_TARGETS][k, 2]], truth, atol=0.05)
        if arrival <= ends[-1]:
            k = int(np.argmax(row[CONTEXT_ARRIVED]))
            assert row[CONTEXT_ARRIVAL_VALID].sum() == 1.0 and row[CONTEXT_ARRIVAL_VALID][k] == 1.0
            expected = (arrival - (ends[k] - PLAN_WAYPOINT_SEGMENT_S)) / PLAN_WAYPOINT_SEGMENT_S
            assert row[CONTEXT_ARRIVAL_FRACTION][k] == pytest.approx(expected, abs=1e-5)
            assert 0.0 <= row[CONTEXT_ARRIVAL_FRACTION][k] <= 1.0
        else:
            assert row[CONTEXT_ARRIVAL_VALID].sum() == 0.0 and row[CONTEXT_ARRIVED].sum() == 0.0


def test_labels_see_the_arrival_on_a_short_remainder():
    series, config = _cohort()
    item = series[0]
    # an anchor 45 s before the truth's end: segment 1 reached (30 s), arrival in segment 2 at half
    end_index = int(np.searchsorted(item.times, item.supervision_times[-1] - 45.0))
    row = segment_labels(item, end_index, config)
    arrival = truth_duration_s(item, end_index)
    assert 30.0 < arrival <= 60.0
    assert row[CONTEXT_VALID][0] == 1.0 and row[CONTEXT_VALID][1] == 0.0
    assert row[CONTEXT_ARRIVED][0] == 0.0 and row[CONTEXT_ARRIVED][1:].all()
    assert row[CONTEXT_ARRIVAL_VALID][1] == 1.0
    assert row[CONTEXT_ARRIVAL_FRACTION][1] == pytest.approx((arrival - 30.0) / 30.0, abs=1e-5)


# ── the features ─────────────────────────────────────────────────────────────


def _window(batch: int = 3, length: int = 31, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = np.arange(length) * 2.0
    rows = []
    for _ in range(batch):
        heading = rng.uniform(-math.pi, math.pi)
        turn = rng.uniform(-0.01, 0.01)
        speed = rng.uniform(60.0, 90.0)
        psi = heading + turn * t
        e = np.cumsum(speed * np.cos(psi)) * 2.0 + rng.uniform(-20000, 20000)
        n = np.cumsum(speed * np.sin(psi)) * 2.0 + rng.uniform(-20000, 20000)
        u = 1500.0 - 3.0 * t
        rows.append(np.column_stack([e, n, u, speed * np.cos(psi), speed * np.sin(psi), np.full(length, -3.0)]))
    return torch.tensor(np.stack(rows), dtype=torch.float64)


ORIGIN = torch.zeros(3, 3, dtype=torch.float64)


def test_segment_features_shape_and_anchor_end():
    window = _window()
    psi = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    features = segment_features(window, psi, ORIGIN, segment_samples=15)
    assert features.shape == (3, 2, len(SEGMENT_FEATURES))
    end_along, end_across, end_up = (SEGMENT_FEATURES.index(n) for n in ("end_along", "end_across", "end_up"))
    # the LAST segment ends at the anchor: its end is the origin of the runway axes
    assert torch.allclose(features[:, -1, [end_along, end_across, end_up]], torch.zeros(3, 3, dtype=torch.float64), atol=1e-9)
    start_along = SEGMENT_FEATURES.index("start_along")
    # the second segment starts where the first ends
    assert torch.allclose(features[:, 1, start_along], features[:, 0, end_along])
    with pytest.raises(ValueError, match="holds no"):
        segment_features(window[:, :10], psi, ORIGIN, segment_samples=15)


def test_segment_features_read_the_course_and_the_threshold_with_the_right_sign():
    # a window flying straight along the course at 80 m/s toward a threshold 20 km ahead
    psi = torch.tensor([0.7], dtype=torch.float64)
    t = torch.arange(31, dtype=torch.float64) * 2.0
    e = -20_000.0 * math.cos(0.7) + 80.0 * t * math.cos(0.7)
    n = -20_000.0 * math.sin(0.7) + 80.0 * t * math.sin(0.7)
    window = torch.stack([e, n, 1500.0 - 3.0 * t, torch.full_like(t, 80.0 * math.cos(0.7)),
                          torch.full_like(t, 80.0 * math.sin(0.7)), torch.full_like(t, -3.0)], dim=1)[None]
    features = segment_features(window, psi, torch.zeros(1, 3, dtype=torch.float64), segment_samples=15)[0, -1]
    name = SEGMENT_FEATURES.index
    assert features[name("direction_cos")] == pytest.approx(1.0, abs=1e-9)
    assert features[name("direction_sin")] == pytest.approx(0.0, abs=1e-9)
    assert features[name("start_along")] < 0 and abs(float(features[name("start_across")])) < 1e-9
    # the threshold is ahead, on the course: bearing cos +1, distance the anchor's own
    assert features[name("threshold_bearing_cos")] == pytest.approx(1.0, abs=1e-9)
    assert features[name("threshold_bearing_sin")] == pytest.approx(0.0, abs=1e-9)
    assert features[name("anchor_distance")] == pytest.approx((20_000.0 - 80.0 * 60.0) / 10_000.0, abs=1e-9)
    # ...and moving the threshold moves the context, not the segment geometry
    moved = segment_features(window, psi, torch.tensor([[500.0, 500.0, 0.0]], dtype=torch.float64), segment_samples=15)[0, -1]
    assert not torch.isclose(moved[name("anchor_distance")], features[name("anchor_distance")])
    assert torch.allclose(moved[: name("anchor_distance")], features[: name("anchor_distance")])


def test_segment_features_are_invariant_to_rotating_the_chart_with_the_course():
    window = _window()
    psi = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    base = segment_features(window, psi, ORIGIN, segment_samples=15)
    theta = 0.9
    c, s = math.cos(theta), math.sin(theta)
    rotated = window.clone()
    for a, b in ((IDX["e"], IDX["n"]), (IDX["edot"], IDX["ndot"])):
        x, y = window[..., a], window[..., b]
        rotated[..., a], rotated[..., b] = x * c - y * s, x * s + y * c
    turned = segment_features(rotated, psi + theta, ORIGIN, segment_samples=15)
    keep = [i for i, name in enumerate(SEGMENT_FEATURES) if name not in ("course_cos", "course_sin")]
    assert torch.allclose(base[..., keep], turned[..., keep], atol=1e-6)
    # the course itself is what moved
    assert not torch.allclose(base[..., SEGMENT_FEATURES.index("course_cos")], turned[..., SEGMENT_FEATURES.index("course_cos")])


# ── the head and its loss ────────────────────────────────────────────────────


def _batch(config: TSConfig, series):
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    windows = training_window_class(config)(series, config, normalizer)
    assert isinstance(windows.context, SegmentPlanContext)
    batch = unpack_batch(windows.batch(list(range(len(windows.index)))))
    return normalizer, windows, batch


@pytest.mark.parametrize("attention", SEGMENT_PLAN_ATTENTIONS)
def test_model_forward_shapes_and_the_probe_context(attention):
    series, config = _cohort(segment_plan_attention=attention)
    normalizer, windows, (x, y, mask, final_time_s, flight_weights, context, _dense) = _batch(config, series)
    model = strategy(config).build_model(config, normalizer)
    assert isinstance(model, SegmentPlanModel)
    prediction = model(x, context)
    m = config.segment_plan_segments
    assert prediction.positions.shape == (len(series), m, 3)
    assert prediction.arrived_logits.shape == (len(series), m) and prediction.fraction_logits.shape == (len(series), m)
    probe = probe_segment_plan_context(2, torch.device("cpu"), config)
    assert set(probe) == set(context) == set(CONTEXT_KEYS)
    assert all(probe[key].shape[1:] == context[key].shape[1:] for key in probe)
    with pytest.raises(ValueError, match="runway course"):
        model(x, None)
    # the model is blind to the truth in the context: every label key randomised, same output
    # (eval mode: dropout would otherwise differ between two forwards)
    blind = {key: (torch.rand_like(value) if key not in (CONTEXT_RUNWAY_HEADING, CONTEXT_TARGET_CHART) else value)
             for key, value in context.items()}
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x, blind).positions, model(x, context).positions)
    model.train()
    loss = segment_plan_loss_components(prediction, flight_weights, config, context)
    assert set(loss.tensors()) == set(SEGMENT_PLAN_LOSS_COMPONENT_NAMES)
    assert float(loss.kinematic) == 0.0 and torch.isfinite(loss.total)
    loss.total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_the_loss_is_zero_on_the_labels_and_blind_to_unreached_segments():
    series, config = _cohort()
    _normalizer, _windows, (x, y, mask, final_time_s, flight_weights, context, _dense) = _batch(config, series)
    big = 30.0
    arrived_logits = (context[CONTEXT_ARRIVED] * 2.0 - 1.0) * big
    fraction = context[CONTEXT_ARRIVAL_FRACTION].clamp(1e-4, 1 - 1e-4)
    exact = SegmentPlanPrediction(
        positions=context[CONTEXT_TARGETS].clone(), arrived_logits=arrived_logits,
        fraction_logits=torch.log(fraction / (1 - fraction)),
    )
    loss = segment_plan_loss_components(exact, flight_weights, config, context)
    assert float(loss.state) == pytest.approx(0.0, abs=1e-6)
    assert float(loss.final_time) == pytest.approx(0.0, abs=1e-3)
    assert float(loss.terminal) == pytest.approx(0.0, abs=1e-6)
    # a wild position in a segment the truth never reaches costs nothing
    unreached = torch.nonzero(context[CONTEXT_VALID] == 0.0)
    assert len(unreached), "the synthetic cohort has a flight with fewer than M segments of truth"
    b, k = (int(v) for v in unreached[0])
    perturbed = exact.positions.clone()
    perturbed[b, k] += 50_000.0
    moved = segment_plan_loss_components(SegmentPlanPrediction(perturbed, exact.arrived_logits, exact.fraction_logits), flight_weights, config, context)
    assert float(moved.state) == pytest.approx(0.0, abs=1e-6)
    # ...and the same wild position in a reached segment does
    b, k = (int(v) for v in torch.nonzero(context[CONTEXT_VALID] == 1.0)[0])
    perturbed = exact.positions.clone()
    perturbed[b, k] += 50_000.0
    moved = segment_plan_loss_components(SegmentPlanPrediction(perturbed, exact.arrived_logits, exact.fraction_logits), flight_weights, config, context)
    assert float(moved.state) > 0.1
    # the weights scale their groups
    doubled = segment_plan_loss_components(
        SegmentPlanPrediction(perturbed, exact.arrived_logits, exact.fraction_logits), flight_weights,
        _config(segment_plan_position_loss_weight=2.0), context,
    )
    assert float(doubled.state) == pytest.approx(2.0 * float(moved.state), rel=1e-6)
    with pytest.raises(ValueError, match="label context"):
        segment_plan_loss_components(exact, flight_weights, config, None)


def test_the_two_token_axes_share_one_head():
    series, config = _cohort(segment_plan_attention=SEGMENT_PLAN_ATTENTION_CHANNELS)
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    channels = strategy(config).build_model(config, normalizer)
    other = _config(segment_plan_attention=SEGMENT_PLAN_ATTENTION_SEGMENTS)
    segments = strategy(other).build_model(other, normalizer)
    head = lambda model: sum(p.numel() for p in model.head.parameters()) + sum(p.numel() for p in model.pool.parameters())
    assert head(channels) == head(segments)
    # the arms differ ONLY in the embedding's input width (K vs F series) and the segment positions
    k, f, d = 2, len(SEGMENT_FEATURES), config.d_model
    assert parameter_count(segments) - parameter_count(channels) == (f - k) * d + k * d


# ── the decode ───────────────────────────────────────────────────────────────


def test_decode_lays_the_waypoints_then_the_threshold_at_the_arrival_time():
    psi = 0.5
    positions = np.array([[2000.0, 100.0, -100.0], [4000.0, 50.0, -200.0], [6000.0, 0.0, -300.0]])
    anchor = np.array([-7000.0, 300.0, 400.0])
    plan = decode_plan(positions, np.array([0.1, 0.6, 0.9]), np.array([0.0, 0.5, 0.0]), anchor, psi)
    assert plan.arrives and plan.arrival_segment == 1 and plan.arrival_probability == pytest.approx(0.6)
    assert plan.arrival_time_s == pytest.approx(45.0)
    np.testing.assert_allclose(plan.times_s, [30.0, 60.0, 90.0])
    de, dn = chart_deltas(positions[:, 0], positions[:, 1], psi)
    np.testing.assert_allclose(plan.waypoints[:, 0], anchor[0] + de)
    np.testing.assert_allclose(plan.waypoints[:, 1], anchor[1] + dn)
    np.testing.assert_allclose(plan.waypoints[:, 2], anchor[2] + positions[:, 2])
    threshold = np.zeros(3)
    offsets, rows = plan_rows(plan, threshold)
    np.testing.assert_allclose(offsets, [30.0, 45.0])
    np.testing.assert_allclose(rows[0], plan.waypoints[0]) and np.testing.assert_allclose(rows[1], threshold)
    durations, padded, final, rows = replay_rows(plan, threshold, 3)
    assert final == pytest.approx(45.0) and rows == 2
    np.testing.assert_allclose(durations, [30.0, 15.0, 30.0])
    np.testing.assert_allclose(padded[2], threshold)
    assert np.all(durations > 0)
    # an arrival fraction of zero still leaves a positive segment
    early = decode_plan(positions, np.array([0.9, 0.9, 0.9]), np.zeros(3), anchor, psi)
    assert early.arrival_segment == 0 and early.arrival_time_s == pytest.approx(MIN_ARRIVAL_S)
    # no segment clears the threshold: every waypoint, capped
    never = decode_plan(positions, np.array([0.1, 0.2, 0.4]), np.zeros(3), anchor, psi, threshold=ARRIVAL_PROBABILITY)
    assert not never.arrives and never.arrival_probability == pytest.approx(0.4)
    offsets, rows = plan_rows(never, threshold)
    np.testing.assert_allclose(offsets, [30.0, 60.0, 90.0]) and np.testing.assert_allclose(rows, never.waypoints)
    durations, padded, final, rows = replay_rows(never, threshold, 3)
    np.testing.assert_allclose(durations, [30.0, 30.0, 30.0])
    assert final == pytest.approx(90.0) and rows == 3


def test_the_readout_scores_the_plan_inside_its_span():
    series, config = _cohort()
    item = series[0]
    anchor = default_anchor(config)
    truth_s = truth_duration_s(item, anchor)
    span_s = config.segment_plan_segments * PLAN_WAYPOINT_SEGMENT_S
    # the truth's own plan: zero waypoint error, zero covered ADE, the arrival on time
    row = segment_labels(item, anchor, config)
    exact = decode_plan(row[CONTEXT_TARGETS], row[CONTEXT_ARRIVED], row[CONTEXT_ARRIVAL_FRACTION],
                        item.values[anchor, list(POSITION_IDX)], runway_heading_rad(item))
    reading = plan_reading(exact, item, anchor, span_s)
    reached = int(row[CONTEXT_VALID].sum())
    assert np.all(np.isfinite(reading["waypoint_error_m"][:reached])) and np.all(np.isnan(reading["waypoint_error_m"][reached:]))
    assert np.nanmax(reading["waypoint_error_m"]) < 0.1
    assert reading["covered_s"] == pytest.approx(min(truth_s, span_s))
    assert reading["truth_arrives"] == (truth_s <= span_s)
    if reading["truth_arrives"]:
        assert reading["plan_arrives"] and abs(reading["arrival_time_error_s"]) < 1e-3
        assert reading["covered_ade_m"] < 1.0   # linear between waypoints against the truth's own rows
    block = summarize_readings([reading], config)
    assert block["flights"] == 1 and block["span_s"] == span_s
    assert block["segment_error_m"][0]["n"] == 1 and block["segment_error_m"][-1]["n"] == (1 if reached == config.segment_plan_segments else 0)


# ── the whole chain ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("attention", SEGMENT_PLAN_ATTENTIONS)
def test_train_checkpoint_forecast_export_and_evaluate(tmp_path, attention):
    series, config = _cohort(n_flights=6, segment_plan_attention=attention, random_train_anchor=True,
                             random_train_anchor_sampling="remaining-path-uniform")
    result = train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    assert result["history"][0]["val_loss"] > 0
    block = result["history"][0]["segment_plan_validation"]
    assert block["flights"] >= 1
    assert set(block["arrival_confusion"]) == {"both", "plan_only", "truth_only", "neither"}
    assert sum(block["arrival_confusion"].values()) == block["flights"]
    assert len(block["segment_error_m"]) == config.segment_plan_segments and block["covered_ade_m"] >= 0.0
    model, loaded, normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.prediction_output == PREDICTION_SEGMENT_PLAN and loaded.segment_plan_attention == attention
    np.testing.assert_allclose(model.channel_mean.numpy(), normalizer.mean, rtol=1e-6)
    np.testing.assert_allclose(model.channel_std.numpy(), normalizer.std, rtol=1e-6)
    assert payload["target_contract"] == SEGMENT_TARGET_CONTRACT
    metadata = json.loads((tmp_path / "run" / "checkpoint_metadata.json").read_text())
    assert metadata["segment_plan"]["segments"] == config.segment_plan_segments
    assert metadata["segment_plan"]["features"] == list(SEGMENT_FEATURES)
    records, metrics = [], []
    out = tmp_path / "pred"
    for index, item in enumerate(series):
        forecast = forecast_approach(model, item, loaded, normalizer, device=torch.device("cpu"))
        offsets = forecast.times - item.times[forecast.anchor]
        assert len(offsets) <= config.segment_plan_segments and np.all(np.diff(np.concatenate([[0.0], offsets])) > 0)
        assert forecast.values.shape[1] == len(config.channels) and np.isfinite(forecast.values).all()
        assert forecast.horizon_capped != forecast.truncated_at_threshold
        if forecast.truncated_at_threshold:
            np.testing.assert_allclose(forecast.values[-1, list(POSITION_IDX)], item.target_chart, atol=1e-6)
            assert forecast.final_time_s == pytest.approx(offsets[-1])
        else:
            assert len(offsets) == config.segment_plan_segments
        fields = strategy(loaded).record_fields(forecast)
        assert set(fields) == {ARRIVAL_SEGMENT_KEY, ARRIVAL_PROBABILITY_KEY}
        assert (fields[ARRIVAL_SEGMENT_KEY] is None) == forecast.horizon_capped
        assert forecast.command_hook_diagnostics is None
        records.append(build_prediction_record(item, forecast, index=index, model_name=loaded.model, horizon_mode=loaded.horizon_mode))
        metrics.append(observed_series_metrics(item, forecast))
    write_batch(records, output_dir=out, config_dict=loaded.to_dict(), flight_metrics=metrics)
    report = evaluate_batch(load_records(out), contexts=terminal_contexts())
    assert report["total"] == len(series) and report["solved"] == len(series)
    # a plan that arrives in its FIRST segment is a one-row forecast, and the record still stands
    item = series[0]
    late = int(np.searchsorted(item.times, item.supervision_times[-1] - 20.0))
    forecast = forecast_approach(model, item, loaded, normalizer, anchor=late, device=torch.device("cpu"))
    assert len(forecast.times) >= 1
    record = build_prediction_record(item, forecast, index=0, model_name=loaded.model, horizon_mode=loaded.horizon_mode)
    assert record.eval_record["final_time_s"] == pytest.approx(forecast.final_time_s, abs=1e-6)
    assert record.source[ARRIVAL_SEGMENT_KEY] == forecast.segment_plan_arrival_segment
    assert np.isfinite(observed_series_metrics(item, forecast)["ade_m"])


def test_the_epoch_line_prints_with_no_arrival_inside_the_span():
    """The L2 campaign's first arm died at epoch 1 formatting a None: no flight yet had both a plan
    arrival and a truth arrival inside the span. The line prints a dash there."""
    from ts_transformer.training.train import segment_plan_epoch_line

    block = {
        "flights": 3, "span_s": 300.0, "covered_ade_m": 512.3, "covered_s_mean": 240.0,
        "segment_error_m": [{"end_s": 30.0, "mean": None, "n": 0}] + [{"end_s": 30.0 * k, "mean": 100.0, "n": 3} for k in range(2, 11)],
        "truth_arrives_share": 0.0, "plan_arrives_share": 0.3,
        "arrival_confusion": {"both": 0, "plan_only": 1, "truth_only": 0, "neither": 2},
        "arrival_time_error_s": {"mean_abs": None, "mean_signed": None, "n": 0},
    }
    line = segment_plan_epoch_line(block)
    assert "covered-ADE=512.3m" in line and "e(30)=—m" in line and "|dT|=—s n=0" in line
    block["arrival_time_error_s"] = {"mean_abs": 12.25, "mean_signed": -3.0, "n": 2}
    assert "|dT|=12.2s n=2" in segment_plan_epoch_line(block)
