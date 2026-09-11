"""Rolled windows (design v5.2): the truth's queue labels a lockstep flight's states as the
policy flies them, the table round-trips and covers, the training draw takes a rolled
window at the share, and one training run records the share and the val readout."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.config import (
    CHECKPOINT_SELECTION_OBJECTIVE,
    PREDICTION_PLAN,
    RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import Normalizer, build_series, training_window_class
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.data.target_conditioning import conditioned_history
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import behind_on_final, lockstep_states, truth_lockstep_policy
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_ROLLED,
    CONTEXT_TARGETS,
    PLAN_TARGET_CONTRACT,
    TARGETS,
    TruthExpert,
    operating_from_labels,
    targets_at,
    targets_from_labels,
)
from ts_transformer.outputs.plan.rolled import (
    METADATA_KEY,
    POLICY_TRUTH,
    RolledDraw,
    load_rolled_windows,
    record_lockstep,
    rolled_table_header,
    write_rolled_windows,
)
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.outputs.plan.strategy import PlanContext
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance

pytestmark = pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)

TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=2.0,
            device="cpu", horizon_mode="normalized", checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE,
            epochs=1, patience=1, batch_size=8)


def _plan_config(**overrides) -> TSConfig:
    return TSConfig(prediction_output=PREDICTION_PLAN, **{**TINY, **overrides})


def _cohort(n_flights: int = 4, **arrivals):
    config = _plan_config()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3, **arrivals)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    return series, config, runway_skeleton(series[0])


def _rolled_table(tmp_path, series, config, skeleton, *, policy_orders=None):
    """The truth-policy table over ``series`` from L-1; ``policy_orders`` collects what the
    policy answered per (flight, step) when given."""
    anchor = default_anchor(config)
    labels = [extract_plan(item, anchor, skeleton) for item in series]
    horizons = [lab.T_s + 30.0 for lab in labels]
    states = lockstep_states(series, [anchor] * len(series), [skeleton] * len(series), [lab.remaining_path_at_anchor_m for lab in labels])
    policy = truth_lockstep_policy(states, labels, horizons)
    if policy_orders is not None:
        inner = policy

        def policy(active):
            orders = inner(active)
            for state, order in zip(active, orders, strict=True):
                instruction = order.instruction
                if instruction is not None and behind_on_final(state, instruction):
                    instruction = None
                policy_orders[(state.series.dataset_id, state.steps)] = instruction
            return orders
    flights, samples = record_lockstep(states, labels, config, policy=policy, time_caps_s=horizons)
    header = rolled_table_header(
        config, policy=POLICY_TRUTH, lockstep_s=30.0, airports=[AIRPORT], splits={"train": len(series)},
        checkpoint={"label": "test", "path": "none"}, generated_at="now", wall_s=0.0,
    )
    return write_rolled_windows(tmp_path / "rolled.npz", samples, header), flights, samples


def test_the_labels_at_a_flown_state_are_the_instruction_the_lockstep_flies(tmp_path):
    """Under the truth policy the recorded target at every step is the instruction the
    policy answered for that state — none where a fix behind an aircraft on the final is
    not flown — read about the aircraft's pose; step 0 is the observed anchor's own label."""
    series, config, skeleton = _cohort(3)
    answered: dict = {}
    table, flights, samples = _rolled_table(tmp_path, series, config, skeleton, policy_orders=answered)
    assert len(samples) == sum(len(flight.orders) for flight in flights) >= 6
    for sample in samples:
        expected = answered[(sample.dataset_id, sample.step)]
        assert sample.targets.next_is_join == (expected is None), (sample.dataset_id, sample.step)
        if expected is not None:
            assert sample.targets.values[TARGETS.index("next_speed_mps")] == pytest.approx(expected.speed_mps)
            assert sample.targets.values[TARGETS.index("next_remaining_m")] == pytest.approx(expected.remaining_m)
        assert sample.window.shape == (config.seq_len, len(config.channels)) and np.all(np.isfinite(sample.window))
    anchor = default_anchor(config)
    for item in series:
        first = next(s for s in samples if s.dataset_id == item.dataset_id and s.step == 0)
        lab = extract_plan(item, anchor, skeleton)
        observed = targets_from_labels(lab, item, anchor, skeleton)
        assert first.targets.next_is_join == observed.next_is_join
        assert np.allclose(first.targets.values * first.targets.valid, observed.values * observed.valid, rtol=1e-4, atol=1e-3)
        assert first.elapsed_s == 0.0
        # the expert at the anchor's own pose reads the extractors' values back exactly
        assert observed.values[TARGETS.index("T_s")] == pytest.approx(lab.T_s)
        assert observed.values[TARGETS.index("remaining_m")] == pytest.approx(lab.remaining_path_at_anchor_m)
    # a pose off the truth's path is labelled with the way back onto it (the nearest row of
    # the leg in force, plus the way there): within the displacement of the anchor's own
    # values, and never the anchor's values less anything flown
    lab = extract_plan(series[0], anchor, skeleton)
    expert = TruthExpert(lab, series[0], anchor, skeleton)
    row = series[0].values[anchor]
    e, n = float(row[IDX["e"]]), float(row[IDX["n"]])
    _instruction, operating = expert.order_at(e + 3_000.0, n, 0.0, 80.0)
    assert abs(operating.remaining_m - lab.remaining_path_at_anchor_m) <= 3_000.0 + 1e-6
    assert abs(operating.T_s - lab.T_s) <= 3_000.0 / 80.0 + 1e-6
    assert operating.remaining_m > 0.0 and operating.T_s > 0.0
    # the same target vector read with no fix ahead is `targets_at` about that pose
    again = targets_at(operating_from_labels(lab), None, 0.0, 0.0, skeleton)
    assert again.next_is_join and again.valid[TARGETS.index("next_ahead_m")] == 0.0


def test_the_table_round_trips_and_refuses_another_contract_or_cohort(tmp_path):
    series, config, skeleton = _cohort(3)
    table, _flights, samples = _rolled_table(tmp_path, series, config, skeleton)
    loaded = load_rolled_windows(table.path)
    assert loaded.samples == len(samples) and loaded.flights == 3 and loaded.sha256 == table.sha256
    assert loaded.header["target_contract"] == PLAN_TARGET_CONTRACT and loaded.header["policy"] == POLICY_TRUTH
    for item in series:
        rows = loaded.samples_for(item.dataset_id)
        assert list(loaded.step[rows]) == list(range(len(rows)))
        assert np.array_equal(loaded.windows[rows[0]], next(s.window for s in samples if s.dataset_id == item.dataset_id and s.step == 0))
    loaded.require_cover(series, config, what="the cohort")
    with pytest.raises(ValueError, match="seq_len"):
        loaded.require_cover(series, _plan_config(seq_len=6), what="the cohort")
    extra, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=5, seed=3), config, airport=AIRPORT)
    with pytest.raises(ValueError, match="cover 3 of 5"):
        loaded.require_cover(extra, config, what="the cohort")
    # a second table extended from the first carries both sample sets and says so
    extended = write_rolled_windows(tmp_path / "rolled2.npz", samples[:2], {**loaded.header, "policy": "model"}, extend=loaded)
    assert extended.samples == loaded.samples + 2 and extended.provenance["extended_from"]["sha256"] == loaded.sha256
    assert extended.header["policy"] == "model"


def test_the_rolled_draw_is_deterministic_and_lands_at_the_share(tmp_path):
    series, config, skeleton = _cohort(3)
    table, _flights, _samples = _rolled_table(tmp_path, series, config, skeleton)
    draw = RolledDraw(table, 0.5)
    seeds = range(400)
    taken = [draw.sample(series[0].dataset_id, seed) for seed in seeds]
    share = sum(k is not None for k in taken) / len(taken)
    assert 0.4 < share < 0.6
    assert taken == [draw.sample(series[0].dataset_id, seed) for seed in seeds]
    rows = set(table.samples_for(series[0].dataset_id).tolist())
    assert {k for k in taken if k is not None} <= rows and len({k for k in taken if k is not None}) > 1
    assert all(RolledDraw(table, 0.0).sample(series[0].dataset_id, seed) is None for seed in seeds)
    assert all(RolledDraw(table, 1.0).sample(series[0].dataset_id, seed) is not None for seed in seeds)
    with pytest.raises(KeyError):
        RolledDraw(table, 1.0).sample("not-a-flight", 3)


def test_a_training_batch_carries_the_rolled_window_at_the_share(tmp_path):
    series, config, skeleton = _cohort(3)
    table, _flights, _samples = _rolled_table(tmp_path, series, config, skeleton)
    config = _plan_config(
        random_train_anchor=True, random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
        plan_rolled_windows_path=str(table.path), plan_rolled_share=1.0,
    )
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    windows = training_window_class(config)(series, config, normalizer, training_input=table)
    assert isinstance(windows.context, PlanContext) and windows.context.rolled is table
    assert "rolled windows" in windows.context.summary
    indices = windows.epoch_indices(7)
    x, _y, mask, _final, _weights, context, _dense = unpack_batch(windows.batch(indices, epoch_seed=7))
    assert torch.all(context[CONTEXT_ROLLED] == 1.0) and torch.all(mask == 0.0)
    for row, index in enumerate(indices):
        s_idx = windows.index[int(index)][0]
        k = windows.context.draw.sample(series[s_idx].dataset_id, 7)
        expected = conditioned_history(normalizer.encode(table.windows[k].astype(np.float64)), windows.conditioning[s_idx])
        assert np.allclose(x[row].numpy(), expected.astype(np.float32), atol=1e-6)
        assert np.allclose(context[CONTEXT_TARGETS][row].numpy(), table.targets[k])
        assert float(context[CONTEXT_NEXT_IS_JOIN][row]) == float(table.next_is_join[k])
    # without the epoch's seed (the validation pass) every row is the observed anchor's
    x0, _y0, mask0, _f0, _w0, context0, _d0 = unpack_batch(windows.batch(indices))
    assert torch.all(context0[CONTEXT_ROLLED] == 0.0) and float(mask0.sum()) > 0.0


def test_train_on_rolled_windows_records_the_share_and_the_val_readout(tmp_path):
    series, config, skeleton = _cohort(8)
    table, _flights, _samples = _rolled_table(tmp_path, series, config, skeleton)
    config = _plan_config(
        random_train_anchor=True, random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
        plan_rolled_windows_path=str(table.path), plan_rolled_share=0.5, epochs=2, patience=2,
    )
    result = train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    assert isinstance(result, dict)
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    epoch = history["history"][0]
    assert 0.0 < epoch["plan_rolled_training"]["share"] < 1.0 and epoch["plan_rolled_training"]["samples"] >= 1
    readout = epoch["plan_rolled_validation"]
    assert readout["flights"] >= 1 and readout["samples"] >= readout["flights"] and np.isfinite(readout["loss"])
    assert set(readout["components"]) == {"state", "final_time", "kinematic", "terminal"}
    metadata = json.loads((tmp_path / "run" / "checkpoint_metadata.json").read_text())
    assert metadata[METADATA_KEY]["sha256"] == table.sha256 and metadata[METADATA_KEY]["policy"] == POLICY_TRUTH
    _model, loaded, _normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.plan_rolled_share == 0.5 and payload[METADATA_KEY]["samples"] == table.samples
