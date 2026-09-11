"""The plan output (design v5 step 3c): its config contract, the labels → targets → order
round trip, the loss on the labels, the drawn replay, and one whole chain — train,
checkpoint, the rolled forecast, export, evaluation — on synthetic arrivals."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from evaluation.metrics import evaluate_batch
from evaluation.records import load_records
from ts_transformer.config import (
    CHECKPOINT_SELECTION_OBJECTIVE,
    PREDICTION_PLAN,
    PREDICTION_STATE,
    RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
    TSConfig,
    default_anchor,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import build_series, training_window_class, Normalizer
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import forecast_approach
from ts_transformer.outputs import strategy
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import draw_order
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    INSTRUCTION,
    TARGETS,
    order_from_prediction,
    targets_from_labels,
    truth_instructions,
)
from ts_transformer.outputs.plan.model import PlanPrediction, plan_loss_components
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.outputs.plan.strategy import PLAN_TARGET_CONTRACT, PlanContext
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint, train
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance, terminal_contexts

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


def test_plan_config_contract():
    config = _plan_config(plan_instruction_loss_weight=2.0)
    assert config.output.plan_instruction_loss_weight == 2.0
    assert "plan · iTransformer · guidance · plan-v1(plan-instruction=2)" in run_display_name(config.to_dict())
    assert run_display_name(_plan_config().to_dict()).startswith("plan · iTransformer · guidance · plan-v1 ·")
    with pytest.raises(ValueError, match="belongs to the plan output"):
        TSConfig(prediction_output=PREDICTION_STATE, plan_operating_loss_weight=2.0, **TINY)
    with pytest.raises(ValueError, match="belongs to the closure output"):
        _plan_config(closure_slowness_knots=8)
    with pytest.raises(ValueError, match="normalized horizon"):
        _plan_config(horizon_mode="window")
    stored = _plan_config(plan_operating_loss_weight=3.0).to_dict()
    assert TSConfig.from_dict(stored).plan_operating_loss_weight == 3.0
    del stored["plan_operating_loss_weight"]
    with pytest.raises(ValueError, match="plan_operating_loss_weight"):
        TSConfig.from_dict(stored)
    assert strategy(config).target_contract(config) == PLAN_TARGET_CONTRACT


def test_the_targets_read_back_as_the_truths_own_instruction():
    series, config, skeleton = _cohort()
    anchor = default_anchor(config)
    seen_fix = False
    for item in series:
        labels = extract_plan(item, anchor, skeleton)
        targets = targets_from_labels(labels, item, anchor, skeleton)
        assert targets.values.shape == (len(TARGETS),) and targets.valid.shape == (len(TARGETS),)
        assert targets.valid[TARGETS.index("T_s")] == 1.0 and targets.valid[TARGETS.index("remaining_m")] == 1.0
        if labels.join_at_anchor:
            # on the final already: no fix ahead, the join is here (its own remaining path)
            assert targets.next_is_join
            assert all(targets.valid[TARGETS.index(name)] == 0.0 for name in INSTRUCTION)
            assert targets.valid[TARGETS.index("d_join_m")] == 1.0
            assert targets.values[TARGETS.index("d_join_m")] == pytest.approx(labels.remaining_path_at_anchor_m, rel=1e-5)
            continue
        instructions = truth_instructions(labels, skeleton)
        e, n = float(item.values[anchor, IDX["e"]]), float(item.values[anchor, IDX["n"]])
        order = order_from_prediction(targets.values, 0.0 if instructions else 1.0, e, n, skeleton)
        assert order.clamped == ()
        assert order.T_s == pytest.approx(labels.T_s, rel=1e-5) and order.d_join_m == pytest.approx(labels.d_join_m, rel=1e-5)
        if instructions:
            seen_fix = True
            assert targets.next_is_join is False
            first = instructions[0]
            assert order.instruction is not None
            assert order.instruction.fix_e == pytest.approx(first.fix_e, abs=0.5)
            assert order.instruction.fix_n == pytest.approx(first.fix_n, abs=0.5)
            wrapped = (order.instruction.heading_out_rad - first.heading_out_rad + math.pi) % (2 * math.pi) - math.pi
            assert abs(wrapped) < 1e-5
            assert order.instruction.speed_mps == pytest.approx(first.speed_mps, rel=1e-5)
            assert order.instruction.remaining_m == pytest.approx(first.remaining_m, rel=1e-5)
        else:
            assert targets.next_is_join and order.instruction is None
    assert seen_fix, "the synthetic offset entries carry a turn: at least one flight has a next fix"


def test_the_loss_is_zero_on_the_labels_and_masks_what_the_track_does_not_define():
    series, config, skeleton = _cohort()
    anchor = default_anchor(config)
    rows = [targets_from_labels(extract_plan(item, anchor, skeleton), item, anchor, skeleton).context(i)
            for i, item in enumerate(series)]
    context = {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}
    values = context[CONTEXT_TARGETS].clone().double()
    logit = torch.where(context[CONTEXT_NEXT_IS_JOIN] > 0.5, 30.0, -30.0).double()
    exact = plan_loss_components(PlanPrediction(values, logit), torch.ones(len(series), dtype=torch.float64), config, context)
    assert float(exact.total) < 1e-6
    # a residual on an UNDEFINED entry costs nothing; on a defined one it costs its scale
    perturbed = values.clone()
    perturbed[:, TARGETS.index("next_ahead_m")] += 10_000.0
    off = plan_loss_components(PlanPrediction(perturbed, logit), torch.ones(len(series), dtype=torch.float64), config, context)
    defined = context[CONTEXT_VALID][:, TARGETS.index("next_ahead_m")].sum()
    if float(defined) == 0.0:
        assert float(off.kinematic) < 1e-6
    else:
        # one instruction entry off by its scale, on every flight that defines it: 1/7 of the group
        assert float(off.kinematic) == pytest.approx(1.0 / len(INSTRUCTION), rel=1e-5)
    # the operating group and the arrival time are separate components
    perturbed = values.clone()
    perturbed[:, TARGETS.index("T_s")] += 100.0
    late = plan_loss_components(PlanPrediction(perturbed, logit), torch.ones(len(series), dtype=torch.float64), config, context)
    assert float(late.final_time) == pytest.approx(1.0, rel=1e-5) and float(late.state) < 1e-6
    # the no-fix flag is priced on every flight: confidently wrong costs ~30 nats each
    wrong = torch.where(context[CONTEXT_NEXT_IS_JOIN] > 0.5, -30.0, 30.0).double()
    flag = plan_loss_components(PlanPrediction(values, wrong), torch.ones(len(series), dtype=torch.float64), config, context)
    assert float(flag.terminal) == pytest.approx(30.0, rel=1e-3)


def test_the_plan_context_and_the_drawn_replay_cover_every_window():
    series, config, skeleton = _cohort()
    config = _plan_config(random_train_anchor=True, random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM)
    normalizer = Normalizer.fit(series)
    windows = training_window_class(config)(series, config, normalizer)
    assert isinstance(windows.context, PlanContext)
    batch = unpack_batch(windows.batch(list(range(min(4, len(windows))))))
    x, _y, _mask, _final, _weights, context, _dense = batch
    assert set(context) >= {CONTEXT_TARGETS, CONTEXT_VALID, CONTEXT_NEXT_IS_JOIN}
    assert context[CONTEXT_TARGETS].shape == (x.shape[0], len(TARGETS))
    # every order the head could emit draws a finite path that ends near the threshold
    anchor = default_anchor(config)
    item = series[0]
    labels = extract_plan(item, anchor, skeleton)
    targets = targets_from_labels(labels, item, anchor, skeleton)
    e, n, u = (float(item.values[anchor, IDX[c]]) for c in ("e", "n", "u"))
    order = order_from_prediction(targets.values, 0.0, e, n, skeleton, next_is_join=targets.next_is_join)
    heading = math.atan2(float(item.values[anchor, IDX["ndot"]]), float(item.values[anchor, IDX["edot"]]))
    speed = math.hypot(float(item.values[anchor, IDX["edot"]]), float(item.values[anchor, IDX["ndot"]]))
    drawn, durations, T = draw_order(order, e, n, u, heading, speed, skeleton, config.pred_len)
    assert drawn.shape == (config.pred_len, len(IDX)) and np.all(np.isfinite(drawn)) and T == pytest.approx(labels.T_s)
    assert durations.sum() == pytest.approx(T)
    last = drawn[-1]
    assert math.hypot(last[IDX["e"]] - skeleton.target_e, last[IDX["n"]] - skeleton.target_n) < 3_000.0
    # the height never climbs (to the instruction's height, to the capture height, then the
    # glidepath capture) and ends on the glidepath near the threshold
    heights = drawn[:, IDX["u"]]
    assert np.all(np.diff(heights) <= 1.0) and heights[-1] < 150.0


def test_train_checkpoint_forecast_export_and_evaluate_one_plan_run(tmp_path):
    series, _config, _skeleton = _cohort(n_flights=8)
    config = _plan_config(random_train_anchor=True, random_train_anchor_sampling=RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM)
    result = train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    assert (tmp_path / "run" / "checkpoint.pt").is_file() and isinstance(result, dict)
    model, loaded, normalizer, payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.prediction_output == PREDICTION_PLAN and payload["target_contract"] == PLAN_TARGET_CONTRACT
    metadata = json.loads((tmp_path / "run" / "checkpoint_metadata.json").read_text())
    assert metadata["plan"]["targets"] == list(TARGETS)
    records, metrics = [], []
    for index, item in enumerate(series[:3]):
        forecast = forecast_approach(model, item, loaded, normalizer, device=torch.device("cpu"))
        assert forecast.prediction_output == PREDICTION_PLAN and forecast.controls is not None
        assert forecast.values.shape[1] == len(IDX) and np.all(np.isfinite(forecast.values))
        orders = forecast.command_hook_diagnostics["planOrders"]
        assert len(orders) == forecast.command_hook_diagnostics["planLegs"] >= 1
        assert set(orders[0]) >= {"T_s", "d_join_m", "remaining_m", "instruction", "clamped", "leg"}
        assert forecast.predicted_final_time_s == pytest.approx(orders[0]["T_s"])
        records.append(build_prediction_record(item, forecast, index=index, model_name=loaded.model, horizon_mode=loaded.horizon_mode))
        metrics.append(observed_series_metrics(item, forecast))
    out = tmp_path / "pred"
    write_batch(records, output_dir=out, config_dict=loaded.to_dict(), flight_metrics=metrics)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mode"].endswith(":plan:test") and all(row["ade_m"] is not None for row in summary["results"])
    states = json.loads((out / summary["results"][0]["states_file"]).read_text())
    assert states["source"]["predictionOutput"] == PREDICTION_PLAN and states["source"]["planRouteKind"] == "rolled"
    assert states["source"]["commandHookDiagnostics"]["planOrders"][0]["T_s"] > 0.0
    report = evaluate_batch(load_records(out), contexts=terminal_contexts())
    assert report["total"] == 3 and report["solved"] == 3
