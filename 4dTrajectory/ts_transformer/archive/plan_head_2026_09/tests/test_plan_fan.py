"""The fan over the next fix (design v5.3 §9 step 3(g)): a K-component mixture over the
instruction group whose top-weight component IS the point prediction, its negative
log-likelihood as the `kinematic` component, and a fan member flown one step deep from
one component."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT
from ts_transformer.config import CHECKPOINT_SELECTION_OBJECTIVE, PREDICTION_PLAN, PREDICTION_STATE, TSConfig, default_anchor
from ts_transformer.data.approach_difficulty import STRATUM_ESTABLISHED, STRATUM_VECTORED
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_ROLLED,
    CONTEXT_SERIES,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    INSTRUCTION,
    OPERATING,
    SCALE_VECTOR,
    SCALES,
    TARGETS,
)
from ts_transformer.outputs.plan.model import (
    FAN_INITIAL_LOG_SIGMA,
    FAN_LOG_SIGMA_MIN,
    PlanOutputModel,
    PlanPrediction,
    _initial_bias,
    _inverse_softplus,
    decode_raw,
    fan_log_sigma,
    fan_rows,
    head_width,
    mixture_nll,
    plan_loss_components,
)
from ts_transformer.backbone.adapters import build_state_forecaster
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.outputs.plan.strategy import rolled_predictions_lockstep
from ts_transformer.experiments.plan_fan_readout import COVERAGE_SIGMAS, displaced, format_table, single_step, summarize
from ts_transformer.outputs.plan.labels import Instruction, PlanOrder
from ts_transformer.run_naming import run_display_name
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance
from ts_transformer.training.train import load_checkpoint, train

TINY = dict(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, final_time_scale_s=2.0,
            device="cpu", horizon_mode="normalized", checkpoint_selection_metric=CHECKPOINT_SELECTION_OBJECTIVE,
            epochs=1, patience=1, batch_size=8)


def _plan_config(**overrides) -> TSConfig:
    return TSConfig(prediction_output=PREDICTION_PLAN, **{**TINY, **overrides})


def test_the_fan_head_is_a_mixture_whose_top_component_is_the_point_prediction():
    with pytest.raises(ValueError, match="plan_fan_components"):
        _plan_config(plan_fan_components=1)
    config = _plan_config(plan_fan_components=3)
    assert "plan-v1(fan=3)" in run_display_name(config.to_dict())
    assert head_width(0) == len(TARGETS) + 1
    assert head_width(3) == len(OPERATING) + 6 * len(INSTRUCTION) + 3 + 1
    raw = torch.randn(5, head_width(3), dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    prediction = decode_raw(raw, 3)
    assert prediction.values.shape == (5, len(TARGETS)) and prediction.fan_values.shape == (5, 3, len(INSTRUCTION))
    full, weights, sigma = fan_rows(prediction)
    assert full.shape == (5, 3, len(TARGETS)) and weights.shape == (5, 3) and sigma.shape == (5, 3, len(INSTRUCTION))
    assert np.allclose(weights.sum(axis=1), 1.0) and np.all(sigma > 0.0)
    top = weights.argmax(axis=1)
    assert np.allclose(full[np.arange(5), top], prediction.values.numpy())
    operating = [TARGETS.index(name) for name in OPERATING]
    assert np.allclose(full[:, :, operating], np.repeat(prediction.values.numpy()[:, None, operating], 3, axis=1))
    i_cos, i_sin = INSTRUCTION.index("next_heading_cos"), INSTRUCTION.index("next_heading_sin")
    assert np.allclose(np.hypot(prediction.fan_values[..., i_cos].numpy(), prediction.fan_values[..., i_sin].numpy()), 1.0, atol=1e-3)
    # the point head's layout and decode are untouched
    point = decode_raw(raw[:, :len(TARGETS) + 1], 0)
    assert point.fan_logits is None and point.values.shape == (5, len(TARGETS))
    with pytest.raises(ValueError, match="no fan"):
        fan_rows(point)
    # the σ parameterisation: bounded below, never clamped above, the init inverted
    assert float(fan_log_sigma(torch.tensor(-50.0))) == pytest.approx(FAN_LOG_SIGMA_MIN, abs=1e-6)
    assert float(fan_log_sigma(torch.tensor(10.0))) == pytest.approx(FAN_LOG_SIGMA_MIN + 10.0, abs=1e-3)
    wide = decode_raw(torch.full((1, head_width(2)), 20.0), 2)
    _full, _w, sigma_wide = fan_rows(wide)
    assert sigma_wide[0, 0, INSTRUCTION.index("next_ahead_m")] == pytest.approx(
        math.exp(FAN_LOG_SIGMA_MIN + 20.0) * SCALES["next_ahead_m"], rel=1e-3,
    )


def test_the_point_heads_layout_and_start_are_unchanged_and_the_field_is_owned_by_the_plan():
    """A stored plan checkpoint (K = 0) keeps its last layer: 15 outputs, the biases of
    the old inline formula; a config without the field loads as 0; another output refuses it."""
    config = _plan_config()
    model = PlanOutputModel(config, build_state_forecaster(config))
    last = model.head[-1]
    bias = last.bias.detach()
    assert tuple(last.weight.shape) == (len(TARGETS) + 1, config.d_model)
    assert float(bias[TARGETS.index("T_s")]) == pytest.approx(_inverse_softplus(300.0 / 100.0))
    assert float(bias[TARGETS.index("next_across_m")]) == 0.0 and float(bias[-1]) == 0.0
    assert float(bias[TARGETS.index("next_heading_cos")]) == 1.0
    assert _initial_bias("h_capture_m") == pytest.approx(600.0 / 500.0)
    stored = config.to_dict()
    del stored["plan_fan_components"]
    assert TSConfig.from_dict(stored).plan_fan_components == 0
    with pytest.raises(ValueError, match="belongs to the plan output"):
        TSConfig(prediction_output=PREDICTION_STATE, plan_fan_components=2, **TINY)
    fan = PlanOutputModel(_plan_config(plan_fan_components=3), build_state_forecaster(config))
    sigma_start = len(OPERATING) + 3 * len(INSTRUCTION)
    fan_bias = fan.head[-1].bias.detach()
    assert float(fan_bias[sigma_start]) == pytest.approx(_inverse_softplus(FAN_INITIAL_LOG_SIGMA - FAN_LOG_SIGMA_MIN))
    across = [float(fan_bias[len(OPERATING) + c * len(INSTRUCTION) + INSTRUCTION.index("next_across_m")]) for c in range(3)]
    assert across[0] < across[1] < across[2] and across[1] == pytest.approx(0.0)


def test_the_mixture_nll_rewards_a_component_on_the_target_and_masks_the_undefined():
    batch, k, p = 2, 2, len(INSTRUCTION)
    scale = torch.as_tensor(SCALE_VECTOR, dtype=torch.float64)
    index = [TARGETS.index(name) for name in INSTRUCTION]
    target = torch.zeros(batch, len(TARGETS), dtype=torch.float64)
    valid = torch.zeros(batch, len(TARGETS), dtype=torch.float64)
    target[0, index] = torch.tensor([5_000.0, -2_000.0, 1.0, 0.0, 90.0, 20_000.0, 900.0], dtype=torch.float64)
    valid[0, index] = 1.0                    # the second sample has no fix ahead: it carries nothing
    weight = torch.ones(batch, dtype=torch.float64)
    log_sigma = torch.full((batch, k, p), math.log(0.5), dtype=torch.float64)
    equal = torch.zeros(batch, k, dtype=torch.float64)

    def nll(fan, logits=equal, t=target, v=valid, w=weight):
        return float(mixture_nll(PlanPrediction(t, torch.zeros(len(t)), fan, log_sigma[:len(t)], logits[:len(t)]), t, v, w, scale))

    on = target[:, index].unsqueeze(1).repeat(1, k, 1)          # both components on the target
    off = on.clone()
    off[:, :, 0] += 10_000.0                                     # both 10 km off ahead
    half = on.clone()
    half[:, 1, 0] += 10_000.0                                    # one on, one off, equal weights
    confident = torch.tensor([[5.0, -5.0]] * batch, dtype=torch.float64)
    assert nll(off) > nll(half) > nll(on)
    assert nll(half, confident) < nll(half) and abs(nll(half, confident) - nll(on)) < 1e-3
    # the sample with no fix ahead contributes nothing: the batch reads as its first sample alone
    assert nll(on) == pytest.approx(nll(on[:1], equal[:1], target[:1], valid[:1], weight[:1]))
    # the dispatch: under the fan `kinematic` IS the NLL at its weight, the other three as the point head's
    config = _plan_config(plan_fan_components=k, plan_instruction_loss_weight=2.0)
    context = {
        CONTEXT_TARGETS: target, CONTEXT_VALID: valid, CONTEXT_NEXT_IS_JOIN: torch.tensor([0.0, 1.0], dtype=torch.float64),
        CONTEXT_SERIES: torch.zeros(batch, dtype=torch.int64), CONTEXT_ROLLED: torch.zeros(batch, dtype=torch.float64),
    }
    fan_prediction = PlanPrediction(target, torch.zeros(batch), half, log_sigma, equal)
    point_prediction = PlanPrediction(target, torch.zeros(batch))
    fanned = plan_loss_components(fan_prediction, weight, config, context)
    pointed = plan_loss_components(point_prediction, weight, config, context)
    assert float(fanned.kinematic) == pytest.approx(2.0 * nll(half))
    assert float(pointed.kinematic) == 0.0      # `values` sits on the target: the L1 is zero
    for name in ("state", "final_time", "terminal"):
        assert float(getattr(fanned, name)) == pytest.approx(float(getattr(pointed, name)))


@pytest.mark.skipif(
    not (DEFAULT_PROCEDURE_ROOT / AIRPORT / "procedure-details").is_dir(),
    reason="the KRDU procedure documents are not on this machine",
)
def test_a_fan_head_trains_and_a_member_flies_one_step_deep(tmp_path):
    config = _plan_config(plan_fan_components=2)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=4, seed=3)
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == 4, report.format()
    skeleton = runway_skeleton(series[0])
    train(series, config, output_dir=tmp_path / "run", data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert loaded.plan_fan_components == 2
    item, anchor, device = series[0], default_anchor(loaded), torch.device("cpu")
    common = (model, [item], loaded, normalizer, [anchor], device, [skeleton])
    top, = rolled_predictions_lockstep(*common)
    members = [rolled_predictions_lockstep(*common, first_component=c)[0] for c in range(2)]
    same, = rolled_predictions_lockstep(*common, first_order=lambda o, s: o)
    for flight in (top, *members, same):
        assert np.all(np.isfinite(flight.forecast.values)) and flight.orders
    assert np.array_equal(same.forecast.values, top.forecast.values)
    # the top-weight component's member IS the top-1 flight; the other differs where the
    # first order flew a fix (the mixture's means start 4 km apart across the course)
    first = [m.orders[0]["flown_fix"] for m in members]
    if first[0] is not None and first[1] is not None:
        assert first[0] != first[1]
        assert any(np.array_equal(m.forecast.values, top.forecast.values) for m in members)
    with pytest.raises(ValueError, match="one step deep"):
        rolled_predictions_lockstep(*common, first_component=0, hold_asks=2)
    # the displacement: the fix moves in runway axes and the schedule coordinate with it
    fix = Instruction(skeleton.target_e + 10_000.0, skeleton.target_n + 2_000.0, 0.3, 80.0, 10_000.0, 500.0)
    order = PlanOrder(300.0, 90.0, 8_000.0, 70.0, 600.0, 10_000.0, 25_000.0, fix, 0.1)
    toward = displaced(order, 5_000.0, 0.0, skeleton)
    d0, xt0 = skeleton.axes(np.array([fix.fix_e]), np.array([fix.fix_n]))
    d1, xt1 = skeleton.axes(np.array([toward.instruction.fix_e]), np.array([toward.instruction.fix_n]))
    assert float(d0[0] - d1[0]) == pytest.approx(5_000.0, abs=1e-3) and float(xt1[0] - xt0[0]) == pytest.approx(0.0, abs=1e-3)
    assert toward.instruction.remaining_m == pytest.approx(5_000.0)
    right = displaced(order, 5_000.0, math.pi / 2, skeleton)
    d2, xt2 = skeleton.axes(np.array([right.instruction.fix_e]), np.array([right.instruction.fix_n]))
    assert float(xt2[0] - xt0[0]) == pytest.approx(5_000.0, abs=1e-3) and float(d2[0] - d0[0]) == pytest.approx(0.0, abs=1e-3)
    assert right.instruction.remaining_m == pytest.approx(10_000.0)
    closing = PlanOrder(300.0, 90.0, 8_000.0, 70.0, 600.0, 10_000.0, 25_000.0, None, 0.9)
    assert displaced(closing, 5_000.0, 0.0, skeleton) is closing



def test_the_readouts_single_step_reading():
    """`plan_fan_readout`: the single-step reading names the top and the nearest component,
    and covers the truth inside a component's 2σ box (the displacement is tested on a real
    skeleton in the gated test below)."""
    k, p = 3, len(TARGETS)
    fan = np.zeros((k, p))
    i_ahead, i_across = TARGETS.index("next_ahead_m"), TARGETS.index("next_across_m")
    fan[:, i_ahead] = [5_000.0, 9_000.0, 20_000.0]
    fan[:, i_across] = [0.0, 3_000.0, -4_000.0]
    weights = np.array([0.2, 0.7, 0.1])
    sigma = np.full((k, len(INSTRUCTION)), 500.0)
    truth = np.zeros(p)
    truth[i_ahead], truth[i_across] = 8_500.0, 2_600.0
    row = single_step(fan, weights, sigma, truth, True)
    assert row["top_component"] == 1 and row["nearest_component"] == 1 and row["covered_2sigma"]
    assert row["top_fix_error_m"] == pytest.approx(math.hypot(500.0, 400.0))
    truth[i_across] = 3_000.0 + COVERAGE_SIGMAS * 500.0 + 1.0     # past component 1's box; 0 and 2 are far ahead/behind
    assert not single_step(fan, weights, sigma, truth, True)["covered_2sigma"]
    assert single_step(fan, weights, sigma, truth, False) == {"top_component": 1, "top_weight": 0.7, "has_truth_fix": False}



def _readout_row(dataset_id: str, *, tortuosity: float, established_at_anchor: bool, fanned: bool) -> dict:
    scored = {"ade_m": 300.0, "chamfer_m": 40.0, "established": True}
    return {
        "dataset_id": dataset_id,
        "difficulty": {"route_tortuosity": tortuosity, "established_at_anchor": established_at_anchor, "remaining_path_m": 20_000.0},
        "single": {"top_component": 1, "top_weight": 0.7, "has_truth_fix": False},
        "top": dict(scored),
        "fan": [dict(scored), dict(scored, ade_m=200.0)] if fanned else None,
        "control": [dict(scored), dict(scored)] if fanned else None,
    }


def test_the_table_renders_a_stratum_with_no_fanned_flight():
    """The established stratum has no fix ahead and so no fan cell: `n/a`, never a crash
    (2026-09-12: a full run died formatting it after every rollout had been flown)."""
    rows = [
        _readout_row("a", tortuosity=1.3, established_at_anchor=False, fanned=True),
        _readout_row("b", tortuosity=1.0, established_at_anchor=True, fanned=False),
    ]
    summary = summarize(rows, 2)
    assert summary[STRATUM_ESTABLISHED]["fan"]["flights"] == 0 and summary[STRATUM_VECTORED]["fan"]["min_ade_mean_m"] == 200.0
    text = format_table(summary, 2)
    assert "n/a" in text and "fan minADE_K mean / p50" in text
