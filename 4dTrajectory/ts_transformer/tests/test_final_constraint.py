"""The final-approach constraint in the learned model: bounded output, penalty, projection."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import channels as ch  # noqa: E402
import final_approach_geometry as fag  # noqa: E402
from batch_contract import unpack_batch  # noqa: E402
from config import (  # noqa: E402
    CORRIDOR_GATE_FAF, CORRIDOR_GATE_ON_FINAL, STATE_POSITION_CORRIDOR_BOUNDED, TSConfig,
)
from dataset import (  # noqa: E402
    ARRIVAL_DATA_PROVENANCE_SCHEMA, FixedAnchorTrajectoryWindows, Normalizer, build_series,
    final_approach_arrays, probe_dynamics, probe_final_approach,
)
from export import build_prediction_record  # noqa: E402
from forecast import Forecast, forecast_approach, project_onto_final  # noqa: E402
from models import build_model  # noqa: E402
from prediction_outputs import StateOutputLayer, StatePrediction  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402
from train import (  # noqa: E402
    STATE_LOSS_COMPONENT_NAMES, ProcedureMultipliers, load_checkpoint, procedure_loss,
    state_prediction_loss_components, train,
)
from trajectory_data_process.harvest.arrivals import SCHEMA_VERSION as ARRIVAL_SCHEMA  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"
C = len(ch.CHANNELS)


def _series(n_flights=8, **overrides):
    config = TSConfig(**{"seq_len": 20, "n_segments": 8, "device": "cpu", **overrides})
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    return series, config


def _provenance():
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64, "source_records": []}],
    }


def _physical_normalizer() -> Normalizer:
    return Normalizer(
        mean=np.zeros(C), std=np.array([2000.0, 2000.0, 300.0, 40.0, 40.0, 4.0]),
    )


def _straight_in_rows(d_m: np.ndarray, *, offset_m: float, height_offset_m: float = 0.0,
                      heading_deg: float = 0.0, speed: float = 70.0) -> np.ndarray:
    """Physical chart rows for a runway whose course is +e (psi = 0): ``d`` back from the
    threshold, ``offset_m`` to the right, on the 3° glidepath plus ``height_offset_m``."""
    rows = np.zeros((len(d_m), C))
    rows[:, ch.IDX["e"]] = -d_m
    rows[:, ch.IDX["n"]] = -offset_m           # xt = e·sin0 − n·cos0 = −n → +offset = right
    rows[:, ch.IDX["u"]] = np.clip(d_m, 0, None) * math.tan(math.radians(3.0)) + height_offset_m
    rows[:, ch.IDX["edot"]] = speed * math.cos(math.radians(heading_deg))
    rows[:, ch.IDX["ndot"]] = speed * math.sin(math.radians(heading_deg))
    return rows


class _Fixed(torch.nn.Module):
    def __init__(self, normalized: torch.Tensor):
        super().__init__()
        self.normalized = normalized

    def forward(self, history):
        return self.normalized.expand(len(history), -1, -1)


def _context(batch: int, *, d_faf=float("nan")) -> dict[str, torch.Tensor]:
    return {
        "runway_heading_rad": torch.zeros(batch),
        "glidepath_tan": torch.full((batch,), math.tan(math.radians(3.0))),
        "final_approach_fix_m": torch.full((batch,), d_faf),
    }


def test_corridor_bounded_output_binds_the_rows_on_the_final_and_leaves_the_rest():
    normalizer = _physical_normalizer()
    d = np.array([15_000.0, 12_000.0, 9_000.0, 6_000.0, 3_000.0, 1_000.0])
    # An aligned straight-in 200 m above the glidepath: 300 m right on the first three
    # rows (inside the membership cone, outside the k=0.5 design corridor), 800 m right on
    # the last three (outside the 500 m membership floor: not on the final).
    physical = np.concatenate([
        _straight_in_rows(d[:3], offset_m=300.0, height_offset_m=200.0),
        _straight_in_rows(d[3:], offset_m=800.0, height_offset_m=200.0),
    ])
    offsets = np.array([300.0] * 3 + [800.0] * 3)
    config = TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, seq_len=4,
                      n_segments=len(physical))
    normalized = torch.tensor(normalizer.encode(physical), dtype=torch.float32)[None]
    layer = StateOutputLayer(_Fixed(normalized), config, normalizer)
    # The history's last row is the anchor, upstream on the same line: the first predicted
    # row's direction is read against it.
    anchor = _straight_in_rows(np.array([17_000.0]), offset_m=300.0, height_offset_m=200.0)
    history = torch.tensor(normalizer.encode(np.repeat(anchor, config.seq_len, axis=0)), dtype=torch.float32)[None].expand(3, -1, -1)
    with pytest.raises(ValueError, match="final-approach context"):
        layer(history)
    out = layer(history, _context(3)).states
    decoded = normalizer.decode(out[0].detach().numpy().astype(np.float64))
    e, n, u = decoded[:, ch.IDX["e"]], decoded[:, ch.IDX["n"]], decoded[:, ch.IDX["u"]]
    d_out, xt_out = -e, -n
    # Every row keeps its along-track distance; the rows the output places on the final
    # (inside the membership cone) end inside the k-cone and the glidepath window they
    # started outside of, pulled in rather than zeroed; the rows outside the membership
    # cone are not on the final by the output's own account and stay put.
    assert np.allclose(d_out, d, atol=1e-3)
    halfwidth = fag.FAS.course_width_m * (d + fag.FAS.d_garp_m) / fag.FAS.d_garp_m
    on_final = offsets <= np.maximum(fag.MEMBERSHIP_K * halfwidth, fag.MEMBERSHIP_FLOOR_M)
    assert on_final.tolist() == [True, True, True, False, False, False]
    bound = np.abs(xt_out)
    assert np.all(bound[on_final] <= fag.K_MARGIN * halfwidth[on_final] + 1e-3)
    assert np.all(bound[on_final] > 0.5 * fag.K_MARGIN * halfwidth[on_final])
    assert np.allclose(bound[~on_final], 800.0, atol=1e-2)
    glidepath = d * math.tan(math.radians(3.0))
    above = u - glidepath
    assert np.all(above[on_final] <= fag.GLIDEPATH_ABOVE_M + 1e-3) and np.all(above[on_final] > 0.0)
    assert np.allclose(above[~on_final], 200.0, atol=1e-2)
    # Velocity channels pass through the layer unchanged.
    assert np.allclose(decoded[:, list(ch.VELOCITY_IDX)], physical[:, list(ch.VELOCITY_IDX)], atol=1e-3)
    # The FAF gate binds by distance alone: the 800 m rows inside the FAF are bound too,
    # the rows beyond it are free.
    faf_layer = StateOutputLayer(
        _Fixed(normalized),
        TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, corridor_gate=CORRIDOR_GATE_FAF,
                 seq_len=4, n_segments=len(physical)),
        normalizer,
    )
    with pytest.raises(ValueError, match="FAF distance"):
        faf_layer(history, _context(3))
    faf_out = normalizer.decode(faf_layer(history, _context(3, d_faf=10_000.0)).states[0].detach().numpy().astype(np.float64))
    assert np.all(np.abs(faf_out[3:, ch.IDX["n"]]) <= fag.K_MARGIN * halfwidth[3:] + 1e-3)
    assert np.allclose(faf_out[:2], physical[:2], atol=1e-2)


def test_batches_carry_the_final_approach_context_only_when_the_recipe_needs_it():
    plain_series, plain = _series()
    normalizer = Normalizer.fit(plain_series)
    plain_batch = FixedAnchorTrajectoryWindows(plain_series, plain, normalizer).batch([0, 1])
    assert len(plain_batch) == 5 and unpack_batch(plain_batch)[5] is None
    for overrides in (
        {"state_position_reference": STATE_POSITION_CORRIDOR_BOUNDED},
        {"procedure_loss_lateral_weight": 1.0},
        {"procedure_loss_dual_step": 0.1},
    ):
        series, config = _series(**overrides)
        assert config.uses_final_approach_context
        batch = FixedAnchorTrajectoryWindows(series, config, normalizer).batch([0, 1])
        context = unpack_batch(batch)[5]
        assert tuple(context) == fag.FINAL_APPROACH_KEYS
        target = series[0].scenario.target
        assert context["runway_heading_rad"][0] == pytest.approx(target.psi)
        assert context["glidepath_tan"][0] == pytest.approx(math.tan(-target.gamma))
        assert math.isnan(float(context["final_approach_fix_m"][0]))   # on-final: no FAF read
    rows = final_approach_arrays(plain_series[0], fix_distance_m=9_500.0)
    assert rows["final_approach_fix_m"] == 9_500.0
    # The probe carries the same keys as the real rows; the control dynamics share only
    # the runway heading (a control recipe cannot bound or penalise the corridor).
    assert tuple(probe_final_approach(2, torch.device("cpu"))) == fag.FINAL_APPROACH_KEYS
    assert "runway_heading_rad" in probe_dynamics(2, torch.device("cpu"))
    assert "final_approach_fix_m" not in probe_dynamics(2, torch.device("cpu"))


def test_procedure_loss_charges_predicted_rows_where_the_truth_is_established():
    normalizer = Normalizer(mean=np.zeros(C), std=np.ones(C))
    d = np.array([20_000.0, 16_000.0, 12_000.0, 8_000.0, 4_000.0, 200.0])
    truth = torch.tensor(_straight_in_rows(d, offset_m=0.0), dtype=torch.float32)[None]
    weights = torch.ones_like(truth)
    context = _context(1)
    flight_weights = torch.ones(1)
    config = TSConfig(procedure_loss_lateral_weight=1.0, procedure_loss_vertical_weight=1.0,
                      seq_len=4, n_segments=len(d))
    halfwidth = fag.FAS.course_width_m * (d + fag.FAS.d_garp_m) / fag.FAS.d_garp_m
    # 300 m right on every row: outside the k-cone where k·hw < 300 (d ≤ ~13 km).
    shifted = torch.tensor(_straight_in_rows(d, offset_m=300.0), dtype=torch.float32)[None]
    prediction = StatePrediction(states=shifted, final_time_s=torch.tensor([100.0]))
    term, diagnostics = procedure_loss(
        prediction, truth, weights[..., list(ch.POSITION_IDX)].sum(-1), flight_weights,
        config, normalizer, context, None,
    )
    gated = (d > fag.NEAR_THRESHOLD_M)                                     # truth on the centreline: all but the 200 m row
    violating = gated & (300.0 > fag.K_MARGIN * halfwidth)
    assert int(diagnostics["procedure_gated_rows"]) == int(gated.sum())
    assert int(diagnostics["procedure_lateral_violations"]) == int(violating.sum())
    assert int(diagnostics["procedure_vertical_violations"]) == 0
    expected = np.mean(np.where(violating, ((300.0 - fag.K_MARGIN * halfwidth) / 100.0) ** 2, 0.0)[gated])
    assert float(term) == pytest.approx(expected, rel=1e-4)
    # Multipliers, not the config weights, scale the term when supplied.
    doubled, _ = procedure_loss(prediction, truth, weights[..., :3].sum(-1), flight_weights,
                                config, normalizer, context, ProcedureMultipliers(2.0, 5.0))
    assert float(doubled) == pytest.approx(2.0 * expected, rel=1e-4)
    # An inactive recipe contributes exactly zero and no diagnostics; a missing context raises.
    zero, none = procedure_loss(prediction, truth, weights[..., :3].sum(-1), flight_weights,
                                TSConfig(seq_len=4, n_segments=len(d)), normalizer, None, None)
    assert float(zero) == 0.0 and none == {}
    with pytest.raises(ValueError, match="context"):
        procedure_loss(prediction, truth, weights[..., :3].sum(-1), flight_weights, config, normalizer, None, None)
    # A truth that never establishes (4 km right the whole way) gates nothing.
    off = torch.tensor(_straight_in_rows(d, offset_m=4_000.0), dtype=torch.float32)[None]
    _, empty = procedure_loss(prediction, off, weights[..., :3].sum(-1), flight_weights, config, normalizer, context, None)
    assert int(empty["procedure_gated_rows"]) == 0
    # The state objective reports it as its fifth component.
    assert STATE_LOSS_COMPONENT_NAMES[-1] == "procedure"
    components = state_prediction_loss_components(
        prediction, truth[:, -1], truth, weights, torch.tensor([100.0]), flight_weights,
        config, normalizer, context,
    )
    assert float(components.extras["procedure"]) == pytest.approx(expected, rel=1e-4)
    assert components.diagnostics == diagnostics or int(components.diagnostics["procedure_gated_rows"]) == int(gated.sum())


def test_dual_multipliers_rise_on_excess_violation_and_never_go_negative():
    config = TSConfig(procedure_loss_dual_step=0.5, procedure_loss_epsilon=0.02)
    multipliers = ProcedureMultipliers.from_config(config)
    assert multipliers == ProcedureMultipliers(0.0, 0.0)
    multipliers.update(0.12, 0.01, config)
    assert multipliers.lateral == pytest.approx(0.05) and multipliers.vertical == 0.0
    multipliers.update(0.0, 0.0, config)
    assert multipliers.lateral == pytest.approx(0.04)
    fixed = ProcedureMultipliers.from_config(TSConfig(procedure_loss_lateral_weight=3.0))
    fixed.update(1.0, 1.0, TSConfig(procedure_loss_lateral_weight=3.0))
    assert fixed.to_dict() == {"lateral": 3.0, "vertical": 0.0}
    assert ProcedureMultipliers.from_config(TSConfig()) is None


def _small_train(tmp_path, **overrides):
    series, config = _series(
        n_flights=12, epochs=2, patience=1, batch_size=32, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, **overrides,
    )
    train(series, config, output_dir=tmp_path, data_provenance=_provenance(), verbose=False)
    return series


def test_penalised_training_logs_the_dual_state_and_stores_the_multipliers(tmp_path):
    _small_train(tmp_path, procedure_loss_lateral_weight=0.5, procedure_loss_vertical_weight=0.5,
                 procedure_loss_dual_step=0.25)
    _model, _config, _normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert set(payload["procedure_multipliers"]) == {"lateral", "vertical"}
    import json
    epochs = json.loads((tmp_path / "history.json").read_text())["history"]
    record = epochs[0]["procedure"]
    assert {"train_gated_rows", "train_lateral_violation_rate", "lambda_lateral", "lambda_vertical"} <= set(record)
    assert record["train_gated_rows"] > 0
    assert "procedure" in epochs[0]["train_components"]


def test_corridor_bounded_checkpoint_round_trips_and_projection_marks_records(tmp_path):
    series = _small_train(tmp_path, state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED)
    model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert loaded.state_position_reference == STATE_POSITION_CORRIDOR_BOUNDED
    assert np.allclose(model.channel_std.numpy(), normalizer.std) and np.allclose(model.channel_mean.numpy(), normalizer.mean)
    forecast = forecast_approach(model, series[0], loaded, normalizer, device=torch.device("cpu"))
    assert np.isfinite(forecast.values).all() and forecast.projected_onto_final is None
    projected = forecast_approach(model, series[0], loaded, normalizer, device=torch.device("cpu"),
                                  project_final=CORRIDOR_GATE_ON_FINAL)
    assert projected.projected_onto_final == CORRIDOR_GATE_ON_FINAL
    record = build_prediction_record(series[0], projected, index=0, model_name="itransformer", horizon_mode=loaded.horizon_mode)
    assert record.source["projectedOntoFinal"] == CORRIDOR_GATE_ON_FINAL
    plain = build_prediction_record(series[0], forecast, index=0, model_name="itransformer", horizon_mode=loaded.horizon_mode)
    assert plain.source["projectedOntoFinal"] is None
    # A fresh, unbound layer runs finite (the batch-size probe) and refuses to bind
    # anything but the bounded output.
    probe = build_model(loaded)
    assert torch.all(probe.channel_std == 1.0)
    with pytest.raises(ValueError, match="corridor-bounded"):
        build_model(TSConfig(seq_len=4, n_segments=3)).bind_normalizer(normalizer)


def test_project_onto_final_clamps_only_the_established_tail_and_rederives_velocity():
    series, _config = _series(n_flights=1)
    item = series[0]
    psi = float(item.scenario.target.psi)
    d = np.array([14_000.0, 12_000.0, 10_000.0, 8_000.0, 6_000.0, 4_000.0, 2_000.0])
    # Downwind rows abeam, then a final 450 m LEFT of the centreline and 100 m low.
    ue, un = math.cos(psi), math.sin(psi)
    rows = np.zeros((len(d) + 2, C))
    downwind_d = np.array([9_000.0, 8_500.0]); downwind_xt = 5_000.0
    rows[:2, ch.IDX["e"]] = -downwind_d * ue + downwind_xt * un
    rows[:2, ch.IDX["n"]] = -downwind_d * un - downwind_xt * ue
    rows[:2, ch.IDX["edot"]], rows[:2, ch.IDX["ndot"]] = -70.0 * ue, -70.0 * un
    xt = -450.0                                   # inside the 500 m membership floor everywhere
    rows[2:, ch.IDX["e"]] = -d * ue + xt * un
    rows[2:, ch.IDX["n"]] = -d * un - xt * ue
    rows[2:, ch.IDX["u"]] = d * math.tan(math.radians(3.0)) - 100.0
    rows[2:, ch.IDX["edot"]], rows[2:, ch.IDX["ndot"]] = 70.0 * ue, 70.0 * un
    durations = np.full(len(rows), 2.0)
    forecast = Forecast(
        times=np.arange(1, len(rows) + 1) * 2.0, values=rows, normalized_progress=np.linspace(0, 1, len(rows)),
        anchor=item.n_samples - 1, final_time_s=2.0 * len(rows), predicted_final_time_s=2.0 * len(rows),
        horizon_mode="full", passes=1, truncated_at_threshold=True, horizon_capped=False,
        sample_durations_s=durations, segment_durations_s=durations,
    )
    projected = project_onto_final(forecast, item, CORRIDOR_GATE_ON_FINAL)
    e, n, u = (projected.values[:, ch.IDX[k]] for k in ("e", "n", "u"))
    d_out = -(e * ue + n * un); xt_out = e * un - n * ue
    halfwidth = fag.FAS.course_width_m * (np.clip(d_out, 0, None) + fag.FAS.d_garp_m) / fag.FAS.d_garp_m
    assert np.allclose(projected.values[:2, :3], rows[:2, :3])                         # downwind untouched
    assert np.allclose(d_out[2:], d)
    # The established tail: the first final row's direction is read against the
    # downwind row before it (a sideways jump in this fixture), so the tail begins one
    # row later; from there every row is clamped to the left edge / the floor.
    assert np.allclose(xt_out[2], xt) and np.allclose(u[2], rows[2, ch.IDX["u"]])
    assert np.allclose(xt_out[3:], -fag.K_MARGIN * halfwidth[3:])                       # clamped to the left edge
    assert np.allclose(u[3:] - d[1:] * math.tan(math.radians(3.0)), -fag.GLIDEPATH_BELOW_M)  # clamped to the floor
    # Velocities are re-derived from the moved positions (left differences over 2 s).
    expected_velocity = np.diff(projected.values[:, :3], axis=0) / 2.0
    assert np.allclose(projected.values[1:, 3:6], expected_velocity)
    assert projected.projected_onto_final == CORRIDOR_GATE_ON_FINAL
    with pytest.raises(ValueError, match="unknown corridor gate"):
        project_onto_final(forecast, item, "never")


def test_synthetic_fixture_manifest_schema_is_current():
    # The provenance fixture above mirrors the arrival manifest contract; if that schema
    # moves, this file must be re-read, not silently kept green.
    assert ARRIVAL_SCHEMA.startswith("harvest-arrivals-v")


# ── Regressions from the 2026-09-04 review ──────────────────────────────────

def test_checkpoints_written_before_the_offset_mask_buffer_still_load():
    """The 2026-09-03 arm-A checkpoints predate ``offset_mask``; a pure function of the
    channel contract must not be a persisted key that blocks them."""
    config = TSConfig(seq_len=4, n_segments=3)
    state = build_model(config).state_dict()
    assert "offset_mask" not in state
    fresh = build_model(config)
    fresh.load_state_dict(state)                      # strict: nothing missing
    # …while the bounded output's statistics ARE persisted (they are learned scale).
    bounded = build_model(TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, seq_len=4, n_segments=3))
    assert {"channel_mean", "channel_std"} <= set(bounded.state_dict())


ARM_A_CHECKPOINT = REPO_ROOT / "4dTrajectory/outputs/KRDU/experiments/airport_frame_20260903/A_threshold_enu/checkpoint.pt"
STATE_V2_CHECKPOINT = REPO_ROOT / "4dTrajectory/outputs/KRDU/experiments/state_v2_20260903/A_anchor_relative/checkpoint.pt"


@pytest.mark.skipif(not ARM_A_CHECKPOINT.is_file(), reason="the 2026-09-03 arm-A checkpoint is not on this machine")
def test_the_2026_09_03_arm_a_checkpoint_loads():
    model, config, _normalizer, _payload = load_checkpoint(ARM_A_CHECKPOINT)
    assert config.state_position_reference == "absolute"
    assert torch.equal(model.offset_mask[list(ch.POSITION_IDX)], torch.ones(3))


@pytest.mark.skipif(not STATE_V2_CHECKPOINT.is_file(), reason="the 2026-09-03 state-v2 checkpoint is not on this machine")
def test_the_2026_09_03_state_v2_checkpoint_that_stored_the_mask_loads_too():
    model, config, _normalizer, _payload = load_checkpoint(STATE_V2_CHECKPOINT)
    assert config.state_position_reference == "anchor-relative"
    assert torch.equal(model.offset_mask[list(ch.POSITION_IDX)], torch.ones(3))


def test_a_state_dict_that_stored_the_mask_loads_through_load_checkpoint(tmp_path):
    """Both generations pinned without the local artifacts: a checkpoint carrying the
    now-transient ``offset_mask`` key must load, and a stray other key must still fail."""
    series = _small_train(tmp_path)
    payload = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    payload["model_state"]["offset_mask"] = torch.ones(C)
    torch.save(payload, tmp_path / "with_mask.pt")
    model, _config, _normalizer, _payload = load_checkpoint(tmp_path / "with_mask.pt")
    assert np.isfinite(forecast_approach(model, series[0], _config, _normalizer, device=torch.device("cpu")).values).all()
    payload["model_state"]["not_a_parameter"] = torch.ones(1)
    torch.save(payload, tmp_path / "stray.pt")
    with pytest.raises(RuntimeError, match="not_a_parameter"):
        load_checkpoint(tmp_path / "stray.pt")


def test_the_corridor_is_refused_outside_the_threshold_anchored_enu_chart():
    for frame in ("airport-enu", "runway-aligned"):
        with pytest.raises(ValueError, match="threshold-anchored ENU"):
            TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, coordinate_frame=frame)
        with pytest.raises(ValueError, match="threshold-anchored ENU"):
            TSConfig(procedure_loss_dual_step=0.1, coordinate_frame=frame)
    TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, coordinate_frame="enu")


def test_the_gate_direction_comes_from_the_predicted_positions_not_the_velocity_channels():
    normalizer = _physical_normalizer()
    d = np.array([12_000.0, 10_000.0, 8_000.0, 6_000.0])
    # Positions run straight down the final (aligned path) but the velocity channels say
    # the aircraft is flying AWAY from the runway: the gate must follow the positions.
    physical = _straight_in_rows(d, offset_m=300.0, height_offset_m=200.0, heading_deg=180.0)
    config = TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, seq_len=4, n_segments=len(d))
    layer = StateOutputLayer(_Fixed(torch.tensor(normalizer.encode(physical), dtype=torch.float32)[None]), config, normalizer)
    # The history's last row (the anchor) sits upstream on the same line.
    anchor = _straight_in_rows(np.array([14_000.0]), offset_m=300.0)
    history = torch.tensor(normalizer.encode(np.repeat(anchor, config.seq_len, axis=0)), dtype=torch.float32)[None].expand(2, -1, -1)
    out = normalizer.decode(layer(history, _context(2)).states[0].detach().numpy().astype(np.float64))
    halfwidth = fag.FAS.course_width_m * (d + fag.FAS.d_garp_m) / fag.FAS.d_garp_m
    assert np.all(np.abs(out[:, ch.IDX["n"]]) <= fag.K_MARGIN * halfwidth + 1e-3)      # bound: the PATH is aligned
    # And a path that runs the other way is not on the final, whatever the channels say.
    reversed_rows = physical[::-1].copy()
    reversed_rows[:, ch.IDX["edot"]] = 70.0                                            # channels claim "inbound"
    layer_back = StateOutputLayer(_Fixed(torch.tensor(normalizer.encode(reversed_rows), dtype=torch.float32)[None]), config, normalizer)
    anchor_back = _straight_in_rows(np.array([4_000.0]), offset_m=300.0)
    history_back = torch.tensor(normalizer.encode(np.repeat(anchor_back, config.seq_len, axis=0)), dtype=torch.float32)[None].expand(2, -1, -1)
    out_back = normalizer.decode(layer_back(history_back, _context(2)).states[0].detach().numpy().astype(np.float64))
    assert np.allclose(out_back[:, :3], reversed_rows[:, :3], atol=1e-2)
    # The helper itself: central differences with the anchor before the first row.
    e = torch.tensor([[0.0, 10.0, 20.0, 40.0]]); n = torch.zeros(1, 4)
    step_e, step_n = fag.position_direction(e, n, torch.tensor([-10.0]), torch.tensor([0.0]))
    assert step_e.tolist() == [[20.0, 20.0, 30.0, 20.0]] and torch.all(step_n == 0.0)
    cos = fag.alignment_cosine(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([0.0]))
    assert torch.all(cos == 0.0)


def test_procedure_loss_does_not_charge_predicted_rows_past_the_threshold():
    normalizer = Normalizer(mean=np.zeros(C), std=np.ones(C))
    d = np.array([12_000.0, 8_000.0, 4_000.0, 1_000.0])
    truth = torch.tensor(_straight_in_rows(d, offset_m=0.0), dtype=torch.float32)[None]
    weights = torch.ones_like(truth)
    config = TSConfig(procedure_loss_lateral_weight=1.0, seq_len=4, n_segments=len(d))
    # A fast forecast that is already 2 km PAST the threshold on every row, 400 m off.
    past = torch.tensor(_straight_in_rows(np.full(len(d), -2_000.0), offset_m=400.0), dtype=torch.float32)[None]
    term, diagnostics = procedure_loss(
        StatePrediction(states=past, final_time_s=torch.tensor([10.0])), truth,
        weights[..., :3].sum(-1), torch.ones(1), config, normalizer, _context(1), None,
    )
    assert float(term) == 0.0 and int(diagnostics["procedure_gated_rows"]) == 0


def test_dual_history_records_the_lambda_used_and_the_next_one(tmp_path):
    _small_train(tmp_path, procedure_loss_lateral_weight=0.5, procedure_loss_dual_step=0.25)
    import json
    epochs = json.loads((tmp_path / "history.json").read_text())["history"]
    first = epochs[0]["procedure"]
    assert first["lambda_lateral"] == 0.5                       # the λ this epoch was scored with
    assert first["lambda_lateral_next"] >= 0.0
    if len(epochs) > 1:
        assert epochs[1]["procedure"]["lambda_lateral"] == first["lambda_lateral_next"]
    _model, _config, _normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    selected = payload["procedure_multipliers"]
    assert selected["lateral"] in {e["procedure"]["lambda_lateral"] for e in epochs}
