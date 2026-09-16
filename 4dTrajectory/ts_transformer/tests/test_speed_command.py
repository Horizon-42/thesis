"""The speed-command control axis, ts side (docs/2026-09-14_specific_force_control_design.md §12).

``control_thrust_parameterization="speed-command"`` makes the head's first column a target
airspeed RELATIVE to the anchor's, Δv, flown by a first-order speed loop through the
specific-force law's thrust. What must hold for it to be the same flight model read another
way, and for nothing stored to move:

* the teacher is the inverse of THIS forward model (a known Δv schedule comes back), and it
  is relative to the anchor the rollout starts from;
* the anchor actuator is the loop target the observed lookback implies, less the anchor's own
  speed — and it is tied to the specific-force column by ``a_Δ = g·τ_V·(n_x − sin γ)``;
* the box, the neutral ("hold the speed you have") and the probe are the contract's;
* the record stays in newtons, carries the command and says which contract it came from;
* the config refuses the pairings not built (the speed floor among them), every named recipe
  pins thrust-fraction, and the name moves only off the default.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_MEMBERS,
    CONTROL_HOOK_OFF,
    CONTROL_HOOK_SPEED_FLOOR,
    CONTROL_IMITATION_TARGET_FITTED,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_SPECIFIC_FORCE,
    CONTROL_SPEED_COMMAND,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
    control_recipe,
    recipe_settings,
)
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.export import build_prediction_record
from ts_transformer.outputs.conditioning import CONDITION_WIDTH
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.outputs.dynamics.backends import RolloutInputs, control_dynamics_backend
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.dynamics.inverse import _drag_force, reference_controls
from ts_transformer.outputs.envelope import (
    MIN_THRUST_FRACTION,
    SPEED_COMMAND_CONTRACT,
    SPEED_LOOP_TIME_CONSTANT_S,
    control_contract,
)
from ts_transformer.run_naming import run_display_name, run_slug

AIRPORT, RUNWAY = "KRDU", "05L"
LAG = {
    "control_dynamics_model": CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    "control_dynamics_backend": CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
}
SPEED_COMMAND = {**LAG, "control_thrust_parameterization": CONTROL_SPEED_COMMAND}
SPECIFIC_FORCE = {**LAG, "control_thrust_parameterization": CONTROL_SPECIFIC_FORCE}
MAX_THRUST_N = 240_000.0
AERO = (122.6, 2.7, 0.02, 0.04, 0.9, 0.1)
FRAME = (37.36, -121.93, 18.0, 0.0)
INITIAL_STATE = (37.55, -121.70, 1800.0, 120.0, -2.4, -0.035, 62_000.0)
HORIZON_S = 240.0


def _config(**overrides) -> TSConfig:
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        n_segments=8, seq_len=16, d_model=32, d_ff=64, n_heads=4, e_layers=1,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def _series(config: TSConfig, n_flights: int = 2):
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3), config, airport=AIRPORT
    )
    assert report.built == n_flights, report.format()
    return series


def _rows(config: TSConfig, series, anchor: int):
    return [
        dynamics_arrays(item, anchor, parameterization=config.control_thrust_parameterization,
                        condition_features=config.control_condition_features)
        for item in series
    ]


# ── the teacher is the inverse of this forward model ─────────────────────────


def test_a_speed_command_schedule_comes_back_through_its_own_inverse():
    """known Δv schedule -> lagged rollout -> dense reference -> inverse -> schedule back.

    A deceleration from 120 m/s in steps (the command relative to the anchor's speed) under a
    gentle coordinated turn, inside the thrust range at every stage (the clamp binding makes
    a command unrecoverable by construction), so the inverse is exact up to the 2 s grid."""
    config = _config(**SPEED_COMMAND, control_rollout_integrator_dt_s=0.1)
    n_segments = int(config.n_segments)
    progress = (np.arange(n_segments) + 0.5) / n_segments
    bank = np.deg2rad(10.0) * np.sin(np.pi * progress)
    controls = np.column_stack((
        -3.0 - 4.0 * progress,
        bank,
        np.cos(INITIAL_STATE[5]) / np.cos(bank),
    ))
    inputs = RolloutInputs(
        initial_state=torch.tensor([INITIAL_STATE], dtype=torch.float64),
        initial_controls=torch.tensor(controls[:1], dtype=torch.float64),
        controls=torch.tensor(controls, dtype=torch.float64).unsqueeze(0),
        segment_durations_s=torch.full((1, n_segments), HORIZON_S / n_segments, dtype=torch.float64),
        aero_params=torch.tensor([AERO], dtype=torch.float64),
        frame_params=torch.tensor([FRAME], dtype=torch.float64),
        max_thrust_n=torch.tensor([MAX_THRUST_N], dtype=torch.float64),
    )
    offsets = np.arange(2.0, HORIZON_S + 1e-9, 2.0)
    rollout = control_dynamics_backend(config).dense_rollout(
        inputs, torch.tensor(offsets, dtype=torch.float64).unsqueeze(0),
        torch.ones((1, len(offsets)), dtype=torch.bool), config,
    )
    times = np.concatenate(([0.0], offsets))
    states = np.concatenate(
        (np.array([INITIAL_STATE]), rollout.query_geodetic_states[0].numpy()), axis=0
    )
    recovered = reference_controls(
        states, times, config=config, aero_params=np.array(AERO), max_thrust_n=MAX_THRUST_N
    )
    midpoints = (np.arange(n_segments) + 0.5) * (HORIZON_S / n_segments)
    sampled = np.column_stack([np.interp(midpoints, times, recovered[:, k]) for k in range(3)])
    error = np.abs(sampled - controls)
    assert error[:, 0].max() < 0.1, f"speed command (m/s): {error[:, 0]}"
    assert error[:, 1].max() < np.deg2rad(0.5), f"bank: {np.rad2deg(error[:, 1])}"
    assert error[:, 2].max() < 5e-3, f"load factor: {error[:, 2]}"
    # The rollout itself flew the loop: the speed ends near the anchor speed + the last command.
    assert states[-1, 3] == pytest.approx(INITIAL_STATE[3] + controls[-1, 0], abs=0.5)


def test_the_anchor_actuator_is_the_loop_target_less_the_anchor_speed():
    """At the anchor the speed-command actuator is ``τ_V·(tangential)``, the specific-force
    actuator ``tangential/g + sin γ`` — so ``a_Δ = g·τ_V·(n_x − sin γ)`` on one track, and bank
    and load are the same numbers."""
    config = _config(**SPEED_COMMAND)
    series = _series(config, n_flights=1)[0]
    anchor = config.seq_len - 1
    speed = dynamics_arrays(series, anchor, parameterization=CONTROL_SPEED_COMMAND,
                            condition_features=config.control_condition_features)
    force = dynamics_arrays(series, anchor, parameterization=CONTROL_SPECIFIC_FORCE,
                            condition_features=config.control_condition_features)
    gamma = speed["initial_state"][5]
    expected = GRAVITY_MPS2 * SPEED_LOOP_TIME_CONSTANT_S * (force["initial_controls"][0] - np.sin(gamma))
    assert speed["initial_controls"][0] == pytest.approx(expected, abs=1e-9)
    assert np.array_equal(speed["initial_controls"][1:], force["initial_controls"][1:])
    assert np.allclose(speed["control_lower"], SPEED_COMMAND_CONTRACT.lower)
    assert np.allclose(speed["control_upper"], SPEED_COMMAND_CONTRACT.upper)


# ── the contract: box, neutral, probe ─────────────────────────────────────────


def test_an_untrained_speed_command_head_outputs_the_neutral():
    """The zeroed head starts every flight at the contract's neutral: Δv = 0, wings level,
    1 g."""
    config = _config(**SPEED_COMMAND)
    model = build_model(config).eval()
    contract = control_contract(CONTROL_SPEED_COMMAND)
    dynamics = {
        "condition": torch.randn(2, CONDITION_WIDTH),
        "control_lower": torch.tensor([contract.lower] * 2, dtype=torch.float32),
        "control_upper": torch.tensor([contract.upper] * 2, dtype=torch.float32),
    }
    with torch.no_grad():
        prediction = model(torch.randn(2, config.seq_len, config.enc_in), dynamics)
    expected = torch.tensor(contract.neutral).view(1, 1, 3).expand_as(prediction.controls)
    assert torch.allclose(prediction.controls, expected, atol=1e-4)


def test_an_untrained_speed_command_forecast_holds_the_anchor_speed():
    """What the neutral MEANS, flown, through the real forecast path: an untrained head brings
    every flight back to its anchor speed. The only excursion is the actuator's own start —
    the command the observed lookback implies at the anchor, which lags back to the neutral —
    so the speed never leaves V₀ by more than that start (plus a margin), and it ends ON V₀.
    The same head under specific-force drifts tens of m/s: no restoring force."""
    config = _config(**SPEED_COMMAND)
    series = _series(config)
    torch.manual_seed(5)
    model = build_model(config).eval()
    forecasts = forecast_control_batch(model, series, config, Normalizer.fit(series),
                                       config.seq_len - 1, torch.device("cpu"))
    for item, forecast in zip(series, forecasts, strict=True):
        row = _rows(config, [item], config.seq_len - 1)[0]
        anchor_speed, start = row["initial_state"][3], abs(row["initial_controls"][0])
        speed = forecast.geodetic_values[:, 3]
        assert forecast.final_time_s > 5.0 * SPEED_LOOP_TIME_CONSTANT_S, forecast.final_time_s
        assert np.abs(speed - anchor_speed).max() < start + 1.0, (item.flight_id, start)
        assert abs(speed[-1] - anchor_speed) < 0.5, (item.flight_id, speed[-1] - anchor_speed)


def test_the_probe_batch_is_in_the_run_contract():
    probe = probe_dynamics(2, torch.device("cpu"), _config(**SPEED_COMMAND))
    assert torch.allclose(probe["control_lower"][0].double(),
                          torch.tensor(SPEED_COMMAND_CONTRACT.lower, dtype=torch.float64))
    assert torch.allclose(probe["initial_controls"][0].double(),
                          torch.tensor(SPEED_COMMAND_CONTRACT.neutral, dtype=torch.float64))


# ── the record ───────────────────────────────────────────────────────────────


def test_a_speed_command_record_says_so_and_prices_its_thrust_where_each_segment_begins():
    config = _config(**SPEED_COMMAND)
    series = _series(config)
    normalizer = Normalizer.fit(series)
    model = build_model(config).eval()
    anchor = config.seq_len - 1
    forecast = forecast_control_batch(model, series, config, normalizer, anchor, torch.device("cpu"))[0]
    record = build_prediction_record(series[0], forecast, index=0, model_name=config.model,
                                     horizon_mode=config.horizon_mode)
    assert forecast.control_parameterization == CONTROL_SPEED_COMMAND
    assert len(forecast.commands) == len(forecast.controls) == config.n_segments
    assert record.states_payload["source"]["controlThrustParameterization"] == CONTROL_SPEED_COMMAND
    segments = record.states_payload["control_segments"]
    assert [s["speed_command_delta"] for s in segments] == pytest.approx(list(forecast.commands[:, 0]))
    assert all("specific_force" not in s for s in segments)
    # Each thrust, evaluated independently at ITS segment's start (numpy drag): the loop's
    # specific force toward V₀ + Δv at that state, clamped to the engine.
    context = _rows(config, series[:1], anchor)[0]
    aero_row, thrust = context["aero_params"], float(context["max_thrust_n"])
    anchor_speed = float(context["initial_state"][3])
    rows = {round(row["t"], 6): row for row in record.states_payload["predicted_states"]}
    for index, segment in enumerate(segments):
        if index == 0:
            alt, speed, gamma, mass = (context["initial_state"][i] for i in (2, 3, 5, 6))
            tolerance = 1e-9
        else:
            row = rows[round(segment["start_t"], 6)]
            alt, speed, gamma, mass = row["alt"], row["V"], row["gamma"], row["m"]
            tolerance = 1e-6
        drag = _drag_force(np.array([alt]), np.array([speed]), np.array([mass]),
                           np.array([segment["load_factor"]]), aero_row)[0]
        weight = mass * GRAVITY_MPS2
        n_x = np.sin(gamma) + (anchor_speed + segment["speed_command_delta"] - speed) / (
            GRAVITY_MPS2 * SPEED_LOOP_TIME_CONSTANT_S)
        expected = np.clip(weight * n_x + drag, MIN_THRUST_FRACTION * thrust, thrust)
        assert segment["thrust"] == pytest.approx(expected, abs=tolerance * weight), index


# ── one real training step ───────────────────────────────────────────────────


def test_a_speed_command_run_trains_through_the_whole_objective():
    """A real window batch, the simple-v3 terms plus the heading-rate term, through the
    lagged speed-command rollout: finite, and the gradient reaches the Δv column."""
    from ts_transformer.data.batch_contract import anchor_state, unpack_batch
    from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows
    from ts_transformer.training.objective import prediction_loss

    config = _config(
        **SPEED_COMMAND, n_segments=4, seq_len=8, d_model=16, d_ff=32,
        control_velocity_loss_weight=1.0, control_imitation_loss_weight=64.0,
        control_heading_rate_loss_weight=1.0, control_rollout_integrator_dt_s=0.5,
    )
    series = _series(config)
    normalizer = Normalizer.fit(series)
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = unpack_batch(
        dataset.batch(np.array([0, 1]))
    )
    assert torch.allclose(dynamics["control_upper"][0].double(),
                          torch.tensor(SPEED_COMMAND_CONTRACT.upper, dtype=torch.float64))
    # The teacher is in the contract's units: relative commands, inside the box.
    assert dynamics["reference_controls"][..., 0].abs().max() < 60.0
    model = build_model(config)
    loss = prediction_loss(
        model(x, dynamics), anchor_state(x, len(config.channels)), target, weights,
        final_time, flight_weights, config, normalizer, dynamics, dense,
    )
    loss.backward()
    assert torch.isfinite(loss)
    grad = model.control_head.control_projection.weight.grad.view(config.n_segments, 3, -1)
    assert torch.isfinite(grad).all() and grad[:, 0].abs().sum() > 0.0


def test_the_barrier_passes_the_speed_command_column_through():
    from ts_transformer.outputs.constraints import build_command_hook
    from ts_transformer.outputs.dynamics import rollout as control_rollout

    _config(**SPEED_COMMAND, control_command_hook="barrier+trombone")
    config = _config(**SPEED_COMMAND, control_command_hook="barrier")
    series = _series(config)
    rows = _rows(config, series, config.seq_len - 1)
    dynamics = {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}
    generator = torch.Generator().manual_seed(2)
    n = int(config.n_segments)
    controls = torch.stack((
        -20.0 * torch.rand(len(series), n, generator=generator, dtype=torch.float64),
        0.6 * (torch.rand(len(series), n, generator=generator, dtype=torch.float64) - 0.5),
        torch.ones(len(series), n, dtype=torch.float64),
    ), dim=-1)
    durations = torch.full((len(series), n), 20.0, dtype=torch.float64)
    hook = build_command_hook(config, dynamics)
    hooked = control_rollout.rollout_control_endpoints(
        controls, durations, dynamics, config, command_hook=hook
    )
    assert hook.diagnostics()["hook_steps"] == len(series) * n
    assert torch.equal(hooked.controls[..., 0], controls[..., 0])
    # The hooked path flies the SAME physics as a plain rollout of what it flew — the law
    # and each flight's own reference speed included (review S1: dropping the law from the
    # hooked path flew Δv as a thrust fraction and passed every other check).
    plain = control_rollout.rollout_control_endpoints(hooked.controls, durations, dynamics, config)
    assert torch.equal(plain.geodetic_states, hooked.geodetic_states)
    assert (hooked.geodetic_states[..., 3] > 40.0).all()


# ── the config and the name ──────────────────────────────────────────────────


def test_speed_command_is_refused_where_it_is_not_built():
    with pytest.raises(ValueError, match="first-order-lag flight model only"):
        _config(control_dynamics_model=CONTROL_DYNAMICS_POINT_MASS,
                control_thrust_parameterization=CONTROL_SPEED_COMMAND)
    with pytest.raises(ValueError, match="fitted-teacher table"):
        _config(**SPEED_COMMAND, control_imitation_loss_weight=1.0,
                control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                control_fitted_teacher_path="teacher.json")
    floors = [hook for hook in CONTROL_HOOKS_AVAILABLE
              if CONTROL_HOOK_SPEED_FLOOR in CONTROL_HOOK_MEMBERS.get(hook, ())]
    assert floors, "no hook contains the speed floor; this check would pass vacuously"
    for hook in floors:
        with pytest.raises(ValueError, match="not built for"):
            _config(**SPEED_COMMAND, control_command_hook=hook)
    for hook in set(CONTROL_HOOKS_AVAILABLE) - set(floors) - {CONTROL_HOOK_OFF}:
        _config(**SPEED_COMMAND, control_command_hook=hook)
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(prediction_output=PREDICTION_STATE,
                 control_thrust_parameterization=CONTROL_SPEED_COMMAND)


def test_every_named_recipe_refuses_the_speed_command():
    for name in CONTROL_RECIPE_NAMES:
        if name == CONTROL_RECIPE_CUSTOM:
            continue
        with pytest.raises(ValueError, match="recipe fields are frozen"):
            TSConfig(**{**recipe_settings(name, keep_name=True), **SPEED_COMMAND})
    TSConfig(**{**recipe_settings("simple-v3", keep_name=False), **SPEED_COMMAND})


def test_the_target_contract_spells_the_speed_command_constants():
    """The loop constant and the box are module constants, so the checkpoint's target
    contract carries them: a checkpoint trained under other constants is refused at load
    (review S2). The other laws' contracts do not change."""
    from ts_transformer.outputs import strategy

    config = _config(**SPEED_COMMAND)
    contract = strategy(config).target_contract(config)
    assert contract.endswith(SPEED_COMMAND_CONTRACT.identity_suffix)
    # verbatim as the stored N6 checkpoints carry it
    assert contract.endswith("+speed-command(tau-v=8s,box=-90..20m/s,neutral=0)-v1")
    # spelled from the constants the law flies and the box the head predicts in
    law = SPEED_COMMAND_CONTRACT.law
    assert f"tau-v={law.speed_time_constant_s:g}s" in contract
    assert f"box={SPEED_COMMAND_CONTRACT.lower[0]:g}..{SPEED_COMMAND_CONTRACT.upper[0]:g}m/s" in contract
    for other in (_config(**LAG), _config(**SPECIFIC_FORCE)):
        assert "speed-command" not in strategy(other).target_contract(other)


def test_the_anchor_actuator_is_clipped_to_the_speed_command_box(monkeypatch):
    """An anchor whose lookback implies a command past the box starts at the box's edge."""
    import ts_transformer.outputs.dynamics.context as context

    config = _config(**SPEED_COMMAND)
    series = _series(config, n_flights=1)[0]
    monkeypatch.setattr(context, "anchor_controls",
                        lambda *args, **kwargs: np.array([500.0, 0.0, 1.0]))
    row = _rows(config, [series], config.seq_len - 1)[0]
    assert row["initial_controls"][0] == SPEED_COMMAND_CONTRACT.upper[0]


def test_the_name_and_the_recipe_carry_the_law():
    config = _config(**SPEED_COMMAND)
    assert control_recipe(config)["thrust_parameterization"] == CONTROL_SPEED_COMMAND
    name = run_display_name(config.to_dict())
    assert "first-order-lag+speed-command @scaled-transport-chart-velocity" in name
    assert "_lag-sc-stcv_" in run_slug(config.to_dict())
    assert run_slug(config.to_dict()) != run_slug(_config(**SPECIFIC_FORCE).to_dict())


def test_the_train_cli_selects_the_speed_command(capsys):
    import argparse

    import ts_transformer.cli.common as cli_common

    parser = argparse.ArgumentParser()
    cli_common.add_data_args(parser)
    cli_common.add_training_args(parser)
    args = parser.parse_args([
        "--data", "unused.json", "--output-dir", "unused-output",
        "--prediction-output", "control", "--horizon-mode", "normalized",
        "--control-dynamics-model", "first-order-lag",
        "--control-dynamics-backend", "scaled-transport-chart-velocity",
        "--control-thrust-parameterization", "speed-command",
    ])
    config, _batch_auto = cli_common.config_from_args(args, parser)
    assert config.control_thrust_parameterization == CONTROL_SPEED_COMMAND
