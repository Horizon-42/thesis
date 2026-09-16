"""The path-angle vertical contract, ts side (docs/2026-09-14_specific_force_control_design.md §14).

``control_thrust_parameterization="specific-force+path-angle"`` makes the head's THIRD column a
path-angle target γ*, flown by a first-order path loop through the load factor the lag RHS
re-solves at every stage; the first column is the specific force, unchanged. What must hold for
it to be the same flight model read another way, and for nothing stored to move:

* the teacher is the inverse of THIS forward model (a known γ* schedule comes back), and the
  target is ABSOLUTE — nothing is taken relative to the anchor;
* the anchor actuator is the target the observed lookback implies, ``γ + τ_γ·γ̇``, and its first
  two columns are the specific-force law's own numbers;
* the box, the neutral (a steady ~3° descent) and the probe are the contract's;
* the loop is closed: a held target is reached and HELD, and a command the load box cannot
  deliver is clipped rather than flown;
* the record stays in newtons with a resolved load factor, carries the path-angle command beside
  it, and says which contract it came from;
* the heading-rate term prices the load the loop resolved, not the γ* column;
* the config refuses the pairings not built (every command hook among them), every named recipe
  pins thrust-fraction, and the name moves only off the default.
"""

from __future__ import annotations

import math

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
    CONTROL_HOOK_OFF,
    CONTROL_IMITATION_TARGET_FITTED,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_SPECIFIC_FORCE,
    CONTROL_SPECIFIC_FORCE_PATH_ANGLE,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CONTROL_THRUST_FRACTION,
    PREDICTION_CONTROL,
    TSConfig,
    control_recipe,
    recipe_settings,
)
from ts_transformer.data.dataset import Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.inference.export import build_prediction_record
from ts_transformer.outputs.conditioning import CONDITION_WIDTH
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.strategy import ControlStrategy
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.outputs.dynamics.backends import (
    RolloutInputs,
    control_dynamics_backend,
)
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.dynamics.inverse import reference_controls
from aerodynamic_model.torch_lag_dynamics import path_angle_load_factor  # noqa: E402
from ts_transformer.outputs.envelope import (
    MAX_LOAD_FACTOR,
    MIN_LOAD_FACTOR,
    PATH_ANGLE_CONTRACT,
    PATH_ANGLE_TIME_CONSTANT_S,
    control_contract,
)
from ts_transformer.run_naming import run_display_name, run_slug

AIRPORT, RUNWAY = "KRDU", "05L"
LAG = {
    "control_dynamics_model": CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    "control_dynamics_backend": CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
}
PATH_ANGLE = {**LAG, "control_thrust_parameterization": CONTROL_SPECIFIC_FORCE_PATH_ANGLE}
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


def _rollout(config: TSConfig, controls: np.ndarray, offsets: np.ndarray, *,
             initial=INITIAL_STATE, horizon_s: float = HORIZON_S):
    inputs = RolloutInputs(
        initial_state=torch.tensor([initial], dtype=torch.float64),
        initial_controls=torch.tensor(controls[:1], dtype=torch.float64),
        controls=torch.tensor(controls, dtype=torch.float64).unsqueeze(0),
        segment_durations_s=torch.full(
            (1, len(controls)), horizon_s / len(controls), dtype=torch.float64
        ),
        aero_params=torch.tensor([AERO], dtype=torch.float64),
        frame_params=torch.tensor([FRAME], dtype=torch.float64),
        max_thrust_n=torch.tensor([MAX_THRUST_N], dtype=torch.float64),
    )
    return control_dynamics_backend(config).dense_rollout(
        inputs, torch.tensor(offsets, dtype=torch.float64).unsqueeze(0),
        torch.ones((1, len(offsets)), dtype=torch.bool), config,
    )


# ── the teacher is the inverse of this forward model ─────────────────────────


def test_a_path_angle_schedule_comes_back_through_its_own_inverse():
    """known γ* schedule -> lagged rollout -> dense reference -> inverse -> schedule back.

    The commands RAMP over segments only a few τ_γ long, so the path is still chasing its
    target where the package's own teacher samples (the segment midpoints) and the ``τ_γ·γ̇``
    term of §14.3 is doing work. The last assertion pins that term: an inverse that read the
    flown path angle as the target — τ_γ = 0 — would miss these commands by an order more than
    the tolerance (review §14.9, finding 9). Bank is held constant so its own step transient
    does not contaminate the early samples; everything stays inside the load box and the thrust
    range, so the inverse is exact up to the 2 s grid."""
    config = _config(**PATH_ANGLE, control_rollout_integrator_dt_s=0.1)
    horizon = 60.0
    n_segments = int(config.n_segments)
    progress = (np.arange(n_segments) + 0.5) / n_segments
    controls = np.column_stack((
        np.full(n_segments, -0.05),
        np.full(n_segments, np.deg2rad(10.0)),
        np.deg2rad(-1.0 - 12.0 * progress),
    ))
    offsets = np.arange(2.0, horizon + 1e-9, 2.0)
    rollout = _rollout(config, controls, offsets, horizon_s=horizon)
    times = np.concatenate(([0.0], offsets))
    states = np.concatenate(
        (np.array([INITIAL_STATE]), rollout.query_geodetic_states[0].numpy()), axis=0
    )
    recovered = reference_controls(
        states, times, config=config, aero_params=np.array(AERO), max_thrust_n=MAX_THRUST_N
    )
    midpoints = (np.arange(n_segments) + 0.5) * (horizon / n_segments)
    sampled = np.column_stack([np.interp(midpoints, times, recovered[:, k]) for k in range(3)])
    error = np.abs(sampled - controls)
    assert error[:, 0].max() < 1e-2, f"specific force (g): {error[:, 0]}"
    assert error[:, 1].max() < np.deg2rad(0.5), f"bank: {np.rad2deg(error[:, 1])}"
    assert error[:, 2].max() < np.deg2rad(0.3), f"path angle: {np.rad2deg(error[:, 2])}"
    # The τ_γ term is what recovers the command: reading the flown path angle as the target
    # instead (τ_γ = 0) misses by far more than the tolerance above.
    flown = np.interp(midpoints, times, states[:, 5])
    assert np.abs(flown - controls[:, 2]).max() > 2.0 * np.deg2rad(0.3), (
        f"the samples are settled; τ_γ is untested: {np.rad2deg(np.abs(flown - controls[:, 2]))}"
    )


def test_the_path_angle_target_is_absolute_and_carries_the_specific_force_columns():
    """The anchor actuator is ``γ + τ_γ·γ̇`` at the anchor — absolute, so it is NOT the
    specific-force law's load column shifted, and its first two columns ARE that law's."""
    config = _config(**PATH_ANGLE)
    series = _series(config, n_flights=1)[0]
    anchor = config.seq_len - 1
    path = dynamics_arrays(series, anchor, parameterization=CONTROL_SPECIFIC_FORCE_PATH_ANGLE,
                           condition_features=config.control_condition_features)
    force = dynamics_arrays(series, anchor, parameterization=CONTROL_SPECIFIC_FORCE,
                            condition_features=config.control_condition_features)
    assert path["initial_controls"][0] == pytest.approx(force["initial_controls"][0], abs=1e-12)
    assert path["initial_controls"][1] == pytest.approx(force["initial_controls"][1], abs=1e-12)
    gamma = float(path["initial_state"][5])
    bank = float(force["initial_controls"][1])
    load = float(force["initial_controls"][2])
    speed = float(path["initial_state"][3])
    path_rate = GRAVITY_MPS2 * (load * math.cos(bank) - math.cos(gamma)) / speed
    assert path["initial_controls"][2] == pytest.approx(
        gamma + PATH_ANGLE_TIME_CONSTANT_S * path_rate, abs=1e-9
    )
    assert np.allclose(path["control_lower"], PATH_ANGLE_CONTRACT.lower)
    assert np.allclose(path["control_upper"], PATH_ANGLE_CONTRACT.upper)


# ── the contract: box, neutral, probe ─────────────────────────────────────────


def test_an_untrained_path_angle_head_outputs_the_neutral():
    """The zeroed head starts every flight at the contract's neutral: a speed hold on a
    descent, wings level, and the teacher's own median path angle."""
    config = _config(**PATH_ANGLE)
    model = build_model(config).eval()
    contract = control_contract(CONTROL_SPECIFIC_FORCE_PATH_ANGLE)
    dynamics = {
        "condition": torch.randn(2, CONDITION_WIDTH),
        "control_lower": torch.tensor([contract.lower] * 2, dtype=torch.float32),
        "control_upper": torch.tensor([contract.upper] * 2, dtype=torch.float32),
    }
    with torch.no_grad():
        prediction = model(torch.randn(2, config.seq_len, config.enc_in), dynamics)
    expected = torch.tensor(contract.neutral).view(1, 1, 3).expand_as(prediction.controls)
    assert torch.allclose(prediction.controls, expected, atol=1e-4)


@pytest.mark.parametrize(
    "aero,mass,thrust",
    (((30.0, 2.2, 0.02, 0.04, 0.9, 0.1), 6_800.0, 22_000.0),
     ((122.6, 2.7, 0.02, 0.04, 0.9, 0.1), 62_000.0, 240_000.0),
     ((427.8, 2.4, 0.02, 0.04, 0.9, 0.1), 251_000.0, 1_026_000.0)),
    ids=("bizjet", "narrowbody", "widebody"),
)
def test_the_neutral_flies_a_steady_descent_on_every_airframe(aero, mass, thrust):
    """What the neutral MEANS, flown, on three airframes that span the cohort: from a level
    start the path settles on the neutral angle within a few τ_γ and stays there — the same
    angle for all three, which is the claim the constant's comment makes."""
    config = _config(**PATH_ANGLE, control_rollout_integrator_dt_s=0.1)
    contract = control_contract(CONTROL_SPECIFIC_FORCE_PATH_ANGLE)
    n_segments = int(config.n_segments)
    controls = np.tile(np.array(contract.neutral), (n_segments, 1))
    offsets = np.arange(5.0, HORIZON_S + 1e-9, 5.0)
    level = (*INITIAL_STATE[:5], 0.0, mass)
    inputs = RolloutInputs(
        initial_state=torch.tensor([level], dtype=torch.float64),
        initial_controls=torch.tensor(controls[:1], dtype=torch.float64),
        controls=torch.tensor(controls, dtype=torch.float64).unsqueeze(0),
        segment_durations_s=torch.full((1, n_segments), HORIZON_S / n_segments, dtype=torch.float64),
        aero_params=torch.tensor([aero], dtype=torch.float64),
        frame_params=torch.tensor([FRAME], dtype=torch.float64),
        max_thrust_n=torch.tensor([thrust], dtype=torch.float64),
    )
    states = control_dynamics_backend(config).dense_rollout(
        inputs, torch.tensor(offsets, dtype=torch.float64).unsqueeze(0),
        torch.ones((1, len(offsets)), dtype=torch.bool), config,
    ).query_geodetic_states[0].numpy()
    settled = states[offsets >= 5.0 * PATH_ANGLE_TIME_CONSTANT_S]
    assert np.allclose(settled[:, 5], contract.neutral[2], atol=np.deg2rad(0.1))
    # no zoom, no stall: the speed stays inside a band the neutral's own n_x explains
    assert settled[:, 3].min() > 70.0 and settled[:, 3].max() < 150.0


def test_a_target_the_load_box_cannot_deliver_is_clipped_not_flown():
    """The loop asks; the box decides. Both ends of the box, each reached by a command INSIDE
    the head's own γ* box, so this is a state the trained head can actually produce.

    The demand has to exceed the box for the test to test anything, which the review found the
    first version did not (§14.9, finding 10): `n = 1 + V·Δγ/(g·τ_γ)` needs Δγ > 19.8° at
    85 m/s to reach 2.0, so the flight is anchored steep and the target is the box's ceiling."""
    config = _config(**PATH_ANGLE, control_rollout_integrator_dt_s=0.05)
    law = control_contract(config.control_thrust_parameterization).law
    assert (law.min_load_factor, law.max_load_factor) == (MIN_LOAD_FACTOR, MAX_LOAD_FACTOR)
    contract = control_contract(CONTROL_SPECIFIC_FORCE_PATH_ANGLE)
    for gamma, target, bound in ((math.radians(-14.0), contract.upper[2], MAX_LOAD_FACTOR),
                                 (math.radians(9.0), contract.lower[2], MIN_LOAD_FACTOR)):
        speed = 85.0
        demanded = math.cos(gamma) + speed * (target - gamma) / (
            GRAVITY_MPS2 * PATH_ANGLE_TIME_CONSTANT_S
        )
        assert not MIN_LOAD_FACTOR <= demanded <= MAX_LOAD_FACTOR, (
            f"the demand {demanded:.2f} is inside the box; this case clips nothing"
        )
        resolved = float(path_angle_load_factor(
            torch.tensor(target), torch.tensor(math.sin(gamma)), torch.tensor(speed),
            torch.tensor(0.0), PATH_ANGLE_TIME_CONSTANT_S, MIN_LOAD_FACTOR, MAX_LOAD_FACTOR,
        ))
        assert resolved == pytest.approx(bound)
    # …and flown, the path rate stays under what that ceiling allows.
    n_segments = int(config.n_segments)
    controls = np.tile(np.array([-0.05, 0.0, contract.upper[2]]), (n_segments, 1))
    offsets = np.arange(1.0, 30.0 + 1e-9, 1.0)
    steep = (*INITIAL_STATE[:3], 85.0, INITIAL_STATE[4], math.radians(-14.0), INITIAL_STATE[6])
    states = _rollout(config, controls, offsets, initial=steep).query_geodetic_states[0].numpy()
    ceiling_dps = math.degrees(GRAVITY_MPS2 * (MAX_LOAD_FACTOR - math.cos(states[0, 5])) / 85.0)
    rate_dps = np.max(np.abs(np.gradient(np.degrees(states[:, 5]), offsets)))
    assert rate_dps <= ceiling_dps * 1.05, f"{rate_dps} deg/s over the box's {ceiling_dps}"
    assert np.all(np.isfinite(states))


def test_the_probe_is_the_contract_s():
    probe = probe_dynamics(2, torch.device("cpu"), _config(**PATH_ANGLE))
    assert np.allclose(probe["control_lower"][0].numpy(), PATH_ANGLE_CONTRACT.lower)
    assert np.allclose(probe["control_upper"][0].numpy(), PATH_ANGLE_CONTRACT.upper)


# ── the record ────────────────────────────────────────────────────────────────


def test_the_record_carries_a_resolved_load_and_the_path_angle_command():
    """The record's third column is the load the loop resolved where each segment begins, the
    command rides beside it under its own contract name, and the record says which contract it
    came from."""
    config = _config(**PATH_ANGLE)
    series = _series(config, n_flights=1)
    model = build_model(config).eval()
    forecasts = forecast_control_batch(
        model, series, config, Normalizer.fit(series), config.seq_len - 1, torch.device("cpu")
    )
    forecast = forecasts[0]
    assert forecast.control_parameterization == CONTROL_SPECIFIC_FORCE_PATH_ANGLE
    assert len(forecast.commands) == len(forecast.controls) == config.n_segments
    record = build_prediction_record(
        series[0], forecast, index=0, model_name=config.model, horizon_mode=config.horizon_mode
    )
    segments = record.states_payload["control_segments"]
    assert [s["path_angle_command"] for s in segments] == pytest.approx(list(forecast.commands[:, 2]))
    assert [s["specific_force"] for s in segments] == pytest.approx(list(forecast.commands[:, 0]))
    loads = np.array([s["load_factor"] for s in segments])
    assert np.all(loads >= MIN_LOAD_FACTOR) and np.all(loads <= MAX_LOAD_FACTOR)
    # …and it is the LAW's own resolver at each segment's start state, not a constant that
    # happens to look like a load (review §14.9, finding 14).
    start_states = [next(s for s in record.states_payload["predicted_states"]
                         if s["t"] >= segment["start_t"] - 1e-9) for segment in segments]
    expected = [float(path_angle_load_factor(
        torch.tensor(segment["path_angle_command"]), torch.tensor(math.sin(state["gamma"])),
        torch.tensor(state["V"]), torch.tensor(segment["bank_rad"]), PATH_ANGLE_TIME_CONSTANT_S,
        MIN_LOAD_FACTOR, MAX_LOAD_FACTOR,
    )) for segment, state in zip(segments, start_states)]
    assert loads == pytest.approx(expected, abs=2e-3)
    assert record.eval_record["source"]["controlThrustParameterization"] == (
        CONTROL_SPECIFIC_FORCE_PATH_ANGLE
    )


# ── the heading-rate term reads the load the loop resolved ─────────────────────


def test_the_heading_rate_term_prices_the_load_the_path_loop_flew():
    """Under this contract the third actuator is γ*, so the term must read the load the loop
    RESOLVED at each endpoint (`_ControlLaw.geodetic_load`) — exactly the ψ row at that load —
    and γ* must reach the turn rate (through the lift) with a finite gradient."""
    from aerodynamic_model.torch_dynamics import heading_rate_rad_s
    from ts_transformer.outputs.control.loss.objective import control_heading_rate_mse
    from ts_transformer.outputs.dynamics.backends import EndpointControlRollout

    config = _config(**PATH_ANGLE, control_heading_rate_loss_weight=8.0)
    generator = torch.Generator().manual_seed(11)
    batch, n = 2, int(config.n_segments)
    states = torch.tensor([INITIAL_STATE] * batch, dtype=torch.float64).unsqueeze(1).repeat(1, n, 1)
    states[..., 3] += torch.rand((batch, n), generator=generator, dtype=torch.float64) * 20.0
    states[..., 5] = torch.deg2rad(-4.0 + 3.0 * torch.rand((batch, n), generator=generator, dtype=torch.float64))
    actual = torch.stack((
        torch.full((batch, n), -0.05, dtype=torch.float64),
        0.4 * torch.rand((batch, n), generator=generator, dtype=torch.float64) - 0.2,
        torch.deg2rad(-5.0 + 4.0 * torch.rand((batch, n), generator=generator, dtype=torch.float64)),
    ), dim=-1).requires_grad_(True)
    dynamics = {
        "aero_params": torch.tensor([AERO] * batch, dtype=torch.float64),
        "max_thrust_n": torch.full((batch,), MAX_THRUST_N, dtype=torch.float64),
        "initial_state": torch.tensor([INITIAL_STATE] * batch, dtype=torch.float64),
        "reference_heading_rate_dps": torch.zeros((batch, n), dtype=torch.float64),
        "reference_heading_rate_weight": torch.ones((batch, n), dtype=torch.float64),
    }
    rollout = EndpointControlRollout(
        channels=torch.zeros((batch, n, 6)), geodetic_states=states, controls=actual.detach(),
        actual_controls=actual,
    )
    loss = control_heading_rate_mse(rollout, config, dynamics)
    load = path_angle_load_factor(
        actual[..., 2], torch.sin(states[..., 5]), states[..., 3], actual[..., 1],
        PATH_ANGLE_TIME_CONSTANT_S, MIN_LOAD_FACTOR, MAX_LOAD_FACTOR,
    )
    expected_dps = torch.rad2deg(heading_rate_rad_s(
        states, torch.stack((torch.zeros_like(load), actual[..., 1], load), dim=-1),
        dynamics["aero_params"].unsqueeze(-2),
    ))
    expected = ((expected_dps / config.control_heading_rate_loss_scale_dps).square()).mean(dim=1)
    assert torch.allclose(loss, expected, rtol=1e-12, atol=0.0)
    # the γ* column is NOT read as a load: a load of ~−0.07 would turn the other way
    assert (load > 0.5).all()
    loss.sum().backward()
    assert torch.isfinite(actual.grad).all() and actual.grad[..., 2].abs().sum() > 0.0


# ── what the axis must not disturb ────────────────────────────────────────────


def test_the_contract_identity_is_spelled_into_the_target_contract():
    config = _config(**PATH_ANGLE)
    contract = ControlStrategy().target_contract(config)
    # verbatim as the stored N7 checkpoints carry it
    assert contract.endswith("+specific-force+path-angle(tau-gamma=3s,box=-15..10deg,neutral=-2.9deg)-v1")
    assert contract.endswith(PATH_ANGLE_CONTRACT.identity_suffix)
    assert f"tau-gamma={PATH_ANGLE_CONTRACT.law.path_angle_time_constant_s:g}s" in contract
    assert "path-angle" not in ControlStrategy().target_contract(_config(**SPECIFIC_FORCE))


def test_the_config_refuses_what_is_not_built():
    with pytest.raises(ValueError, match="first-order-lag"):
        _config(**{**PATH_ANGLE, "control_dynamics_model": CONTROL_DYNAMICS_POINT_MASS})
    with pytest.raises(ValueError, match="does not say which coordinate"):
        _config(**PATH_ANGLE, control_imitation_target=CONTROL_IMITATION_TARGET_FITTED,
                control_fitted_teacher_path="teacher.json", control_imitation_loss_weight=1.0)
    for hook in CONTROL_HOOKS_AVAILABLE:
        if hook == CONTROL_HOOK_OFF:
            continue
        with pytest.raises(ValueError, match="load-factor column"):
            _config(**PATH_ANGLE, control_command_hook=hook)


def test_every_named_recipe_still_pins_thrust_fraction():
    """A named recipe is frozen: the new value is reachable only from `custom`."""
    for name in CONTROL_RECIPE_NAMES:
        if name == CONTROL_RECIPE_CUSTOM:
            continue
        settings = recipe_settings(name, keep_name=True)
        assert settings["control_thrust_parameterization"] == CONTROL_THRUST_FRACTION, name
        with pytest.raises(ValueError, match="recipe fields are frozen"):
            TSConfig(**{**settings, **PATH_ANGLE})
    TSConfig(**{**recipe_settings("simple-v3", keep_name=False), **PATH_ANGLE})


def test_the_name_and_the_recipe_dict_move_only_off_the_default():
    default = _config(**LAG)
    moved = _config(**PATH_ANGLE)
    assert "path-angle" not in run_display_name(default.to_dict())
    assert "specific-force+path-angle" in run_display_name(moved.to_dict())
    assert "lag-sfpa" in run_slug(moved.to_dict())
    assert "thrust_parameterization" not in control_recipe(default)
    assert control_recipe(moved)["thrust_parameterization"] == CONTROL_SPECIFIC_FORCE_PATH_ANGLE
