"""The lag model's speed-command law (ts_transformer/docs/2026-09-14_specific_force_control_design.md §12).

Under :class:`SpeedCommandLaw` the first actuator is a speed command relative to the anchor
airspeed, and the RHS asks for the specific force ``sin γ + (V₀ + a_Δ − V)/(g·τ_V)``, flown
through the specific-force law's own thrust. So wherever the clamp does not bind the speed
equation must read ``V' = (V₀ + a_Δ − V)/τ_V`` — for every airframe — and where it binds it
must read the thrust-fraction law's at that bound. A rollout under a constant command must
settle on the command, which is the restoring force the specific-force law lacks.
"""

from __future__ import annotations

import math

import pytest
import torch

from aerodynamic_model.torch_lag_dynamics import (
    UNSCALED_CHART,
    SpecificForceLaw,
    SpeedCommandLaw,
    THRUST_FRACTION_LAW,
    lag_rhs,
    lag_state_from_geodetic,
    lag_state_scale,
    lag_step_context,
    rollout_piecewise_constant,
    rollout_piecewise_constant_hooked,
    rollout_piecewise_constant_at_times,
)

FRAME = torch.tensor([[35.9, -78.8, 120.0, 0.3]], dtype=torch.float64)
TAU = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float64)
MIN_FRACTION = -0.2
TAU_V = 8.0
AIRFRAMES = (
    ((30.0, 2.2, 0.02, 0.04, 0.9, 0.1), 6_800.0, 22_000.0),
    ((124.6, 2.7, 0.02, 0.04, 0.9, 0.1), 66_300.0, 233_000.0),
    ((427.8, 2.4, 0.02, 0.04, 0.9, 0.1), 251_000.0, 1_026_000.0),
)


def _lag_state(airframe, *, speed=85.0, gamma=-0.05, actuators):
    aero, mass, _thrust = airframe
    geodetic = torch.tensor([[35.95, -78.75, 900.0, speed, 1.1, gamma, mass]], dtype=torch.float64)
    scale = lag_state_scale(UNSCALED_CHART, FRAME)
    state = lag_state_from_geodetic(geodetic, torch.tensor([actuators], dtype=torch.float64), FRAME, scale)
    return state, torch.tensor([aero], dtype=torch.float64), scale


def _rhs(law, state, aero, thrust, scale, *, reference_speed=85.0):
    """The one lag RHS; the speed loop's reference is the initial state's airspeed."""
    initial = state.new_zeros((1, 7))
    initial[0, 3] = reference_speed
    return lag_rhs(state, state[..., 7:], aero, lag_step_context(law, FRAME, TAU, thrust, scale, initial), law)


def _speed_rate(state: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    velocity, velocity_rate = state[..., 3:6], rate[..., 3:6]
    return (velocity * velocity_rate).sum(-1) / velocity.norm(dim=-1)


@pytest.mark.parametrize("airframe", AIRFRAMES, ids=("bizjet", "narrowbody", "widebody"))
@pytest.mark.parametrize("reference,command", ((85.0, -3.0), (90.0, -5.0), (80.0, 4.0), (85.0, 0.0)))
def test_the_speed_loop_moves_every_airframe_toward_its_command_alike(airframe, reference, command):
    state, aero, scale = _lag_state(airframe, actuators=(command, 0.2, 1.05))
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    rate = _rhs(SpeedCommandLaw(MIN_FRACTION, TAU_V), state, aero, thrust, scale, reference_speed=reference)
    speed = state[..., 3:6].norm(dim=-1)
    expected = (reference + command - speed) / TAU_V
    assert _speed_rate(state, rate).item() == pytest.approx(expected.item(), rel=1e-9, abs=1e-12)
    assert torch.equal(rate[..., 7:], torch.zeros_like(rate[..., 7:]))


@pytest.mark.parametrize("airframe", AIRFRAMES, ids=("bizjet", "narrowbody", "widebody"))
@pytest.mark.parametrize("command,fraction", ((500.0, 1.0), (-500.0, MIN_FRACTION)))
def test_where_the_clamp_binds_the_loop_flies_the_thrust_fractions_bound(airframe, command, fraction):
    """The loop admits the thrust-fraction law's thrusts: a command far past either end
    flies δ = 1 or δ = −0.2."""
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    sc_state, aero, scale = _lag_state(airframe, actuators=(command, 0.2, 1.05))
    tf_state, _aero, _scale = _lag_state(airframe, actuators=(fraction, 0.2, 1.05))
    sc_rate = _rhs(SpeedCommandLaw(MIN_FRACTION, TAU_V), sc_state, aero, thrust, scale)
    tf_rate = _rhs(THRUST_FRACTION_LAW, tf_state, aero, thrust, scale)
    assert torch.allclose(sc_rate[..., :7], tf_rate[..., :7], rtol=0.0, atol=1e-12)


def _rollout_args(commands, *, speed=85.0, device="cpu"):
    aero, mass, thrust = AIRFRAMES[1]
    return dict(
        chart_scale=UNSCALED_CHART,
        initial_geodetic_states=torch.tensor(
            [[35.95, -78.75, 900.0, speed, 1.1, -0.05, mass]], dtype=torch.float64, device=device
        ),
        initial_controls=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, device=device),
        commands=commands.to(device),
        segment_durations_s=torch.full(commands.shape[:2], 15.0, dtype=torch.float64, device=device),
        aero_params=torch.tensor([aero], dtype=torch.float64, device=device),
        frame_params=FRAME.to(device),
        time_constants_s=TAU.to(device),
        max_thrust_n=torch.tensor([thrust], dtype=torch.float64, device=device),
    )


def test_a_constant_command_settles_on_the_commanded_speed():
    """A −10 m/s step held for 120 s: the speed follows the actuator lag and the loop's τ_V
    in series, ``V(t) = V₀ + Δv·(1 − (τ_V e^{−t/τ_V} − τ_T e^{−t/τ_T})/(τ_V − τ_T))``, and
    ends on the command. Under the specific-force law the same flight has no such attractor."""
    commands = torch.tensor([[[-10.0, 0.0, 1.0]] * 8], dtype=torch.float64)
    law = SpeedCommandLaw(min_thrust_fraction=MIN_FRACTION, speed_time_constant_s=TAU_V)
    queries = torch.arange(1.0, 120.0, 1.0, dtype=torch.float64).unsqueeze(0)
    dense = rollout_piecewise_constant_at_times(
        **_rollout_args(commands), query_offsets_s=queries,
        query_valid=torch.ones_like(queries, dtype=torch.bool), control_law=law, integrator_dt_s=0.1,
    )
    speed = dense.query_states[0, :, 3:6].norm(dim=-1)
    t, tau_t = queries[0], TAU[0].item()
    expected = 85.0 - 10.0 * (1.0 - (TAU_V * torch.exp(-t / TAU_V) - tau_t * torch.exp(-t / tau_t)) / (TAU_V - tau_t))
    assert (speed - expected).abs().max().item() < 1e-3
    assert speed[-1].item() == pytest.approx(75.0, abs=0.01)


def _two_flights():
    aero, mass, thrust = AIRFRAMES[1]
    commands = torch.zeros((2, 4, 3), dtype=torch.float64)
    commands[..., 2] = 1.0
    return dict(
        chart_scale=UNSCALED_CHART,
        initial_geodetic_states=torch.tensor(
            [[35.95, -78.75, 900.0, 80.0, 1.1, -0.05, mass], [35.95, -78.75, 900.0, 95.0, 1.1, -0.05, mass]],
            dtype=torch.float64,
        ),
        initial_controls=torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=torch.float64),
        commands=commands,
        segment_durations_s=torch.full((2, 4), 10.0, dtype=torch.float64),
        aero_params=torch.tensor([aero] * 2, dtype=torch.float64),
        frame_params=FRAME.expand(2, -1),
        time_constants_s=TAU,
        max_thrust_n=torch.tensor([thrust] * 2, dtype=torch.float64),
        control_law=SpeedCommandLaw(min_thrust_fraction=MIN_FRACTION, speed_time_constant_s=TAU_V),
    )


def _endpoint_speeds(args):
    return rollout_piecewise_constant(**args)[:, -1, 3:6].norm(dim=-1)


def _hooked_speeds(args):
    ends, _flown = rollout_piecewise_constant_hooked(
        **args, command_hook=lambda state, command, duration, index, reference: command
    )
    return ends[:, -1, 3:6].norm(dim=-1)


def _dense_speeds(args):
    end = args["segment_durations_s"].sum(dim=-1, keepdim=True)
    dense = rollout_piecewise_constant_at_times(
        **args, query_offsets_s=end, query_valid=torch.ones_like(end, dtype=torch.bool)
    )
    return dense.query_states[:, -1, 3:6].norm(dim=-1)


@pytest.mark.parametrize("speeds", (_endpoint_speeds, _hooked_speeds, _dense_speeds),
                         ids=("endpoints", "hooked", "dense"))
def test_the_reference_speed_is_each_flights_own_anchor_speed(speeds):
    """Two flights at 80 and 95 m/s with the same zero command each hold their OWN speed —
    in every rollout the lag model offers (each packs its own step context)."""
    assert speeds(_two_flights()).tolist() == pytest.approx([80.0, 95.0], abs=1e-6)


def test_a_speed_command_law_refuses_a_bad_time_constant_or_floor():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="speed_time_constant_s"):
            SpeedCommandLaw(min_thrust_fraction=MIN_FRACTION, speed_time_constant_s=bad)
    with pytest.raises(ValueError, match="below the installed thrust"):
        SpeedCommandLaw(min_thrust_fraction=1.0, speed_time_constant_s=TAU_V)
    SpecificForceLaw(min_thrust_fraction=MIN_FRACTION)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile path")
def test_the_speed_command_step_compiles_and_backpropagates_on_cuda():
    commands = torch.tensor([[[-6.0, 0.1, 1.0], [-10.0, -0.1, 1.0], [-12.0, 0.0, 1.0]]], dtype=torch.float64)
    law = SpeedCommandLaw(min_thrust_fraction=MIN_FRACTION, speed_time_constant_s=TAU_V)
    leaf = commands.clone().cuda().requires_grad_(True)
    args = _rollout_args(commands, device="cuda")
    args["commands"] = leaf
    ends = rollout_piecewise_constant(**args, control_law=law)
    cpu = rollout_piecewise_constant(**_rollout_args(commands), control_law=law)
    assert torch.allclose(ends.detach().cpu(), cpu, rtol=1e-9, atol=1e-6)
    ends[..., 3].sum().backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad[..., 0].abs().sum() > 0.0
    assert math.isfinite(float(ends[0, -1, 3:6].norm()))
