"""The lag model's two control laws (ts_transformer/docs/2026-09-14_specific_force_control_design.md).

Under the specific-force law the RHS re-solves the thrust at every stage as
``clamp(W·a_x + D, T_lo, T_max)`` and subtracts the same drag, so the speed equation must
read ``V' = g·(a_x - sin γ)`` wherever the clamp does not bind — for every airframe — and
must read exactly the thrust-fraction law's where it does. The thrust-fraction law is pinned
to its own composition so the refactor that made room for the second law cannot move it.
Since the laws share one RHS (2026-09-16), a law is exercised through that RHS and its own two
resolvers, and the geodetic reading the record uses is pinned to the one the RHS flies.
"""

from __future__ import annotations

import math

import pytest
import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, flight_aerodynamics
from aerodynamic_model.torch_lag_dynamics import (
    UNSCALED_CHART,
    THRUST_FRACTION_LAW,
    SpecificForceLaw,
    lag_rhs,
    lag_state_from_geodetic,
    lag_state_scale,
    lag_step_context,
    rollout_piecewise_constant,
    rollout_piecewise_constant_at_times,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    transport_chart_kinematics,
    transport_chart_rhs,
)

FRAME = torch.tensor([[35.9, -78.8, 120.0, 0.3]], dtype=torch.float64)
TAU = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float64)
MIN_FRACTION = -0.2
#: Three airframes spanning the fleet: a light bizjet, the median narrowbody, a widebody.
#: ``(S, Cl_max, Cd0, k, stall_threshold, k_stall)``, landing mass, installed thrust.
AIRFRAMES = (
    ((30.0, 2.2, 0.02, 0.04, 0.9, 0.1), 6_800.0, 22_000.0),
    ((124.6, 2.7, 0.02, 0.04, 0.9, 0.1), 66_300.0, 233_000.0),
    ((427.8, 2.4, 0.02, 0.04, 0.9, 0.1), 251_000.0, 1_026_000.0),
)


def _geodetic(airframe, *, speed=85.0, gamma=-0.05, altitude=900.0):
    return torch.tensor(
        [[35.95, -78.75, altitude, speed, 1.1, gamma, airframe[1]]], dtype=torch.float64
    )


def _lag_state(airframe, *, speed=85.0, gamma=-0.05, altitude=900.0, actuators):
    geodetic = _geodetic(airframe, speed=speed, gamma=gamma, altitude=altitude)
    scale = lag_state_scale(UNSCALED_CHART, FRAME)
    state = lag_state_from_geodetic(
        geodetic, torch.tensor([actuators], dtype=torch.float64), FRAME, scale
    )
    return state, torch.tensor([airframe[0]], dtype=torch.float64), scale


def _rhs(law, state, commands, aero, thrust, scale):
    context = lag_step_context(law, FRAME, TAU, thrust, scale, state.new_zeros((1, 7)))
    return lag_rhs(state, commands, aero, context, law)


def _speed_rate(state: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """``d|v|/dt`` from the chart velocity and its rate; the transport term is ⊥ v."""
    velocity, velocity_rate = state[..., 3:6], rate[..., 3:6]
    return (velocity * velocity_rate).sum(-1) / velocity.norm(dim=-1)


@pytest.mark.parametrize("airframe", AIRFRAMES, ids=("bizjet", "narrowbody", "widebody"))
@pytest.mark.parametrize("specific_force", (-0.1, -0.04, 0.0, 0.05))
def test_drag_cancels_so_the_specific_force_moves_every_airframe_alike(airframe, specific_force):
    state, aero, scale = _lag_state(airframe, actuators=(specific_force, 0.2, 1.05))
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    commands = state[..., 7:]           # actuators at their commands: the lag rows are zero
    rate = _rhs(SpecificForceLaw(MIN_FRACTION), state, commands, aero, thrust, scale)
    sin_gamma = state[..., 5] / state[..., 3:6].norm(dim=-1)
    expected = GRAVITY_MPS2 * (specific_force - sin_gamma)
    assert _speed_rate(state, rate).item() == pytest.approx(expected.item(), rel=1e-9, abs=1e-12)
    assert torch.equal(rate[..., 7:], torch.zeros_like(rate[..., 7:]))


@pytest.mark.parametrize("airframe", AIRFRAMES, ids=("bizjet", "narrowbody", "widebody"))
@pytest.mark.parametrize("specific_force,fraction", ((5.0, 1.0), (-5.0, MIN_FRACTION)))
def test_where_the_clamp_binds_the_law_is_the_thrust_fractions_at_that_bound(
    airframe, specific_force, fraction
):
    """The two laws admit the same thrusts: n_x past either end flies δ = 1 or δ = -0.2."""
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    sf_state, aero, scale = _lag_state(airframe, actuators=(specific_force, 0.2, 1.05))
    tf_state, _aero, _scale = _lag_state(airframe, actuators=(fraction, 0.2, 1.05))
    sf_rate = _rhs(SpecificForceLaw(MIN_FRACTION), sf_state, sf_state[..., 7:], aero, thrust, scale)
    tf_rate = _rhs(THRUST_FRACTION_LAW, tf_state, tf_state[..., 7:], aero, thrust, scale)
    assert torch.allclose(sf_rate[..., :7], tf_rate[..., :7], rtol=0.0, atol=1e-12)


def test_the_thrust_fraction_law_is_its_own_composition_unchanged():
    """``lag_rhs`` IS the chart RHS at ``T = δ·T_max`` plus the actuators' ODE, bit for bit."""
    airframe = AIRFRAMES[1]
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    state, aero, scale = _lag_state(airframe, actuators=(0.13, -0.3, 1.1))
    commands = torch.tensor([[0.3, 0.1, 1.0]], dtype=torch.float64)
    actual = state[..., 7:]
    physical = torch.cat((actual[..., :1] * thrust.unsqueeze(-1), actual[..., 1:]), dim=-1)
    expected = torch.cat(
        (transport_chart_rhs(state[..., :7], physical, aero, FRAME), (commands - actual) / TAU),
        dim=-1,
    )
    assert torch.equal(_rhs(THRUST_FRACTION_LAW, state, commands, aero, thrust, scale), expected)


def test_the_geodetic_reading_resolves_the_thrust_the_rhs_flies():
    """The record and the heading-rate loss read a law off a GEODETIC state
    (``geodetic_controls``); it must resolve what the RHS resolves at the same chart state."""
    airframe = AIRFRAMES[2]
    law = SpecificForceLaw(MIN_FRACTION)
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    actual = torch.tensor([[-0.03, 0.2, 1.2]], dtype=torch.float64)
    geodetic = _geodetic(airframe, speed=92.0, altitude=1500.0)
    state, aero, scale = _lag_state(airframe, speed=92.0, altitude=1500.0, actuators=actual[0].tolist())
    kinematics = transport_chart_kinematics(state[..., :7], FRAME)
    parameters = law.parameter_matrix(thrust, geodetic, dtype=torch.float64, device=torch.device("cpu"))
    drag = flight_aerodynamics(kinematics.condition, actual[..., 2], aero).drag_n
    via_chart = law.resolve_thrust(kinematics.condition, actual, drag, thrust, parameters)
    via_geodetic = law.geodetic_controls(
        geodetic.unsqueeze(1), actual.unsqueeze(1), max_thrust_n=thrust,
        initial_geodetic_states=geodetic, aero_params=aero,
    )
    assert torch.allclose(via_geodetic[:, 0, 0], via_chart, rtol=1e-12, atol=0.0)
    assert torch.equal(via_geodetic[0, 0, 1:], actual[0, 1:])


def _schedule(n_segments=6):
    progress = (torch.arange(n_segments, dtype=torch.float64) + 0.5) / n_segments
    return torch.stack(
        (-0.05 + 0.03 * torch.sin(2 * math.pi * progress),
         0.2 * torch.sin(math.pi * progress),
         1.0 + 0.03 * torch.cos(2 * math.pi * progress)),
        dim=-1,
    ).unsqueeze(0)


def _rollout_args(controls, device="cpu"):
    aero, mass, thrust = AIRFRAMES[1]
    return dict(
        chart_scale=UNSCALED_CHART,
        initial_geodetic_states=torch.tensor(
            [[35.95, -78.75, 900.0, 85.0, 1.1, -0.05, mass]], dtype=torch.float64, device=device
        ),
        initial_controls=controls[:, 0].to(device),
        commands=controls.to(device),
        segment_durations_s=torch.full(controls.shape[:2], 10.0, dtype=torch.float64, device=device),
        aero_params=torch.tensor([aero], dtype=torch.float64, device=device),
        frame_params=FRAME.to(device),
        time_constants_s=TAU.to(device),
        max_thrust_n=torch.tensor([thrust], dtype=torch.float64, device=device),
    )


#: The last segment-end state of the thrust-fraction rollout below, computed by the code
#: BEFORE the control laws existed (commit 775b59e) — and bit-identical on this branch when
#: it was pinned. The tolerance only absorbs platform float noise; a moved law misses by
#: orders of magnitude more.
_THRUST_FRACTION_GOLDEN = (
    4685.750582414903, 10800.245627847104, 372.13569857235234, -39.39653876235639,
    102.6278141165774, -10.678724190889788, 66300.0, 0.09674904587908417,
    0.05237038891475323, 1.0259806626580166,
)


def test_the_thrust_fraction_law_reproduces_the_rollout_from_before_the_laws():
    controls = _schedule()
    controls[..., 0] = 0.1 + 0.05 * controls[..., 0]
    for law in ({}, {"control_law": THRUST_FRACTION_LAW}):
        ends = rollout_piecewise_constant(**_rollout_args(controls), **law)
        assert ends[0, -1].tolist() == pytest.approx(_THRUST_FRACTION_GOLDEN, rel=1e-12, abs=1e-12)


def test_the_specific_force_rollout_flies_the_commanded_speed_rate():
    """Across a whole rollout (lag included) the speed rate is g·(a_x - sin γ), where the
    actuator a_x is read off the rolled state itself."""
    controls = _schedule()
    args = _rollout_args(controls)
    law = SpecificForceLaw(min_thrust_fraction=MIN_FRACTION)
    spacing = 0.25
    queries = torch.arange(spacing, 60.0, spacing, dtype=torch.float64).unsqueeze(0)
    dense = rollout_piecewise_constant_at_times(
        **args, query_offsets_s=queries, query_valid=torch.ones_like(queries, dtype=torch.bool),
        control_law=law, integrator_dt_s=0.05,
    )
    states = dense.query_states[0]
    velocity = states[:, 3:6]
    speed = velocity.norm(dim=-1)
    finite_difference = torch.gradient(speed, spacing=(queries[0],))[0]
    expected = GRAVITY_MPS2 * (states[:, 7] - states[:, 5] / speed)
    # A central difference across a command switch sees the actuator's derivative kink
    # there (the lag keeps a_x continuous, not its rate): score the stencils inside a hold.
    boundaries = torch.cumsum(args["segment_durations_s"][0], dim=0)
    clear = (queries[0].unsqueeze(-1) - boundaries).abs().min(dim=-1).values > 1.5 * spacing
    clear[0] = clear[-1] = False
    assert (finite_difference - expected)[clear].abs().max().item() < 1e-3


def test_a_specific_force_law_refuses_a_floor_at_or_above_the_ceiling():
    with pytest.raises(ValueError, match="below the installed thrust"):
        SpecificForceLaw(min_thrust_fraction=1.0)
    with pytest.raises(ValueError, match="below the installed thrust"):
        SpecificForceLaw(min_thrust_fraction=float("nan"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile path")
def test_the_specific_force_step_compiles_and_backpropagates_on_cuda():
    controls = _schedule().requires_grad_(True)
    args = _rollout_args(controls.detach(), device="cuda")
    args["commands"] = controls.cuda()
    law = SpecificForceLaw(min_thrust_fraction=MIN_FRACTION)
    ends = rollout_piecewise_constant(**args, control_law=law)
    cpu = rollout_piecewise_constant(**_rollout_args(controls.detach()), control_law=law)
    assert torch.allclose(ends.detach().cpu(), cpu, rtol=1e-9, atol=1e-6)
    ends[..., 3].sum().backward()
    assert torch.isfinite(controls.grad).all() and controls.grad[..., 0].abs().sum() > 0.0
