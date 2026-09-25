"""The lag model's path-angle law (ts_transformer/docs/2026-09-14_specific_force_control_design.md §14).

Under :class:`PathAngleLaw` the THIRD actuator is a path-angle target and the RHS re-solves the
load factor ``n = [cos γ + V(γ* − γ)/(g τ_γ)]/cos φ`` at every stage, flown through the
specific-force law's own thrust. So wherever neither the load box nor the stall clamp binds the
path equation must read ``γ' = (γ* − γ)/τ_γ`` — for every airframe — a rollout under a constant
target must settle on that target, and where the box binds the law must fly the box's own load,
not the one the target asks for.
"""

from __future__ import annotations

import math

import pytest
import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from aerodynamic_model.torch_lag_dynamics import (
    UNSCALED_CHART,
    PathAngleLaw,
    SpecificForceLaw,
    lag_rhs,
    lag_state_from_geodetic,
    lag_state_scale,
    lag_step_context,
    rollout_piecewise_constant,
    rollout_piecewise_constant_at_times,
)

FRAME = torch.tensor([[35.9, -78.8, 120.0, 0.3]], dtype=torch.float64)
TAU = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float64)
MIN_FRACTION = -0.2
TAU_GAMMA = 3.0
LOAD_BOX = (0.2, 2.0)
AIRFRAMES = (
    ((30.0, 2.2, 0.02, 0.04, 0.9, 0.1), 6_800.0, 22_000.0),
    ((124.6, 2.7, 0.02, 0.04, 0.9, 0.1), 66_300.0, 233_000.0),
    ((427.8, 2.4, 0.02, 0.04, 0.9, 0.1), 251_000.0, 1_026_000.0),
)


def _lag_state(airframe, *, speed=85.0, gamma=-0.05, actuators):
    aero, mass, _thrust = airframe
    geodetic = torch.tensor([[35.95, -78.75, 900.0, speed, 1.1, gamma, mass]], dtype=torch.float64)
    scale = lag_state_scale(UNSCALED_CHART, FRAME)
    state = lag_state_from_geodetic(
        geodetic, torch.tensor([actuators], dtype=torch.float64), FRAME, scale
    )
    return state, torch.tensor([aero], dtype=torch.float64), scale


def _path_rate(state: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """dγ/dt from the chart velocity and its rate, γ = asin(v_u / |v|)."""
    velocity, velocity_rate = state[..., 3:6], rate[..., 3:6]
    speed = velocity.norm(dim=-1)
    speed_rate = (velocity * velocity_rate).sum(-1) / speed
    return (velocity_rate[..., 2] * speed - velocity[..., 2] * speed_rate) / (
        speed * torch.sqrt(speed**2 - velocity[..., 2] ** 2)
    )


def _law_rhs(law, state, aero, scale, thrust):
    """The one lag RHS under ``law``, actuators at their commands."""
    context = lag_step_context(law, FRAME, TAU, thrust, scale, state.new_zeros((1, 7)))
    return lag_rhs(state, state[..., 7:], aero, context, law)


def _rhs(state, aero, scale, thrust):
    return _law_rhs(PathAngleLaw(MIN_FRACTION, TAU_GAMMA, *LOAD_BOX), state, aero, scale, thrust)


def _sf_rhs(state, aero, scale, thrust):
    return _law_rhs(SpecificForceLaw(MIN_FRACTION), state, aero, scale, thrust)


@pytest.mark.parametrize("target_deg", (-3.0, -2.0, -1.0, 0.0, 1.5))
def test_the_path_loop_moves_every_airframe_toward_its_target_alike(target_deg):
    """One state, three airframes: the path rate is the SAME number and it is the loop's
    ``(γ* − γ)/τ_γ``.

    Exactly the same to 1e-12 across airframes — that is the invariance — and within 0.3 % of
    the ideal rate, the WGS84 transport term the chart RHS adds and the flat-earth loop does
    not subtract (`PathAngleLaw`'s docstring: it moves the loop's fixed point by ~0.002° at
    85 m/s, three orders below the contract's 0.11° break-even)."""
    gamma = math.radians(-2.0)
    expected = (math.radians(target_deg) - gamma) / TAU_GAMMA
    rates = []
    for airframe in AIRFRAMES:
        state, aero, scale = _lag_state(
            airframe, gamma=gamma, actuators=(-0.05, 0.0, math.radians(target_deg))
        )
        thrust = torch.tensor([airframe[2]], dtype=torch.float64)
        rate = _rhs(state, aero, scale, thrust)
        assert torch.equal(rate[..., 7:], torch.zeros_like(rate[..., 7:]))
        rates.append(_path_rate(state, rate).item())
    assert rates[0] == pytest.approx(rates[1], rel=0.0, abs=1e-12)
    assert rates[0] == pytest.approx(rates[2], rel=0.0, abs=1e-12)
    if expected != 0.0:
        assert rates[0] == pytest.approx(expected, rel=3e-3)
    else:
        assert abs(rates[0]) < 2e-5


@pytest.mark.parametrize("airframe", AIRFRAMES, ids=("bizjet", "narrowbody", "widebody"))
def test_a_target_the_box_cannot_deliver_flies_the_box(airframe):
    """A 30° pull asks for a load far past the ceiling; the law must fly the ceiling, which is
    the specific-force law's own RHS at that load — not the load the target implies."""
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    pa_state, aero, scale = _lag_state(airframe, actuators=(-0.05, 0.0, math.radians(30.0)))
    sf_state, _aero, _scale = _lag_state(airframe, actuators=(-0.05, 0.0, LOAD_BOX[1]))
    pa_rate = _rhs(pa_state, aero, scale, thrust)
    sf_rate = _sf_rhs(sf_state, aero, scale, thrust)
    assert torch.allclose(pa_rate[..., :7], sf_rate[..., :7], rtol=0.0, atol=1e-12)


def test_a_constant_target_settles_on_the_commanded_path_angle():
    """A −4° target held for 120 s from a −3° start: the path follows the actuator lag and the
    loop's τ_γ in series and ends on the target, with the speed still the specific force's."""
    aero, mass, thrust = AIRFRAMES[1]
    target = math.radians(-4.0)
    commands = torch.tensor([[[-0.05, 0.0, target]] * 8], dtype=torch.float64)
    queries = torch.arange(1.0, 120.0, 1.0, dtype=torch.float64).unsqueeze(0)
    dense = rollout_piecewise_constant_at_times(
        initial_geodetic_states=torch.tensor(
            [[35.95, -78.75, 2000.0, 85.0, 1.1, math.radians(-3.0), mass]], dtype=torch.float64
        ),
        initial_controls=torch.tensor([[-0.05, 0.0, math.radians(-3.0)]], dtype=torch.float64),
        commands=commands,
        segment_durations_s=torch.full((1, 8), 15.0, dtype=torch.float64),
        aero_params=torch.tensor([aero], dtype=torch.float64),
        frame_params=FRAME,
        time_constants_s=TAU,
        max_thrust_n=torch.tensor([thrust], dtype=torch.float64),
        query_offsets_s=queries,
        query_valid=torch.ones_like(queries, dtype=torch.bool),
        control_law=PathAngleLaw(MIN_FRACTION, TAU_GAMMA, *LOAD_BOX),
        integrator_dt_s=0.1, chart_scale=UNSCALED_CHART,
    )
    speed = dense.query_states[0, :, 3:6].norm(dim=-1)
    gamma = torch.asin(dense.query_states[0, :, 5] / speed)
    t, tau_load = queries[0], TAU[2].item()
    expected = math.radians(-3.0) + (target - math.radians(-3.0)) * (
        1.0
        - (TAU_GAMMA * torch.exp(-t / TAU_GAMMA) - tau_load * torch.exp(-t / tau_load))
        / (TAU_GAMMA - tau_load)
    )
    assert (gamma - expected).abs().max().item() < math.radians(0.02)
    assert gamma[-1].item() == pytest.approx(target, abs=math.radians(0.01))
    # the specific force still owns the speed, unchanged by the vertical law:
    # V' = g(n_x − sin γ) at every sample.
    speed_rate = torch.gradient(speed, spacing=(queries[0],))[0]
    expected_rate = GRAVITY_MPS2 * (-0.05 - torch.sin(gamma))
    # away from the one-sided difference at each end of the sampled window
    assert (speed_rate - expected_rate)[1:-1].abs().max().item() < 2e-3


def test_the_law_reduces_to_the_specific_force_law_when_the_target_is_the_current_path():
    """γ* = γ asks for the trim load `cos γ / cos φ`, so the two laws agree where the
    specific-force law flies that load: the path-angle law adds no vertical authority of its
    own, it only decides which load is asked for."""
    airframe = AIRFRAMES[1]
    gamma, bank = math.radians(-3.0), math.radians(12.0)
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    pa_state, aero, scale = _lag_state(airframe, gamma=gamma, actuators=(-0.05, bank, gamma))
    trim = math.cos(gamma) / math.cos(bank)
    sf_state, _aero, _scale = _lag_state(airframe, gamma=gamma, actuators=(-0.05, bank, trim))
    pa_rate = _rhs(pa_state, aero, scale, thrust)
    sf_rate = _sf_rhs(sf_state, aero, scale, thrust)
    assert torch.allclose(pa_rate[..., :7], sf_rate[..., :7], rtol=0.0, atol=1e-12)


def test_the_stall_clamp_still_decides_what_the_loop_gets():
    """A target whose load is INSIDE the box but past what the wing can make at this speed: the
    RHS's stall clamp must limit the flown path rate below the loop's own, and the same state
    flown at the stall-capped load must give the same rates.

    §14.2's claim is that feasibility is unchanged — the head may ask, the aircraft refuses.
    Nothing else in either test file exercises the clamp (review §14.9, finding 13)."""
    aero, mass, thrust_n = AIRFRAMES[1]
    slow, gamma = 62.0, math.radians(-3.0)     # ~1.1 x the 1-g stall speed of this airframe
    target = math.radians(4.0)
    thrust = torch.tensor([thrust_n], dtype=torch.float64)
    state, aero_t, scale = _lag_state(
        AIRFRAMES[1], speed=slow, gamma=gamma, actuators=(-0.05, 0.0, target)
    )
    demanded = math.cos(gamma) + slow * (target - gamma) / (GRAVITY_MPS2 * TAU_GAMMA)
    assert LOAD_BOX[0] < demanded < LOAD_BOX[1], f"the box clips first at n={demanded:.2f}"
    cl_max_load = (
        0.5 * 1.225 * (1.0 - 2.25577e-5 * 900.0) ** 5.25588 * slow**2 * aero[0] * aero[1]
        / (mass * GRAVITY_MPS2)
    )
    assert demanded > cl_max_load, f"the wing can make n={demanded:.2f} here; nothing to clamp"
    rate = _rhs(state, aero_t, scale, thrust)
    flown_dps = math.degrees(_path_rate(state, rate).item())
    loop_dps = math.degrees((target - gamma) / TAU_GAMMA)
    assert 0.0 < flown_dps < loop_dps, f"flown {flown_dps:.3f}, the loop asked {loop_dps:.3f}"
    # and it is the SAME as the specific-force law flying the capped load: the clamp, not the law
    capped, _aero, _scale = _lag_state(
        AIRFRAMES[1], speed=slow, gamma=gamma, actuators=(-0.05, 0.0, demanded)
    )
    sf_rate = _sf_rhs(capped, aero_t, scale, thrust)
    assert torch.allclose(rate[..., :7], sf_rate[..., :7], rtol=0.0, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compile path")
def test_the_path_angle_step_compiles_and_backpropagates_on_cuda():
    aero, mass, thrust = AIRFRAMES[1]
    commands = torch.tensor(
        [[[-0.05, 0.1, math.radians(-2.0)], [-0.06, -0.1, math.radians(-3.0)],
          [-0.05, 0.0, math.radians(-4.0)]]], dtype=torch.float64
    )
    law = PathAngleLaw(MIN_FRACTION, TAU_GAMMA, *LOAD_BOX)

    def args(device):
        return dict(
            chart_scale=UNSCALED_CHART,
            initial_geodetic_states=torch.tensor(
                [[35.95, -78.75, 2000.0, 85.0, 1.1, math.radians(-2.0), mass]],
                dtype=torch.float64, device=device,
            ),
            initial_controls=torch.tensor(
                [[-0.05, 0.0, math.radians(-2.0)]], dtype=torch.float64, device=device
            ),
            segment_durations_s=torch.full((1, 3), 15.0, dtype=torch.float64, device=device),
            aero_params=torch.tensor([aero], dtype=torch.float64, device=device),
            frame_params=FRAME.to(device),
            time_constants_s=TAU.to(device),
            max_thrust_n=torch.tensor([thrust], dtype=torch.float64, device=device),
        )

    leaf = commands.clone().cuda().requires_grad_(True)
    ends = rollout_piecewise_constant(**args("cuda"), commands=leaf, control_law=law)
    cpu = rollout_piecewise_constant(**args("cpu"), commands=commands, control_law=law)
    assert torch.allclose(ends.detach().cpu(), cpu, rtol=1e-9, atol=1e-6)
    ends[..., 5].sum().backward()
    assert torch.isfinite(leaf.grad).all() and leaf.grad[..., 2].abs().sum() > 0.0


def test_the_law_refuses_an_impossible_contract():
    SpecificForceLaw(MIN_FRACTION)          # the shared engine-floor contract
    with pytest.raises(ValueError, match="path_angle_time_constant_s"):
        PathAngleLaw(MIN_FRACTION, 0.0, *LOAD_BOX)
    with pytest.raises(ValueError, match="load box"):
        PathAngleLaw(MIN_FRACTION, TAU_GAMMA, 2.0, 0.2)
    with pytest.raises(ValueError, match="min_thrust_fraction"):
        PathAngleLaw(1.5, TAU_GAMMA, *LOAD_BOX)


def test_a_batch_flies_flight_by_flight():
    """Two flights of different airframes in one call fly what each flies alone."""
    (aero_a, mass_a, thrust_a), (aero_b, mass_b, thrust_b) = AIRFRAMES[0], AIRFRAMES[2]
    law = PathAngleLaw(MIN_FRACTION, TAU_GAMMA, *LOAD_BOX)
    commands = torch.tensor(
        [[[-0.05, 0.0, math.radians(-4.0)]] * 4, [[-0.04, 0.1, math.radians(-2.0)]] * 4],
        dtype=torch.float64,
    )
    states = torch.tensor(
        [[35.95, -78.75, 2000.0, 78.0, 1.1, math.radians(-3.0), mass_a],
         [35.95, -78.75, 2200.0, 92.0, 1.0, math.radians(-2.5), mass_b]],
        dtype=torch.float64,
    )
    queries = torch.arange(5.0, 60.0, 5.0, dtype=torch.float64).expand(2, -1).contiguous()
    kwargs = dict(
        chart_scale=UNSCALED_CHART,
        segment_durations_s=torch.full((2, 4), 15.0, dtype=torch.float64),
        aero_params=torch.tensor([aero_a, aero_b], dtype=torch.float64),
        frame_params=FRAME.expand(2, -1),
        time_constants_s=TAU,
        max_thrust_n=torch.tensor([thrust_a, thrust_b], dtype=torch.float64),
        query_offsets_s=queries,
        query_valid=torch.ones_like(queries, dtype=torch.bool),
        control_law=law,
        integrator_dt_s=0.1,
    )
    both = rollout_piecewise_constant_at_times(
        initial_geodetic_states=states, initial_controls=commands[:, 0], commands=commands, **kwargs
    )
    for row in (0, 1):
        alone = rollout_piecewise_constant_at_times(
            initial_geodetic_states=states[row : row + 1],
            initial_controls=commands[row : row + 1, 0],
            commands=commands[row : row + 1],
            **{
                **kwargs,
                "segment_durations_s": kwargs["segment_durations_s"][row : row + 1],
                "aero_params": kwargs["aero_params"][row : row + 1],
                "frame_params": kwargs["frame_params"][row : row + 1],
                "max_thrust_n": kwargs["max_thrust_n"][row : row + 1],
                "query_offsets_s": queries[row : row + 1],
                "query_valid": torch.ones_like(queries[row : row + 1], dtype=torch.bool),
            },
        )
        assert torch.allclose(
            both.query_states[row], alone.query_states[0], rtol=0.0, atol=1e-10
        )


def test_the_geodetic_reading_resolves_the_load_the_rhs_flies():
    """The heading-rate loss and the record resolve the path loop's load off a GEODETIC state
    (``geodetic_load``); it must be the load the RHS flies at the same chart state: the specific-
    force law flying that load gives the path-angle law's own rates."""
    airframe = AIRFRAMES[1]
    law = PathAngleLaw(MIN_FRACTION, TAU_GAMMA, *LOAD_BOX)
    gamma, bank, target = math.radians(-2.5), math.radians(15.0), math.radians(-4.0)
    thrust = torch.tensor([airframe[2]], dtype=torch.float64)
    state, aero, scale = _lag_state(airframe, gamma=gamma, actuators=(-0.05, bank, target))
    geodetic = torch.tensor([[[35.95, -78.75, 900.0, 85.0, 1.1, gamma, airframe[1]]]], dtype=torch.float64)
    load = law.geodetic_load(
        geodetic, state[:, None, 7:], max_thrust_n=thrust, initial_geodetic_states=geodetic[:, 0]
    )
    flown, _aero, _scale = _lag_state(airframe, gamma=gamma, actuators=(-0.05, bank, float(load)))
    assert torch.allclose(
        _rhs(state, aero, scale, thrust)[..., :7], _sf_rhs(flown, aero, scale, thrust)[..., :7],
        rtol=0.0, atol=1e-9,
    )

