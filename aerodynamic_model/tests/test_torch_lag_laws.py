"""The lag model's control-law PROTOCOL (2026-09-16): four laws, one RHS.

Each law used to own a copy of the RHS, the RK4 step, the context unpacking and the compile
entry points; now it owns only its two resolvers, its step-context columns and two compile entry
points, and everything else is shared. What that structure must not lose:

* every law's rollout is the one it flew before the laws shared an RHS (goldens captured on the
  per-law code, commit 82e27e8 — the thrust-fraction law's own golden predates the laws and lives
  in ``test_torch_lag_specific_force.py``);
* each law's compile entry points fly THAT law (they name their class; a slip would fly another
  law on CUDA only, where the suite rarely looks);
* the geodetic readers the record and the heading-rate loss use resolve, over a real batch, what
  each flight resolves alone — including the one per-flight column (the speed command's anchor
  airspeed) — and refuse a shape they would otherwise broadcast into nonsense.
"""

from __future__ import annotations

import math

import pytest
import torch

from aerodynamic_model.torch_lag_dynamics import (
    THRUST_FRACTION_LAW,
    PathAngleLaw,
    SpecificForceLaw,
    SpeedCommandLaw,
    lag_state_from_geodetic,
    lag_state_scale,
    lag_step_context,
    rk4_lag_step,
    rollout_piecewise_constant,
    rollout_piecewise_constant_at_times,
)

FRAME = torch.tensor([[35.9, -78.8, 120.0, 0.3]], dtype=torch.float64)
TAU = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float64)
AERO, MASS, THRUST = (124.6, 2.7, 0.02, 0.04, 0.9, 0.1), 66_300.0, 233_000.0
LAWS = {
    "thrust-fraction": THRUST_FRACTION_LAW,
    "specific-force": SpecificForceLaw(-0.2),
    "speed-command": SpeedCommandLaw(-0.2, 8.0),
    "path-angle": PathAngleLaw(-0.2, 3.0, 0.2, 2.0),
}


def _schedule(first, third):
    progress = (torch.arange(6, dtype=torch.float64) + 0.5) / 6
    return torch.stack(
        (first(progress), 0.2 * torch.sin(math.pi * progress), third(progress)), dim=-1
    ).unsqueeze(0)


#: ``(law, commands, last segment-end state)`` — computed by the per-law RHS of 82e27e8 and
#: bit-identical on the shared one. The tolerance only absorbs platform float noise.
GOLDENS = {
    "specific-force": (
        LAWS["specific-force"],
        _schedule(lambda p: -0.05 + 0.03 * torch.sin(2 * math.pi * p),
                  lambda p: 1.0 + 0.03 * torch.cos(2 * math.pi * p)),
        (4724.855600684948, 10649.05796164399, 391.50016395544066, -36.37642882270384,
         90.64233537073386, -9.262297150085574, 66300.0, -0.06501908241831658,
         0.05237038891475323, 1.0259806626580166),
    ),
    "speed-command": (
        LAWS["speed-command"],
        _schedule(lambda p: -8.0 + 4.0 * torch.sin(2 * math.pi * p),
                  lambda p: 1.0 + 0.03 * torch.cos(2 * math.pi * p)),
        (4649.657360686857, 10031.250358352607, 437.3317047535967, -34.85888476478415,
         65.77674464921543, -6.975821539497677, 66300.0, -10.002544322442212,
         0.05237038891475323, 1.0259806626580166),
    ),
    "path-angle": (
        LAWS["path-angle"],
        _schedule(lambda p: -0.05 + 0.03 * torch.sin(2 * math.pi * p),
                  lambda p: math.radians(-3.0) + math.radians(1.5) * torch.cos(2 * math.pi * p)),
        (4772.43889409788, 10657.679282160574, 495.2926768519652, -32.83166906716392,
         80.90345691539873, -2.688994754513843, 66300.0, -0.06501908241831658,
         0.05237038891475323, -0.02968747229821311),
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_every_law_flies_the_rollout_it_flew_before_the_laws_shared_an_rhs(name):
    law, commands, golden = GOLDENS[name]
    ends = rollout_piecewise_constant(
        torch.tensor([[35.95, -78.75, 900.0, 85.0, 1.1, -0.05, MASS]], dtype=torch.float64),
        commands[:, 0], commands, torch.full((1, 6), 10.0, dtype=torch.float64),
        torch.tensor([AERO], dtype=torch.float64), FRAME, TAU,
        torch.tensor([THRUST], dtype=torch.float64), control_law=law,
    )
    assert ends[0, -1].tolist() == pytest.approx(golden, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name", sorted(LAWS))
def test_a_laws_compile_entry_points_fly_that_law(name):
    """Run eagerly, both entry points equal the generic step under their own class — and differ
    from every other law's at a state where the laws disagree."""
    law = LAWS[name]
    geodetic = torch.tensor([[35.95, -78.75, 900.0, 85.0, 1.1, -0.05, MASS]], dtype=torch.float64)
    actuators = torch.tensor([[0.1, 0.2, -0.04]], dtype=torch.float64)
    scale = lag_state_scale(None, FRAME)
    state = lag_state_from_geodetic(geodetic, actuators, FRAME, scale)
    commands = torch.tensor([[0.15, 0.1, -0.05]], dtype=torch.float64)
    aero = torch.tensor([AERO], dtype=torch.float64)
    thrust = torch.tensor([THRUST], dtype=torch.float64)
    dt = torch.tensor([0.5], dtype=torch.float64)
    context = lag_step_context(law, FRAME, TAU, thrust, scale, geodetic)
    expected = rk4_lag_step(state, commands, aero, dt, context, type(law))
    assert torch.equal(type(law).step_inference(state, commands, aero, dt, context), expected)
    assert torch.equal(type(law).step_autograd(state, commands, aero, dt, context), expected)
    for other_name, other in LAWS.items():
        if other_name == name:
            continue
        other_context = lag_step_context(other, FRAME, TAU, thrust, scale, geodetic)
        assert not torch.equal(type(other).step_inference(state, commands, aero, dt, other_context), expected)


def _batch():
    """Two flights of different airframes and anchor speeds, four segment-start states each."""
    generator = torch.Generator().manual_seed(5)
    initial = torch.tensor(
        [[35.95, -78.75, 900.0, 85.0, 1.1, -0.05, MASS],
         [35.90, -78.70, 1800.0, 118.0, -0.4, -0.03, 180_000.0]], dtype=torch.float64,
    )
    states = initial.unsqueeze(1).repeat(1, 4, 1)
    states[..., 2] += 300.0 * torch.rand((2, 4), generator=generator, dtype=torch.float64)
    states[..., 3] += 15.0 * torch.rand((2, 4), generator=generator, dtype=torch.float64)
    states[..., 5] = torch.deg2rad(-4.0 + 3.0 * torch.rand((2, 4), generator=generator, dtype=torch.float64))
    actual = torch.stack((
        -0.1 + 0.2 * torch.rand((2, 4), generator=generator, dtype=torch.float64),
        -0.3 + 0.6 * torch.rand((2, 4), generator=generator, dtype=torch.float64),
        torch.deg2rad(-5.0 + 4.0 * torch.rand((2, 4), generator=generator, dtype=torch.float64)),
    ), dim=-1)
    aero = torch.tensor([AERO, (427.8, 2.4, 0.02, 0.04, 0.9, 0.1)], dtype=torch.float64)
    thrust = torch.tensor([THRUST, 800_000.0], dtype=torch.float64)
    return initial, states, actual, aero, thrust


@pytest.mark.parametrize("name", sorted(LAWS))
def test_the_geodetic_readers_resolve_each_flight_as_it_resolves_alone(name):
    law = LAWS[name]
    initial, states, actual, aero, thrust = _batch()
    both = law.geodetic_controls(
        states, actual, max_thrust_n=thrust, initial_geodetic_states=initial, aero_params=aero
    )
    loads = law.geodetic_load(states, actual, max_thrust_n=thrust, initial_geodetic_states=initial)
    assert both.shape == (2, 4, 3) and loads.shape == (2, 4)
    assert torch.equal(both[..., 2], loads)
    for row in (0, 1):
        alone = law.geodetic_controls(
            states[row : row + 1], actual[row : row + 1], max_thrust_n=thrust[row : row + 1],
            initial_geodetic_states=initial[row : row + 1], aero_params=aero[row : row + 1],
        )
        # equal up to the vectorised kernels' remainder-loop rounding (atan2, sqrt over 8 vs 4
        # elements), never a broadcasting error, which would be orders larger
        assert torch.allclose(both[row], alone[0], rtol=1e-13, atol=1e-9)
    if name == "speed-command":
        # the reference is each flight's OWN anchor airspeed: swapping the anchors moves the thrust
        swapped = law.geodetic_controls(
            states, actual, max_thrust_n=thrust, initial_geodetic_states=initial.flip(0),
            aero_params=aero,
        )
        assert not torch.equal(swapped[..., 0], both[..., 0])


def test_the_geodetic_readers_refuse_a_shape_they_would_broadcast():
    law = LAWS["path-angle"]
    initial, states, actual, aero, thrust = _batch()
    with pytest.raises(ValueError, match=r"\[B, N, 7\] states and \[B, N, 3\] actuators"):
        law.geodetic_load(states[:, 0], actual[:, 0], max_thrust_n=thrust, initial_geodetic_states=initial)
    with pytest.raises(ValueError, match=r"\[B, N, 7\] states and \[B, N, 3\] actuators"):
        law.geodetic_controls(states, actual[:, :3], max_thrust_n=thrust,
                              initial_geodetic_states=initial, aero_params=aero)


def test_the_parameter_columns_take_the_consumers_dtype():
    """A float32 installed thrust must not round a law's float constants for a float64 consumer
    (review 2026-09-16, R1 finding 2): the columns are cast in the dtype the consumer names."""
    law = LAWS["path-angle"]
    thrust32 = torch.tensor([THRUST], dtype=torch.float32)
    initial = torch.tensor([[35.95, -78.75, 900.0, 85.0, 1.1, -0.05, MASS]], dtype=torch.float64)
    columns = law.parameter_matrix(thrust32, initial, dtype=torch.float64, device=torch.device("cpu"))
    assert columns.dtype == torch.float64
    assert columns[0, 2].item() == 0.2 and columns[0, 1].item() == 3.0


def test_the_path_loop_keeps_the_path_when_the_bank_moves():
    """Why a hook that moves the bank leaves the path-angle target alone: the loop divides by the flown
    bank at every stage, so the same target flies the same path angle at 0° and at 25° of bank."""
    law = LAWS["path-angle"]
    target = math.radians(-3.5)
    paths = []
    for bank_deg in (0.0, 25.0):
        commands = torch.tensor([[[-0.05, math.radians(bank_deg), target]] * 6], dtype=torch.float64)
        queries = torch.arange(1.0, 60.0, 1.0, dtype=torch.float64).unsqueeze(0)
        dense = rollout_piecewise_constant_at_times(
            torch.tensor([[35.95, -78.75, 1500.0, 85.0, 1.1, math.radians(-3.0), MASS]], dtype=torch.float64),
            torch.tensor([[-0.05, 0.0, math.radians(-3.0)]], dtype=torch.float64), commands,
            torch.full((1, 6), 10.0, dtype=torch.float64), torch.tensor([AERO], dtype=torch.float64), FRAME, TAU,
            torch.tensor([THRUST], dtype=torch.float64), queries, torch.ones_like(queries, dtype=torch.bool),
            control_law=law, integrator_dt_s=0.1,
        )
        velocity = dense.query_states[0, :, 3:6]
        paths.append(torch.asin(velocity[:, 2] / velocity.norm(dim=-1)))
    settled = slice(20, None)
    assert torch.allclose(paths[0][settled], paths[1][settled], rtol=0.0, atol=math.radians(0.05))
    assert abs(float(paths[1][-1]) - target) < math.radians(0.05)

