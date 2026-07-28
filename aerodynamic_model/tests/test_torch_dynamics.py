"""Numerical contract tests between the differentiable and CasADi dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aerodynamic_model.casadi_simulator import CasadiSimulator
from aerodynamic_model.common import GeodeticState, LoadFactorControl
from aerodynamic_model.rollout import rollout_piecewise_constant as casadi_rollout
from aerodynamic_model.torch_dynamics import geodetic_step, rollout_piecewise_constant
from aircraft.aero_params import aero_params_for_aircraft
from aircraft.aircraft_sets import A320


def _aero_tensor(batch: int = 1) -> torch.Tensor:
    aero = aero_params_for_aircraft(A320)
    row = [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall]
    return torch.tensor([row] * batch, dtype=torch.float64)


def _state_tensor(state: GeodeticState) -> torch.Tensor:
    return torch.tensor(
        [[state.latitude, state.longitude, state.altitude, state.V,
          state.psi, state.gamma, state.m]],
        dtype=torch.float64,
    )


def _state_array(state: GeodeticState) -> np.ndarray:
    return np.array([
        state.latitude, state.longitude, state.altitude, state.V,
        state.psi, state.gamma, state.m,
    ])


@pytest.mark.parametrize(
    "control",
    [
        LoadFactorControl(thrust=42_000.0, bank_rad=math.radians(12.0), load_factor=1.05),
        # Deliberately above Cl_max demand: verifies the capped-lift/stall-drag branch too.
        LoadFactorControl(thrust=20_000.0, bank_rad=math.radians(-20.0), load_factor=3.2),
    ],
)
def test_torch_step_matches_casadi_simulator_including_stall(control):
    initial = GeodeticState(
        latitude=35.8801,
        longitude=-78.7880,
        altitude=940.0,
        V=78.0,
        psi=math.radians(147.0),
        gamma=math.radians(-3.1),
        m=A320.landing_mass,
    )
    expected = CasadiSimulator(A320, 0.5).step(initial, control, 0.5)
    actual = geodetic_step(
        _state_tensor(initial),
        torch.tensor([[control.thrust, control.bank_rad, control.load_factor]], dtype=torch.float64),
        _aero_tensor(),
        torch.tensor([0.5], dtype=torch.float64),
    )[0].detach().numpy()

    np.testing.assert_allclose(actual, _state_array(expected), rtol=2e-12, atol=2e-9)


def test_nonuniform_torch_rollout_matches_casadi_segment_endpoints():
    initial = GeodeticState(
        latitude=38.7420,
        longitude=-90.3650,
        altitude=1200.0,
        V=92.0,
        psi=math.radians(110.0),
        gamma=math.radians(-2.0),
        m=A320.landing_mass,
    )
    numeric_controls = [
        LoadFactorControl(55_000.0, math.radians(8.0), 1.03),
        LoadFactorControl(38_000.0, math.radians(-5.0), 0.98),
        LoadFactorControl(25_000.0, 0.0, 1.01),
    ]
    durations = [1.3, 2.1, 0.75]
    samples = casadi_rollout(
        CasadiSimulator(A320, 0.5),
        initial,
        numeric_controls,
        sum(durations),
        integrator_dt=0.5,
        segment_durations=durations,
    )
    boundaries = np.cumsum(durations)
    expected = np.stack([
        _state_array(next(sample.state for sample in samples if abs(sample.t - boundary) < 1e-9))
        for boundary in boundaries
    ])

    controls = torch.tensor(
        [[[c.thrust, c.bank_rad, c.load_factor] for c in numeric_controls]],
        dtype=torch.float64,
    )
    actual = rollout_piecewise_constant(
        _state_tensor(initial),
        controls,
        torch.tensor([durations], dtype=torch.float64),
        _aero_tensor(),
        integrator_dt_s=0.5,
    )[0].detach().numpy()

    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-8)


def test_batched_heterogeneous_durations_match_independent_casadi_rollouts():
    """The shorter batch member must stop stepping while the longer one remains active."""
    initial_states = [
        GeodeticState(35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass),
        GeodeticState(38.74, -90.36, 1300.0, 95.0, 1.7, -0.03, A320.landing_mass),
    ]
    controls_by_flight = [
        [
            LoadFactorControl(45_000.0, 0.08, 1.02),
            LoadFactorControl(36_000.0, -0.04, 0.99),
            LoadFactorControl(30_000.0, 0.01, 1.01),
        ],
        [
            LoadFactorControl(60_000.0, -0.06, 1.04),
            LoadFactorControl(48_000.0, 0.03, 1.00),
            LoadFactorControl(33_000.0, -0.02, 0.98),
        ],
    ]
    durations_by_flight = [[0.4, 1.1, 0.6], [1.3, 0.2, 0.75]]
    expected_by_flight = []
    for initial, controls, durations in zip(
        initial_states, controls_by_flight, durations_by_flight
    ):
        samples = casadi_rollout(
            CasadiSimulator(A320, 0.5),
            initial,
            controls,
            sum(durations),
            integrator_dt=0.5,
            segment_durations=durations,
        )
        expected_by_flight.append(np.stack([
            _state_array(next(
                sample.state for sample in samples if abs(sample.t - boundary) < 1e-9
            ))
            for boundary in np.cumsum(durations)
        ]))

    actual = rollout_piecewise_constant(
        torch.cat([_state_tensor(state) for state in initial_states], dim=0),
        torch.tensor([
            [[control.thrust, control.bank_rad, control.load_factor] for control in controls]
            for controls in controls_by_flight
        ], dtype=torch.float64),
        torch.tensor(durations_by_flight, dtype=torch.float64),
        _aero_tensor(batch=2),
        integrator_dt_s=0.5,
    ).detach().numpy()

    np.testing.assert_allclose(
        actual, np.stack(expected_by_flight), rtol=3e-12, atol=3e-8
    )


def test_rollout_backpropagates_to_controls_and_nonuniform_durations():
    initial = torch.tensor(
        [[35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass]],
        dtype=torch.float64,
    )
    controls = torch.tensor(
        [[[45_000.0, 0.08, 1.02], [36_000.0, -0.04, 0.99]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    durations = torch.tensor([[0.8, 1.2]], dtype=torch.float64, requires_grad=True)
    states = rollout_piecewise_constant(
        initial, controls, durations, _aero_tensor(), integrator_dt_s=0.5
    )
    loss = states[..., 2:6].square().mean()
    loss.backward()

    assert controls.grad is not None and torch.isfinite(controls.grad).all()
    assert durations.grad is not None and torch.isfinite(durations.grad).all()
    assert torch.count_nonzero(controls.grad) > 0
    assert torch.count_nonzero(durations.grad) == durations.numel()
