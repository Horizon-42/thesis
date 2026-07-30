"""Numerical contract tests between the differentiable and CasADi dynamics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import aerodynamic_model.torch_dynamics as torch_dynamics
from aerodynamic_model.torch_dense_rollout import rollout_piecewise_constant_at_times
from aerodynamic_model.casadi_simulator import CasadiSimulator
from aerodynamic_model.common import GeodeticState, LoadFactorControl
from aerodynamic_model.rollout import rollout_piecewise_constant as casadi_rollout
from aerodynamic_model.torch_dynamics import (
    geodetic_step,
    rollout_piecewise_constant,
)
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


def _segment_barrier_rollout(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    durations: torch.Tensor,
    aero_params: torch.Tensor,
    *,
    integrator_dt_s: float,
) -> torch.Tensor:
    """Pre-optimization reference: all batch rows synchronize at every segment."""
    state = initial_states
    endpoints = []
    dt_cap = torch.as_tensor(
        integrator_dt_s, dtype=initial_states.dtype, device=initial_states.device
    )
    for segment in range(controls.shape[1]):
        remaining = durations[:, segment]
        steps = int(torch.ceil(remaining.detach().max() / integrator_dt_s).item())
        for _ in range(steps):
            step_dt = torch.minimum(remaining, dt_cap)
            active = step_dt > 0.0
            stepped = geodetic_step(state, controls[:, segment], aero_params, step_dt)
            state = torch.where(active.unsqueeze(-1), stepped, state)
            remaining = (remaining - step_dt).clamp(min=0.0)
        endpoints.append(state)
    return torch.stack(endpoints, dim=1)


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


def test_dense_rollout_records_fixed_queries_and_backpropagates_through_switch_times():
    initial_state = GeodeticState(
        35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass
    )
    initial = _state_tensor(initial_state)
    numeric_controls = [
        LoadFactorControl(45_000.0, 0.08, 1.02),
        LoadFactorControl(36_000.0, -0.04, 0.99),
    ]
    controls = torch.tensor(
        [[
            [control.thrust, control.bank_rad, control.load_factor]
            for control in numeric_controls
        ]],
        dtype=torch.float64,
        requires_grad=True,
    )
    durations = torch.tensor([[1.3, 2.7]], dtype=torch.float64, requires_grad=True)
    result = rollout_piecewise_constant_at_times(
        initial,
        controls,
        durations,
        _aero_tensor(),
        torch.tensor([[2.0, 4.0]], dtype=torch.float64),
        torch.tensor([[True, True]]),
        integrator_dt_s=0.5,
    )

    assert result.query_states.shape == (1, 2, 7)
    assert result.segment_end_states.shape == (1, 2, 7)
    torch.testing.assert_close(
        result.query_states[:, -1], result.segment_end_states[:, -1]
    )

    simulator = CasadiSimulator(A320, 0.5)
    state = initial_state
    previous = 0.0
    expected_queries = []
    expected_endpoints = []
    for event in sorted({*np.arange(0.5, 4.01, 0.5), 1.3, 4.0}):
        control = numeric_controls[0 if 0.5 * (previous + event) < 1.3 else 1]
        state = simulator.step(state, control, event - previous)
        previous = event
        if event in (2.0, 4.0):
            expected_queries.append(_state_array(state))
        if event in (1.3, 4.0):
            expected_endpoints.append(_state_array(state))
    np.testing.assert_allclose(
        result.query_states[0].detach().numpy(),
        np.stack(expected_queries),
        rtol=3e-12,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        result.segment_end_states[0].detach().numpy(),
        np.stack(expected_endpoints),
        rtol=3e-12,
        atol=3e-8,
    )
    loss = result.query_states[..., 2:6].square().mean()
    loss.backward()

    assert controls.grad is not None and torch.isfinite(controls.grad).all()
    assert durations.grad is not None and torch.isfinite(durations.grad).all()
    assert torch.count_nonzero(controls.grad) > 0
    # Fixed queries depend on switch boundaries, not on unused slack after the final query.
    # The first duration moves the 1.3 s switch and must therefore receive state gradient.
    assert durations.grad[0, 0] != 0.0


def test_discrete_adjoint_uses_one_dense_state_layout(monkeypatch):
    """Every reverse step must hit the same static compiled-kernel layout."""
    observed_strides = []
    original_step = torch_dynamics._rollout_step

    def recording_step(state, controls, aero_params, dt_s):
        if torch.is_grad_enabled():
            observed_strides.append(state.stride())
        return original_step(state, controls, aero_params, dt_s)

    monkeypatch.setattr(torch_dynamics, "_rollout_step", recording_step)
    initial = torch.tensor(
        [
            [35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass],
            [38.74, -90.36, 1300.0, 95.0, 1.7, -0.03, A320.landing_mass],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    controls = torch.tensor(
        [
            [[45_000.0, 0.08, 1.02], [36_000.0, -0.04, 0.99]],
            [[60_000.0, -0.06, 1.04], [48_000.0, 0.03, 1.00]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    durations = torch.tensor(
        [[1.1, 0.9], [0.6, 1.4]], dtype=torch.float64, requires_grad=True
    )

    rollout_piecewise_constant(
        initial,
        controls,
        durations,
        _aero_tensor(batch=2),
        integrator_dt_s=0.5,
    ).sum().backward()

    assert len(observed_strides) > 1
    assert set(observed_strides) == {(7, 1)}


def test_single_flight_rollouts_do_not_leak_horizon_length_into_step_strides(monkeypatch):
    """Sequential inference flights must reuse one compiled-kernel input layout.

    A transpose of ``[1,S,*]`` can still report itself contiguous while retaining an
    ``S``-dependent stride on the singleton batch dimension.  Different learned durations
    then consume one Dynamo specialization per flight until inference hits its cache limit.
    """
    observed_strides = []
    original_step = torch_dynamics._rollout_step

    def recording_step(state, controls, aero_params, dt_s):
        observed_strides.append((controls.stride(), dt_s.stride()))
        return original_step(state, controls, aero_params, dt_s)

    monkeypatch.setattr(torch_dynamics, "_rollout_step", recording_step)
    initial = torch.tensor(
        [[35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass]],
        dtype=torch.float64,
    )
    controls = torch.tensor(
        [[[45_000.0, 0.08, 1.02], [36_000.0, -0.04, 0.99]]],
        dtype=torch.float64,
    )

    with torch.no_grad():
        for durations in ([0.5, 1.0], [0.5, 1.5]):
            rollout_piecewise_constant(
                initial,
                controls,
                torch.tensor([durations], dtype=torch.float64),
                _aero_tensor(),
                integrator_dt_s=0.5,
            )

    assert observed_strides
    assert set(observed_strides) == {((3, 1), (1,))}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_inference_step_accepts_many_partial_batch_shapes():
    """Validation replay must not exhaust Dynamo's static-shape recompile cache."""
    torch.compiler.reset()
    torch_dynamics._COMPILED_CUDA_INFERENCE_STEP = None
    try:
        state_row = torch.tensor(
            [35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass],
            dtype=torch.float64,
            device="cuda",
        )
        control_row = torch.tensor(
            [45_000.0, 0.08, 1.02], dtype=torch.float64, device="cuda"
        )
        with torch.no_grad():
            for batch in range(1, 11):
                stepped = torch_dynamics._rollout_step(
                    state_row.expand(batch, -1).contiguous(),
                    control_row.expand(batch, -1).contiguous(),
                    _aero_tensor(batch).to("cuda"),
                    torch.full((batch,), 0.5, dtype=torch.float64, device="cuda"),
                )
                assert stepped.shape == (batch, 7)
                assert torch.isfinite(stepped).all()
    finally:
        torch.compiler.reset()
        torch_dynamics._COMPILED_CUDA_INFERENCE_STEP = None


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_global_schedule_matches_segment_barriers_in_outputs_and_gradients(device):
    initial_rows = torch.tensor(
        [
            [35.88, -78.79, 900.0, 80.0, 2.2, -0.04, A320.landing_mass],
            [38.74, -90.36, 1300.0, 95.0, 1.7, -0.03, A320.landing_mass],
        ],
        dtype=torch.float64,
        device=device,
    )
    control_rows = torch.tensor(
        [
            [[45_000.0, 0.08, 1.02], [36_000.0, -0.04, 0.99], [30_000.0, 0.01, 1.01]],
            [[60_000.0, -0.06, 1.04], [48_000.0, 0.03, 1.00], [33_000.0, -0.02, 0.98]],
        ],
        dtype=torch.float64,
        device=device,
    )
    # Includes exact dt multiples and values immediately on either side of a boundary.
    duration_rows = torch.tensor(
        [[0.5, 1.0, 1.5000000001], [1.4999999999, 0.51, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    weights = torch.linspace(
        0.5, 1.5, 2 * 3 * 7, dtype=torch.float64, device=device
    ).reshape(2, 3, 7)

    def evaluated(rollout):
        initial = initial_rows.clone().requires_grad_()
        controls = control_rows.clone().requires_grad_()
        durations = duration_rows.clone().requires_grad_()
        aero = _aero_tensor(batch=2).to(device).requires_grad_()
        states = rollout(
            initial,
            controls,
            durations,
            aero,
            integrator_dt_s=0.5,
        )
        gradients = torch.autograd.grad(
            (states * weights).sum(), (initial, controls, durations, aero)
        )
        return states.detach(), tuple(gradient.detach() for gradient in gradients)

    expected_states, expected_gradients = evaluated(_segment_barrier_rollout)
    actual_states, actual_gradients = evaluated(rollout_piecewise_constant)

    state_tolerance = (
        {"rtol": 3e-12, "atol": 3e-8}
        if device.type == "cuda"
        else {"rtol": 0.0, "atol": 0.0}
    )
    torch.testing.assert_close(actual_states, expected_states, **state_tolerance)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-8)
