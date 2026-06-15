import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_least_squares_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "least_squares_transcription_optimizor.py"
    )
    optimization_dir = str(module_path.parent)
    if optimization_dir not in sys.path:
        sys.path.insert(0, optimization_dir)

    spec = importlib.util.spec_from_file_location(
        "least_squares_transcription_optimizor",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_least_squares_transcription_imports():
    module = load_least_squares_module()

    assert (
        module.LeastSquaresTranscriptionOptimizor.__name__
        == "LeastSquaresTranscriptionOptimizor"
    )


def test_variable_scale_uses_aircraft_specific_thrust_bound():
    module = load_least_squares_module()
    aircraft = SimpleNamespace(
        max_thrust_n=1026000.0,
        approach_thrust_guess_n=140000.0,
    )
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=SimpleNamespace(simulator=SimpleNamespace(aircraft=aircraft)),
        n_segments=2,
        arrival_time_s=100.0,
    )

    scale = optimizer.build_variable_scale()

    np.testing.assert_allclose(scale[:6], [1026000.0, 1.0, 0.1, 1026000.0, 1.0, 0.1])


def test_optimize_trajectory_builds_least_squares_problem(monkeypatch):
    module = load_least_squares_module()
    geodetic_simulator = FakeGeodeticSimulator(module)
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=geodetic_simulator,
        n_segments=2,
        arrival_time_s=100.0,
        dt=100.0,
    )
    initial_state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    target_state = module.GeodeticState(3.0, 4.0, 900.0, 115.0, 0.2, -0.02, 70000.0)

    def fake_least_squares(fun, x0, bounds, max_nfev, ftol, xtol, gtol, x_scale):
        assert len(x0) == optimizer.n_segments * (
            optimizer.control_dim + optimizer.state_dim
        )
        assert len(bounds) == 2
        assert len(bounds[0]) == len(x0)
        assert len(bounds[1]) == len(x0)
        assert max_nfev == 1000
        assert ftol == xtol == gtol == 1e-6
        np.testing.assert_allclose(x_scale, optimizer.build_variable_scale())

        residuals = fun(x0)
        assert residuals.shape == (
            optimizer.n_segments * optimizer.state_dim
            + optimizer.state_dim
            + optimizer.n_segments * optimizer.state_dim
            + optimizer.n_segments * optimizer.control_dim,
        )
        assert len(geodetic_simulator.calls) == 8
        assert all(call["dt"] == 50.0 for call in geodetic_simulator.calls)
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "least_squares", fake_least_squares)

    final_time, node_control, node_state = optimizer.optimize_trajectory(
        initial_state,
        target_state,
    )

    assert final_time == 100.0
    assert node_control.shape == (2, 3)
    assert node_state.shape == (2, 6)


def test_terminal_residual_is_weighted_above_defects():
    module = load_least_squares_module()
    geodetic_simulator = FakeGeodeticSimulator(module)
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=geodetic_simulator,
        n_segments=1,
        arrival_time_s=10.0,
        dt=10.0,
    )
    initial_state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    target_state = module.GeodeticState(1.0, 2.0, 900.0, 120.0, 0.1, -0.01, 70000.0)
    node_control_guess = np.array([[0.0, 0.0, 0.0]])
    node_state_guess = optimizer.geodetic_state_to_array(initial_state).reshape(1, 6)
    z = np.hstack((node_control_guess.flatten(), node_state_guess.flatten()))

    residuals = optimizer.trajectory_residuals(z, initial_state, target_state)
    terminal_start = optimizer.n_segments * optimizer.state_dim
    terminal_residual = residuals[
        terminal_start:terminal_start + optimizer.state_dim
    ]

    expected_terminal = (
        module._TERMINAL_RESIDUAL_WEIGHT
        * optimizer.state_constraint_error(node_state_guess[0], optimizer.geodetic_state_to_array(target_state))
    )
    np.testing.assert_allclose(terminal_residual, expected_terminal)
    assert terminal_residual[2] == module._TERMINAL_RESIDUAL_WEIGHT


def test_replay_state_residual_uses_controls_not_node_state():
    module = load_least_squares_module()
    geodetic_simulator = FakeGeodeticSimulator(module)
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=geodetic_simulator,
        n_segments=2,
        arrival_time_s=10.0,
        dt=10.0,
    )
    initial_state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    target_state = module.GeodeticState(1.0, 2.0, 900.0, 120.0, 0.1, -0.01, 70000.0)
    node_control_guess = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    node_state_guess = np.vstack((
        optimizer.geodetic_state_to_array(target_state),
        optimizer.geodetic_state_to_array(target_state),
    ))
    z = np.hstack((node_control_guess.flatten(), node_state_guess.flatten()))

    residuals = optimizer.trajectory_residuals(z, initial_state, target_state)
    replay_start = (
        optimizer.n_segments * optimizer.state_dim
        + optimizer.state_dim
    )
    replay_residual = residuals[
        replay_start:replay_start + optimizer.n_segments * optimizer.state_dim
    ]
    first_replay_state = optimizer.geodetic_state_to_array(
        module.GeodeticState(
            latitude=initial_state.latitude + 0.1,
            longitude=initial_state.longitude + 0.2,
            altitude=initial_state.altitude + 1.0,
            V=initial_state.V + 2.0,
            psi=initial_state.psi + 0.01,
            gamma=initial_state.gamma + 0.001,
            m=initial_state.m,
        )
    )
    second_replay_state = optimizer.geodetic_state_to_array(
        module.GeodeticState(
            latitude=initial_state.latitude + 0.2,
            longitude=initial_state.longitude + 0.4,
            altitude=initial_state.altitude + 2.0,
            V=initial_state.V + 4.0,
            psi=initial_state.psi + 0.02,
            gamma=initial_state.gamma + 0.002,
            m=initial_state.m,
        )
    )

    expected_replay_residual = (
        module._REPLAY_STATE_RESIDUAL_WEIGHT
        * np.concatenate((
            optimizer.state_constraint_error(
                first_replay_state,
                node_state_guess[0],
            ),
            optimizer.state_constraint_error(
                second_replay_state,
                node_state_guess[1],
            ),
        ))
    )
    np.testing.assert_allclose(replay_residual, expected_replay_residual)


def test_replay_residual_adds_one_state_block_per_segment():
    module = load_least_squares_module()
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=FakeGeodeticSimulator(module),
        n_segments=3,
        arrival_time_s=12.0,
        dt=12.0,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    node_control_guess = np.zeros((optimizer.n_segments, optimizer.control_dim))
    node_state_guess = np.tile(
        optimizer.geodetic_state_to_array(state),
        (optimizer.n_segments, 1),
    )
    z = np.hstack((node_control_guess.flatten(), node_state_guess.flatten()))

    residuals = optimizer.trajectory_residuals(z, state, state)

    assert residuals.shape == (
        optimizer.n_segments * optimizer.state_dim
        + optimizer.state_dim
        + optimizer.n_segments * optimizer.state_dim
        + optimizer.n_segments * optimizer.control_dim,
    )


def test_initial_invalid_residual_raises_before_solver(monkeypatch):
    module = load_least_squares_module()
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=FailingGeodeticSimulator(),
        n_segments=1,
        arrival_time_s=20.0,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("least_squares should not run for an invalid initial guess")

    monkeypatch.setattr(module, "least_squares", fail_if_called)

    with pytest.raises(ValueError, match="initial trajectory guess is not simulatable"):
        optimizer.optimize_trajectory(state, state)


def test_least_squares_failure_raises(monkeypatch):
    module = load_least_squares_module()
    optimizer = module.LeastSquaresTranscriptionOptimizor(
        sim_server=FakeGeodeticSimulator(module),
        n_segments=1,
        arrival_time_s=20.0,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fake_least_squares(*args, **kwargs):
        return SimpleNamespace(success=False, x=np.array([]), message="bad residual")

    monkeypatch.setattr(module, "least_squares", fake_least_squares)

    with pytest.raises(ValueError, match="Optimization failed: bad residual"):
        optimizer.optimize_trajectory(state, state)


class FakeGeodeticSimulator:
    def __init__(self, module):
        self.module = module
        self.calls = []

    def step(self, state, control, dt):
        self.calls.append({"state": state, "control": control, "dt": dt})
        return self.module.GeodeticState(
            latitude=state.latitude + 0.1,
            longitude=state.longitude + 0.2,
            altitude=state.altitude + 1.0,
            V=state.V + 2.0,
            psi=state.psi + 0.01,
            gamma=state.gamma + 0.001,
            m=state.m,
        )


class FailingGeodeticSimulator:
    def step(self, state, control, dt):
        del state, control, dt
        raise ValueError("simulation stopped: altitude below 0")
