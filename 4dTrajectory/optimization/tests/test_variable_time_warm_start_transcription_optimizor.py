import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_variable_time_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "variable_time_warm_start_transcription_optimizor.py"
    )
    optimization_dir = str(module_path.parent)
    if optimization_dir not in sys.path:
        sys.path.insert(0, optimization_dir)

    spec = importlib.util.spec_from_file_location(
        "variable_time_warm_start_transcription_optimizor",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variable_time_warm_start_transcription_imports():
    module = load_variable_time_module()

    assert (
        module.VariableTimeWarmStartTranscriptionOptimizor.__name__
        == "VariableTimeWarmStartTranscriptionOptimizor"
    )


def test_unpack_z_includes_variable_final_time():
    module = load_variable_time_module()
    optimizer = module.VariableTimeWarmStartTranscriptionOptimizor(
        sim_server=object(),
        n_segments=2,
        arrival_time_s=75.0,
    )
    z = np.arange(
        1
        + optimizer.n_segments * optimizer.control_dim
        + optimizer.n_segments * optimizer.state_dim,
        dtype=float,
    )

    final_time, node_control, node_state = optimizer.unpack_z(z)

    assert final_time == 0.0
    np.testing.assert_array_equal(
        node_control,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    np.testing.assert_array_equal(
        node_state,
        np.array([
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ]),
    )


def test_optimize_uses_variable_time_warm_start_for_slsqp(monkeypatch):
    module = load_variable_time_module()
    optimizer = module.VariableTimeWarmStartTranscriptionOptimizor(
        sim_server=FakeGeodeticSimulator(module),
        n_segments=2,
        arrival_time_s=100.0,
        dt=100.0,
    )
    initial_state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    target_state = module.GeodeticState(3.0, 4.0, 900.0, 115.0, 0.2, -0.02, 70000.0)
    calls = []

    def fake_least_squares(fun, x0, bounds, max_nfev, ftol, xtol, gtol, x_scale):
        calls.append({"phase": "least_squares", "x0": x0.copy()})
        assert len(x0) == 1 + optimizer.n_segments * (
            optimizer.control_dim + optimizer.state_dim
        )
        assert x0[0] == 100.0
        np.testing.assert_allclose(bounds[0][0], 50.0)
        np.testing.assert_allclose(bounds[1][0], 150.0)
        assert max_nfev == 2
        assert ftol == xtol == gtol == 1e-6
        assert len(x_scale) == len(x0)
        residuals = fun(x0)
        assert residuals.shape == (
            optimizer.n_segments * optimizer.state_dim
            + optimizer.state_dim
            + optimizer.n_segments * optimizer.state_dim
            + optimizer.n_segments * optimizer.control_dim
            + 1,
        )
        warm_x = x0.copy()
        warm_x[0] = 96.0
        warm_x[1] = 11000.0
        return SimpleNamespace(success=False, x=warm_x, message="max nfev")

    def fake_minimize(fun, x0, bounds, constraints, method, options):
        calls.append({"phase": "minimize", "x0": x0.copy()})
        assert x0[0] == 96.0
        assert x0[1] == 11000.0
        assert method == "SLSQP"
        assert options == {"maxiter": 1000, "ftol": 1e-6}
        assert len(bounds) == len(x0)
        assert fun(x0) >= 0.0
        assert constraints[0]["fun"](x0).shape == (
            optimizer.n_segments * optimizer.state_dim,
        )
        assert constraints[1]["fun"](x0).shape == (optimizer.state_dim,)
        final_x = x0.copy()
        final_x[0] = 94.0
        return SimpleNamespace(success=True, x=final_x, message="")

    monkeypatch.setattr(module, "least_squares", fake_least_squares)
    monkeypatch.setattr(module, "minimize", fake_minimize)

    final_time, node_control, node_state = optimizer.optimize_trajectory(
        initial_state,
        target_state,
    )

    assert [call["phase"] for call in calls] == ["least_squares", "minimize"]
    assert final_time == 94.0
    assert node_control[0, 0] == 11000.0
    assert node_control.shape == (2, 3)
    assert node_state.shape == (2, 6)


def test_initial_invalid_residual_raises_before_variable_time_solver(monkeypatch):
    module = load_variable_time_module()
    optimizer = module.VariableTimeWarmStartTranscriptionOptimizor(
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


class FakeGeodeticSimulator:
    def __init__(self, module):
        self.module = module

    def step(self, state, control, dt):
        del control, dt
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
