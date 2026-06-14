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

    def fake_least_squares(fun, x0, bounds, max_nfev, ftol, xtol, gtol):
        assert len(x0) == optimizer.n_segments * (
            optimizer.control_dim + optimizer.state_dim
        )
        assert len(bounds) == 2
        assert len(bounds[0]) == len(x0)
        assert len(bounds[1]) == len(x0)
        assert max_nfev == 1000
        assert ftol == xtol == gtol == 1e-6

        residuals = fun(x0)
        assert residuals.shape == (
            optimizer.n_segments * optimizer.state_dim
            + optimizer.state_dim
            + optimizer.n_segments * optimizer.control_dim,
        )
        assert len(geodetic_simulator.calls) == optimizer.n_segments
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
