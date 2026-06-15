import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_single_shooting_module():
    module_path = Path(__file__).resolve().parents[1] / "single_shooting_optimizor.py"
    optimization_dir = str(module_path.parent)
    if optimization_dir not in sys.path:
        sys.path.insert(0, optimization_dir)

    spec = importlib.util.spec_from_file_location(
        "single_shooting_optimizor",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_shooting_imports():
    module = load_single_shooting_module()

    assert module.SingleShootingOptimizor.__name__ == "SingleShootingOptimizor"


def test_trim_attack_guess_uses_aircraft_state_and_aero_model():
    module = load_single_shooting_module()
    optimizer = module.SingleShootingOptimizor(
        module.GeodeticSimulator(),
        n_control_segments=10,
    )
    state = module.GeodeticState(
        latitude=35.8,
        longitude=-78.8,
        altitude=700.0,
        V=87.45,
        psi=np.radians(45.0),
        gamma=np.radians(-3.0),
        m=78000.0,
    )

    attack_guess = optimizer.estimate_trim_attack_rad(state)

    assert np.radians(8.0) < attack_guess <= module._MAX_ATTACK_RAD


def test_initial_unsimulatable_guess_raises_before_minimize(monkeypatch):
    module = load_single_shooting_module()
    optimizer = module.SingleShootingOptimizor(
        FailingGeodeticSimulator(),
        n_control_segments=2,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("minimize should not run for an invalid initial guess")

    monkeypatch.setattr(module, "minimize", fail_if_called)

    with pytest.raises(ValueError, match="initial control guess is not simulatable"):
        optimizer.optimize_trajectory(state, state)


def test_optimize_trajectory_passes_trim_attack_guess_to_minimize(monkeypatch):
    module = load_single_shooting_module()
    optimizer = module.SingleShootingOptimizor(
        module.GeodeticSimulator(),
        n_control_segments=1,
        dt=0.2,
    )
    initial_state = module.GeodeticState(
        latitude=35.8,
        longitude=-78.8,
        altitude=700.0,
        V=87.45,
        psi=np.radians(45.0),
        gamma=np.radians(-3.0),
        m=78000.0,
    )
    target_state = module.GeodeticState(
        latitude=35.8,
        longitude=-78.8,
        altitude=700.0,
        V=87.45,
        psi=np.radians(45.0),
        gamma=np.radians(-3.0),
        m=78000.0,
    )
    expected_attack = optimizer.estimate_trim_attack_rad(initial_state)

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        del fun, bounds, constraints
        assert method == "SLSQP"
        assert options == {"maxiter": 1000, "ftol": 1e-6}
        _, controls = optimizer.unpack_z(x0)
        np.testing.assert_allclose(controls[:, 2], [expected_attack])
        assert controls[0, 2] != module._DEFAULT_ATTACK_GUESS_RAD
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    final_time, controls, node_state = optimizer.optimize_trajectory(
        initial_state,
        target_state,
    )

    assert final_time == 1.0
    assert controls.shape == (1, 3)
    assert node_state is None


class FailingGeodeticSimulator:
    def step(self, state, control, dt):
        del state, control, dt
        raise ValueError("simulation stopped: altitude below 0")
