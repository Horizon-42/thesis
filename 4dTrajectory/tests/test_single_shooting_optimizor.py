import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_single_shooting_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "optimization"
        / "single_shooting_optimizor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "single_shooting_optimizor",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def test_single_shooting_optimizor_imports_from_optimization_dir():
    module = load_single_shooting_module()

    assert module.SingleShootingOptimizor.__name__ == "SingleShootingOptimizor"


def test_unpack_z_uses_final_time_and_control_layout():
    module = load_single_shooting_module()
    optimizer = module.SingleShootingOptimizor(
        sim=FakeGeodeticSimulator(module),
        n_control_segments=2,
    )
    z = np.arange(1 + optimizer.n_control_segments * optimizer.control_dim, dtype=float)

    final_time, controls = optimizer.unpack_z(z)

    assert final_time == 0.0
    np.testing.assert_array_equal(
        controls,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )


def test_angular_difference_uses_shortest_wrapped_distance():
    module = load_single_shooting_module()

    difference = module.SingleShootingOptimizor.angular_difference_rad(
        math.radians(179.0),
        math.radians(-179.0),
    )

    assert math.isclose(difference, math.radians(-2.0))


def test_position_constraint_error_uses_only_position():
    module = load_single_shooting_module()
    state = module.GeodeticState(
        latitude=1.0 / 111.32,
        longitude=1.0 / 111.32,
        altitude=1100.0,
        V=130.0,
        psi=math.radians(179.0),
        gamma=math.radians(-1.0),
        m=70000.0,
    )
    target_state = module.GeodeticState(
        latitude=0.0,
        longitude=0.0,
        altitude=1000.0,
        V=120.0,
        psi=math.radians(-179.0),
        gamma=math.radians(-3.0),
        m=70000.0,
    )

    error = module.SingleShootingOptimizor.position_constraint_error(
        state,
        target_state,
    )

    np.testing.assert_allclose(error, np.array([1.0, 1.0, 1.0]), atol=1e-6)


def test_terminal_soft_cost_uses_speed_and_angles():
    module = load_single_shooting_module()
    state = module.GeodeticState(
        latitude=1.0,
        longitude=2.0,
        altitude=1100.0,
        V=130.0,
        psi=math.radians(179.0),
        gamma=math.radians(-1.0),
        m=70000.0,
    )
    target_state = module.GeodeticState(
        latitude=0.0,
        longitude=0.0,
        altitude=1000.0,
        V=120.0,
        psi=math.radians(-179.0),
        gamma=math.radians(-3.0),
        m=70000.0,
    )

    cost = module.SingleShootingOptimizor.terminal_soft_cost(
        state,
        target_state,
    )

    assert math.isclose(cost, math.sqrt(1.0**2 + (-0.2)**2 + 1.0**2))


def test_segment_simulate_splits_duration_into_bounded_substeps():
    module = load_single_shooting_module()
    simulator = FakeGeodeticSimulator(module)
    optimizer = module.SingleShootingOptimizor(
        sim=simulator,
        n_control_segments=1,
        dt=0.4,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    control = module.Control(12000.0, 0.0, 0.0)

    optimizer.segment_simulate(state, control, duration=1.0)

    assert len(simulator.calls) == 3
    np.testing.assert_allclose(
        [call["dt"] for call in simulator.calls],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    )


def test_optimize_trajectory_builds_single_shooting_minimize_problem(monkeypatch):
    module = load_single_shooting_module()
    target_state = module.GeodeticState(3.0, 4.0, 900.0, 115.0, 0.2, -0.02, 70000.0)
    simulator = TargetStateSimulator(target_state)
    optimizer = module.SingleShootingOptimizor(
        sim=simulator,
        n_control_segments=2,
        dt=10000.0,
    )
    initial_state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        assert len(x0) == 1 + optimizer.n_control_segments * optimizer.control_dim
        assert method == "SLSQP"
        assert len(bounds) == len(x0)
        assert constraints["type"] == "eq"
        assert options == {"maxiter": 1000, "ftol": 1e-6}

        final_time, controls = optimizer.unpack_z(x0)
        assert final_time > 0.0
        np.testing.assert_allclose(
            controls,
            np.array([
                [12000.0, 0.0, math.radians(4.0)],
                [12000.0, 0.0, math.radians(4.0)],
            ]),
        )
        # optimize_trajectory runs a feasibility pre-check (one terminal_objective +
        # one terminal_position_constraint) before handing the problem to minimize;
        # drop those calls so we measure only the solver-evaluation rollouts below.
        simulator.calls.clear()
        assert math.isclose(fun(x0), final_time * 1e-4)
        np.testing.assert_allclose(constraints["fun"](x0), np.zeros(3))
        assert len(simulator.calls) == optimizer.n_control_segments * 2
        np.testing.assert_allclose(
            [call["dt"] for call in simulator.calls],
            [final_time / 2.0] * 4,
        )
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    final_time, controls, node_state = optimizer.optimize_trajectory(
        initial_state,
        target_state,
    )

    assert final_time > 0.0
    assert controls.shape == (2, 3)
    assert node_state is None


def test_objective_and_constraint_penalize_unsimulatable_candidates():
    module = load_single_shooting_module()
    optimizer = module.SingleShootingOptimizor(
        sim=FailingGeodeticSimulator(),
        n_control_segments=1,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    candidate = np.array([10.0, 12000.0, 0.0, math.radians(4.0)])

    # An unsimulatable candidate is scored with the infeasible sentinel (not raised),
    # so the solver can still evaluate and step away from it.
    assert optimizer.terminal_objective(candidate, state, state) == module._INFEASIBLE_OBJECTIVE_VALUE
    np.testing.assert_array_equal(
        optimizer.terminal_position_constraint(candidate, state, state),
        np.full(3, module._INFEASIBLE_OBJECTIVE_VALUE),
    )


def test_optimize_trajectory_rejects_unsimulatable_initial_guess():
    module = load_single_shooting_module()
    optimizer = module.SingleShootingOptimizor(
        sim=FailingGeodeticSimulator(),
        n_control_segments=1,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    # The pre-flight feasibility check fails loudly rather than handing the solver a
    # problem whose initial guess cannot even be simulated.
    with pytest.raises(ValueError, match="not simulatable"):
        optimizer.optimize_trajectory(state, state)


def test_single_shooting_optimizor_rejects_invalid_dt():
    module = load_single_shooting_module()

    with pytest.raises(ValueError, match="dt must be positive and finite"):
        module.SingleShootingOptimizor(
            sim=FakeGeodeticSimulator(module),
            n_control_segments=1,
            dt=0.0,
        )


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


class TargetStateSimulator:
    def __init__(self, target_state):
        self.target_state = target_state
        self.calls = []

    def step(self, state, control, dt):
        self.calls.append({"state": state, "control": control, "dt": dt})
        return self.target_state


class FailingGeodeticSimulator:
    def step(self, state, control, dt):
        del state, control, dt
        raise ValueError("simulation stopped: altitude below 0")
