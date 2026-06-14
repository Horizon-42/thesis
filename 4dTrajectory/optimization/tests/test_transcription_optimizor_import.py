import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_transcription_module():
    module_path = Path(__file__).resolve().parents[1] / "transcription_optimizor.py"
    spec = importlib.util.spec_from_file_location(
        "transcription_optimizor",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def test_transcription_optimizor_imports_from_repo_root():
    module = load_transcription_module()

    assert module.TranscriptionOptimizor.__name__ == "TranscriptionOptimizor"


def test_unpack_z_uses_final_time_controls_and_states_layout():
    module = load_transcription_module()
    optimizer = module.TranscriptionOptimizor(sim_server=object(), n_segments=2)
    z = np.arange(
        1 + optimizer.n_segments * optimizer.control_dim +
        optimizer.n_segments * optimizer.state_dim,
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


def test_geodetic_state_to_array_uses_geodetic_state_field_names():
    module = load_transcription_module()
    state = module.GeodeticState(
        latitude=51.1139,
        longitude=-114.0203,
        altitude=1084.0,
        V=135.0,
        psi=0.3,
        gamma=-0.05,
        m=64000.0,
    )

    array = module.TranscriptionOptimizor.geodetic_state_to_array(state)

    np.testing.assert_allclose(
        array,
        np.array([51.1139, -114.0203, 1084.0, 135.0, 0.3, -0.05]),
    )


def test_array_to_geodetic_state_restores_fixed_mass():
    module = load_transcription_module()

    state = module.TranscriptionOptimizor.array_to_geodetic_state(
        np.array([51.1139, -114.0203, 1084.0, 135.0, 0.3, -0.05]),
        mass=64000.0,
    )

    assert state == module.GeodeticState(
        latitude=51.1139,
        longitude=-114.0203,
        altitude=1084.0,
        V=135.0,
        psi=0.3,
        gamma=-0.05,
        m=64000.0,
    )


def test_optimize_trajectory_builds_compatible_minimize_problem(monkeypatch):
    module = load_transcription_module()
    geodetic_simulator = FakeGeodeticSimulator(module)
    optimizer = module.TranscriptionOptimizor(
        sim_server=geodetic_simulator,
        n_segments=2,
    )
    initial_state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)
    target_state = module.GeodeticState(3.0, 4.0, 900.0, 115.0, 0.2, -0.02, 70000.0)

    def fake_minimize(fun, x0, bounds, constraints, method, options):
        assert len(x0) == 1 + optimizer.n_segments * (
            optimizer.control_dim + optimizer.state_dim
        )
        assert len(bounds) == len(x0)
        assert method == "SLSQP"
        assert options == {"maxiter": 1000, "ftol": 1e-6}
        assert fun(x0) == 100.0

        _, node_control, node_state = optimizer.unpack_z(x0)
        np.testing.assert_allclose(
            node_control,
            np.array([[12000.0, 0.0, 0.0], [12000.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(
            node_state,
            np.array([
                [2.0, 3.0, 950.0, 117.5, 0.15, -0.015],
                [3.0, 4.0, 900.0, 115.0, 0.2, -0.02],
            ]),
        )

        state_bounds = bounds[
            1 + optimizer.n_segments * optimizer.control_dim:
        ]
        assert state_bounds[:optimizer.state_dim] == [
            (-90.0, 90.0),
            (-180.0, 180.0),
            (0.0, None),
            (1.0, None),
            (None, None),
            (-(np.pi / 2.0 - 1e-3), np.pi / 2.0 - 1e-3),
        ]

        defect_values = constraints[0]["fun"](x0)
        final_state_values = constraints[1]["fun"](x0)

        assert defect_values.shape == (optimizer.n_segments * optimizer.state_dim,)
        assert final_state_values.shape == (optimizer.state_dim,)
        assert len(geodetic_simulator.calls) == optimizer.n_segments
        assert all(
            isinstance(call["state"], module.GeodeticState)
            for call in geodetic_simulator.calls
        )
        assert all(
            isinstance(call["control"], module.Control)
            for call in geodetic_simulator.calls
        )
        assert all(call["state"].m == initial_state.m for call in geodetic_simulator.calls)
        assert all(call["dt"] == 50.0 for call in geodetic_simulator.calls)
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    final_time, node_control, node_state = optimizer.optimize_trajectory(
        initial_state,
        target_state,
    )

    assert final_time == 100.0
    assert node_control.shape == (2, 3)
    assert node_state.shape == (2, 6)


def test_defect_constraints_turn_simulator_failures_into_infeasible_residual(monkeypatch):
    module = load_transcription_module()
    optimizer = module.TranscriptionOptimizor(
        sim_server=FailingGeodeticSimulator(),
        n_segments=2,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fake_minimize(fun, x0, bounds, constraints, method, options):
        del fun, bounds, method, options
        defect_values = constraints[0]["fun"](x0)

        assert defect_values.shape == (optimizer.n_segments * optimizer.state_dim,)
        np.testing.assert_allclose(defect_values, np.full(defect_values.shape, 1e9))
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    final_time, node_control, node_state = optimizer.optimize_trajectory(state, state)

    assert final_time == 100.0
    assert node_control.shape == (2, 3)
    assert node_state.shape == (2, 6)


def test_default_runway_guess_stays_inside_simulator_domain(monkeypatch):
    module = load_transcription_module()
    optimizer = module.TranscriptionOptimizor(
        sim_server=module.GeodeticSimulator(),
        n_segments=10,
    )
    initial_state = module.GeodeticState(
        latitude=35.878659,
        longitude=-78.7873,
        altitude=1000.0,
        V=120.0,
        psi=0.0,
        gamma=0.0,
        m=78000.0,
    )
    target_state = module.GeodeticState(
        latitude=35.874,
        longitude=-78.802,
        altitude=111.86,
        V=70.0,
        psi=np.radians(45.0),
        gamma=np.radians(-4.0),
        m=78000.0,
    )

    def fake_minimize(fun, x0, bounds, constraints, method, options):
        del fun, bounds, method, options
        defect_values = constraints[0]["fun"](x0)

        assert np.all(np.isfinite(defect_values))
        assert not np.all(defect_values == 1e9)
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    optimizer.optimize_trajectory(initial_state, target_state)


def test_optimize_trajectory_rejects_unsimulatable_endpoint_state():
    module = load_transcription_module()
    optimizer = module.TranscriptionOptimizor(
        sim_server=FakeGeodeticSimulator(module),
        n_segments=1,
    )
    initial_state = module.GeodeticState(
        latitude=1.0,
        longitude=2.0,
        altitude=1000.0,
        V=0.0,
        psi=0.1,
        gamma=-0.01,
        m=70000.0,
    )
    target_state = module.GeodeticState(1.0, 2.0, 900.0, 115.0, 0.2, -0.02, 70000.0)

    with pytest.raises(ValueError, match="initial_state.V must be >= 1.0"):
        optimizer.optimize_trajectory(initial_state, target_state)


def test_optimize_trajectory_raises_when_minimize_fails(monkeypatch):
    module = load_transcription_module()
    optimizer = module.TranscriptionOptimizor(
        sim_server=FakeGeodeticSimulator(module),
        n_segments=1,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(success=False, x=np.array([]), message="bad search")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    with pytest.raises(ValueError, match="Optimization failed: bad search"):
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
        raise ValueError("Required step size is less than spacing between numbers.")
