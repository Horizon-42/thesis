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
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        ]),
    )


def test_geodetic_state_to_array_uses_simulation_server_field_names():
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
        np.array([51.1139, -114.0203, 1084.0, 135.0, 0.3, -0.05, 64000.0]),
    )


def test_optimize_trajectory_builds_compatible_minimize_problem(monkeypatch):
    module = load_transcription_module()
    sim_server = FakeSimulationServer(module)
    optimizer = module.TranscriptionOptimizor(sim_server=sim_server, n_segments=2)
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

        defect_values = constraints[0]["fun"](x0)
        final_state_values = constraints[1]["fun"](x0)

        assert defect_values.shape == (optimizer.n_segments * optimizer.state_dim,)
        assert final_state_values.shape == (optimizer.state_dim,)
        assert len(sim_server.calls) == optimizer.n_segments
        assert all(
            isinstance(call["state"], module.GeodeticState)
            for call in sim_server.calls
        )
        assert all(isinstance(call["control"], module.Control) for call in sim_server.calls)
        assert all(call["dt"] == 50.0 for call in sim_server.calls)
        return SimpleNamespace(success=True, x=x0, message="")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    final_time, node_control, node_state = optimizer.optimize_trajectory(
        initial_state,
        target_state,
    )

    assert final_time == 100.0
    assert node_control.shape == (2, 3)
    assert node_state.shape == (2, 7)


def test_optimize_trajectory_raises_when_minimize_fails(monkeypatch):
    module = load_transcription_module()
    optimizer = module.TranscriptionOptimizor(
        sim_server=FakeSimulationServer(module),
        n_segments=1,
    )
    state = module.GeodeticState(1.0, 2.0, 1000.0, 120.0, 0.1, -0.01, 70000.0)

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(success=False, x=np.array([]), message="bad search")

    monkeypatch.setattr(module, "minimize", fake_minimize)

    with pytest.raises(ValueError, match="Optimization failed: bad search"):
        optimizer.optimize_trajectory(state, state)


class FakeSimulationServer:
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
