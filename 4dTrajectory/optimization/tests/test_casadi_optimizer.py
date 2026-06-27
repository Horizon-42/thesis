import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from geokit import kt_to_ms
from aircraft.aircraft_sets import C172
from aircraft.aero_params import AeroParams
from aerodynamic_model.common import GeodeticState


def load_casadi_optimizer_module():
    module_path = Path(__file__).resolve().parents[1] / "casadi_optimizer.py"
    spec = importlib.util.spec_from_file_location("casadi_optimizer", module_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def test_casadi_optimizer_imports_from_repo_root():
    module = load_casadi_optimizer_module()

    assert module.make_multiple_shooting_solver.__name__ == "make_multiple_shooting_solver"


def test_segment_integrate_expr_uses_integer_substeps_for_numeric_duration():
    module = load_casadi_optimizer_module()
    calls = []

    def fake_step_func(*, x_geo, u, aero_params, dt):
        calls.append(dt)
        return {"x_geo_next": x_geo + 1.0}

    result = module.segement_integrate_expr(
        fake_step_func,
        ca.DM.zeros(6),
        ca.DM.zeros(3),
        ca.DM.zeros(6),
        dt=0.4,
        duration=1.0,
        n_steps=3,
    )

    assert len(calls) == 3
    np.testing.assert_allclose(calls, [1.0 / 3.0] * 3)
    np.testing.assert_allclose(
        np.array(result, dtype=float).reshape(-1),
        np.ones(6) * 3,
    )


def test_make_multiple_shooting_solver_builds_symbolic_nlp(monkeypatch):
    module = load_casadi_optimizer_module()
    captured = {}

    def fake_nlpsol(name, plugin, nlp):
        captured["name"] = name
        captured["plugin"] = plugin
        captured["nlp"] = nlp
        return SimpleNamespace(name=name)

    monkeypatch.setattr(module.ca, "nlpsol", fake_nlpsol)

    solver, lbw, ubw, lbg, ubg = module.make_multiple_shooting_solver(
        segment_num=2,
        dt=0.5,
        max_duration=10.0,
        aero_params_obj=AeroParams(S=122.6),
        aircraft_meta={
            "max_thrust": 240000.0,
            "min_load_factor": 0.0,
            "max_load_factor": 3.0,
            "min_terminal_speed": 0.0,
            "min_altitude": 0.0,
        },
    )

    assert solver.name == "solver"
    assert captured["plugin"] == "ipopt"
    assert captured["nlp"]["x"].shape == (2 * 3 + 2 * 6, 1)
    assert captured["nlp"]["g"].shape == (2 * 6 + 6, 1)
    assert captured["nlp"]["p"].shape == (7 + 7 + 1, 1)
    assert len(lbw) == len(ubw) == 2 * 3 + 2 * 6
    assert len(lbg) == len(ubg) == 2 * 6 + 6


def test_make_multiple_shooting_solver_uses_pure_symbolic_parameters(monkeypatch):
    module = load_casadi_optimizer_module()
    captured = {}

    def fake_segment_integrate(step_func, x_start, u, aero_params, dt, duration, n_steps=None):
        return ca.SX.sym("predicted_next", 6)

    def fake_nlpsol(name, plugin, nlp):
        captured["nlp"] = nlp
        return SimpleNamespace(name=name)

    monkeypatch.setattr(module, "segement_integrate_expr", fake_segment_integrate)
    monkeypatch.setattr(module.ca, "nlpsol", fake_nlpsol)

    module.make_multiple_shooting_solver(
        segment_num=1,
        dt=0.5,
        max_duration=10.0,
        aero_params_obj=AeroParams(S=122.6),
        aircraft_meta={
            "max_thrust": 240000.0,
            "min_load_factor": 0.0,
            "max_load_factor": 3.0,
            "min_terminal_speed": 0.0,
            "min_altitude": 0.0,
        },
    )

    assert captured["nlp"]["p"].is_symbolic()


def test_optimize_trajectory_runs_real_ipopt_for_fixed_time_target():
    module = load_casadi_optimizer_module()
    duration = 0.2
    speed = kt_to_ms(C172.terminal_speed_kt) + 5.0
    state = GeodeticState(
        latitude=51.1139,
        longitude=-114.0203,
        altitude=1000.0,
        V=speed,
        psi=0.0,
        gamma=0.0,
        m=C172.mass_kg,
    )
    optimizer = module.CasadiOptimizer(
        n_segments=2,
        dt=0.1,
        max_duration=0.4,
        aircraft=C172,
    )
    target = propagate_with_nominal_control(module, optimizer, state, duration)

    duration_opt, controls, states = optimizer.optimize_trajectory(
        state,
        target,
        duration=duration,
    )

    assert optimizer.solver.stats()["success"]
    assert duration_opt == duration
    assert controls.shape == (2, 3)
    assert states.shape == (2, 6)
    np.testing.assert_allclose(
        states[-1],
        np.array([
            target.latitude,
            target.longitude,
            target.altitude,
            target.V,
            target.psi,
            target.gamma,
        ]),
        atol=1e-5,
    )


def test_optimize_trajectory_uses_supplied_initial_guess():
    module = load_casadi_optimizer_module()
    optimizer = object.__new__(module.CasadiOptimizer)
    optimizer.n_segments = 1
    optimizer.max_duration = 10.0
    optimizer.lbw = [0.0] * 9
    optimizer.ubw = [100.0] * 9
    optimizer.lbg = [0.0] * 12
    optimizer.ubg = [0.0] * 12
    optimizer.build_initial_guess = lambda initial, target: [99.0] * 9
    supplied_guess = [float(i) for i in range(9)]

    class FakeSolver:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"x": ca.DM([1.0, 2.0, 3.0, 51.0, -114.0, 1000.0, 40.0, 0.1, 0.0])}

        def stats(self):
            return {"success": True}

    optimizer.solver = FakeSolver()
    state = GeodeticState(51.0, -114.0, 1000.0, 40.0, 0.1, 0.0, C172.mass_kg)

    optimizer.optimize_trajectory(
        state,
        state,
        duration=5.0,
        initial_guess=supplied_guess,
    )

    assert optimizer.solver.kwargs["x0"] == supplied_guess


def test_optimize_time_to_target_reuses_solver_with_duration_parameter():
    module = load_casadi_optimizer_module()
    calls = []

    class FakeOptimizer:
        dt = 1.0
        solution_to_initial_guess = staticmethod(module.CasadiOptimizer.solution_to_initial_guess)

        def optimize_trajectory(self, initial_state, target_state, duration=None, initial_guess=None):
            calls.append((duration, initial_guess))
            if duration < 6.0:
                raise ValueError("too short")
            return (
                duration,
                np.full((1, 3), duration),
                np.full((1, 6), duration),
            )

    result = module.CasadiOptimizer.optimize_time_to_target(
        FakeOptimizer(),
        object(),
        object(),
        max_duration=10.0,
        max_attempts=3,
    )

    assert result[0] == 6.25
    assert [call[0] for call in calls] == [10.0, 5.0, 7.5, 6.25]
    assert calls[0][1] is None
    np.testing.assert_allclose(calls[1][1], [10.0] * 9)
    np.testing.assert_allclose(calls[2][1], [10.0] * 9)
    np.testing.assert_allclose(calls[3][1], [7.5] * 9)


def test_optimize_time_to_target_raises_when_first_solve_fails():
    module = load_casadi_optimizer_module()
    calls = []

    class FakeOptimizer:
        solution_to_initial_guess = staticmethod(module.CasadiOptimizer.solution_to_initial_guess)

        def optimize_trajectory(self, initial_state, target_state, duration=None, initial_guess=None):
            calls.append((duration, initial_guess))
            if duration < 7.0 or duration == 10.0:
                raise ValueError("failed")
            return (
                duration,
                np.full((1, 3), duration),
                np.full((1, 6), duration),
            )

    with pytest.raises(ValueError, match="failed"):
        module.CasadiOptimizer.optimize_time_to_target(
            FakeOptimizer(),
            object(),
            object(),
            max_duration=10.0,
            max_attempts=2,
        )

    assert calls == [(10.0, None)]


def propagate_with_nominal_control(module, optimizer, state, duration):
    step_func = module.make_geo_step_from_enu_integrator()["step_func"]
    x = ca.DM(optimizer.geo_state_to_decision_vector(state))
    u = ca.DM([C172.approach_thrust_guess_n, 0.0, 1.0])
    aero_params = ca.DM([
        optimizer.aero_params.S,
        optimizer.aero_params.Cl_max,
        optimizer.aero_params.Cd0,
        optimizer.aero_params.k,
        optimizer.aero_params.stall_threshold,
        optimizer.aero_params.k_stall,
    ])
    segment_substeps = max(1, math.ceil((optimizer.max_duration / optimizer.n_segments) / optimizer.dt))
    step_dt = duration / optimizer.n_segments / segment_substeps
    for _ in range(optimizer.n_segments):
        for _ in range(segment_substeps):
            x = step_func(
                x_geo=x,
                u=u,
                aero_params=aero_params,
                dt=step_dt,
            )["x_geo_next"]
    return GeodeticState(*np.array(x, dtype=float).reshape(-1).tolist())
