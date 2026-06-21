import importlib.util
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np

from aerodynamic_model.casadi_simulator import AeroParams


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
        },
    )

    assert solver.name == "solver"
    assert captured["plugin"] == "ipopt"
    assert captured["nlp"]["x"].shape == (1 + 2 * 3 + 2 * 6, 1)
    assert captured["nlp"]["g"].shape == (2 * 6 + 6, 1)
    assert len(lbw) == len(ubw) == 1 + 2 * 3 + 2 * 6
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
        },
    )

    assert captured["nlp"]["p"].is_symbolic()
