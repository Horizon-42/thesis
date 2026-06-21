import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from aerodynamic_model.aircraft_sets import C172
from aerodynamic_model.casadi_simulator import AeroParams
from aerodynamic_model.common import GeodeticState


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "casadi_direct_collocation_optimizer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "casadi_direct_collocation_optimizer",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_imports_from_repo_root():
    module = load_module()

    assert (
        module.make_direct_collocation_solver.__name__
        == "make_direct_collocation_solver"
    )


def test_hermite_simpson_defect_is_zero_for_constant_state_with_zero_rhs():
    module = load_module()
    # A trivial rhs that returns 0 everywhere should produce a zero
    # defect when both knots are identical -- a sanity check that the
    # algebraic form is wired correctly.
    rhs_func = ca.Function(
        "rhs_zero",
        [ca.SX.sym("x", 7), ca.SX.sym("u", 3), ca.SX.sym("p", 6)],
        [ca.SX.zeros(7)],
    )
    x = ca.DM([10.0, -5.0, 1000.0, 60.0, 0.1, -0.01, 70000.0])
    u = ca.DM([1000.0, 0.0, 1.0])
    p = ca.DM([122.6, 1.5, 0.02, 0.04, 0.9, 0.1])

    defect = module.hermite_simpson_defect_expr(rhs_func, x, x, u, p, 0.5)

    np.testing.assert_allclose(
        np.array(defect, dtype=float).reshape(-1),
        np.zeros(7),
    )


def test_make_direct_collocation_solver_builds_symbolic_nlp(monkeypatch):
    module = load_module()
    captured = {}

    def fake_nlpsol(name, plugin, nlp):
        captured["plugin"] = plugin
        captured["nlp"] = nlp
        return SimpleNamespace(name=name)

    monkeypatch.setattr(module.ca, "nlpsol", fake_nlpsol)

    solver, lbw, ubw, lbg, ubg = module.make_direct_collocation_solver(
        segment_num=2,
        aero_params_obj=AeroParams(S=122.6),
        aircraft_meta={
            "max_thrust": 240000.0,
            "min_load_factor": 0.5,
            "max_load_factor": 2.0,
            "min_terminal_speed": 60.0,
            "min_altitude": 25.0,
        },
    )

    assert solver.name == "solver"
    assert captured["plugin"] == "ipopt"
    # 2 controls × 3 + 2 states × 6 = 18 decision variables.
    assert captured["nlp"]["x"].shape == (2 * 3 + 2 * 6, 1)
    # 2 segment defects × 6 + 1 terminal × 6 = 18 equality constraints.
    assert captured["nlp"]["g"].shape == (2 * 6 + 6, 1)
    # 7 (initial) + 7 (target) + 1 (duration) = 15 parameter slots.
    assert captured["nlp"]["p"].shape == (7 + 7 + 1, 1)
    assert len(lbw) == len(ubw) == 2 * 3 + 2 * 6
    assert len(lbg) == len(ubg) == 2 * 6 + 6


def test_optimize_trajectory_solves_real_ipopt_problem():
    """End-to-end: generate a feasible target by forward-integrating the
    same continuous RHS with the optimiser's nominal control, then ask
    the optimiser to recover that target.  This avoids cross-frame
    discrepancies from the re-anchored stepper that the previous
    optimiser test had to absorb."""
    module = load_module()
    duration = 2.0
    n_segments = 4

    speed = C172.terminal_speed_kt * 0.51444 + 10.0
    state = GeodeticState(
        latitude=51.1139,
        longitude=-114.0203,
        altitude=1000.0,
        V=speed,
        psi=0.0,
        gamma=0.0,
        m=C172.mass_kg,
    )

    optimizer = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments,
        dt=0.2,
        max_duration=duration,
        aircraft=C172,
    )

    target = _propagate_with_fixed_enu_rhs(state, optimizer, duration)

    duration_opt, controls, states = optimizer.optimize_trajectory(
        state,
        target,
        duration=duration,
    )

    assert optimizer.solver.stats()["success"]
    assert duration_opt == duration
    assert controls.shape == (n_segments, 3)
    assert states.shape == (n_segments, 6)

    # The terminal node should match the target after the round-trip
    # geodetic <-> ENU <-> dynamics <-> ENU <-> geodetic.  Tolerances
    # have to absorb both IPOPT's defect convergence and the cubic
    # interpolation residual of Hermite-Simpson, hence the 1e-3 atol.
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
        atol=1e-3,
    )


def test_optimize_trajectory_uses_supplied_initial_guess():
    module = load_module()
    optimizer = object.__new__(module.CasadiDirectCollocationOptimizer)
    optimizer.n_segments = 1
    optimizer.max_duration = 5.0
    optimizer.lbw = [0.0] * 9
    optimizer.ubw = [100.0] * 9
    optimizer.lbg = [0.0] * 12
    optimizer.ubg = [0.0] * 12
    optimizer._build_initial_guess = lambda initial, target: [99.0] * 9
    supplied_guess = [float(i) for i in range(9)]

    class FakeSolver:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return {"x": ca.DM([1.0, 2.0, 3.0, 0.0, 0.0, 1000.0, 40.0, 0.1, 0.0])}

        def stats(self):
            return {"success": True}

    optimizer.solver = FakeSolver()
    state = GeodeticState(51.0, -114.0, 1000.0, 40.0, 0.1, 0.0, C172.mass_kg)

    optimizer.optimize_trajectory(
        state,
        state,
        duration=2.0,
        initial_guess=supplied_guess,
    )

    assert optimizer.solver.kwargs["x0"] == supplied_guess


def test_make_direct_collocation_solver_free_time_builds_symbolic_nlp(monkeypatch):
    """Free-time NLP has one extra decision variable (T) and one extra
    bound slot.  Constraint count is unchanged from the fixed-time NLP
    because T is on the variable side, not the constraint side."""
    module = load_module()
    captured = {}

    def fake_nlpsol(name, plugin, nlp):
        captured["plugin"] = plugin
        captured["nlp"] = nlp
        return SimpleNamespace(name=name)

    monkeypatch.setattr(module.ca, "nlpsol", fake_nlpsol)

    solver, lbw, ubw, lbg, ubg = module.make_direct_collocation_solver_free_time(
        segment_num=2,
        aero_params_obj=AeroParams(S=122.6),
        aircraft_meta={
            "max_thrust": 240000.0,
            "min_load_factor": 0.5,
            "max_load_factor": 2.0,
            "min_terminal_speed": 60.0,
            "min_altitude": 25.0,
            "min_duration": 1.0,
        },
    )

    assert solver.name == "solver"
    assert captured["plugin"] == "ipopt"
    # 2 controls × 3 + 2 states × 6 + 1 duration = 19 decision variables.
    assert captured["nlp"]["x"].shape == (2 * 3 + 2 * 6 + 1, 1)
    # Constraints unchanged: 2 segment defects × 6 + 1 terminal × 6 = 18.
    assert captured["nlp"]["g"].shape == (2 * 6 + 6, 1)
    # 7 (initial) + 7 (target) + 1 (max_duration scale) = 15 parameter slots.
    assert captured["nlp"]["p"].shape == (7 + 7 + 1, 1)
    assert len(lbw) == len(ubw) == 2 * 3 + 2 * 6 + 1


def test_optimize_free_time_solves_real_ipopt_problem():
    """The free-time solver should find an arrival time strictly less
    than ``max_duration`` and reach the target without bisection."""
    module = load_module()
    n_segments = 4
    feasible_duration = 2.0
    max_duration = 6.0  # generous upper bound

    speed = C172.terminal_speed_kt * 0.51444 + 10.0
    state = GeodeticState(
        latitude=51.1139,
        longitude=-114.0203,
        altitude=1000.0,
        V=speed,
        psi=0.0,
        gamma=0.0,
        m=C172.mass_kg,
    )

    optimizer = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments,
        dt=0.2,
        max_duration=max_duration,
        aircraft=C172,
    )
    target = _propagate_with_fixed_enu_rhs(state, optimizer, feasible_duration)

    final_time, controls, states = optimizer.optimize_free_time(
        state, target, max_duration,
    )

    assert optimizer.free_time_solver.stats()["success"]
    assert controls.shape == (n_segments, 3)
    assert states.shape == (n_segments, 6)
    # The optimum should sit BELOW the upper bound -- otherwise the
    # solver did not actually shrink T (which would mean the minimum-
    # time objective is not active).
    assert final_time < max_duration - 0.1
    # And it should be close to the truly feasible duration we used to
    # generate the target.  IPOPT may report something slightly larger
    # because of the control-effort regulariser, but it should be in
    # the right neighbourhood.
    assert abs(final_time - feasible_duration) < 0.5

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
        atol=1e-3,
    )


def test_optimizer_does_not_expose_optimize_time_to_target():
    # Direct collocation only ships ``optimize_free_time`` -- the
    # CasadiOptimizer-style bisection wrapper has been removed since it
    # is no longer needed.  The backend dispatcher knows to call the
    # right method per optimiser.
    module = load_module()
    assert not hasattr(module.CasadiDirectCollocationOptimizer, "optimize_time_to_target")


def test_optimize_trajectory_raises_on_solver_failure():
    module = load_module()
    optimizer = object.__new__(module.CasadiDirectCollocationOptimizer)
    optimizer.n_segments = 1
    optimizer.max_duration = 5.0
    optimizer.lbw = [0.0] * 9
    optimizer.ubw = [100.0] * 9
    optimizer.lbg = [0.0] * 12
    optimizer.ubg = [0.0] * 12
    optimizer._build_initial_guess = lambda initial, target: [0.0] * 9

    class FailingSolver:
        def __call__(self, **kwargs):
            return {"x": ca.DM.zeros(9)}

        def stats(self):
            return {"success": False, "return_status": "infeasible"}

    optimizer.solver = FailingSolver()
    state = GeodeticState(51.0, -114.0, 1000.0, 40.0, 0.1, 0.0, C172.mass_kg)

    with pytest.raises(ValueError, match="infeasible"):
        optimizer.optimize_trajectory(state, state, duration=2.0)


def _propagate_with_fixed_enu_rhs(
    state: GeodeticState,
    optimizer,
    duration: float,
) -> GeodeticState:
    """Forward RK4 of the same continuous fixed-ENU RHS the optimiser
    uses.  Anchoring at the initial geodetic point keeps the propagator
    and the optimiser in the same frame."""
    from aerodynamic_model.casadi_simulator import make_dynamics_model

    rhs_func = make_dynamics_model()["rhs_func"]
    aero_params = ca.DM([
        optimizer.aero_params.S,
        optimizer.aero_params.Cl_max,
        optimizer.aero_params.Cd0,
        optimizer.aero_params.k,
        optimizer.aero_params.stall_threshold,
        optimizer.aero_params.k_stall,
    ])
    u = ca.DM([C172.approach_thrust_guess_n, 0.0, 1.0])

    anchor_lat = state.latitude
    anchor_lon = state.longitude
    # Build a small CasADi function in the test (not the module) so the
    # test does not lean on private helpers.
    lat_s, lon_s, alt_s = ca.SX.sym("lat"), ca.SX.sym("lon"), ca.SX.sym("alt")
    rlat_s, rlon_s = ca.SX.sym("rlat"), ca.SX.sym("rlon")
    from aerodynamic_model.casadi_coordinates_converter import (
        enu_to_geodetic_expr,
        geodetic_to_enu_expr,
    )
    e_s, n_s, u_s = geodetic_to_enu_expr(lat_s, lon_s, alt_s, rlat_s, rlon_s, 0.0)
    to_enu = ca.Function("to_enu", [lat_s, lon_s, alt_s, rlat_s, rlon_s], [ca.vertcat(e_s, n_s, u_s)])

    e_in = ca.SX.sym("e")
    n_in = ca.SX.sym("n")
    u_in = ca.SX.sym("u")
    lat_o, lon_o, alt_o = enu_to_geodetic_expr(e_in, n_in, u_in, rlat_s, rlon_s, 0.0)
    to_geo = ca.Function("to_geo", [e_in, n_in, u_in, rlat_s, rlon_s], [ca.vertcat(lat_o, lon_o, alt_o)])

    enu = to_enu(state.latitude, state.longitude, state.altitude, anchor_lat, anchor_lon)
    x = ca.DM([
        float(enu[0]),
        float(enu[1]),
        state.altitude,
        state.V,
        state.psi,
        state.gamma,
        state.m,
    ])
    n_steps = max(1, int(round(duration / 0.05)))
    dt = duration / n_steps
    for _ in range(n_steps):
        k1 = rhs_func(x, u, aero_params)
        k2 = rhs_func(x + 0.5 * dt * k1, u, aero_params)
        k3 = rhs_func(x + 0.5 * dt * k2, u, aero_params)
        k4 = rhs_func(x + dt * k3, u, aero_params)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    geo = to_geo(float(x[0]), float(x[1]), float(x[2]), anchor_lat, anchor_lon)
    return GeodeticState(
        latitude=float(geo[0]),
        longitude=float(geo[1]),
        altitude=float(geo[2]),
        V=float(x[3]),
        psi=float(x[4]),
        gamma=float(x[5]),
        m=float(x[6]),
    )
