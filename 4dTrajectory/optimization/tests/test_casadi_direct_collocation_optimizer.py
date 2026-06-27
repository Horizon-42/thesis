import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from aerodynamic_model.aircraft_sets import A320, C172
from aerodynamic_model.casadi_simulator import AeroParams, aero_params_for_aircraft
from aerodynamic_model.common import GeodeticState

# Keep the optimisation dir importable for modules loaded via importlib.
_OPTIMIZATION_DIR = Path(__file__).resolve().parents[1]
if str(_OPTIMIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(_OPTIMIZATION_DIR))


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


def test_terminal_bank_constraint_expr_matches_coordinated_bank():
    module = load_module()
    # Heading turns 0 -> 0.05 rad over state_h = 2 s at V = 100, gamma = 0:
    # expr = V cos(gamma) * (dpsi / state_h) = 100 * 1 * 0.05 / 2 = 2.5 (= g*tan(mu)).
    n_prev = ca.DM([0.0, 0.0, 1000.0, 100.0, 0.0, 0.0])
    n_term = ca.DM([0.0, 0.0, 1000.0, 100.0, 0.05, 0.0])
    start = ca.DM([0.0, 0.0, 1000.0, 100.0, 0.0, 0.0, 70000.0])

    expr, lb, ub = module.terminal_bank_constraint_expr(
        [n_prev, n_term], start, 2.0, math.radians(5.0),
    )

    assert float(expr) == pytest.approx(2.5)
    assert ub == pytest.approx(9.81 * math.tan(math.radians(5.0)))
    assert lb == pytest.approx(-ub)


def test_control_smoothness_cost_weights_segment_changes():
    module = load_module()
    meta = {"max_thrust": 240000.0, "min_load_factor": 0.5, "max_load_factor": 2.0}
    # Two control segments; bank changes by pi/2 (scaled diff = 1.0), thrust
    # and load unchanged.  With w_bank = 3, the single pair contributes
    # (3 * 1)^2 = 9, averaged over 1 pair => 9.
    u0 = ca.DM([0.0, 0.0, 1.0])
    u1 = ca.DM([0.0, math.pi / 2.0, 1.0])
    cost = module._control_smoothness_cost([u0, u1], meta, (0.1, 3.0, 1.0))
    assert float(cost) == pytest.approx(9.0)
    # No change between segments => zero smoothness cost.
    assert float(module._control_smoothness_cost([u0, u0], meta, (0.1, 3.0, 1.0))) == pytest.approx(0.0)


def test_unwrap_target_heading_picks_short_turn():
    module = load_module()
    # init +170 deg, target given as -170 deg: the short turn is +20 deg, so
    # the target heading must unwrap to +190 deg (same heading, other branch).
    init = [0.0, 0.0, 0.0, 0.0, math.radians(170.0), 0.0, 0.0]
    tgt = [0.0, 0.0, 0.0, 0.0, math.radians(-170.0), 0.0, 0.0]
    assert math.degrees(module._unwrap_target_heading(init, tgt)[4]) == pytest.approx(190.0)
    # already within +-180 deg: unchanged.
    init2 = [0.0, 0.0, 0.0, 0.0, math.radians(10.0), 0.0, 0.0]
    tgt2 = [0.0, 0.0, 0.0, 0.0, math.radians(40.0), 0.0, 0.0]
    assert math.degrees(module._unwrap_target_heading(init2, tgt2)[4]) == pytest.approx(40.0)


def test_optimize_handles_heading_wrap_across_180_deg():
    """A turn whose target heading sits on the other side of the +-180 branch
    cut (e.g. KRDU R32 SINNO: -135 deg -> +135 deg) must still solve -- the
    target heading is unwrapped to the shortest turn."""
    import casadi as ca
    from aerodynamic_model.casadi_simulator import (
        AeroParams, make_geodetic_step_integrator,
    )

    module = load_module()
    ap = aero_params_for_aircraft(C172)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    step = make_geodetic_step_integrator(transport="approx")["step_func"]

    duration = 2.0
    speed = C172.terminal_speed_kt * 0.51444 + 10.0
    init = GeodeticState(51.1139, -114.0203, 1000.0, speed,
                         math.radians(179.0), 0.0, C172.mass_kg)
    # Propagate a left turn so the heading just crosses +180 deg.
    u = ca.DM([C172.approach_thrust_guess_n, math.radians(10.0), 1.05])
    x = ca.DM([init.latitude, init.longitude, init.altitude,
               init.V, init.psi, init.gamma, init.m])
    for _ in range(int(duration / 0.05)):
        x = step(x_geo=x, u=u, aero_params=aero, dt=0.05)["x_geo_next"]
    a = np.array(x).reshape(-1)
    # Supply the target heading WRAPPED into [-180, 180] deg, as the frontend would.
    wrapped_psi = (a[4] + math.pi) % (2 * math.pi) - math.pi
    assert wrapped_psi < 0  # crossed +180 -> now negative branch
    target = GeodeticState(a[0], a[1], a[2], a[3], wrapped_psi, a[5], C172.mass_kg)

    optimizer = module.CasadiDirectCollocationOptimizer(
        n_segments=4, dt=0.2, max_duration=6.0, aircraft=C172,
        max_terminal_bank_deg=89.0,
    )
    _, _, states = optimizer.optimize_trajectory(init, target, duration=duration)
    assert optimizer.solver.stats()["success"]
    # Terminal heading equals the propagated heading modulo 360 deg.
    diff = (math.degrees(states[-1][4]) - math.degrees(a[4])) % 360.0
    assert min(diff, 360.0 - diff) < 1.0


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
    # 2 segment defects × 6 + 1 terminal × 6 = 18 equality constraints,
    # plus 1 terminal bank inequality.
    assert captured["nlp"]["g"].shape == (2 * 6 + 6 + 1, 1)
    # 7 (initial) + 7 (target) + 1 (duration) = 15 parameter slots.
    assert captured["nlp"]["p"].shape == (7 + 7 + 1, 1)
    assert len(lbw) == len(ubw) == 2 * 3 + 2 * 6
    assert len(lbg) == len(ubg) == 2 * 6 + 6 + 1


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

    target = _propagate_with_geodetic_rhs(state, optimizer, duration)

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
    optimizer.state_substeps = 1
    optimizer.collocation_scheme = "hermiteSimpson"
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
    # 2 segment defects × 6 + 1 terminal × 6 = 18 equality constraints,
    # plus 1 terminal bank inequality.
    assert captured["nlp"]["g"].shape == (2 * 6 + 6 + 1, 1)
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
    target = _propagate_with_geodetic_rhs(state, optimizer, feasible_duration)

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


def test_optimize_free_time_raw_is_playback_consistent():
    """The geodetic transcription shares ONE continuous RHS with the
    playback integrator, so the raw (un-polished) controls already land
    at the target when replayed -- no multiple-shooting polish needed.

    Replaying the piecewise-constant controls through the geodetic
    stepper (a fine RK4 of the same continuous dynamics the collocation
    defects approximate) should reach the target up to the Hermite-Simpson
    discretisation error, which is small on this short horizon.
    """
    module = load_module()
    n_segments = 4
    feasible_duration = 2.0
    max_duration = 6.0

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
    target = _propagate_with_geodetic_rhs(state, optimizer, feasible_duration)

    final_time, controls, _ = optimizer.optimize_free_time(
        state, target, max_duration,
    )

    # Replay the raw controls through the continuous geodetic stepper --
    # this is the playback path the frontend's dynamics now share.
    from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator

    step = make_geodetic_step_integrator(transport="approx")["step_func"]
    aero_params = ca.DM([
        optimizer.aero_params.S,
        optimizer.aero_params.Cl_max,
        optimizer.aero_params.Cd0,
        optimizer.aero_params.k,
        optimizer.aero_params.stall_threshold,
        optimizer.aero_params.k_stall,
    ])
    x = ca.DM([
        state.latitude, state.longitude, state.altitude,
        state.V, state.psi, state.gamma, state.m,
    ])
    segment_h = final_time / n_segments
    for k in range(n_segments):
        u = ca.DM([float(controls[k, 0]), float(controls[k, 1]), float(controls[k, 2])])
        remaining = segment_h
        while remaining > 1e-9:
            dt = min(0.05, remaining)
            x = step(x_geo=x, u=u, aero_params=aero_params, dt=dt)["x_geo_next"]
            remaining -= dt

    playback = np.array(x).reshape(-1)
    # Raw HS controls land on target up to the collocation discretisation
    # error: metre-level horizontally on this short horizon, with NO
    # polish step.  (The old fixed-ENU path needed polish to get here.)
    assert abs(playback[0] - target.latitude) < 1e-3
    assert abs(playback[1] - target.longitude) < 1e-3
    assert abs(playback[2] - target.altitude) < 5.0
    assert abs(playback[3] - target.V) < 1.0


def test_dense_state_keeps_playback_consistent_on_long_coarse_control_horizon():
    """The whole point of the dense-state transcription: even with a COARSE
    control mesh over a LONG horizon (where a single HS step per control
    segment would drift kilometres), replaying the raw controls through the
    re-anchored RK4 playback simulator still lands within a few metres --
    because the state is collocated on N*M sub-intervals (auto-selected M).
    """
    from aerodynamic_model.aircraft_sets import A320
    from aerodynamic_model.casadi_simulator import (
        AeroParams, CasadiSimulator, make_geodetic_step_integrator,
    )
    from aerodynamic_model.common import LoadFactorControl

    module = load_module()
    n_segments = 10
    horizon = 150.0
    max_duration = 260.0
    state = GeodeticState(35.60, -78.50, 1600.0, 90.0,
                          math.radians(-40), math.radians(-3), A320.mass_kg)

    optimizer = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=max_duration, aircraft=A320,
    )
    # M is auto-selected to keep the state step a few seconds even though
    # the control segments are tens of seconds long.
    assert optimizer.state_substeps >= 4

    # Feasible target: propagate the A320 nominal approach control for the
    # horizon through the geodetic dynamics (on its own dynamics manifold).
    ap = aero_params_for_aircraft(A320)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    gstep = make_geodetic_step_integrator(transport="approx")["step_func"]
    u_nom = ca.DM([A320.approach_thrust_guess_n, 0.0, 1.0])
    xp = ca.DM([state.latitude, state.longitude, state.altitude,
                state.V, state.psi, state.gamma, state.m])
    for _ in range(int(horizon / 0.05)):
        xp = gstep(x_geo=xp, u=u_nom, aero_params=aero, dt=0.05)["x_geo_next"]
    tp = np.array(xp).reshape(-1)
    target = GeodeticState(tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], A320.mass_kg)

    # Fixed-time path (robust convergence); the dense-state behaviour under
    # test is identical for the free-time path.
    final_time, controls, _ = optimizer.optimize_trajectory(
        state, target, duration=horizon,
    )

    # Replay through the same re-anchored RK4 stepper the frontend playback
    # uses (NOT the geodetic stepper) -- they share the continuous RHS.
    sim = CasadiSimulator(aircraft=A320, dt=0.2)
    s = state
    segment_h = final_time / n_segments
    for k in range(n_segments):
        u = LoadFactorControl(
            thrust=float(controls[k, 0]),
            bank_rad=float(controls[k, 1]),
            load_factor=float(controls[k, 2]),
        )
        remaining = segment_h
        while remaining > 1e-9:
            dt = min(0.2, remaining)
            s = sim.step(s, u, dt)
            remaining -= dt

    R = 6_371_000.0
    horiz = R * math.hypot(
        math.radians(s.latitude - target.latitude),
        math.radians(s.longitude - target.longitude) * math.cos(math.radians(target.latitude)),
    )
    assert horiz < 5.0
    assert abs(s.altitude - target.altitude) < 5.0
    assert abs(s.V - target.V) < 1.0


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
    optimizer.state_substeps = 1
    optimizer.collocation_scheme = "hermiteSimpson"
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


def _propagate_with_geodetic_rhs(
    state: GeodeticState,
    optimizer,
    duration: float,
) -> GeodeticState:
    """Forward RK4 of the same continuous geodetic RHS the optimiser
    uses, via the geodetic stepper.  Generating the target this way means
    the optimiser is asked to recover a point that is exactly on its own
    dynamics manifold (no cross-frame discrepancy to absorb)."""
    from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator

    step = make_geodetic_step_integrator(transport="approx")["step_func"]
    aero_params = ca.DM([
        optimizer.aero_params.S,
        optimizer.aero_params.Cl_max,
        optimizer.aero_params.Cd0,
        optimizer.aero_params.k,
        optimizer.aero_params.stall_threshold,
        optimizer.aero_params.k_stall,
    ])
    u = ca.DM([C172.approach_thrust_guess_n, 0.0, 1.0])

    x = ca.DM([
        state.latitude,
        state.longitude,
        state.altitude,
        state.V,
        state.psi,
        state.gamma,
        state.m,
    ])
    n_steps = max(1, int(round(duration / 0.05)))
    dt = duration / n_steps
    for _ in range(n_steps):
        x = step(x_geo=x, u=u, aero_params=aero_params, dt=dt)["x_geo_next"]

    return GeodeticState(
        latitude=float(x[0]),
        longitude=float(x[1]),
        altitude=float(x[2]),
        V=float(x[3]),
        psi=float(x[4]),
        gamma=float(x[5]),
        m=float(x[6]),
    )


# --------------------------------------------------------------------------
# Selectable defect "fitting equation" (collocation scheme) comparison
# --------------------------------------------------------------------------

def test_defect_scheme_registry_lists_all_schemes():
    module = load_module()
    # Geodetic, normalized-geodetic and local-ENU dynamics each take all three
    # fittings; the re-anchored ENU is discrete so it is shooting-only.
    assert set(module._DEFECT_SCHEMES) == {
        "trapezoidal", "hermiteSimpson", "rk4",
        "trapezoidalFullTransport", "hermiteSimpsonFullTransport", "rk4FullTransport",
        "trapezoidalNormalized", "hermiteSimpsonNormalized", "rk4Normalized",
        "trapezoidalNormalizedFullTransport", "hermiteSimpsonNormalizedFullTransport",
        "rk4NormalizedFullTransport",
        "localEnuTrapezoidal", "localEnuHermiteSimpson", "localEnu",
        "reanchoredEnu",
    }
    # Only the *Normalized schemes carry the metric-position decision transform
    # (both the approx- and the full-transport normalized variants).
    assert module._NORMALIZED_SCHEMES == frozenset({
        "trapezoidalNormalized", "hermiteSimpsonNormalized", "rk4Normalized",
        "trapezoidalNormalizedFullTransport", "hermiteSimpsonNormalizedFullTransport",
        "rk4NormalizedFullTransport",
    })
    # The bare optimiser keeps Hermite-Simpson for backward compatibility.
    assert module._DEFAULT_SCHEME == "hermiteSimpson"


def test_trapezoidal_and_rk4_defects_zero_for_constant_state_zero_rhs():
    module = load_module()
    rhs_func = ca.Function(
        "rhs_zero",
        [ca.SX.sym("x", 7), ca.SX.sym("u", 3), ca.SX.sym("p", 6)],
        [ca.SX.zeros(7)],
    )
    x = ca.DM([10.0, -5.0, 1000.0, 60.0, 0.1, -0.01, 70000.0])
    u = ca.DM([1000.0, 0.0, 1.0])
    p = ca.DM([122.6, 1.5, 0.02, 0.04, 0.9, 0.1])

    for defect_expr in (module.trapezoidal_defect_expr, module.rk4_defect_expr):
        defect = defect_expr(rhs_func, x, x, u, p, 0.5)
        np.testing.assert_allclose(
            np.array(defect, dtype=float).reshape(-1), np.zeros(7), atol=1e-12,
        )


def test_unknown_collocation_scheme_raises():
    module = load_module()
    with pytest.raises(ValueError, match="unknown collocation_scheme"):
        module.CasadiDirectCollocationOptimizer(
            n_segments=10, dt=0.2, max_duration=120.0, aircraft=C172,
            collocation_scheme="quartic",
        )


def test_solver_backend_validation_and_default():
    module = load_module()
    assert module._DEFAULT_SOLVER_BACKEND == "ipopt"
    assert set(module._SOLVER_BACKENDS) == {"ipopt", "sqpmethod"}
    with pytest.raises(ValueError, match="unknown solver_backend"):
        module.CasadiDirectCollocationOptimizer(
            n_segments=10, dt=0.2, max_duration=120.0, aircraft=C172,
            solver_backend="lbfgs",
        )
    # The default optimiser records ipopt as its backend.
    opt = module.CasadiDirectCollocationOptimizer(
        n_segments=4, dt=0.2, max_duration=30.0, aircraft=C172,
    )
    assert opt.solver_backend == "ipopt"


def _replay_geodetic_horiz_miss(aircraft, init, controls, horizon):
    """Replay piecewise-constant controls through the fine geodetic
    integrator and return the horizontal miss to ``controls``' endpoint
    target (set by the caller)."""
    from aerodynamic_model.casadi_simulator import (
        AeroParams, make_geodetic_step_integrator,
    )

    step = make_geodetic_step_integrator(transport="approx")["step_func"]
    ap = aero_params_for_aircraft(aircraft)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    x = ca.DM([init.latitude, init.longitude, init.altitude,
               init.V, init.psi, init.gamma, init.m])
    seg = horizon / len(controls)
    for row in controls:
        u = ca.DM([float(row[0]), float(row[1]), float(row[2])])
        rem = seg
        while rem > 1e-9:
            h = min(0.05, rem)
            x = step(x_geo=x, u=u, aero_params=aero, dt=h)["x_geo_next"]
            rem -= h
    return np.array(x).reshape(-1)


def test_collocation_schemes_form_an_accuracy_ladder():
    """All three defect schemes solve the same feasible problem and pin
    their NODES on the target; replaying the controls reveals the order
    ladder -- the crude trapezoidal (2nd order) drifts more than the
    higher-order Hermite-Simpson and rk4 (both 4th order)."""
    from aerodynamic_model.aircraft_sets import A320

    module = load_module()
    n_segments = 10
    horizon = 90.0
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0,
                         math.radians(40.0), math.radians(-3.0), A320.mass_kg)

    # Feasible on-manifold target: propagate the A320 nominal approach
    # control through the optimiser's own geodetic RHS.
    from aerodynamic_model.casadi_simulator import (
        AeroParams, make_geodetic_step_integrator,
    )
    ap = aero_params_for_aircraft(A320)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    step = make_geodetic_step_integrator(transport="approx")["step_func"]
    u = ca.DM([A320.approach_thrust_guess_n, 0.0, 1.0])
    xp = ca.DM([init.latitude, init.longitude, init.altitude,
                init.V, init.psi, init.gamma, init.m])
    for _ in range(int(horizon / 0.05)):
        xp = step(x_geo=xp, u=u, aero_params=aero, dt=0.05)["x_geo_next"]
    tp = np.array(xp).reshape(-1)
    target = GeodeticState(tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], A320.mass_kg)

    R = 6_371_000.0

    def node_and_playback_miss(scheme):
        opt = module.CasadiDirectCollocationOptimizer(
            n_segments=n_segments, dt=0.2, max_duration=horizon * 1.6,
            aircraft=A320, collocation_scheme=scheme,
        )
        _, controls, states = opt.optimize_trajectory(init, target, duration=horizon)
        node_miss = R * math.hypot(
            math.radians(states[-1][0] - target.latitude),
            math.radians(states[-1][1] - target.longitude) * math.cos(math.radians(target.latitude)),
        )
        end = _replay_geodetic_horiz_miss(A320, init, controls, horizon)
        pb_miss = R * math.hypot(
            math.radians(end[0] - target.latitude),
            math.radians(end[1] - target.longitude) * math.cos(math.radians(target.latitude)),
        )
        assert controls.shape == (n_segments, 3)
        assert states.shape == (n_segments, 6)
        return node_miss, pb_miss

    trap_node, trap_pb = node_and_playback_miss("trapezoidal")
    hs_node, hs_pb = node_and_playback_miss("hermiteSimpson")
    rk4_node, rk4_pb = node_and_playback_miss("rk4")

    # Every scheme pins its own nodes on the target (terminal equality).
    for node_miss in (trap_node, hs_node, rk4_node):
        assert node_miss < 1.0

    # The higher-order schemes reproduce the trajectory to sub-metre; the
    # crude trapezoidal drifts noticeably more.
    assert hs_pb < 2.0
    assert rk4_pb < 2.0
    assert trap_pb > hs_pb


def test_full_transport_schemes_registered_and_solve():
    """The geodetic FULL-transport schemes (exact transport: the psi cross term
    the default ``approx`` schemes drop) are registered and solve a feasible
    problem, pinning their nodes on the target.  Full transport changes the
    optimum only negligibly (the cross term is ~O(e^2 sinγ)), so the whole node
    path stays within a few metres of the approx-transport optimum."""
    from aerodynamic_model.aircraft_sets import A320
    from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator

    module = load_module()
    for scheme in ("hermiteSimpsonFullTransport", "trapezoidalFullTransport", "rk4FullTransport"):
        assert scheme in module._DEFECT_SCHEMES

    n_segments = 10
    horizon = 90.0
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0,
                         math.radians(40.0), math.radians(-3.0), A320.mass_kg)

    # On-manifold target: propagate the nominal approach control through the
    # FULL-transport geodetic RHS (so the full scheme is exactly feasible).
    ap = aero_params_for_aircraft(A320)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    step = make_geodetic_step_integrator(transport="full")["step_func"]
    u = ca.DM([A320.approach_thrust_guess_n, 0.0, 1.0])
    xp = ca.DM([init.latitude, init.longitude, init.altitude,
                init.V, init.psi, init.gamma, init.m])
    for _ in range(int(horizon / 0.05)):
        xp = step(x_geo=xp, u=u, aero_params=aero, dt=0.05)["x_geo_next"]
    tp = np.array(xp).reshape(-1)
    target = GeodeticState(tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], A320.mass_kg)
    R = 6_371_000.0

    def solve(scheme):
        opt = module.CasadiDirectCollocationOptimizer(
            n_segments=n_segments, dt=0.2, max_duration=horizon * 1.6,
            aircraft=A320, collocation_scheme=scheme,
        )
        _, controls, states = opt.optimize_trajectory(init, target, duration=horizon)
        assert controls.shape == (n_segments, 3)
        node_miss = R * math.hypot(
            math.radians(states[-1][0] - target.latitude),
            math.radians(states[-1][1] - target.longitude) * math.cos(math.radians(target.latitude)),
        )
        return node_miss, states

    full_miss, full_states = solve("hermiteSimpsonFullTransport")
    approx_miss, approx_states = solve("hermiteSimpson")
    assert full_miss < 1.0
    assert approx_miss < 1.0
    # Whole-path separation between the full- and approx-transport optima: the
    # cross term is tiny, so every node agrees to within a few metres.
    max_sep = max(
        R * math.hypot(
            math.radians(f[0] - a[0]),
            math.radians(f[1] - a[1]) * math.cos(math.radians(target.latitude)),
        )
        for f, a in zip(full_states, approx_states)
    )
    assert max_sep < 5.0


def test_reanchored_enu_scheme_is_consistent_with_the_enu_playback():
    """The ``reanchoredEnu`` defect IS the re-anchored ENU one-step map the
    frontend playback (``CasadiSimulator``) runs, so a solution replayed
    through that exact simulator lands on the target with a small miss
    (no geodetic-vs-ENU model gap; only the step-size discretisation remains)."""
    from aerodynamic_model.aircraft_sets import A320
    from aerodynamic_model.casadi_simulator import (
        AeroParams, CasadiSimulator, make_geodetic_step_integrator,
    )
    from aerodynamic_model.common import LoadFactorControl

    module = load_module()
    n_segments = 10
    horizon = 90.0
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0,
                         math.radians(40.0), math.radians(-3.0), A320.mass_kg)

    ap = aero_params_for_aircraft(A320)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    gstep = make_geodetic_step_integrator(transport="approx")["step_func"]
    u = ca.DM([A320.approach_thrust_guess_n, 0.0, 1.0])
    xp = ca.DM([init.latitude, init.longitude, init.altitude,
                init.V, init.psi, init.gamma, init.m])
    for _ in range(int(horizon / 0.05)):
        xp = gstep(x_geo=xp, u=u, aero_params=aero, dt=0.05)["x_geo_next"]
    tp = np.array(xp).reshape(-1)
    target = GeodeticState(tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], A320.mass_kg)

    opt = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=horizon * 1.6,
        aircraft=A320, collocation_scheme="reanchoredEnu",
    )
    assert opt.collocation_scheme == "reanchoredEnu"
    final_time, controls, states = opt.optimize_trajectory(init, target, duration=horizon)
    assert controls.shape == (n_segments, 3)
    assert states.shape == (n_segments, 6)

    # Replay through the exact ENU stepper the playback uses.
    sim = CasadiSimulator(aircraft=A320, dt=0.2)
    s = init
    seg = final_time / n_segments
    for k in range(n_segments):
        cu = LoadFactorControl(thrust=float(controls[k, 0]),
                               bank_rad=float(controls[k, 1]),
                               load_factor=float(controls[k, 2]))
        remaining = seg
        while remaining > 1e-9:
            dt = min(0.2, remaining)
            s = sim.step(s, cu, dt)
            remaining -= dt

    R = 6_371_000.0
    horiz = R * math.hypot(
        math.radians(s.latitude - target.latitude),
        math.radians(s.longitude - target.longitude) * math.cos(math.radians(target.latitude)),
    )
    assert horiz < 5.0
    assert abs(s.altitude - target.altitude) < 5.0


def test_geodetic_enu_conversion_round_trips_and_anchors_at_ref():
    """Regression for the localEnu dynamics: the forward
    ``geodetic_state_to_enu_expr`` must invert ``enu_state_to_geodetic_expr``
    (so the fixed-frame collocation is self-consistent), and a point converted
    against its OWN ref must sit at the ENU origin with V/psi/gamma unchanged."""
    from aerodynamic_model.casadi_simulator import (
        geodetic_state_to_enu_expr, enu_state_to_geodetic_expr,
    )

    x_geo = [35.60, -78.50, 1500.0, 120.0, math.radians(40.0), math.radians(-2.0)]
    ref = ca.DM([35.87, -78.80, 0.0])         # a fixed anchor ~30 km away

    enu = np.array(ca.DM(geodetic_state_to_enu_expr(ca.DM(x_geo), ref)), dtype=float).ravel()
    back = np.array(ca.DM(enu_state_to_geodetic_expr(ca.DM(list(enu)), ref)), dtype=float).ravel()
    # forward then inverse recovers the geodetic state.
    np.testing.assert_allclose(back[:3], x_geo[:3], atol=1e-6)          # lat,lon,alt
    np.testing.assert_allclose(back[3], x_geo[3], atol=1e-6)            # V
    np.testing.assert_allclose([back[4], back[5]], x_geo[4:6], atol=1e-9)

    # Anchored at its own point -> ENU origin, speed/heading/pitch unchanged.
    ref_self = ca.DM([x_geo[0], x_geo[1], 0.0])
    enu0 = np.array(ca.DM(geodetic_state_to_enu_expr(ca.DM(x_geo), ref_self)), dtype=float).ravel()
    np.testing.assert_allclose(enu0[0:2], [0.0, 0.0], atol=1e-6)        # east, north
    np.testing.assert_allclose(enu0[2], x_geo[2], atol=1e-6)           # up == alt
    np.testing.assert_allclose([enu0[3], enu0[4], enu0[5]],
                               [x_geo[3], x_geo[4], x_geo[5]], atol=1e-9)


def test_local_enu_scheme_solves_with_every_fitting():
    """localEnu is a CONTINUOUS dynamics (flat RHS in a fixed ENU tangent frame
    at the target), so -- like the geodetic dynamics -- it takes every fitting.
    All three variants solve and return the standard tuple shape; the terminal
    node is pinned on the target.  (Its accuracy far from the anchor is studied
    in dynamics_comparison_30km, not here.)"""
    from aerodynamic_model.aircraft_sets import A320
    from aerodynamic_model.casadi_simulator import (
        AeroParams, make_geodetic_step_integrator,
    )

    module = load_module()
    n_segments = 10
    horizon = 90.0
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0,
                         math.radians(40.0), math.radians(-3.0), A320.mass_kg)

    ap = aero_params_for_aircraft(A320)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    gstep = make_geodetic_step_integrator(transport="approx")["step_func"]
    u = ca.DM([A320.approach_thrust_guess_n, 0.0, 1.0])
    xp = ca.DM([init.latitude, init.longitude, init.altitude,
                init.V, init.psi, init.gamma, init.m])
    for _ in range(int(horizon / 0.05)):
        xp = gstep(x_geo=xp, u=u, aero_params=aero, dt=0.05)["x_geo_next"]
    tp = np.array(xp).reshape(-1)
    target = GeodeticState(tp[0], tp[1], tp[2], tp[3], tp[4], tp[5], A320.mass_kg)
    R = 6_371_000.0

    for scheme in ("localEnuTrapezoidal", "localEnuHermiteSimpson", "localEnu"):
        opt = module.CasadiDirectCollocationOptimizer(
            n_segments=n_segments, dt=0.2, max_duration=horizon * 1.6,
            aircraft=A320, collocation_scheme=scheme,
        )
        assert opt.collocation_scheme == scheme
        _, controls, states = opt.optimize_trajectory(init, target, duration=horizon)
        assert controls.shape == (n_segments, 3)
        assert states.shape == (n_segments, 6)
        node_miss = R * math.hypot(
            math.radians(states[-1][0] - target.latitude),
            math.radians(states[-1][1] - target.longitude) * math.cos(math.radians(target.latitude)),
        )
        assert node_miss < 1.0


def test_normalized_scheme_solves_loose_window_where_geodetic_fails():
    """Regression (KRDU HEAVE -> RW05L): the metric-position *Normalized*
    geodetic scheme stays well-conditioned and converges on the default mesh
    with a loose arrival window, where the plain (radian-state) geodetic scheme
    exhausts IPOPT's iteration budget.

    HEAVE -> RW05L is a ~180 deg turn + ~1700 m descent over a ~9.6 km
    straight-line span; at n_segments=10 with max_duration=1000 s (far longer
    than the ~250 s the trajectory needs) the radian-state geodetic NLP is badly
    scaled.  Expressing position as METRES from the target conditions the NLP so
    the same solve converges and lands on the target.
    """
    module = load_module()
    n_segments = 10
    max_duration = 1000.0

    state = GeodeticState(
        latitude=35.91816944, longitude=-78.89327222, altitude=1828.8,
        V=(A320.terminal_speed_kt + 25) * 0.51444,
        psi=math.radians(225.0), gamma=0.0, m=A320.mass_kg,
    )
    target = GeodeticState(
        latitude=35.87446907, longitude=-78.80194912,
        altitude=367.0 * 0.3048 + A320.threshold_crossing_height_m,
        V=A320.terminal_speed_kt * 0.51444,
        psi=math.radians(45.0), gamma=math.radians(-3.0), m=A320.mass_kg,
    )

    optimizer = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=max_duration, aircraft=A320,
        collocation_scheme="hermiteSimpsonNormalized",
    )
    final_time, controls, states = optimizer.optimize_free_time(
        state, target, max_duration,
    )

    assert optimizer.free_time_solver.stats()["success"]
    assert controls.shape == (n_segments, 3)
    # The solver shrank T to the natural duration, far below the loose window.
    assert final_time < 0.5 * max_duration

    np.testing.assert_allclose(
        states[-1],
        np.array([
            target.latitude, target.longitude, target.altitude,
            target.V, target.psi, target.gamma,
        ]),
        atol=1e-3,
    )

    # The cold-start / free-time wall-clock split is still recorded for the log.
    timings = optimizer.last_solve_timings
    assert set(timings) == {"coldStartS", "freeTimeSolveS", "solveTotalS"}
    assert timings["solveTotalS"] == pytest.approx(
        timings["coldStartS"] + timings["freeTimeSolveS"],
    )


def test_normalized_scheme_matches_plain_geodetic_on_a_benign_problem():
    """Normalization is a pure change of DECISION variables (the geodetic RHS is
    evaluated at the exact reconstructed lat/lon), so on a problem the plain
    geodetic scheme also solves, the normalized scheme returns the SAME
    trajectory -- same arrival time and same terminal state."""
    module = load_module()
    n_segments = 4
    feasible_duration = 2.0
    max_duration = 6.0

    speed = C172.terminal_speed_kt * 0.51444 + 10.0
    state = GeodeticState(51.1139, -114.0203, 1000.0, speed, 0.0, 0.0, C172.mass_kg)

    base = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=max_duration, aircraft=C172,
        collocation_scheme="hermiteSimpson",
    )
    target = _propagate_with_geodetic_rhs(state, base, feasible_duration)
    norm = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=max_duration, aircraft=C172,
        collocation_scheme="hermiteSimpsonNormalized",
    )

    t_base, _, s_base = base.optimize_free_time(state, target, max_duration)
    t_norm, _, s_norm = norm.optimize_free_time(state, target, max_duration)

    assert base.free_time_solver.stats()["success"]
    assert norm.free_time_solver.stats()["success"]
    assert t_norm == pytest.approx(t_base, abs=0.05)
    np.testing.assert_allclose(s_norm[-1], s_base[-1], atol=1e-3)


def test_normalized_full_transport_matches_plain_full_transport():
    """The metric-position normalization is orthogonal to the transport model:
    on a benign problem the normalized FULL-transport scheme returns the SAME
    trajectory as the plain FULL-transport scheme (same arrival time + terminal
    state) — exactly as the approx-transport pair does above."""
    from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator

    module = load_module()
    n_segments = 4
    feasible_duration = 2.0
    max_duration = 6.0

    speed = C172.terminal_speed_kt * 0.51444 + 10.0
    state = GeodeticState(51.1139, -114.0203, 1000.0, speed, 0.0, 0.0, C172.mass_kg)

    plain = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=max_duration, aircraft=C172,
        collocation_scheme="hermiteSimpsonFullTransport",
    )
    # Feasible target ON the full-transport manifold (propagate the full RHS).
    step = make_geodetic_step_integrator(transport="full")["step_func"]
    aero = ca.DM([
        plain.aero_params.S, plain.aero_params.Cl_max, plain.aero_params.Cd0,
        plain.aero_params.k, plain.aero_params.stall_threshold, plain.aero_params.k_stall,
    ])
    u = ca.DM([C172.approach_thrust_guess_n, 0.0, 1.0])
    x = ca.DM([state.latitude, state.longitude, state.altitude,
               state.V, state.psi, state.gamma, state.m])
    n_steps = max(1, int(round(feasible_duration / 0.05)))
    for _ in range(n_steps):
        x = step(x_geo=x, u=u, aero_params=aero, dt=feasible_duration / n_steps)["x_geo_next"]
    xt = np.array(x).reshape(-1)
    target = GeodeticState(xt[0], xt[1], xt[2], xt[3], xt[4], xt[5], C172.mass_kg)

    norm = module.CasadiDirectCollocationOptimizer(
        n_segments=n_segments, dt=0.2, max_duration=max_duration, aircraft=C172,
        collocation_scheme="hermiteSimpsonNormalizedFullTransport",
    )

    t_plain, _, s_plain = plain.optimize_free_time(state, target, max_duration)
    t_norm, _, s_norm = norm.optimize_free_time(state, target, max_duration)

    assert plain.free_time_solver.stats()["success"]
    assert norm.free_time_solver.stats()["success"]
    assert t_norm == pytest.approx(t_plain, abs=0.05)
    np.testing.assert_allclose(s_norm[-1], s_plain[-1], atol=1e-3)
