"""The unified CollocationOptimizer: foundation math, unconstrained + procedure-constrained
solves, every scheme, the guards, and the regressions (FAF convention, sea-level altitude floor).

Constraints are optional: ``segments=None`` is the unconstrained optimiser (ADS-B / runway
target), a list of legs is the procedure-constrained one. Both share this file.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_OPT_DIR = Path(__file__).resolve().parents[2]      # 4dTrajectory/optimization
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as ca  # noqa: E402
import approach_constraints as ac  # noqa: E402
from approach_constraints import geometry as ac_geo  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator  # noqa: E402
from aircraft.aircraft_sets import A320  # noqa: E402
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402

from collocation import CollocationOptimizer, _DEFECT_SCHEMES, altitude_floor_m, ALTITUDE_FLOOR_MARGIN_M  # noqa: E402
from collocation import schemes as _schemes  # noqa: E402
from collocation import components as _components  # noqa: E402
from collocation import optimizer as _optimizer  # noqa: E402

MTOW = A320.mass.max_takeoff_kg


# ── shared scenario builders ───────────────────────────────────────────────
def _rollout_samples(init, horizon, dt=0.05):
    """Forward-integrate the geodetic RHS from ``init`` (a reachable trajectory to pin against)."""
    ap = aero_params_for_aircraft(A320)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    step = make_geodetic_step_integrator(transport="full")["step_func"]
    u = ca.DM([A320.approach.thrust_guess_n, 0.0, 1.0])
    x = ca.DM([init.latitude, init.longitude, init.altitude, init.V, init.psi, init.gamma, init.m])
    samples = [np.array(x).reshape(-1)]
    for _ in range(int(horizon / dt)):
        x = step(x_geo=x, u=u, aero_params=aero, dt=dt)["x_geo_next"]
        samples.append(np.array(x).reshape(-1))
    return samples


def _reachable_target(init, horizon):
    return GeodeticState(*_rollout_samples(init, horizon)[-1][:6], MTOW)


def _approach(S, frame, fracs=(0.30, 0.55, 0.78, 1.0)):
    """INITIAL / INTERMEDIATE / FINAL_LPV segments from rollout samples at ``fracs``."""
    smp = lambda f: S[int(f * (len(S) - 1))]
    IAF, IF, FAF, LTP = (frame.to_ne(*smp(f)[:2]) for f in fracs)
    ft = 0.3048
    inbound = -FAF / float(np.hypot(*FAF))
    d_garp = (9023.0 + 1000.0) * ft
    gpa = math.degrees(math.atan2(smp(fracs[2])[2] - S[-1][2], float(np.hypot(*(FAF - LTP)))))
    lpv = ac.LpvFinalSpec(
        ltp_ne=LTP, fpap_ne=9023.0 * ft * inbound, garp_ne=d_garp * inbound,
        course_width_m=max(350.0 * ft, math.tan(math.radians(1.5)) * d_garp),
        tdze_m=S[-1][2] - 50.0 * ft, tch_m=50.0 * ft, gpa_deg=gpa, below_m=120.0, above_m=120.0,
        # Vertical gate, as the backend always sets it: glidepath window binds from the FAF
        # toward the runway; upstream, the published FAF floor (below the flown altitude here).
        d_faf_m=float(np.hypot(*(FAF - LTP))),
        prefaf_floor_m=float(smp(fracs[2])[2]) - 200.0,
    )
    segments = [
        ac.SegmentSpec(ac.SegmentKind.INITIAL, IAF, IF, "IAF", "IF", halfwidth_m=1852.0),
        ac.SegmentSpec(ac.SegmentKind.INTERMEDIATE, IF, FAF, "IF", "FAF", halfwidth_m=1852.0),
        ac.SegmentSpec(ac.SegmentKind.FINAL_LPV, FAF, LTP, "FAF", "LTP", lpv=lpv),
    ]
    return segments, FAF, LTP


def _one_lpv_segment(ltp, faf):
    ft = 0.3048
    norm = float(np.hypot(*faf))
    inbound = -np.asarray(faf, float) / norm
    d_garp = (9023.0 + 1000.0) * ft
    lpv = ac.LpvFinalSpec(
        ltp_ne=np.asarray(ltp, float), fpap_ne=9023.0 * ft * inbound, garp_ne=d_garp * inbound,
        course_width_m=max(350.0 * ft, math.tan(math.radians(1.5)) * d_garp),
        tdze_m=120.0, tch_m=50.0 * ft, gpa_deg=3.0, below_m=120.0, above_m=120.0,
    )
    return ac.SegmentSpec(ac.SegmentKind.FINAL_LPV, faf, ltp, "FAF", "LTP", lpv=lpv)


def _faf_intercept_deg(states, frame, faf_ne):
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    i = int(np.argmin([np.hypot(*(p - faf_ne)) for p in ne]))
    # FAF -> LTP(0,0) course from the ONE course-math source, in the model's psi convention.
    fac = math.degrees(ac_geo.course_bearing(faf_ne, [0.0, 0.0]))
    return abs(((math.degrees(states[i][4]) - fac + 180) % 360) - 180)


# a mild reachable straight-in used by the unconstrained solves
def _straight_in(heading_deg=40.0, alt=1500.0):
    return GeodeticState(35.60, -78.50, alt, 90.0, math.radians(heading_deg), math.radians(-3.0), MTOW)


# ── foundation math (moved to schemes / components) ────────────────────────
@pytest.mark.parametrize("defect_fn", [
    _schemes.hermite_simpson_defect_expr,
    _schemes.trapezoidal_defect_expr,
    _schemes.rk4_defect_expr,
])
def test_defect_is_zero_for_constant_state_with_zero_rhs(defect_fn):
    zero_rhs = lambda x, u, aero: ca.DM.zeros(x.shape)
    x = ca.DM([0.5, -1.3, 300.0, 80.0, 0.1, -0.05, 60000.0])
    d = defect_fn(zero_rhs, x, x, ca.DM([0, 0, 1]), ca.DM.zeros(6), 2.0)
    assert float(ca.norm_inf(d)) == pytest.approx(0.0, abs=1e-12)


def test_scheme_registry_has_every_scheme():
    assert _schemes._DEFAULT_SCHEME in _DEFECT_SCHEMES
    # 3 fittings × {geodetic, +fullTransport, +normalized, +normalizedFullTransport} + 3 localEnu + reanchored
    assert len(_DEFECT_SCHEMES) == 16
    assert _schemes._NORMALIZED_FULL_TRANSPORT_SCHEMES <= set(_DEFECT_SCHEMES)


def test_unwrap_target_heading_picks_short_turn():
    ip = _components._geodetic_state_to_decision(_straight_in(heading_deg=-135.0))
    tp = _components._geodetic_state_to_decision(_straight_in(heading_deg=135.0))
    unwrapped = _components._unwrap_target_heading(ip, tp)
    assert abs(unwrapped[4] - ip[4]) <= math.pi + 1e-9      # within a half-turn, not the long way


def test_altitude_floor_is_a_margin_below_the_target():
    assert altitude_floor_m(1000.0) == 1000.0 - ALTITUDE_FLOOR_MARGIN_M
    assert altitude_floor_m(16.0) < 16.0        # below a near-sea-level threshold
    assert altitude_floor_m(150.0) < 150.0


def test_unwrap_angle_picks_the_nearest_branch():
    assert _components.unwrap_angle(0.785, 7.0) == pytest.approx(0.785 + 2.0 * math.pi)
    assert _components.unwrap_angle(0.785, 0.9) == pytest.approx(0.785)
    assert _components.unwrap_angle(-2.356, 3.93) == pytest.approx(-2.356 + 2.0 * math.pi)


# ── procedure constraint row functions (white-box) ─────────────────────────
def _join_for_rows():
    """A _FinalJoin over the _one_lpv_segment geometry: FAF at (5000, 0), course −π/2
    (flying south toward the LTP at the origin), branch course −π/2, window [1000, 2000]
    upstream of the FAF."""
    seg = _one_lpv_segment(np.array([0.0, 0.0]), np.array([5000.0, 0.0]))
    return _optimizer._FinalJoin(
        phase=0, lpv=seg.lpv, d_faf_m=5000.0, max_offset_m=2000.0, min_offset_m=1000.0,
        course_branch_rad=-math.pi / 2.0,
    )


def _eval_rows(rows, node_syms, values):
    f = ca.Function("f", node_syms, [expr for expr, _lb, _ub in rows])
    outs = f(*values)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    return [np.array(o).ravel() for o in outs], [(lb, ub) for _e, lb, ub in rows]


def test_fac_join_rows_contract():
    join = _join_for_rows()
    node = ca.SX.sym("x", 6)
    rows = _optimizer._fac_join_rows(node, join, math.radians(30.0))
    assert len(rows) == 4                                    # xtk eq + window lo/hi + psi box
    assert rows[0][1] == rows[0][2] == 0.0                   # cross-track is an EQUALITY
    assert rows[3][1] == pytest.approx(-math.radians(30.0))  # psi box bounds
    # a node ON the course, 1.5 km before the FAF, aligned: every row inside its bounds
    vals, bounds = _eval_rows(rows, [node], [ca.DM([6500.0, 0.0, 800.0, 80.0, -math.pi / 2, 0.0])])
    for v, (lb, ub) in zip(vals, bounds):
        assert lb - 1e-9 <= float(v[0]) <= ub + 1e-9
    # at the FAF exactly: the window's lower row is violated (too late to join)
    vals, _ = _eval_rows(rows, [node], [ca.DM([5000.0, 0.0, 800.0, 80.0, -math.pi / 2, 0.0])])
    assert float(vals[1][0]) > 0.0


def test_fac_alignment_rows_switch_tiers_at_the_faf():
    join = _join_for_rows()
    nodes = [ca.SX.sym(f"x{i}", 6) for i in range(2)]
    rows = _optimizer._fac_alignment_rows(
        nodes, join, math.radians(10.0), math.radians(30.0))
    assert len(rows) == 2 and all(ub == 0.0 for _e, _lb, ub in rows)
    course = -math.pi / 2.0

    def violations(d_upstream, dev_rad):
        vals, _ = _eval_rows(
            rows, nodes,
            [ca.DM([join.d_faf_m + d_upstream, 0.0, 800.0, 80.0, course + dev_rad, 0.0]),
             ca.DM([join.d_faf_m + d_upstream, 0.0, 800.0, 80.0, course + dev_rad, 0.0])],
        )
        return max(float(v.max()) for v in vals)

    assert violations(+1500.0, math.radians(20.0)) <= 0.0    # upstream: 20 deg ok (loose 30)
    assert violations(-1500.0, math.radians(20.0)) > 0.0     # past the FAF: 20 deg VIOLATES (10)
    assert violations(-1500.0, math.radians(5.0)) <= 0.0     # past the FAF: 5 deg ok
    assert violations(-1500.0, math.radians(-20.0)) > 0.0    # symmetric on the other side


def test_psi_corridor_bounds_forbid_full_windings():
    # The constrained ψ variable bounds are the route's heading hull ± the manoeuvre slack —
    # a ±2π winding excursion (the nsp=2 looping local optima) is OUTSIDE the variable box and
    # cannot even be visited. Unconstrained solves keep the generic ±3π.
    init = _straight_in()
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, _FAF, _LTP = _approach(S, frame)
    opt = CollocationOptimizer(A320, segments=segments)
    nlp, lbw, ubw, *_rest, layout = opt._build(init, target, 200.0)
    n_seg = opt.n_seg_per_phase
    psi_idx = n_seg * _schemes.CONTROL_DIM + 4          # first node's ψ slot in phase 0's block
    lo, hi = lbw[psi_idx], ubw[psi_idx]
    assert hi - lo < 2.0 * math.pi + 2.0 * math.radians(90.0) + 1e-9   # hull(≤2π... route ≈ small) + slack
    assert hi < 3.0 * math.pi - 1e-9 and lo > -3.0 * math.pi + 1e-9    # tighter than the generic box
    # winding a full turn from the initial heading leaves the corridor
    assert init.psi + 2.0 * math.pi > hi
    # unconstrained: generic ±3π untouched
    unopt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")
    _n, ulbw, uubw, *_r, _l = unopt._build(init, target, 200.0)
    upsi = unopt.n_segments * _schemes.CONTROL_DIM + 4
    assert uubw[upsi] == pytest.approx(3.0 * math.pi)
    assert ulbw[upsi] == pytest.approx(-3.0 * math.pi)


def test_prefaf_and_terminal_row_shapes():
    node = ca.SX.sym("x", 6)
    disc = _optimizer._prefaf_fix_rows(node, np.array([100.0, 200.0]), 926.0)
    assert len(disc) == 1 and disc[0][2] == 0.0
    pin = _optimizer._terminal_pin_rows(node, np.zeros(6))
    assert len(pin) == 1 and int(pin[0][0].shape[0]) == 6
    assert pin[0][1] == pin[0][2] == 0.0


# ── guards ─────────────────────────────────────────────────────────────────
def test_guards():
    seg = [_one_lpv_segment(np.array([0.0, 0.0]), np.array([5000.0, 0.0]))]
    with pytest.raises(ValueError, match="normalized full-transport"):
        CollocationOptimizer(A320, segments=seg, scheme="hermiteSimpson")
    with pytest.raises(ValueError, match="non-empty"):
        CollocationOptimizer(A320, segments=[])
    with pytest.raises(ValueError, match="ipopt"):
        CollocationOptimizer(A320, segments=seg,
                             scheme="hermiteSimpsonNormalizedFullTransport", solver_backend="sqpmethod")
    with pytest.raises(ValueError, match="unknown scheme"):
        CollocationOptimizer(A320, scheme="nope")
    with pytest.raises(ValueError, match="unknown solver_backend"):
        CollocationOptimizer(A320, solver_backend="nope")
    with pytest.raises(ValueError):
        CollocationOptimizer(A320, min_speed_ms=-1.0)
    with pytest.raises(ValueError, match="max_join_offset_m"):
        CollocationOptimizer(A320, max_join_offset_m=-1.0)


def test_guards_frame_anchor_contract():
    # The segments must be in the TARGET-anchored (n, e) frame: the procedure must END at the
    # origin (= the LTP = the pinned terminal), and every LPV spec must anchor its LTP there.
    from dataclasses import replace
    off_origin = [_one_lpv_segment(np.array([4000.0, 0.0]), np.array([9000.0, 0.0]))]
    with pytest.raises(ValueError, match="target-anchored"):
        CollocationOptimizer(A320, segments=off_origin)
    good = _one_lpv_segment(np.array([0.0, 0.0]), np.array([5000.0, 0.0]))
    bad_lpv = replace(good, lpv=replace(good.lpv, ltp_ne=np.array([500.0, 0.0])))
    with pytest.raises(ValueError, match="LPV segment"):
        CollocationOptimizer(A320, segments=[bad_lpv])


def test_phase_plan_adds_transition_phase_only_for_a_far_start():
    # A start farther than the fix-passage tolerance from the first leg's start fix -> an
    # UNCONSTRAINED start->first-fix transition phase is prepended (whose end node must then
    # PASS the fix); a start within the tolerance joins the procedure directly. For an LPV-first
    # procedure (no RNP box) the generic 2 km fallback applies.
    seg = [_one_lpv_segment(np.array([0.0, 0.0]), np.array([5000.0, 0.0]))]
    opt = CollocationOptimizer(A320, segments=seg)
    near = opt._phase_plan(np.array([5100.0, 0.0]), None)
    assert len(near) == 1 and near[0][1] is seg[0]
    far = opt._phase_plan(np.array([9000.0, 0.0]), None)
    assert len(far) == 2
    assert far[0][1] is None and np.allclose(far[0][0], [5000.0, 0.0])   # transition -> the FAF
    assert far[1][1] is seg[0]

    # Box-first procedure: the threshold IS the passage-disc radius (k·RNP = 0.5·1852 = 926 m) —
    # one decision, no dead zone where a near-but-not-at start would skip the fix.
    boxed = [
        ac.SegmentSpec(ac.SegmentKind.INITIAL, [9000.0, 0.0], [5000.0, 0.0], "IAF", "FAF",
                       halfwidth_m=1852.0),
        _one_lpv_segment(np.array([0.0, 0.0]), np.array([5000.0, 0.0])),
    ]
    opt_boxed = CollocationOptimizer(A320, segments=boxed)
    assert opt_boxed._first_fix_join_tolerance_m() == pytest.approx(926.0)
    inside = opt_boxed._phase_plan(np.array([9500.0, 0.0]), None)      # 500 m -> at the fix
    assert len(inside) == 2
    between = opt_boxed._phase_plan(np.array([10500.0, 0.0]), None)    # 1.5 km -> must fly to it
    assert len(between) == 3 and between[0][1] is None


# ── unconstrained solves ───────────────────────────────────────────────────
def test_unconstrained_free_time_reaches_target():
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")
    final_time, controls, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    assert controls.shape == (opt.n_segments, 3)
    assert states.shape == (opt.n_segments, 6)
    assert final_time < 120.0 * 1.6 - 1.0                    # the time objective actually shrank T
    np.testing.assert_allclose(
        states[-1], [target.latitude, target.longitude, target.altitude, target.V, target.psi, target.gamma],
        atol=1e-2)
    # the dense plan subsamples to the returned endpoints exactly (playback-consistent contract)
    dense = opt.last_dense_states_geo
    m = opt.state_substeps or _components.select_state_substeps(120.0 * 1.6, opt.n_segments)
    np.testing.assert_allclose(dense[m - 1:: m], states, atol=1e-9)


def test_unconstrained_fixed_time_hits_the_duration():
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")
    final_time, _c, states = opt.optimize_trajectory(init, target, duration=150.0)
    assert final_time == pytest.approx(150.0, abs=1e-2)
    np.testing.assert_allclose(
        states[-1], [target.latitude, target.longitude, target.altitude, target.V, target.psi, target.gamma],
        atol=1e-2)


def test_fixed_time_objective_weights_control_effort_at_one():
    # REGRESSION: the fixed-time objective must weight control effort at 1.0 (the old fixed-time
    # NLP's behaviour), NOT the free-time 1e-3 tie-breaker (which is only appropriate when the time
    # term dominates). White-box check: ∂cost/∂thrust at the all-equal-control initial guess — where
    # the smoothness gradient vanishes — is purely effort_weight · ∂effort, so the fixed/free ratio
    # pins the two weights at 1.0 / 1e-3 = 1000×. A 1e-3 fixed-time weight (the bug) makes it 1×.
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")

    def dcost_dthrust0(fixed_duration):
        nlp, *_bounds, x0, _layout = opt._build(init, target, 150.0, fixed_duration=fixed_duration)
        grad = ca.Function("g", [nlp["x"]], [ca.gradient(nlp["f"], nlp["x"])])
        return abs(float(np.array(grad(x0))[0]))       # ∂/∂ thrust of phase 0, segment 0

    assert dcost_dthrust0(150.0) == pytest.approx(dcost_dthrust0(None) * 1000.0, rel=1e-6)


def test_supplied_initial_guess_is_accepted():
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")
    ft, _c, _s = opt.optimize_free_time(init, target, 120.0 * 1.6)
    seed = opt._solve_fixed_raw(init, target, 120.0 * 1.6)
    ft2, _c2, _s2 = opt.optimize_free_time(init, target, 120.0 * 1.6, initial_guess=seed)
    assert ft2 == pytest.approx(ft, rel=0.2)


def test_handles_heading_wrap_across_180_deg():
    # start heading -170°, target +170°: a ~20° turn that must NOT go the long way round the cut.
    init = _straight_in(heading_deg=-170.0)
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme="hermiteSimpsonNormalizedFullTransport")
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    np.testing.assert_allclose(states[-1][:2], [target.latitude, target.longitude], atol=1e-2)


def test_near_sea_level_target_solves():
    # REGRESSION (KMSY/KSMF): the altitude floor is target-relative, so a threshold BELOW the old
    # absolute 25 m floor is feasible (was Infeasible). Start low so the target lands ~17.8 m MSL.
    init = _straight_in(alt=620.0)
    target = _reachable_target(init, 120.0)
    assert target.altitude < 25.0
    opt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    assert states[-1][2] == pytest.approx(target.altitude, abs=1.0)


def test_free_time_raises_on_infeasible_problem():
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport")
    with pytest.raises(ValueError, match="optimization failed"):
        # 5 s to fly a ~2 min straight-in: no feasible trajectory at any control.
        opt.optimize_trajectory(init, target, duration=5.0)


@pytest.mark.parametrize("scheme", sorted(_DEFECT_SCHEMES))
def test_every_scheme_solves_and_pins_the_target(scheme):
    """Each registered dynamics×fitting scheme solves a benign unconstrained problem; the terminal
    pin puts the last node ON the target (position) regardless of the scheme's interior fidelity."""
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    opt = CollocationOptimizer(A320, scheme=scheme)
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    np.testing.assert_allclose(states[-1][:2], [target.latitude, target.longitude], atol=1e-3)


def test_normalized_matches_plain_geodetic_on_a_benign_problem():
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    plain = CollocationOptimizer(A320, scheme="hermiteSimpson").optimize_free_time(init, target, 200.0)
    norm = CollocationOptimizer(A320, scheme="hermiteSimpsonNormalized").optimize_free_time(init, target, 200.0)
    assert norm[0] == pytest.approx(plain[0], rel=0.05)           # same optimal time
    np.testing.assert_allclose(norm[2][-1], plain[2][-1], atol=1e-2)   # same terminal state


# ── procedure-constrained solves ───────────────────────────────────────────
def test_constrained_joins_the_course_passes_fixes_and_reaches_threshold():
    init = _straight_in()
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)

    opt = CollocationOptimizer(A320, segments=segments)
    final_time, controls, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    nsp = opt.n_seg_per_phase
    # the start is well away from the IAF -> a start->IAF transition phase is prepended
    assert controls.shape == ((len(segments) + 1) * nsp, 3)
    assert final_time == pytest.approx(sum(opt.segment_durations_s), rel=1e-6)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0                       # threshold reached

    # Pre-FAF fix passage: the phase ending at the START of the leg into the FAF (the IF here)
    # must deliver the aircraft within that leg's k·RNP disc. The ONLY fix-passage requirement.
    if_fix = np.asarray(segments[1].start_ne, float)
    join_tolerance = segments[1].k_margin * segments[1].halfwidth_m
    assert float(np.hypot(*(ne[2 * nsp - 1] - if_fix))) <= join_tolerance + 1.0

    # FAC join, established EARLY: on the course, at least 1/5 of the final leg BEFORE the FAF
    # and at most half the previous (intermediate) leg before it.
    lpv = segments[-1].lpv
    jn = ne[(len(segments)) * nsp - 1]                  # end of the pre-final phase's block
    assert abs(float(ac.fac_cross_track(jn[0], jn[1], lpv))) < 1.0
    d_join = float(ac.fac_distance_to_ltp(jn[0], jn[1], lpv))
    d_faf = float(ac.fac_distance_to_ltp(FAF[0], FAF[1], lpv))
    prev = segments[-2]
    max_offset = 0.5 * float(np.hypot(*(np.asarray(prev.end_ne) - np.asarray(prev.start_ne))))
    max_offset = max(max_offset, 0.2 * d_faf)
    assert d_faf + 0.2 * d_faf - 1.0 <= d_join <= d_faf + max_offset + 1.0
    assert _faf_intercept_deg(states, frame, FAF) <= 30.0 + 1e-6        # standard intercept

    blk = slice(-nsp, None)                                             # the final LPV leg's block
    viol = ac.segment_violations_from_components(
        segments[-1], ne[blk, 0], ne[blk, 1], states[blk, 2], states[blk, 5])
    assert max(float(np.ravel(v).max()) for v in viol.values()) <= 1.0


def test_route_unwrapped_terminal_heading_solves_double_dogleg():
    # REGRESSION (KRDU H05LZ): two same-direction 90° corners accumulate the route heading to
    # target_psi + 2π. The plain initial-heading unwrap ties at exactly π and can pick the wrong
    # branch — pinning a terminal ψ the fix-by-fix route cannot reach without an impossible
    # extra turn inside the final corridor (Infeasible_Problem_Detected). The route-chained
    # unwrap pins the branch the route actually reaches.
    from dataclasses import replace
    frame = ac.TargetFrame(35.60, -78.50)
    ltp = np.array([0.0, 0.0])
    faf = np.array([-7071.0, -7071.0])            # final course = +π/4 (model convention)
    f2 = np.array([-707.0, -13435.0])             # intermediate course −π/4: +90° at the FAF
    f1 = np.array([4879.0, -7849.0])              # initial course −3π/4: +90° at F2
    # Backend-realistic floors + the pre-FAF vertical gate regularize the otherwise wildly
    # under-constrained vertical (without them IPOPT wanders through dive-and-climb iterates).
    final = _one_lpv_segment(ltp, faf)
    final = replace(final, lpv=replace(final.lpv, d_faf_m=10000.0, prefaf_floor_m=600.0))
    segments = [
        ac.SegmentSpec(ac.SegmentKind.INITIAL, f1, f2, "F1", "F2", halfwidth_m=1852.0,
                       step_downs=[ac.StepDown(s_from_start_m=7900.0, min_alt_m=750.0)],
                       max_descent_deg=4.7),
        ac.SegmentSpec(ac.SegmentKind.INTERMEDIATE, f2, faf, "F2", "FAF", halfwidth_m=1852.0,
                       step_downs=[ac.StepDown(s_from_start_m=9000.0, min_alt_m=600.0)],
                       max_descent_deg=4.7),
        final,
    ]
    init_ll = frame.to_latlon([11243.0, -1485.0])
    init = GeodeticState(float(init_ll[0]), float(init_ll[1]), 1300.0, 90.0,
                         5.0 * math.pi / 4.0, 0.0, MTOW)      # heading = target ψ + π (the tie)
    tll = frame.to_latlon(ltp)
    target = GeodeticState(float(tll[0]), float(tll[1]), 120.0 + 50.0 * 0.3048, 76.0,
                           math.pi / 4.0, math.radians(-3.0), MTOW)
    # Unit-level: the chained-course unwrap picks the route's branch (ψ_target + 2π), where the
    # plain initial-heading unwrap ties at exactly π and can pick ψ_target. The hull spans the
    # chain (init 3.93 … target 7.07), feeding the ψ-corridor variable bounds.
    probe = CollocationOptimizer(A320, segments=segments)
    unwrapped, hull_lo, hull_hi = probe._route_psi_profile(
        np.array([11243.0, -1485.0]), 5.0 * math.pi / 4.0, math.pi / 4.0)
    assert unwrapped == pytest.approx(math.pi / 4.0 + 2.0 * math.pi, abs=1e-9)
    assert hull_lo == pytest.approx(5.0 * math.pi / 4.0, abs=0.15)   # init heading end of hull
    assert hull_hi == pytest.approx(unwrapped, abs=1e-9)             # target end of hull
    # Solve-level, at a configuration that converges on this double-90° geometry (several nsp
    # values crawl to Max_Iterations — a known NLP-hardness limit of the knife-edge synthetic,
    # not a branch bug; the deterministic branch check is the unit assert above; see CLAUDE.md
    # 2026-07-04).
    opt = CollocationOptimizer(A320, segments=segments,
                               scheme="trapezoidalNormalizedFullTransport", n_seg_per_phase=5)
    _t, _c, states = opt.optimize_free_time(init, target, 600.0)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - ltp))) < 1.0
    # the terminal heading lands on the ROUTE's branch: ψ_target + 2π
    assert states[-1][4] == pytest.approx(math.pi / 4.0 + 2.0 * math.pi, abs=1e-3)
    # the pre-FAF fix (F2) is the one forced passage (end of the F1->F2 phase, which is the
    # second block: [transition, F1-leg, F2-leg(join), final])
    nsp = opt.n_seg_per_phase
    assert float(np.hypot(*(ne[2 * nsp - 1] - f2))) <= 0.5 * 1852.0 + 1.0


def test_only_the_prefaf_fix_carries_a_passage_disc():
    # E -> A -> B -> FAF -> LTP with A ~3 km OFF the direct E->B line: A (and the entry E) are
    # laterally FREE, so min-time cuts the corner past A; B (the pre-FAF fix) must be passed
    # within its leg's k·RNP disc.
    frame = ac.TargetFrame(35.60, -78.50)
    ltp = np.array([0.0, 0.0])
    faf = np.array([8000.0, 0.0])
    b = np.array([12000.0, 0.0])
    a = np.array([16000.0, 3000.0])
    e = np.array([20000.0, 0.0])
    segments = [
        ac.SegmentSpec(ac.SegmentKind.INITIAL, e, a, "E", "A", halfwidth_m=1852.0),
        ac.SegmentSpec(ac.SegmentKind.INITIAL, a, b, "A", "B", halfwidth_m=1852.0),
        ac.SegmentSpec(ac.SegmentKind.INTERMEDIATE, b, faf, "B", "FAF", halfwidth_m=1852.0),
        _one_lpv_segment(ltp, faf),
    ]
    e_ll = frame.to_latlon(e)
    init = GeodeticState(float(e_ll[0]), float(e_ll[1]), 1200.0, 90.0,
                         -math.pi / 2.0, 0.0, MTOW)            # at E, heading down-course
    tll = frame.to_latlon(ltp)
    target = GeodeticState(float(tll[0]), float(tll[1]), 120.0 + 50.0 * 0.3048, 76.0,
                           -math.pi / 2.0, math.radians(-3.0), MTOW)
    opt = CollocationOptimizer(A320, segments=segments)
    _t, _c, states = opt.optimize_free_time(init, target, 400.0)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    nsp = opt.n_seg_per_phase
    tol = 0.5 * 1852.0
    assert float(np.hypot(*(ne[-1] - ltp))) < 1.0
    assert float(np.hypot(*(ne[nsp - 1] - a))) > tol + 100.0        # A skipped (free)
    assert float(np.hypot(*(ne[2 * nsp - 1] - b))) <= tol + 1.0     # B passed (the disc)


def test_zero_join_offset_clamps_to_the_min_upstream_point():
    # max_join_offset_m=0 is clamped up to the 1/5-final minimum: the join collapses to the
    # single on-course point 1/5 of the final leg BEFORE the FAF (never at/after the FAF).
    init = _straight_in()
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)
    opt = CollocationOptimizer(A320, segments=segments, max_join_offset_m=0.0)
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    nsp = opt.n_seg_per_phase
    lpv = segments[-1].lpv
    jn = ne[(len(segments)) * nsp - 1]
    d_faf = float(ac.fac_distance_to_ltp(FAF[0], FAF[1], lpv))
    assert abs(float(ac.fac_cross_track(jn[0], jn[1], lpv))) < 1.0
    assert float(ac.fac_distance_to_ltp(jn[0], jn[1], lpv)) == pytest.approx(1.2 * d_faf, abs=1.0)
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0


def test_constrained_intercepts_final_from_an_offset_start():
    init = _straight_in()
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)
    offset = frame.to_latlon([float(np.hypot(*FAF)) + 4000.0, 4000.0])
    init_off = GeodeticState(float(offset[0]), float(offset[1]), S[0][2] + 300.0, 95.0,
                             math.radians(80.0), 0.0, MTOW)
    opt = CollocationOptimizer(A320, segments=segments)
    _t, _c, states = opt.optimize_free_time(init_off, target, 200.0)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0
    assert _faf_intercept_deg(states, frame, FAF) <= 30.0 + 1e-6


@pytest.mark.parametrize("heading_deg", [135.0, -100.0])
def test_constrained_solves_a_non_fixed_point_runway_heading(heading_deg):
    # REGRESSION (KRDU RW32): the FAF final-approach course must be the model's math-ENU heading,
    # NOT the compass bearing (they agree only at 45°/225°). A NW/SE runway reproduced it —
    # Infeasible before the fix, a clean aligned solve after.
    init = _straight_in(heading_deg=heading_deg)
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)
    opt = CollocationOptimizer(A320, segments=segments)
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0
    assert _faf_intercept_deg(states, frame, FAF) <= 30.0 + 1e-6


def test_constrained_fixed_time_hits_the_duration():
    init = _straight_in()
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, _FAF, LTP = _approach(S, frame)
    opt = CollocationOptimizer(A320, segments=segments)
    final_time, _c, states = opt.optimize_trajectory(init, target, duration=150.0)
    assert final_time == pytest.approx(150.0, abs=1e-2)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0


def test_constrained_enforces_step_down_floor_and_descent_cap():
    # The backend always codes an at-or-above floor + a descent cap on the pre-final legs; this
    # exercises those NLP rows end-to-end (incl. the moc_floor staircase on a CasADi vector).
    from dataclasses import replace
    init = _straight_in()
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    frame = ac.TargetFrame(target.latitude, target.longitude)

    def _segments_with(floor_offset_m, caps=(4.7, 3.5)):
        segments, FAF, LTP = _approach(S, frame)
        smp = lambda f: S[int(f * (len(S) - 1))]
        for i, (cap, end_frac) in enumerate(zip(caps, (0.55, 0.78))):
            leg_len = float(np.hypot(*(np.asarray(segments[i].end_ne) - np.asarray(segments[i].start_ne))))
            floor_alt = smp(end_frac)[2] + floor_offset_m
            segments[i] = replace(
                segments[i],
                step_downs=[ac.StepDown(s_from_start_m=leg_len, min_alt_m=floor_alt)],
                max_descent_deg=cap,
            )
        return segments, FAF, LTP

    # Floors slightly below the flown altitudes -> feasible, solves, and each leg's own phase
    # nodes (the ones the NLP applies that leg's rows to) satisfy the coded floor.
    segments, _FAF, LTP = _segments_with(floor_offset_m=-150.0)
    opt = CollocationOptimizer(A320, segments=segments)
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0
    nsp = opt.n_seg_per_phase          # phase 0 is the start->IAF transition
    for i, seg in enumerate(segments[:2]):
        blk = slice((i + 1) * nsp, (i + 2) * nsp)
        viol = ac.segment_violations_from_components(
            seg, ne[blk, 0], ne[blk, 1], states[blk, 2], states[blk, 5], include_lateral=False)
        floor_key = next(k for k in viol if k.endswith(".floor"))
        assert float(np.ravel(viol[floor_key]).max()) <= 1.0

    # An impossible floor (1 km above the start) -> the rows bind -> infeasible, raises.
    bad_segments, _FAF, _LTP = _segments_with(floor_offset_m=2000.0)
    bad = CollocationOptimizer(A320, segments=bad_segments)
    with pytest.raises(ValueError, match="failed"):
        bad.optimize_free_time(init, target, 120.0 * 1.6)


def test_terminal_bank_uses_previous_node_when_last_phase_has_one_node():
    # REGRESSION: with a single state node in the last phase, the terminal bank's psi_dot needs
    # the PREVIOUS phase's terminal node as ``prev``. A zeros placeholder turned the row into a
    # bogus bound on the ABSOLUTE heading (V·cosγ·ψ) instead of the heading RATE.
    init = _straight_in()
    target = _reachable_target(init, 120.0)
    ltp, mid, faf = np.array([0.0, 0.0]), np.array([2500.0, 0.0]), np.array([5000.0, 0.0])
    segments = [
        ac.SegmentSpec(ac.SegmentKind.INITIAL, faf, mid, "FAF", "MID", halfwidth_m=1852.0),
        _one_lpv_segment(ltp, mid),
    ]
    opt = CollocationOptimizer(A320, segments=segments, n_seg_per_phase=1, state_substeps=1)
    nlp, *_rest, layout = opt._build(init, target, 200.0)
    n_phases = len(layout["phase_nseg"])
    per_block = _schemes.CONTROL_DIM + _schemes.STATE_DIM      # n_seg = 1, m_sub = 1
    psi_prev = nlp["x"][(n_phases - 2) * per_block + _schemes.CONTROL_DIM + 4]
    bank_row = nlp["g"][-1]                                    # free-time build: bank is last
    assert ca.depends_on(bank_row, psi_prev)


def test_constrained_solves_a_near_sea_level_threshold():
    # REGRESSION (KMSY/KSMF): a threshold below the old absolute 25 m floor must solve.
    init = _straight_in(alt=620.0)
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], MTOW)
    assert target.altitude < 25.0
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)
    opt = CollocationOptimizer(A320, segments=segments)
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0
    assert states[-1][2] == pytest.approx(target.altitude, abs=1.0)


# ── the two modes, compared on ONE shared scenario set ─────────────────────
def test_unconstrained_vs_constrained_speed_and_success_rate(capsys):
    """The two ways the pipeline optimises a scenario, on the SAME inputs: unconstrained
    initial->target (asdb/runway) vs procedure-constrained (runway_cons). Reports success rate +
    solve time; both must solve the shared reachable set. The constrained solve does strictly more
    work (one phase per leg + corridor/glidepath/FAF), so this quantifies the cost."""
    scenarios = []
    for h in (40.0, 135.0, -100.0):
        init = _straight_in(heading_deg=h)
        S = _rollout_samples(init, 120.0)
        target = GeodeticState(*S[-1][:6], MTOW)
        frame = ac.TargetFrame(target.latitude, target.longitude)
        segments, _FAF, _LTP = _approach(S, frame)
        scenarios.append((init, target, segments))

    def benchmark(make_optimizer):
        solved, total_s = 0, 0.0
        for init, target, segments in scenarios:
            optimizer = make_optimizer(segments)
            started = time.perf_counter()
            try:
                optimizer.optimize_free_time(init, target, 120.0 * 1.6)
                solved += 1
            except Exception:      # noqa: BLE001 — a failed solve counts against the success rate
                pass
            total_s += time.perf_counter() - started
        return solved, total_s

    un_ok, un_s = benchmark(lambda _s: CollocationOptimizer(A320, scheme="trapezoidalNormalizedFullTransport"))
    co_ok, co_s = benchmark(lambda s: CollocationOptimizer(A320, segments=s))

    n = len(scenarios)
    with capsys.disabled():
        print(f"\n  optimizer comparison over {n} shared scenarios (success | total | per-scenario):")
        print(f"    unconstrained: {un_ok}/{n} | {un_s:5.2f}s | {un_s / n:.2f}s")
        print(f"    constrained  : {co_ok}/{n} | {co_s:5.2f}s | {co_s / n:.2f}s")
    assert un_ok == n
    assert co_ok == n
    assert un_s > 0.0 and co_s > 0.0


def test_first_leg_entry_floor():
    from collocation.optimizer import _first_leg_entry_floor_m
    from dataclasses import replace
    stepped = ac.SegmentSpec(
        ac.SegmentKind.INITIAL, np.array([1000.0, 0.0]), np.array([0.0, 0.0]), "A", "B",
        halfwidth_m=1852.0,
        step_downs=[ac.StepDown(s_from_start_m=400.0, min_alt_m=750.0),
                    ac.StepDown(s_from_start_m=800.0, min_alt_m=600.0)],
    )
    # the HIGHEST step binds at the leg's entry (the staircase descends along track)
    assert _first_leg_entry_floor_m(stepped) == 750.0
    plain = ac.SegmentSpec(
        ac.SegmentKind.INITIAL, np.array([1000.0, 0.0]), np.array([0.0, 0.0]), "A", "B",
        halfwidth_m=1852.0, base_floor_m=42.0,
    )
    assert _first_leg_entry_floor_m(plain) == 42.0
    lpv_seg = _one_lpv_segment(np.array([0.0, 0.0]), np.array([8000.0, 0.0]))
    with_prefaf = replace(lpv_seg, lpv=replace(lpv_seg.lpv, prefaf_floor_m=600.0))
    assert _first_leg_entry_floor_m(with_prefaf) == 600.0
    without = replace(lpv_seg, lpv=replace(lpv_seg.lpv, prefaf_floor_m=None))
    assert _first_leg_entry_floor_m(without) == 0.0


def test_transition_phase_altitude_floor_binds():
    # The start->first-fix transition carries no leg rows; its altitude is capped at
    # min(start altitude, the first leg's published entry floor) - the margin, so the
    # min-time "dive for speed then climb back" excursion (observed to target-300 m on
    # real batches) cannot exist. The whole plan also respects the global target floor.
    frame = ac.TargetFrame(35.60, -78.50)
    ltp = np.array([0.0, 0.0])
    faf = np.array([8000.0, 0.0])
    b = np.array([12000.0, 0.0])
    segments = [
        ac.SegmentSpec(ac.SegmentKind.INTERMEDIATE, b, faf, "B", "FAF", halfwidth_m=1852.0,
                       step_downs=[ac.StepDown(s_from_start_m=3000.0, min_alt_m=700.0)]),
        _one_lpv_segment(ltp, faf),
    ]
    s_ll = frame.to_latlon(np.array([22000.0, 0.0]))   # 10 km before B -> transition phase
    init = GeodeticState(float(s_ll[0]), float(s_ll[1]), 1500.0, 90.0,
                         -math.pi / 2.0, 0.0, MTOW)
    tll = frame.to_latlon(ltp)
    target = GeodeticState(float(tll[0]), float(tll[1]), 120.0 + 50.0 * 0.3048, 76.0,
                           -math.pi / 2.0, math.radians(-3.0), MTOW)
    opt = CollocationOptimizer(A320, segments=segments)
    _t, _c, states = opt.optimize_free_time(init, target, 400.0)

    transition_floor = min(init.altitude, 700.0) - ALTITUDE_FLOOR_MARGIN_M
    nsp = opt.n_seg_per_phase
    assert min(s[2] for s in states[:nsp]) >= transition_floor - 1e-3
    # global floor over the DENSE plan: never below target - margin anywhere
    global_floor = target.altitude - ALTITUDE_FLOOR_MARGIN_M
    dense_alts = [float(row[2]) for row in opt.last_dense_states_geo]
    assert min(dense_alts) >= global_floor - 1e-3
    # and the solve still lands on the target
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - ltp))) < 1.0
