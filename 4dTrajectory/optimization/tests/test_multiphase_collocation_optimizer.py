"""Multiphase constrained-approach optimiser: guards + a real convergent, feasible solve.

One phase per leg (start->IAF transition + the procedure legs); fixes pinned at the phase
boundaries (exact membership, no along-track partition); free per-phase time.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_OPT_DIR = Path(__file__).resolve().parents[1]
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as ca  # noqa: E402
import approach_constraints as ac  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator  # noqa: E402
from aircraft.aircraft_sets import A320  # noqa: E402
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from multiphase_collocation_optimizer import MultiphaseCollocationOptimizer  # noqa: E402


def _final_segment(ltp, faf):
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


def test_guards():
    seg = [_final_segment(np.array([0.0, 0.0]), np.array([5000.0, 0.0]))]
    with pytest.raises(ValueError, match="normalized full-transport"):
        MultiphaseCollocationOptimizer(A320, seg, scheme="hermiteSimpson")
    with pytest.raises(ValueError, match="at least one segment"):
        MultiphaseCollocationOptimizer(A320, [])
    with pytest.raises(ValueError, match="ipopt"):
        MultiphaseCollocationOptimizer(A320, seg, solver_backend="sqpmethod")


def _rollout_samples(init, horizon, dt=0.05):
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


def _approach(S, frame, fracs=(0.30, 0.55, 0.78, 1.0)):
    """Build INITIAL/INTERMEDIATE/FINAL_LPV segments from rollout samples at ``fracs``."""
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
    )
    segments = [
        ac.SegmentSpec(ac.SegmentKind.INITIAL, IAF, IF, "IAF", "IF", halfwidth_m=1852.0),
        ac.SegmentSpec(ac.SegmentKind.INTERMEDIATE, IF, FAF, "IF", "FAF", halfwidth_m=1852.0),
        ac.SegmentSpec(ac.SegmentKind.FINAL_LPV, FAF, LTP, "FAF", "LTP", lpv=lpv),
    ]
    return segments, FAF, LTP


def _faf_intercept_deg(states, frame, faf_ne):
    """Heading-vs-final-approach-course angle (deg) at the state nearest the FAF."""
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    i = int(np.argmin([np.hypot(*(p - faf_ne)) for p in ne]))
    fac = math.degrees(math.atan2(-faf_ne[0], -faf_ne[1]))   # FAF -> LTP(0,0), model-ENU (0=E, CCW)
    return abs(((math.degrees(states[i][4]) - fac + 180) % 360) - 180)


def test_multiphase_free_approach_pins_faf_and_threshold_only():
    # rollout-derived (reachable) straight-in; the start is NOT a procedure fix.
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0, math.radians(40.0), math.radians(-3.0),
                         A320.mass.max_takeoff_kg)
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], A320.mass.max_takeoff_kg)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)

    opt = MultiphaseCollocationOptimizer(A320, segments)   # max_intercept default 30 deg
    final_time, controls, states = opt.optimize_free_time(init, target, 120.0 * 1.6)

    nsp = opt.n_seg_per_phase
    assert controls.shape == (len(segments) * nsp, 3)       # one phase per leg, no transition phase
    assert states.shape == (len(segments) * nsp, 6)
    assert final_time == pytest.approx(sum(opt.segment_durations_s), rel=1e-6)

    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    # only the FAF + threshold are pinned (the pre-final horizontal path is free)
    assert min(float(np.hypot(*(p - FAF))) for p in ne) < 1.0          # FAF crossed
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0                       # threshold reached
    assert _faf_intercept_deg(states, frame, FAF) <= 30.0 + 1e-6        # standard intercept
    # the final LPV leg satisfies its corridor + glidepath
    blk = slice((len(segments) - 1) * nsp, len(segments) * nsp)
    viol = ac.segment_violations_from_components(
        segments[-1], ne[blk, 0], ne[blk, 1], states[blk, 2], np.radians(states[blk, 5])
    )
    assert max(float(np.ravel(v).max()) for v in viol.values()) <= 1.0


def test_multiphase_intercepts_final_from_an_offset_start():
    # The HEAVE-class win: a start OFF the final approach line still converges, with the optimiser
    # turning onto final at <= 30 deg (instead of an unflyable sharp turn pinned at the FAF).
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0, math.radians(40.0), math.radians(-3.0),
                         A320.mass.max_takeoff_kg)
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], A320.mass.max_takeoff_kg)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)
    # shove the start ~4 km to the side of the final line (free pre-final path must turn onto final)
    offset = frame.to_latlon([float(np.hypot(*FAF)) + 4000.0, 4000.0])
    init_off = GeodeticState(float(offset[0]), float(offset[1]), S[0][2] + 300.0, 95.0,
                             math.radians(80.0), 0.0, A320.mass.max_takeoff_kg)

    opt = MultiphaseCollocationOptimizer(A320, segments)
    _t, _c, states = opt.optimize_free_time(init_off, target, 200.0)
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0
    assert _faf_intercept_deg(states, frame, FAF) <= 30.0 + 1e-6


@pytest.mark.parametrize("heading_deg", [135.0, -100.0])
def test_multiphase_solves_a_non_fixed_point_runway_heading(heading_deg):
    # REGRESSION (KRDU RW32). The FAF final-approach course must be the model's math-ENU heading
    # (0 = East, CCW toward North) — the SAME convention as the state ``psi`` the FAF-intercept is
    # compared against — NOT the compass bearing. The two agree ONLY near the 45deg/225deg fixed
    # points: KRDU's 05/23 runways sit there and masked a 180deg convention error in ``fac_rad``,
    # while RW32 (~315deg) hit it and made EVERY constrained solve infeasible (the intercept
    # demanded a heading ~180deg opposed to the pinned target). A runway heading well away from a
    # fixed point (here NW ~135deg and SE ~-100deg) reproduces it: infeasible before the fix, and a
    # clean aligned solve after. (The two existing tests above use ~40deg == on the fixed point, so
    # they could not catch this.)
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0, math.radians(heading_deg), math.radians(-3.0),
                         A320.mass.max_takeoff_kg)
    S = _rollout_samples(init, 120.0)
    target = GeodeticState(*S[-1][:6], A320.mass.max_takeoff_kg)
    frame = ac.TargetFrame(target.latitude, target.longitude)
    segments, FAF, LTP = _approach(S, frame)

    opt = MultiphaseCollocationOptimizer(A320, segments)
    _t, _c, states = opt.optimize_free_time(init, target, 120.0 * 1.6)   # raised Infeasible pre-fix
    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    assert float(np.hypot(*(ne[-1] - LTP))) < 1.0                        # threshold reached
    assert _faf_intercept_deg(states, frame, FAF) <= 30.0 + 1e-6         # aligned at the FAF
