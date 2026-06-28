"""Procedure path-constraints wired into the direct-collocation NLP (normalized full transport).

Wiring + guards (fast: no solve), plus one real constrained IPOPT solve that must converge AND
return a trajectory inside the corridor + glidepath.
"""

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

import constraints as ac  # noqa: E402
from constraints import examples as ac_examples  # noqa: E402
from casadi_direct_collocation_optimizer import (  # noqa: E402
    _NORMALIZED_FULL_TRANSPORT_SCHEMES,
    CasadiDirectCollocationOptimizer,
)
from aircraft.aircraft_sets import A320  # noqa: E402

SCHEME = "trapezoidalNormalizedFullTransport"


def _segments_and_spans():
    segs = ac_examples.build_example_segments()
    lengths = [float(np.hypot(*(s.end_ne - s.start_ne))) for s in segs]
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    spans = [(float(cum[i]), float(cum[i + 1])) for i in range(len(segs))]
    return segs, spans


def _make(scheme, *, segs=None, spans=None):
    return CasadiDirectCollocationOptimizer(
        n_segments=4, dt=5.0, max_duration=600.0, aircraft=A320,
        collocation_scheme=scheme, constraint_segments=segs, constraint_spans=spans,
    )


def test_constraints_add_inequality_rows():
    segs, spans = _segments_and_spans()
    baseline = _make(SCHEME)
    constrained = _make(SCHEME, segs=segs, spans=spans)
    extra = len(constrained.lbg) - len(baseline.lbg)
    assert extra > 0, "constrained NLP must gain inequality rows"
    # the extra rows are one-sided: lb = -inf, ub = 0
    assert constrained.ubg[-extra:] == [0.0] * extra
    assert all(np.isneginf(v) for v in constrained.lbg[-extra:])
    # free-time NLP gets the same rows
    assert len(constrained.free_time_lbg) - len(baseline.free_time_lbg) == extra


def test_constraints_rejected_on_non_normalized_scheme():
    segs, spans = _segments_and_spans()
    with pytest.raises(ValueError, match="normalized full-transport"):
        _make("hermiteSimpson", segs=segs, spans=spans)


def test_spans_must_align_with_segments():
    segs, spans = _segments_and_spans()
    with pytest.raises(ValueError, match="align 1:1"):
        _make(SCHEME, segs=segs, spans=spans[:-1])


def test_all_normalized_full_transport_schemes_accept_constraints():
    segs, spans = _segments_and_spans()
    for scheme in _NORMALIZED_FULL_TRANSPORT_SCHEMES:
        opt = _make(scheme, segs=segs, spans=spans)
        assert len(opt.lbg) > 0


def _rollout_samples(init, aircraft, horizon, dt=0.05):
    """Forward-propagate the geodetic RHS (full transport) -> a dynamically reachable path."""
    import casadi as ca
    from aerodynamic_model.casadi_simulator import make_geodetic_step_integrator
    from aircraft.aero_params import aero_params_for_aircraft

    ap = aero_params_for_aircraft(aircraft)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    step = make_geodetic_step_integrator(transport="full")["step_func"]
    u = ca.DM([aircraft.approach.thrust_guess_n, 0.0, 1.0])
    x = ca.DM([init.latitude, init.longitude, init.altitude, init.V, init.psi, init.gamma, init.m])
    samples = [np.array(x).reshape(-1)]
    for _ in range(int(horizon / dt)):
        x = step(x_geo=x, u=u, aero_params=aero, dt=dt)["x_geo_next"]
        samples.append(np.array(x).reshape(-1))
    return samples


def test_constrained_solve_converges_and_returns_a_feasible_trajectory():
    """A real IPOPT solve: the smooth corridor + glidepath must converge from the cold start and
    the returned nodes must lie inside the constraints. (Regression for the |cross-track| kink that
    made the gradient-based solve fail — the corridor is now two smooth rows.)"""
    import math
    from aerodynamic_model.common import GeodeticState

    horizon = 90.0
    init = GeodeticState(35.60, -78.50, 1500.0, 90.0,
                         math.radians(40.0), math.radians(-3.0), A320.mass.max_takeoff_kg)
    samples = _rollout_samples(init, A320, horizon)
    target = GeodeticState(*samples[-1][:6], A320.mass.max_takeoff_kg)

    frame = ac.TargetFrame(target.latitude, target.longitude)
    faf_ne = frame.to_ne(samples[0][0], samples[0][1])
    norm = float(np.hypot(*faf_ne))
    inbound = -faf_ne / norm
    ft = 0.3048
    d_garp = (9023.0 + 1000.0) * ft
    lpv = ac.LpvFinalSpec(
        ltp_ne=np.array([0.0, 0.0]),
        fpap_ne=9023.0 * ft * inbound,
        garp_ne=d_garp * inbound,
        course_width_m=max(350.0 * ft, math.tan(math.radians(1.5)) * d_garp),
        tdze_m=target.altitude - 50.0 * ft,
        tch_m=50.0 * ft,
        gpa_deg=3.0,
        below_m=60.0,
        above_m=60.0,
    )
    seg = ac.SegmentSpec(ac.SegmentKind.FINAL_LPV, start_ne=faf_ne, end_ne=np.array([0.0, 0.0]),
                         lpv=lpv, k_margin=0.5)

    opt = CasadiDirectCollocationOptimizer(
        n_segments=10, dt=0.2, max_duration=horizon * 1.6, aircraft=A320,
        collocation_scheme=SCHEME, constraint_segments=[seg], constraint_spans=[(0.0, norm)],
    )
    _, _, states = opt.optimize_trajectory(init, target, duration=horizon)
    assert opt.solver.stats()["success"]

    ne = np.array([frame.to_ne(s[0], s[1]) for s in states])
    viol = ac.segment_violations_from_components(
        seg, ne[:, 0], ne[:, 1], states[:, 2], states[:, 5]
    )
    worst = max(float(np.ravel(v).max()) for v in viol.values())
    assert worst <= 1.0, f"returned path violates constraints by {worst:.2f} m"
