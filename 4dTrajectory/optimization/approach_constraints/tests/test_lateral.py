"""Lateral corridor ⑤–⑦: box + LPV angular corridor + course half-width (scalar-component API)."""

import numpy as np

from approach_constraints import lateral
from approach_constraints.segments import LpvFinalSpec

A = np.array([0.0, 0.0])
B = np.array([10.0, 0.0])


def _lpv():
    return LpvFinalSpec(
        ltp_ne=np.array([0.0, 0.0]),
        fpap_ne=np.array([-3000.0, 0.0]),
        garp_ne=np.array([-3305.0, 0.0]),
        course_width_m=106.7,
        tdze_m=120.0,
        tch_m=15.24,
        gpa_deg=3.0,
    )


def test_box_corridor():
    # two smooth rows (right, left); their max is the |e_xt| - margin "abs" form
    right, left = lateral.box_corridor_violation(3.0, 7.0, A, B, 20.0, k=0.5)
    assert np.isclose(max(right, left), -3.0)        # inside by 3 m
    right2, left2 = lateral.box_corridor_violation(3.0, 15.0, A, B, 20.0, k=0.5)
    assert np.isclose(max(right2, left2), 5.0)       # outside by 5 m
    # vectorised over nodes
    n = np.array([3.0, 3.0]); e = np.array([7.0, 15.0])
    r, l = lateral.box_corridor_violation(n, e, A, B, 20.0, k=0.5)
    assert np.allclose(np.maximum(r, l), [-3.0, 5.0])


def test_lpv_halfwidth_converges_to_course_width_at_ltp():
    lpv = _lpv()
    assert np.isclose(lateral.lpv_course_halfwidth(0.0, 0.0, lpv), 106.7, rtol=1e-3)
    wide = lateral.lpv_course_halfwidth(9260.0, 0.0, lpv)   # near the PFAF
    assert np.isclose(wide, 106.7 * (9260.0 + 3305.0) / 3305.0, rtol=1e-3)
    assert wide > 106.7


def test_lpv_corridor_on_and_off_centerline():
    lpv = _lpv()
    r, l = lateral.lpv_corridor_violation(5000.0, 0.0, lpv, k=0.5)   # on centerline -> inside
    assert max(r, l) < 0.0
    r2, l2 = lateral.lpv_corridor_violation(0.0, 200.0, lpv, k=0.5)  # 200 m off near threshold
    assert max(r2, l2) > 0.0


def test_fix_passage_disc_is_metre_scaled():
    fix = np.array([1000.0, -500.0])
    # inside / on the boundary / outside — and metre-scaled near the boundary
    assert lateral.fix_passage_violation(1000.0, -500.0, fix, 926.0) < 0.0        # at the fix
    assert np.isclose(lateral.fix_passage_violation(1000.0, 426.0, fix, 926.0), 0.0)  # on the rim
    ten_m_out = lateral.fix_passage_violation(1000.0, 436.0, fix, 926.0)
    assert np.isclose(ten_m_out, 10.0, atol=0.1)                                  # ≈ metres over
    # vectorised over nodes
    n = np.array([1000.0, 3000.0])
    e = np.array([-500.0, -500.0])
    v = lateral.fix_passage_violation(n, e, fix, 926.0)
    assert v[0] < 0.0 < v[1]


def test_fac_cross_track_and_distance_to_ltp():
    lpv = _lpv()   # course along +n, LTP at the origin, GARP at n = -3305
    assert np.isclose(lateral.fac_cross_track(5000.0, 0.0, lpv), 0.0)   # on the course
    assert np.isclose(abs(lateral.fac_cross_track(5000.0, 120.0, lpv)), 120.0)
    assert np.isclose(lateral.fac_distance_to_ltp(0.0, 0.0, lpv), 0.0)  # at the threshold
    assert np.isclose(lateral.fac_distance_to_ltp(9260.0, 0.0, lpv), 9260.0)


def test_fac_join_window_established_before_the_faf():
    lpv = _lpv()
    d_faf, max_off, min_off = 9260.0, 4000.0, 1852.0   # min = "1/5 of the final" style bound
    inside = lateral.fac_join_window_violation(11500.0, 0.0, lpv, d_faf, max_off, min_off)
    assert max(float(v) for v in inside) <= 0.0            # within [d_faf+min, d_faf+max]
    at_faf = lateral.fac_join_window_violation(9260.0, 0.0, lpv, d_faf, max_off, min_off)
    assert max(float(v) for v in at_faf) > 0.0             # AT the FAF: too late now
    downstream = lateral.fac_join_window_violation(8000.0, 0.0, lpv, d_faf, max_off, min_off)
    assert max(float(v) for v in downstream) > 0.0         # between FAF and runway: forbidden
    too_far = lateral.fac_join_window_violation(13500.0, 0.0, lpv, d_faf, max_off, min_off)
    assert max(float(v) for v in too_far) > 0.0            # beyond the upstream window
    # both offsets 0 collapse the window to exactly the FAF (backward-compatible default)
    exact = lateral.fac_join_window_violation(9260.0, 0.0, lpv, d_faf, 0.0)
    assert max(float(v) for v in exact) <= 1e-9
    upstream = lateral.fac_join_window_violation(9300.0, 0.0, lpv, d_faf, 0.0)
    assert max(float(v) for v in upstream) > 0.0
