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
