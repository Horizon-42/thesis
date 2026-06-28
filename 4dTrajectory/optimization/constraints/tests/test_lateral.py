"""TODO ⑤–⑦ (lateral). xfail until implemented; scalar-component API."""

import numpy as np
import pytest

from constraints import lateral
from constraints.segments import LpvFinalSpec

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


@pytest.mark.xfail(reason="TODO ⑤ box_corridor_violation", strict=False)
def test_box_corridor():
    assert np.isclose(lateral.box_corridor_violation(3.0, 7.0, A, B, 20.0, k=0.5), -3.0)
    n = np.array([3.0, 3.0])
    e = np.array([7.0, 15.0])
    assert np.allclose(lateral.box_corridor_violation(n, e, A, B, 20.0, k=0.5), [-3.0, 5.0])


@pytest.mark.xfail(reason="TODO ⑥ lpv_course_halfwidth", strict=False)
def test_lpv_halfwidth_converges_to_course_width_at_ltp():
    lpv = _lpv()
    assert np.isclose(lateral.lpv_course_halfwidth(0.0, 0.0, lpv), 106.7, rtol=1e-3)
    wide = lateral.lpv_course_halfwidth(9260.0, 0.0, lpv)   # near the PFAF
    assert np.isclose(wide, 106.7 * (9260.0 + 3305.0) / 3305.0, rtol=1e-3)
    assert wide > 106.7


@pytest.mark.xfail(reason="TODO ⑦ lpv_corridor_violation", strict=False)
def test_lpv_corridor_on_and_off_centerline():
    lpv = _lpv()
    assert lateral.lpv_corridor_violation(5000.0, 0.0, lpv, k=0.5) < 0.0   # on centerline -> inside
    assert lateral.lpv_corridor_violation(0.0, 200.0, lpv, k=0.5) > 0.0    # 200 m off near threshold
