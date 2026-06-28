"""TODO ⑧, ⑨, ⑩ (vertical). xfail until implemented; scalar-component API.

The window/floor *wrappers* are provided but call the TODO functions, so they xfail too until the
underlying piece is done.
"""

import math

import numpy as np
import pytest

from constraints import vertical
from constraints.segments import LpvFinalSpec, StepDown


def _lpv():
    return LpvFinalSpec(
        ltp_ne=np.array([0.0, 0.0]),
        fpap_ne=np.array([-3000.0, 0.0]),
        garp_ne=np.array([-3305.0, 0.0]),
        course_width_m=106.7,
        tdze_m=120.0,
        tch_m=15.24,
        gpa_deg=3.0,
        below_m=15.0,
        above_m=30.0,
    )


@pytest.mark.xfail(reason="TODO ⑧ glidepath_altitude", strict=False)
def test_glidepath_altitude():
    lpv = _lpv()
    assert np.isclose(vertical.glidepath_altitude(0.0, lpv), 120.0 + 15.24)
    assert np.isclose(
        vertical.glidepath_altitude(1000.0, lpv),
        120.0 + 15.24 + 1000.0 * math.tan(math.radians(3.0)),
    )


@pytest.mark.xfail(reason="TODO ⑧ via the provided window wrapper", strict=False)
def test_glidepath_window_on_path_is_satisfied():
    lpv = _lpv()
    gp = 120.0 + 15.24 + 1000.0 * math.tan(math.radians(3.0))
    low, high = vertical.glidepath_window_violation(gp, 1000.0, lpv)
    assert low <= 0.0 and high <= 0.0
    low2, _ = vertical.glidepath_window_violation(gp - 100.0, 1000.0, lpv)  # well below -> "too low"
    assert low2 > 0.0


@pytest.mark.xfail(reason="TODO ⑨ moc_floor", strict=False)
def test_moc_floor_staircase():
    steps = [StepDown(2000.0, 1524.0), StepDown(5000.0, 914.0)]
    assert np.isclose(vertical.moc_floor(3000.0, steps), 914.0)   # only the 5000 m step ahead
    assert np.isclose(vertical.moc_floor(1000.0, steps), 1524.0)  # both ahead -> the higher one
    assert np.isclose(vertical.moc_floor(6000.0, steps), 0.0)     # none ahead -> base
    # vectorised over nodes
    assert np.allclose(vertical.moc_floor(np.array([1000.0, 3000.0, 6000.0]), steps),
                       [1524.0, 914.0, 0.0])


@pytest.mark.xfail(reason="TODO ⑩ (bonus) descent_gradient_violation", strict=False)
def test_descent_gradient_cap():
    assert vertical.descent_gradient_violation(math.radians(-2.5), 3.0) < 0.0   # shallow -> ok
    assert vertical.descent_gradient_violation(math.radians(-4.0), 3.0) > 0.0   # too steep
