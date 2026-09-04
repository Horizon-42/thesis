"""The FAS cone is one definition, shared by the optimizer bridge and the learned corridor."""

import math

import numpy as np
import pytest
from geokit import FT_M

from flight_scenarios.fas_geometry import (
    COURSE_WIDTH_FLOOR_M,
    FPAP_FLOOR_M,
    GARP_BEYOND_FPAP_M,
    course_halfwidth_m,
    fas_course_geometry,
)


def test_unknown_runway_length_uses_the_9023_ft_floor_and_the_350_ft_width():
    fas = fas_course_geometry()
    assert fas.d_fpap_m == pytest.approx(9023.0 * FT_M)
    assert fas.d_garp_m == pytest.approx((9023.0 + 1000.0) * FT_M)
    # tan(1.5°) · 3055 m ≈ 80 m sits below the 350 ft floor, so the floor binds.
    assert fas.course_width_m == pytest.approx(350.0 * FT_M)
    assert fas.course_width_m == COURSE_WIDTH_FLOOR_M
    assert FPAP_FLOOR_M + GARP_BEYOND_FPAP_M == fas.d_garp_m


def test_a_long_runway_widens_the_course():
    fas = fas_course_geometry(runway_length_m=4000.0)
    assert fas.d_fpap_m == 4000.0
    assert fas.course_width_m == pytest.approx(math.tan(math.radians(1.5)) * fas.d_garp_m)
    assert fas.course_width_m > COURSE_WIDTH_FLOOR_M


def test_halfwidth_is_the_course_width_at_the_ltp_and_grows_linearly():
    fas = fas_course_geometry()
    assert course_halfwidth_m(0.0, fas) == pytest.approx(fas.course_width_m)
    d = np.array([0.0, 5_000.0, 10_000.0])
    hw = course_halfwidth_m(d, fas)
    assert np.allclose(hw, fas.course_width_m * (d + fas.d_garp_m) / fas.d_garp_m)
    assert hw[2] - hw[1] == pytest.approx(hw[1] - hw[0])
