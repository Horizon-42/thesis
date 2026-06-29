"""TargetFrame is provided complete, so these PASS now (they guard the reference transform)."""

import numpy as np
from geokit import DEG2RAD, WGS84_A

from approach_constraints.frame import TargetFrame


def test_target_maps_to_origin():
    fr = TargetFrame(35.8776, -78.7875)
    assert np.allclose(fr.to_ne(35.8776, -78.7875), [0.0, 0.0])


def test_one_degree_north_is_R_times_deg2rad():
    fr = TargetFrame(0.0, 0.0)
    n, e = fr.to_ne(1.0, 0.0)
    assert n > 0.0
    assert np.isclose(n, WGS84_A * DEG2RAD)
    assert np.isclose(e, 0.0)


def test_east_uses_target_latitude_cosine():
    fr = TargetFrame(60.0, 0.0)
    _n, e = fr.to_ne(60.0, 1.0)
    assert np.isclose(e, WGS84_A * DEG2RAD * np.cos(60.0 * DEG2RAD))


def test_round_trip_array():
    fr = TargetFrame(35.0, -78.0)
    lats = np.array([35.0, 35.1, 34.9])
    lons = np.array([-78.0, -77.9, -78.2])
    ne = fr.to_ne(lats, lons)
    back = fr.to_latlon(ne)
    assert np.allclose(back[:, 0], lats)
    assert np.allclose(back[:, 1], lons)
