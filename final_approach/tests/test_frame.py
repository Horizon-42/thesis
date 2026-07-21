"""The runway frame: axis signs, the compass convention, and the chart scale."""

from __future__ import annotations

import math

import pytest
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from final_approach import RunwayFrame, TrackPoint

# A due-north runway keeps the arithmetic checkable by hand: along == north, cross == east.
NORTH = RunwayFrame(ident="36", lat=35.0, lon=-78.0, elevation_m=100.0, course_deg=0.0)


def _offset(frame: RunwayFrame, *, north_m: float = 0.0, east_m: float = 0.0, alt_m: float = 100.0):
    return TrackPoint(
        lat=frame.lat + north_m / METRES_PER_DEG_LAT,
        lon=frame.lon + east_m / metres_per_deg_lon(frame.lat),
        alt_m=alt_m,
    )


def test_threshold_projects_to_the_origin():
    p = NORTH.project(TrackPoint(NORTH.lat, NORTH.lon, NORTH.elevation_m))
    assert p.along_m == pytest.approx(0.0, abs=1e-6)
    assert p.cross_m == pytest.approx(0.0, abs=1e-6)
    assert p.height_m == pytest.approx(0.0, abs=1e-9)


def test_before_the_threshold_is_negative_along():
    """The sign convention the whole package depends on: approaching == negative."""
    p = NORTH.project(_offset(NORTH, north_m=-3000.0))
    assert p.along_m == pytest.approx(-3000.0, abs=0.5)


def test_cross_track_is_right_positive():
    """Heading north, east is to the RIGHT."""
    assert NORTH.project(_offset(NORTH, east_m=250.0)).cross_m == pytest.approx(250.0, abs=0.5)
    assert NORTH.project(_offset(NORTH, east_m=-250.0)).cross_m == pytest.approx(-250.0, abs=0.5)


def test_height_is_relative_to_threshold_elevation():
    assert NORTH.project(_offset(NORTH, alt_m=415.0)).height_m == pytest.approx(315.0)


def test_course_is_compass_not_math_enu():
    """course_deg=90 must mean EAST. Under math-ENU it would mean north, rotating the
    frame 90 deg -- the exact mix-up that reads an aligned aircraft as an intercept."""
    east = RunwayFrame(ident="09", lat=35.0, lon=-78.0, elevation_m=0.0, course_deg=90.0)
    p = east.project(_offset(east, east_m=-2000.0, alt_m=0.0))
    assert p.along_m == pytest.approx(-2000.0, abs=0.5)
    assert p.cross_m == pytest.approx(0.0, abs=0.5)


def test_opposite_ends_share_a_centreline_but_flip_along():
    """Why cross-track alone cannot separate the two ends of one runway."""
    south = RunwayFrame(ident="18", lat=NORTH.lat, lon=NORTH.lon, elevation_m=100.0, course_deg=180.0)
    point = _offset(NORTH, north_m=-3000.0)
    assert NORTH.project(point).along_m == pytest.approx(-3000.0, abs=0.5)
    assert south.project(point).along_m == pytest.approx(+3000.0, abs=0.5)
    # Both see it on the centreline -- the offset is identical in magnitude.
    assert abs(south.project(point).cross_m) == pytest.approx(abs(NORTH.project(point).cross_m), abs=0.5)


def test_distance_is_horizontal_only():
    far = _offset(NORTH, north_m=-3000.0, alt_m=5000.0)
    assert NORTH.distance_m(far) == pytest.approx(3000.0, abs=0.5)
