"""Start-state and target (end-of-track) kinematics."""

import math

import pytest

from flight_scenarios.start_state import (
    final_state_from_track,
    initial_state_from_track,
)

# A synthetic 100 m/s due-EAST, level track at the equator. Using the WGS84 semi-major
# axis (geokit's default), 500 m east over a 5 s window is this many degrees of longitude:
_LON_STEP_DEG = 500.0 / (math.pi / 180.0 * 6378137.0)
DUE_EAST_LEVEL = [
    [0.0, 0.0, 0.0, 1000.0],
    [5.0, _LON_STEP_DEG, 0.0, 1000.0],
    [10.0, 2 * _LON_STEP_DEG, 0.0, 1000.0],
]


def test_too_few_waypoints_raises():
    with pytest.raises(ValueError):
        initial_state_from_track([[0.0, 0.0, 0.0, 1000.0]], mass_kg=78000.0)


def test_due_east_level_track():
    s = initial_state_from_track(DUE_EAST_LEVEL, mass_kg=78000.0)
    # position is read straight off the anchor sample
    assert (s.latitude, s.longitude, s.altitude) == (0.0, 0.0, 1000.0)
    assert s.m == 78000.0
    # kinematics estimated over the 5 s window
    assert s.V == pytest.approx(100.0, rel=1e-2)
    assert s.psi == pytest.approx(math.pi / 2, abs=1e-2)   # due east
    assert s.gamma == pytest.approx(0.0, abs=1e-3)          # level


def test_climbing_track_has_positive_gamma():
    # same 500 m horizontal step, +250 m climb over the window
    climbing = [[0.0, 0.0, 0.0, 1000.0], [5.0, _LON_STEP_DEG, 0.0, 1250.0]]
    s = initial_state_from_track(climbing, mass_kg=1157.0)
    assert s.gamma == pytest.approx(math.atan2(250.0, 500.0), abs=1e-2)
    assert s.gamma > 0


def test_final_state_is_anchored_at_track_end():
    # The target is the state at the LAST sample; same due-east level track.
    s = final_state_from_track(DUE_EAST_LEVEL, mass_kg=78000.0)
    assert (s.latitude, s.longitude, s.altitude) == (0.0, 2 * _LON_STEP_DEG, 1000.0)
    assert s.V == pytest.approx(100.0, rel=1e-2)
    assert s.psi == pytest.approx(math.pi / 2, abs=1e-2)   # still due east
    assert s.gamma == pytest.approx(0.0, abs=1e-3)


def test_final_state_too_few_waypoints_raises():
    with pytest.raises(ValueError):
        final_state_from_track([[0.0, 0.0, 0.0, 1000.0]], mass_kg=78000.0)
