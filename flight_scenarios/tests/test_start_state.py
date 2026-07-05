"""Start-state and target (end-of-track) kinematics."""

import math

import pytest

from flight_scenarios.start_state import (
    final_state_from_track,
    initial_state_from_track,
    state_samples_from_track,
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
    # math-ENU heading: psi = 0 is due East (NOT the compass bearing pi/2)
    assert s.psi == pytest.approx(0.0, abs=1e-2)           # due east
    assert s.gamma == pytest.approx(0.0, abs=1e-3)          # level


def test_due_north_level_track():
    # Locks the convention on the other axis: due North must be psi = +pi/2 in math-ENU
    # (the compass bearing of due North is 0 — the reflection bug would return 0 here).
    lat_step_deg = 500.0 / (math.pi / 180.0 * 6378137.0)
    due_north = [
        [0.0, 0.0, 0.0, 1000.0],
        [5.0, 0.0, lat_step_deg, 1000.0],
        [10.0, 0.0, 2 * lat_step_deg, 1000.0],
    ]
    s = initial_state_from_track(due_north, mass_kg=78000.0)
    assert s.V == pytest.approx(100.0, rel=1e-2)
    assert s.psi == pytest.approx(math.pi / 2, abs=1e-2)   # due north


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
    assert s.psi == pytest.approx(0.0, abs=1e-2)           # still due east (math-ENU)
    assert s.gamma == pytest.approx(0.0, abs=1e-3)


def test_final_state_too_few_waypoints_raises():
    with pytest.raises(ValueError):
        final_state_from_track([[0.0, 0.0, 0.0, 1000.0]], mass_kg=78000.0)


def test_velocity_is_robust_to_a_stuck_sample():
    # Clean 100 m/s due-east motion with one *stuck* (duplicate-position) sample in the
    # middle — the kind of low-altitude ADS-B jitter that makes a 2-point estimate under-read.
    # The least-squares fit must drop the stuck report and recover the true speed.
    step = _LON_STEP_DEG / 5.0  # one second of 100 m/s due east
    stuck = [
        [0.0, 0.0, 0.0, 1000.0],
        [1.0, 1 * step, 0.0, 1000.0],
        [2.0, 1 * step, 0.0, 1000.0],  # STUCK: identical position to t=1
        [3.0, 3 * step, 0.0, 1000.0],
        [4.0, 4 * step, 0.0, 1000.0],
        [5.0, 5 * step, 0.0, 1000.0],
    ]
    s = final_state_from_track(stuck, mass_kg=78000.0)
    assert s.V == pytest.approx(100.0, rel=2e-2)          # not dragged down by the stuck sample
    assert s.psi == pytest.approx(0.0, abs=1e-2)          # still due east (math-ENU)


def test_state_samples_from_track_derives_every_sample():
    samples = state_samples_from_track(DUE_EAST_LEVEL, mass_kg=78000.0)
    assert len(samples) == len(DUE_EAST_LEVEL)
    assert [t for t, _ in samples] == pytest.approx([0.0, 5.0, 10.0])
    for _, state in samples:
        assert state.V == pytest.approx(100.0, rel=1e-4)   # 500 m per 5 s window
        assert state.psi == pytest.approx(0.0, abs=1e-9)   # due east (math-ENU)
        assert state.gamma == pytest.approx(0.0, abs=1e-9)
        assert state.m == 78000.0
    # positions read straight off the samples
    assert samples[1][1].longitude == pytest.approx(_LON_STEP_DEG)
    assert samples[-1][1].altitude == 1000.0


def test_state_samples_from_track_requires_two_waypoints():
    with pytest.raises(ValueError):
        state_samples_from_track([[0.0, 0.0, 0.0, 1000.0]], mass_kg=78000.0)
