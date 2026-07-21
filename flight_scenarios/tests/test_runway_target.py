"""Runway-threshold target state."""

import math

import pytest

from aircraft.aircraft_sets import A320
from flight_scenarios.runway_target import find_threshold, threshold_target_state


def test_find_threshold_known():
    thr = find_threshold("KRDU", "05L")
    assert thr is not None
    assert thr["ident"] == "05L"
    assert thr["heading_deg"] == 45.0


def test_find_threshold_case_insensitive_and_unknown():
    assert find_threshold("krdu", "05l") is not None
    assert find_threshold("KRDU", "99X") is None
    assert find_threshold("ZZZZ", "05L") is None


def test_threshold_target_uses_threshold_and_approach_envelope():
    thr = find_threshold("KRDU", "05L")
    t = threshold_target_state("KRDU", "05L", A320, mass_kg=66000.0)
    assert (t.latitude, t.longitude) == (thr["lat"], thr["lon"])
    # crosses the threshold at the coded crossing height above its elevation
    assert t.altitude == pytest.approx(thr["elevation_m"] + A320.approach.threshold_crossing_height_m)
    assert t.V == pytest.approx(A320.approach.reference_speed_ms)          # Vref
    # 05L is a compass bearing of 45deg; in math-ENU psi = 90 - 45 = 45deg (the diagonal is
    # the reflection's fixed point, so this case alone can't catch the convention — see below).
    assert t.psi == pytest.approx(math.radians(45.0))
    assert t.gamma == pytest.approx(math.radians(-A320.approach.glide_angle_deg))  # descending
    assert t.m == 66000.0


def test_threshold_target_psi_is_math_enu_not_compass():
    # A non-diagonal runway, where compass bearing != math-ENU heading, locks the convention.
    thr = find_threshold("KSTL", "30L")
    assert thr["heading_deg"] == 302.0                    # compass bearing
    t = threshold_target_state("KSTL", "30L", A320, mass_kg=66000.0)
    expected = math.radians(90.0 - 302.0)                 # math-ENU
    expected = (expected + math.pi) % (2.0 * math.pi) - math.pi  # wrap to [-pi, pi]
    assert t.psi == pytest.approx(expected)               # = radians(148), NOT radians(302)


def test_threshold_target_unknown_returns_none():
    assert threshold_target_state("KRDU", "99X", A320, mass_kg=66000.0) is None


def test_threshold_target_prefers_the_manifest_published_path_point():
    target = threshold_target_state(
        "KRDU",
        "05L",
        A320,
        mass_kg=66000.0,
        published_target={
            "lat": 1.0,
            "lon": 2.0,
            "elevation_msl_m": 100.0,
            "course_deg": 180.0,
            "threshold_crossing_height_m": 17.0,
            "published_glidepath_deg": 3.2,
        },
    )
    assert (target.latitude, target.longitude, target.altitude) == (1.0, 2.0, 117.0)
    assert target.psi == pytest.approx(math.radians(-90.0))
    assert target.gamma == pytest.approx(math.radians(-3.2))
