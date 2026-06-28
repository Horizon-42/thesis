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
    assert t.psi == pytest.approx(math.radians(45.0))                       # runway heading
    assert t.gamma == pytest.approx(math.radians(-A320.approach.glide_angle_deg))  # descending
    assert t.m == 66000.0


def test_threshold_target_unknown_returns_none():
    assert threshold_target_state("KRDU", "99X", A320, mass_kg=66000.0) is None
