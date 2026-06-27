"""Tests for geokit speed unit conversions."""

import pytest

import geokit
from geokit import units as U


def test_kt_to_ms_exact():
    # 1 knot = 1 nm/h = 1852 m / 3600 s
    assert U.kt_to_ms(1.0) == pytest.approx(1852.0 / 3600.0)
    assert U.kt_to_ms(100.0) == pytest.approx(51.44444444)


def test_ft_min_to_ms_exact():
    # 1 ft/min = 0.3048 m / 60 s
    assert U.ft_min_to_ms(1.0) == pytest.approx(0.3048 / 60.0)
    assert U.ft_min_to_ms(1000.0) == pytest.approx(5.08)


def test_kmh_and_mph():
    assert U.kmh_to_ms(3.6) == pytest.approx(1.0)
    assert U.mph_to_ms(1.0) == pytest.approx(0.44704)


def test_length_conversions():
    assert U.nm_to_m(1.0) == pytest.approx(1852.0)
    assert U.ft_to_m(1.0) == pytest.approx(0.3048)
    assert U.m_to_nm(1852.0) == pytest.approx(1.0)
    assert U.m_to_ft(0.3048) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "forward, backward",
    [
        (U.kt_to_ms, U.ms_to_kt),
        (U.ft_min_to_ms, U.ms_to_ft_min),
        (U.kmh_to_ms, U.ms_to_kmh),
        (U.mph_to_ms, U.ms_to_mph),
        (U.nm_to_m, U.m_to_nm),
        (U.ft_to_m, U.m_to_ft),
    ],
)
def test_round_trips(forward, backward):
    for value in (0.0, 1.0, 137.5, 9000.0):
        assert backward(forward(value)) == pytest.approx(value)


def test_public_exports():
    for name in ("kt_to_ms", "ms_to_kt", "ft_min_to_ms", "nm_to_m", "ft_to_m", "KT_MS", "FT_MIN_MS"):
        assert hasattr(geokit, name)
