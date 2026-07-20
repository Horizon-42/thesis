"""Tests for geokit numeric helpers + constant sanity."""

import math

import pytest

import geokit
from geokit import constants as C
from geokit import geodesy as G


# ── Constants ────────────────────────────────────────────────────────────────

def test_wgs84_constants():
    assert C.WGS84_A == 6_378_137.0
    # e^2 derived from the defining flattening
    assert C.WGS84_E2 == pytest.approx(0.0066943799901, rel=0, abs=1e-12)
    # mean radius literal matches (2a + b)/3 to <1 m
    assert C.EARTH_RADIUS_MEAN_M == pytest.approx((2 * C.WGS84_A + C.WGS84_B) / 3.0, abs=1.0)


def test_default_sphere_radius_is_wgs84_a():
    assert C.SPHERE_RADIUS_M == C.WGS84_A


def test_unit_constants():
    assert C.NM_M == 1852.0
    assert C.FT_M == 0.3048
    assert C.KT_MS == pytest.approx(1852.0 / 3600.0)


# ── Haversine ────────────────────────────────────────────────────────────────

def test_haversine_zero_distance():
    assert G.haversine_m(35.0, -78.0, 35.0, -78.0) == pytest.approx(0.0)


def test_haversine_one_degree_latitude():
    # One degree of latitude ~= pi/180 * R; default radius = WGS84 a.
    d = G.haversine_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(math.radians(1.0) * C.WGS84_A, rel=1e-9)


def test_haversine_km_is_metres_over_1000():
    assert G.haversine_km(35.0, -78.0, 36.0, -79.0) == pytest.approx(
        G.haversine_m(35.0, -78.0, 36.0, -79.0) / 1000.0
    )


def test_haversine_radius_override_switches_value():
    a = G.haversine_m(35.0, -78.0, 36.0, -79.0)
    mean = G.haversine_m(35.0, -78.0, 36.0, -79.0, radius_m=C.EARTH_RADIUS_MEAN_M)
    assert mean < a  # mean radius < WGS84 a
    assert mean / a == pytest.approx(C.EARTH_RADIUS_MEAN_M / C.WGS84_A, rel=1e-12)


def test_equirectangular_matches_haversine_short_span():
    # Over a short span the small-angle distance ~= haversine.
    h = G.haversine_m(35.0, -78.0, 35.01, -78.01)
    e = G.equirectangular_distance_m(35.0, -78.0, 35.01, -78.01)
    assert e == pytest.approx(h, rel=1e-4)


# ── Bearing ──────────────────────────────────────────────────────────────────

def test_bearing_due_north():
    assert G.bearing_rad(35.0, -78.0, 36.0, -78.0) == pytest.approx(0.0, abs=1e-9)


def test_bearing_due_east():
    assert G.bearing_rad(0.0, 0.0, 0.0, 1.0) == pytest.approx(math.pi / 2.0, abs=1e-9)


# ── Flat-Earth helpers ───────────────────────────────────────────────────────

def test_metres_per_deg_lon_shrinks_with_latitude():
    assert G.metres_per_deg_lon(0.0) == pytest.approx(C.METRES_PER_DEG_LAT)
    assert G.metres_per_deg_lon(60.0) == pytest.approx(C.METRES_PER_DEG_LAT * 0.5)


def test_flat_distance_matches_haversine_short_span():
    h = G.haversine_m(35.0, -78.0, 35.02, -78.0)
    f = G.flat_distance_m(35.0, -78.0, 35.02, -78.0)
    assert f == pytest.approx(h, rel=2e-3)


def test_metres_per_degree_precise_is_near_constant():
    lon_m, lat_m = G.metres_per_degree_precise(35.0)
    assert lat_m == pytest.approx(110_941.0, abs=200.0)
    assert lon_m == pytest.approx(91_290.0, abs=400.0)


def test_wgs84_curvature_radii_pinned_at_the_closed_form_landmarks():
    # Equator: R_M = a(1 - e^2) (the ellipse's tightest meridional curvature),
    # R_N = a exactly. Pole: both equal a / sqrt(1 - e^2). Values are the exact
    # closed forms evaluated independently — a formula regression (e.g. swapping
    # the exponents, or the two radii) moves them by kilometres.
    r_m, r_n = G.wgs84_curvature_radii(0.0)
    assert r_m == pytest.approx(6_335_439.327, abs=1e-3)
    assert r_n == pytest.approx(C.WGS84_A, abs=1e-9)

    r_m, r_n = G.wgs84_curvature_radii(90.0)
    assert r_m == pytest.approx(6_399_593.626, abs=1e-3)
    assert r_n == pytest.approx(r_m, abs=1e-6)

    # Mid-latitude spot check (45 deg), and the ordering R_M < R_N off the poles.
    r_m, r_n = G.wgs84_curvature_radii(45.0)
    assert r_m == pytest.approx(6_367_381.816, abs=1e-3)
    assert r_n == pytest.approx(6_388_838.290, abs=1e-3)
    assert r_m < r_n


# ── Bounding box ─────────────────────────────────────────────────────────────

def test_bounds_from_radius_basic():
    west, south, east, north = G.bounds_from_radius_km(35.0, -78.0, 30.0)
    assert west < -78.0 < east
    assert south < 35.0 < north
    # latitude half-span ~= 30 km / 111.32 km/deg
    assert (north - 35.0) == pytest.approx(30.0 / (C.METRES_PER_DEG_LAT / 1000.0), rel=1e-9)


def test_bounds_pole_guard():
    west, south, east, north = G.bounds_from_radius_km(90.0, 0.0, 30.0)
    assert east - west == pytest.approx(360.0)  # dlon clamped to 180


# ── Public surface ───────────────────────────────────────────────────────────

def test_public_exports():
    for name in ("haversine_m", "bearing_rad", "WGS84_A", "SPHERE_RADIUS_M", "NM_M"):
        assert hasattr(geokit, name)
