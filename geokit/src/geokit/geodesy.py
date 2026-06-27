"""Numeric geodesy helpers (plain Python / stdlib ``math``).

Canonical argument order is ``(lat, lon, ...)`` with degrees in / metres out, unless a
name says otherwise. CasADi-symbolic equivalents live in
``aerodynamic_model/casadi_coordinates_converter.py`` (that module imports the constants
from here so the *values* stay unified — see the consolidation doc).
"""

from __future__ import annotations

import math

from .constants import (
    METRES_PER_DEG_LAT,
    SPHERE_RADIUS_M,
)


# ── Great-circle / spherical distance ────────────────────────────────────────

def haversine_m(
    lat1: float, lon1: float, lat2: float, lon2: float, radius_m: float = SPHERE_RADIUS_M
) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(a))


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float, radius_m: float = SPHERE_RADIUS_M
) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    return haversine_m(lat1, lon1, lat2, lon2, radius_m) / 1000.0


def equirectangular_distance_m(
    lat1: float, lon1: float, lat2: float, lon2: float, radius_m: float = SPHERE_RADIUS_M
) -> float:
    """Small-angle (equirectangular) horizontal distance, metres.

    Cheaper than haversine and accurate over short spans; used where one point is a small
    offset from the other (e.g. per-sample trajectory deviation).
    """
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    return radius_m * math.hypot(dphi, dlmb * math.cos(math.radians(lat2)))


# ── Bearing ──────────────────────────────────────────────────────────────────

def bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2.

    Returns radians, 0 = North, positive = clockwise (toward East).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlmb)
    return math.atan2(y, x)


# ── Flat-Earth (local-tangent) helpers ───────────────────────────────────────

def metres_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude at the given latitude."""
    return METRES_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def flat_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate ground distance for short segments (< ~20 km), metres.

    Flat-Earth: longitude scaled by ``cos(mean_lat)``. Faster than haversine and fine for
    runway / procedure geometry; use :func:`haversine_m` when accuracy over long spans matters.
    """
    mid_lat = 0.5 * (lat1 + lat2)
    dx = (lon2 - lon1) * metres_per_deg_lon(mid_lat)
    dy = (lat2 - lat1) * METRES_PER_DEG_LAT
    return math.hypot(dx, dy)


def metres_per_degree_precise(lat_deg: float) -> tuple[float, float]:
    """Precise (lon, lat) metres-per-degree from the WGS84 series expansion.

    More accurate than the flat ``METRES_PER_DEG_LAT`` constant; reserved for terrain
    projection where the latitude dependence matters. Returns ``(m_per_deg_lon, m_per_deg_lat)``.
    """
    lat = math.radians(lat_deg)
    m_lat = 111_132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat) - 0.0023 * math.cos(6 * lat)
    m_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)
    return (m_lon, m_lat)


# ── Bounding box ─────────────────────────────────────────────────────────────

def bounds_from_radius_km(
    lat: float, lon: float, radius_km: float
) -> tuple[float, float, float, float]:
    """Bounding box of ±``radius_km`` around a point: (west, south, east, north).

    Uses the unified :data:`METRES_PER_DEG_LAT` for latitude and a cos-scaled value for
    longitude, with a pole guard so ``cos(lat) -> 0`` does not blow up.
    """
    km_per_deg_lat = METRES_PER_DEG_LAT / 1000.0
    dlat = radius_km / km_per_deg_lat
    cos_lat = math.cos(math.radians(lat))
    dlon = 180.0 if abs(cos_lat) < 1e-6 else radius_km / (km_per_deg_lat * cos_lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
