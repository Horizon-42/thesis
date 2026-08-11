"""geokit — canonical geodetic / coordinate / unit conversions for AeroViz-4D.

One source of truth for the constants and numeric helpers that were previously
re-derived across the data pipeline, optimizer/aero, and (via a generated JSON) the
frontend. See ``docs/coordinate-conversion-consolidation.md``.
"""

from __future__ import annotations

from .constants import (
    DEG2RAD,
    EARTH_RADIUS_MEAN_M,
    FT_M,
    FT_MIN_MS,
    KMH_MS,
    KT_MS,
    METRES_PER_DEG_LAT,
    MPH_MS,
    NM_M,
    RAD2DEG,
    SPHERE_RADIUS_M,
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    WGS84_E_PRIME2,
    WGS84_F,
)
from .geodesy import (
    bearing_rad,
    bounds_from_radius_km,
    compass_bearing_to_math_enu_rad,
    equirectangular_distance_m,
    flat_distance_m,
    haversine_km,
    haversine_m,
    metres_per_deg_lon,
    metres_per_degree_precise,
    wgs84_curvature_radii,
)
from .units import (
    ft_min_to_ms,
    ft_to_m,
    kmh_to_ms,
    kt_to_ms,
    m_to_ft,
    m_to_nm,
    mph_to_ms,
    ms_to_ft_min,
    ms_to_kmh,
    ms_to_kt,
    ms_to_mph,
    nm_to_m,
)

__all__ = [
    # constants
    "WGS84_A",
    "WGS84_B",
    "WGS84_F",
    "WGS84_E2",
    "WGS84_E_PRIME2",
    "EARTH_RADIUS_MEAN_M",
    "SPHERE_RADIUS_M",
    "METRES_PER_DEG_LAT",
    "NM_M",
    "FT_M",
    "KT_MS",
    "FT_MIN_MS",
    "KMH_MS",
    "MPH_MS",
    "DEG2RAD",
    "RAD2DEG",
    # geodesy
    "haversine_m",
    "haversine_km",
    "equirectangular_distance_m",
    "bearing_rad",
    "compass_bearing_to_math_enu_rad",
    "metres_per_deg_lon",
    "flat_distance_m",
    "metres_per_degree_precise",
    "wgs84_curvature_radii",
    "bounds_from_radius_km",
    # speed units
    "kt_to_ms",
    "ms_to_kt",
    "ft_min_to_ms",
    "ms_to_ft_min",
    "kmh_to_ms",
    "ms_to_kmh",
    "mph_to_ms",
    "ms_to_mph",
    # length units
    "nm_to_m",
    "m_to_nm",
    "ft_to_m",
    "m_to_ft",
]
