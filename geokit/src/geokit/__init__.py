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
    KT_MS,
    METRES_PER_DEG_LAT,
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
    equirectangular_distance_m,
    flat_distance_m,
    haversine_km,
    haversine_m,
    metres_per_deg_lon,
    metres_per_degree_precise,
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
    "DEG2RAD",
    "RAD2DEG",
    # geodesy
    "haversine_m",
    "haversine_km",
    "equirectangular_distance_m",
    "bearing_rad",
    "metres_per_deg_lon",
    "flat_distance_m",
    "metres_per_degree_precise",
    "bounds_from_radius_km",
]
