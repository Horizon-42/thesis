"""Great-circle distance helpers shared across processing stages."""

from __future__ import annotations

import math


EARTH_RADIUS_KM = 6371.0
KM_PER_NM = 1.852


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    return haversine_km(lat1, lon1, lat2, lon2) / KM_PER_NM


def bounds_from_radius_km(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Bounding box of +/- ``radius_km`` around a point, as west, south, east, north."""
    dlat = radius_km / 110.574
    dlon = radius_km / (111.320 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
