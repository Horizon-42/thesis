"""Great-circle distance helpers shared across processing stages.

Thin re-export of the canonical implementations in the ``geokit`` package, kept here so
existing imports (``trajectory_data_process.geo``) and the km-based public API keep
working. The actual math/constants live in one place now — see
``docs/coordinate-conversion-consolidation.md``.
"""

from __future__ import annotations

from geokit import NM_M, bounds_from_radius_km, haversine_km

__all__ = ["haversine_km", "distance_nm", "bounds_from_radius_km"]


def distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    return haversine_km(lat1, lon1, lat2, lon2) / (NM_M / 1000.0)
