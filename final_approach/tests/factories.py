"""Geometry factories shared by final-approach tests."""

from __future__ import annotations

import math

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from final_approach import RunwayFrame, TrackPoint


FRAME = RunwayFrame(
    ident="05L",
    lat=35.8745,
    lon=-78.802,
    elevation_m=111.86,
    course_deg=45.0,
)
ALTITUDE_QUANTUM_M = 25.0 * 0.3048


def synthetic_approach(
    *,
    glidepath_deg: float = 3.0,
    tch_m: float = 17.5,
    cross_m: float = 0.0,
    start_along_m: float = -12_000.0,
    end_along_m: float = -325.0,
    step_m: float = 77.0,
    quantise: bool = False,
    frame: RunwayFrame = FRAME,
) -> list[TrackPoint]:
    """A straight approach sampled from far to near in time order."""
    slope = math.tan(math.radians(glidepath_deg))
    metres_per_lon_degree = metres_per_deg_lon(frame.lat)
    east_hat = math.sin(math.radians(frame.course_deg))
    north_hat = math.cos(math.radians(frame.course_deg))
    points = []
    along_m = start_along_m
    while along_m <= end_along_m:
        height_m = tch_m - along_m * slope
        if quantise:
            height_m = (
                round((height_m + frame.elevation_m) / ALTITUDE_QUANTUM_M)
                * ALTITUDE_QUANTUM_M
                - frame.elevation_m
            )
        east_m = along_m * east_hat + cross_m * north_hat
        north_m = along_m * north_hat - cross_m * east_hat
        points.append(
            TrackPoint(
                lat=frame.lat + north_m / METRES_PER_DEG_LAT,
                lon=frame.lon + east_m / metres_per_lon_degree,
                alt_m=frame.elevation_m + height_m,
            )
        )
        along_m += step_m
    return points
