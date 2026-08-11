"""Build a target :class:`GeodeticState` from a published runway threshold.

An alternative to deriving the target from the end of the observed track (which, near the
ground, is noisy / at touchdown / sometimes corrupt): the **runway threshold** is a clean,
well-defined approach endpoint — the threshold position at the threshold-crossing height, on
the runway heading, at the aircraft's reference approach speed and glidepath descent.

Threshold geometry comes from ``trajectory_data_process/config/runway_thresholds.json``
(``airports[CODE].runways[].thresholds[] = {ident, lat, lon, elevation_m, heading_deg}``).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from aerodynamic_model.common import GeodeticState
from aircraft.aircraft_sets import Aircraft
from geokit import compass_bearing_to_math_enu_rad

_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "trajectory_data_process" / "config" / "runway_thresholds.json"
)


@lru_cache(maxsize=None)
def _load_airports(path: Path = _THRESHOLDS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["airports"]


def find_threshold(
    airport: str | None, runway: str | None, *, path: Path = _THRESHOLDS_PATH
) -> dict[str, Any] | None:
    """The threshold record ``{ident, lat, lon, elevation_m, heading_deg}`` for
    ``(airport, runway)``, or ``None`` if it isn't in the config."""
    airports = _load_airports(path)
    record = airports.get((airport or "").upper())
    if record is None:
        return None
    want = (runway or "").upper()
    for runway_pair in record.get("runways", []):
        for threshold in runway_pair.get("thresholds", []):
            if str(threshold.get("ident", "")).upper() == want:
                return threshold
    return None


def threshold_target_state(
    airport: str | None,
    runway: str | None,
    aircraft: Aircraft,
    *,
    mass_kg: float,
    published_target: dict[str, Any] | None = None,
) -> GeodeticState | None:
    """A target state at the runway threshold, or ``None`` if the threshold is unknown.

    Position = threshold lat/lon at ``elevation + threshold_crossing_height``; ``V`` = the
    aircraft's reference approach speed (Vref); ``psi`` = the runway heading (in the model's
    math-ENU convention); ``gamma`` = the coded glidepath descent (negative).
    """
    if published_target is not None:
        tch = published_target.get("threshold_crossing_height_m")
        glidepath = published_target.get("published_glidepath_deg")
        if tch is None or glidepath is None:
            return None
        lat = float(published_target["lat"])
        lon = float(published_target["lon"])
        altitude = float(published_target["elevation_msl_m"]) + float(tch)
        course_deg = float(published_target["course_deg"])
        glidepath_deg = float(glidepath)
    else:
        # Kept for synthetic/in-memory scenarios. Canonical harvested arrivals always
        # carry ``runway_target`` decoded from the CIFP.
        threshold = find_threshold(airport, runway)
        if threshold is None:
            return None
        lat = float(threshold["lat"])
        lon = float(threshold["lon"])
        altitude = (
            float(threshold["elevation_m"])
            + aircraft.approach.threshold_crossing_height_m
        )
        course_deg = float(threshold["heading_deg"])
        glidepath_deg = aircraft.approach.glide_angle_deg
    # ``heading_deg`` is a compass bearing (0 = N, clockwise); the modeling layer's psi is the
    # math-ENU heading (0 = E, CCW toward N), so psi = 90deg - bearing (then wrap to [-pi, pi]).
    psi = compass_bearing_to_math_enu_rad(math.radians(course_deg))
    return GeodeticState(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        V=aircraft.approach.reference_speed_ms,
        psi=psi,
        gamma=math.radians(-glidepath_deg),
        m=mass_kg,
    )
