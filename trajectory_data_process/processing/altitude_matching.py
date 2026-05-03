"""Altitude matching helpers for OpenSky-derived training records.

OpenSky `/tracks/all` contains barometric altitude only. This module joins
geometric altitude from `/states/all` state vectors into a derived point set
without modifying the raw track payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateAltitudeSample:
    """A state-vector altitude sample with source metadata."""

    icao24: str
    time: float
    baro_altitude_m: float | None
    geo_altitude_m: float
    source_response_id: str | None = None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_state_altitude_samples(
    payload: dict[str, Any],
    *,
    source_response_id: str | None = None,
) -> list[StateAltitudeSample]:
    """Extract valid geometric-altitude samples from an OpenSky states payload."""
    samples: list[StateAltitudeSample] = []
    for row in payload.get("states") or []:
        if not row or len(row) <= 13:
            continue
        icao24 = str(row[0] or "").lower().strip()
        if not icao24:
            continue

        # State vector indexes follow OpenSky REST docs:
        # 3=time_position, 4=last_contact, 7=baro_altitude, 13=geo_altitude.
        sample_time = _finite_float(row[3])
        if sample_time is None:
            sample_time = _finite_float(row[4])
        geo_altitude = _finite_float(row[13])
        if sample_time is None or geo_altitude is None:
            continue

        samples.append(
            StateAltitudeSample(
                icao24=icao24,
                time=sample_time,
                baro_altitude_m=_finite_float(row[7]),
                geo_altitude_m=geo_altitude,
                source_response_id=source_response_id,
            )
        )
    return samples


def _nearest_sample(
    *,
    icao24: str,
    waypoint_time: float,
    samples: list[StateAltitudeSample],
    max_age_sec: float,
) -> StateAltitudeSample | None:
    best: StateAltitudeSample | None = None
    best_delta: float | None = None
    for sample in samples:
        if sample.icao24 != icao24:
            continue
        delta = abs(sample.time - waypoint_time)
        if delta > max_age_sec:
            continue
        if best_delta is None or delta < best_delta:
            best = sample
            best_delta = delta
    return best


def build_dual_altitude_points(
    track: dict[str, Any],
    samples: list[StateAltitudeSample],
    *,
    max_age_sec: float = 15.0,
    require_geo_altitude: bool = True,
) -> dict[str, Any]:
    """Build derived points with both barometric and geometric altitude.

    Raw `track.path` is only read. Missing `geo_altitude_m` is counted as
    quarantined when geometric altitude is required.
    """
    icao24 = str(track.get("icao24") or "").lower().strip()
    points: list[dict[str, Any]] = []
    stats = {
        "points_total": 0,
        "points_with_baro_altitude": 0,
        "points_with_geo_altitude": 0,
        "points_with_both_altitudes": 0,
        "points_quarantined_missing_geo_altitude": 0,
    }

    for raw_index, wp in enumerate(track.get("path") or []):
        if not wp or len(wp) < 6:
            continue
        t, lat, lon, baro_altitude_m, true_track_deg, on_ground = wp[:6]
        time_value = _finite_float(t)
        lat_value = _finite_float(lat)
        lon_value = _finite_float(lon)
        baro_value = _finite_float(baro_altitude_m)
        if time_value is None or lat_value is None or lon_value is None:
            continue

        stats["points_total"] += 1
        if baro_value is not None:
            stats["points_with_baro_altitude"] += 1

        sample = _nearest_sample(
            icao24=icao24,
            waypoint_time=time_value,
            samples=samples,
            max_age_sec=max_age_sec,
        )
        if sample is not None:
            stats["points_with_geo_altitude"] += 1

        if baro_value is None or sample is None:
            if sample is None and require_geo_altitude:
                stats["points_quarantined_missing_geo_altitude"] += 1
            if require_geo_altitude:
                continue

        if baro_value is not None and sample is not None:
            stats["points_with_both_altitudes"] += 1

        point = {
            "raw_index": raw_index,
            "time": time_value,
            "lat": lat_value,
            "lon": lon_value,
            "baro_altitude_m": baro_value,
            "geo_altitude_m": None if sample is None else sample.geo_altitude_m,
            "true_track_deg": _finite_float(true_track_deg),
            "on_ground": bool(on_ground),
            "altitude_sources": {
                "baro_altitude_m": "opensky_tracks_all_baro_altitude_m",
                "geo_altitude_m": None if sample is None else "opensky_states_all_geo_altitude_m",
            },
            "geo_altitude_match": None
            if sample is None
            else {
                "method": "nearest_state_vector",
                "source_time": sample.time,
                "delta_t_sec": abs(sample.time - time_value),
                "source_response_id": sample.source_response_id,
            },
        }
        points.append(point)

    return {"points": points, "quality": stats}

