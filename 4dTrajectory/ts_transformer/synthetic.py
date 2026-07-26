"""Synthetic arrival tracks, for smoke-testing the pipeline without harvested ADS-B.

This exists so the whole path — build series -> train -> forecast -> evaluation records ->
`python -m evaluation` — can be exercised before any real data lands, and so the test suite
has fixtures that are geometrically meaningful rather than random noise.

It is **not** a data augmentation tool and must never be trained on as though it were real:
these tracks are piecewise-straight, noise-free in their intent, and contain none of the
things that make real terminal-airspace prediction hard (vectoring, holding, speed control,
wind, the runway-configuration multimodality the survey in ``4dTrajectory/docs`` calls the
central challenge). A model scoring well here has learned to extend a straight line.

Geometry per flight: an entry point on the 25 km ring at a random bearing offset from the
approach course, a straight leg to a 5 NM final approach fix, then a straight final along
the runway course, descending a 3-degree glidepath onto the threshold.
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import numpy as np

from flight_scenarios.runway_target import find_threshold

from coordinate_frames import ENUFrame

ENTRY_RADIUS_M = 25_000.0     # matches trajectory_data_process ENTRY_RADIUS_KM
FAF_DISTANCE_M = 5.0 * 1852.0  # 5 NM
GLIDE_ANGLE_DEG = 3.0
SAMPLE_INTERVAL_S = 1.0
# Height above the threshold at the 25 km ring. Chosen so the descent from the ring to the
# FAF averages ~3.5 degrees rather than the ~5.7 a 2200 m entry would force — steeper than
# anything an arrival actually flies, which would have made the vertical channel unlearnable
# for the wrong reason.
ENTRY_HEIGHT_M = 1600.0


def synthetic_arrivals(
    airport: str,
    runway: str,
    *,
    n_flights: int = 24,
    seed: int = 7,
    entry_offset_deg: float = 50.0,
    entry_speed_ms: float = 135.0,
    threshold_speed_ms: float = 72.0,
    jitter_m: float = 25.0,
) -> list[dict[str, Any]]:
    """``n_flights`` flight dicts in the czml-input shape, landing on ``airport``/``runway``.

    Raises if the threshold is not in ``trajectory_data_process/config/runway_thresholds.json``
    — a synthetic track anchored at a guessed runway would be quietly meaningless.
    """
    threshold = find_threshold(airport, runway)
    if threshold is None:
        raise ValueError(f"no published threshold for {airport}/{runway}")

    elevation_m = float(threshold["elevation_m"])
    heading_deg = float(threshold["heading_deg"])
    # The SAME ENU projection the data build uses, applied in reverse —
    # so synthetic waypoints and the training-side projection cannot drift apart.
    frame = ENUFrame(lat0=float(threshold["lat"]), lon0=float(threshold["lon"]),
                     alt0=elevation_m)

    # Direction of TRAVEL on final, in ENU. Compass heading H -> (east, north) = (sin H, cos H).
    course = math.radians(heading_deg)
    inbound = np.array([math.sin(course), math.cos(course)])

    rng = np.random.default_rng(seed)
    flights: list[dict[str, Any]] = []

    for i in range(n_flights):
        # Entry sits on the ring, offset from the reciprocal of the approach course.
        offset = math.radians(rng.uniform(-entry_offset_deg, entry_offset_deg))
        back = -inbound
        entry = ENTRY_RADIUS_M * np.array([
            back[0] * math.cos(offset) - back[1] * math.sin(offset),
            back[0] * math.sin(offset) + back[1] * math.cos(offset),
        ])
        faf = -FAF_DISTANCE_M * inbound

        legs = [(entry, faf), (faf, np.zeros(2))]
        speeds = (float(entry_speed_ms + rng.normal(0.0, 6.0)),
                  float(threshold_speed_ms + rng.normal(0.0, 3.0)))

        points: list[tuple[float, float, float, float]] = []  # (t, e, n, u)
        t = 0.0
        for leg_index, (start, end) in enumerate(legs):
            length = float(np.linalg.norm(end - start))
            direction = (end - start) / length
            v_start = speeds[0] if leg_index == 0 else speeds[0] * 0.75 + speeds[1] * 0.25
            v_end = speeds[0] * 0.75 + speeds[1] * 0.25 if leg_index == 0 else speeds[1]

            travelled = 0.0
            while travelled < length:
                fraction = travelled / length
                speed = v_start + (v_end - v_start) * fraction
                position = start + direction * travelled
                distance_to_threshold = float(np.linalg.norm(position))
                # On the glidepath inside the FAF; a shallower cruise-descent outside it.
                if distance_to_threshold <= FAF_DISTANCE_M:
                    height = distance_to_threshold * math.tan(math.radians(GLIDE_ANGLE_DEG))
                else:
                    faf_height = FAF_DISTANCE_M * math.tan(math.radians(GLIDE_ANGLE_DEG))
                    beyond = (distance_to_threshold - FAF_DISTANCE_M) / (ENTRY_RADIUS_M - FAF_DISTANCE_M)
                    height = faf_height + beyond * (ENTRY_HEIGHT_M - faf_height)
                points.append((t, float(position[0]), float(position[1]), height))
                t += SAMPLE_INTERVAL_S
                travelled += speed * SAMPLE_INTERVAL_S
        points.append((t, 0.0, 0.0, 0.0))

        noise = rng.normal(0.0, jitter_m, size=(len(points), 3))
        waypoints = []
        for k, pt in enumerate(points):
            lat, lon = frame.latlon_from_horizontal(
                pt[1] + noise[k, 0], pt[2] + noise[k, 1]
            )
            # Waypoint order is [t, lon, lat, alt] — lon BEFORE lat (the czml-input trap
            # channels.py documents).
            waypoints.append([
                float(pt[0]), round(lon, 6), round(lat, 6),
                round(elevation_m + pt[3] + noise[k, 2] * 0.4, 1),
            ])

        # Ids and transponder addresses are made unique PER RUNWAY. Generating "SYN000" for
        # every runway would collide across a multi-runway set, and the split key is built
        # from (id, runway, icao24, landing time) — colliding synthetic flights would hide
        # exactly the leak that key exists to prevent. crc32, NOT the builtin hash(): that
        # one is salted per process (PYTHONHASHSEED), which would give the "same" seeded
        # flights a different identity in every interpreter — training in one process and
        # `predict --split test` on regenerated data in another would intersect on zero
        # flight ids.
        tag = f"{runway}{i:03d}"
        flights.append({
            "id": f"SYN{tag}",
            "callsign": f"SYN{tag}",
            "type": "A320",
            "icao24": f"{(0xC00000 + i + 977 * (zlib.crc32(runway.encode()) % 89)) & 0xFFFFFF:06x}",
            "dep_airport": None,
            "arr_airport": airport,
            "runway": runway,
            "landing_time_utc": f"2026-07-{1 + i % 28:02d}T{i % 24:02d}:00:00Z",
            "altitude_source": "synthetic",
            "waypoints": waypoints,
        })

    return flights
