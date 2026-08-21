"""Truncate landing tracks to their ARRIVAL segment; classify local circuits.

Harvested landing tracks are validated by their END (closest point at the runway
threshold + runway heading), so a flight that DEPARTED the same airport and came
back — or a local GA/pattern circuit that never left the terminal area — is kept
WHOLE, starting at its takeoff. Those are not TMA arrivals: as optimizer
scenarios they put the start on the field next to the target, and for
multi-aircraft interaction studies the meaningful boundary is the ENTRY into the
terminal area.

Rule (walking BACKWARD from touchdown): the arrival segment starts at the first
sample after the LAST run of at least ``hysteresis`` consecutive samples OUTSIDE
the ``entry_radius_km`` ring around the airport — i.e. the flight's final entry
into the ring. The hysteresis means a single jittery fix beyond the ring cannot
cut a track. A track that NEVER leaves the ring is a LOCAL circuit, not an
arrival: it is excluded from the arrivals dataset (kept in a review side file,
never silently dropped).

The local-circuit test is DISTANCE from the destination field, so it only sees a
flight that took off where it landed. A takeoff from a NEIGHBOURING field inside
the ring passes it — and lands in the dataset as a "coverage-limited arrival"
whose first sample is on a runway a few kilometres away. That is what the
altitude test below catches; the two together are the one question "does this
segment contain a takeoff", asked from the two directions a track can answer it.

Waypoint rows are the CZML-input shape ``[t_offset_s, lon_deg, lat_deg, alt_m]``;
the truncated segment's times are REBASED to 0 so every downstream consumer
(scenario building, reference records, CZML rendering, evaluation baselines)
sees one canonical arrival segment. ``entry_time_utc`` (the absolute time the
flight crossed into the ring — the boundary condition multi-aircraft interaction
studies need) is derived from ``landing_time_utc`` minus the segment duration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from trajectory_data_process.geo import haversine_km

# The terminal-entry ring. Must sit INSIDE the harvest crop radius (30 km), or
# no track could ever be "outside" it. 25 km leaves a 5 km buffer.
ENTRY_RADIUS_KM = 25.0
# Consecutive samples beyond the ring required to count as "outside" (GPS jitter
# on a single fix must not cut a track).
ENTRY_HYSTERESIS_SAMPLES = 3
# A never-outside track is a LOCAL circuit only if it STARTS at the field (a
# takeoff); one starting farther out but inside the ring is a coverage-limited
# arrival (e.g. an ADS-B gap ate its pre-ring stretch) and must be kept.
LOCAL_START_RADIUS_KM = 5.0
# A segment whose FIRST sample is this close to the landing runway's elevation
# began on a runway: it contains a takeoff and is not a terminal-area arrival.
# Measured over the 42 725 rostered arrivals of the five-airport fleet, the
# first-sample height above the landing runway is bimodal with an EMPTY band
# either side of this value: 75 flights at or below 82.1 m (every one of them within
# 2 km of a satellite field — KRHV/KPAO/KNUQ at KSJC, KSAC/KMCC at KSMF, KNEW at
# KMSY) and then nothing until 175.3 m, an airborne arrival 11 km from the
# nearest field. Altitude alone decides it: reported ground speed does NOT
# separate the two populations, because a jet at rotation still reads 71–80 m/s
# while on the runway, squarely inside the approach-speed range.
GROUND_START_AGL_M = 100.0


def arrival_segment(
    waypoints: list[list[float]],
    airport_lat: float,
    airport_lon: float,
    *,
    field_elevation_m: float,
    entry_radius_km: float = ENTRY_RADIUS_KM,
    hysteresis: int = ENTRY_HYSTERESIS_SAMPLES,
) -> tuple[str, list[list[float]]]:
    """Classify one track and cut it to its arrival segment.

    ``field_elevation_m`` is the landing runway's elevation in the SAME vertical
    datum as the waypoint altitudes (the harvest stores both as HAE). It is
    required and has no default: a wrong datum here shifts the ground test by the
    geoid separation (~33 m over the US) without failing, and the caller is the
    only place that knows which datum its rows carry.

    Returns ``(kind, segment)``:
      * ``("arrival", rows)`` — the final-entry→touchdown rows, times rebased to 0.
        This applies to PLAIN arrivals too (their pre-ring stretch, at most the
        5 km harvest-crop annulus, is dropped), so every arrival in the dataset
        shares ONE entry boundary — the state distribution ON the ring is exactly
        the boundary condition interaction studies need.
      * ``("local", waypoints)`` — the track never left the ring (original rows).
      * ``("takeoff", waypoints)`` — the segment that survived the cut starts on
        the ground, so it contains a takeoff from a field inside the ring
        (original rows, for the review side file).

    The ground test is applied to the segment the cut PRODUCED, not to the raw
    track: a flight that departs a neighbouring field, leaves the ring and comes
    back is a genuine arrival whose takeoff was already cut away.
    """
    kind, segment = _entry_cut(
        waypoints, airport_lat, airport_lon,
        entry_radius_km=entry_radius_km, hysteresis=hysteresis,
    )
    if kind == "arrival" and segment[0][3] - field_elevation_m <= GROUND_START_AGL_M:
        return "takeoff", waypoints
    return kind, segment


def _entry_cut(
    waypoints: list[list[float]],
    airport_lat: float,
    airport_lon: float,
    *,
    entry_radius_km: float,
    hysteresis: int,
) -> tuple[str, list[list[float]]]:
    """The ring geometry alone: ``("arrival", cut rows)`` or ``("local", rows)``."""
    outside = [
        haversine_km(w[2], w[1], airport_lat, airport_lon) > entry_radius_km
        for w in waypoints
    ]
    run = 0
    for i in range(len(waypoints) - 1, -1, -1):
        if outside[i]:
            run += 1
            if run >= hysteresis:
                # The run spans [i, i + run - 1]; the arrival begins at the first
                # sample after it (the final crossing INTO the ring).
                return "arrival", _rebased(waypoints[i + run:])
        else:
            run = 0
    if outside[0] and run >= 1:
        # The track starts with a SHORT (< hysteresis) outside stretch — within the
        # jitter tolerance, so nothing is cut: still an arrival, kept whole.
        return "arrival", _rebased(waypoints)
    start_dist_km = haversine_km(waypoints[0][2], waypoints[0][1], airport_lat, airport_lon)
    if start_dist_km <= LOCAL_START_RADIUS_KM:
        # Never left the ring AND started at the field: a takeoff→circuit→landing.
        return "local", waypoints
    # Never left the ring but started well away from the field: a coverage-limited
    # arrival — keep it whole.
    return "arrival", _rebased(waypoints)


def _rebased(rows: list[list[float]]) -> list[list[float]]:
    t0 = rows[0][0]
    return [[r[0] - t0, r[1], r[2], r[3]] for r in rows]


def truncate_flights(
    flights: list[dict[str, Any]],
    airport_lat: float,
    airport_lon: float,
    *,
    field_elevation_m: float,
    entry_radius_km: float = ENTRY_RADIUS_KM,
    hysteresis: int = ENTRY_HYSTERESIS_SAMPLES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split CZML-input flights into ``(arrivals, locals, takeoffs)``.

    Arrivals get their waypoints replaced by the (rebased) arrival segment plus
    provenance fields: ``arrival_truncated`` (whether anything was cut),
    ``cut_samples`` (how many), ``arrival_duration_s`` and ``entry_time_utc``.
    Locals and takeoffs are returned unmodified for the review side file.
    """
    arrivals: list[dict[str, Any]] = []
    locals_: list[dict[str, Any]] = []
    takeoffs: list[dict[str, Any]] = []
    buckets = {"local": locals_, "takeoff": takeoffs}
    for flight in flights:
        waypoints = flight["waypoints"]
        kind, segment = arrival_segment(
            waypoints, airport_lat, airport_lon,
            field_elevation_m=field_elevation_m,
            entry_radius_km=entry_radius_km, hysteresis=hysteresis,
        )
        if kind != "arrival":
            buckets[kind].append(flight)
            continue
        out = dict(flight)
        out["waypoints"] = segment
        out["arrival_truncated"] = len(segment) < len(waypoints)
        out["cut_samples"] = len(waypoints) - len(segment)
        duration_s = float(segment[-1][0])
        out["arrival_duration_s"] = duration_s
        out["entry_time_utc"] = _entry_time_utc(flight.get("landing_time_utc"), duration_s)
        arrivals.append(out)
    return arrivals, locals_, takeoffs


def _entry_time_utc(landing_time_utc: str | None, duration_s: float) -> str | None:
    """Absolute ring-entry time = landing time − segment duration (None if unknown)."""
    if not landing_time_utc:
        return None
    landing = datetime.fromisoformat(landing_time_utc.replace("Z", "+00:00"))
    entry = landing - timedelta(seconds=duration_s)
    return entry.strftime("%Y-%m-%dT%H:%M:%SZ")
