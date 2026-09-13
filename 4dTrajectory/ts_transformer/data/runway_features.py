"""The approach geometry runway-intent R0 and R1 share, R1's per-anchor feature vector, and
R1.1's candidate-symmetric rows built from it (`candidate_rows`).

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §12. CAUSAL by construction: a feature at anchor
``i`` reads the flight's waypoints ``[0, i]`` and the airport context strictly before the anchor's
wall-clock time (`data.runway_context`). The flown path LEFT decides which samples are anchors —
an evaluation grid — and is never a feature value: it is the future.

Waypoints are the arrival slice's ``[t_since_entry_s, lon, lat, alt_m]`` rows.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from ts_transformer.data.runway_context import ENTRY_SECTORS, RunwayContext, course_deg

ENTRY = "entry"
#: The own-track trend feature: the displacement over this much of the past.
HISTORY_S = 60.0
#: Callsign prefixes kept as their own column, per airport; the rest share one "other" column.
TOP_AIRLINES = 15
#: A "time since" feature is capped here (minutes) and read as "none in the window" beyond it.
SINCE_CAP_MIN = 180.0
CONTEXT_WINDOWS_MIN = (30.0, 60.0)


def remaining_path_m(waypoints: Sequence[Sequence[float]]) -> list[float]:
    """Flown horizontal path from each sample to the last one (the landing), metres."""
    remaining = [0.0] * len(waypoints)
    for i in range(len(waypoints) - 2, -1, -1):
        _, lon0, lat0, _ = waypoints[i]
        _, lon1, lat1, _ = waypoints[i + 1]
        step = math.hypot(
            (lon1 - lon0) * metres_per_deg_lon(0.5 * (lat0 + lat1)),
            (lat1 - lat0) * METRES_PER_DEG_LAT,
        )
        remaining[i] = remaining[i + 1] + step
    return remaining


def anchors(waypoints: Sequence[Sequence[float]], bins_km: Iterable[float]) -> dict[str, int]:
    """``entry`` -> 0, and ``<D>km`` -> the first sample with at most D km left, for every D
    the flight's slice actually starts beyond (a flight entering with 12 km left has no 15 km
    anchor — it is counted out of that bin, and the bin's coverage says so)."""
    remaining = remaining_path_m(waypoints)
    out = {ENTRY: 0}
    for distance_km in bins_km:
        limit = distance_km * 1000.0
        if remaining[0] <= limit:
            continue
        out[f"{distance_km:g}km"] = next(i for i, r in enumerate(remaining) if r <= limit)
    return out


def ring_anchors(
    waypoints: Sequence[Sequence[float]], reference: tuple[float, float], radii_km: Iterable[float]
) -> dict[str, int]:
    """``entry`` -> 0, and ``r<R>km`` -> the FIRST sample within R km of the airport reference —
    a query a controller could issue as the aircraft crosses the ring, and one that does not
    depend on the landing runway. `anchors` measures the path left to the TRUE threshold, so at its
    anchor the true runway's along-track distance is the bin value — a label pattern a model reads
    (R1 review, 2026-09-13: +1.5–3.4 points of side accuracy at 15–20 km); R1 anchors on rings."""
    ref_lon, ref_lat = reference
    distance = [math.hypot(*_local_m(float(w[1]), float(w[2]), ref_lon, ref_lat)) for w in waypoints]
    out = {ENTRY: 0}
    for radius_km in radii_km:
        inside = next((i for i, d in enumerate(distance) if d <= radius_km * 1000.0), None)
        if inside is not None and inside > 0:
            out[f"r{radius_km:g}km"] = inside
    return out


def minutes_since_each(context: RunwayContext, anchor_time: datetime, candidates: Sequence[str]) -> list[float]:
    """Minutes since each candidate's last landing before ``anchor_time``, capped at `SINCE_CAP_MIN`
    (a runway not landed on within the cap reads as the cap)."""
    since = {r: SINCE_CAP_MIN for r in candidates}
    for landing in reversed(context.recent(anchor_time, timedelta(minutes=SINCE_CAP_MIN))):
        if landing.runway in since and since[landing.runway] == SINCE_CAP_MIN:
            since[landing.runway] = (anchor_time - landing.time).total_seconds() / 60.0
    return [since[r] for r in candidates]


def track_course_at(waypoints: Sequence[Sequence[float]], index: int) -> float:
    """The course INTO the anchor sample (a backward difference: nothing after the anchor);
    at the entry sample, the only one with nothing before it, the first segment's course."""
    i0, i1 = (index - 1, index) if index > 0 else (0, 1)
    _, lon0, lat0, _ = waypoints[i0]
    _, lon1, lat1, _ = waypoints[i1]
    return course_deg(lon0, lat0, lon1, lat1)


def _local_m(lon: float, lat: float, ref_lon: float, ref_lat: float) -> tuple[float, float]:
    return (lon - ref_lon) * metres_per_deg_lon(ref_lat), (lat - ref_lat) * METRES_PER_DEG_LAT


def airline(flight: Mapping[str, Any]) -> str:
    """The callsign's operator prefix; a flight with no callsign has none."""
    return str(flight.get("callsign") or "")[:3]


@dataclass(frozen=True)
class FeatureSpace:
    """One airport's feature columns, fixed before any sample is built."""

    candidates: tuple[str, ...]
    targets: Mapping[str, Mapping[str, Any]]   # runway -> {lat, lon, course_deg}
    reference: tuple[float, float]             # lon, lat the entry sectors are measured from
    airlines: tuple[str, ...]
    names: tuple[str, ...]
    #: column name -> interpretable group, for grouped permutation importance
    groups: Mapping[str, str]


def feature_space(
    candidates: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    reference: tuple[float, float],
    vocabulary_flights: Iterable[Mapping[str, Any]],
) -> FeatureSpace:
    """The columns. Which operator prefixes get one is read from ``vocabulary_flights``' callsigns
    — a vocabulary, no label (the R1 runner passes every flight it may use)."""
    counts = Counter(airline(flight) for flight in vocabulary_flights)
    counts.pop("", None)
    airlines = tuple(prefix for prefix, _ in counts.most_common(TOP_AIRLINES))
    columns: list[tuple[str, str]] = []
    for window in CONTEXT_WINDOWS_MIN:
        columns += [(f"share{window:g}_{r}", "configuration") for r in candidates]
        columns.append((f"landings{window:g}", "configuration"))
    columns += [(f"last_{r}", "configuration") for r in candidates]
    columns.append(("min_since_last", "configuration"))
    columns += [(f"sector_last_{r}", "same-sector landing") for r in candidates]
    columns.append(("min_since_sector_last", "same-sector landing"))
    columns += [("wind_from_east_kt", "wind"), ("wind_from_north_kt", "wind"),
                ("wind_speed_kt", "wind"), ("metar_age_min", "wind"), ("metar_missing", "wind")]
    columns += [(f"headwind_{r}", "wind") for r in candidates]
    columns += [("east_km", "own position & track"), ("north_km", "own position & track"),
                ("dist_km", "own position & track"), ("bearing_sin", "own position & track"),
                ("bearing_cos", "own position & track")]
    for r in candidates:
        columns += [(f"along_{r}_km", "own position & track"), (f"cross_{r}_km", "own position & track"),
                    (f"course_diff_cos_{r}", "own position & track"),
                    (f"course_diff_sin_{r}", "own position & track")]
    columns += [("course_sin", "own position & track"), ("course_cos", "own position & track"),
                ("groundspeed_mps", "own position & track"), ("alt_m", "own position & track"),
                ("vrate_mps", "own position & track"), ("d_east_60s_km", "own position & track"),
                ("d_north_60s_km", "own position & track"), ("t_since_entry_s", "own position & track")]
    columns += [(f"sector_{k}", "entry sector") for k in range(ENTRY_SECTORS)]
    columns += [(f"airline_{a}", "airline") for a in airlines] + [("airline_other", "airline")]
    columns += [("hour_sin", "time of day"), ("hour_cos", "time of day")]
    return FeatureSpace(
        candidates=tuple(candidates), targets=targets, reference=reference, airlines=airlines,
        names=tuple(name for name, _ in columns), groups=dict(columns),
    )


def anchor_features(
    space: FeatureSpace,
    context: RunwayContext,
    flight: Mapping[str, Any],
    index: int,
    *,
    sector: int,
    anchor_time: datetime,
) -> np.ndarray:
    """One sample: the columns of ``space`` at waypoint ``index`` (see the module docstring)."""
    waypoints = flight["waypoints"]
    values: list[float] = []
    since_cap = timedelta(minutes=SINCE_CAP_MIN)

    for window in CONTEXT_WINDOWS_MIN:
        recent = Counter(landing.runway for landing in context.recent(anchor_time, timedelta(minutes=window)))
        total = sum(recent.values())
        values += [recent[r] / total if total else 0.0 for r in space.candidates]
        values.append(float(total))
    history = context.recent(anchor_time, since_cap)
    last = history[-1] if history else None
    values += [float(last is not None and last.runway == r) for r in space.candidates]
    values.append((anchor_time - last.time).total_seconds() / 60.0 if last else SINCE_CAP_MIN)
    same = [landing for landing in context.recent(anchor_time, context.sector_window) if landing.sector == sector]
    sector_last = same[-1] if same else None
    values += [float(sector_last is not None and sector_last.runway == r) for r in space.candidates]
    values.append(
        (anchor_time - sector_last.time).total_seconds() / 60.0 if sector_last
        else context.sector_window.total_seconds() / 60.0
    )

    wind = context.wind_at(anchor_time)
    usable = wind is not None and wind.direction_deg is not None and wind.speed_kt is not None
    if usable:
        rad = math.radians(wind.direction_deg)
        values += [wind.speed_kt * math.sin(rad), wind.speed_kt * math.cos(rad), wind.speed_kt,
                   (anchor_time - wind.valid).total_seconds() / 60.0, 0.0]
        values += [wind.speed_kt * math.cos(math.radians(wind.direction_deg - float(space.targets[r]["course_deg"])))
                    for r in space.candidates]
    else:
        values += [0.0, 0.0, 0.0, SINCE_CAP_MIN, 1.0] + [0.0] * len(space.candidates)

    t_rel, lon, lat, alt = (float(v) for v in waypoints[index])
    ref_lon, ref_lat = space.reference
    east, north = _local_m(lon, lat, ref_lon, ref_lat)
    bearing = math.atan2(east, north)
    values += [east / 1000.0, north / 1000.0, math.hypot(east, north) / 1000.0,
               math.sin(bearing), math.cos(bearing)]
    course = track_course_at(waypoints, index)
    for r in space.candidates:
        target = space.targets[r]
        te, tn = _local_m(lon, lat, float(target["lon"]), float(target["lat"]))
        runway_rad = math.radians(float(target["course_deg"]))
        ue, un = math.sin(runway_rad), math.cos(runway_rad)
        # distance to go along the inbound course (+ before the threshold), cross-track (+ right)
        values += [-(te * ue + tn * un) / 1000.0, (te * un - tn * ue) / 1000.0]
        diff = math.radians(course - float(target["course_deg"]))
        values += [math.cos(diff), math.sin(diff)]
    previous = waypoints[index - 1] if index > 0 else waypoints[index]
    dt = max(t_rel - float(previous[0]), 1e-6)
    pe, pn = _local_m(float(previous[1]), float(previous[2]), ref_lon, ref_lat)
    groundspeed = math.hypot(east - pe, north - pn) / dt if index > 0 else 0.0
    vrate = (alt - float(previous[3])) / dt if index > 0 else 0.0
    past = index
    while past > 0 and t_rel - float(waypoints[past][0]) < HISTORY_S:
        past -= 1
    he, hn = _local_m(float(waypoints[past][1]), float(waypoints[past][2]), ref_lon, ref_lat)
    values += [math.sin(math.radians(course)), math.cos(math.radians(course)), groundspeed, alt,
               vrate, (east - he) / 1000.0, (north - hn) / 1000.0, t_rel - float(waypoints[0][0])]

    values += [float(sector == k) for k in range(ENTRY_SECTORS)]
    operator = airline(flight)
    values += [float(operator == a) for a in space.airlines] + [float(operator not in space.airlines)]
    hour = anchor_time.hour + anchor_time.minute / 60.0
    values += [math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)]
    row = np.asarray(values, dtype=np.float64)
    assert row.shape == (len(space.names),), (row.shape, len(space.names))
    return row


#: R1.1 (plan §14): the columns of one CANDIDATE's row, each read in that runway's own terms, so one
#: learned rule applies to every runway — "the recent landings went here" is learned once, from all
#: of them. No column names a runway.
CANDIDATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("prior_share", "static prior"),            # the runway's share of the partition's training landings
    ("share30", "configuration"),
    ("share60", "configuration"),
    ("group_share30", "configuration"),         # its direction group's share of the last 30 min
    ("last", "configuration"),                  # it took the last landing
    ("min_since", "configuration"),             # minutes since its last landing, capped
    ("b1", "configuration"),                    # it is B1's pick
    ("sector_last", "same-sector landing"),
    ("headwind_kt", "wind"),
    ("crosswind_kt", "wind"),                   # magnitude
    ("along_km", "own position & track"),
    ("cross_km", "own position & track"),
    ("course_diff_cos", "own position & track"),
    ("course_diff_sin", "own position & track"),
    ("d_along_60s_km", "own position & track"),  # the last 60 s of motion in its axes
    ("d_cross_60s_km", "own position & track"),
    ("airline_share", "airline"),               # the operator's share of training landings on it
)
#: R1's columns every candidate row of a sample carries unchanged. Alone they move every candidate's
#: score alike, so under the softmax they only act through the per-candidate columns. R1's day-level
#: columns (the raw wind, the report's age, the time of day) and its one-hot identities (airline,
#: entry sector) are left out: R1 read the first two as a fingerprint of the usual configuration
#: (plan §13.3 point 4), and a one-hot identity has no candidate-relative form.
SHARED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("landings30", "configuration"),
    ("landings60", "configuration"),
    ("min_since_last", "configuration"),
    ("min_since_sector_last", "same-sector landing"),
    ("metar_missing", "wind"),
    ("dist_km", "own position & track"),
    ("groundspeed_mps", "own position & track"),
    ("alt_m", "own position & track"),
    ("vrate_mps", "own position & track"),
    ("t_since_entry_s", "own position & track"),
)
CANDIDATE_ROW_NAMES: tuple[str, ...] = tuple(name for name, _ in CANDIDATE_COLUMNS + SHARED_COLUMNS)
CANDIDATE_ROW_GROUPS: dict[str, str] = dict(CANDIDATE_COLUMNS + SHARED_COLUMNS)


def candidate_rows(
    space: FeatureSpace,
    flat: np.ndarray,
    *,
    group_of: np.ndarray,
    prior_share: np.ndarray,
    b1_pick: np.ndarray,
    minutes_since: np.ndarray,
    airline_share: np.ndarray,
) -> np.ndarray:
    """R1.1's table ``[n, C, len(CANDIDATE_ROW_NAMES)]``: for each of the ``n`` samples, one row per
    candidate runway. ``flat`` is R1's matrix (columns ``space.names``) and supplies everything it
    already holds, re-indexed per candidate; the rest is passed in — ``prior_share`` ``[C]``,
    ``b1_pick`` ``[n]`` (candidate index), ``minutes_since`` / ``airline_share`` ``[n, C]``;
    ``group_of`` ``[C]`` is each candidate's direction group."""
    column = {name: i for i, name in enumerate(space.names)}
    n, count = len(flat), len(space.candidates)

    def per(pattern: str) -> np.ndarray:
        return flat[:, [column[pattern.format(r=r)] for r in space.candidates]]

    def shared(name: str) -> np.ndarray:
        return np.broadcast_to(flat[:, [column[name]]], (n, count))

    share30 = per("share30_{r}")
    group_share30 = np.stack([share30[:, group_of == group_of[c]].sum(axis=1) for c in range(count)], axis=1)
    course = np.radians([float(space.targets[r]["course_deg"]) for r in space.candidates])
    ue, un = np.sin(course), np.cos(course)
    from_east, from_north = flat[:, [column["wind_from_east_kt"]]], flat[:, [column["wind_from_north_kt"]]]
    d_east, d_north = flat[:, [column["d_east_60s_km"]]], flat[:, [column["d_north_60s_km"]]]
    values = {
        "prior_share": np.broadcast_to(np.asarray(prior_share, dtype=np.float64), (n, count)),
        "share30": share30,
        "share60": per("share60_{r}"),
        "group_share30": group_share30,
        "last": per("last_{r}"),
        "min_since": minutes_since,
        "b1": (np.asarray(b1_pick)[:, None] == np.arange(count)[None, :]).astype(np.float64),
        "sector_last": per("sector_last_{r}"),
        "headwind_kt": per("headwind_{r}"),
        # the wind vector across each course (R1's headwind is the component along it)
        "crosswind_kt": np.abs(from_east * un - from_north * ue),
        "along_km": per("along_{r}_km"),
        "cross_km": per("cross_{r}_km"),
        "course_diff_cos": per("course_diff_cos_{r}"),
        "course_diff_sin": per("course_diff_sin_{r}"),
        # the same axes `anchor_features` reads: along = distance to go, cross = right of the course
        "d_along_60s_km": -(d_east * ue + d_north * un),
        "d_cross_60s_km": d_east * un - d_north * ue,
        "airline_share": airline_share,
    }
    columns = [values[name] for name, _ in CANDIDATE_COLUMNS] + [shared(name) for name, _ in SHARED_COLUMNS]
    return np.stack(columns, axis=2)

