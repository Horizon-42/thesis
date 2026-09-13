"""Runway-intent R1 features (`data.runway_features`): causal, and in the runway's own axes."""

import math
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon
from ts_transformer.data.runway_context import ContextLanding, RunwayContext, WindReport
from ts_transformer.data.runway_features import anchor_features, feature_space, ring_anchors

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
THRESHOLD = {"lat": 35.87, "lon": -78.79, "course_deg": 45.0}
TARGETS = {"05L": THRESHOLD, "23R": {**THRESHOLD, "course_deg": 225.0}}


def _context() -> RunwayContext:
    return RunwayContext(
        [ContextLanding(T0 - timedelta(minutes=5), "05L", sector=5)],
        {r: t["course_deg"] for r, t in TARGETS.items()}, Counter({"05L": 3, "23R": 1}),
        [WindReport(T0 - timedelta(minutes=40), 50.0, 8.0)],
        window=timedelta(minutes=30), sector_window=timedelta(minutes=60),
        metar_delay=timedelta(minutes=10), calm_kt=3.0,
    )


def _inbound(distance_km: float, cross_km: float = 0.0) -> tuple[float, float]:
    """A position ``distance_km`` before the 05L threshold on its course, ``cross_km`` right of it."""
    course = math.radians(45.0)
    east = -distance_km * 1000 * math.sin(course) + cross_km * 1000 * math.cos(course)
    north = -distance_km * 1000 * math.cos(course) - cross_km * 1000 * math.sin(course)
    return (THRESHOLD["lon"] + east / metres_per_deg_lon(THRESHOLD["lat"]),
            THRESHOLD["lat"] + north / METRES_PER_DEG_LAT)


def _flight(n: int = 40) -> dict:
    waypoints = []
    for i in range(n):
        lon, lat = _inbound(20.0 - 0.5 * i, cross_km=1.0 if i < 20 else 0.0)
        waypoints.append([4.0 * i, lon, lat, 2000.0 - 40.0 * i])
    return {"callsign": "AAL123", "waypoints": waypoints}


def _space(flight: dict):
    return feature_space(["05L", "23R"], TARGETS, (THRESHOLD["lon"], THRESHOLD["lat"]), [flight])


def test_a_feature_never_reads_past_its_anchor():
    flight = _flight()
    space = _space(flight)
    index = 15
    full = anchor_features(space, _context(), flight, index, sector=5, anchor_time=T0)
    cut = {**flight, "waypoints": flight["waypoints"][: index + 1]}
    altered = {**flight, "waypoints": flight["waypoints"][: index + 1]
               + [[w[0], w[1] + 0.3, w[2] - 0.3, 0.0] for w in flight["waypoints"][index + 1:]]}
    np.testing.assert_array_equal(full, anchor_features(space, _context(), cut, index, sector=5, anchor_time=T0))
    np.testing.assert_array_equal(full, anchor_features(space, _context(), altered, index, sector=5, anchor_time=T0))


def test_the_runway_axes_read_distance_to_go_and_the_right_hand_side():
    flight = _flight()
    space = _space(flight)
    row = dict(zip(space.names, anchor_features(space, _context(), flight, 0, sector=5, anchor_time=T0)))
    assert math.isclose(row["along_05L_km"], 20.0, abs_tol=0.05)
    assert math.isclose(row["cross_05L_km"], 1.0, abs_tol=0.05)
    # the reciprocal runway sees the same aircraft from the other end: behind it, on its left
    assert math.isclose(row["along_23R_km"], -20.0, abs_tol=0.05)
    assert math.isclose(row["cross_23R_km"], -1.0, abs_tol=0.05)


def test_the_context_columns_read_the_airport_before_the_anchor():
    flight = _flight()
    space = _space(flight)
    row = dict(zip(space.names, anchor_features(space, _context(), flight, 10, sector=5, anchor_time=T0)))
    assert row["share30_05L"] == 1.0 and row["landings30"] == 1.0 and row["last_05L"] == 1.0
    assert math.isclose(row["min_since_last"], 5.0)
    assert row["sector_last_05L"] == 1.0
    assert row["metar_missing"] == 0.0 and math.isclose(row["metar_age_min"], 40.0)
    assert row["headwind_05L"] > 0 > row["headwind_23R"]
    assert row["airline_AAL"] == 1.0 and row["airline_other"] == 0.0
    # nothing has happened yet at the first landing's own minute
    early = dict(zip(space.names, anchor_features(
        space, _context(), flight, 10, sector=5, anchor_time=T0 - timedelta(minutes=5))))
    assert early["landings30"] == 0.0 and early["last_05L"] == 0.0


def test_ring_anchors_do_not_depend_on_the_landing_runway():
    """Two flights on the same path that land on different runways get the same anchors: the
    rings are around the airport reference, never measured to a threshold."""
    flight = _flight()
    reference = (THRESHOLD["lon"], THRESHOLD["lat"])
    got = ring_anchors(flight["waypoints"], reference, [15.0, 10.0, 3.0, 0.1])
    assert got["entry"] == 0
    assert got == ring_anchors([list(w) for w in flight["waypoints"]], reference, [15.0, 10.0, 3.0, 0.1])
    lon, lat = flight["waypoints"][got["r10km"]][1:3]
    before = flight["waypoints"][got["r10km"] - 1][1:3]
    distance = lambda p: math.hypot((p[0] - reference[0]) * metres_per_deg_lon(reference[1]),  # noqa: E731
                                    (p[1] - reference[1]) * METRES_PER_DEG_LAT)
    assert distance((lon, lat)) <= 10_000.0 < distance(before)
    # a ring the track never enters (it ends at 0.5 km from the reference) has no anchor
    assert "r0.1km" not in got
