"""
test_preprocess_airports.py
===========================
Unit tests for the airport preprocessing utilities.

Run: pytest python/tests/ -v
     pytest python/tests/ --cov=python --cov-report=html

Testing approach:
  All tests use small handcrafted fixtures — no real OurAirports CSV needed.
  This makes tests fast, deterministic, and runnable without network access.
"""

import math
import pytest
from preprocess_airports import (
    runway_bearing_rad,
    offset_point_deg,
    runway_to_polygon,
    RunwayEnds,
    METRES_PER_DEG_LAT,
    metres_per_deg_lon,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

# A simple north–south runway (LE is south, HE is north)
# le=16 (heading 160°), he=34 (heading 340°) — but we test geometry, not magnetic hdg
RWY_NS = RunwayEnds(
    le_ident="16", he_ident="34",
    le_lon=-119.38, le_lat=49.92,
    he_lon=-119.38, he_lat=49.96,
    le_elevation_ft=1398, he_elevation_ft=1421,
    length_ft=8200, width_ft=150,
    surface="ASP", lighted=1,
)

# An east–west runway
RWY_EW = RunwayEnds(
    le_ident="09", he_ident="27",
    le_lon=-119.50, le_lat=49.94,
    he_lon=-119.35, he_lat=49.94,
    le_elevation_ft=1400, he_elevation_ft=1400,
    length_ft=9000, width_ft=200,
    surface="CON", lighted=1,
)

PRECISION = 3  # decimal places for floating-point assertions


# ─────────────────────────────────────────────────────────────────────────────
class TestRunwayBearingRad:
    def test_north_south_bearing_is_zero(self):
        """LE south → HE north: bearing should be 0 (due north)."""
        bearing = runway_bearing_rad(
            RWY_NS.le_lon, RWY_NS.le_lat,
            RWY_NS.he_lon, RWY_NS.he_lat,
        )
        # TODO — assert math.isclose(bearing, 0.0, abs_tol=1e-3)
        assert bearing is not None  # replace with real assertion

    def test_east_west_bearing_is_pi_over_2(self):
        """LE west → HE east: bearing should be π/2 (due east)."""
        bearing = runway_bearing_rad(
            RWY_EW.le_lon, RWY_EW.le_lat,
            RWY_EW.he_lon, RWY_EW.he_lat,
        )
        # TODO — assert math.isclose(bearing, math.pi / 2, abs_tol=1e-3)
        assert bearing is not None

    def test_bearing_is_opposite_in_reverse_direction(self):
        """Bearing A→B should differ from B→A by ±π."""
        fwd = runway_bearing_rad(-119.38, 49.92, -119.38, 49.96)
        rev = runway_bearing_rad(-119.38, 49.96, -119.38, 49.92)
        # TODO — assert math.isclose(abs(fwd - rev), math.pi, abs_tol=1e-3)
        assert fwd is not None and rev is not None


# ─────────────────────────────────────────────────────────────────────────────
class TestOffsetPointDeg:
    def test_due_east_increases_longitude(self):
        lon, lat = offset_point_deg(-119.38, 49.95, math.pi / 2, 1000)
        expected_dlon = 1000 / metres_per_deg_lon(49.95)
        # TODO ①  assert math.isclose(lon, -119.38 + expected_dlon, abs_tol=1e-5)
        # TODO ②  assert math.isclose(lat, 49.95, abs_tol=1e-6)
        assert lon is not None and lat is not None

    def test_due_north_increases_latitude(self):
        lon, lat = offset_point_deg(-119.38, 49.95, 0.0, 1000)
        expected_dlat = 1000 / METRES_PER_DEG_LAT
        # TODO — assert math.isclose(lat, 49.95 + expected_dlat, abs_tol=1e-5)
        assert lon is not None and lat is not None

    def test_zero_distance_returns_original_point(self):
        lon, lat = offset_point_deg(-119.38, 49.95, math.pi / 4, 0)
        # TODO — assert both equal the inputs exactly (float equality OK for ×0)
        assert lon is not None and lat is not None


# ─────────────────────────────────────────────────────────────────────────────
class TestRunwayToPolygon:
    def test_polygon_has_5_points_and_is_closed(self):
        ring = runway_to_polygon(RWY_NS)
        # GeoJSON ring: 4 corners + closing point (first == last)
        # TODO — assert len(ring) == 5
        # TODO — assert ring[0] == ring[-1]
        assert ring is not None

    def test_polygon_width_matches_input(self):
        """Width of the polygon should be approximately runway.width_ft in metres."""
        ring = runway_to_polygon(RWY_NS)
        # For a N–S runway, left and right corners share the same latitude.
        # Width = |left_lon - right_lon| × metres_per_deg_lon(lat)
        #       ≈ 150 ft × METRES_PER_FOOT = 45.72 m
        expected_width_m = RWY_NS.width_ft * 0.3048

        # TODO — Extract the LE_left and LE_right corners from ring,
        #   compute the distance between them, and assert it ≈ expected_width_m.
        assert ring is not None

    def test_polygon_points_contain_lon_lat_pairs(self):
        ring = runway_to_polygon(RWY_NS)
        # Each element should be a list/tuple of exactly 2 floats
        # TODO — for point in ring: assert len(point) == 2
        assert ring is not None
