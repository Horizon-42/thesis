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
from data_layout import airport_data_path, upsert_airports_index
from preprocess_airports import (
    runway_bearing_rad,
    runway_bearing_from_metadata,
    offset_point_deg,
    runway_to_polygon,
    landing_zone_polygon,
    build_runway_geojson,
    load_airport_config,
    write_airport_config,
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

RWY_WITH_DISPLACED = RunwayEnds(
    le_ident="16", he_ident="34",
    le_lon=-119.3789978, le_lat=49.9660988,
    he_lon=-119.3759995, he_lat=49.9462013,
    le_elevation_ft=1421, he_elevation_ft=1370,
    length_ft=8900, width_ft=200,
    surface="ASP", lighted=1,
    le_displaced_threshold_ft=1200,
    he_displaced_threshold_ft=400,
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
        assert math.isclose(bearing, 0.0, abs_tol=10**(-PRECISION))

    def test_east_west_bearing_is_pi_over_2(self):
        """LE west → HE east: bearing should be π/2 (due east)."""
        bearing = runway_bearing_rad(
            RWY_EW.le_lon, RWY_EW.le_lat,
            RWY_EW.he_lon, RWY_EW.he_lat,
        )
        assert math.isclose(bearing, math.pi / 2, abs_tol=10**(-PRECISION))

    def test_bearing_is_opposite_in_reverse_direction(self):
        """Bearing A→B should differ from B→A by ±π."""
        fwd = runway_bearing_rad(-119.38, 49.92, -119.38, 49.96)
        rev = runway_bearing_rad(-119.38, 49.96, -119.38, 49.92)
        assert fwd is not None and rev is not None  
        assert math.isclose((fwd - rev) % (2 * math.pi), math.pi, abs_tol=10**(-PRECISION))

    def test_metadata_heading_overrides_coordinate_bearing(self):
        rwy = RunwayEnds(
            le_ident="16", he_ident="34",
            le_lon=-119.38, le_lat=49.92,
            he_lon=-119.36, he_lat=49.98,
            le_elevation_ft=0, he_elevation_ft=0,
            length_ft=8000, width_ft=150,
            surface="ASP", lighted=1,
            le_heading_degT=175.0,
        )
        bearing = runway_bearing_from_metadata(rwy)
        assert math.isclose(bearing, math.radians(175.0), abs_tol=10**(-PRECISION))


# ─────────────────────────────────────────────────────────────────────────────
class TestOffsetPointDeg:
    def test_due_east_increases_longitude(self):
        lon, lat = offset_point_deg(-119.38, 49.95, math.pi / 2, 1000)
        expected_dlon = 1000 / metres_per_deg_lon(49.95)
        # ①  assert math.isclose(lon, -119.38 + expected_dlon, abs_tol=10**(-PRECISION))
        # ②  assert math.isclose(lat, 49.95, abs_tol=10**(-PRECISION))
        assert lon is not None and lat is not None
        assert math.isclose(lon, -119.38 + expected_dlon, abs_tol=10**(-PRECISION))
        assert math.isclose(lat, 49.95, abs_tol=10**(-PRECISION))

    def test_due_north_increases_latitude(self):
        lon, lat = offset_point_deg(-119.38, 49.95, 0.0, 1000)
        expected_dlat = 1000 / METRES_PER_DEG_LAT
        # — assert math.isclose(lat, 49.95 + expected_dlat, abs_tol=1e-5)
        assert lon is not None and lat is not None
        assert math.isclose(lat, 49.95 + expected_dlat, abs_tol=10**(-PRECISION))

    def test_zero_distance_returns_original_point(self):
        lon, lat = offset_point_deg(-119.38, 49.95, math.pi / 4, 0)
        # — assert both equal the inputs exactly (float equality OK for ×0)
        assert lon is not None and lat is not None
        assert lon == -119.38
        assert lat == 49.95


# ─────────────────────────────────────────────────────────────────────────────
class TestRunwayToPolygon:
    def test_polygon_has_5_points_and_is_closed(self):
        ring = runway_to_polygon(RWY_NS)
        # GeoJSON ring: 4 corners + closing point (first == last)
        # — assert len(ring) == 5
        # — assert ring[0] == ring[-1]
        assert ring is not None
        assert len(ring) == 5
        assert ring[0] == ring[-1]

    def test_polygon_width_matches_input(self):
        """Width of the polygon should be approximately runway.width_ft in metres."""
        ring = runway_to_polygon(RWY_NS)
        # For a N–S runway, left and right corners share the same latitude.
        # Width = |left_lon - right_lon| × metres_per_deg_lon(lat)
        #       ≈ 150 ft × METRES_PER_FOOT = 45.72 m
        expected_width_m = RWY_NS.width_ft * 0.3048

        # — Extract the LE_left and LE_right corners from ring,
        #   compute the distance between them, and assert it ≈ expected_width_m.
        assert ring is not None
        le_left = ring[0]  # [lon, lat]
        le_right = ring[1]  # [lon, lat]
        lat = le_left[1]  # latitude of the corners (should be the same)
        actual_width_m = abs(le_left[0] - le_right[0]) * metres_per_deg_lon(lat)
        # Flat-plane conversion introduces centimetre-level rounding differences.
        assert math.isclose(actual_width_m, expected_width_m, abs_tol=0.02)

    def test_polygon_points_contain_lon_lat_pairs(self):
        ring = runway_to_polygon(RWY_NS)
        # Each element should be a list/tuple of exactly 2 floats
        # — for point in ring: assert len(point) == 2
        assert ring is not None
        for point in ring:
            assert isinstance(point, (list, tuple))
            assert len(point) == 2
            assert all(isinstance(coord, float) for coord in point)

    def test_displaced_thresholds_expand_surface_polygon(self):
        landing_ring = landing_zone_polygon(RWY_WITH_DISPLACED)
        surface_ring = runway_to_polygon(RWY_WITH_DISPLACED)

        # Compare centerline lengths (metres) from LE/HE center points.
        landing_le = ((landing_ring[0][0] + landing_ring[1][0]) / 2, (landing_ring[0][1] + landing_ring[1][1]) / 2)
        landing_he = ((landing_ring[2][0] + landing_ring[3][0]) / 2, (landing_ring[2][1] + landing_ring[3][1]) / 2)
        surface_le = ((surface_ring[0][0] + surface_ring[1][0]) / 2, (surface_ring[0][1] + surface_ring[1][1]) / 2)
        surface_he = ((surface_ring[2][0] + surface_ring[3][0]) / 2, (surface_ring[2][1] + surface_ring[3][1]) / 2)

        def flat_dist_m(a, b):
            lat = (a[1] + b[1]) / 2
            dx = (b[0] - a[0]) * metres_per_deg_lon(lat)
            dy = (b[1] - a[1]) * METRES_PER_DEG_LAT
            return math.hypot(dx, dy)

        assert flat_dist_m(surface_le, surface_he) > flat_dist_m(landing_le, landing_he)


class TestRunwayGeojson:
    def test_outputs_two_polygons_per_runway(self):
        fc = build_runway_geojson([RWY_NS], airport_ident="CYLW")
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2

        zone_types = {f["properties"]["zone_type"] for f in fc["features"]}
        assert zone_types == {"runway_surface", "landing_zone"}


class TestAirportConfig:
    def test_load_airport_config_reads_lon_lat_from_airports_csv(self, tmp_path):
        csv_path = tmp_path / "airports.csv"
        csv_path.write_text(
            "\n".join([
                '"id","ident","latitude_deg","longitude_deg","gps_code","icao_code"',
                '1,"CYYC",51.118822,-114.009933,"CYYC","CYYC"',
            ]),
            encoding="utf-8",
        )

        airport = load_airport_config(csv_path, "CYYC", height_m=15000)

        assert airport == {
            "code": "CYYC",
            "lon": -114.009933,
            "lat": 51.118822,
            "height": 15000,
        }

    def test_write_airport_config_writes_json(self, tmp_path):
        csv_path = tmp_path / "airports.csv"
        out_path = tmp_path / "airport.json"
        csv_path.write_text(
            "\n".join([
                '"id","ident","latitude_deg","longitude_deg","gps_code","icao_code"',
                '1,"CYVR",49.193901,-123.183998,"CYVR","CYVR"',
            ]),
            encoding="utf-8",
        )

        airport = write_airport_config(csv_path, "cyvr", out_path, height_m=12000)

        assert airport["code"] == "CYVR"
        assert out_path.read_text(encoding="utf-8") == (
            '{\n'
            '  "code": "CYVR",\n'
            '  "lon": -123.183998,\n'
            '  "lat": 49.193901,\n'
            '  "height": 12000\n'
            '}'
        )

    def test_airport_data_path_uses_airport_scoped_layout(self):
        path = airport_data_path("krdu", "airport.json")
        assert path.as_posix().endswith("/public/data/airports/KRDU/airport.json")

    def test_upsert_airports_index_writes_sorted_manifest_and_preserves_default(self, tmp_path):
        index_path = tmp_path / "index.json"

        upsert_airports_index(
            airport_code="KRDU",
            airport_name="Raleigh-Durham International Airport",
            lat=35.878659,
            lon=-78.7873,
            default_airport="KRDU",
            index_path=index_path,
        )
        manifest = upsert_airports_index(
            airport_code="CYVR",
            airport_name="Vancouver International Airport",
            lat=49.193901,
            lon=-123.183998,
            index_path=index_path,
        )

        assert manifest["defaultAirport"] == "KRDU"
        assert [airport["code"] for airport in manifest["airports"]] == ["CYVR", "KRDU"]
