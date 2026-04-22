"""
test_generate_czml.py
=====================
Unit tests for the CZML generator.

These tests verify the JSON structure of the generated CZML without loading
it into CesiumJS.  Structure bugs (wrong key names, wrong array layout)
are caught here, not in the browser.
"""

from datetime import datetime, timezone
import math
from pathlib import Path

from generate_czml import (
    build_document_packet,
    build_position_property,
    build_orientation_property,
    build_flight_packet,
    build_czml,
    compute_velocity_orientation,
    default_input_path,
    default_output_path,
)

START_DT = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)
SAMPLE_WAYPOINTS = [
    (0,    -119.38, 49.95, 4500),
    (120,  -119.40, 49.90, 3800),
    (240,  -119.42, 49.85, 3200),
]


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildDocumentPacket:
    def test_id_is_document(self):
        doc = build_document_packet(START_DT, START_DT, 60)
        assert doc["id"] == "document"

    def test_clock_interval_format(self):
        from datetime import timedelta
        end_dt = START_DT + timedelta(hours=1)
        doc = build_document_packet(START_DT, end_dt, 60)
        assert doc["clock"]["interval"] == "2026-04-01T08:00:00Z/2026-04-01T09:00:00Z"

    def test_multiplier_passed_through(self):
        doc = build_document_packet(START_DT, START_DT, 120)
        assert doc["clock"]["multiplier"] == 120


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildPositionProperty:
    def test_cartographic_degrees_length(self):
        prop = build_position_property(START_DT, SAMPLE_WAYPOINTS)
        assert len(prop["cartographicDegrees"]) == 12

    def test_first_four_values_match_first_waypoint(self):
        prop = build_position_property(START_DT, SAMPLE_WAYPOINTS)
        arr = prop["cartographicDegrees"]
        assert arr[0] == 0
        assert arr[1] == -119.38
        assert arr[2] == 49.95
        assert arr[3] == 4500

    def test_epoch_is_iso_utc_string(self):
        prop = build_position_property(START_DT, SAMPLE_WAYPOINTS)
        assert prop["epoch"] == "2026-04-01T08:00:00Z"

    def test_empty_waypoints_gives_empty_array(self):
        prop = build_position_property(START_DT, [])
        assert prop["cartographicDegrees"] == []


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildFlightPacket:
    def setup_method(self):
        self.packet = build_flight_packet(
            "UAL123", "United 123", "B738",
            SAMPLE_WAYPOINTS, START_DT, (255, 140, 0, 200)
        )

    def test_id(self):
        assert self.packet["id"] == "UAL123"

    def test_model_gltf_path(self):
        assert self.packet["model"]["gltf"] == "/models/aircraft.glb"

    def test_orientation_has_unit_quaternion(self):
        ori = self.packet["orientation"]
        assert "unitQuaternion" in ori
        assert "epoch" in ori
        # 3 waypoints → 3 samples × 5 values each (t, x, y, z, w)
        assert len(ori["unitQuaternion"]) == 15

    def test_trail_color_in_path(self):
        assert self.packet["path"]["material"]["solidColor"]["color"]["rgba"] == [255, 140, 0, 200]


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures for build_czml (no mock data — use explicit inline flights)
TEST_FLIGHTS = [
    {
        "id": "TEST1", "callsign": "Test One", "type": "B738",
        "waypoints": [[0, -114.0, 51.0, 5000], [300, -114.01, 51.12, 1100]],
    },
    {
        "id": "TEST2", "callsign": "Test Two", "type": "A320",
        "waypoints": [[0, -113.5, 51.2, 6000], [600, -114.01, 51.12, 1100]],
    },
]


class TestBuildCzml:
    def test_first_element_is_document(self):
        czml = build_czml(TEST_FLIGHTS, START_DT)
        assert czml[0]["id"] == "document"

    def test_length_equals_flights_plus_one(self):
        czml = build_czml(TEST_FLIGHTS, START_DT)
        assert len(czml) == len(TEST_FLIGHTS) + 1

    def test_end_time_derived_from_max_offset(self):
        """Document packet end time should equal start + max offset."""
        czml = build_czml(TEST_FLIGHTS, START_DT)
        doc = czml[0]
        interval = doc["clock"]["interval"]
        assert interval.endswith("2026-04-01T08:10:00Z")


# ─────────────────────────────────────────────────────────────────────────────
class TestComputeVelocityOrientation:
    """Verify heading / pitch derived from the 3-D velocity vector."""

    def test_northbound_heading_near_zero(self):
        # Flying due north: same longitude, increasing latitude
        wps = [(0, -114.0, 50.0, 3000), (60, -114.0, 51.0, 3000)]
        orients = compute_velocity_orientation(wps)
        heading, pitch = orients[0]
        assert abs(heading) < 0.01  # ≈ 0 (North)
        assert abs(pitch) < 0.01   # level flight

    def test_eastbound_heading_near_pi_half(self):
        # Flying due east: same latitude, increasing longitude
        wps = [(0, -114.0, 50.0, 3000), (60, -113.0, 50.0, 3000)]
        orients = compute_velocity_orientation(wps)
        heading, _pitch = orients[0]
        assert abs(heading - math.pi / 2) < 0.05  # ≈ π/2 (East)

    def test_descending_gives_negative_pitch(self):
        wps = [(0, -114.0, 50.0, 5000), (60, -114.0, 50.1, 3000)]
        orients = compute_velocity_orientation(wps)
        _heading, pitch = orients[0]
        assert pitch < 0  # descending

    def test_single_waypoint_returns_zero(self):
        wps = [(0, -114.0, 50.0, 3000)]
        orients = compute_velocity_orientation(wps)
        assert orients == [(0.0, 0.0)]

    def test_last_waypoint_uses_backward_difference(self):
        wps = [
            (0, -114.0, 50.0, 3000),
            (60, -114.0, 51.0, 3000),
        ]
        orients = compute_velocity_orientation(wps)
        # Both waypoints should have the same orientation (same segment)
        assert orients[0] == orients[1]


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildOrientationProperty:
    def test_structure_has_epoch_and_quaternion(self):
        wps = [(0, -114.0, 50.0, 3000), (60, -114.0, 51.0, 3000)]
        prop = build_orientation_property(START_DT, wps)
        assert "epoch" in prop
        assert "unitQuaternion" in prop
        assert prop["interpolationAlgorithm"] == "LINEAR"

    def test_flat_array_length(self):
        wps = [(0, -114.0, 50.0, 3000), (60, -114.0, 51.0, 3000)]
        prop = build_orientation_property(START_DT, wps)
        # 2 waypoints × 5 values (t, x, y, z, w)
        assert len(prop["unitQuaternion"]) == 10

    def test_quaternions_are_unit_length(self):
        wps = SAMPLE_WAYPOINTS
        prop = build_orientation_property(START_DT, wps)
        arr = prop["unitQuaternion"]
        for i in range(0, len(arr), 5):
            x, y, z, w = arr[i + 1], arr[i + 2], arr[i + 3], arr[i + 4]
            norm = math.sqrt(x*x + y*y + z*z + w*w)
            assert abs(norm - 1.0) < 1e-9, f"Quaternion at index {i} not unit: {norm}"


class TestDefaultPaths:
    def test_default_output_path_is_airport_scoped(self):
        output_path = default_output_path("krdu")
        assert output_path.as_posix().endswith("/public/data/airports/KRDU/trajectories.czml")

    def test_default_input_path_prefers_latest_airport_output(self, monkeypatch, tmp_path):
        airport_dir = tmp_path / "krdu"
        airport_dir.mkdir(parents=True)
        older = airport_dir / "krdu_czml_input_20260401T080000Z.json"
        newer = airport_dir / "krdu_czml_input_20260402T080000Z.json"
        older.write_text("[]", encoding="utf-8")
        newer.write_text("[]", encoding="utf-8")

        monkeypatch.setattr("generate_czml.OPENSKY_OUTPUT_ROOT", tmp_path)

        assert default_input_path("KRDU") == newer

    def test_default_input_path_falls_back_to_airport_specific_placeholder(self, monkeypatch, tmp_path):
        monkeypatch.setattr("generate_czml.OPENSKY_OUTPUT_ROOT", tmp_path)

        assert default_input_path("CYVR") == Path(tmp_path) / "cyvr" / "cyvr_czml_input_latest.json"
