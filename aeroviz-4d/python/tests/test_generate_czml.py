"""
test_generate_czml.py
=====================
Unit tests for the CZML generator.

These tests verify the JSON structure of the generated CZML without loading
it into CesiumJS.  Structure bugs (wrong key names, wrong array layout)
are caught here, not in the browser.
"""

from datetime import datetime, timezone
from generate_czml import (
    build_document_packet,
    build_position_property,
    build_flight_packet,
    build_czml,
    MOCK_FLIGHTS,
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

    def test_orientation_velocity_reference(self):
        assert self.packet["orientation"]["velocityReference"] == "#UAL123.position"

    def test_trail_color_in_path(self):
        assert self.packet["path"]["material"]["solidColor"]["color"]["rgba"] == [255, 140, 0, 200]


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildCzml:
    def test_first_element_is_document(self):
        czml = build_czml(MOCK_FLIGHTS, START_DT)
        assert czml[0]["id"] == "document"

    def test_length_equals_flights_plus_one(self):
        czml = build_czml(MOCK_FLIGHTS, START_DT)
        assert len(czml) == len(MOCK_FLIGHTS) + 1

    def test_end_time_derived_from_max_offset(self):
        """Document packet end time should equal start + max offset."""
        czml = build_czml(MOCK_FLIGHTS, START_DT)
        doc = czml[0]
        interval = doc["clock"]["interval"]
        assert interval.endswith("2026-04-01T08:19:00Z")
