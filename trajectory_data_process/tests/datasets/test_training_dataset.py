from __future__ import annotations

import copy
import unittest

from trajectory_data_process.datasets.training_dataset import (
    attach_training_points_or_quarantine,
    make_raw_track_record,
)


class TrainingDatasetTests(unittest.TestCase):
    def test_make_raw_track_record_keeps_track_payload_unchanged(self) -> None:
        track = {
            "icao24": "abc123",
            "path": [[1000, 49.98, -119.31, 1500.0, 210.0, False]],
        }
        original = copy.deepcopy(track)

        record = make_raw_track_record(
            airport="cylw",
            fetch_profile="terminal_all",
            candidate_sources=["arrival", "area"],
            source_response_ids=["sha256:source"],
            flight_metadata={"estArrivalAirport": "CYLW"},
            track=track,
        )

        self.assertEqual(track, original)
        self.assertEqual(record["airport"], "CYLW")
        self.assertEqual(record["track"], original)
        self.assertEqual(record["source"]["candidate_sources"], ["arrival", "area"])
        self.assertTrue(record["raw_track_id"].startswith("raw_track:sha256:"))

    def test_attach_training_points_requires_geo_altitude(self) -> None:
        event = {
            "event_id": "evt1",
            "airport": "CYLW",
            "label": "pass",
            "raw_waypoint_range": {"start_raw_index": 1, "end_raw_index": 2},
        }
        points = [
            {"raw_index": 1, "baro_altitude_m": 1500.0, "geo_altitude_m": 1490.0},
            {"raw_index": 2, "baro_altitude_m": 1510.0, "geo_altitude_m": None},
        ]

        ready, quarantine = attach_training_points_or_quarantine(
            event,
            raw_track_id="raw_track:sha256:abc",
            dual_altitude_points=points,
            require_geo_altitude=True,
        )

        self.assertIsNone(ready)
        self.assertIsNotNone(quarantine)
        self.assertEqual(quarantine["reason"], "missing_geo_altitude")
        self.assertEqual(quarantine["detail"]["missing_raw_indexes"], [2])

    def test_attach_training_points_adds_quality_counts(self) -> None:
        event = {
            "event_id": "evt1",
            "airport": "CYLW",
            "label": "pass",
            "quality": {"complete": True},
            "raw_waypoint_range": {"start_raw_index": 1, "end_raw_index": 2},
        }
        points = [
            {"raw_index": 1, "baro_altitude_m": 1500.0, "geo_altitude_m": 1490.0},
            {"raw_index": 2, "baro_altitude_m": 1510.0, "geo_altitude_m": 1500.0},
        ]

        ready, quarantine = attach_training_points_or_quarantine(
            event,
            raw_track_id="raw_track:sha256:abc",
            dual_altitude_points=points,
        )

        self.assertIsNone(quarantine)
        self.assertIsNotNone(ready)
        assert ready is not None
        self.assertEqual(len(ready["training_points"]), 2)
        self.assertEqual(ready["source"]["raw_track_id"], "raw_track:sha256:abc")
        self.assertEqual(ready["quality"]["points_with_both_altitudes"], 2)


if __name__ == "__main__":
    unittest.main()
