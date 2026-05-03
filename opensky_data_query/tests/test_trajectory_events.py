from __future__ import annotations

import copy
import unittest

from opensky_data_query.trajectory_events import (
    extract_complete_airport_events,
    parse_track_for_analysis,
)


AIRPORT_LAT = 0.0
AIRPORT_LON = 0.0


def sample_track(*path: list[object]) -> dict[str, object]:
    return {
        "icao24": "abc123",
        "callsign": "TST123",
        "startTime": 1000,
        "endTime": 1100,
        "path": list(path),
    }


class TrajectoryEventsTests(unittest.TestCase):
    def test_parse_track_for_analysis_keeps_raw_index_and_sorts_derived_view(self) -> None:
        track = sample_track(
            [30, 0.1000, 0.0, 1600.0, 180.0, False],
            [10, 0.1200, 0.0, 1700.0, 180.0, False],
            [20, None, 0.0, 1700.0, 180.0, False],
        )

        points = parse_track_for_analysis(track, airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON)

        self.assertEqual([point.time for point in points], [10.0, 30.0])
        self.assertEqual([point.raw_index for point in points], [1, 0])

    def test_extract_complete_pass_event_without_mutating_raw_path(self) -> None:
        track = sample_track(
            [1000, 0.1000, 0.0, 2000.0, 180.0, False],
            [1010, 0.0600, 0.0, 1900.0, 180.0, False],
            [1020, 0.0200, 0.0, 1850.0, 180.0, False],
            [1030, 0.0600, 0.0, 1900.0, 0.0, False],
            [1040, 0.1000, 0.0, 2000.0, 0.0, False],
        )
        original_path = copy.deepcopy(track["path"])

        events = extract_complete_airport_events(
            track,
            airport="CYLW",
            airport_lat=AIRPORT_LAT,
            airport_lon=AIRPORT_LON,
            airport_elev_m=0.0,
            radius_nm=5.0,
            candidate_sources=["area"],
            flight_metadata={"estDepartureAirport": "KSEA", "estArrivalAirport": "CYVR"},
        )

        self.assertEqual(track["path"], original_path)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["label"], "pass")
        self.assertTrue(event["complete_radius_crossing"])
        self.assertEqual(event["raw_waypoint_range"], {"start_raw_index": 1, "end_raw_index": 3})
        self.assertEqual(
            event["boundary_crossings"]["entry"]["source_segment"],
            {"before_raw_index": 0, "after_raw_index": 1},
        )
        self.assertEqual(
            event["boundary_crossings"]["exit"]["source_segment"],
            {"before_raw_index": 3, "after_raw_index": 4},
        )
        self.assertAlmostEqual(event["boundary_crossings"]["entry"]["radius_nm"], 5.0)

    def test_arrival_candidate_with_conflicting_geometry_is_ambiguous(self) -> None:
        track = sample_track(
            [1000, 0.1000, 0.0, 3000.0, 180.0, False],
            [1010, 0.0600, 0.0, 2900.0, 180.0, False],
            [1020, 0.0200, 0.0, 2800.0, 180.0, False],
            [1030, 0.0600, 0.0, 2900.0, 0.0, False],
            [1040, 0.1000, 0.0, 3000.0, 0.0, False],
        )

        events = extract_complete_airport_events(
            track,
            airport="CYLW",
            airport_lat=AIRPORT_LAT,
            airport_lon=AIRPORT_LON,
            airport_elev_m=0.0,
            radius_nm=5.0,
            candidate_sources=["arrival"],
            flight_metadata={"estArrivalAirport": "CYLW"},
            low_altitude_agl_m=600.0,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "ambiguous")
        self.assertTrue(events[0]["label_evidence"]["estArrivalAirport_matches"])


if __name__ == "__main__":
    unittest.main()

