from __future__ import annotations

import unittest

from trajectory_data_process.processing.trajectory_events import extract_airport_events
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


def _trajectory(arr=None, dep=None, geo_floor=1850.0) -> Trajectory:
    coords = [
        (0, 0.10, 2000.0),
        (10, 0.06, 1900.0),
        (20, 0.02, geo_floor),
        (30, 0.06, 1900.0),
        (40, 0.10, 2000.0),
    ]
    points = [
        TrajectoryPoint(t, lat, 0.0, geo, geo + 20, 180.0, on_ground=False) for t, lat, geo in coords
    ]
    return Trajectory(icao24="abc123", callsign="TST123", dep_airport=dep, arr_airport=arr, points=points)


class TrajectoryEventsTests(unittest.TestCase):
    def test_pass_event_high_over_airport(self) -> None:
        events = extract_airport_events(
            _trajectory(), airport="TST", airport_lat=0.0, airport_lon=0.0, airport_elev_m=0.0
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "pass")
        self.assertEqual(events[0]["point_range"], {"start_index": 1, "end_index": 3})

    def test_arrival_low_over_airport_is_landing(self) -> None:
        events = extract_airport_events(
            _trajectory(arr="TST", geo_floor=300.0),
            airport="TST",
            airport_lat=0.0,
            airport_lon=0.0,
            airport_elev_m=0.0,
        )

        self.assertEqual(events[0]["label"], "landing")
        self.assertTrue(events[0]["label_evidence"]["arr_airport_matches"])

    def test_no_event_when_never_inside_radius(self) -> None:
        far = Trajectory(
            icao24="zzz",
            callsign=None,
            dep_airport=None,
            arr_airport=None,
            points=[
                TrajectoryPoint(0, 1.0, 0.0, 2000.0, 2000.0, None, False),
                TrajectoryPoint(10, 1.1, 0.0, 2000.0, 2000.0, None, False),
            ],
        )

        self.assertEqual(
            extract_airport_events(far, airport="TST", airport_lat=0.0, airport_lon=0.0, airport_elev_m=0.0),
            [],
        )


if __name__ == "__main__":
    unittest.main()
