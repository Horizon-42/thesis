from __future__ import annotations

import unittest

from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.processing.czml_export import trajectory_to_czml_flight
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


AIRPORT_LAT, AIRPORT_LON = 35.8776, -78.7875


def _approach_trajectory() -> Trajectory:
    coords = [
        (0, 35.95, AIRPORT_LON, 1500.0),
        (60, 35.91, AIRPORT_LON, 1000.0),
        (120, AIRPORT_LAT, AIRPORT_LON, 500.0),
    ]
    points = [
        TrajectoryPoint(
            time=t, lat=lat, lon=lon, geo_altitude_m=geo, baro_altitude_m=geo + 20,
            heading_deg=225.0, on_ground=False,
        )
        for t, lat, lon, geo in coords
    ]
    return Trajectory(icao24="abc123", callsign="TST123", dep_airport="KJFK", arr_airport="KRDU", points=points)


class CzmlExportTests(unittest.TestCase):
    def test_uses_geometric_altitude_for_waypoints(self) -> None:
        flight = trajectory_to_czml_flight(_approach_trajectory(), airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON)

        self.assertIsNotNone(flight)
        assert flight is not None
        self.assertEqual(flight["id"], "TST123")
        self.assertEqual(flight["altitude_source"], "opensky_history_geoaltitude_m")
        # Final waypoint altitude is the geometric altitude (500), not baro (520).
        self.assertEqual(flight["waypoints"][-1][3], 500.0)

    def test_keeps_trajectory_at_matching_runway_threshold(self) -> None:
        threshold = RunwayThreshold("KRDU", "23R", AIRPORT_LAT, AIRPORT_LON, 124.0, 225.0)

        flight = trajectory_to_czml_flight(
            _approach_trajectory(), airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON, runway_threshold=threshold
        )

        self.assertIsNotNone(flight)
        assert flight is not None
        self.assertEqual(flight["runway"], "23R")

    def test_rejects_trajectory_at_different_threshold(self) -> None:
        far_threshold = RunwayThreshold("KRDU", "05L", 35.0, -78.0, 124.0, 45.0)

        flight = trajectory_to_czml_flight(
            _approach_trajectory(), airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON, runway_threshold=far_threshold
        )

        self.assertIsNone(flight)

    def test_rejects_distant_trajectory(self) -> None:
        far = Trajectory(
            icao24="zzz",
            callsign="FAR",
            dep_airport=None,
            arr_airport=None,
            points=[
                TrajectoryPoint(0, 10.0, 10.0, 9000.0, 9000.0, None, False),
                TrajectoryPoint(60, 10.1, 10.1, 9000.0, 9000.0, None, False),
            ],
        )

        self.assertIsNone(trajectory_to_czml_flight(far, airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON))


if __name__ == "__main__":
    unittest.main()
