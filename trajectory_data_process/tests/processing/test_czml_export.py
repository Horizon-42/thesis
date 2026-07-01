from __future__ import annotations

import unittest

from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.processing.czml_export import trajectory_to_czml_flight
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint


AIRPORT_LAT, AIRPORT_LON = 35.8776, -78.7875


# Final approach to runway 23R (heading 225°): the ground track runs south-west into
# the threshold, so the geometry agrees with the runway heading.
def _approach_trajectory() -> Trajectory:
    coords = [
        (0, 35.9176, -78.7381, 1500.0),
        (60, 35.8976, -78.7628, 1000.0),
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


THRESHOLD = RunwayThreshold("KRDU", "23R", AIRPORT_LAT, AIRPORT_LON, 124.0, 225.0)


def _landing_trajectory() -> Trajectory:
    coords = [(0, 35.9176, -78.7381, 900.0), (30, 35.8976, -78.7628, 500.0), (60, AIRPORT_LAT, AIRPORT_LON, 150.0)]
    points = [TrajectoryPoint(t, lat, lon, geo, geo + 20, 225.0, False) for t, lat, lon, geo in coords]
    return Trajectory(icao24="abc123", callsign="LND1", dep_airport="KJFK", arr_airport="KRDU", points=points)


def _departure_trajectory() -> Trajectory:
    coords = [(0, AIRPORT_LAT, AIRPORT_LON, 150.0), (30, 35.8976, -78.7628, 500.0), (60, 35.9176, -78.7381, 900.0)]
    points = [TrajectoryPoint(t, lat, lon, geo, geo + 20, 45.0, False) for t, lat, lon, geo in coords]
    return Trajectory(icao24="dep999", callsign="DEP1", dep_airport="KRDU", arr_airport="KJFK", points=points)


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

    def test_landing_only_keeps_descent_to_threshold(self) -> None:
        flight = trajectory_to_czml_flight(
            _landing_trajectory(), airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON,
            runway_threshold=THRESHOLD, landing_only=True,
        )

        self.assertIsNotNone(flight)
        assert flight is not None
        self.assertEqual(flight["runway"], "23R")
        self.assertTrue(flight["landing_time_utc"].endswith("Z"))
        # Aligned with the 225° runway heading -> accepted.
        self.assertTrue(flight["heading_ok"])
        self.assertAlmostEqual(flight["approach_course_deg"], 225.0, delta=2.0)
        self.assertAlmostEqual(flight["course_error_deg"], 0.0, delta=2.0)

    def test_landing_only_flags_heading_mismatch_from_geometry(self) -> None:
        # Descends onto the 23R threshold but the ground track is due south (~180°),
        # 45° off the 225° runway heading — even though the ADS-B track *claims* 225°.
        # Geometry catches the misalignment: the landing is tagged, not dropped, so it
        # can be reviewed for a false kill.
        coords = [(0, 35.9376, AIRPORT_LON, 900.0), (30, 35.9076, AIRPORT_LON, 500.0), (60, AIRPORT_LAT, AIRPORT_LON, 150.0)]
        points = [TrajectoryPoint(t, lat, lon, geo, geo + 20, 225.0, False) for t, lat, lon, geo in coords]
        traj = Trajectory(icao24="mis1", callsign="MIS1", dep_airport=None, arr_airport=None, points=points)

        flight = trajectory_to_czml_flight(
            traj, airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON,
            runway_threshold=THRESHOLD, landing_only=True,
        )

        self.assertIsNotNone(flight)
        assert flight is not None
        self.assertFalse(flight["heading_ok"])
        self.assertEqual(flight["runway"], "23R")
        self.assertAlmostEqual(flight["approach_course_deg"], 180.0, delta=1.0)
        self.assertAlmostEqual(flight["course_error_deg"], 45.0, delta=1.0)
        # The ADS-B track alone would have accepted it — geometry is what rejects it.
        self.assertAlmostEqual(flight["track_error_deg"], 0.0, delta=0.1)

    def test_landing_only_crops_waypoints_to_radius(self) -> None:
        # An early point ~58 km from the airport, then a normal SW approach. A 30 km
        # crop keeps only the in-radius points (up to the anchor).
        coords = [
            (0, 36.30, -78.40, 3000.0),          # ~58 km out -> cropped
            (60, 35.9176, -78.7381, 900.0),
            (90, 35.8976, -78.7628, 500.0),
            (120, AIRPORT_LAT, AIRPORT_LON, 150.0),
        ]
        points = [TrajectoryPoint(t, lat, lon, geo, geo + 20, 225.0, False) for t, lat, lon, geo in coords]
        traj = Trajectory(icao24="rad1", callsign="RAD1", dep_airport="KJFK", arr_airport="KRDU", points=points)

        cropped = trajectory_to_czml_flight(
            traj, airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON,
            runway_threshold=THRESHOLD, landing_only=True, crop_radius_km=30.0,
        )
        full = trajectory_to_czml_flight(
            traj, airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON,
            runway_threshold=THRESHOLD, landing_only=True, crop_radius_km=200.0,
        )

        self.assertIsNotNone(cropped)
        assert cropped is not None
        self.assertEqual(len(cropped["waypoints"]), 3)   # the 58 km point is dropped
        self.assertEqual(len(full["waypoints"]), 4)      # a wide radius keeps it
        self.assertTrue(cropped["heading_ok"])           # cropping does not affect the direction test

    def test_landing_only_rejects_high_overflight(self) -> None:
        # A jet descending at altitude that merely clips the threshold laterally
        # (aligned heading, prior descent) must not count as a landing.
        coords = [(0, 35.9176, -78.7381, 9000.0), (30, 35.8976, -78.7628, 8500.0), (60, AIRPORT_LAT, AIRPORT_LON, 8000.0)]
        points = [TrajectoryPoint(t, lat, lon, geo, geo + 20, 225.0, False) for t, lat, lon, geo in coords]
        overflight = Trajectory(
            icao24="ovf1", callsign="OVF1", dep_airport=None, arr_airport=None, points=points
        )

        flight = trajectory_to_czml_flight(
            overflight, airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON,
            runway_threshold=THRESHOLD, landing_only=True,
        )

        self.assertIsNone(flight)

    def test_landing_only_rejects_departure(self) -> None:
        flight = trajectory_to_czml_flight(
            _departure_trajectory(), airport_lat=AIRPORT_LAT, airport_lon=AIRPORT_LON,
            runway_threshold=THRESHOLD, landing_only=True,
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
