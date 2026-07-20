"""A landing belongs to ONE runway, not to every parallel runway near it.

Each threshold used to be classified independently, so a landing on KSJC 30L -- 250 m from
30R and on the identical heading, both inside the 1000 m capture radius -- was written into
both runways' files. It showed up downstream as an observed lateral error whose median WAS
the parallel separation: KSTL 30L 397 m, KSJC 30R 234 m.
"""

from __future__ import annotations

import math
import unittest

from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.processing.czml_export import (
    _parallel_thresholds,
    _wins_against_parallel_runways,
    classify_landing_flights,
)
from trajectory_data_process.trajectory import Trajectory, TrajectoryPoint

# KSJC, from runways.csv: 30L and 30R are parallel (heading 318.8 / 319.0), ~250 m apart.
# 12R is the OTHER END of 30L's runway -- 180 deg away, and a full rollout stops on top of it.
K30L = RunwayThreshold(airport="KSJC", ident="30L", lat=37.35100, lon=-121.91700,
                       elevation_m=17.37, heading_deg=318.8)
K30R = RunwayThreshold(airport="KSJC", ident="30R", lat=37.35230, lon=-121.91500,
                       elevation_m=16.76, heading_deg=319.0)
K12R = RunwayThreshold(airport="KSJC", ident="12R", lat=37.37370, lon=-121.94200,
                       elevation_m=11.58, heading_deg=138.8)
K11 = RunwayThreshold(airport="KSJC", ident="11", lat=37.36590, lon=-121.93700,
                      elevation_m=12.80, heading_deg=139.0)
SIBLINGS = [K30L, K30R, K12R, K11]


def _approach_to(threshold: RunwayThreshold, *, n: int = 60) -> Trajectory:
    """A straight 3 deg approach down ``threshold``'s centreline, ending at the threshold."""
    course = math.radians(threshold.heading_deg)
    points = []
    for i in range(n):
        back_m = (n - 1 - i) * 100.0
        # Walk back UP the approach: opposite the landing direction.
        dlat = -back_m * math.cos(course) / METRES_PER_DEG_LAT
        dlon = -back_m * math.sin(course) / metres_per_deg_lon(threshold.lat)
        points.append(
            TrajectoryPoint(
                time=1_700_000_000 + i * 4,
                lat=threshold.lat + dlat,
                lon=threshold.lon + dlon,
                geo_altitude_m=threshold.elevation_m + 15.0 + back_m * math.tan(math.radians(3.0)),
                baro_altitude_m=None,
                heading_deg=threshold.heading_deg,
                on_ground=False,
            )
        )
    return Trajectory(
        icao24="abc123", callsign="TEST01", dep_airport=None, arr_airport="KSJC", points=points
    )


class ParallelSelectionTest(unittest.TestCase):
    def test_parallel_runway_is_a_competitor(self) -> None:
        idents = [t.ident for t in _parallel_thresholds(K30L, SIBLINGS, heading_tolerance_deg=20.0)]
        self.assertIn("30R", idents)

    def test_opposite_end_of_the_same_runway_is_not_a_competitor(self) -> None:
        """A full rollout ends on 12R's threshold and would win any distance comparison."""
        idents = [t.ident for t in _parallel_thresholds(K30L, SIBLINGS, heading_tolerance_deg=20.0)]
        self.assertNotIn("12R", idents)

    def test_crossing_runway_is_not_a_competitor(self) -> None:
        idents = [t.ident for t in _parallel_thresholds(K30L, SIBLINGS, heading_tolerance_deg=20.0)]
        self.assertNotIn("11", idents)

    def test_threshold_never_competes_with_itself(self) -> None:
        idents = [t.ident for t in _parallel_thresholds(K30L, SIBLINGS, heading_tolerance_deg=20.0)]
        self.assertNotIn("30L", idents)

    def test_missing_heading_yields_no_competitors_rather_than_crashing(self) -> None:
        headless = RunwayThreshold("KSJC", "30L", 37.351, -121.917, 17.37, None)
        self.assertEqual(_parallel_thresholds(headless, SIBLINGS, heading_tolerance_deg=20.0), [])


class ArbitrationTest(unittest.TestCase):
    def test_a_30l_approach_wins_for_30l(self) -> None:
        points = _approach_to(K30L).points
        self.assertTrue(_wins_against_parallel_runways(points, K30L, [K30R]))

    def test_a_30l_approach_loses_for_30r(self) -> None:
        points = _approach_to(K30L).points
        self.assertFalse(_wins_against_parallel_runways(points, K30R, [K30L]))

    def test_no_competitors_means_always_wins(self) -> None:
        points = _approach_to(K30L).points
        self.assertTrue(_wins_against_parallel_runways(points, K30L, []))


class ClassificationTest(unittest.TestCase):
    def _classify(self, threshold, siblings):
        accepted, _ = classify_landing_flights(
            [_approach_to(K30L)],
            airport_lat=37.3626,
            airport_lon=-121.9290,
            runway_threshold=threshold,
            sibling_thresholds=siblings,
            exclude_ground=False,
        )
        return accepted

    def test_landing_is_accepted_by_its_own_runway(self) -> None:
        self.assertEqual(len(self._classify(K30L, SIBLINGS)), 1)

    def test_same_landing_is_rejected_by_the_parallel_runway(self) -> None:
        self.assertEqual(len(self._classify(K30R, SIBLINGS)), 0)

    def test_without_siblings_the_bug_reproduces(self) -> None:
        """Guards the wiring: an unthreaded caller silently restores double-assignment."""
        self.assertEqual(len(self._classify(K30R, [])), 1)


if __name__ == "__main__":
    unittest.main()
