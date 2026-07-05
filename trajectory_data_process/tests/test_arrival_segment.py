"""Arrival-segment truncation: ring entry, hysteresis, local classification."""

from __future__ import annotations

import math
import unittest

from trajectory_data_process.arrival_segment import (
    ENTRY_RADIUS_KM,
    arrival_segment,
    truncate_flights,
)

AIRPORT_LAT, AIRPORT_LON = 35.878659, -78.7873
# ~1 km of longitude at this latitude, for building tracks by distance.
KM_LON = 1.0 / (111.32 * math.cos(math.radians(AIRPORT_LAT)))


def _wp(t: float, dist_km: float, alt: float = 1000.0) -> list[float]:
    """A waypoint ``dist_km`` due east of the airport: [t, lon, lat, alt]."""
    return [t, AIRPORT_LON + dist_km * KM_LON, AIRPORT_LAT, alt]


def _inbound(t0: float, d0: float, d1: float, n: int, dt: float = 10.0) -> list[list[float]]:
    return [_wp(t0 + i * dt, d0 + (d1 - d0) * i / max(1, n - 1)) for i in range(n)]


class ArrivalSegmentTests(unittest.TestCase):
    def test_plain_arrival_cut_at_the_final_ring_entry(self) -> None:
        # 29 km -> 0 km straight in: the pre-ring stretch (>25 km) is dropped, so
        # every arrival shares the ring as its entry boundary; times rebase to 0.
        track = _inbound(0.0, 29.0, 0.0, 30)
        kind, segment = arrival_segment(track, AIRPORT_LAT, AIRPORT_LON)
        self.assertEqual(kind, "arrival")
        self.assertLess(len(segment), len(track))
        self.assertEqual(segment[0][0], 0.0)                       # rebased
        first_dist = abs(segment[0][1] - AIRPORT_LON) / KM_LON
        self.assertLessEqual(first_dist, ENTRY_RADIUS_KM)          # starts just inside the ring
        self.assertGreater(first_dist, ENTRY_RADIUS_KM - 2.0)      # ...not deep inside it

    def test_depart_and_return_keeps_only_the_final_inbound(self) -> None:
        # Takeoff at the field -> out to 28 km (outside the ring) -> back in to land.
        outbound = _inbound(0.0, 0.5, 28.0, 20)
        inbound = _inbound(200.0, 28.0, 0.0, 20)
        kind, segment = arrival_segment(outbound + inbound, AIRPORT_LAT, AIRPORT_LON)
        self.assertEqual(kind, "arrival")
        # the segment starts inside the ring on the RETURN leg (t rebased to 0,
        # strictly shorter than the full loop) and ends at the field
        self.assertEqual(segment[0][0], 0.0)
        self.assertLess(len(segment), 40)
        dists = [abs(w[1] - AIRPORT_LON) / KM_LON for w in segment]
        self.assertLessEqual(dists[0], ENTRY_RADIUS_KM)
        self.assertLess(dists[-1], 0.6)
        # monotone inbound: no outbound samples survived
        self.assertTrue(all(d <= ENTRY_RADIUS_KM + 1e-6 for d in dists))

    def test_single_outlier_beyond_the_ring_does_not_cut(self) -> None:
        # One jittery fix at 26 km in an otherwise inside-the-ring final approach:
        # below the hysteresis, so the track is NOT cut there.
        track = _inbound(0.0, 24.0, 10.0, 10)
        track[4] = _wp(track[4][0], 26.0)          # lone outlier
        track += _inbound(100.0, 10.0, 0.0, 10)
        kind, segment = arrival_segment(track, AIRPORT_LAT, AIRPORT_LON)
        self.assertEqual(kind, "arrival")
        self.assertEqual(len(segment), len(track))  # kept whole

    def test_local_circuit_never_leaving_the_ring(self) -> None:
        # Takeoff, a 12 km pattern, landing — never beyond 25 km: a local, not an arrival.
        track = _inbound(0.0, 0.5, 12.0, 10) + _inbound(100.0, 12.0, 0.0, 10)
        kind, segment = arrival_segment(track, AIRPORT_LAT, AIRPORT_LON)
        self.assertEqual(kind, "local")
        self.assertEqual(segment, track)            # untouched, for the review file


class TruncateFlightsTests(unittest.TestCase):
    def test_splits_and_annotates(self) -> None:
        arrival_track = _inbound(0.0, 29.0, 0.0, 30)
        local_track = _inbound(0.0, 0.5, 12.0, 10) + _inbound(100.0, 12.0, 0.0, 10)
        flights = [
            {"id": "AAL1", "landing_time_utc": "2026-06-18T10:10:00Z", "waypoints": arrival_track},
            {"id": "N123", "landing_time_utc": "2026-06-18T11:00:00Z", "waypoints": local_track},
        ]
        arrivals, locals_ = truncate_flights(flights, AIRPORT_LAT, AIRPORT_LON)

        self.assertEqual([f["id"] for f in arrivals], ["AAL1"])
        self.assertEqual([f["id"] for f in locals_], ["N123"])
        kept = arrivals[0]
        self.assertTrue(kept["arrival_truncated"])
        self.assertEqual(kept["cut_samples"], len(arrival_track) - len(kept["waypoints"]))
        self.assertEqual(kept["arrival_duration_s"], kept["waypoints"][-1][0])
        # entry time = landing time − segment duration
        expected_entry_offset = 600 - kept["arrival_duration_s"]  # 10:10 landing
        self.assertEqual(
            kept["entry_time_utc"],
            f"2026-06-18T10:{int(expected_entry_offset // 60):02d}:{int(expected_entry_offset % 60):02d}Z",
        )
        # the raw input flight dict is not mutated
        self.assertNotIn("arrival_truncated", flights[0])
        self.assertEqual(len(flights[0]["waypoints"]), len(arrival_track))


if __name__ == "__main__":
    unittest.main()


class CoverageLimitedArrivalTests(unittest.TestCase):
    def test_never_outside_but_starting_far_from_the_field_is_an_arrival(self) -> None:
        # An ADS-B gap ate the pre-ring stretch: the track starts at 18 km (inside
        # the ring) and descends in — an arrival, NOT a local circuit.
        track = _inbound(0.0, 18.0, 0.0, 20)
        kind, segment = arrival_segment(track, AIRPORT_LAT, AIRPORT_LON)
        self.assertEqual(kind, "arrival")
        self.assertEqual(len(segment), len(track))
