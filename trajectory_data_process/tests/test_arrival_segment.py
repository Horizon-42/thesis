"""Arrival-segment truncation: ring entry, hysteresis, local/takeoff classification."""

from __future__ import annotations

import math
import unittest

from trajectory_data_process.arrival_segment import (
    ENTRY_RADIUS_KM,
    GROUND_START_AGL_M,
    arrival_segment,
    truncate_flights,
)

AIRPORT_LAT, AIRPORT_LON = 35.878659, -78.7873
FIELD_ELEVATION_M = 0.0
# ~1 km of longitude at this latitude, for building tracks by distance.
KM_LON = 1.0 / (111.32 * math.cos(math.radians(AIRPORT_LAT)))


def _wp(t: float, dist_km: float, alt: float = 1000.0) -> list[float]:
    """A waypoint ``dist_km`` due east of the airport: [t, lon, lat, alt]."""
    return [t, AIRPORT_LON + dist_km * KM_LON, AIRPORT_LAT, alt]


def _inbound(t0: float, d0: float, d1: float, n: int, dt: float = 10.0) -> list[list[float]]:
    return [_wp(t0 + i * dt, d0 + (d1 - d0) * i / max(1, n - 1)) for i in range(n)]


def _climb_out(
    t0: float, d0: float, d1: float, n: int, alt1: float, dt: float = 10.0
) -> list[list[float]]:
    """A departure leg: starts at field elevation and climbs to ``alt1``."""
    return [
        _wp(
            t0 + i * dt,
            d0 + (d1 - d0) * i / max(1, n - 1),
            FIELD_ELEVATION_M + (alt1 - FIELD_ELEVATION_M) * i / max(1, n - 1),
        )
        for i in range(n)
    ]


class ArrivalSegmentTests(unittest.TestCase):
    def test_plain_arrival_cut_at_the_final_ring_entry(self) -> None:
        # 29 km -> 0 km straight in: the pre-ring stretch (>25 km) is dropped, so
        # every arrival shares the ring as its entry boundary; times rebase to 0.
        track = _inbound(0.0, 29.0, 0.0, 30)
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
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
        kind, segment = arrival_segment(
            outbound + inbound, AIRPORT_LAT, AIRPORT_LON,
            field_elevation_m=FIELD_ELEVATION_M,
        )
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
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual(kind, "arrival")
        self.assertEqual(len(segment), len(track))  # kept whole

    def test_local_circuit_never_leaving_the_ring(self) -> None:
        # Takeoff, a 12 km pattern, landing — never beyond 25 km: a local, not an arrival.
        track = _inbound(0.0, 0.5, 12.0, 10) + _inbound(100.0, 12.0, 0.0, 10)
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
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
        arrivals, locals_, takeoffs = truncate_flights(
            flights, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )

        self.assertEqual([f["id"] for f in arrivals], ["AAL1"])
        self.assertEqual([f["id"] for f in locals_], ["N123"])
        self.assertEqual(takeoffs, [])
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
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual(kind, "arrival")
        self.assertEqual(len(segment), len(track))


class SatelliteFieldTakeoffTests(unittest.TestCase):
    """A takeoff from a NEIGHBOURING field inside the ring is not an arrival.

    The local-circuit test measures distance from the DESTINATION, so a departure
    9 km away passes it; on the real fleet that admitted 75 such flights, 64 of
    them at KSJC (KRHV 7 km, KNUQ 11 km, KPAO 21 km).
    """

    def test_ground_start_at_a_neighbouring_field_is_a_takeoff(self) -> None:
        # Departs a field 9 km east, climbs to 1500 m, turns back and lands.
        # Never leaves the 25 km ring and starts >5 km out, so the ring geometry
        # alone would have called it a coverage-limited arrival.
        track = _climb_out(0.0, 9.0, 20.0, 20, 1500.0) + _inbound(300.0, 20.0, 0.0, 25)
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual(kind, "takeoff")
        self.assertEqual(segment, track)            # untouched, for the review file

    def test_the_test_is_altitude_not_speed(self) -> None:
        # A jet at rotation reads 71-80 m/s on the runway, inside the approach-speed
        # range, so only the altitude separates the two populations. Rebuilding the
        # same departure with a fast first sample must still classify as a takeoff.
        track = _climb_out(0.0, 9.0, 20.0, 20, 1500.0) + _inbound(300.0, 20.0, 0.0, 25)
        track[0] = _wp(track[0][0], 9.0, FIELD_ELEVATION_M)
        track[1] = _wp(track[1][0], 8.2, FIELD_ELEVATION_M)   # 80 m/s down the runway
        kind, _ = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual(kind, "takeoff")

    def test_a_low_arrival_above_the_threshold_is_kept(self) -> None:
        # Reception begins airborne at 9 km, just above the ground band: a genuine
        # coverage-limited arrival, which the ground test must not take.
        track = _inbound(0.0, 9.0, 0.0, 20)
        for row in track:
            row[3] = FIELD_ELEVATION_M + GROUND_START_AGL_M + 1.0
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual(kind, "arrival")
        self.assertEqual(len(segment), len(track))

    def test_a_departure_that_leaves_the_ring_keeps_its_arrival(self) -> None:
        # Same departure, but it flies out past the ring before returning. The cut
        # already removed the takeoff, so what remains IS an arrival — the ground
        # test reads the segment the cut produced, never the raw track.
        track = _climb_out(0.0, 9.0, 28.0, 25, 3000.0) + _inbound(400.0, 28.0, 0.0, 30)
        kind, segment = arrival_segment(
            track, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual(kind, "arrival")
        self.assertLess(len(segment), len(track))
        self.assertEqual(segment[0][0], 0.0)

    def test_truncate_flights_routes_takeoffs_to_their_own_bucket(self) -> None:
        takeoff_track = _climb_out(0.0, 9.0, 20.0, 20, 1500.0) + _inbound(300.0, 20.0, 0.0, 25)
        flights = [
            {"id": "AAL1", "landing_time_utc": "2026-06-18T10:10:00Z",
             "waypoints": _inbound(0.0, 29.0, 0.0, 30)},
            {"id": "N456", "landing_time_utc": "2026-06-18T11:00:00Z",
             "waypoints": takeoff_track},
        ]
        arrivals, locals_, takeoffs = truncate_flights(
            flights, AIRPORT_LAT, AIRPORT_LON, field_elevation_m=FIELD_ELEVATION_M
        )
        self.assertEqual([f["id"] for f in arrivals], ["AAL1"])
        self.assertEqual(locals_, [])
        self.assertEqual([f["id"] for f in takeoffs], ["N456"])
        # returned unmodified, like a local circuit
        self.assertNotIn("arrival_truncated", takeoffs[0])
