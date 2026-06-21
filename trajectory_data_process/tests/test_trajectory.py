from __future__ import annotations

import unittest

import pandas as pd

from trajectory_data_process.trajectory import build_trajectories_from_history


def _row(t: str, lat: float, lon: float, baro=1000.0, geo=980.0, **extra) -> dict:
    base = {
        "time": t,
        "icao24": "abc123",
        "lat": lat,
        "lon": lon,
        "baroaltitude": baro,
        "geoaltitude": geo,
        "heading": 180.0,
        "onground": False,
        "callsign": "TST123",
    }
    base.update(extra)
    return base


class TrajectoryBuildTests(unittest.TestCase):
    def test_builds_single_trajectory_with_both_altitudes(self) -> None:
        df = pd.DataFrame([_row("2026-04-19T10:00:00Z", 0.10, 0.0), _row("2026-04-19T10:00:10Z", 0.08, 0.0)])

        trajectories = build_trajectories_from_history(df)

        self.assertEqual(len(trajectories), 1)
        traj = trajectories[0]
        self.assertEqual(traj.icao24, "abc123")
        self.assertEqual(traj.callsign, "TST123")
        self.assertEqual(traj.points[0].geo_altitude_m, 980.0)
        self.assertEqual(traj.points[0].baro_altitude_m, 1000.0)
        self.assertEqual(traj.geo_altitude_coverage, 1.0)

    def test_splits_on_large_time_gap(self) -> None:
        df = pd.DataFrame(
            [
                _row("2026-04-19T10:00:00Z", 0.1, 0.0),
                _row("2026-04-19T10:05:00Z", 0.1, 0.0),
                _row("2026-04-19T10:40:00Z", 0.1, 0.0),
                _row("2026-04-19T10:40:10Z", 0.1, 0.0),
            ]
        )

        trajectories = build_trajectories_from_history(df, max_gap_sec=900)

        self.assertEqual([len(t.points) for t in trajectories], [2, 2])

    def test_drops_points_without_geo_altitude_when_required(self) -> None:
        df = pd.DataFrame(
            [
                _row("2026-04-19T10:00:00Z", 0.10, 0.0, geo=None),
                _row("2026-04-19T10:00:10Z", 0.08, 0.0, geo=980.0),
                _row("2026-04-19T10:00:20Z", 0.06, 0.0, geo=970.0),
            ]
        )

        trajectories = build_trajectories_from_history(df, require_geo_altitude=True)

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(len(trajectories[0].points), 2)
        self.assertTrue(all(p.geo_altitude_m is not None for p in trajectories[0].points))

    def test_missing_geoaltitude_column_raises_when_required(self) -> None:
        df = pd.DataFrame(
            [{"time": "2026-04-19T10:00:00Z", "icao24": "abc123", "lat": 0.1, "lon": 0.0, "baroaltitude": 1000.0}]
        )

        with self.assertRaises(ValueError):
            build_trajectories_from_history(df, require_geo_altitude=True)


if __name__ == "__main__":
    unittest.main()
