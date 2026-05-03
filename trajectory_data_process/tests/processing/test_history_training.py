from __future__ import annotations

import unittest

import pandas as pd

from trajectory_data_process.processing.history_training import (
    build_training_records_from_history,
    history_group_to_dual_altitude_points,
    history_group_to_track,
    iter_history_track_segments,
)


class HistoryTrainingTests(unittest.TestCase):
    def test_history_group_to_track_uses_baroaltitude_path(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "time": "2026-04-19T10:00:00Z",
                    "icao24": "abc123",
                    "lat": 0.10,
                    "lon": 0.0,
                    "baroaltitude": 1000.0,
                    "geoaltitude": 980.0,
                    "heading": 180.0,
                    "onground": False,
                    "callsign": "TST123",
                }
            ]
        )

        track = history_group_to_track(df)

        self.assertEqual(track["icao24"], "abc123")
        self.assertEqual(track["callsign"], "TST123")
        self.assertEqual(track["path"][0][3], 1000.0)

    def test_history_group_to_dual_altitude_points_keeps_both_altitudes(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "time": "2026-04-19T10:00:00Z",
                    "icao24": "abc123",
                    "lat": 0.10,
                    "lon": 0.0,
                    "baroaltitude": 1000.0,
                    "geoaltitude": 980.0,
                    "heading": 180.0,
                    "onground": False,
                }
            ]
        )

        points = history_group_to_dual_altitude_points(df)

        self.assertEqual(points[0]["baro_altitude_m"], 1000.0)
        self.assertEqual(points[0]["geo_altitude_m"], 980.0)
        self.assertEqual(points[0]["geo_altitude_match"]["method"], "same_history_row")

    def test_dual_altitude_raw_index_matches_filtered_track_path(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "time": "2026-04-19T10:00:00Z",
                    "icao24": "abc123",
                    "lat": 0.10,
                    "lon": 0.0,
                    "baroaltitude": None,
                    "geoaltitude": 900.0,
                },
                {
                    "time": "2026-04-19T10:00:10Z",
                    "icao24": "abc123",
                    "lat": 0.08,
                    "lon": 0.0,
                    "baroaltitude": 1000.0,
                    "geoaltitude": 980.0,
                },
            ]
        )

        track = history_group_to_track(df)
        points = history_group_to_dual_altitude_points(df)

        self.assertEqual(len(track["path"]), 1)
        self.assertEqual(points[0]["raw_index"], 0)
        self.assertEqual(points[0]["time"], track["path"][0][0])

    def test_iter_history_track_segments_splits_large_time_gaps(self) -> None:
        df = pd.DataFrame(
            [
                {"time": pd.Timestamp("2026-04-19T10:00:00Z"), "icao24": "abc123", "lat": 0.1, "lon": 0.0},
                {"time": pd.Timestamp("2026-04-19T10:05:00Z"), "icao24": "abc123", "lat": 0.1, "lon": 0.0},
                {"time": pd.Timestamp("2026-04-19T10:40:00Z"), "icao24": "abc123", "lat": 0.1, "lon": 0.0},
            ]
        )

        segments = iter_history_track_segments(df, max_gap_sec=900)

        self.assertEqual([len(segment) for segment in segments], [2, 1])

    def test_build_training_records_from_history_emits_pass_event(self) -> None:
        rows = []
        for t, lat, baro, geo in [
            ("2026-04-19T10:00:00Z", 0.10, 2000.0, 1980.0),
            ("2026-04-19T10:00:10Z", 0.06, 1900.0, 1880.0),
            ("2026-04-19T10:00:20Z", 0.02, 1850.0, 1830.0),
            ("2026-04-19T10:00:30Z", 0.06, 1900.0, 1880.0),
            ("2026-04-19T10:00:40Z", 0.10, 2000.0, 1980.0),
        ]:
            rows.append(
                {
                    "time": t,
                    "icao24": "abc123",
                    "lat": lat,
                    "lon": 0.0,
                    "baroaltitude": baro,
                    "geoaltitude": geo,
                    "heading": 180.0,
                    "onground": False,
                    "callsign": "TST123",
                    "estdepartureairport": "KAAA",
                    "estarrivalairport": "KBBB",
                }
            )
        df = pd.DataFrame(rows)

        result = build_training_records_from_history(
            df,
            airport="KRDU",
            airport_lat=0.0,
            airport_lon=0.0,
            airport_elev_m=0.0,
            radius_nm=5.0,
            low_altitude_agl_m=600.0,
        )

        self.assertEqual(len(result["raw_tracks"]), 1)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["label"], "pass")
        self.assertEqual(result["events"][0]["quality"]["points_with_both_altitudes"], 3)
        self.assertEqual(result["raw_tracks"][0]["source"]["endpoint"], "traffic.data.opensky.history")
        self.assertEqual(result["quarantine"], [])


if __name__ == "__main__":
    unittest.main()
