from __future__ import annotations

import copy
import unittest

from trajectory_data_process.processing.altitude_matching import (
    build_dual_altitude_points,
    parse_state_altitude_samples,
)


class AltitudeMatchingTests(unittest.TestCase):
    def test_parse_state_altitude_samples_uses_geo_altitude_field(self) -> None:
        payload = {
            "states": [
                [
                    "ABC123",
                    "TST123",
                    "Canada",
                    1004,
                    1005,
                    -119.31,
                    49.98,
                    1500.0,
                    False,
                    120.0,
                    210.0,
                    0.0,
                    None,
                    1468.2,
                ],
                ["ABC123", "BAD", "Canada", 1010, 1010, -119.0, 50.0, 1500.0, False, None, None, None, None, None],
            ]
        }

        samples = parse_state_altitude_samples(payload, source_response_id="sha256:state")

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].icao24, "abc123")
        self.assertEqual(samples[0].baro_altitude_m, 1500.0)
        self.assertEqual(samples[0].geo_altitude_m, 1468.2)
        self.assertEqual(samples[0].source_response_id, "sha256:state")

    def test_build_dual_altitude_points_matches_nearest_state_without_mutating_track(self) -> None:
        track = {
            "icao24": "abc123",
            "path": [
                [1000, 49.98, -119.31, 1500.0, 210.0, False],
                [1020, 49.97, -119.32, 1490.0, 211.0, False],
            ],
        }
        original_path = copy.deepcopy(track["path"])
        samples = parse_state_altitude_samples(
            {
                "states": [
                    ["abc123", None, None, 1004, 1004, None, None, 1501.0, False, None, None, None, None, 1468.2],
                    ["abc123", None, None, 1030, 1030, None, None, 1492.0, False, None, None, None, None, 1458.0],
                ]
            },
            source_response_id="sha256:state",
        )

        result = build_dual_altitude_points(track, samples, max_age_sec=15)

        self.assertEqual(track["path"], original_path)
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["points"][0]["baro_altitude_m"], 1500.0)
        self.assertEqual(result["points"][0]["geo_altitude_m"], 1468.2)
        self.assertEqual(result["points"][0]["geo_altitude_match"]["delta_t_sec"], 4.0)
        self.assertEqual(result["quality"]["points_with_both_altitudes"], 2)

    def test_missing_geo_altitude_is_quarantined_not_filled_from_baro(self) -> None:
        track = {
            "icao24": "abc123",
            "path": [
                [1000, 49.98, -119.31, 1500.0, 210.0, False],
                [1100, 49.97, -119.32, 1490.0, 211.0, False],
            ],
        }
        samples = parse_state_altitude_samples(
            {
                "states": [
                    ["abc123", None, None, 1002, 1002, None, None, 1501.0, False, None, None, None, None, 1468.2],
                ]
            }
        )

        result = build_dual_altitude_points(track, samples, max_age_sec=15, require_geo_altitude=True)

        self.assertEqual(len(result["points"]), 1)
        self.assertEqual(result["points"][0]["geo_altitude_m"], 1468.2)
        self.assertNotEqual(result["points"][0]["geo_altitude_m"], result["points"][0]["baro_altitude_m"])
        self.assertEqual(result["quality"]["points_total"], 2)
        self.assertEqual(result["quality"]["points_quarantined_missing_geo_altitude"], 1)


if __name__ == "__main__":
    unittest.main()
