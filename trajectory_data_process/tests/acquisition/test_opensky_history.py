from __future__ import annotations

import unittest

import pandas as pd

from trajectory_data_process.acquisition.opensky_history import (
    AIRPORT_HISTORY_COLUMNS,
    history_rows_as_records,
)


class OpenSkyHistoryTests(unittest.TestCase):
    def test_geoaltitude_is_a_requested_column(self) -> None:
        self.assertIn("geoaltitude", AIRPORT_HISTORY_COLUMNS)
        self.assertIn("baroaltitude", AIRPORT_HISTORY_COLUMNS)

    def test_history_rows_as_records_is_json_safe(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-04-19T10:00:00Z"),
                    "icao24": "abc123",
                    "lat": 35.5,
                    "lon": -78.5,
                    "geoaltitude": 980.0,
                    "baroaltitude": float("nan"),
                }
            ]
        )

        records = history_rows_as_records(df)

        self.assertEqual(records[0]["time"], "2026-04-19T10:00:00Z")
        self.assertEqual(records[0]["icao24"], "abc123")
        self.assertIsNone(records[0]["baroaltitude"])
        self.assertEqual(records[0]["geoaltitude"], 980.0)


if __name__ == "__main__":
    unittest.main()
