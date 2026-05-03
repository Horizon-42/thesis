from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from opensky_data_query.opensky_history_db import (
    AIRPORT_HISTORY_COLUMNS,
    HISTORY_COLUMNS,
    normalize_history_dataframe,
    write_history_rows,
)


class OpenSkyHistoryDbTests(unittest.TestCase):
    def test_normalize_history_dataframe_sorts_by_icao24_and_time(self) -> None:
        df = pd.DataFrame(
            [
                {"icao24": "b", "time": "2026-04-19T10:00:10Z", "lat": 2.0},
                {"icao24": "a", "time": "2026-04-19T10:00:20Z", "lat": 3.0},
                {"icao24": "a", "time": "2026-04-19T10:00:00Z", "lat": 1.0},
            ]
        )

        out = normalize_history_dataframe(df)

        self.assertEqual(out["icao24"].tolist(), ["a", "a", "b"])
        self.assertEqual(out["lat"].tolist(), [1.0, 3.0, 2.0])
        self.assertTrue(str(out["time"].dtype).startswith("datetime64"))

    def test_write_history_rows_uses_history_partition(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-04-19T10:00:00Z"),
                    "icao24": "abc123",
                    "lat": 35.9,
                    "lon": -78.8,
                    "baroaltitude": 1000.0,
                    "geoaltitude": 980.0,
                }
            ]
        )

        with TemporaryDirectory() as tmp:
            path = write_history_rows(
                Path(tmp),
                airport="krdu",
                df=df,
                fetched_at=datetime(2026, 5, 3, 13, tzinfo=timezone.utc),
            )

            self.assertEqual(path.name, "rows.jsonl")
            self.assertIn("history_rows/v2/airport=KRDU/year=2026/month=05/day=03/hour=13", str(path))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["icao24"], "abc123")
            self.assertEqual(rows[0]["geoaltitude"], 980.0)

    def test_write_history_rows_preserves_returned_row_order(self) -> None:
        df = pd.DataFrame(
            [
                {"time": pd.Timestamp("2026-04-19T10:00:10Z"), "icao24": "b"},
                {"time": pd.Timestamp("2026-04-19T10:00:00Z"), "icao24": "a"},
            ]
        )

        with TemporaryDirectory() as tmp:
            path = write_history_rows(
                Path(tmp),
                airport="krdu",
                df=df,
                fetched_at=datetime(2026, 5, 3, 13, tzinfo=timezone.utc),
                query_name="airport_ops",
            )

            self.assertEqual(path.name, "airport_ops.jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["icao24"] for row in rows], ["b", "a"])

    def test_history_columns_include_both_altitudes(self) -> None:
        self.assertIn("baroaltitude", HISTORY_COLUMNS)
        self.assertIn("geoaltitude", HISTORY_COLUMNS)

    def test_airport_history_columns_request_flights_table_airports(self) -> None:
        self.assertIn("FlightsData4.estdepartureairport", AIRPORT_HISTORY_COLUMNS)
        self.assertIn("FlightsData4.estarrivalairport", AIRPORT_HISTORY_COLUMNS)

    def test_normalize_history_dataframe_canonicalizes_flights_columns(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "time": "2026-04-19T10:00:00Z",
                    "icao24": "ABC123",
                    "FlightsData4.estArrivalAirport": "KRDU",
                }
            ]
        )

        out = normalize_history_dataframe(df)

        self.assertIn("estarrivalairport", out.columns)
        self.assertEqual(out.loc[0, "icao24"], "abc123")
        self.assertEqual(out.loc[0, "estarrivalairport"], "KRDU")


if __name__ == "__main__":
    unittest.main()
