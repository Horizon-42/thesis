from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from trajectory_data_process.acquisition.download_opensky_history import (
    build_download_tasks,
    iter_time_chunks,
    run_download_for_airport,
)


class DownloadOpenSkyHistoryTests(unittest.TestCase):
    def test_iter_time_chunks_keeps_partial_final_chunk(self) -> None:
        start = datetime(2026, 4, 19, 10, tzinfo=timezone.utc)
        stop = datetime(2026, 4, 19, 12, 30, tzinfo=timezone.utc)

        chunks = iter_time_chunks(start, stop, chunk_hours=1)

        self.assertEqual(
            chunks,
            [
                (
                    datetime(2026, 4, 19, 10, tzinfo=timezone.utc),
                    datetime(2026, 4, 19, 11, tzinfo=timezone.utc),
                ),
                (
                    datetime(2026, 4, 19, 11, tzinfo=timezone.utc),
                    datetime(2026, 4, 19, 12, tzinfo=timezone.utc),
                ),
                (
                    datetime(2026, 4, 19, 12, tzinfo=timezone.utc),
                    datetime(2026, 4, 19, 12, 30, tzinfo=timezone.utc),
                ),
            ],
        )

    def test_build_download_tasks_adds_terminal_area_after_airport_ops(self) -> None:
        start = datetime(2026, 4, 19, 10, tzinfo=timezone.utc)
        stop = datetime(2026, 4, 19, 12, tzinfo=timezone.utc)

        tasks = build_download_tasks(
            airport="krdu",
            start=start,
            stop=stop,
            fetch_profile="terminal_all",
            chunk_hours=1,
            run_id="run",
            bbox_lat_pad=0.5,
            bbox_lon_pad=0.6,
            airport_lat=35.878659,
            airport_lon=-78.7873,
        )

        self.assertEqual([task.query for task in tasks], ["airport_ops", "terminal_area", "airport_ops", "terminal_area"])
        self.assertEqual(tasks[0].airport_filter, "KRDU")
        self.assertIsNone(tasks[1].airport_filter)
        self.assertEqual(len(tasks[1].bounds or ()), 4)
        for actual, expected in zip(tasks[1].bounds or (), (-79.3873, 35.378659, -78.1873, 36.378659)):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(tasks[0].query_name.startswith("airport_ops_20260419T100000Z_20260419T110000Z_run"))

    def test_run_download_for_airport_writes_rows_and_manifest(self) -> None:
        calls: list[dict[str, Any]] = []

        def fake_fetcher(**kwargs: Any) -> pd.DataFrame:
            calls.append(kwargs)
            return pd.DataFrame(
                [
                    {
                        "time": kwargs["start"],
                        "icao24": "a4f7cb",
                        "lat": 35.8816,
                        "lon": -78.7932,
                        "baroaltitude": 182.88,
                        "geoaltitude": 129.54,
                    }
                ]
            )

        with TemporaryDirectory() as tmp:
            manifest = run_download_for_airport(
                airport="krdu",
                start=datetime(2026, 4, 19, 10, tzinfo=timezone.utc),
                stop=datetime(2026, 4, 19, 11, tzinfo=timezone.utc),
                output_root=Path(tmp),
                fetch_profile="airport_ops",
                chunk_hours=1,
                bbox_lat_pad=0.45,
                bbox_lon_pad=0.60,
                aeroviz_root=Path(tmp),
                cached=True,
                run_id="test_run",
                created_at=datetime(2026, 5, 22, 8, 30, tzinfo=timezone.utc),
                fetcher=fake_fetcher,
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["airport"], "KRDU")
            self.assertIsNone(calls[0]["bounds"])
            self.assertTrue(calls[0]["cached"])
            self.assertEqual(manifest["history_rows"]["count"], 1)

            output = manifest["history_rows"]["outputs"][0]
            rows_path = Path(output["path"])
            rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["icao24"], "a4f7cb")
            self.assertEqual(rows[0]["geoaltitude"], 129.54)

            manifest_path = Path(manifest["manifest_path"])
            self.assertTrue(manifest_path.exists())
            written = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(written["run_id"], "test_run")
            self.assertEqual(written["history_rows"]["outputs"][0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
