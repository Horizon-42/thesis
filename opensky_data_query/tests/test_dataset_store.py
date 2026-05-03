from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from opensky_data_query.dataset_store import (
    find_cached_source_response,
    partition_path,
    sha256_text,
    write_jsonl_records,
    write_source_response,
)


class DatasetStoreTests(unittest.TestCase):
    def test_partition_path_supports_hour_and_day(self) -> None:
        ts = datetime(2026, 4, 15, 13, 22, tzinfo=timezone.utc)
        root = Path("/tmp/out")

        hourly = partition_path(root, "raw_tracks", airport="cylw", timestamp=ts)
        daily = partition_path(root, "raw_tracks", airport="cylw", timestamp=ts, granularity="day")

        self.assertEqual(
            hourly,
            root / "raw_tracks" / "v2" / "airport=CYLW" / "year=2026" / "month=04" / "day=15" / "hour=13",
        )
        self.assertEqual(
            daily,
            root / "raw_tracks" / "v2" / "airport=CYLW" / "year=2026" / "month=04" / "day=15",
        )

    def test_write_source_response_preserves_body_and_indexes_it(self) -> None:
        body = '{"states":[["abc123",null]],"time":1776257000}\n'
        fetched_at = datetime(2026, 4, 15, 13, 0, 2, tzinfo=timezone.utc)

        with TemporaryDirectory() as tmp:
            record = write_source_response(
                Path(tmp),
                airport="CYLW",
                fetched_at=fetched_at,
                endpoint="/states/all",
                params={"icao24": "abc123", "time": 1776257000},
                body_text=body,
                http_status=200,
            )

            body_path = Path(record["body_full_path"])
            self.assertEqual(body_path.read_text(encoding="utf-8"), body)
            self.assertEqual(record["body_sha256"], sha256_text(body))
            self.assertEqual(record["source_id"], f"sha256:{sha256_text(body)}")

            index_path = body_path.parent / "source_index.jsonl"
            rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["body_path"], body_path.name)
            self.assertNotIn("body_full_path", rows[0])

    def test_write_jsonl_records_appends_sorted_json_objects(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "airport_events" / "events.jsonl"
            count = write_jsonl_records(path, [{"b": 2, "a": 1}, {"id": "x"}])

            self.assertEqual(count, 2)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ['{"a": 1, "b": 2}', '{"id": "x"}'],
            )

    def test_find_cached_source_response_returns_matching_body(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = write_source_response(
                root,
                airport="CYLW",
                fetched_at=datetime(2026, 4, 15, 13, 0, tzinfo=timezone.utc),
                endpoint="/tracks/all",
                params={"icao24": "abc123", "time": 1000},
                body_text='{"path":[]}',
                http_status=200,
            )

            cached = find_cached_source_response(
                root,
                airport="cylw",
                endpoint="/tracks/all",
                params={"time": 1000, "icao24": "abc123"},
            )

            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertEqual(cached["source_id"], written["source_id"])
            self.assertEqual(Path(cached["body_full_path"]).read_text(encoding="utf-8"), '{"path":[]}')


if __name__ == "__main__":
    unittest.main()
