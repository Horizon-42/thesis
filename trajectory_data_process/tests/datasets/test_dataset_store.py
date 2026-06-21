from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from trajectory_data_process.datasets.dataset_store import (
    load_jsonl_records,
    partition_path,
    write_jsonl_records,
)


class DatasetStoreTests(unittest.TestCase):
    def test_partition_path_supports_hour_and_day(self) -> None:
        ts = datetime(2026, 4, 15, 13, 22, tzinfo=timezone.utc)
        root = Path("/tmp/out")

        hourly = partition_path(root, "raw_tracks", airport="krdu", timestamp=ts)
        daily = partition_path(root, "raw_tracks", airport="krdu", timestamp=ts, granularity="day")

        self.assertEqual(
            hourly,
            root / "raw_tracks" / "v3" / "airport=KRDU" / "year=2026" / "month=04" / "day=15" / "hour=13",
        )
        self.assertEqual(
            daily,
            root / "raw_tracks" / "v3" / "airport=KRDU" / "year=2026" / "month=04" / "day=15",
        )

    def test_write_and_load_jsonl_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "airport_events" / "events.jsonl"
            count = write_jsonl_records(path, [{"b": 2, "a": 1}, {"id": "x"}])

            self.assertEqual(count, 2)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ['{"a": 1, "b": 2}', '{"id": "x"}'],
            )
            self.assertEqual(load_jsonl_records(path), [{"a": 1, "b": 2}, {"id": "x"}])

    def test_load_jsonl_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_jsonl_records(Path("/tmp/does-not-exist.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
