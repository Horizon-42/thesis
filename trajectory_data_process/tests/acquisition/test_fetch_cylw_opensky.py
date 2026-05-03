from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from trajectory_data_process.acquisition.fetch_cylw_opensky import (
    _retry_after_seconds,
    _tracks_all_too_old,
    find_cached_track_payload,
)


class FetchOpenSkyTests(unittest.TestCase):
    def test_retry_after_seconds_uses_numeric_header(self) -> None:
        self.assertEqual(_retry_after_seconds("12", fallback=90.0), 12.0)

    def test_retry_after_seconds_falls_back_for_missing_or_invalid_header(self) -> None:
        self.assertEqual(_retry_after_seconds(None, fallback=90.0), 90.0)
        self.assertEqual(_retry_after_seconds("not-a-number", fallback=90.0), 90.0)

    def test_tracks_all_too_old_uses_thirty_day_window(self) -> None:
        day = 24 * 3600
        now = 100 * day

        self.assertFalse(_tracks_all_too_old(int(now - 29 * day), now=now))
        self.assertFalse(_tracks_all_too_old(0, now=now))
        self.assertTrue(_tracks_all_too_old(int(now - 31 * day), now=now))

    def test_find_cached_track_payload_reads_legacy_raw_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "krdu_raw_20260419T225418Z.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "airport": "KRDU",
                        "tracks": [
                            {
                                "icao24": "ac92b1",
                                "startTime": 1776600191,
                                "endTime": 1776607479,
                                "path": [[1776603203, 35.9, -78.8, 1000, 180, False]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            cached = find_cached_track_payload(root, airport="KRDU", icao24="AC92B1", t_ref=1776603203)

            self.assertIsNotNone(cached)
            assert cached is not None
            track, source_ids = cached
            self.assertEqual(track["icao24"], "ac92b1")
            self.assertEqual(source_ids, ["legacy_raw_json:krdu_raw_20260419T225418Z.json"])


if __name__ == "__main__":
    unittest.main()
