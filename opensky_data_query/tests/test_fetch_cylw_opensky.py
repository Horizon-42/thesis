from __future__ import annotations

import unittest

from opensky_data_query.fetch_cylw_opensky import _retry_after_seconds, _tracks_all_too_old


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


if __name__ == "__main__":
    unittest.main()
