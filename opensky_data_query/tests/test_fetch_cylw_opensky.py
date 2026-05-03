from __future__ import annotations

import unittest

from opensky_data_query.fetch_cylw_opensky import _retry_after_seconds


class FetchOpenSkyTests(unittest.TestCase):
    def test_retry_after_seconds_uses_numeric_header(self) -> None:
        self.assertEqual(_retry_after_seconds("12", fallback=90.0), 12.0)

    def test_retry_after_seconds_falls_back_for_missing_or_invalid_header(self) -> None:
        self.assertEqual(_retry_after_seconds(None, fallback=90.0), 90.0)
        self.assertEqual(_retry_after_seconds("not-a-number", fallback=90.0), 90.0)


if __name__ == "__main__":
    unittest.main()

