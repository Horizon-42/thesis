from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from opensky_data_query.download_adsblol_history import (
    SplitFileReader,
    candidate_tags,
    parse_day,
)


class DownloadAdsblolHistoryTests(unittest.TestCase):
    def test_parse_day_accepts_dash_and_dot_dates(self) -> None:
        self.assertEqual(parse_day("2026-04-19").isoformat(), "2026-04-19")
        self.assertEqual(parse_day("2026.04.19").isoformat(), "2026-04-19")

    def test_candidate_tags_match_release_naming(self) -> None:
        day = parse_day("2026-04-19")

        self.assertEqual(
            candidate_tags(day, "prod"),
            [
                "v2026.04.19-planes-readsb-prod-0",
                "v2026.04.19-planes-readsb-prod-1",
            ],
        )
        self.assertEqual(candidate_tags(day, "staging"), ["v2026.04.19-planes-readsb-staging-0"])

    def test_split_file_reader_reads_files_as_one_stream(self) -> None:
        with TemporaryDirectory() as tmp:
            part1 = Path(tmp) / "part.aa"
            part2 = Path(tmp) / "part.ab"
            part1.write_bytes(b"abc")
            part2.write_bytes(b"def")

            reader = SplitFileReader([part1, part2])
            try:
                self.assertEqual(reader.read(4), b"abcd")
                self.assertEqual(reader.read(4), b"ef")
                self.assertEqual(reader.read(4), b"")
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
