from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trajectory_data_process.build_arrivals import (
    discover_airports,
    landing_files,
    merge_landing_flights,
)


def _flight(fid: str, icao: str, t: str) -> dict:
    return {"id": fid, "callsign": fid, "type": "UNK", "icao24": icao, "landing_time_utc": t, "waypoints": [[0, 0, 0, 0], [1, 0, 0, 1]]}


class MergeTests(unittest.TestCase):
    def _write(self, d: Path, code: str, ident: str, flights: list[dict]) -> Path:
        path = d / f"{code}_{ident}_landings.json"
        path.write_text(json.dumps(flights), encoding="utf-8")
        return path

    def test_merges_and_dedupes_across_files_without_touching_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            a = self._write(d, "KRDU", "23R", [_flight("AAL1", "a1", "2026-06-18T10:00:00Z")])
            # same landing duplicated in another file + a namesake that is a different landing
            b = self._write(d, "KRDU", "05L", [
                _flight("AAL1", "a1", "2026-06-18T10:00:00Z"),   # duplicate -> dropped
                _flight("AAL1", "b2", "2026-06-18T11:00:00Z"),   # namesake -> kept AS-IS
            ])

            merged = merge_landing_flights([a, b])

            self.assertEqual(len(merged), 2)
            # ids pass through untouched: positional _N re-uniquing gave one flight
            # different ids in different views. Uniqueness lives in flight_key
            # (icao24 + landing time), which CZML generation derives entity ids from.
            self.assertEqual([m["id"] for m in merged], ["AAL1", "AAL1"])
            self.assertEqual({m["icao24"] for m in merged}, {"a1", "b2"})

    def test_landing_files_all_vs_subset(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d, "KRDU", "23R", [_flight("A", "a", "t")])
            self._write(d, "KRDU", "05L", [_flight("B", "b", "t")])

            self.assertEqual(len(landing_files(d, "KRDU", None)), 2)
            self.assertEqual([p.name for p in landing_files(d, "KRDU", ["23R"])], ["KRDU_23R_landings.json"])

    def test_missing_requested_runway_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                landing_files(Path(tmp), "KRDU", ["99X"])

    def test_discover_airports_finds_dirs_with_landing_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "KRDU").mkdir()
            self._write(root / "KRDU", "KRDU", "23R", [_flight("A", "a", "t")])
            (root / "KSJC").mkdir()
            self._write(root / "KSJC", "KSJC", "30L", [_flight("B", "b", "t")])
            (root / "empty").mkdir()  # no *_landings.json -> not discovered

            self.assertEqual(discover_airports(root), ["KRDU", "KSJC"])

    def test_discover_airports_missing_root(self) -> None:
        self.assertEqual(discover_airports(Path("/tmp/does-not-exist-xyz")), [])


if __name__ == "__main__":
    unittest.main()
