from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trajectory_data_process.acquisition.airports import resolve_airport_profile


def _aeroviz_root(tmp: str) -> Path:
    root = Path(tmp)
    data_dir = root / "public" / "data" / "common"
    data_dir.mkdir(parents=True)
    (data_dir / "airports.csv").write_text(
        "ident,gps_code,icao_code,latitude_deg,longitude_deg,elevation_ft\n"
        "KRDU,KRDU,KRDU,35.8776,-78.7875,435\n",
        encoding="utf-8",
    )
    return root


class AirportProfileTests(unittest.TestCase):
    def test_resolves_center_and_elevation(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = resolve_airport_profile("krdu", _aeroviz_root(tmp))
        self.assertEqual(profile.code, "KRDU")
        self.assertAlmostEqual(profile.lat, 35.8776)
        self.assertAlmostEqual(profile.elevation_m, 435 * 0.3048, places=3)

    def test_unknown_airport_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                resolve_airport_profile("ZZZZ", _aeroviz_root(tmp))


if __name__ == "__main__":
    unittest.main()
