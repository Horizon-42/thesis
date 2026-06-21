from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trajectory_data_process.acquisition.runways import resolve_runway_threshold


HEADER = (
    '"id","airport_ref","airport_ident","length_ft","width_ft","surface","lighted","closed",'
    '"le_ident","le_latitude_deg","le_longitude_deg","le_elevation_ft","le_heading_degT","le_displaced_threshold_ft",'
    '"he_ident","he_latitude_deg","he_longitude_deg","he_elevation_ft","he_heading_degT","he_displaced_threshold_ft"'
)
ROW = (
    '242724,3844,"KRDU",10000,150,"CON",1,0,'
    '"05L",35.8745002746582,-78.802001953125,367,45,,'
    '"23R",35.893798828125,-78.77799987792969,409,225,'
)


def _csv(tmp: str) -> Path:
    path = Path(tmp) / "runways.csv"
    path.write_text(HEADER + "\n" + ROW + "\n", encoding="utf-8")
    return path


class RunwayThresholdTests(unittest.TestCase):
    def test_resolves_low_end(self) -> None:
        with TemporaryDirectory() as tmp:
            t = resolve_runway_threshold("KRDU", "05L", _csv(tmp))
        self.assertEqual(t.ident, "05L")
        self.assertAlmostEqual(t.lat, 35.8745002746582)
        self.assertAlmostEqual(t.lon, -78.802001953125)
        self.assertAlmostEqual(t.elevation_m, 367 * 0.3048, places=3)
        self.assertEqual(t.heading_deg, 45.0)

    def test_resolves_high_end(self) -> None:
        with TemporaryDirectory() as tmp:
            t = resolve_runway_threshold("krdu", "23R", _csv(tmp))
        self.assertEqual(t.ident, "23R")
        self.assertAlmostEqual(t.lat, 35.893798828125)
        self.assertEqual(t.heading_deg, 225.0)

    def test_unknown_runway_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                resolve_runway_threshold("KRDU", "99X", _csv(tmp))


if __name__ == "__main__":
    unittest.main()
