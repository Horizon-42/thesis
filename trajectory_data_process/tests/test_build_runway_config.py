"""The runway config must publish LANDING thresholds, not pavement ends.

Taking the pavement end where a runway has a displaced threshold put the approach target
short by the displacement -- 775 m at KSJC 30L/30R, a 40.6 m altitude error on a 3 deg
glidepath -- which corrupted both the optimizer target and the evaluation gates.
"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from geokit import FT_M, haversine_m

from trajectory_data_process.acquisition.runways import (
    _pavement_end,
    landing_thresholds_from_row,
    resolve_runway_threshold,
)

# KSJC 12R/30L, verbatim from OurAirports runways.csv: 30L is displaced 2542 ft (775 m).
KSJC_12R_30L = {
    "le_ident": "12R",
    "le_latitude_deg": "37.37369918823242",
    "le_longitude_deg": "-121.94200134277344",
    "le_elevation_ft": "38",
    "le_heading_degT": "138.8",
    "le_displaced_threshold_ft": "1302",
    "he_ident": "30L",
    "he_latitude_deg": "37.35100173950195",
    "he_longitude_deg": "-121.91699981689453",
    "he_elevation_ft": "57",
    "he_heading_degT": "318.8",
    "he_displaced_threshold_ft": "2542",
}

# KRDU 05R/23L: no displacement anywhere, so the thresholds must stay at the pavement ends.
KRDU_05R_23L = {
    "le_ident": "05R",
    "le_latitude_deg": "35.864601135253906",
    "le_longitude_deg": "-78.79730224609375",
    "le_elevation_ft": "397",
    "le_heading_degT": "45",
    "le_displaced_threshold_ft": "",
    "he_ident": "23L",
    "he_latitude_deg": "35.87919998168945",
    "he_longitude_deg": "-78.77940368652344",
    "he_elevation_ft": "431",
    "he_heading_degT": "225",
    "he_displaced_threshold_ft": "",
}


def _by_ident(row: dict[str, str]) -> dict[str, dict]:
    return {t["ident"]: t for t in landing_thresholds_from_row(row)}


class DisplacedThresholdTest(unittest.TestCase):
    def test_displaced_threshold_moves_down_the_centreline_by_the_displacement(self) -> None:
        thresholds = _by_ident(KSJC_12R_30L)
        pavement = _pavement_end(KSJC_12R_30L, "he")
        moved = haversine_m(
            pavement["lat"], pavement["lon"], thresholds["30L"]["lat"], thresholds["30L"]["lon"]
        )
        self.assertAlmostEqual(moved, 2542 * FT_M, delta=1.0)
        self.assertAlmostEqual(thresholds["30L"]["displaced_threshold_m"], 774.8, places=1)

    def test_displacement_moves_toward_the_far_end_not_away(self) -> None:
        """A displaced threshold is INSIDE the runway; moving the wrong way doubles the error."""
        thresholds = _by_ident(KSJC_12R_30L)
        far = _pavement_end(KSJC_12R_30L, "le")
        pavement = _pavement_end(KSJC_12R_30L, "he")
        full = haversine_m(pavement["lat"], pavement["lon"], far["lat"], far["lon"])
        remaining = haversine_m(
            thresholds["30L"]["lat"], thresholds["30L"]["lon"], far["lat"], far["lon"]
        )
        self.assertLess(remaining, full)

    def test_elevation_is_interpolated_along_the_runway_slope(self) -> None:
        """30L end is 57 ft, 12R end 38 ft; 775 m in, the surface has dropped."""
        thresholds = _by_ident(KSJC_12R_30L)
        pavement_elevation_m = 57 * FT_M
        self.assertLess(thresholds["30L"]["elevation_m"], pavement_elevation_m)
        expected = (57 + (38 - 57) * (2542 / 11013.0)) * FT_M
        self.assertAlmostEqual(thresholds["30L"]["elevation_m"], expected, delta=0.3)

    def test_heading_is_unchanged_by_displacement(self) -> None:
        self.assertEqual(_by_ident(KSJC_12R_30L)["30L"]["heading_deg"], 318.8)

    def test_both_ends_of_one_runway_are_displaced_independently(self) -> None:
        thresholds = _by_ident(KSJC_12R_30L)
        self.assertAlmostEqual(thresholds["12R"]["displaced_threshold_m"], 396.8, places=1)
        self.assertAlmostEqual(thresholds["30L"]["displaced_threshold_m"], 774.8, places=1)

    def test_undisplaced_runway_keeps_the_pavement_end_exactly(self) -> None:
        thresholds = _by_ident(KRDU_05R_23L)
        for ident, side in (("05R", "le"), ("23L", "he")):
            pavement = _pavement_end(KRDU_05R_23L, side)
            self.assertEqual(thresholds[ident]["displaced_threshold_m"], 0.0)
            self.assertAlmostEqual(thresholds[ident]["lat"], pavement["lat"], places=6)
            self.assertAlmostEqual(thresholds[ident]["lon"], pavement["lon"], places=6)
            self.assertAlmostEqual(
                thresholds[ident]["elevation_m"], round(pavement["elevation_m"], 2), places=2
            )

    def test_missing_displacement_column_is_zero_not_a_crash(self) -> None:
        row = dict(KRDU_05R_23L)
        del row["le_displaced_threshold_ft"]
        del row["he_displaced_threshold_ft"]
        self.assertEqual(_by_ident(row)["23L"]["displaced_threshold_m"], 0.0)

    def test_glidepath_altitude_error_the_bug_caused(self) -> None:
        """The regression this guards: 775 m short on a 3 deg path is ~40 m of altitude."""
        displacement_m = _by_ident(KSJC_12R_30L)["30L"]["displaced_threshold_m"]
        self.assertAlmostEqual(displacement_m * math.tan(math.radians(3.0)), 40.6, delta=0.5)

    def test_zero_length_with_displacement_raises_rather_than_silently_undisplacing(self) -> None:
        """A row whose ends coincide cannot place a displaced threshold; guessing hides 40 m."""
        row = dict(KSJC_12R_30L)
        row["le_latitude_deg"] = row["he_latitude_deg"]
        row["le_longitude_deg"] = row["he_longitude_deg"]
        with self.assertRaises(ValueError):
            landing_thresholds_from_row(row)


class ResolverAgreementTest(unittest.TestCase):
    """The --runway download path and the config generator must name the SAME point.

    They used to diverge: the generator learned displaced thresholds while
    resolve_runway_threshold kept pavement ends — 775 m apart at KSJC 30L, which also
    shifts the landing anchor and hence landing_time_utc, the flight_key identity field.
    """

    def test_resolver_returns_the_displaced_landing_threshold(self) -> None:
        row = dict(KSJC_12R_30L, airport_ident="KSJC")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "runways.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            resolved = resolve_runway_threshold("KSJC", "30L", path)
        published = _by_ident(KSJC_12R_30L)["30L"]
        self.assertAlmostEqual(resolved.lat, published["lat"], places=7)
        self.assertAlmostEqual(resolved.lon, published["lon"], places=7)
        self.assertAlmostEqual(resolved.elevation_m, published["elevation_m"], places=2)
        self.assertEqual(resolved.heading_deg, published["heading_deg"])


if __name__ == "__main__":
    unittest.main()
