"""The published decision altitude: read off the plates, stored in the configuration.

The CIFP has no minima record at all, so this is the one runway fact in the pipeline
that comes from a PDF. These tests pin the two things that can go wrong silently: a
column misread on a plate, and the decision altitude being confused with the height the
plate publishes it against (which is above the TOUCHDOWN ZONE, not the threshold).
"""

from __future__ import annotations

import json
import tempfile
import shutil
import unittest
from pathlib import Path

from geokit import FT_M

from trajectory_data_process.extract_approach_minima import (
    _LNAV_VNAV_LABEL,
    _LPV,
    _labelled_row,
    _touchdown_zone_elevation,
    minima_for_airport,
    read_airport_plates,
    read_plate,
)
from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.approach_minima import (
    NO_VERTICAL_MINIMA,
    PublishedMinima,
    RUNWAY_THRESHOLDS_SCHEMA,
    no_vertical_minima,
)

CIFP = Path("data/CIFP/CIFP_260806/FAACIFP18")
CONFIG = Path("trajectory_data_process/config/runway_thresholds.json")
CHARTS = Path("data/RNAV_CHARTS")
FLEET = ("KRDU", "KSJC", "KSTL", "KSMF", "KMSY")

# The three thresholds in the fleet with no LPV, and why. KRDU 32 and KSMF 35R publish
# Baro-VNAV minima, which AIM 5-4-5 f 5 also puts a missed approach point at; KRDU 14
# has no RNAV procedure at all, in the plates or in the CIFP (TD9).
NOT_LPV = {"KRDU 32": "lnav_vnav", "KSMF 35R": "lnav_vnav", "KRDU 14": NO_VERTICAL_MINIMA}
# MUST match ``manoeuvre.instructions.ALTITUDE_BIN_M / 2`` (1000 ft bins, so 500 ft).
# Mirrored rather than imported because the harvest suite does not put the ts_transformer
# tree on its path. Every published decision altitude in this fleet falls inside word 0,
# which is why a go-around's timing cannot be judged from the word (two-tier plan D75).
ALTITUDE_WORD_HALF_BIN_FT = 500.0

# Hand-read off the plates, one per shape the parser has to handle: a plain LPV sheet, a
# sidestep sheet that prints two TDZEs, the lowest and highest LPV in the fleet, a TDZE
# below sea level, and the two runways whose only vertical guidance is Baro-VNAV.
BY_HAND = {
    ("KRDU", "05L"): ("lpv", 598, 214, 384),
    ("KSJC", "30L"): ("lpv", 257, 200, 57),     # sidestep sheet: also prints TDZE 30R 55
    ("KSTL", "12L"): ("lpv", 951, 410, 541),
    ("KMSY", "20"): ("lpv", 249, 250, -1),      # TDZE below sea level
    ("KRDU", "32"): ("lnav_vnav", 820, 391, 429),
    ("KSMF", "35R"): ("lnav_vnav", 311, 287, 24),
}


class PlateParse(unittest.TestCase):
    def test_the_hand_read_plates_still_parse_to_the_hand_read_figures(self) -> None:
        """Six sheets read by eye, against what the parser makes of them.

        The parser's own DA - height == TDZE check cannot catch a wrong ROW, because
        every row on a plate satisfies it (KRDU 05L: 598-214, 748-364 and 840-456 all
        give 384). Only an independent reading can, which is what this is.
        """
        for (code, ident), expected in BY_HAND.items():
            plate = read_airport_plates(CHARTS / code)[ident]
            actual = (
                plate.service,
                plate.decision_altitude_ft_msl,
                plate.decision_height_above_touchdown_ft,
                plate.touchdown_zone_elevation_ft,
            )
            self.assertEqual(actual, expected, f"{code} {ident}")

    def test_the_whole_fleet_parses_and_the_service_split_is_the_published_one(self) -> None:
        services: dict[str, str | None] = {}
        for code in FLEET:
            for runway, plate in read_airport_plates(CHARTS / code).items():
                services[f"{code} {runway}"] = plate.service
        self.assertEqual(len(services), 25)      # KRDU 14 has no plate at all
        self.assertEqual(
            {name: service for name, service in services.items() if service != "lpv"},
            {"KRDU 32": "lnav_vnav", "KSMF 35R": "lnav_vnav"},
        )

    def test_rnp_plates_are_not_read(self) -> None:
        """RNAV (RNP) Z minima are a different service on a different authorisation."""
        rnp = CHARTS / "KRDU" / "00516RRZ5L.PDF"
        self.assertTrue(rnp.exists())
        self.assertIsNone(read_plate(rnp))

    def test_two_gps_plates_for_one_runway_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("A.PDF", "B.PDF"):
                shutil.copy(CHARTS / "KRDU" / "00516RY5L.PDF", directory / name)
            with self.assertRaisesRegex(ValueError, "two RNAV \\(GPS\\) plates"):
                read_airport_plates(directory)

    def test_a_plate_for_an_unlisted_threshold_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "the configuration does not list"):
            minima_for_airport("KRDU", CHARTS, ["05L"])


class StoredMinima(unittest.TestCase):
    def test_the_fleet_reads_back_through_load_airport(self) -> None:
        guided = 0
        for code in FLEET:
            for runway in load_airport(code, config_file=CONFIG, cifp_file=CIFP).runways:
                expected = NOT_LPV.get(f"{code} {runway.ident}", "lpv")
                self.assertEqual(runway.published_minima.service, expected, runway.ident)
                guided += runway.published_minima.vertically_guided
        self.assertEqual(guided, 25)

    def test_a_threshold_without_minima_raises_rather_than_defaulting(self) -> None:
        runway = load_airport("KRDU", config_file=CONFIG, cifp_file=CIFP).runway("14")
        self.assertFalse(runway.published_minima.vertically_guided)
        with self.assertRaisesRegex(ValueError, "no vertically guided minima"):
            runway.decision_height_above_threshold_m

    def test_height_above_threshold_is_not_the_plate_height_above_touchdown(self) -> None:
        """The two reference points differ, and using the wrong one is a silent error."""
        runway = load_airport("KRDU", config_file=CONFIG, cifp_file=CIFP).runway("05L")
        self.assertAlmostEqual(runway.published_minima.decision_height_above_touchdown_ft, 214.0)
        self.assertAlmostEqual(runway.decision_height_above_threshold_m / FT_M, 231.2, places=1)

    def test_the_threshold_is_never_above_the_touchdown_zone(self) -> None:
        """The slope runs one way: the landing threshold is at or below the TDZE, so the
        height above the threshold is at or above the plate's. The gap reaches 23.9 ft
        (KSTL 29), which is why the two are not interchangeable."""
        gaps = [
            runway.decision_height_above_threshold_m / FT_M
            - runway.published_minima.decision_height_above_touchdown_ft
            for code in FLEET
            for runway in load_airport(code, config_file=CONFIG, cifp_file=CIFP).runways
            if runway.published_minima.vertically_guided
        ]
        self.assertGreater(min(gaps), -0.5)      # -0.5 ft covers the plate's whole-foot rounding
        self.assertAlmostEqual(max(gaps), 23.9, places=1)

    def test_every_decision_altitude_sits_inside_altitude_word_zero(self) -> None:
        highest = max(
            runway.decision_height_above_threshold_m / FT_M
            for code in FLEET
            for runway in load_airport(code, config_file=CONFIG, cifp_file=CIFP).runways
            if runway.published_minima.vertically_guided
        )
        self.assertAlmostEqual(highest, 423.4, places=1)   # KSTL 12L
        self.assertLess(423.4, ALTITUDE_WORD_HALF_BIN_FT)

    def test_an_older_configuration_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "runway_thresholds.json"
            config = json.loads(CONFIG.read_text(encoding="utf-8"))
            config["schema_version"] = "runway-thresholds-v2"
            stale.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, RUNWAY_THRESHOLDS_SCHEMA):
                load_airport("KRDU", config_file=stale, cifp_file=CIFP)


class RowSelection(unittest.TestCase):
    """The parse's three silent-skip paths, each now a loud failure."""

    # The shape that matters: the label is printed on its own line, the figures under it.
    STACKED = ["LPV", "DA 598/18 214 (200-12 )", "LNAV/", "DA 748/35 364 (400-58 )", "VNAV"]

    def test_a_stacked_lpv_label_is_read_as_lpv_not_as_the_row_below_it(self) -> None:
        self.assertEqual(_labelled_row(self.STACKED, _LPV), (598, 214))
        self.assertEqual(_labelled_row(self.STACKED, _LNAV_VNAV_LABEL), (748, 364))

    def test_the_lnav_mda_row_is_never_taken_for_a_decision_altitude(self) -> None:
        self.assertIsNone(_labelled_row(["LNAV MDA 840/24 456 (500-114 )"], _LNAV_VNAV_LABEL))

    def test_a_sidestep_plate_uses_this_runways_labelled_tdze(self) -> None:
        lines = ["ELEV 62 D TDZE 30L 57", "TDZE 30R 55"]
        self.assertEqual(_touchdown_zone_elevation(Path("x.PDF"), lines, "30L"), 57)
        self.assertEqual(_touchdown_zone_elevation(Path("x.PDF"), lines, "30R"), 55)

    def test_two_unlabelled_candidates_raise_rather_than_letting_the_check_choose(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot tell which TDZE"):
            _touchdown_zone_elevation(Path("x.PDF"), ["TDZE 57", "TDZE 55"], "30L")

    def test_a_tdze_ending_a_line_does_not_pair_with_the_next_lines_digit(self) -> None:
        """KRDU 32 prints "ELEV 435 TDZE" then "0" from the airport diagram below it."""
        self.assertEqual(
            _touchdown_zone_elevation(Path("x.PDF"), ["TDZE 429", "ELEV 435 TDZE", "0"], "32"),
            429,
        )


class MinimaContract(unittest.TestCase):
    def test_a_hand_edited_config_with_inconsistent_figures_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not the plate's TDZE"):
            PublishedMinima("lpv", 598.0, 200.0, 384.0, "c.PDF", "p", "a", "eff", None)

    def test_no_minima_carries_a_reason_and_no_figures(self) -> None:
        with self.assertRaisesRegex(ValueError, "must say why"):
            no_vertical_minima("   ")
        with self.assertRaisesRegex(ValueError, "carries no plate figures"):
            PublishedMinima(NO_VERTICAL_MINIMA, 598.0, None, None, None, None, None, None, "why")

    def test_incomplete_published_minima_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            PublishedMinima("lpv", 598.0, 214.0, 384.0, None, "p", "a", "eff", None)

    def test_from_config_requires_every_key(self) -> None:
        with self.assertRaises(KeyError):
            PublishedMinima.from_config({"service": "lpv"})


if __name__ == "__main__":
    unittest.main()
