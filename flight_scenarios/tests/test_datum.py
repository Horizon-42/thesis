"""The observed altitude datum must be converted exactly once, on the way into modeling.

Observed ADS-B altitude is ellipsoidal (HAE); thresholds and gates are MSL. Left alone the
gap is ~30 m, which passed 0.5 % of real KRDU landings through a +/-3 m vertical gate.
"""

from __future__ import annotations

import unittest

from flight_scenarios.datum import (
    HAE_ALTITUDE_SOURCE,
    MSL_ALTITUDE_SOURCE,
    flight_to_msl,
    flights_to_msl,
    geoid_undulation_m,
    waypoints_to_msl,
)

KRDU_LAT, KRDU_LON = 35.8792, -78.7794
KRDU_UNDULATION_M = -33.53


def _flight(**over):
    base = {
        "id": "TEST123",
        "icao24": "abc123",
        "runway": "23L",
        "altitude_source": HAE_ALTITUDE_SOURCE,
        "waypoints": [
            [0.0, KRDU_LON, KRDU_LAT, 1000.0],
            [1.0, KRDU_LON, KRDU_LAT, 900.0],
        ],
    }
    base.update(over)
    return base


class GeoidTest(unittest.TestCase):
    def test_undulation_matches_published_egm96(self) -> None:
        (n,) = geoid_undulation_m([KRDU_LAT], [KRDU_LON])
        self.assertAlmostEqual(n, KRDU_UNDULATION_M, delta=0.5)

    def test_undulation_is_not_zero(self) -> None:
        """PROJ silently returns a no-op 'ballpark' transform when the grid is missing."""
        (n,) = geoid_undulation_m([KRDU_LAT], [KRDU_LON])
        self.assertGreater(abs(n), 10.0)

    def test_undulation_varies_by_location(self) -> None:
        krdu, ksjc = geoid_undulation_m([KRDU_LAT, 37.3626], [KRDU_LON, -121.9290])
        self.assertNotAlmostEqual(krdu, ksjc, places=1)


class ConversionTest(unittest.TestCase):
    def test_msl_is_higher_than_hae_where_the_geoid_is_below_the_ellipsoid(self) -> None:
        """N is negative over the US, so H_MSL = h_HAE - N comes out ABOVE the input."""
        converted = waypoints_to_msl([[0.0, KRDU_LON, KRDU_LAT, 1000.0]])
        self.assertAlmostEqual(converted[0][3], 1000.0 - KRDU_UNDULATION_M, delta=0.5)
        self.assertGreater(converted[0][3], 1000.0)

    def test_only_altitude_changes(self) -> None:
        (row,) = waypoints_to_msl([[7.0, KRDU_LON, KRDU_LAT, 500.0]])
        self.assertEqual(row[0], 7.0)
        self.assertEqual(row[1], KRDU_LON)
        self.assertEqual(row[2], KRDU_LAT)

    def test_empty_track_is_not_an_error(self) -> None:
        self.assertEqual(waypoints_to_msl([]), [])

    def test_conversion_retags_the_altitude_source(self) -> None:
        self.assertEqual(flight_to_msl(_flight())["altitude_source"], MSL_ALTITUDE_SOURCE)

    def test_conversion_is_not_applied_twice(self) -> None:
        once = flight_to_msl(_flight())
        twice = flight_to_msl(once)
        self.assertEqual(once["waypoints"], twice["waypoints"])

    def test_unknown_datum_raises_rather_than_being_guessed(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            flight_to_msl(_flight(altitude_source="barometric_ft"))
        self.assertIn("barometric_ft", str(ctx.exception))

    def test_missing_datum_raises_rather_than_defaulting(self) -> None:
        with self.assertRaises(ValueError):
            flight_to_msl(_flight(altitude_source=None))

    def test_synthetic_tracks_are_already_msl_and_pass_through(self) -> None:
        """synthetic.py builds waypoints as threshold elevation_m + height, i.e. MSL."""
        original = _flight(altitude_source="synthetic")
        converted = flight_to_msl(original)
        self.assertEqual(converted["waypoints"][0][3], 1000.0)
        self.assertEqual(converted["altitude_source"], "synthetic")

    def test_input_flight_is_not_mutated(self) -> None:
        original = _flight()
        flight_to_msl(original)
        self.assertEqual(original["altitude_source"], HAE_ALTITUDE_SOURCE)
        self.assertEqual(original["waypoints"][0][3], 1000.0)

    def test_batch_converts_every_flight(self) -> None:
        converted = flights_to_msl([_flight(), _flight(id="OTHER")])
        self.assertTrue(all(f["altitude_source"] == MSL_ALTITUDE_SOURCE for f in converted))


class SeamTest(unittest.TestCase):
    """Both modeling entry points must read observed tracks through the SAME loader.

    A scenario built on MSL beside a reference record built on HAE is a silent 30 m
    disagreement about the same flight -- the "batch edition" failure mode where one of two
    parallel paths gets updated and the other doesn't.
    """

    def test_scenario_builder_goes_through_the_converting_loader(self) -> None:
        import inspect

        from flight_scenarios import load_observed_flights
        from flight_scenarios.build import build_scenarios_from_czml_input

        self.assertIn("load_observed_flights", inspect.getsource(build_scenarios_from_czml_input))
        self.assertIn("flights_to_msl", inspect.getsource(load_observed_flights))

    def test_single_flight_builder_converts_by_construction(self) -> None:
        """build_scenario must not trust its caller: the bug reached three load paths."""
        import inspect

        from flight_scenarios.build import build_scenario

        self.assertIn("flight_to_msl", inspect.getsource(build_scenario))

    def test_ts_dataset_builder_converts_before_reading_raw_waypoints(self) -> None:
        """state_samples_from_track takes a bare waypoint list, so its caller must convert."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "4dTrajectory" / "ts_transformer" / "dataset.py"
        ).read_text(encoding="utf-8")
        convert = source.index("flight = flight_to_msl(flight)")
        self.assertLess(convert, source.index("state_samples_from_track(waypoints"))
        self.assertLess(convert, source.index("build_scenario(flight"))

    def test_reference_writer_goes_through_the_converting_loader(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "4dTrajectory" / "optimization" / "scenario_optimization.py"
        ).read_text(encoding="utf-8")
        start = source.index("def write_reference_records")
        body = source[start : source.index("\ndef ", start + 1)]
        self.assertIn("load_observed_flights", body)
        self.assertNotIn("json.loads(Path(observed_tracks)", body)


if __name__ == "__main__":
    unittest.main()
