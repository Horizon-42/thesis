from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from trajectory_data_process.acquisition.airports import AirportProfile
from trajectory_data_process.acquisition.runways import RunwayThreshold
from trajectory_data_process.landings import (
    check_history_access,
    download_airport_landings,
    iter_airport_entries,
)

PROFILE = AirportProfile("KRDU", 35.8776, -78.7875, 132.0)
THRESHOLD = RunwayThreshold("KRDU", "23R", 35.8938, -78.778, 124.7, 225.0)
START = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)


def _landing_df() -> pd.DataFrame:
    rows = []
    for sec, lat, lon, geo in [
        ("2026-04-19T11:30:00Z", 35.95, -78.79, 900.0),
        ("2026-04-19T11:30:30Z", 35.92, -78.785, 500.0),
        ("2026-04-19T11:31:00Z", 35.8938, -78.778, 150.0),
    ]:
        rows.append(
            {
                "time": sec, "icao24": "abc123", "lat": lat, "lon": lon,
                "baroaltitude": geo + 20, "geoaltitude": geo, "heading": 225.0,
                "onground": False, "callsign": "LND1",
                "estarrivalairport": "KRDU", "estdepartureairport": "KJFK",
            }
        )
    return pd.DataFrame(rows)


class _StubFetch:
    """Returns the landing dataframe on the first chunk, empty after."""

    def __init__(self, df: pd.DataFrame, always: bool = False) -> None:
        self.df = df
        self.always = always
        self.calls = 0
        self.last_kwargs: dict = {}

    def __call__(self, **kwargs) -> pd.DataFrame:
        self.calls += 1
        self.last_kwargs = kwargs
        if self.always or self.calls == 1:
            return self.df.copy()
        return pd.DataFrame(columns=list(self.df.columns))


class LandingDownloadTests(unittest.TestCase):
    def test_collects_requested_count(self) -> None:
        fetch = _StubFetch(_landing_df())

        collected = download_airport_landings(
            profile=PROFILE, thresholds=[THRESHOLD], count=1, start=START,
            max_lookback_days=1.0, chunk_hours=6.0, fetch_history_fn=fetch,
        )

        self.assertEqual(len(collected["23R"]), 1)
        self.assertEqual(collected["23R"][0]["runway"], "23R")
        self.assertEqual(fetch.calls, 1)  # stops as soon as the count is met

    def test_gives_up_on_dry_runway_instead_of_full_lookback(self) -> None:
        empty = _StubFetch(pd.DataFrame(columns=list(_landing_df().columns)), always=True)

        collected = download_airport_landings(
            profile=PROFILE, thresholds=[THRESHOLD], count=5, start=START,
            max_lookback_days=10.0, chunk_hours=12.0, dry_give_up_chunks=3,
            fetch_history_fn=empty,
        )

        self.assertEqual(len(collected["23R"]), 0)
        self.assertEqual(empty.calls, 3)  # gave up after 3 dry chunks, not the full 20

    def test_dedupes_same_landing_across_chunks(self) -> None:
        fetch = _StubFetch(_landing_df(), always=True)

        collected = download_airport_landings(
            profile=PROFILE, thresholds=[THRESHOLD], count=5, start=START,
            max_lookback_days=0.75, chunk_hours=6.0, fetch_history_fn=fetch,
        )

        self.assertEqual(len(collected["23R"]), 1)  # same landing not double-counted
        self.assertGreaterEqual(fetch.calls, 2)


class ResumeTests(unittest.TestCase):
    def test_preloaded_at_count_makes_no_queries(self) -> None:
        fetch = _StubFetch(_landing_df(), always=True)
        preloaded = {"23R": [{"icao24": "old1", "landing_time_utc": "2026-04-01T00:00:00Z", "runway": "23R"}]}

        collected = download_airport_landings(
            profile=PROFILE, thresholds=[THRESHOLD], count=1, start=START,
            max_lookback_days=1.0, preloaded=preloaded, fetch_history_fn=fetch,
        )

        self.assertEqual(fetch.calls, 0)            # already satisfied: no download
        self.assertEqual(collected["23R"], preloaded["23R"])

    def test_preloaded_partial_merges_and_dedupes(self) -> None:
        fetch = _StubFetch(_landing_df(), always=True)
        preloaded = {"23R": [{"icao24": "old1", "landing_time_utc": "2026-04-01T00:00:00Z", "runway": "23R"}]}

        collected = download_airport_landings(
            profile=PROFILE, thresholds=[THRESHOLD], count=2, start=START,
            max_lookback_days=0.75, preloaded=preloaded, fetch_history_fn=fetch,
        )

        self.assertEqual(len(collected["23R"]), 2)  # kept the old one, added one new landing
        self.assertEqual(collected["23R"][0]["icao24"], "old1")


class BboxQueryTests(unittest.TestCase):
    def test_download_uses_bbox_not_airport_join(self) -> None:
        fetch = _StubFetch(_landing_df())

        download_airport_landings(
            profile=PROFILE, thresholds=[THRESHOLD], count=1, start=START,
            max_lookback_days=1.0, bbox_radius_km=30.0, fetch_history_fn=fetch,
        )

        self.assertIn("bounds", fetch.last_kwargs)
        self.assertNotIn("airport", fetch.last_kwargs)
        west, south, east, north = fetch.last_kwargs["bounds"]
        self.assertLess(west, PROFILE.lon)  # box brackets the airport centre
        self.assertGreater(north, PROFILE.lat)


class PreflightTests(unittest.TestCase):
    def test_probes_once_and_returns_on_success(self) -> None:
        fetch = _StubFetch(_landing_df(), always=True)

        check_history_access(profile=PROFILE, reference=START, fetch_history_fn=fetch)

        self.assertEqual(fetch.calls, 1)
        self.assertIn("bounds", fetch.last_kwargs)

    def test_propagates_access_error(self) -> None:
        def denied(**_kwargs):
            raise RuntimeError("OpenSky history query was denied (PERMISSION_DENIED).")

        with self.assertRaises(RuntimeError):
            check_history_access(profile=PROFILE, reference=START, fetch_history_fn=denied)


class ConfigLoaderTests(unittest.TestCase):
    def test_iter_airport_entries_builds_profiles_and_thresholds(self) -> None:
        config = {
            "airports": {
                "KRDU": {
                    "lat": 35.8776, "lon": -78.7875, "elevation_m": 132.0,
                    "runways": [
                        {"thresholds": [
                            {"ident": "05L", "lat": 35.87, "lon": -78.80, "elevation_m": 111.0, "heading_deg": 45.0},
                            {"ident": "23R", "lat": 35.89, "lon": -78.77, "elevation_m": 124.0, "heading_deg": 225.0},
                        ]}
                    ],
                }
            }
        }

        entries = iter_airport_entries(config)

        self.assertEqual(len(entries), 1)
        profile, thresholds = entries[0]
        self.assertEqual(profile.code, "KRDU")
        self.assertEqual([t.ident for t in thresholds], ["05L", "23R"])


if __name__ == "__main__":
    unittest.main()
