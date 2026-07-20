"""Pin the flight_identity MIRROR to the canonical vector.

``flight_identity.flight_key`` must match ``flight_scenarios.identity.flight_key``
exactly; this package cannot import that one (it would pull the modeling tree), so the
same pinned example guards both copies: ``test_scenario_optimization.py`` asserts it
against the canonical function, this file against the mirror. If either copy drifts,
one of the two suites fails.
"""

from flight_identity import flight_key


def test_pinned_canonical_vector_matches_the_canonical_function():
    src = {
        "id": "EJA969",
        "runway": "05R",
        "icao24": "ad7f04",
        "landing_time_utc": "2026-06-18T21:37:36Z",
    }
    assert flight_key(src, 0) == "EJA969_05R_ad7f04_20260618T213736Z"


def test_missing_fields_are_skipped_and_index_is_the_last_resort():
    assert flight_key({"id": "AAL1"}, 3) == "AAL1"
    assert flight_key({}, 3) == "flight3"


def test_filename_unsafe_characters_collapse_to_underscores():
    assert flight_key({"id": "AAL 1/x"}, 0) == "AAL_1_x"
