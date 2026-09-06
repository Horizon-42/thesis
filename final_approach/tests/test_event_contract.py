"""The serialized threshold event's schema, validated by the seam both packages share.

``validate_event`` is what the harvest writes against and what ``evaluation.arrival``
reads through; its rejection cases live here, with the contract, rather than in the
consumer that merely lets the error through.
"""

from __future__ import annotations

from typing import Any

import pytest

from final_approach.event_contract import (
    CENSORED_EVENT_METHOD,
    EVENT_SCHEMA_VERSION,
    validate_event,
)


def censored_event() -> dict[str, Any]:
    """A right-censored estimated event as the harvest serializes it."""
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "estimated",
        "method": CENSORED_EVENT_METHOD,
        "observability": "right_censored",
        "runway": "05L",
        "threshold_crossing_lat": 35.0,
        "threshold_crossing_lon": -78.0,
        "threshold_crossing_altitude_m": 160.0,
        "altitude_datum": "hae",
        "signed_cross_track_m": 0.0,
        "source_sample_range": [0, 1],
        "event_time_s": None,
        "interpolation_fraction": None,
        "extrapolation_distance_m": 325.0,
        "uncertainty": {"status": "uncalibrated"},
        "crossing_ground_speed_m_s": 70.0,
    }


def test_a_current_estimated_event_validates_to_its_status():
    assert validate_event(censored_event()) == "estimated"


def test_the_obsolete_hybrid_method_is_rejected():
    hybrid = censored_event()
    hybrid["method"] = "direct_lateral_fitted_vertical"
    with pytest.raises(ValueError, match="unsupported"):
        validate_event(hybrid)


@pytest.mark.parametrize(
    "dropped",
    [("method",), ("observability",), ("method", "observability")],
    ids=["no-method", "no-observability", "neither"],
)
def test_an_estimated_event_missing_its_discriminators_is_rejected(dropped):
    """Dropping BOTH is the case a bare ``.get(method) != observability`` misses.

    ``None != None`` is False, so such an event validated clean and then fell through
    to the censored branch -- graded as a real crossing on the strength of two absent
    fields. Each single-field case already failed; only the pair did not.
    """
    event = censored_event()
    for key in dropped:
        del event[key]
    with pytest.raises(ValueError, match="unsupported"):
        validate_event(event)


@pytest.mark.parametrize("schema", ["observed-threshold-event-v7", None])
def test_an_obsolete_or_missing_schema_requires_local_reclassification(schema):
    legacy = censored_event()
    if schema is None:
        del legacy["schema_version"]
    else:
        legacy["schema_version"] = schema
    with pytest.raises(ValueError, match="reclassify-existing"):
        validate_event(legacy)
