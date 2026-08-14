"""Observed records reject stale event geometry and report honest coverage."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trajectory_data_process.harvest.airports import Runway, runway_data_fingerprint
from trajectory_data_process.harvest.observed import (
    observed_record,
    source_event_availability,
)


def _runway() -> Runway:
    return Runway(
        airport="KAAA", ident="18", lat=35.0, lon=-78.0,
        elevation_hae_m=130.0, elevation_msl_m=100.0,
        course_deg=180.0, hae_minus_msl_m=30.0,
        threshold_crossing_height_m=15.0, published_glidepath_deg=3.0,
        width_m=45.72, lpv_course_width_m=106.75,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
    )


def test_observed_record_rejects_event_from_a_different_runway_cycle():
    runway = _runway()
    event = {
        "schema_version": "observed-threshold-event-v6",
        "status": "estimated",
        "method": "final_segment_robust_fit",
        "method_version": 6,
        "runway": runway.ident,
        "runway_data_fingerprint": runway_data_fingerprint(runway),
    }
    track = {
        "flight_key": "TEST_18_abc123_20260812T000000Z",
        "callsign": "TEST",
        "icao24": "abc123",
        "landing_time_utc": "2026-08-12T00:00:00Z",
        "observed_threshold_event": event,
        "samples": [[0.0, -78.0, 35.05, 500.0], [1.0, -78.0, 35.04, 450.0]],
    }
    newer = replace(runway, procedure_source_cycle="2026-09-03")

    with pytest.raises(ValueError, match="runway-data fingerprint.*reclassify"):
        observed_record(track, newer)


def test_observed_record_preserves_a_current_unavailable_event_for_indeterminate_evaluation():
    runway = _runway()
    event = {
        "schema_version": "observed-threshold-event-v6",
        "status": "unavailable",
        "method": "final_segment_robust_fit",
        "method_version": 6,
        "runway": runway.ident,
        "runway_data_fingerprint": runway_data_fingerprint(runway),
        "unavailable_reason": "selected final inbound pass has no fittable segment",
    }
    track = {
        "flight_key": "TEST_18_abc123_20260812T000000Z",
        "callsign": "TEST",
        "icao24": "abc123",
        "landing_time_utc": "2026-08-12T00:00:00Z",
        "observed_threshold_event": event,
        "samples": [[0.0, -78.0, 35.05, 500.0], [1.0, -78.0, 35.04, 450.0]],
    }

    record = observed_record(track, runway)

    assert record["source"]["observed_threshold_event"] == event


def test_event_availability_counts_ambiguous_and_unassignable_candidates():
    source = {
        "total": 4,
        "counts": {
            "assigned": 1,
            "ambiguous": 1,
            "unassignable": 1,
            "not_landing": 1,
        },
        "records": [
            {"outcome": "assigned", "event_status": "estimated"},
            {"outcome": "ambiguous", "event_status": "unavailable"},
            {"outcome": "unassignable", "event_status": "unavailable"},
            {"outcome": "not_landing", "event_status": "unavailable"},
        ],
    }

    availability = source_event_availability(source)

    assert availability["denominator"] == "arrival_candidates_excluding_not_landing"
    assert availability["event_denominator"] == 3
    assert availability["event_estimated"] == 1
    assert availability["event_unavailable"] == 2
    assert availability["event_estimated_rate"] == pytest.approx(1 / 3)
    assert availability["excluded_not_landing"] == 1
