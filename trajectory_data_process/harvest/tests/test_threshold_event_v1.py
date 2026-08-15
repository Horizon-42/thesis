"""Contract tests for the single threshold-event resolver."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from final_approach import Assignment, Projected, TrackPoint, fit_final_segment
from evaluation.context import assessment_for_runway
from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
    threshold_frame_fingerprint,
)
from trajectory_data_process.harvest.classify import classify_track
from trajectory_data_process.harvest.threshold_event import (
    build_observed_threshold_event,
)
from trajectory_data_process.harvest.tracks import Sample, Track


def _runway() -> Runway:
    return Runway(
        airport="KFIT",
        ident="18",
        lat=35.0,
        lon=-78.0,
        elevation_hae_m=120.0,
        elevation_msl_m=90.0,
        course_deg=180.0,
        hae_minus_msl_m=30.0,
        threshold_crossing_height_m=15.0,
        published_glidepath_deg=3.0,
        width_m=45.72,
        lpv_course_width_m=106.75,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
    )


def _sample(
    runway: Runway,
    *,
    time_s: float,
    along_m: float,
    cross_m: float,
    height_m: float,
    speed_m_s: float | None = 80.0,
) -> Sample:
    point = runway.frame("hae").unproject(
        Projected(along_m, cross_m, height_m)
    )
    return Sample(
        time_s,
        point.lat,
        point.lon,
        point.alt_m,
        False,
        reported_ground_speed_m_s=speed_m_s,
        last_position_update_s=time_s,
        last_contact_s=time_s,
    )


def _direct_track(*, speed_m_s: float = 80.0) -> Track:
    runway = _runway()
    slope = math.tan(math.radians(3.0))
    samples = [
        _sample(
            runway,
            time_s=float(index),
            along_m=float(along_m),
            cross_m=0.0,
            height_m=15.0 - slope * along_m,
            speed_m_s=speed_m_s,
        )
        for index, along_m in enumerate(range(-8_000, 0, 100))
    ]
    samples[-1] = _sample(
        runway,
        time_s=79.0,
        along_m=-100.0,
        cross_m=-10.0,
        height_m=30.0,
        speed_m_s=speed_m_s,
    )
    samples.append(
        _sample(
            runway,
            time_s=81.5,
            along_m=100.0,
            cross_m=30.0,
            height_m=10.0,
            speed_m_s=speed_m_s,
        )
    )
    return Track("abc123", "FIT123", tuple(samples))


def test_direct_event_interpolates_time_lateral_and_vertical_with_one_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("a source-valid direct bracket must not run a trajectory fit")

    monkeypatch.setattr(
        "final_approach.assign.fit_final_segment", unexpected_fit
    )
    runway = _runway()
    classified = classify_track(
        _direct_track(), Airport("KFIT", 35.0, -78.0, 90.0, (runway,))
    )

    event = classified.observed_threshold_event
    assert classified.fit is None
    assert event["schema_version"] == "runway-threshold-event-v1"
    assert event["method"] == "direct_linear_bracket"
    assert event["observability"] == "within_observed_support"
    assert event["source_sample_range"] == [79, 80]
    assert event["interpolation_fraction"] == pytest.approx(0.5)
    assert event["event_time_s"] == pytest.approx(80.25)
    assert event["signed_cross_track_m"] == pytest.approx(10.0)
    assert event["threshold_crossing_altitude_m"] == pytest.approx(140.0)
    assert event["extrapolation_distance_m"] == 0.0
    assert event["uncertainty"] == {"status": "uncalibrated"}
    assert "candidate_fits" not in event
    assert "fit_window_m" not in event
    assert "threshold_crossing_height_m" not in event["threshold_frame_snapshot"]
    assert "lpv_course_width_m" not in event["threshold_frame_snapshot"]
    assert "width_m" not in event["threshold_frame_snapshot"]


def test_plausible_bracket_with_failed_source_integrity_does_not_fallback_to_fit() -> None:
    runway = _runway()
    track = _direct_track(speed_m_s=10.0)
    classified = classify_track(
        track, Airport("KFIT", 35.0, -78.0, 90.0, (runway,))
    )

    event = classified.observed_threshold_event
    assert classified.outcome == "unassignable"
    assert classified.fit is None
    assert event["status"] == "unavailable"
    assert event["method"] == "none"
    assert event["observability"] == "invalid_support"


def test_censored_event_reuses_winning_assignment_fit_without_refitting(
) -> None:
    runway = _runway()
    slope = math.tan(math.radians(3.0))
    track = Track(
        "abc123",
        "FIT123",
        tuple(
            _sample(
                runway,
                time_s=float(index),
                along_m=float(along_m),
                cross_m=2.0,
                height_m=15.0 - slope * along_m,
            )
            for index, along_m in enumerate(range(-5_000, 0, 100))
        ),
    )
    points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in track.samples
    ]
    fit = fit_final_segment(points, runway.frame("hae"))
    assert fit is not None

    event = build_observed_threshold_event(
        track,
        runway,
        Assignment("assigned", runway.ident, fit, {}, None, None),
    )

    assert event["status"] == "estimated"
    assert event["method"] == "censored_robust_line"
    assert event["observability"] == "right_censored"
    assert event["source_sample_range"] == [
        fit.first_sample_index,
        fit.last_sample_index,
    ]
    assert event["extrapolation_distance_m"] == pytest.approx(100.0, abs=1.0)
    assert event["event_time_s"] is None
    assert event["uncertainty"] == {"status": "uncalibrated"}
    assert set(event["diagnostics"]["fit"]["altitude_fit"]) == {
        "rms_residual_m",
        "max_abs_residual_m",
    }


def test_physical_frame_fingerprint_excludes_evaluation_policy() -> None:
    runway = _runway()
    policy_only_change = replace(
        runway,
        threshold_crossing_height_m=18.0,
        published_glidepath_deg=3.2,
        width_m=60.0,
        lpv_course_width_m=120.0,
        width_source="different-policy-source",
    )
    physical_change = replace(runway, lat=runway.lat + 0.0001)

    assert threshold_frame_fingerprint(policy_only_change) == \
        threshold_frame_fingerprint(runway)
    assert threshold_frame_fingerprint(physical_change) != \
        threshold_frame_fingerprint(runway)
    assert assessment_for_runway(
        policy_only_change
    ).evaluation_context_fingerprint != assessment_for_runway(
        runway
    ).evaluation_context_fingerprint
