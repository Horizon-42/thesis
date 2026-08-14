"""Track classification keeps identity and arrival endpoint on the assigned final pass."""

from __future__ import annotations

import math

import pytest

from final_approach import Assignment, Projected, TrackPoint, fit_final_segment
from geokit import METRES_PER_DEG_LAT
from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.classify import classify_track
from trajectory_data_process.harvest.store import track_record
from trajectory_data_process.harvest.threshold_event import (
    build_observed_threshold_event,
    _fit_ensemble,
    _fitted_event,
    select_observed_threshold_bracket,
)
from trajectory_data_process.harvest.tracks import Sample, Track


LAT = 35.0
LON = -78.0
ELEVATION_M = 100.0


def _airport() -> Airport:
    runway = Runway(
        airport="KFIT",
        ident="36",
        lat=LAT,
        lon=LON,
        elevation_hae_m=ELEVATION_M,
        elevation_msl_m=ELEVATION_M,
        course_deg=0.0,
        hae_minus_msl_m=0.0,
        threshold_crossing_height_m=15.0,
        published_glidepath_deg=3.0,
        width_m=45.72,
        lpv_course_width_m=106.75,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
        position_source="faa_cifp_path_point",
        vertical_source="faa_cifp_path_point",
    )
    return Airport("KFIT", LAT, LON, ELEVATION_M, (runway,))


def _sample(time_s: float, along_m: float, height_m: float) -> Sample:
    return Sample(
        time_s=time_s,
        lat=LAT + along_m / METRES_PER_DEG_LAT,
        lon=LON,
        alt_hae_m=ELEVATION_M + height_m,
        on_ground=False,
    )


def test_landing_anchor_belongs_to_last_inbound_pass_not_earlier_high_overflight():
    # The first pass wins a whole-track horizontal argmin because it samples the threshold
    # exactly, but it is 600 m high and travelling in the reciprocal direction.
    early = [
        _sample(float(i), along_m, 1000.0 - i * 80.0)
        for i, along_m in enumerate((5000.0, 4000.0, 3000.0, 2000.0, 1000.0, 0.0))
    ]

    # The final pass is a normal 3-degree inbound approach. Its last observation is 100 m
    # short of the threshold, so only final-pass selection can keep identity on this landing.
    slope = math.tan(math.radians(3.0))
    final = [
        _sample(
            float(len(early) + i),
            along_m,
            15.0 - slope * along_m,
        )
        for i, along_m in enumerate(range(-5000, 0, 100))
    ]
    track = Track("abc123", "FIT123", tuple([*early, *final]))

    classified = classify_track(track, _airport())

    assert classified.outcome == "assigned"
    assert classified.runway == "36"
    assert classified.landing_sample_index == len(track.samples) - 1
    assert classified.landing_sample_index > len(early)
    runway_frame = _airport().runway("36").frame("hae")
    endpoint = track.samples[classified.landing_sample_index]
    assert runway_frame.project(
        TrackPoint(endpoint.lat, endpoint.lon, endpoint.alt_hae_m)
    ).along_m == pytest.approx(-100.0)

    stored = track_record(classified)
    event = stored["observed_threshold_event"]
    assert event["status"] == "estimated"
    assert event["runway"] == "36"
    assert event["altitude_datum"] == "hae"
    assert event["runway_data_fingerprint"]
    assert event["runway_data"]["procedure_source_cycle"] == "2026-08-06"
    # The extrapolation uses the empirically selected 3 km primary window. The
    # winning 5 km assignment fit remains separately preserved for audit.
    assert event["method"] == "final_segment_robust_fit"
    assert event["source_sample_range"] == [
        27, len(track.samples) - 3
    ]
    assert event["assignment_fit"]["source_sample_range"] == [
        len(early), len(track.samples) - 3
    ]
    assert event["threshold_crossing_altitude_m"] == pytest.approx(
        ELEVATION_M + 15.0, abs=0.5
    )
    assert event["signed_cross_track_m"] == pytest.approx(0.0, abs=0.5)
    assert event["extrapolation_m"] == pytest.approx(300.0, abs=1.0)


def test_event_fit_cannot_switch_to_samples_after_the_assignment_pass():
    runway = _airport().runway("36")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(3.0))
    selected_samples = [
        _sample(float(index), along_m, 15.0 - slope * along_m)
        for index, along_m in enumerate(range(-5_000, -200, 100))
    ]
    selected_points = [
        TrackPoint(sample.lat, sample.lon, sample.alt_hae_m)
        for sample in selected_samples
    ]
    assignment_fit = fit_final_segment(selected_points, frame)
    assert assignment_fit is not None

    # A later approach-like stretch is deliberately appended.  The assignment has
    # already selected the first pass, so event-window fitting must not search this
    # later data and silently change the physical event.
    later_samples = [
        _sample(
            float(len(selected_samples) + index),
            along_m,
            80.0 - slope * along_m,
        )
        for index, along_m in enumerate(range(-3_000, -200, 100))
    ]
    track = Track("abc123", "FIT123", tuple([*selected_samples, *later_samples]))
    event = build_observed_threshold_event(
        track,
        runway,
        Assignment("assigned", runway.ident, assignment_fit, {}, None, None),
    )

    assert event["status"] == "estimated"
    assert event["source_sample_range"][1] <= assignment_fit.last_sample_index


def _track_with_threshold_bracket(
    *,
    crossing_height_m: float,
    jump_cross_m: float = 0.0,
    bracket_cross_m: float = 0.0,
    position_update_gap_s: float = 2.5,
    reported_ground_speed_m_s: float = 80.0,
):
    runway = _airport().runway("36")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(3.0))
    samples = []
    for index, along_m in enumerate(range(-8_000, 0, 100)):
        height_m = 15.0 - slope * along_m
        point = frame.unproject(Projected(float(along_m), 0.0, height_m))
        samples.append(Sample(float(index), point.lat, point.lon, point.alt_m, False))
    before = frame.unproject(
        Projected(-100.0, bracket_cross_m, crossing_height_m)
    )
    after = frame.unproject(
        Projected(100.0, bracket_cross_m + jump_cross_m, crossing_height_m)
    )
    before_time = float(len(samples) - 1)
    after_time = float(len(samples))
    samples[-1] = Sample(
        before_time,
        before.lat,
        before.lon,
        before.alt_m,
        False,
        reported_ground_speed_m_s=reported_ground_speed_m_s,
        last_position_update_s=after_time - position_update_gap_s,
        last_contact_s=before_time,
    )
    samples.append(Sample(
        after_time,
        after.lat,
        after.lon,
        after.alt_m,
        False,
        reported_ground_speed_m_s=reported_ground_speed_m_s,
        last_position_update_s=after_time,
        last_contact_s=after_time,
    ))
    return Track("abc123", "FIT123", tuple(samples))


def test_valid_threshold_bracket_anchors_one_robust_three_dimensional_fit():
    classified = classify_track(
        _track_with_threshold_bracket(crossing_height_m=25.0), _airport()
    )

    event = classified.observed_threshold_event
    assert event["schema_version"] == "observed-threshold-event-v6"
    assert event["method"] == "final_segment_robust_fit"
    assert event["method_version"] == 6
    assert "component_methods" not in event
    # The bracket selects and terminates the physical pass.  Both coordinates of
    # the event then come from one robust pre-threshold fit; the bracket's raw
    # altitude is not silently mixed into that fitted point.
    assert event["threshold_crossing_altitude_m"] == pytest.approx(
        ELEVATION_M + 15.0, abs=0.1
    )
    assert event["source_sample_range"] == [51, 77]
    assert event["threshold_bracket"]["source_sample_range"] == [79, 80]
    assert event["threshold_bracket"]["height_at_threshold_m"] == pytest.approx(25.0)
    assert event["lateral_extrapolation_m"] == pytest.approx(300.0, abs=1.0)
    assert event["vertical_extrapolation_m"] == pytest.approx(300.0, abs=1.0)
    assert event["threshold_bracket"]["position_update_gap_s"] == pytest.approx(2.5)
    assert event["threshold_bracket"]["reported_ground_speed_mean_m_s"] == 80.0
    assert event["threshold_bracket"]["position_derived_speed_m_s"] == pytest.approx(
        80.0, abs=0.2
    )


def test_implausible_threshold_jump_falls_back_to_validated_extrapolation():
    classified = classify_track(
        _track_with_threshold_bracket(
            crossing_height_m=25.0,
            jump_cross_m=1_000.0,
        ),
        _airport(),
    )

    event = classified.observed_threshold_event
    assert event["method"] == "final_segment_robust_fit"
    assert event["fit_window_m"] == [-3000.0, -300.0]
    assert {tuple(candidate["window_m"]) for candidate in event["candidate_fits"]} == {
        (-3000.0, -300.0),
        (-4000.0, -300.0),
        (-5000.0, -300.0),
    }
    assert event["uncertainty_95_m"]["vertical_effective"] == pytest.approx(
        event["uncertainty_95_m"]["vertical_statistical"]
        + event["uncertainty_95_m"]["vertical_window_sensitivity"]
    )
    assert event["uncertainty_95_m"]["lateral_effective"] >= 10.5
    assert event["bracket_rejections"][0]["reason"] == \
        "position displacement disagrees with ADS-B reported ground speed"


def test_speed_valid_bracket_outside_landing_structure_is_not_direct():
    classified = classify_track(
        _track_with_threshold_bracket(
            crossing_height_m=25.0,
            bracket_cross_m=1_500.0,
        ),
        _airport(),
    )

    event = classified.observed_threshold_event
    assert event["method"] == "final_segment_robust_fit"
    assert event["signed_cross_track_m"] == pytest.approx(0.0, abs=0.5)
    assert any(
        rejection["reason"] == "threshold bracket is outside landing structure"
        for rejection in event["bracket_rejections"]
    )


def test_low_lateral_bracket_hundreds_of_metres_high_is_not_direct():
    classified = classify_track(
        _track_with_threshold_bracket(crossing_height_m=120.0),
        _airport(),
    )

    event = classified.observed_threshold_event
    assert event["method"] == "final_segment_robust_fit"
    assert event["threshold_crossing_altitude_m"] == pytest.approx(
        ELEVATION_M + 15.0, abs=0.5
    )


def test_bracket_selection_uses_cross_track_not_the_later_parallel_threshold():
    base = _airport().runway("36")
    east_shift_deg = 200.0 / (
        METRES_PER_DEG_LAT * math.cos(math.radians(LAT))
    )
    parallel = Runway(
        **{
            **base.__dict__,
            "ident": "36R",
            "lat": base.lat + 100.0 / METRES_PER_DEG_LAT,
            "lon": base.lon + east_shift_deg,
        }
    )
    slope = math.tan(math.radians(3.0))
    samples = []
    for index, along_m in enumerate(range(-5_000, 201, 100)):
        point = base.frame("hae").unproject(
            Projected(float(along_m), 0.0, 15.0 - slope * along_m)
        )
        samples.append(
            Sample(
                float(index),
                point.lat,
                point.lon,
                point.alt_m,
                False,
                reported_ground_speed_m_s=100.0,
                last_position_update_s=float(index),
                last_contact_s=float(index),
            )
        )
    track = Track("abc123", "FIT123", tuple(samples))

    selection = select_observed_threshold_bracket(
        track,
        (base, parallel),
        max_structural_cross_m=1_000.0,
        max_structural_height_m=100.0,
    )

    assert selection.outcome == "assigned"
    assert selection.bracket is not None
    assert selection.bracket.runway.ident == "36"
    assert selection.bracket.source_sample_range == (49, 50)
    assert selection.scores_m["36"] == pytest.approx(0.0, abs=0.5)
    assert selection.scores_m["36R"] == pytest.approx(200.0, abs=1.0)


def test_valid_bracket_cannot_publish_a_structurally_incompatible_fit():
    runway = _airport().runway("36")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(3.0))
    samples = []
    for index, along_m in enumerate(range(-8_000, 0, 100)):
        cross_m = 0.0 if along_m == -100 else 1_500.0
        point = frame.unproject(
            Projected(float(along_m), cross_m, 15.0 - slope * along_m)
        )
        samples.append(Sample(float(index), point.lat, point.lon, point.alt_m, False))
    after = frame.unproject(Projected(100.0, 0.0, 15.0))
    samples[-1] = Sample(
        samples[-1].time_s,
        samples[-1].lat,
        samples[-1].lon,
        samples[-1].alt_hae_m,
        False,
        reported_ground_speed_m_s=100.0,
        last_position_update_s=79.0,
        last_contact_s=79.0,
    )
    samples.append(
        Sample(
            81.0,
            after.lat,
            after.lon,
            after.alt_m,
            False,
            reported_ground_speed_m_s=100.0,
            last_position_update_s=81.0,
            last_contact_s=81.0,
        )
    )

    classified = classify_track(Track("abc123", "FIT123", tuple(samples)), _airport())

    assert classified.outcome == "assigned"
    assert classified.fit is None
    assert classified.observed_threshold_event["status"] == "unavailable"
    assert classified.observed_threshold_event["method"] == \
        "final_segment_robust_fit"
    assert "structurally incompatible" in classified.observed_threshold_event[
        "unavailable_reason"
    ]


def test_state_row_time_does_not_create_a_false_position_jump():
    classified = classify_track(
        _track_with_threshold_bracket(
            crossing_height_m=25.0,
            # About 500 m in the one-second state-row interval, but 100 m/s
            # over the real five-second ADS-B position-update interval.
            jump_cross_m=458.0,
            position_update_gap_s=5.0,
            reported_ground_speed_m_s=100.0,
        ),
        _airport(),
    )

    event = classified.observed_threshold_event
    assert event["method"] == "final_segment_robust_fit"
    assert event["threshold_bracket"]["sample_gap_s"] == 1.0
    assert event["threshold_bracket"]["position_update_gap_s"] == 5.0
    assert event["threshold_bracket"]["position_derived_speed_m_s"] == pytest.approx(
        100.0, abs=1.0
    )
    assert event["threshold_bracket"]["reported_ground_speed_mean_m_s"] == 100.0


def test_later_valid_bracket_does_not_reuse_an_earlier_approach_fit():
    track = _track_with_threshold_bracket(
        crossing_height_m=25.0,
        jump_cross_m=1_000.0,
    )
    runway = _airport().runway("36")
    frame = runway.frame("hae")
    samples = list(track.samples)
    returned_before = frame.unproject(Projected(-100.0, 0.0, 25.0))
    valid_after = frame.unproject(Projected(100.0, 0.0, 25.0))
    samples.extend(
        (
            Sample(
                82.0,
                returned_before.lat,
                returned_before.lon,
                returned_before.alt_m,
                False,
                reported_ground_speed_m_s=100.0,
                last_position_update_s=82.0,
                last_contact_s=82.0,
            ),
            Sample(
                84.0,
                valid_after.lat,
                valid_after.lon,
                valid_after.alt_m,
                False,
                reported_ground_speed_m_s=100.0,
                last_position_update_s=84.0,
                last_contact_s=84.0,
            ),
        )
    )

    classified = classify_track(
        Track(track.icao24, track.callsign, tuple(samples)),
        _airport(),
    )

    event = classified.observed_threshold_event
    assert classified.outcome == "assigned"
    assert classified.fit is None
    assert classified.landing_sample_index == 81
    assert event["method"] == "final_segment_robust_fit"
    assert event["status"] == "unavailable"
    assert event["unavailable_reason"] == \
        "selected final inbound pass has no fittable segment"
    assert event["threshold_bracket"]["source_sample_range"] == [81, 82]
    assert event["bracket_rejections"][0]["source_sample_range"] == [79, 80]
    assert event["bracket_rejections"][0]["reason"] == \
        "position displacement disagrees with ADS-B reported ground speed"


@pytest.mark.parametrize(
    ("outer_m", "expected_window"),
    (
        (-4_000, [-4_000.0, -300.0]),
        (-5_000, [-5_000.0, -300.0]),
    ),
)
def test_wider_primary_fit_reports_statistical_and_window_diagnostics_without_proxy_floor(
    outer_m: int,
    expected_window: list[float],
):
    runway = _airport().runway("36")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(3.0))
    points = [
        frame.unproject(Projected(float(along_m), 0.0, 15.0 - slope * along_m))
        for along_m in range(outer_m, outer_m + 1_000, 100)
    ]
    assignment_fit = fit_final_segment(
        points,
        frame,
        window_m=(-5_000.0, -300.0),
    )
    assert assignment_fit is not None

    ensemble = _fit_ensemble(runway, points, assignment_fit)
    assert ensemble is not None
    event = _fitted_event(
        runway,
        assignment_fit,
        ensemble=ensemble,
        bracket=None,
        bracket_rejections=(),
    )

    assert event["status"] == "estimated"
    assert event["fit_window_m"] == expected_window
    assert "empirical_calibration" not in event
    assert "vertical_empirical_floor" not in event["uncertainty_95_m"]
    assert event["uncertainty_95_m"]["vertical_effective"] == pytest.approx(
        event["uncertainty_95_m"]["vertical_statistical"]
        + event["uncertainty_95_m"]["vertical_window_sensitivity"]
    )
