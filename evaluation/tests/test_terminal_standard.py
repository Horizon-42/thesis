"""Regression tests for the threshold-event verdict standard."""

from __future__ import annotations

import math

import pytest

from evaluation import AssessmentContext, evaluate_batch, evaluate_record, record_from_dict
from evaluation.context import assessment_for_runway
from trajectory_data_process.harvest.airports import Runway


TARGET = {
    "lat": 35.9,
    "lon": -78.8,
    "alt": 130.0,
    "V": 70.0,
    "psi": 0.8,
    "gamma": -0.05,
    "m": 60_000.0,
}


def _record(*, cross_m: float = 0.0, vertical_m: float = 0.0) -> dict:
    event = {
        "schema_version": "observed-threshold-event-v4",
        "status": "estimated",
        "method": "final_segment_window_ensemble",
        "method_version": 4,
        "component_methods": {
            "lateral": "final_segment_window_ensemble",
            "vertical": "final_segment_window_ensemble",
        },
        "runway": "05L",
        "threshold_crossing_lat": TARGET["lat"],
        "threshold_crossing_lon": TARGET["lon"],
        "threshold_crossing_altitude_m": TARGET["alt"] + 30.0 + vertical_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": cross_m,
        "cross_track_sigma_m": 0.25,
        "altitude_sigma_m": 0.25,
        "component_source_sample_ranges": {
            "lateral": [0, 1], "vertical": [0, 1],
        },
        "fit_window_m": [-5_000.0, -300.0],
        "sample_count": 8,
        "along_track_span_m": 4_000.0,
        "extrapolation_m": 325.0,
        "cross_track_fit": {
            "rms_residual_m": 0.5,
            "max_abs_residual_m": 1.0,
            "rho": 0.0,
            "n_effective": 8.0,
        },
        "altitude_fit": {
            "rms_residual_m": 0.5,
            "max_abs_residual_m": 1.0,
            "rho": 0.0,
            "n_effective": 8.0,
        },
    }
    first = {"t": 0.0, **TARGET, "lat": 35.8, "alt": 500.0}
    last = {"t": 100.0, **TARGET, "lat": 35.89, "alt": 150.0}
    return {
        "source": {
            "id": "TEST123",
            "subject": "observed",
            "arr_airport": "KRDU",
            "runway": "05L",
            "flight_key": "TEST123_05L_abc123_20260812T000000Z",
            "hae_minus_msl_m": 30.0,
            "observed_threshold_event": event,
        },
        "initial_state": {key: value for key, value in first.items() if key != "t"},
        "target_state": TARGET,
        "final_time_s": 100.0,
        "states": [first, last],
        "controls": [],
    }


def _computed_record(*, vertical_m: float = 0.0) -> dict:
    first = {"t": 0.0, **TARGET, "lat": 35.8, "alt": 500.0}
    last = {"t": 100.0, **TARGET, "alt": TARGET["alt"] + vertical_m}
    return {
        "source": {
            "id": "IDEAL123",
            "subject": "optimized",
            "arr_airport": "KRDU",
            "runway": "05L",
        },
        "initial_state": {key: value for key, value in first.items() if key != "t"},
        "target_state": TARGET,
        "final_time_s": 100.0,
        "states": [first, last],
        "controls": [{"thrust": 1.0}, {"thrust": 1.0}],
    }


def _lpv_context() -> AssessmentContext:
    return AssessmentContext(
        benchmark="lpv",
        airport="KRDU",
        runway="05L",
        runway_course_deg=45.0,
        runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy",
        runway_source_cycle="2026-08-06",
        procedure_source="faa_cifp_path_point",
        procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=144.76,
        threshold_elevation_msl_m=114.76,
        threshold_crossing_height_m=15.24,
        lpv_lateral_fsd_m=106.75,
    )


def test_lpv_vertical_and_overall_use_the_resolved_threshold_bound():
    result = evaluate_record(record_from_dict(_record()), context=_lpv_context())

    assert result.lateral_result == "pass"
    assert result.vertical_result == "pass"
    assert result.vertical_lower_bound_m == pytest.approx(-7.5)
    assert result.vertical_upper_bound_m == pytest.approx(7.5)
    assert result.verdict == "pass"
    assert result.success is True


@pytest.mark.parametrize("vertical_m", [-7.5, 7.5])
def test_ideal_lpv_trajectory_passes_at_the_exact_vertical_bound(vertical_m):
    result = evaluate_record(
        record_from_dict(_computed_record(vertical_m=vertical_m)),
        context=_lpv_context(),
    )

    assert result.vertical_result == "pass"
    assert result.verdict == "pass"


@pytest.mark.parametrize("vertical_m", [-7.5001, 7.5001])
def test_ideal_lpv_trajectory_fails_just_outside_vertical_bound(vertical_m):
    result = evaluate_record(
        record_from_dict(_computed_record(vertical_m=vertical_m)),
        context=_lpv_context(),
    )

    assert result.vertical_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("vertical",)


def test_observed_uncertainty_is_diagnostic_and_does_not_shrink_vertical_gate():
    result = evaluate_record(
        record_from_dict(_record(vertical_m=7.2)), context=_lpv_context()
    )

    assert result.vertical_interval_m == pytest.approx((6.71, 7.69))
    assert result.vertical_result == "pass"
    assert result.verdict == "pass"


def test_nominal_extrapolated_event_can_pass_with_large_estimator_uncertainty():
    payload = _record()
    payload["source"]["observed_threshold_event"]["altitude_sigma_m"] = 13.0 / 1.96

    result = evaluate_record(record_from_dict(payload), context=_lpv_context())

    assert result.vertical_interval_m == pytest.approx((-13.0, 13.0))
    assert result.vertical_result == "pass"
    assert result.verdict == "pass"


def test_observed_point_outside_gate_fails_even_when_interval_overlaps_gate():
    payload = _record(vertical_m=8.0)
    payload["source"]["observed_threshold_event"]["altitude_sigma_m"] = 2.0

    result = evaluate_record(record_from_dict(payload), context=_lpv_context())

    assert result.vertical_interval_m == pytest.approx((4.08, 11.92))
    assert result.vertical_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("vertical",)


def test_runway_edge_failure_controls_with_lpv_vertical_available():
    result = evaluate_record(
        record_from_dict(_record(cross_m=30.0)), context=_lpv_context()
    )

    assert result.lateral_bound_m == pytest.approx(22.86)
    assert result.lateral_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("lateral",)
    assert result.reason is None


def test_estimator_uncertainty_does_not_change_lateral_point_verdict():
    inside = _record(cross_m=22.8)
    inside["source"]["observed_threshold_event"]["cross_track_sigma_m"] = 2.0
    outside = _record(cross_m=23.0)
    outside["source"]["observed_threshold_event"]["cross_track_sigma_m"] = 2.0

    pass_result = evaluate_record(record_from_dict(inside), context=_lpv_context())
    fail_result = evaluate_record(record_from_dict(outside), context=_lpv_context())

    assert pass_result.lateral_interval_m == pytest.approx((18.88, 26.72))
    assert pass_result.lateral_result == "pass"
    assert fail_result.lateral_interval_m == pytest.approx((19.08, 26.92))
    assert fail_result.lateral_result == "fail"


def test_lnav_vnav_fallback_has_a_real_vertical_gate():
    context = AssessmentContext(
        benchmark="rnp_apch_lnav_vnav_baro",
        airport="KRDU",
        runway="05L",
        runway_course_deg=45.0,
        runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy",
        runway_source_cycle="2026-08-06",
        procedure_source="faa_terminal_procedure",
        procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=144.76,
        threshold_elevation_msl_m=114.76,
        threshold_crossing_height_m=15.24,
        baro_vnav_approved=True,
    )

    result = evaluate_record(
        record_from_dict(_record(cross_m=5.0, vertical_m=10.0)), context=context
    )

    assert result.lateral_result == "pass"
    assert result.vertical_result == "pass"
    assert result.vertical_lower_bound_m == pytest.approx(-22.0)
    assert result.vertical_upper_bound_m == pytest.approx(22.0)
    assert result.verdict == "pass"


def test_non_lpv_fallback_keeps_lateral_result_when_path_reference_is_unavailable():
    runway = Runway(
        airport="KRDU",
        ident="05L",
        lat=TARGET["lat"],
        lon=TARGET["lon"],
        elevation_hae_m=130.0,
        elevation_msl_m=100.0,
        course_deg=45.0,
        hae_minus_msl_m=30.0,
        threshold_crossing_height_m=None,
        published_glidepath_deg=None,
        width_m=45.72,
        lpv_course_width_m=None,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
    )
    context = assessment_for_runway(runway, baro_vnav_approved=True)

    result = evaluate_record(record_from_dict(_computed_record()), context=context)

    assert result.lateral_result == "pass"
    assert result.vertical_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert result.deviation is not None
    assert result.deviation.vertical_m is None
    assert result.reason == "authoritative Baro-VNAV threshold path reference unavailable"

    report = evaluate_batch(
        [record_from_dict(_computed_record())],
        contexts={("KRDU", "05L"): context},
    )
    assert report["vertical_m"] is None
    assert report["trajectories"][0]["vertical_m"] is None


def test_subject_is_required_instead_of_guessed():
    payload = _record()
    del payload["source"]["subject"]

    with pytest.raises(ValueError, match="source.subject"):
        record_from_dict(payload)


def test_non_finite_state_is_rejected_at_the_record_boundary():
    payload = _record()
    payload["states"][-1]["alt"] = math.nan

    with pytest.raises(ValueError, match=r"states\[1\]\.alt.*finite"):
        record_from_dict(payload)
