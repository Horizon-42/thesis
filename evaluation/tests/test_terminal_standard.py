"""Regression tests for the threshold-event verdict standard."""

from __future__ import annotations

import math

import pytest

from evaluation import AssessmentContext, evaluate_record, record_from_dict


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
        "status": "estimated",
        "method": "final_segment_linear_fit",
        "method_version": 1,
        "runway": "05L",
        "threshold_crossing_lat": TARGET["lat"],
        "threshold_crossing_lon": TARGET["lon"],
        "threshold_crossing_altitude_m": TARGET["alt"] + 30.0 + vertical_m,
        "altitude_datum": "hae",
        "signed_cross_track_m": cross_m,
        "cross_track_sigma_m": 0.25,
        "altitude_sigma_m": 0.25,
        "source_sample_range": [0, 1],
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
        lpv_lateral_fsd_m=106.75,
        lpv_vertical_fsd_m=None,
    )


def test_lpv_vertical_and_overall_remain_indeterminate_without_rtca_scale():
    result = evaluate_record(record_from_dict(_record()), context=_lpv_context())

    assert result.lateral_result == "pass"
    assert result.vertical_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert result.success is False


def test_runway_edge_failure_controls_even_when_lpv_vertical_is_indeterminate():
    result = evaluate_record(
        record_from_dict(_record(cross_m=30.0)), context=_lpv_context()
    )

    assert result.lateral_bound_m == pytest.approx(22.86)
    assert result.lateral_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("lateral",)
    assert result.reason is None


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
