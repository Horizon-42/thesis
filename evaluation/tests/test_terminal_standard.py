"""Regression tests for the threshold-event verdict standard."""

from __future__ import annotations

import math

import pytest

from evaluation import AssessmentContext, evaluate_batch, evaluate_record, record_from_dict
from evaluation.context import assessment_for_runway
from evaluation.tests.factories import (
    TARGET,
    assessment_context,
    observed_payload,
    trajectory_payload,
)
from trajectory_data_process.harvest.airports import Runway


def test_lpv_reference_uses_the_common_rnav_terminal_vertical_bound():
    result = evaluate_record(
        record_from_dict(observed_payload()), context=assessment_context()
    )

    assert result.lateral_result == "pass"
    assert result.vertical_result == "pass"
    assert result.vertical_lower_bound_m == pytest.approx(-22.0)
    assert result.vertical_upper_bound_m == pytest.approx(22.0)
    assert result.verdict == "pass"
    assert result.success is True


@pytest.mark.parametrize("vertical_m", [-22.0, 22.0])
def test_ideal_rnav_trajectory_passes_at_the_exact_vertical_bound(vertical_m):
    result = evaluate_record(
        record_from_dict(trajectory_payload(final_alt=TARGET["alt"] + vertical_m)),
        context=assessment_context(),
    )

    assert result.vertical_result == "pass"
    assert result.verdict == "pass"


@pytest.mark.parametrize("vertical_m", [-22.0001, 22.0001])
def test_ideal_rnav_trajectory_fails_just_outside_vertical_bound(vertical_m):
    result = evaluate_record(
        record_from_dict(trajectory_payload(final_alt=TARGET["alt"] + vertical_m)),
        context=assessment_context(),
    )

    assert result.vertical_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("vertical",)


def test_uncalibrated_observed_event_keeps_the_point_estimate_gate():
    result = evaluate_record(
        record_from_dict(observed_payload(vertical_m=21.9)),
        context=assessment_context(),
    )

    assert result.vertical_result == "pass"
    assert result.verdict == "pass"


def test_observed_point_outside_gate_fails_without_a_fabricated_interval():
    payload = observed_payload(vertical_m=22.1)

    result = evaluate_record(record_from_dict(payload), context=assessment_context())

    assert result.vertical_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("vertical",)


def test_runway_edge_failure_controls_with_lpv_vertical_available():
    result = evaluate_record(
        record_from_dict(observed_payload(cross_m=30.0)),
        context=assessment_context(),
    )

    assert result.lateral_bound_m == pytest.approx(22.86)
    assert result.lateral_result == "fail"
    assert result.verdict == "fail"
    assert result.violations == ("lateral",)
    assert result.reason is None


def test_lateral_point_estimate_controls_the_lateral_verdict():
    inside = observed_payload(cross_m=22.8)
    outside = observed_payload(cross_m=23.0)

    pass_result = evaluate_record(record_from_dict(inside), context=assessment_context())
    fail_result = evaluate_record(record_from_dict(outside), context=assessment_context())

    assert pass_result.lateral_result == "pass"
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
        record_from_dict(observed_payload(cross_m=5.0, vertical_m=10.0)),
        context=context,
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

    result = evaluate_record(record_from_dict(trajectory_payload()), context=context)

    assert result.lateral_result == "pass"
    assert result.vertical_result == "indeterminate"
    assert result.verdict == "indeterminate"
    assert result.deviation is not None
    assert result.deviation.vertical_m is None
    assert result.reason == "authoritative Baro-VNAV threshold path reference unavailable"

    report = evaluate_batch(
        [record_from_dict(trajectory_payload())],
        contexts={("KRDU", "05L"): context},
    )
    assert report["vertical_m"] is None
    assert report["trajectories"][0]["vertical_m"] is None


def test_subject_is_required_instead_of_guessed():
    payload = observed_payload()
    del payload["source"]["subject"]

    with pytest.raises(ValueError, match="source.subject"):
        record_from_dict(payload)


def test_non_finite_state_is_rejected_at_the_record_boundary():
    payload = observed_payload()
    payload["states"][-1]["alt"] = math.nan

    with pytest.raises(ValueError, match=r"states\[1\]\.alt.*finite"):
        record_from_dict(payload)
