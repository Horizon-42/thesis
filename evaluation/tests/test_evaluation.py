"""Record validation, batch methodology, and reference-span regressions."""

from __future__ import annotations

import json
import math

import pytest

from evaluation import (
    AssessmentContext,
    compare_to_reference,
    evaluate_batch,
    load_record,
    record_from_dict,
)


def context() -> AssessmentContext:
    return AssessmentContext(
        benchmark="rnp_apch_lnav_vnav_baro", airport="KRDU", runway="05L",
        runway_course_deg=0.0, runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy", runway_source_cycle="2026-08-06",
        procedure_source="faa_terminal_procedure", procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=130.0,
        threshold_elevation_msl_m=100.0,
        threshold_crossing_height_m=30.0,
        baro_vnav_approved=True,
    )


def payload(*, subject="optimized", final_lat=35.0, final_lon=-78.0,
            final_alt=130.0, final_t=100.0):
    target = {"lat": 35.0, "lon": -78.0, "alt": 130.0, "V": 70.0,
              "psi": 0.0, "gamma": -0.05, "m": 60_000.0}
    first = {"t": 0.0, **target, "lat": 34.9, "alt": 1000.0}
    last = {"t": final_t, **target, "lat": final_lat, "lon": final_lon,
            "alt": final_alt}
    return {
        "source": {"id": "TEST1", "subject": subject, "arr_airport": "KRDU",
                   "runway": "05L", "icao24": "abc123",
                   "landing_time_utc": "2026-08-12T00:00:00Z"},
        "initial_state": {key: value for key, value in first.items() if key != "t"},
        "target_state": target, "final_time_s": final_t, "states": [first, last],
        "controls": [] if subject == "observed" else [{"thrust": 1.0}] * 2,
    }


def contexts():
    return {("KRDU", "05L"): context()}


@pytest.mark.parametrize(
    "location",
    ["initial", "target", "first", "interior", "last", "time", "control"],
)
def test_non_finite_values_are_rejected_at_the_record_boundary(location):
    value = payload()
    if location == "initial": value["initial_state"]["alt"] = math.nan
    elif location == "target": value["target_state"]["lat"] = math.inf
    elif location == "first": value["states"][0]["lon"] = -math.inf
    elif location == "interior": value["states"].insert(1, dict(value["states"][0], alt=math.nan, t=50.0)); value["controls"].insert(1, {"thrust": 1.0})
    elif location == "last": value["states"][-1]["alt"] = math.nan
    elif location == "time": value["final_time_s"] = math.nan
    else: value["controls"][0]["thrust"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        record_from_dict(value)


def test_json_nan_token_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload()).replace("130.0", "NaN", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON"):
        load_record(path)


def test_batch_serializes_context_methodology_and_three_way_counts():
    report = evaluate_batch([record_from_dict(payload())], contexts=contexts())
    assert report["schema_version"] == "terminal-approach-evaluation-v3"
    assert report["verdict_counts"] == {"pass": 1, "fail": 0, "indeterminate": 0}
    assert report["assessment_contexts"][0]["runway_source_cycle"] == "2026-08-06"
    assert report["assessment_contexts"][0]["desired_threshold_altitude_msl_m"] == 130.0
    assert report["methodology"]["event"]["observed"].endswith("no evaluation refit")
    uncertainty = report["methodology"]["uncertainty"]
    assert uncertainty["classification"] == "diagnostic_only_not_used_by_verdict"
    assert uncertainty["verdict_rule"] == \
        "point_estimate_against_inclusive_component_bounds"
    lpv = report["methodology"]["terminal_vertical"]["lpv"]
    assert lpv["scale_model"] == "do229_lpv_angular_min_clamped"
    assert lpv["one_sided_minimum_fsd_m"] == 15.0
    assert lpv["normal_fsd_fraction"] == 0.5
    assert lpv["effective_threshold_bound_m"] == 7.5
    assert {item["location"] for item in lpv["sources"]} == {
        "§§2.2.4.4.4 and 2.2.5.4.4",
        "Volume II, Part C, Chapter 5, Section B, §5.3.3.1.1.1(b)",
        "Chapter 2, page 2-15, Glidepath - GPS Source",
    }
    json.dumps(report, allow_nan=False)


def test_observed_availability_is_supplied_from_the_unfiltered_harvest_roster():
    from evaluation.tests.test_terminal_standard import _lpv_context, _record

    availability = {
        "denominator": "arrival_candidates_excluding_not_landing",
        "event_denominator": 3,
        "event_estimated": 1,
        "event_unavailable": 2,
        "event_estimated_rate": 1 / 3,
        "excluded_not_landing": 1,
    }
    report = evaluate_batch(
        [record_from_dict(_record())],
        contexts={("KRDU", "05L"): _lpv_context()},
        observed_availability=availability,
    )

    assert report["total"] == 1
    assert report["observed"] == availability


def test_computed_trajectory_that_does_not_reach_threshold_fails_the_event():
    report = evaluate_batch(
        [record_from_dict(payload(final_lat=34.999))], contexts=contexts()
    )
    row = report["trajectories"][0]
    assert row["event_status"] == "not_reached"
    assert row["verdict"] == "fail"


def test_observed_report_copies_the_policy_free_event_for_audit():
    from evaluation.tests.test_terminal_standard import _record, _lpv_context

    record = record_from_dict(_record())
    report = evaluate_batch(
        [record], contexts={("KRDU", "05L"): _lpv_context()}
    )
    copied = report["trajectories"][0]["observed_threshold_event"]
    assert copied == record.source["observed_threshold_event"]
    assert "verdict" not in copied and "benchmark" not in copied


def _write_pair(tmp_path, *, reference_end_lat: float):
    ours = payload()
    ours["reference_file"] = "reference.json"
    ref = payload(subject="observed", final_lat=reference_end_lat)
    (tmp_path / "reference.json").write_text(json.dumps(ref), encoding="utf-8")
    path = tmp_path / "ours.json"
    path.write_text(json.dumps(ours), encoding="utf-8")
    record = load_record(path)
    reference = load_record(tmp_path / "reference.json")
    return record, reference


def test_reference_comparison_rejects_paths_that_end_roughly_325_m_apart(tmp_path):
    record, reference = _write_pair(tmp_path, reference_end_lat=35.0 - 325.0 / 111_319.5)
    with pytest.raises(ValueError, match="different physical spans"):
        compare_to_reference(record, reference)
    report = evaluate_batch([record], contexts=contexts())
    block = report["trajectories"][0]["reference"]
    assert block["comparison_status"] == "skipped"
    assert block["end_gap_m"] == pytest.approx(325.0, rel=0.01)
    assert "flight_time_delta_s" not in block
    assert report["reference"] is None


def test_reference_comparison_runs_when_both_physical_endpoints_match(tmp_path):
    record, reference = _write_pair(tmp_path, reference_end_lat=35.0)
    comparison = compare_to_reference(record, reference)
    assert comparison.path_lateral_m["max"] == pytest.approx(0.0, abs=1e-6)
    report = evaluate_batch([record], contexts=contexts())
    assert report["reference"]["compared"] == 1


def test_empty_failed_path_skips_reference_with_strict_json_null_gaps(tmp_path):
    ours = payload()
    ours.update(
        reference_file="reference.json",
        final_time_s=None,
        states=[],
        controls=[],
        reason="solver failed",
    )
    reference = payload(subject="observed")
    (tmp_path / "reference.json").write_text(
        json.dumps(reference), encoding="utf-8"
    )
    path = tmp_path / "ours.json"
    path.write_text(json.dumps(ours), encoding="utf-8")

    report = evaluate_batch([load_record(path)], contexts=contexts())

    block = report["trajectories"][0]["reference"]
    assert block["comparison_status"] == "skipped"
    assert block["start_gap_m"] is None
    assert block["end_gap_m"] is None
    json.dumps(report, allow_nan=False)
