"""Record validation, batch methodology, and reference-span regressions."""

from __future__ import annotations

import json
import math

import pytest

from evaluation import (
    compare_to_reference,
    evaluate_batch,
    load_record,
    record_from_dict,
)
from evaluation.tests.factories import (
    assessment_context,
    observed_payload,
    trajectory_payload,
)


def contexts():
    return {
        ("KRDU", "05L"): assessment_context(
            benchmark="rnp_apch_lnav_vnav_baro"
        )
    }


@pytest.mark.parametrize(
    "location",
    ["initial", "target", "first", "interior", "last", "time", "control"],
)
def test_non_finite_values_are_rejected_at_the_record_boundary(location):
    value = trajectory_payload()
    if location == "initial": value["initial_state"]["alt"] = math.nan
    elif location == "target": value["target_state"]["lat"] = math.inf
    elif location == "first": value["states"][0]["lon"] = -math.inf
    elif location == "interior":
        value["states"].insert(
            1, dict(value["states"][0], alt=math.nan, t=50.0)
        )
        value["controls"].insert(1, {"thrust": 1.0})
    elif location == "last": value["states"][-1]["alt"] = math.nan
    elif location == "time": value["final_time_s"] = math.nan
    else: value["controls"][0]["thrust"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        record_from_dict(value)


def test_json_nan_token_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(trajectory_payload()).replace("130.0", "NaN", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-standard JSON"):
        load_record(path)


def test_batch_serializes_context_methodology_and_three_way_counts():
    report = evaluate_batch(
        [record_from_dict(trajectory_payload())], contexts=contexts()
    )
    assert report["schema_version"] == "terminal-approach-evaluation-v6"
    assert report["verdict_counts"] == {"pass": 1, "fail": 0, "indeterminate": 0}
    assert report["assessment_contexts"][0]["runway_source_cycle"] == "2026-08-06"
    assert report["assessment_contexts"][0]["desired_threshold_altitude_msl_m"] == 130.0
    assert report["methodology"]["event"]["observed"].endswith("no evaluation refit")
    uncertainty = report["methodology"]["uncertainty"]
    assert uncertainty == {
        "verdict_rule": "point_estimate_against_inclusive_component_bounds",
        "observed_status": "uncalibrated",
        "unmodelled_sources": [
            "ADS-B geometric-altitude update alignment and measurement error",
            "runway/FAS survey uncertainty",
            "geoid/datum uncertainty",
            "model-form and extrapolation uncertainty",
        ],
    }
    vertical = report["methodology"]["terminal_vertical"]
    acceptance = vertical["common_rnav_terminal_acceptance"]
    assert acceptance["standard_id"] == "icao_doc_9613_rnp_apch_fas_22m"
    assert acceptance["lower_m"] == -22.0
    assert acceptance["upper_m"] == 22.0
    assert acceptance["source"]["location"] == (
        "Volume II, Part C, Chapter 5, Section A, §5.3.4.4.7"
    )
    assert acceptance["claim_boundary"] == (
        "terminal final-approach geometry; not touchdown or landing certification"
    )
    deviation = report["trajectories"][0]["deviation"]
    assert "lateral_sigma_m" not in deviation
    assert "vertical_sigma_m" not in deviation
    assert "lateral_interval_m" not in deviation
    assert "vertical_interval_m" not in deviation
    json.dumps(report, allow_nan=False)


def test_observed_availability_is_supplied_from_the_unfiltered_harvest_roster():
    availability = {
        "denominator": "arrival_candidates_excluding_not_landing",
        "event_denominator": 3,
        "event_estimated": 1,
        "event_unavailable": 2,
        "event_estimated_rate": 1 / 3,
        "excluded_not_landing": 1,
        "source_integrity_excluded_candidates": 0,
    }
    report = evaluate_batch(
        [record_from_dict(observed_payload())],
        contexts={("KRDU", "05L"): assessment_context()},
        observed_availability=availability,
    )

    assert report["total"] == 1
    assert report["observed"] == availability


def test_computed_trajectory_that_does_not_reach_threshold_fails_the_event():
    report = evaluate_batch(
        [record_from_dict(trajectory_payload(final_lat=34.999))],
        contexts=contexts(),
    )
    row = report["trajectories"][0]
    assert row["event_status"] == "not_reached"
    assert row["verdict"] == "fail"


def test_observed_report_copies_the_policy_free_event_for_audit():
    record = record_from_dict(observed_payload())
    report = evaluate_batch(
        [record], contexts={("KRDU", "05L"): assessment_context()}
    )
    copied = report["trajectories"][0]["observed_threshold_event"]
    assert copied == record.source["observed_threshold_event"]
    assert "verdict" not in copied and "benchmark" not in copied


def _write_pair(tmp_path, *, reference_end_lat: float):
    ours = trajectory_payload()
    ours["reference_file"] = "reference.json"
    ref = trajectory_payload(subject="observed", final_lat=reference_end_lat)
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
    ours = trajectory_payload()
    ours.update(
        reference_file="reference.json",
        final_time_s=None,
        states=[],
        controls=[],
        reason="solver failed",
    )
    reference = trajectory_payload(subject="observed")
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
