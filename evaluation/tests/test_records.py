"""The record boundary and the manifest-only read side.

Everything the evaluator reads off a record's ``source`` is checked once, here at the
boundary; the batch read side follows ``summary.json`` and never globs or falls back to
loading the whole batch to discover what it contains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evaluation import evaluate_batch, record_from_dict
from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG, contexts_for_input
from evaluation.records import iter_records, record_files, roster_context_keys
from evaluation.tests.factories import (
    assessment_context,
    observed_payload,
    trajectory_payload,
    write_batch,
)


@pytest.mark.parametrize("key", ["arr_airport", "runway"])
def test_the_airport_and_runway_are_required_at_the_boundary(key):
    """They select the assessment context a verdict is measured against; a record that
    cannot name them used to be accepted and fail later, mid-batch, at first use."""
    value = trajectory_payload()
    del value["source"][key]
    with pytest.raises(ValueError, match=f"source.{key}"):
        record_from_dict(value)
    value["source"][key] = ""
    with pytest.raises(ValueError, match=f"source.{key}"):
        record_from_dict(value)


def test_an_observed_record_with_an_event_requires_its_datum_offset():
    value = observed_payload()
    del value["source"]["hae_minus_msl_m"]
    with pytest.raises(ValueError, match="hae_minus_msl_m"):
        record_from_dict(value)


def test_a_reference_track_without_an_event_needs_no_datum_offset():
    """The optimizer's and ts_transformer's comparison references are observed records
    that never pass through runway assignment: no event, nothing to cross-check."""
    value = trajectory_payload(subject="observed")
    del value["source"]["hae_minus_msl_m"]
    assert record_from_dict(value).source.get("observed_threshold_event") is None


def test_the_harvest_observed_roster_names_its_airport_once(tmp_path):
    """``harvest.observed`` writes the airport at the top of the roster and only the
    runway per row; the modeling batches name it per row. Both resolve."""
    batch = write_batch(tmp_path, [trajectory_payload(), trajectory_payload()])
    manifest = batch / "summary.json"
    summary = json.loads(manifest.read_text())
    for row in summary["results"]:
        del row["arr_airport"]
    summary["airport"] = "KRDU"
    manifest.write_text(json.dumps(summary), encoding="utf-8")
    assert roster_context_keys(batch) == [("KRDU", "05L"), ("KRDU", "05L")]


def test_the_state_clock_must_increase_strictly():
    """``t`` is the one channel with an ordering contract: the crossing interpolation
    and every flight-time delta read it."""
    value = trajectory_payload()
    value["states"].insert(1, dict(value["states"][0], t=150.0))
    value["controls"].insert(1, dict(value["controls"][0]))
    with pytest.raises(ValueError, match=r"states\[2\]\.t"):
        record_from_dict(value)
    value["states"][1]["t"] = 0.0     # a tie is not increasing either
    with pytest.raises(ValueError, match=r"states\[1\]\.t"):
        record_from_dict(value)


def test_a_rostered_batch_is_read_through_its_roster_only(tmp_path):
    batch = write_batch(tmp_path, [trajectory_payload(), trajectory_payload()])
    (tmp_path / "orphan_eval.json").write_text(
        json.dumps(trajectory_payload()), encoding="utf-8"
    )
    files = record_files(batch)
    assert [file.name for file in files] == ["record_0_eval.json", "record_1_eval.json"]
    assert roster_context_keys(batch) == [("KRDU", "05L"), ("KRDU", "05L")]
    assert sum(1 for _ in iter_records(batch)) == 2


def test_a_roster_row_that_cannot_name_its_airport_raises_instead_of_loading_the_batch(
    tmp_path,
):
    """The old fallback silently switched to materializing every record (~1 MB each)."""
    batch = write_batch(tmp_path, [trajectory_payload()])
    manifest = batch / "summary.json"
    rows = json.loads(manifest.read_text())
    rows["results"][0]["arr_airport"] = None
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="names no runway, or no airport"):
        roster_context_keys(batch)


def test_a_roster_that_lists_a_missing_record_raises_with_the_file_named(tmp_path):
    batch = write_batch(tmp_path, [trajectory_payload()])
    (batch / "record_0_eval.json").unlink()
    with pytest.raises(ValueError, match="missing record"):
        record_files(batch)


def test_a_directory_without_a_manifest_is_rejected_with_the_crafted_message(tmp_path):
    (tmp_path / "loose_eval.json").write_text(
        json.dumps(trajectory_payload()), encoding="utf-8"
    )
    assert roster_context_keys(tmp_path) is None
    with pytest.raises(ValueError, match="has no summary.json manifest"):
        record_files(tmp_path)
    args = argparse.Namespace(config=DEFAULT_CONFIG, cifp=DEFAULT_CIFP)
    with pytest.raises(ValueError, match="has no summary.json manifest"):
        contexts_for_input(tmp_path, args)


def test_a_loose_record_file_resolves_its_own_airport(tmp_path):
    loose = tmp_path / "one_eval.json"
    loose.write_text(json.dumps(trajectory_payload()), encoding="utf-8")
    assert roster_context_keys(loose) is None
    assert record_files(loose) == [loose]
    args = argparse.Namespace(config=DEFAULT_CONFIG, cifp=DEFAULT_CIFP)
    contexts = contexts_for_input(loose, args)
    assert ("KRDU", "05L") in contexts


def test_observed_availability_is_refused_at_the_first_computed_record():
    availability = {
        "denominator": "arrival_candidates_excluding_not_landing",
        "event_denominator": 1, "event_estimated": 1, "event_unavailable": 0,
        "event_estimated_rate": 1.0, "excluded_not_landing": 0,
        "source_integrity_excluded_candidates": 0,
    }
    contexts = {("KRDU", "05L"): assessment_context()}
    with pytest.raises(ValueError, match="observed-only batch; record 'TEST1' is 'optimized'"):
        evaluate_batch(
            [record_from_dict(observed_payload()), record_from_dict(trajectory_payload())],
            contexts=contexts,
            observed_availability=availability,
        )
    with pytest.raises(ValueError, match="denominator"):
        evaluate_batch(
            [], contexts=contexts, observed_availability={**availability, "denominator": "x"}
        )


def test_an_empty_batch_names_itself_empty_not_mixed():
    report = evaluate_batch([], contexts={("KRDU", "05L"): assessment_context()})
    assert report["subject"] == "empty"
    assert report["total"] == 0
    assert report["verdict_counts"] == {"pass": 0, "fail": 0, "indeterminate": 0}


def test_the_composite_fails_on_any_failed_component_even_with_an_open_one():
    """Fail beats indeterminate: a record with no vertical reference but a lateral
    miss is a failed approach, not an open question."""
    context = assessment_context(benchmark="rnp_apch_lnav_vnav_baro", baro_vnav_approved=False)
    value = trajectory_payload(final_lon=-77.999)          # ~90 m right of centreline
    report = evaluate_batch([record_from_dict(value)], contexts={("KRDU", "05L"): context})
    [row] = report["trajectories"]
    assert row["lateral_result"] == "fail"
    assert row["vertical_result"] == "indeterminate"
    assert row["verdict"] == "fail"
    assert row["violations"] == ["lateral"]
