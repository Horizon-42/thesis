"""The evaluation-to-training seam is an identity roster, not model logic."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dataset  # noqa: E402
import run_ts_pipeline  # noqa: E402
from lateral_eligibility import (  # noqa: E402
    EVALUATION_REPORT_SCHEMA,
    LATERAL_PASS_ROSTER_SCHEMA,
    build_lateral_pass_roster,
    default_evaluation_report_path,
    default_lateral_pass_roster_path,
)


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    arrivals = tmp_path / "arrivals"
    approach = tmp_path / "approach"
    arrivals.mkdir()
    approach.mkdir()
    manifest = arrivals / "manifest.json"
    records = [
        {"flight_key": key, "source_sha256": char * 64}
        for key, char in (("A", "a"), ("B", "b"), ("C", "c"))
    ]
    manifest.write_text(
        json.dumps({"airport": "KRDU", "records": records}), encoding="utf-8"
    )
    report = approach / "evaluation_report.json"
    report.write_text(
        json.dumps(
            {
                # The producer's constant, not a literal: a fixture pinned to the old
                # version is what let the v4 -> v5 bump ship green past this suite.
                "schema_version": EVALUATION_REPORT_SCHEMA,
                "subject": "observed",
                "trajectories": [
                    {
                        "airport": "KRDU",
                        "flight_key": "A",
                        "lateral_result": "pass",
                        "vertical_result": "fail",
                    },
                    {
                        "airport": "KRDU",
                        "flight_key": "B",
                        "lateral_result": "fail",
                        "vertical_result": "pass",
                    },
                    {
                        "airport": "KRDU",
                        "flight_key": "C",
                        "lateral_result": "pass",
                        "vertical_result": "indeterminate",
                    },
                    {
                        "airport": "KRDU",
                        "flight_key": "EVALUATION_ONLY",
                        "lateral_result": "pass",
                        "vertical_result": "pass",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, report


def test_lateral_pass_roster_uses_only_the_lateral_component(tmp_path: Path) -> None:
    manifest, report = _write_sources(tmp_path)
    output = default_lateral_pass_roster_path(manifest)

    roster = build_lateral_pass_roster(manifest, report, output)

    assert output.is_file()
    assert roster["schema_version"] == LATERAL_PASS_ROSTER_SCHEMA
    assert roster["airport"] == "KRDU"
    assert roster["eligible_flight_keys"] == ["A", "C"]
    assert roster["counts"] == {
        "arrival_candidates": 3,
        "eligible_lateral_pass": 2,
        "excluded_lateral_fail": 1,
        "excluded_lateral_indeterminate": 0,
        "evaluation_only": 1,
    }
    assert roster["sources"]["arrival_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert roster["sources"]["evaluation_report_sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    assert default_evaluation_report_path(manifest) == report


def test_lateral_pass_roster_rejects_missing_arrival_evaluation(tmp_path: Path) -> None:
    manifest, report = _write_sources(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["trajectories"] = [
        row for row in payload["trajectories"] if row["flight_key"] != "B"
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing evaluation result.*B"):
        build_lateral_pass_roster(
            manifest, report, default_lateral_pass_roster_path(manifest)
        )


def test_arrival_provenance_is_filtered_by_a_bound_roster(tmp_path: Path) -> None:
    manifest, report = _write_sources(tmp_path)
    roster_path = default_lateral_pass_roster_path(manifest)
    build_lateral_pass_roster(manifest, report, roster_path)

    provenance = dataset.arrival_data_provenance(
        manifest, eligibility_rosters=[roster_path]
    )
    entry = provenance["manifests"][0]

    assert [row["flight_key"] for row in entry["source_records"]] == ["A", "C"]
    assert entry["eligibility"]["policy"] == "evaluation.lateral_result == pass"
    assert entry["eligibility"]["roster_sha256"] == hashlib.sha256(
        roster_path.read_bytes()
    ).hexdigest()
    assert sum(
        len(keys)
        for keys in dataset.flight_keys_by_split(provenance, dataset.TSConfig()).values()
    ) == 2


def test_arrival_provenance_rejects_roster_bound_to_old_manifest(tmp_path: Path) -> None:
    manifest, report = _write_sources(tmp_path)
    roster_path = default_lateral_pass_roster_path(manifest)
    build_lateral_pass_roster(manifest, report, roster_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"].append({"flight_key": "D", "source_sha256": "d" * 64})
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different arrival manifest"):
        dataset.arrival_data_provenance(
            manifest, eligibility_rosters=[roster_path]
        )


def test_pipeline_passes_one_lateral_roster_per_manifest() -> None:
    plan = run_ts_pipeline.TrainingPlan(
        ("KRDU", "KMSY"),
        "itransformer",
        training_mode="pooled",
    )

    args = plan._data_args()

    assert args.count("--data") == 2
    assert args.count("--eligibility-roster") == 2
    assert set(plan.eligibility_rosters) == {
        default_lateral_pass_roster_path(path) for path in plan.data_manifests
    }
    prediction = run_ts_pipeline.PredictionPlan(
        plan, "KRDU", ("eval",), split="val"
    )
    predict_command = prediction.steps()[0][1]
    assert predict_command[predict_command.index("--eligibility-roster") + 1] == str(
        default_lateral_pass_roster_path(prediction.data_manifest)
    )
