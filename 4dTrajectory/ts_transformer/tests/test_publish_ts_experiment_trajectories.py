import hashlib
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

import publish_ts_experiment_trajectories as publisher


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _indexed_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "experiments"
    run = root / "campaign" / "stage" / "run_seed1337"
    checkpoint = run / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _write_json(run / "checkpoint_metadata.json", {
        "checkpoint_sha256": checkpoint_sha,
        "arrival_manifests": {"KRDU": "manifest-sha"},
    })
    _write_json(run / "history.json", {
        "config": {
            "model": "itransformer",
            "prediction_output": "control",
            "horizon_mode": "normalized",
            "seed": 1337,
        },
    })
    index = root / "index.json"
    _write_json(index, {
        "root": str(root),
        "entries": [
            {
                "path": "campaign/stage/run_seed1337",
                "campaign_id": "campaign",
                "run_id": "stage_run_seed1337",
                "kind": "training",
                "status": "completed",
                "artifacts": ["checkpoint.pt", "history.json"],
            },
            {
                "path": "campaign/comparison",
                "kind": "comparison",
                "status": "completed",
                "artifacts": ["report.json"],
            },
        ],
    })
    return index, checkpoint


def test_discovers_only_completed_training_checkpoints(tmp_path):
    index, checkpoint = _indexed_checkpoint(tmp_path)

    discovered = publisher.discover_checkpoints(index)

    assert len(discovered) == 1
    assert discovered[0].experiment_id == "campaign/stage/run_seed1337"
    assert discovered[0].campaign == "campaign"
    assert discovered[0].checkpoint == checkpoint
    assert discovered[0].config["prediction_output"] == "control"


def test_publication_plan_reuses_existing_prediction_evaluation_and_czml_contract(
    monkeypatch, tmp_path,
):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        device="cuda",
    )

    commands = dict(plan.commands())
    assert commands["predict"][2:4] == ["predict", "--checkpoint"]
    assert commands["predict"][-4:] == ["--split", "val", "--device", "cuda"]
    assert commands["evaluate"][1:3] == ["-m", "evaluation"]
    assert "--result-source" in commands["publish-czml"]
    assert commands["publish-czml"][commands["publish-czml"].index("--result-source") + 1] \
        == "experiment"
    assert commands["publish-czml"][commands["publish-czml"].index("--dataset-split") + 1] \
        == "val"
    assert plan.category.endswith("_val")
    assert "comparison" in plan.comparison_dir.parts
    assert "horizon: normalized time" in plan.category_label
    assert plan.experiment_metadata["horizonMode"] == "normalized"


def test_prediction_source_uses_prediction_category_without_experiment_metadata(tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "train",
        result_source="prediction",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
    )

    publish = dict(plan.commands())["publish-czml"]
    assert publish[publish.index("--result-source") + 1] == "prediction"
    assert "--experiment-id" not in publish
    assert plan.category.startswith("prediction_")
    assert "Predicted" in plan.category_label


def test_refreshes_horizon_metadata_for_an_archived_publication(monkeypatch, tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
    )
    manifest = plan.comparison_dir.parent / "categories.json"
    _write_json(manifest, {"categories": [{
        "key": plan.category,
        "label": "legacy label",
        "experiment": {"id": experiment.experiment_id},
    }]})

    assert publisher.refresh_category_metadata(plan)

    category = json.loads(manifest.read_text())["categories"][0]
    assert category["label"] == plan.category_label
    assert category["experiment"] == plan.experiment_metadata


def test_publication_plan_cannot_access_outer_test(tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]

    with pytest.raises(ValueError, match="development splits"):
        publisher.PublicationPlan(experiment, "KRDU", "test")


def test_rebuild_index_keeps_metrics_and_failure_reasons(tmp_path):
    completed = tmp_path / "one" / "KRDU" / "val" / publisher.PUBLICATION_MANIFEST
    blocked = tmp_path / "two" / "KRDU" / "train" / publisher.PUBLICATION_MANIFEST
    _write_json(completed, {
        "schemaVersion": publisher.PUBLICATION_SCHEMA,
        "status": "completed",
        "accuracy": {"ade_m": {"mean": 100.0}},
        "evaluation": {"success_rate": 0.5},
    })
    _write_json(blocked, {
        "schemaVersion": publisher.PUBLICATION_SCHEMA,
        "status": "blocked",
        "failure": "manifest mismatch",
    })

    document = publisher.rebuild_publication_index(tmp_path)

    assert document["counts"]["completed"] == 1
    assert document["counts"]["blocked"] == 1
    assert document["publications"][0]["accuracy"]["ade_m"]["mean"] == 100.0
    assert document["publications"][1]["failure"] == "manifest mismatch"


def test_archive_retains_exact_records_and_keeps_aggregate_outputs(monkeypatch, tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "train",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
    )
    plan.output_dir.mkdir(parents=True)
    (plan.output_dir / "references").mkdir()
    (plan.output_dir / "one_states.json").write_text("states")
    (plan.output_dir / "one_eval.json").write_text("eval")
    (plan.output_dir / "references" / "one_reference_eval.json").write_text("reference")
    (plan.output_dir / "summary.json").write_text("summary")
    (plan.output_dir / "evaluation_report.json").write_text("evaluation")

    archived = publisher.archive_prediction_records(plan)

    assert archived == 3
    assert not (plan.output_dir / "one_states.json").exists()
    assert not (plan.output_dir / "references").exists()
    assert (plan.output_dir / "summary.json").read_text() == "summary"
    assert (plan.output_dir / "evaluation_report.json").read_text() == "evaluation"
    with tarfile.open(plan.records_archive, "r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == [
            "one_states.json",
            "one_eval.json",
            "references/one_reference_eval.json",
        ]

    # Simulate cleanup interruption: one loose member survives after the complete archive
    # was committed. Recovery must preserve the full archive instead of replacing it with
    # this one-file subset.
    (plan.output_dir / "one_eval.json").write_text("eval")
    assert publisher.archive_prediction_records(plan) == 1
    with tarfile.open(plan.records_archive, "r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == [
            "one_states.json",
            "one_eval.json",
            "references/one_reference_eval.json",
        ]


def test_interrupt_marks_current_job_failed_and_terminates_batch(monkeypatch, tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "harvest" / "KRDU" / "arrivals" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"manifest")
    experiment = replace(
        experiment,
        arrival_manifests={"KRDU": hashlib.sha256(b"manifest").hexdigest()},
    )
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
    )

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(publisher.subprocess, "run", interrupt)

    with pytest.raises(KeyboardInterrupt):
        publisher.run_publication(plan, dry_run=False, force=False, fail_fast=False)

    publication = json.loads(plan.publication_manifest.read_text())
    assert publication["status"] == "failed"
    assert publication["failure"].startswith("KeyboardInterrupt")


def test_main_returns_failure_when_any_publication_is_blocked(monkeypatch, tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    monkeypatch.setattr(
        publisher,
        "run_publication",
        lambda *_args, **_kwargs: "blocked",
    )

    exit_code = publisher.main(["--experiment-index", str(index)])

    assert exit_code != 0


def test_normalizes_repository_relative_checkpoint_path():
    assert publisher._normalize_checkpoint_id(
        "4dTrajectory/outputs/POOLED/experiments/"
        "campaign/stage/run_seed1337/checkpoint.pt"
    ) == "campaign/stage/run_seed1337"


def test_publication_manifest_serializes_external_output_roots(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    index, _checkpoint = _indexed_checkpoint(repository)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", repository)
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "external-output",
        harvest_root=tmp_path / "external-harvest",
        frontend_airports_root=tmp_path / "external-frontend",
    )

    document = publisher._publication_document(plan, status="running")

    assert document["rawOutputDir"] == str(plan.output_dir)
    assert document["frontendDir"] == str(plan.comparison_dir)
