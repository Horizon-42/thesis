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
    # The label is the canonical run grammar (run_naming) plus the run id as meta.
    assert plan.category_label.startswith(
        "Validation split (model selection) — Experiment: control · "
    )
    assert plan.category_label.endswith("run_seed1337")
    assert plan.experiment_metadata["horizonMode"] == "normalized"
    assert plan.experiment_metadata["label"].endswith("run_seed1337")


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


def test_refresh_labels_only_walks_publication_manifests(monkeypatch, tmp_path):
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
    _write_json(plan.publication_manifest, {
        "schemaVersion": publisher.PUBLICATION_SCHEMA,
        "status": "completed",
        "experimentId": experiment.experiment_id,
        "campaign": experiment.campaign,
        "runId": experiment.run_id,
        "checkpoint": experiment.checkpoint_relative,
        "airport": "KRDU",
        "split": "val",
        "resultSource": "experiment",
        "category": plan.category,
        "config": experiment.config,
    })
    manifest = plan.comparison_dir.parent / "categories.json"
    _write_json(manifest, {"categories": [{"key": plan.category, "label": "legacy"}]})

    seen, patched = publisher.refresh_labels_from_manifests(
        tmp_path / "published", tmp_path / "frontend"
    )

    assert (seen, patched) == (1, 1)
    category = json.loads(manifest.read_text())["categories"][0]
    assert category["label"] == plan.category_label
    assert category["experiment"]["label"] == plan.experiment_metadata["label"]
    assert category["resultSource"] == "experiment"


def test_reused_prediction_dir_skips_predict_and_is_never_archived(monkeypatch, tmp_path):
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    campaign_records = tmp_path / "experiments" / "campaign" / "stage" / "run_seed1337_pred_val"
    _write_json(campaign_records / "summary.json", {
        "checkpoint": str(checkpoint), "split": "val", "results": [],
    })
    (campaign_records / "flight_states.json").write_text("{}", encoding="utf-8")
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=campaign_records,
    )

    commands = dict(plan.commands())
    assert "predict" not in commands
    assert commands["evaluate"][commands["evaluate"].index("--input") + 1] == str(campaign_records)
    publish = commands["publish-czml"]
    assert publish[publish.index("--summary") + 1] == str(campaign_records / "summary.json")
    # The evaluation report and the publication manifest stay in the publisher's own directory,
    # so the campaign's prediction directory is only ever read.
    assert plan.evaluation_report.parent == plan.output_dir
    assert plan.output_dir != campaign_records
    assert publisher._loose_prediction_records(plan.output_dir) == []


# ── --category-variant: two prediction dirs, one checkpoint ─────────────────

def _plain_prediction_dir(tmp_path: Path, checkpoint: Path, name: str) -> Path:
    """A prediction directory with no anytime block — an ordinary predict run."""
    directory = tmp_path / name
    _write_json(directory / "summary.json", {
        "checkpoint": str(checkpoint), "split": "val",
        "results": [{"id": "AAL1", "arr_airport": "KRDU"}],
    })
    return directory


def _variant_plan(tmp_path, experiment, directory, variant):
    return publisher.PublicationPlan(
        experiment, "KRDU", "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=directory,
        variant=variant,
    )


def test_a_category_variant_separates_two_publications_of_one_checkpoint(
    monkeypatch, tmp_path,
):
    """A predict-time variant (an inference projection, a hook applied only at predict) has
    no checkpoint of its own — it reuses its baseline's. Everything a category is named from
    comes from that checkpoint, so without a variant the two would be one category."""
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    baseline = publisher.PublicationPlan(
        experiment, "KRDU", "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=_plain_prediction_dir(tmp_path, checkpoint, "base_pred_val"),
    )
    projected = _variant_plan(
        tmp_path, experiment,
        _plain_prediction_dir(tmp_path, checkpoint, "project_pred_val"),
        publisher._parse_category_variant("project-on-final=projection, on-final gate"),
    )

    assert projected.category == f"{baseline.category[: -len('_val')]}_project-on-final_val"
    assert projected.category != baseline.category
    assert projected.output_dir == baseline.output_dir / "project-on-final"
    metadata = projected.experiment_metadata
    assert metadata["id"] == f"{experiment.experiment_id}@project-on-final"
    assert metadata["id"] != baseline.experiment_metadata["id"]
    # The variant does NOT move the picker heading: this is still the training campaign's arm.
    assert metadata["group"] == baseline.experiment_metadata["group"] == experiment.campaign
    assert metadata["label"].endswith("projection, on-final gate")
    assert projected.category_label.endswith(metadata["label"])
    publish = dict(projected.commands())["publish-czml"]
    assert publish[publish.index("--experiment-id") + 1] == metadata["id"]


def test_a_variant_slug_defaults_its_own_label_and_rejects_an_unusable_one():
    plain = publisher._parse_category_variant("barrier-infer-soft")
    assert plain.key_suffix == plain.id_suffix == plain.label_suffix == "barrier-infer-soft"
    assert plain.group is None
    # The slug becomes part of a category key, so a leading digit or a path separator is out.
    for bad in ("12km", "a/b", "", "with space", "slug="):
        with pytest.raises(ValueError):
            publisher._parse_category_variant(bad)


def test_a_second_prediction_dir_on_one_category_is_refused_not_overwritten(
    monkeypatch, tmp_path,
):
    """The whole point of the variant: without one, the second publication would silently
    replace the first's CZML, evaluation report and picker entry."""
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "harvest" / "KRDU" / "arrivals" / "manifest.json"
    _write_json(manifest, {})
    monkeypatch.setattr(publisher, "_sha256", lambda path: (
        "manifest-sha" if path == manifest else experiment.checkpoint_sha256
    ))
    first = _plain_prediction_dir(tmp_path, checkpoint, "base_pred_val")
    second = _plain_prediction_dir(tmp_path, checkpoint, "project_pred_val")
    published = publisher.PublicationPlan(
        experiment, "KRDU", "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=first,
    )
    _write_json(published.publication_manifest, {
        "schemaVersion": publisher.PUBLICATION_SCHEMA, "status": "completed",
        "predictionDir": publisher._path_for_manifest(first),
    })

    # The same directory again is a legitimate republication...
    assert published.preflight_error() is None
    # ...a different one under the same name is not.
    collides = publisher.PublicationPlan(
        experiment, "KRDU", "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=second,
    )
    assert "--category-variant" in (collides.preflight_error() or "")
    # ...and giving it one clears the collision by giving it its own category.
    assert _variant_plan(
        tmp_path, experiment, second,
        publisher._parse_category_variant("project-on-final"),
    ).preflight_error() is None


def test_a_directory_that_names_its_own_bin_refuses_a_second_identity(monkeypatch, tmp_path):
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = _variant_plan(
        tmp_path, experiment, _anytime_records(tmp_path, checkpoint),
        publisher._parse_category_variant("something-else"),
    )

    with pytest.raises(ValueError, match="one identity"):
        _ = plan.category


def test_a_variant_publication_refreshes_from_its_manifest(monkeypatch, tmp_path):
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = _variant_plan(
        tmp_path, experiment,
        _plain_prediction_dir(tmp_path, checkpoint, "project_pred_val"),
        publisher._parse_category_variant("project-faf=projection, FAF gate"),
    )
    _write_json(plan.evaluation_report, {"trajectories": [], "summary": {}})
    _write_json(plan.publication_manifest, publisher._publication_document(
        plan, status="completed", completed_steps=("evaluate", "publish-czml"),
    ))
    categories = plan.comparison_dir.parent / "categories.json"
    _write_json(categories, {"categories": [{"key": plan.category, "label": "stale"}]})

    seen, patched = publisher.refresh_labels_from_manifests(
        tmp_path / "published", tmp_path / "frontend"
    )

    assert (seen, patched) == (1, 1)
    entry = json.loads(categories.read_text())["categories"][0]
    assert entry["label"] == plan.category_label
    assert entry["experiment"] == plan.experiment_metadata


# ── anytime bins: one category per re-anchored bin ──────────────────────────

def _anytime_records(tmp_path: Path, checkpoint: Path, *, limit: int = 0,
                     records: int = 236, schema: str | None = None) -> Path:
    """A record directory as `run_ts_anytime_curve.py --write-records` writes one."""
    directory = tmp_path / "anytime_records_20260908" / "records" / "L1_native32" / "12km"
    _write_json(directory / "summary.json", {
        "checkpoint": str(checkpoint), "split": "val",
        "results": [{"id": "AAL1", "arr_airport": "KRDU"}],
        "anytime": {
            "schema": schema or publisher.ANYTIME_RECORDS_SCHEMA,
            "campaign": "anytime_records_20260908",
            "arm": "A0-fixed",
            "label": "L1_native32",
            "bin_m": 12000.0,
            "bin_label": "12km",
            "anchor_rule": "the closest sample",
            "min_future_s": 60.0,
            "split": "val",
            "limit": limit,
            "split_flights": 1404,
            "measured_flights": limit or 1404,
            "records": records,
            "coverage": records / (limit or 1404),
        },
    })
    return directory


def _anytime_plan(tmp_path: Path, experiment, directory: Path) -> "publisher.PublicationPlan":
    return publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=directory,
    )


def test_an_anytime_bin_publishes_as_its_own_category_under_the_campaign(
    monkeypatch, tmp_path,
):
    """A bin is a re-anchored SUBSET of the split, so it gets its own key, its own picker
    entry (the picker dedupes by experiment id) and the campaign as its group — otherwise
    only the first bin published would ever be selectable, under the training campaign."""
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = _anytime_plan(tmp_path, experiment, _anytime_records(tmp_path, checkpoint))
    plain = publisher.PublicationPlan(
        experiment, "KRDU", "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
    )

    assert plan.category == f"{plain.category[: -len('_val')]}_a12km_val"
    assert plan.category != plain.category
    assert plan.experiment_group == "anytime_records_20260908"
    metadata = plan.experiment_metadata
    assert metadata["id"] == f"{experiment.experiment_id}@12km"
    assert metadata["id"] != plain.experiment_metadata["id"]
    assert metadata["group"] == "anytime_records_20260908"
    assert metadata["label"].endswith("@ 12km remaining (236 of 1404 flights)")
    assert plan.category_label.endswith(metadata["label"])
    # Its own evaluation report and publication manifest: bins must not overwrite each
    # other's verdicts in the split's directory.
    assert plan.output_dir == plain.output_dir / "a12km"
    publish = dict(plan.commands())["publish-czml"]
    assert publish[publish.index("--experiment-id") + 1] == metadata["id"]
    assert publish[publish.index("--experiment-group") + 1] == "anytime_records_20260908"


def test_an_anytime_label_states_the_limit_it_was_measured_under(monkeypatch, tmp_path):
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    directory = _anytime_records(tmp_path, checkpoint, limit=300, records=236)

    plan = _anytime_plan(tmp_path, experiment, directory)

    assert plan.experiment_metadata["label"].endswith(
        "@ 12km remaining (236 of 300 flights, --limit 300 of 1404 in the split)"
    )


def test_an_anytime_publication_manifest_refreshes_to_the_same_label(monkeypatch, tmp_path):
    """The bin rides in the manifest, so a label refresh never reopens the record directory
    (which is read-only, and may have been archived elsewhere) to recompute the same name."""
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    plan = _anytime_plan(
        tmp_path, experiment, _anytime_records(tmp_path, checkpoint, limit=300)
    )
    _write_json(plan.evaluation_report, {"trajectories": [], "summary": {}})
    _write_json(plan.publication_manifest, publisher._publication_document(
        plan, status="completed", completed_steps=("evaluate", "publish-czml"),
    ))
    manifest = plan.comparison_dir.parent / "categories.json"
    _write_json(manifest, {"categories": [{"key": plan.category, "label": "stale"}]})

    seen, patched = publisher.refresh_labels_from_manifests(
        tmp_path / "published", tmp_path / "frontend"
    )

    assert (seen, patched) == (1, 1)
    category = json.loads(manifest.read_text())["categories"][0]
    assert category["label"] == plan.category_label
    assert category["experiment"] == plan.experiment_metadata


def test_a_pooled_anytime_bin_is_refused_rather_than_filed_under_one_airport(
    monkeypatch, tmp_path,
):
    """The runner replays the checkpoint's WHOLE split and offers no airport narrowing, so a
    pooled checkpoint's bin is one cohort over every airport it was trained on. The publisher
    fans out one plan per airport — publishing that directory would file all five airports'
    flights under each airport's category and print the pooled count as that airport's."""
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "harvest" / "KRDU" / "arrivals" / "manifest.json"
    _write_json(manifest, {})
    monkeypatch.setattr(publisher, "_sha256", lambda path: (
        "manifest-sha" if path == manifest else experiment.checkpoint_sha256
    ))
    directory = _anytime_records(tmp_path, checkpoint)
    summary = json.loads((directory / "summary.json").read_text())
    summary["results"] = [{"id": "AAL1", "arr_airport": "KRDU"},
                          {"id": "UAL2", "arr_airport": "KSJC"}]
    _write_json(directory / "summary.json", summary)

    plan = _anytime_plan(tmp_path, experiment, directory)

    assert "not just KRDU" in (plan.preflight_error() or "")


def test_the_same_checkpoint_cannot_be_given_two_prediction_dirs(monkeypatch, tmp_path):
    """Publishing several bins of one checkpoint is the expected workflow, and the obvious
    attempt is to repeat the flag — which used to keep the last one silently."""
    index, checkpoint = _indexed_checkpoint(tmp_path)
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    directory = _anytime_records(tmp_path, checkpoint)

    with pytest.raises(SystemExit):
        publisher.main([
            "--experiment-index", str(index),
            "--checkpoint", "campaign/stage/run_seed1337",
            "--reuse-prediction-dir", f"campaign/stage/run_seed1337={directory}",
            "--reuse-prediction-dir", f"campaign/stage/run_seed1337={directory.parent}",
        ])


def test_an_unknown_anytime_schema_is_refused_rather_than_published_as_the_split(
    monkeypatch, tmp_path,
):
    index, checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    directory = _anytime_records(tmp_path, checkpoint, schema="ts-anytime-records-v99")

    plan = _anytime_plan(tmp_path, experiment, directory)
    with pytest.raises(ValueError, match="unknown schema"):
        _ = plan.category


def test_reused_prediction_dir_must_match_the_checkpoint_and_split(monkeypatch, tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    manifest = tmp_path / "harvest" / "KRDU" / "arrivals" / "manifest.json"
    _write_json(manifest, {})
    monkeypatch.setattr(
        publisher, "_sha256", lambda path: (
            "manifest-sha" if path == manifest else experiment.checkpoint_sha256
        ),
    )
    foreign = tmp_path / "foreign_pred_val"
    _write_json(foreign / "summary.json", {
        "checkpoint": str(tmp_path / "other" / "checkpoint.pt"), "split": "val",
    })
    plan = publisher.PublicationPlan(
        experiment,
        "KRDU",
        "val",
        raw_output_root=tmp_path / "published",
        harvest_root=tmp_path / "harvest",
        frontend_airports_root=tmp_path / "frontend",
        prediction_dir=foreign,
    )

    assert "were produced by" in (plan.preflight_error() or "")

    _write_json(foreign / "summary.json", {
        "checkpoint": str(experiment.checkpoint), "split": "train",
    })
    assert "'train' split" in (plan.preflight_error() or "")

    _write_json(foreign / "summary.json", {
        "checkpoint": str(experiment.checkpoint), "split": "val",
    })
    assert plan.preflight_error() is None


def test_reused_prediction_dir_requires_a_summary(tmp_path):
    index, _checkpoint = _indexed_checkpoint(tmp_path)
    experiment = publisher.discover_checkpoints(index)[0]

    with pytest.raises(ValueError, match="no summary.json"):
        publisher.PublicationPlan(
            experiment, "KRDU", "val", prediction_dir=tmp_path / "nowhere",
        )


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
