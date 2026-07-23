"""Top-level TS orchestration must pass the same arrival manifest as the TS loader."""

from __future__ import annotations

import hashlib
import json

import run_ts_pipeline as pipeline


def test_ts_pipeline_discovers_and_passes_arrival_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    manifest = pipeline.HARVEST_ROOT / "KRDU" / "arrivals" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"schema_version": "harvest-arrivals-v1", "airport": "KRDU"}),
        encoding="utf-8",
    )

    assert pipeline.discover_k_airports() == ["KRDU"]
    plan = pipeline.Plan("KRDU", "itransformer", "window", ("eval",))
    assert plan.data_manifest == manifest
    commands = [command for _label, command in plan.steps()]
    assert all(str(manifest) in command for command in commands[:2])
    assert pipeline.run_cell(plan, dry_run=True, skip_train=False) is True


def test_prediction_outputs_and_frontend_categories_are_split_specific(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")

    test = pipeline.Plan("KRDU", "itransformer", "window", ("czml",), split="test")
    train = pipeline.Plan("KRDU", "itransformer", "window", ("czml",), split="train")

    assert test.pred_dir != train.pred_dir
    assert test.category != train.category
    assert test.pred_dir.name.endswith("_test")
    assert train.category.endswith("_train")
    assert "test" in test.label.lower()
    assert "train" in train.label.lower()


def test_skip_train_rejects_checkpoint_for_a_different_arrival_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "outputs")
    manifest = pipeline.HARVEST_ROOT / "KRDU" / "arrivals" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"generation":2}', encoding="utf-8")

    plan = pipeline.Plan("KRDU", "itransformer", "window", ("eval",))
    plan.train_dir.mkdir(parents=True)
    plan.checkpoint.write_bytes(b"checkpoint")
    plan.checkpoint_metadata.write_text(json.dumps({
        "schema_version": pipeline.CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": hashlib.sha256(plan.checkpoint.read_bytes()).hexdigest(),
        "arrival_manifest_sha256": hashlib.sha256(b'{"generation":1}').hexdigest(),
    }), encoding="utf-8")

    assert "arrival manifest" in (plan.checkpoint_reuse_error() or "")
    assert plan.checkpoint_exists() is False
