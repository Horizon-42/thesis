"""Top-level TS orchestration must pass the same arrival manifest as the TS loader."""

from __future__ import annotations

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
