"""Pipeline planning around harvest manifests, without running subprocesses."""

from __future__ import annotations

import json

import pytest

import run_scenario_pipeline as pipeline


def test_airport_discovery_includes_tracks_that_evaluate_only_will_promote(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_TRACKS_ROOT", tmp_path)
    manifest = tmp_path / "KRDU" / "tracks" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"airport": "KRDU"}), encoding="utf-8")

    assert pipeline.discover_k_airports() == ["KRDU"]


def test_dry_run_accepts_arrival_manifest_scheduled_by_observed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_TRACKS_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "SCENARIOS_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "optimization")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    tracks = pipeline.HARVEST_TRACKS_ROOT / "KRDU" / "tracks" / "manifest.json"
    tracks.parent.mkdir(parents=True)
    tracks.write_text(json.dumps({"airport": "KRDU"}), encoding="utf-8")

    assert pipeline.run_observed("KRDU", dry_run=True) is True
    assert pipeline.run_for_airport(
        "KRDU",
        "runway",
        False,
        ("eval",),
        dry_run=True,
        skip_optimize=False,
        input_will_exist=True,
    ) is True


def test_fitted_adsb_plan_uses_fitted_target_and_renamed_category(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HARVEST_TRACKS_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(pipeline, "SCENARIOS_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(pipeline, "OPT_OUTPUTS_ROOT", tmp_path / "optimization")
    monkeypatch.setattr(pipeline, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")

    plan = pipeline.Plan("KRDU", "fitted-adsb", False, ("eval",))
    assert plan.category == "fitted_adsb"
    assert plan.label == "Fitted ADS-B crossing"
    assert plan.scenarios.name == "KRDU_arrivals_fitted_adsb_scenarios.json"
    scenario_command = plan.steps()[0][1]
    assert "--target-from-fitted-adsb" in scenario_command
    assert "adsb" not in pipeline.TARGET_TYPES


def test_fitted_adsb_rejects_procedure_constraints():
    with pytest.raises(ValueError, match="incompatible"):
        pipeline.Plan("KRDU", "fitted-adsb", True, ("eval",))


def test_legacy_adsb_target_type_is_no_longer_a_mode():
    with pytest.raises(ValueError, match="unknown target type"):
        pipeline.Plan("KRDU", "adsb", False, ("eval",))
