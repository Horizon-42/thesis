"""Preparation and optimization pipeline planning, without running subprocesses."""

from __future__ import annotations

import json
import hashlib
import subprocess

import pytest

import prepare_scenario_inputs as prepare
import run_scenario_optimization as optimize


def test_prepare_long_command_reports_elapsed_heartbeat(monkeypatch, capsys):
    class FakeProcess:
        def __init__(self):
            self.waits = 0

        def wait(self, *, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return 0

    monkeypatch.setattr(prepare.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    ticks = iter((10.0, 41.0, 42.0))
    monkeypatch.setattr(prepare.time, "monotonic", lambda: next(ticks))

    prepare._run_command_with_progress(["fake"], label="KRDU observed", interval_s=30.0)

    output = capsys.readouterr().out
    assert "[progress] KRDU observed: started" in output
    assert "[progress] KRDU observed: still running (31s elapsed)" in output
    assert "[progress] KRDU observed: completed in 32s" in output


def test_airport_discovery_includes_tracks_that_evaluate_only_will_promote(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "HARVEST_TRACKS_ROOT", tmp_path)
    manifest = tmp_path / "KRDU" / "tracks" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"airport": "KRDU"}), encoding="utf-8")

    assert prepare.discover_k_airports() == ["KRDU"]


def test_dry_run_accepts_arrival_manifest_scheduled_by_observed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "HARVEST_TRACKS_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(prepare, "SCENARIOS_DIR", tmp_path / "scenarios")
    tracks = prepare.HARVEST_TRACKS_ROOT / "KRDU" / "tracks" / "manifest.json"
    tracks.parent.mkdir(parents=True)
    tracks.write_text(json.dumps({"airport": "KRDU"}), encoding="utf-8")

    assert prepare.run_observed("KRDU", dry_run=True) is True
    assert prepare.run_for_airport(
        "KRDU",
        "runway",
        dry_run=True,
        input_will_exist=True,
    ) is True


def test_preparation_builds_each_distinct_target_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "HARVEST_TRACKS_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(prepare, "SCENARIOS_DIR", tmp_path / "scenarios")

    fitted = prepare.scenario_command("KRDU", "fitted-adsb")
    runway = prepare.scenario_command("KRDU", "runway")
    assert fitted[-1] == "--target-from-fitted-adsb"
    assert runway[-1] == "--target-from-threshold"
    assert any(part.endswith("KRDU_arrivals_fitted_adsb_scenarios.json") for part in fitted)
    assert any(part.endswith("KRDU_arrivals_threshold_scenarios.json") for part in runway)


def test_optimizer_uses_prepared_scenario_without_rebuilding_it(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "HARVEST_TRACKS_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(optimize, "SCENARIOS_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(optimize, "OPT_OUTPUTS_ROOT", tmp_path / "optimization")
    monkeypatch.setattr(optimize, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")

    plan = optimize.Plan("KRDU", "fitted-adsb", False, ("eval",))
    assert plan.category == "fitted_adsb"
    assert plan.label == "Fitted ADS-B crossing"
    assert plan.scenarios.name == "KRDU_arrivals_fitted_adsb_scenarios.json"
    commands = [cmd for _, cmd in plan.steps()]
    assert str(optimize.OPT_SCRIPT) in commands[0]
    assert commands[0][commands[0].index("--references-dir") + 1] == \
        "../shared_references/fitted_adsb"
    assert all("flight_scenarios" not in cmd for command in commands for cmd in command)
    assert "adsb" not in optimize.TARGET_TYPES


def test_optimizer_discovers_only_prepared_k_airports(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "SCENARIOS_DIR", tmp_path)
    (tmp_path / "KRDU_arrivals_fitted_adsb_scenarios.json").touch()
    (tmp_path / "KMSY_arrivals_threshold_scenarios.json").touch()
    (tmp_path / "CYYC_arrivals_threshold_scenarios.json").touch()

    assert optimize.discover_k_airports() == ["KMSY", "KRDU"]


def test_fitted_adsb_rejects_procedure_constraints():
    with pytest.raises(ValueError, match="incompatible"):
        optimize.Plan("KRDU", "fitted-adsb", True, ("eval",))


def test_runway_modes_share_one_canonical_reference_directory():
    runway = optimize.Plan("KRDU", "runway", False, ("eval",))
    constrained = optimize.Plan("KRDU", "runway", True, ("eval",))
    assert runway.references_dir == constrained.references_dir == \
        "../shared_references/runway"


def test_comparison_publication_always_uses_its_committed_evaluation_report():
    plan = optimize.Plan("KRDU", "runway", False, ("czml",))
    steps = plan.steps(reuse=True)

    assert [label.split(" ", 1)[1] for label, _ in steps] == [
        "evaluation report",
        "comparison CZML",
    ]
    comparison_cmd = steps[-1][1]
    report_flag = comparison_cmd.index("--evaluation-report")
    assert comparison_cmd[report_flag + 1] == str(plan.report)


def test_skip_optimize_rejects_an_incomplete_summary(tmp_path):
    plan = optimize.Plan("KRDU", "runway", False, ("eval",))
    plan.opt_dir = tmp_path
    plan.summary = tmp_path / "summary.json"
    plan.scenarios = tmp_path / "scenarios.json"
    plan.arrivals_manifest = tmp_path / "arrivals_manifest.json"
    plan.scenarios.write_text("[]")
    plan.arrivals_manifest.write_text("{}")
    plan.summary.write_text(json.dumps({
        "total": 1,
        "solved": 1,
        "failed": 0,
        "results": [{
            "id": "AFR074",
            "runway": "05L",
            "status": "solved",
            "states_file": "missing_states.json",
            "eval_file": "missing_eval.json",
        }],
    }), encoding="utf-8")

    assert plan.optimization_exists() is False
    assert "missing" in (plan.optimization_reuse_error() or "")


def test_skip_optimize_accepts_a_complete_solved_roster(tmp_path):
    plan = optimize.Plan("KRDU", "runway", False, ("eval",))
    plan.opt_dir = tmp_path
    plan.summary = tmp_path / "summary.json"
    plan.scenarios = tmp_path / "scenarios.json"
    plan.arrivals_manifest = tmp_path / "arrivals_manifest.json"
    plan.scenarios.write_text("[]")
    plan.arrivals_manifest.write_text("{}")
    states_name = "AFR074_05L_states.json"
    eval_name = "AFR074_05L_eval.json"
    reference_name = "references/AFR074_05L_reference_eval.json"
    (tmp_path / states_name).write_text("{}")
    (tmp_path / "references").mkdir()
    reference_path = tmp_path / reference_name
    reference_path.write_text("{}")
    (tmp_path / eval_name).write_text(json.dumps({
        "source": {"id": "AFR074", "runway": "05L"},
        "states": [],
        "controls": [],
        "final_time_s": 10.0,
        "states_ref": {"file": states_name, "key": "simulator_states"},
        "reference_file": reference_name,
    }))
    plan.summary.write_text(json.dumps({
        "total": 1,
        "solved": 1,
        "failed": 0,
        "results": [{
            "id": "AFR074",
            "runway": "05L",
            "status": "solved",
            "states_file": states_name,
            "eval_file": eval_name,
        }],
    }))
    (tmp_path / "references" / "manifest.json").write_text(json.dumps({
        "schema_version": "optimization-references-v2-sha256",
        "source_signature": {
            "scenarios_sha256": hashlib.sha256(plan.scenarios.read_bytes()).hexdigest(),
            "arrivals_manifest_sha256": hashlib.sha256(
                plan.arrivals_manifest.read_bytes()
            ).hexdigest(),
        },
        "records": [{
            "file": reference_path.name,
            "identity": {
                "flight_key": "AFR074_05L_unknown_unknown",
                "id": "AFR074",
                "runway": "05L",
                "icao24": None,
                "landing_time_utc": None,
            },
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        }],
    }))

    assert plan.optimization_reuse_error() is None
    assert plan.optimization_exists() is True


def test_legacy_adsb_target_type_is_no_longer_a_mode():
    with pytest.raises(ValueError, match="unknown target type"):
        optimize.Plan("KRDU", "adsb", False, ("eval",))
