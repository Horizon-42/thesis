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
    assert "--target-from-fitted-adsb" in fitted
    assert "--target-from-threshold" in runway
    assert any(part.endswith("KRDU_arrivals_fitted_adsb_scenarios.json") for part in fitted)
    assert any(part.endswith("KRDU_arrivals_threshold_scenarios.json") for part in runway)

    # Both target datasets carry the SAME per-runway cap: the selection is roster-derived
    # and target-independent, so the two datasets must cover the same flights or the
    # per-flight comparison between fitted_adsb and runway silently stops being paired.
    for command in (fitted, runway):
        assert command[command.index("--max-per-runway") + 1] == str(
            prepare.DEFAULT_MAX_PER_RUNWAY
        )
    # 0 means "every rostered arrival" and must not emit a cap of zero.
    assert "--max-per-runway" not in prepare.scenario_command(
        "KRDU", "runway", max_per_runway=0
    )


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
        "optimization_config": plan.optimization_config,
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
    # The reference quotes its observed states from the shared sibling track store, so an
    # intact reference means the track is intact too (contract v3). The record→track path
    # mapping is the shared evaluation_export helper the validator itself uses.
    track_path = optimize.observed_track_path(reference_path)
    track_path.parent.mkdir()
    track_path.write_text('{"states": []}')
    (tmp_path / eval_name).write_text(json.dumps({
        "source": {"id": "AFR074", "runway": "05L"},
        "states": [],
        "controls": [],
        "final_time_s": 10.0,
        "states_ref": {"file": states_name, "key": "simulator_states"},
        "reference_file": reference_name,
    }))
    plan.summary.write_text(json.dumps({
        "optimization_config": plan.optimization_config,
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
        "schema_version": optimize.REFERENCE_CACHE_SCHEMA,
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
            "track_sha256": hashlib.sha256(track_path.read_bytes()).hexdigest(),
        }],
    }))

    assert plan.optimization_reuse_error() is None
    assert plan.optimization_exists() is True

    # A reference whose shared track went missing is NOT reusable: the record on its own
    # carries no states, so an absent track is an unreadable reference, not a cosmetic gap.
    track_path.unlink()
    assert "observed track" in (plan.optimization_reuse_error() or "")


def test_skip_optimize_rejects_a_batch_from_a_different_solver_configuration(
    tmp_path,
):
    plan = optimize.Plan(
        "KRDU",
        "runway",
        False,
        ("eval",),
        fitting="rk4",
        n_segments=12,
        state_substeps=4,
    )
    plan.summary = tmp_path / "summary.json"
    plan.summary.write_text(json.dumps({
        "optimization_config": {
            **plan.optimization_config,
            "fitting": "hs",
            "transcription_scheme": "hermiteSimpsonNormalizedFullTransport",
        },
        "total": 0,
        "solved": 0,
        "failed": 0,
        "results": [],
    }))

    assert "configuration" in (plan.optimization_reuse_error() or "")


def test_legacy_adsb_target_type_is_no_longer_a_mode():
    with pytest.raises(ValueError, match="unknown target type"):
        optimize.Plan("KRDU", "adsb", False, ("eval",))


def test_footprint_estimate_tracks_outputs_and_existing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(optimize, "HARVEST_TRACKS_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(optimize, "SCENARIOS_DIR", tmp_path / "scenarios")
    monkeypatch.setattr(optimize, "OPT_OUTPUTS_ROOT", tmp_path / "optimization")
    monkeypatch.setattr(optimize, "COMPARISON_AIRPORTS_ROOT", tmp_path / "frontend")
    scenarios = optimize.SCENARIOS_DIR / "KRDU_arrivals_threshold_scenarios.json"
    scenarios.parent.mkdir(parents=True)
    scenarios.write_text(json.dumps([{}, {}]), encoding="utf-8")   # two prepared flights

    both = optimize.Plan("KRDU", "runway", False, ("czml", "eval"))
    full = optimize.estimate_footprint_bytes([both])
    assert full > 0

    # Dropping the CZML output must shrink the estimate — the space-check refusal message
    # suggests exactly that remedy, and it used to change real usage but not the estimate.
    eval_only = optimize.Plan("KRDU", "runway", False, ("eval",))
    assert optimize.estimate_footprint_bytes([eval_only]) < full

    # Artifacts already on disk are netted per family: a --resume restart or a
    # --skip-optimize rebuild no longer demands the full footprint again.
    both.opt_dir.mkdir(parents=True)
    (both.opt_dir / "existing_states.json").write_bytes(
        b"x" * (2 * (optimize._RECORD_BYTES_FIXED + optimize._RECORD_BYTES_ROLLOUT))
    )
    netted = optimize.estimate_footprint_bytes([both])
    assert netted < full
    # ...and a family never goes negative, however large the existing tree is.
    (both.opt_dir / "huge_leftover.json").write_bytes(b"x" * (4 * 1024 * 1024))
    assert optimize.estimate_footprint_bytes([both]) >= 0


def test_runner_and_batch_share_the_reference_cache_contract():
    # run_scenario_optimization validates reference caches that scenario_optimization
    # writes. Both now IMPORT the whole contract — schema string, hash primitive, and
    # record→track path mapping — from evaluation_export, so this seam test pins that
    # neither side has regrown a restated copy.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "4dTrajectory" / "optimization"))
    import evaluation_export as ee
    import scenario_optimization as so

    assert optimize.REFERENCE_CACHE_SCHEMA == so.REFERENCE_CACHE_SCHEMA == ee.REFERENCE_CACHE_SCHEMA
    assert optimize.OBSERVED_TRACKS_DIR == so.OBSERVED_TRACKS_DIR == ee.OBSERVED_TRACKS_DIR
    assert optimize.observed_track_path is ee.observed_track_path
    assert optimize.file_sha256 is ee.file_sha256
