"""Safety and layout contracts for the destructive pipeline cleaner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import clean_pipeline_data as cleaner


def _write(path: Path, text: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_dry_run_rosters_each_harvest_category_without_deleting(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    outputs = tmp_path / "trajectory_data_process" / "outputs"
    harvest = outputs / "harvest"
    tracks = _write(harvest / "KAAA" / "tracks" / "manifest.json")
    arrivals = _write(harvest / "KAAA" / "arrivals" / "manifest.json")
    approach = _write(harvest / "KAAA" / "approach" / "summary.json")
    legacy = _write(outputs / "landings" / "KAAA" / "KAAA_18_landings.json")

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", tmp_path / "4dTrajectory" / "outputs")
    monkeypatch.setattr(
        cleaner,
        "COMPARISON_AIRPORTS_ROOT",
        tmp_path / "aeroviz-4d" / "public" / "data" / "airports",
    )
    monkeypatch.setattr(cleaner, "HARVEST_ROOT", harvest)
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())
    monkeypatch.setattr(
        cleaner.sys,
        "argv",
        ["clean_pipeline_data.py", "--all-airports", "--dry-run"],
    )

    cleaner.main()

    output = capsys.readouterr().out
    assert "harvest arrivals" in output
    assert "harvest approach" in output
    assert "dry-run" in output
    assert all(path.exists() for path in (tracks, arrivals, approach, legacy))


def test_default_clean_removes_derived_harvest_but_keeps_downloaded_tracks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harvest = tmp_path / "trajectory_data_process" / "outputs" / "harvest"
    tracks = _write(harvest / "KAAA" / "tracks" / "manifest.json")
    arrivals = _write(harvest / "KAAA" / "arrivals" / "manifest.json")
    approach = _write(harvest / "KAAA" / "approach" / "summary.json")

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", tmp_path / "4dTrajectory" / "outputs")
    monkeypatch.setattr(
        cleaner,
        "COMPARISON_AIRPORTS_ROOT",
        tmp_path / "aeroviz-4d" / "public" / "data" / "airports",
    )
    monkeypatch.setattr(cleaner, "HARVEST_ROOT", harvest)
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, kept = cleaner.deletion_groups(airports=None)

    planned = {path for _, files in groups for path in files}
    assert tracks not in planned
    assert {arrivals, approach} <= planned
    assert any("harvest tracks" in note for note in kept)


def test_default_clean_deletes_preparation_outputs_without_touching_tracks_or_static_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harvest = tmp_path / "trajectory_data_process" / "outputs" / "harvest"
    tracks = _write(harvest / "KAAA" / "tracks" / "manifest.json")
    arrivals = _write(harvest / "KAAA" / "arrivals" / "manifest.json")
    approach = _write(harvest / "KAAA" / "approach" / "summary.json")
    scenarios = _write(
        tmp_path / "flight_scenarios" / "outputs" /
        "KAAA_arrivals_fitted_adsb_scenarios.json"
    )
    optimizer = _write(
        tmp_path / "4dTrajectory" / "outputs" / "KAAA" / "runway" / "summary.json"
    )
    prediction = _write(
        tmp_path / "4dTrajectory" / "outputs" / "KAAA" /
        "ts_pred_itransformer_val" / "summary.json",
        json.dumps({"split": "val"}),
    )
    checkpoint = _write(
        tmp_path / "4dTrajectory" / "outputs" / "KAAA" /
        "ts_itransformer_window" / "checkpoint.pt"
    )
    history = _write(checkpoint.parent / "history.json")
    test_release = _write(checkpoint.parent / "test_release.json")
    formal_experiment = _write(
        tmp_path / "4dTrajectory" / "outputs" / "POOLED" / "experiments" /
        "campaign" / "run" / "experiment_manifest.json"
    )
    unknown_model_output = _write(
        tmp_path / "4dTrajectory" / "outputs" / "KAAA" /
        "manual_diagnostic" / "result.json"
    )
    airport_dir = tmp_path / "aeroviz-4d" / "public" / "data" / "airports" / "KAAA"
    observed = _write(airport_dir / "trajectories.czml")
    landings = _write(airport_dir / "landings" / "index.json")
    observed_report = _write(
        airport_dir / "comparison" / "observed" / "evaluation_report.json"
    )
    _write(
        airport_dir / "comparison" / "categories.json",
        json.dumps({"categories": [{"key": "observed", "dir": "observed"}]}),
    )
    static_airport = _write(airport_dir / "airport.json")

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        cleaner,
        "SCENARIOS_DIR",
        tmp_path / "flight_scenarios" / "outputs",
    )
    monkeypatch.setattr(
        cleaner,
        "OPT_OUTPUTS_ROOT",
        tmp_path / "4dTrajectory" / "outputs",
    )
    monkeypatch.setattr(
        cleaner,
        "COMPARISON_AIRPORTS_ROOT",
        tmp_path / "aeroviz-4d" / "public" / "data" / "airports",
    )
    monkeypatch.setattr(cleaner, "HARVEST_ROOT", harvest)
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())
    monkeypatch.setattr(
        cleaner.sys,
        "argv",
        ["clean_pipeline_data.py", "--all-airports", "--yes"],
    )

    cleaner.main()

    assert tracks.exists()
    assert static_airport.exists()
    assert all(
        path.exists()
        for path in (
            checkpoint,
            history,
            test_release,
            formal_experiment,
            unknown_model_output,
        )
    )
    assert all(
        not path.exists()
        for path in (
            arrivals,
            approach,
            scenarios,
            optimizer,
            prediction,
            observed,
            landings,
            observed_report,
        )
    )


def test_deletion_plan_is_allowlisted_and_never_selects_experiment_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    outputs = tmp_path / "4dTrajectory" / "outputs"
    optimizer = _write(outputs / "KAAA" / "fitted_adsb" / "summary.json")
    references = _write(outputs / "KAAA" / "shared_references" / "runway" / "manifest.json")
    prediction = _write(
        outputs / "KAAA" / "ts_pred_model_val" / "summary.json",
        json.dumps({"split": "val"}),
    )
    final_prediction = _write(
        outputs / "KAAA" / "ts_pred_model_test" / "summary.json",
        json.dumps({"split": "test"}),
    )
    final_prediction_state = _write(
        final_prediction.parent / "flight_states.json"
    )
    ambiguous_prediction = _write(
        outputs / "KAAA" / "ts_pred_model_unknown" / "summary.json",
        json.dumps({"model": "itransformer"}),
    )
    missing_summary_state = _write(
        outputs / "KAAA" / "ts_pred_model_missing_summary" / "flight_states.json"
    )

    checkpoint = _write(outputs / "KAAA" / "ts_model" / "checkpoint.pt")
    metadata = _write(outputs / "KAAA" / "ts_model" / "checkpoint_metadata.json")
    history = _write(outputs / "KAAA" / "ts_model" / "history.json")
    release = _write(outputs / "KAAA" / "ts_model" / "test_release.json")
    nested_test_prediction = _write(
        outputs / "KAAA" / "ts_model" / "pred_test" / "evaluation_report.json"
    )
    experiment = _write(
        outputs / "POOLED" / "experiments" / "campaign" / "run" /
        "experiment_manifest.json"
    )
    parked = _write(outputs / "KAAA" / "_parked" / "checkpoint.pt")
    unknown = _write(outputs / "KAAA" / "diagnostic" / "result.json")

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", outputs)
    monkeypatch.setattr(
        cleaner,
        "COMPARISON_AIRPORTS_ROOT",
        tmp_path / "aeroviz-4d" / "public" / "data" / "airports",
    )
    monkeypatch.setattr(
        cleaner,
        "HARVEST_ROOT",
        tmp_path / "trajectory_data_process" / "outputs" / "harvest",
    )
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, kept = cleaner.deletion_groups(airports={"KAAA"})
    planned = {path for _, files in groups for path in files}

    assert {optimizer, references, prediction} <= planned
    assert planned.isdisjoint({
        checkpoint,
        metadata,
        history,
        release,
        nested_test_prediction,
        final_prediction,
        final_prediction_state,
        ambiguous_prediction,
        missing_summary_state,
        experiment,
        parked,
        unknown,
    })
    assert any("experiment and training outputs" in note for note in kept)
    assert any("final-test prediction" in note for note in kept)


def test_mixed_frontend_comparison_is_preserved_when_it_contains_experiment_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    airports = tmp_path / "aeroviz-4d" / "public" / "data" / "airports"
    comparison = airports / "KAAA" / "comparison"
    ordinary = _write(comparison / "runway" / "comparison_index.json")
    experiment = _write(comparison / "experiment_formal_val" / "comparison_index.json")
    registry = _write(
        comparison / "categories.json",
        json.dumps({
            "categories": [
                {"key": "runway", "dir": "runway"},
                {
                    "key": "experiment_formal_val",
                    "dir": "experiment_formal_val",
                    "datasetSplit": "val",
                    "resultSource": "experiment",
                },
            ],
        }),
    )

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", tmp_path / "4dTrajectory" / "outputs")
    monkeypatch.setattr(cleaner, "COMPARISON_AIRPORTS_ROOT", airports)
    monkeypatch.setattr(
        cleaner,
        "HARVEST_ROOT",
        tmp_path / "trajectory_data_process" / "outputs" / "harvest",
    )
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, kept = cleaner.deletion_groups(airports={"KAAA"})
    planned = {path for _, files in groups for path in files}

    assert planned.isdisjoint({ordinary, experiment, registry})
    assert any("experiment or final-test publication" in note for note in kept)


@pytest.mark.parametrize("with_registry", [False, True])
def test_unowned_frontend_comparison_content_fails_closed(
    tmp_path: Path,
    monkeypatch,
    with_registry: bool,
) -> None:
    airports = tmp_path / "aeroviz-4d" / "public" / "data" / "airports"
    comparison = airports / "KAAA" / "comparison"
    ordinary = _write(comparison / "runway" / "comparison_index.json")
    curated = _write(comparison / "curated-notes.txt")
    if with_registry:
        _write(
            comparison / "categories.json",
            json.dumps({"categories": [{"key": "runway", "dir": "runway"}]}),
        )

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", tmp_path / "4dTrajectory" / "outputs")
    monkeypatch.setattr(cleaner, "COMPARISON_AIRPORTS_ROOT", airports)
    monkeypatch.setattr(
        cleaner,
        "HARVEST_ROOT",
        tmp_path / "trajectory_data_process" / "outputs" / "harvest",
    )
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, kept = cleaner.deletion_groups(airports={"KAAA"})
    planned = {path for _, files in groups for path in files}

    assert planned.isdisjoint({ordinary, curated})
    assert any("frontend comparison" in note for note in kept)


def test_all_airports_scope_does_not_treat_pooled_outputs_as_an_airport(
    tmp_path: Path,
    monkeypatch,
) -> None:
    outputs = tmp_path / "4dTrajectory" / "outputs"
    airport_output = _write(outputs / "KAAA" / "runway" / "summary.json")
    pooled_output = _write(outputs / "POOLED" / "runway" / "summary.json")

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", outputs)
    monkeypatch.setattr(
        cleaner,
        "COMPARISON_AIRPORTS_ROOT",
        tmp_path / "aeroviz-4d" / "public" / "data" / "airports",
    )
    monkeypatch.setattr(
        cleaner,
        "HARVEST_ROOT",
        tmp_path / "trajectory_data_process" / "outputs" / "harvest",
    )
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, _ = cleaner.deletion_groups(airports=None)
    planned = {path for _, files in groups for path in files}

    assert airport_output in planned
    assert pooled_output not in planned


def test_airport_scope_never_selects_another_airports_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenarios = tmp_path / "flight_scenarios" / "outputs"
    models = tmp_path / "4dTrajectory" / "outputs"
    airports = tmp_path / "aeroviz-4d" / "public" / "data" / "airports"
    harvest = tmp_path / "trajectory_data_process" / "outputs" / "harvest"

    selected = {
        _write(scenarios / "KAAA_arrivals_threshold_scenarios.json"),
        _write(models / "KAAA" / "runway" / "summary.json"),
        _write(airports / "KAAA" / "comparison" / "runway" / "index.json"),
        _write(harvest / "KAAA" / "arrivals" / "manifest.json"),
    }
    other = {
        _write(scenarios / "KBBB_arrivals_threshold_scenarios.json"),
        _write(models / "KBBB" / "runway" / "summary.json"),
        _write(airports / "KBBB" / "comparison" / "runway" / "index.json"),
        _write(harvest / "KBBB" / "arrivals" / "manifest.json"),
    }
    _write(
        airports / "KAAA" / "comparison" / "categories.json",
        json.dumps({"categories": [{"key": "runway", "dir": "runway"}]}),
    )
    _write(
        airports / "KBBB" / "comparison" / "categories.json",
        json.dumps({"categories": [{"key": "runway", "dir": "runway"}]}),
    )

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", scenarios)
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", models)
    monkeypatch.setattr(cleaner, "COMPARISON_AIRPORTS_ROOT", airports)
    monkeypatch.setattr(cleaner, "HARVEST_ROOT", harvest)
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, _ = cleaner.deletion_groups(airports={"KAAA"})
    planned = {path for _, files in groups for path in files}

    assert selected <= planned
    assert planned.isdisjoint(other)


def test_observed_cleanup_selects_only_the_canonical_czml_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    airports = tmp_path / "aeroviz-4d" / "public" / "data" / "airports"
    canonical = _write(airports / "KAAA" / "trajectories.czml")
    lookalike = _write(airports / "KAAA" / "trajectories.czml.curated-copy")

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "flight_scenarios" / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", tmp_path / "4dTrajectory" / "outputs")
    monkeypatch.setattr(cleaner, "COMPARISON_AIRPORTS_ROOT", airports)
    monkeypatch.setattr(
        cleaner,
        "HARVEST_ROOT",
        tmp_path / "trajectory_data_process" / "outputs" / "harvest",
    )
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())

    groups, _, _ = cleaner.deletion_groups(airports={"KAAA"})
    planned = {path for _, files in groups for path in files}

    assert canonical in planned
    assert lookalike not in planned


def test_cli_requires_explicit_airport_scope(monkeypatch) -> None:
    monkeypatch.setattr(cleaner.sys, "argv", ["clean_pipeline_data.py", "--dry-run"])

    with pytest.raises(SystemExit, match="2"):
        cleaner.main()


def test_failed_transactional_staging_restores_every_selected_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _write(tmp_path / "outputs" / "first.json", "first")
    second = _write(tmp_path / "outputs" / "second.json", "second")
    staging = tmp_path / ".clean-staging"
    real_move = Path.replace

    def fail_on_second(source: Path, destination: Path) -> Path:
        if source == second and staging in destination.parents:
            raise OSError("simulated staging failure")
        return real_move(source, destination)

    monkeypatch.setattr(cleaner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cleaner, "SCENARIOS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(cleaner, "OPT_OUTPUTS_ROOT", tmp_path / "models")
    monkeypatch.setattr(cleaner, "COMPARISON_AIRPORTS_ROOT", tmp_path / "airports")
    monkeypatch.setattr(cleaner, "HARVEST_ROOT", tmp_path / "harvest")
    monkeypatch.setattr(cleaner, "_tracked_files", lambda: frozenset())
    monkeypatch.setattr(cleaner, "_move_file", fail_on_second)

    with pytest.raises(OSError, match="simulated staging failure"):
        cleaner.delete_files_transactionally([first, second], staging_root=staging)

    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"
    assert not staging.exists()
