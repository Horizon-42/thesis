"""Safety and layout contracts for the destructive pipeline cleaner."""

from __future__ import annotations

from pathlib import Path

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
        ["clean_pipeline_data.py", "--include-downloads", "--dry-run"],
    )

    cleaner.main()

    output = capsys.readouterr().out
    assert "harvest tracks" in output
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

    groups, _, kept = cleaner.deletion_groups(
        include_parked=False, include_downloads=False
    )

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
    airport_dir = tmp_path / "aeroviz-4d" / "public" / "data" / "airports" / "KAAA"
    observed = _write(airport_dir / "trajectories.czml")
    landings = _write(airport_dir / "landings" / "index.json")
    observed_report = _write(
        airport_dir / "comparison" / "observed" / "evaluation_report.json"
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
        ["clean_pipeline_data.py", "--yes"],
    )

    cleaner.main()

    assert tracks.exists()
    assert static_airport.exists()
    assert all(
        not path.exists()
        for path in (arrivals, approach, scenarios, observed, landings, observed_report)
    )
