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


def test_default_clean_keeps_every_harvest_category(tmp_path: Path, monkeypatch) -> None:
    harvest = tmp_path / "trajectory_data_process" / "outputs" / "harvest"
    harvest_files = {
        _write(harvest / "KAAA" / "tracks" / "manifest.json"),
        _write(harvest / "KAAA" / "arrivals" / "manifest.json"),
        _write(harvest / "KAAA" / "approach" / "summary.json"),
    }

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
    assert planned.isdisjoint(harvest_files)
    assert any("harvest data" in note for note in kept)
