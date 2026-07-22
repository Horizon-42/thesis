"""The legacy-named CLI is only a multi-airport adapter for harvest."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from trajectory_data_process.download_landings import build_parser, harvest_argv
from trajectory_data_process.harvest import __main__ as harvest_cli
from trajectory_data_process.harvest.__main__ import (
    _completed_download_manifest,
    _resolve_download_options,
)
from trajectory_data_process.harvest.store import HarvestPaths


def test_wrapper_forwards_one_airport_to_the_canonical_harvest_cli():
    args = build_parser().parse_args(
        [
            "--airports", "krdu",
            "--count", "12",
            "--entry-radius-km", "20",
            "--evaluate-only",
            "--full-redownload",
            "--no-publish",
            "--multiplier", "30",
        ]
    )

    argv = harvest_argv(args, "KRDU")
    assert argv[:4] == ["--airport", "KRDU", "--count", "12"]
    assert argv[argv.index("--entry-radius-km") + 1] == "20.0"
    assert "--evaluate-only" in argv
    assert "--full-redownload" in argv
    assert "--no-publish" in argv
    assert argv[argv.index("--multiplier") + 1] == "30"


def _write_manifest(root: Path, provenance: dict[str, str]) -> HarvestPaths:
    paths = HarvestPaths(root=root, code="KRDU")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(json.dumps({"provenance": provenance}), encoding="utf-8")
    return paths


def test_download_reuses_previous_start_and_enables_opensky_cache(tmp_path: Path):
    paths = _write_manifest(
        tmp_path,
        {
            "start_utc": "2026-07-01T12:34:56+00:00",
            "scanned_to_utc": "2026-06-01T00:00:00+00:00",
        },
    )
    messages: list[str] = []

    start, cached = _resolve_download_options(
        requested_start=None,
        paths=paths,
        full_redownload=False,
        no_cache=False,
        log=messages.append,
    )

    assert start == datetime(2026, 7, 1, 12, 34, 56, tzinfo=timezone.utc)
    assert cached is True
    assert "reusing previous download start" in messages[0]


def test_download_reuses_legacy_scanned_to_timestamp(tmp_path: Path):
    paths = _write_manifest(
        tmp_path, {"scanned_to_utc": "2026-06-01T00:00:00+00:00"}
    )

    start, cached = _resolve_download_options(
        requested_start=None,
        paths=paths,
        full_redownload=False,
        no_cache=False,
        log=lambda _message: None,
    )

    assert start == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert cached is True


def test_download_reuses_interrupted_checkpoint_start_for_cache_keys(tmp_path: Path):
    paths = HarvestPaths(root=tmp_path, code="KSJC")
    paths.checkpoint.mkdir(parents=True)
    paths.checkpoint_state.write_text(
        json.dumps(
            {
                "version": 1,
                "start_utc": "2026-07-22T11:50:11.064893+00:00",
            }
        ),
        encoding="utf-8",
    )

    start, cached = _resolve_download_options(
        requested_start=None,
        paths=paths,
        full_redownload=False,
        no_cache=False,
        log=lambda _message: None,
    )

    assert start == datetime(
        2026, 7, 22, 11, 50, 11, 64893, tzinfo=timezone.utc
    )
    assert cached is True


def test_explicit_start_overrides_previous_download_start(tmp_path: Path):
    paths = _write_manifest(
        tmp_path, {"start_utc": "2026-07-01T12:34:56+00:00"}
    )

    start, cached = _resolve_download_options(
        requested_start="2026-07-20T09:00:00Z",
        paths=paths,
        full_redownload=False,
        no_cache=False,
    )

    assert start == datetime(2026, 7, 20, 9, tzinfo=timezone.utc)
    assert cached is True


def test_full_redownload_ignores_previous_start_and_disables_cache(tmp_path: Path):
    paths = _write_manifest(
        tmp_path, {"start_utc": "2026-07-01T12:34:56+00:00"}
    )

    start, cached = _resolve_download_options(
        requested_start=None,
        paths=paths,
        full_redownload=True,
        no_cache=False,
        log=lambda _message: None,
    )

    assert start is None
    assert cached is False


def test_completed_download_accepts_a_runway_that_was_given_up(tmp_path: Path):
    paths = HarvestPaths(root=tmp_path, code="KMSY")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "total": 703,
                "records": [{}] * 703,
                "per_runway": {"02": 700, "20": 3},
                "provenance": {
                    "radius_km": 30.0,
                    "start_utc": "2026-07-22T00:00:00+00:00",
                    "given_up": ["20"],
                },
            }
        ),
        encoding="utf-8",
    )

    completed = _completed_download_manifest(
        paths,
        expected_runways={"02", "20"},
        target_per_runway=600,
        radius_km=30.0,
        requested_start=None,
    )

    assert completed is not None


def test_download_is_not_complete_when_a_runway_is_below_target_without_give_up(
    tmp_path: Path,
):
    paths = HarvestPaths(root=tmp_path, code="KMSY")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "total": 703,
                "records": [{}] * 703,
                "per_runway": {"02": 700, "20": 3},
                "provenance": {
                    "radius_km": 30.0,
                    "start_utc": "2026-07-22T00:00:00+00:00",
                    "given_up": [],
                },
            }
        ),
        encoding="utf-8",
    )

    completed = _completed_download_manifest(
        paths,
        expected_runways={"02", "20"},
        target_per_runway=600,
        radius_km=30.0,
        requested_start=None,
    )

    assert completed is None


def _write_completed_cli_fixture(root: Path) -> Path:
    config = root / "runways.json"
    config.write_text(
        json.dumps(
            {
                "airports": {
                    "KAAA": {
                        "runways": [{"thresholds": [{"ident": "18"}]}]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    paths = HarvestPaths(root=root / "outputs", code="KAAA")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "total": 600,
                "records": [{}] * 600,
                "per_runway": {"18": 600},
                "provenance": {
                    "radius_km": 30.0,
                    "start_utc": "2026-07-22T00:00:00+00:00",
                    "given_up": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_completed_airport_is_skipped_before_airport_loading(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    config = _write_completed_cli_fixture(tmp_path)
    monkeypatch.setattr(
        harvest_cli,
        "load_airport",
        lambda *_args, **_kwargs: pytest.fail("completed airport must be skipped"),
    )

    result = harvest_cli.main(
        [
            "--airport", "KAAA",
            "--count", "600",
            "--config", str(config),
            "--output", str(tmp_path / "outputs"),
        ]
    )

    assert result == 0
    assert "completed tracks already exist; skipping" in capsys.readouterr().out


def test_full_redownload_bypasses_completed_airport_skip(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = _write_completed_cli_fixture(tmp_path)

    class AirportLoadReached(Exception):
        pass

    monkeypatch.setattr(
        harvest_cli,
        "load_airport",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AirportLoadReached),
    )

    with pytest.raises(AirportLoadReached):
        harvest_cli.main(
            [
                "--airport", "KAAA",
                "--count", "600",
                "--config", str(config),
                "--output", str(tmp_path / "outputs"),
                "--full-redownload",
            ]
        )
