"""The legacy-named CLI is only a multi-airport adapter for harvest."""

import json
from datetime import datetime, timezone
from pathlib import Path

from trajectory_data_process.download_landings import build_parser, harvest_argv
from trajectory_data_process.harvest.__main__ import _resolve_download_options
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
