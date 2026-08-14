"""Safe, batched rebuild of source-timed tracks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trajectory_data_process.harvest import freshness_rebuild as rebuild_module
from trajectory_data_process.harvest.__main__ import build_parser
from trajectory_data_process.harvest.adsb_metadata import AdsbStateMetadata
from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.freshness_rebuild import rebuild_fresh_tracks
from trajectory_data_process.harvest.store import (
    ALTITUDE_DATUM,
    ALTITUDE_SOURCE,
    HarvestPaths,
)


def _airport() -> Airport:
    return Airport(
        "KAAA",
        35.0,
        -78.0,
        100.0,
        (
            Runway(
                airport="KAAA",
                ident="18",
                lat=35.0,
                lon=-78.0,
                elevation_hae_m=130.0,
                elevation_msl_m=100.0,
                course_deg=0.0,
                hae_minus_msl_m=30.0,
                threshold_crossing_height_m=15.0,
                published_glidepath_deg=3.0,
                width_m=45.72,
                lpv_course_width_m=106.75,
                runway_source_cycle="2026-08-06",
                procedure_source_cycle="2026-08-06",
            ),
        ),
    )


def _source(root: Path) -> HarvestPaths:
    paths = HarvestPaths(root, "KAAA")
    key = "TEST_not_landing_abc123_20260801T001019Z"
    relative = f"not_landing/{key}.json"
    record = {
        "flight_key": key,
        "icao24": "abc123",
        "callsign": "TEST",
        "outcome": "not_landing",
        "runway": None,
        "landing_time_utc": None,
        "landing_sample_index": None,
        "start_time_utc": "2026-08-01T00:10:00Z",
        "duration_s": 20.0,
        "max_sample_gap_s": 1.0,
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "assignment": {"outcome": "not_landing"},
        "observed_threshold_event": {"status": "unavailable"},
        "samples": [
            [float(index), -78.5, 35.5 + index * 0.0001, 500.0 - index]
            for index in range(20)
        ],
    }
    path = paths.tracks / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    paths.manifest.write_text(
        json.dumps(
            {
                "airport": "KAAA",
                "altitude_source": ALTITUDE_SOURCE,
                "altitude_datum": ALTITUDE_DATUM,
                "counts": {
                    "assigned": 0,
                    "ambiguous": 0,
                    "unassignable": 0,
                    "not_landing": 1,
                },
                "per_runway": {},
                "total": 1,
                "provenance": {"original": True},
                "records": [
                    {
                        "flight_key": key,
                        "file": relative,
                        "outcome": "not_landing",
                        "runway": None,
                        "icao24": "abc123",
                        "callsign": "TEST",
                        "landing_time_utc": None,
                        "landing_sample_index": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return paths


class _Metadata:
    provenance = {"test": True}

    def __init__(self) -> None:
        self.calls = 0

    def lookup_many(self, queries):
        self.calls += 1
        return [
            AdsbStateMetadata(70.0, state_time - 0.25, state_time - 0.1)
            for _icao24, state_time in queries
        ]


def _snapshot(paths: HarvestPaths) -> dict[str, bytes]:
    return {
        str(path.relative_to(paths.airport)): path.read_bytes()
        for path in sorted(paths.airport.rglob("*"))
        if path.is_file()
    }


def test_rebuild_is_batched_source_timed_and_does_not_change_source(tmp_path):
    source = _source(tmp_path / "source")
    destination = HarvestPaths(tmp_path / "staging", "KAAA")
    metadata = _Metadata()
    before = _snapshot(source)

    manifest = rebuild_fresh_tracks(
        _airport(), source, destination, metadata=metadata, batch_tracks=512
    )

    assert _snapshot(source) == before
    assert metadata.calls == 1
    assert manifest["source_integrity_complete"] is True
    assert manifest["source_integrity"]["source_total"] == 1
    assert manifest["source_integrity"]["excluded_total"] == 0
    assert "source_integrity" not in manifest["provenance"]["freshness_rebuild"]
    [row] = manifest["records"]
    rebuilt = json.loads((destination.tracks / row["file"]).read_text())
    assert rebuilt["samples"][0][0] == 0.0
    assert rebuilt["duration_s"] == pytest.approx(19.0)
    assert rebuilt["source_integrity"]["position_time_basis"] == "lastposupdate"


def test_rebuild_refuses_same_or_existing_destination(tmp_path):
    source = _source(tmp_path / "source")
    metadata = _Metadata()
    with pytest.raises(ValueError, match="must differ"):
        rebuild_fresh_tracks(_airport(), source, source, metadata=metadata)

    nested = HarvestPaths(source.root / "nested", "KAAA")
    with pytest.raises(ValueError, match="roots may not be nested"):
        rebuild_fresh_tracks(_airport(), source, nested, metadata=metadata)

    destination = HarvestPaths(tmp_path / "staging", "KAAA")
    destination.airport.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        rebuild_fresh_tracks(_airport(), source, destination, metadata=metadata)


def test_source_change_aborts_before_staging_commit(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    destination = HarvestPaths(tmp_path / "staging", "KAAA")
    fingerprints = iter(["before", "after"])
    monkeypatch.setattr(
        rebuild_module, "_source_fingerprint", lambda *_args: next(fingerprints)
    )

    with pytest.raises(RuntimeError, match="source harvest changed"):
        rebuild_fresh_tracks(
            _airport(), source, destination, metadata=_Metadata(), log=lambda _line: None
        )

    assert not destination.airport.exists()


def test_space_preflight_fails_before_writing_airport_output(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    destination = HarvestPaths(tmp_path / "staging", "KAAA")
    monkeypatch.setattr(
        rebuild_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1024),
    )

    with pytest.raises(OSError, match="insufficient staging space"):
        rebuild_fresh_tracks(_airport(), source, destination, metadata=_Metadata())

    assert not destination.airport.exists()


def test_cli_exposes_freshness_rebuild_as_an_exclusive_mode(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        ["--airport", "KAAA", "--rebuild-fresh-from", str(tmp_path / "source")]
    )
    assert args.rebuild_fresh_from == tmp_path / "source"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--airport",
                "KAAA",
                "--rebuild-fresh-from",
                str(tmp_path / "source"),
                "--evaluate-only",
            ]
        )
