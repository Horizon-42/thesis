"""Manifest-aware harvest merging preserves source tracks transactionally."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.__main__ import build_parser
from trajectory_data_process.harvest.merge import merge_stored_tracks
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


def _rewrite_record(paths: HarvestPaths, **updates: object) -> None:
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    row = manifest["records"][0]
    record_path = paths.tracks / row["file"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update(updates)
    for field in ("icao24", "callsign"):
        if field in updates:
            row[field] = updates[field]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")


def _write_harvest(
    root: Path,
    *,
    key: str,
    relative: str,
    icao24: str,
    provenance: dict,
) -> HarvestPaths:
    paths = HarvestPaths(root, "KAAA")
    record = {
        "flight_key": key,
        "icao24": icao24,
        "callsign": key.split("_")[0],
        "outcome": "not_landing",
        "runway": None,
        "landing_time_utc": None,
        "landing_sample_index": None,
        "start_time_utc": "2026-05-01T00:00:00Z",
        "duration_s": 1.0,
        "max_sample_gap_s": 1.0,
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "assignment": {
            "outcome": "not_landing",
            "runway": None,
            "scores_m": {},
            "margin_m": None,
            "reason": "test",
        },
        "samples": [[0.0, -78.0, 35.0, 100.0], [1.0, -78.0, 35.0, 99.0]],
    }
    record_path = paths.tracks / relative
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    paths.manifest.write_text(
        json.dumps(
            {
                "airport": "KAAA",
                "written_utc": "2026-05-02T00:00:00Z",
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
                "provenance": provenance,
                "records": [
                    {
                        "flight_key": key,
                        "file": relative,
                        "outcome": "not_landing",
                        "runway": None,
                        "icao24": icao24,
                        "callsign": record["callsign"],
                        "landing_time_utc": None,
                        "landing_sample_index": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_merge_stages_hard_links_preserves_provenance_and_invalidates_views(tmp_path):
    destination = _write_harvest(
        tmp_path / "current",
        key="NOW_not_landing_abc001_20260701T000000Z",
        relative="not_landing/NOW_not_landing_abc001_20260701T000000Z.json",
        icao24="abc001",
        provenance={"window": "current"},
    )
    source = _write_harvest(
        tmp_path / "may",
        key="MAY_not_landing_abc002_20260501T000000Z",
        relative="not_landing/MAY_not_landing_abc002_20260501T000000Z.json",
        icao24="abc002",
        provenance={"window": "may"},
    )
    source_record = source.tracks / source.manifest.parent.joinpath(
        "not_landing/MAY_not_landing_abc002_20260501T000000Z.json"
    ).relative_to(source.tracks)
    (destination.airport / "arrivals").mkdir()
    (destination.airport / "arrivals" / "stale.json").write_text("{}", encoding="utf-8")
    destination.approach.mkdir()
    (destination.approach / "stale.json").write_text("{}", encoding="utf-8")
    source_hash = hashlib.sha256(source.manifest.read_bytes()).hexdigest()

    merged = merge_stored_tracks(
        destination,
        [source],
        airport=_airport(),
        metadata_lookup=lambda _icao24, _time_s: None,
        metadata_provenance={"test": True},
    )

    assert merged["total"] == 2
    assert merged["counts"]["not_landing"] == 2
    assert not (destination.airport / "arrivals").exists()
    assert not destination.approach.exists()
    may_row = next(row for row in merged["records"] if row["icao24"] == "abc002")
    assert not os.path.samefile(source_record, destination.tracks / may_row["file"])
    merge = merged["provenance"]["merge"]
    assert merge["network_access"] is False
    assert [item["provenance"]["window"] for item in merge["sources"]] == [
        "current",
        "may",
    ]
    assert merge["sources"][1]["manifest_sha256"] == source_hash


def test_merge_rejects_duplicate_identity_without_changing_destination(tmp_path):
    key = "SAME_not_landing_abc001_20260501T000000Z"
    destination = _write_harvest(
        tmp_path / "current",
        key=key,
        relative=f"not_landing/{key}.json",
        icao24="abc001",
        provenance={"window": "current"},
    )
    source = _write_harvest(
        tmp_path / "may",
        key=key,
        relative=f"not_landing/{key}.json",
        icao24="abc001",
        provenance={"window": "may"},
    )
    before = destination.manifest.read_bytes()

    with pytest.raises(ValueError, match="duplicate flight_key"):
        merge_stored_tracks(
            destination,
            [source],
            airport=_airport(),
            metadata_lookup=lambda _icao24, _time_s: None,
            metadata_provenance={"test": True},
        )

    assert destination.manifest.read_bytes() == before
    assert json.loads(destination.manifest.read_text(encoding="utf-8"))["total"] == 1


def test_merge_rejects_a_manifest_path_that_escapes_tracks(tmp_path):
    destination = _write_harvest(
        tmp_path / "current",
        key="NOW_not_landing_abc001_20260701T000000Z",
        relative="not_landing/NOW.json",
        icao24="abc001",
        provenance={"window": "current"},
    )
    source = _write_harvest(
        tmp_path / "may",
        key="MAY_not_landing_abc002_20260501T000000Z",
        relative="not_landing/MAY.json",
        icao24="abc002",
        provenance={"window": "may"},
    )
    manifest = json.loads(source.manifest.read_text(encoding="utf-8"))
    manifest["records"][0]["file"] = "../outside.json"
    source.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes tracks directory"):
        merge_stored_tracks(
            destination,
            [source],
            airport=_airport(),
            metadata_lookup=lambda _icao24, _time_s: None,
            metadata_provenance={"test": True},
        )


def test_merge_rejects_post_reclassification_collision_without_committing(tmp_path):
    destination = _write_harvest(
        tmp_path / "current",
        key="OLD_A_not_landing_abc001_20260501T000001Z",
        relative="not_landing/old-a.json",
        icao24="abc001",
        provenance={"window": "current"},
    )
    source = _write_harvest(
        tmp_path / "may",
        key="OLD_B_assigned_abc001_20260501T000001Z",
        relative="assigned/18/old-b.json",
        icao24="abc001",
        provenance={"window": "may"},
    )
    _rewrite_record(destination, callsign="SAME")
    _rewrite_record(source, callsign="SAME")
    before = destination.manifest.read_bytes()
    (destination.airport / "arrivals").mkdir()
    stale = destination.airport / "arrivals" / "stale.json"
    stale.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate flight_key.*reclassification"):
        merge_stored_tracks(
            destination,
            [source],
            airport=_airport(),
            metadata_lookup=lambda _icao24, _time_s: None,
            metadata_provenance={"test": True},
        )

    assert destination.manifest.read_bytes() == before
    assert stale.read_text(encoding="utf-8") == "{}"


def test_merge_validates_reclassification_before_committing(tmp_path):
    destination = _write_harvest(
        tmp_path / "current",
        key="NOW_not_landing_abc001_20260701T000000Z",
        relative="not_landing/now.json",
        icao24="abc001",
        provenance={"window": "current"},
    )
    source = _write_harvest(
        tmp_path / "may",
        key="MAY_not_landing_abc002_20260501T000000Z",
        relative="not_landing/may.json",
        icao24="abc002",
        provenance={"window": "may"},
    )
    _rewrite_record(source, start_time_utc="not-an-iso-instant")
    before = destination.manifest.read_bytes()
    (destination.airport / "arrivals").mkdir()
    stale = destination.airport / "arrivals" / "stale.json"
    stale.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="start_time_utc"):
        merge_stored_tracks(
            destination,
            [source],
            airport=_airport(),
            metadata_lookup=lambda _icao24, _time_s: None,
            metadata_provenance={"test": True},
        )

    assert destination.manifest.read_bytes() == before
    assert stale.read_text(encoding="utf-8") == "{}"


def test_cli_exposes_merge_as_an_exclusive_no_download_mode(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        ["--airport", "KAAA", "--merge-source", str(tmp_path / "may")]
    )
    assert args.merge_source == [tmp_path / "may"]

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--airport",
                "KAAA",
                "--merge-source",
                str(tmp_path / "may"),
                "--evaluate-only",
            ]
        )
