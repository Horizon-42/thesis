"""Stored samples can be reclassified without an OpenSky download."""

from __future__ import annotations

import json
import math

import pytest

from final_approach import Projected

from trajectory_data_process.harvest.__main__ import build_parser
from trajectory_data_process.harvest.adsb_metadata import AdsbStateMetadata
from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
    threshold_frame_fingerprint,
)
from trajectory_data_process.harvest.classify import classify_track
from trajectory_data_process.harvest.reclassify import (
    _reclassification_order,
    _stored_track,
    reclassify_stored_tracks,
)
from trajectory_data_process.harvest.store import HarvestPaths, track_record
from trajectory_data_process.harvest.tracks import (
    SOURCE_INTEGRITY_SCHEMA,
    Sample,
    SourceIntegrity,
    Track,
    source_timed_final_block,
)


def _airport() -> Airport:
    runway = Runway(
        airport="KAAA", ident="18", lat=35.0, lon=-78.0,
        elevation_hae_m=130.0, elevation_msl_m=100.0,
        course_deg=0.0, hae_minus_msl_m=30.0,
        threshold_crossing_height_m=15.0, published_glidepath_deg=3.0,
        width_m=45.72, lpv_course_width_m=106.75,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
    )
    return Airport("KAAA", 35.0, -78.0, 100.0, (runway,))


def _classified():
    runway = _airport().runway("18")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(3.0))
    samples = []
    for index, along_m in enumerate(range(-8_000, 0, 100)):
        point = frame.unproject(Projected(
            float(along_m), 0.0,
            runway.threshold_crossing_height_m - slope * along_m,
        ))
        samples.append(Sample(
            1_786_492_800.0 + index,
            point.lat, point.lon, point.alt_m, False,
        ))
    return classify_track(Track("abc123", "TEST1", tuple(samples)), _airport())


def _metadata(_icao24, time_s):
    return AdsbStateMetadata(70.0, time_s, time_s)


def test_reclassify_existing_reuses_samples_and_adds_current_frame_fingerprint(tmp_path):
    paths = HarvestPaths(tmp_path, "KAAA")
    classified = _classified()
    legacy = track_record(classified)
    legacy["observed_threshold_event"].pop("threshold_frame_fingerprint", None)
    legacy["observed_threshold_event"].pop("threshold_frame_snapshot", None)
    relative = f"assigned/18/{legacy['flight_key']}.json"
    path = paths.tracks / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(legacy), encoding="utf-8")
    paths.manifest.write_text(json.dumps({
        "airport": "KAAA",
        "altitude_source": legacy["altitude_source"],
        "altitude_datum": legacy["altitude_datum"],
        "counts": {"assigned": 1, "ambiguous": 0, "unassignable": 0, "not_landing": 0},
        "total": 1,
        "provenance": {"original": True},
        "records": [{
            "flight_key": legacy["flight_key"], "file": relative,
            "outcome": "assigned", "runway": "18",
            "icao24": "abc123", "callsign": "TEST1",
            "landing_time_utc": legacy["landing_time_utc"],
            "landing_sample_index": legacy["landing_sample_index"],
        }],
    }), encoding="utf-8")
    samples_before = legacy["samples"]

    manifest = reclassify_stored_tracks(
        _airport(),
        paths,
        metadata_lookup=_metadata,
        metadata_provenance={"test": True},
    )

    assert manifest["total"] == 1
    [row] = manifest["records"]
    rewritten = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
    assert rewritten["samples"] == samples_before
    assert rewritten["observed_threshold_event"]["threshold_frame_fingerprint"] == \
        threshold_frame_fingerprint(_airport().runway("18"))
    assert row["event_status"] == "estimated"
    assert manifest["provenance"]["reclassification"]["network_access"] is False
    assert manifest["provenance"]["reclassification"]["adsb_metadata"] == {
        "test": True
    }


def test_cli_exposes_an_exclusive_no_download_reclassification_mode():
    parser = build_parser()
    args = parser.parse_args(["--airport", "KAAA", "--reclassify-existing"])
    assert args.reclassify_existing is True
    assert args.jobs >= 1


def _classified_source_timed():
    """A source-timed track (integrity present), classified — the worker-side path."""
    runway = _airport().runway("18")
    frame = runway.frame("hae")
    slope = math.tan(math.radians(3.0))
    raw = []
    for index, along_m in enumerate(range(-8_000, -100, 100)):
        point = frame.unproject(Projected(
            float(along_m), 0.0,
            runway.threshold_crossing_height_m - slope * along_m,
        ))
        time_s = 1_786_496_400.0 + index
        raw.append(Sample(
            time_s, point.lat, point.lon, point.alt_m, False,
            70.0, time_s, time_s,
        ))
    samples, integrity = source_timed_final_block(raw)
    return classify_track(
        Track("abc124", "TEST2", tuple(samples), integrity), _airport()
    )


def _write_two_record_root(root) -> HarvestPaths:
    paths = HarvestPaths(root, "KAAA")
    rows = []
    for classified in (_classified(), _classified_source_timed()):
        record = track_record(classified)
        relative = f"assigned/18/{record['flight_key']}.json"
        path = paths.tracks / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        rows.append({
            "flight_key": record["flight_key"], "file": relative,
            "outcome": "assigned", "runway": "18",
            "icao24": record["icao24"], "callsign": record["callsign"],
            "landing_time_utc": record["landing_time_utc"],
            "landing_sample_index": record["landing_sample_index"],
        })
    paths.manifest.write_text(json.dumps({
        "airport": "KAAA",
        "altitude_source": "opensky_history_geoaltitude_m",
        "altitude_datum": "hae",
        "counts": {"assigned": 2, "ambiguous": 0, "unassignable": 0, "not_landing": 0},
        "total": 2,
        "provenance": {"original": True},
        "records": rows,
    }), encoding="utf-8")
    return paths


def test_parallel_reclassification_is_byte_identical_to_serial(tmp_path):
    """jobs is throughput only: same manifest, same record bytes, any worker count.

    The two-record roster covers both worker outcomes: the source-timed track is
    classified INSIDE a worker, while the legacy track (no source_integrity) comes
    back unclassified for the parent's batched metadata path.
    """
    serial_paths = _write_two_record_root(tmp_path / "serial")
    parallel_paths = _write_two_record_root(tmp_path / "parallel")

    serial = reclassify_stored_tracks(
        _airport(), serial_paths,
        metadata_lookup=_metadata, metadata_provenance={"test": True},
    )
    parallel = reclassify_stored_tracks(
        _airport(), parallel_paths,
        metadata_lookup=_metadata, metadata_provenance={"test": True},
        jobs=2,
    )

    assert serial["provenance"]["reclassification"].pop("jobs") == 1
    assert parallel["provenance"]["reclassification"].pop("jobs") == 2
    for manifest in (serial, parallel):
        manifest["provenance"]["reclassification"].pop("completed_utc")
    assert serial == parallel
    for row in serial["records"]:
        assert (serial_paths.tracks / row["file"]).read_bytes() == \
            (parallel_paths.tracks / row["file"]).read_bytes()

    # The worker-classified track went through the CURRENT estimator: its censored
    # event carries the extrapolated crossing ground speed (constant 70 m/s track).
    timed_row = next(
        row for row in parallel["records"] if row["icao24"] == "abc124"
    )
    rewritten = json.loads(
        (parallel_paths.tracks / timed_row["file"]).read_text(encoding="utf-8")
    )
    event = rewritten["observed_threshold_event"]
    assert event["method"] == "censored_robust_line"
    assert event["crossing_ground_speed_m_s"] == pytest.approx(70.0)


def test_source_timed_track_round_trip_preserves_subsecond_time_and_speed(tmp_path):
    base = _classified().track
    samples = tuple(
        Sample(
            sample.time_s + 0.375,
            sample.lat,
            sample.lon,
            sample.alt_hae_m,
            sample.on_ground,
            71.25,
            sample.time_s + 0.375,
            sample.time_s + 0.4,
        )
        for sample in base.samples
    )
    integrity = SourceIntegrity(
        SOURCE_INTEGRITY_SCHEMA,
        input_rows=len(samples),
        metadata_missing_rows=0,
        stale_last_contact_rows=0,
        stale_position_rows=0,
        future_timestamp_rows=0,
        inconsistent_position_groups=0,
        geoaltitude_async_groups=0,
        held_rows_removed=0,
        coverage_gap_count=0,
        retained_rows=len(samples),
    )
    classified = classify_track(
        Track(base.icao24, base.callsign, samples, integrity), _airport()
    )
    record = track_record(classified)
    path = tmp_path / "track.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    restored = _stored_track(record, path)

    assert restored.start_s == pytest.approx(samples[0].time_s, abs=0.0005)
    assert restored.samples[0].reported_ground_speed_m_s == 71.25
    assert restored.source_integrity == integrity


def test_reclassify_preserves_population_integrity_audit_without_nested_copy(tmp_path):
    base = _classified().track
    integrity = SourceIntegrity(
        SOURCE_INTEGRITY_SCHEMA,
        input_rows=len(base.samples),
        metadata_missing_rows=0,
        stale_last_contact_rows=0,
        stale_position_rows=0,
        future_timestamp_rows=0,
        inconsistent_position_groups=0,
        geoaltitude_async_groups=0,
        held_rows_removed=0,
        coverage_gap_count=0,
        retained_rows=len(base.samples),
    )
    classified = classify_track(
        Track(base.icao24, base.callsign, base.samples, integrity), _airport()
    )
    record = track_record(classified)
    relative = f"assigned/18/{record['flight_key']}.json"
    paths = HarvestPaths(tmp_path, "KAAA")
    path = paths.tracks / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    population_audit = {
        "schema_version": SOURCE_INTEGRITY_SCHEMA,
        "source_total": 2,
        "source_counts": {
            "assigned": 2,
            "ambiguous": 0,
            "unassignable": 0,
            "not_landing": 0,
        },
        "output_total": 1,
        "excluded_total": 1,
        "excluded": [
            {
                "source_flight_key": "EXCLUDED_18_def456_20260801T000000Z",
                "source_outcome": "assigned",
                "reason": "final fresh position block has fewer than two samples",
                "source_integrity": integrity.to_dict(),
            }
        ],
        "totals": {},
    }
    paths.manifest.write_text(
        json.dumps(
            {
                "schema_version": "harvest-tracks-v2-source-timing",
                "airport": "KAAA",
                "altitude_source": record["altitude_source"],
                "altitude_datum": record["altitude_datum"],
                "counts": {
                    "assigned": 1,
                    "ambiguous": 0,
                    "unassignable": 0,
                    "not_landing": 0,
                },
                "per_runway": {"18": 1},
                "total": 1,
                "source_integrity_complete": True,
                "source_integrity": population_audit,
                "provenance": {
                    "freshness_rebuild": {
                        "source_integrity": population_audit,
                    }
                },
                "records": [
                    {
                        "flight_key": record["flight_key"],
                        "file": relative,
                        "outcome": "assigned",
                        "runway": "18",
                        "icao24": "abc123",
                        "callsign": "TEST1",
                        "landing_time_utc": record["landing_time_utc"],
                        "landing_sample_index": record["landing_sample_index"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = reclassify_stored_tracks(
        _airport(),
        paths,
        metadata_lookup=lambda *_args: pytest.fail(
            "source-timed reclassification must not query ADS-B metadata"
        ),
        metadata_provenance={"unused": True},
    )

    assert manifest["source_integrity"] == population_audit
    assert "source_integrity" not in manifest["provenance"]["freshness_rebuild"]


def test_reclassification_orders_every_outcome_by_canonical_flight_time():
    rows = [
        {
            "flight_key": "LATE_not_landing_abc003_20260701T000000Z",
            "outcome": "not_landing",
            "landing_time_utc": None,
        },
        {
            "flight_key": "MIDDLE_18_abc002_20260601T000000Z",
            "outcome": "assigned",
            "landing_time_utc": "2026-06-01T00:00:00Z",
        },
        {
            "flight_key": "EARLY_unassignable_abc001_20260501T000000Z",
            "outcome": "unassignable",
            "landing_time_utc": None,
        },
    ]

    ordered = sorted(rows, key=_reclassification_order)

    assert [row["flight_key"].split("_", 1)[0] for row in ordered] == [
        "EARLY",
        "MIDDLE",
        "LATE",
    ]
