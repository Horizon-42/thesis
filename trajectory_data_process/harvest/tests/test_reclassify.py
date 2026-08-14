"""Stored samples can be reclassified without an OpenSky download."""

from __future__ import annotations

import json
import math

from final_approach import Projected

from trajectory_data_process.harvest.__main__ import build_parser
from trajectory_data_process.harvest.airports import Airport, Runway, runway_data_fingerprint
from trajectory_data_process.harvest.classify import classify_track
from trajectory_data_process.harvest.reclassify import (
    _reclassification_order,
    reclassify_stored_tracks,
)
from trajectory_data_process.harvest.store import HarvestPaths, track_record
from trajectory_data_process.harvest.tracks import Sample, Track


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


def test_reclassify_existing_reuses_samples_and_adds_current_frame_fingerprint(tmp_path):
    paths = HarvestPaths(tmp_path, "KAAA")
    classified = _classified()
    legacy = track_record(classified)
    legacy["observed_threshold_event"].pop("runway_data_fingerprint", None)
    legacy["observed_threshold_event"].pop("runway_data", None)
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
        metadata_lookup=lambda _icao24, _time_s: None,
        metadata_provenance={"test": True},
    )

    assert manifest["total"] == 1
    [row] = manifest["records"]
    rewritten = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
    assert rewritten["samples"] == samples_before
    assert rewritten["observed_threshold_event"]["runway_data_fingerprint"] == \
        runway_data_fingerprint(_airport().runway("18"))
    assert row["event_status"] == "estimated"
    assert manifest["provenance"]["reclassification"]["network_access"] is False
    assert manifest["provenance"]["reclassification"]["adsb_metadata"] == {
        "test": True
    }


def test_cli_exposes_an_exclusive_no_download_reclassification_mode():
    parser = build_parser()
    args = parser.parse_args(["--airport", "KAAA", "--reclassify-existing"])
    assert args.reclassify_existing is True


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
