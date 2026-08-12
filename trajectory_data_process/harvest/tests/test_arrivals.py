"""The model-ready arrival manifest is the only modeling/training data contract."""

from __future__ import annotations

import json

import pytest

from flight_scenarios.identity import flight_key
from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
    runway_data_fingerprint,
)
from trajectory_data_process.harvest.arrivals import (
    _anchor_index,
    load_arrival_flights,
    write_arrival_records,
)
from trajectory_data_process.harvest.store import ALTITUDE_DATUM, ALTITUDE_SOURCE, HarvestPaths


def _runway(
    ident: str,
    *,
    tch_m: float | None = 15.0,
    glidepath_deg: float | None = 3.0,
) -> Runway:
    return Runway(
        airport="KAAA",
        ident=ident,
        lat=0.0,
        lon=0.0,
        elevation_hae_m=10.0,
        elevation_msl_m=10.0,
        course_deg=90.0,
        hae_minus_msl_m=0.0,
        threshold_crossing_height_m=tch_m,
        published_glidepath_deg=glidepath_deg,
        width_m=45.72,
        lpv_course_width_m=106.75 if tch_m is not None else None,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
        position_source="faa_cifp_path_point",
        vertical_source="faa_cifp_path_point",
    )


def _airport() -> Airport:
    return Airport(
        code="KAAA",
        lat=0.0,
        lon=0.0,
        elevation_msl_m=10.0,
        runways=(
            _runway("18"),
            _runway("19", tch_m=None),
            _runway("20", glidepath_deg=None),
        ),
    )


def _source_track(
    paths: HarvestPaths,
    *,
    callsign: str,
    icao24: str,
    runway: str,
    samples: list[list[float]],
    landing_sample_index: int,
) -> dict:
    landing_time = f"1970-01-01T00:00:{landing_sample_index:02d}Z"
    key = flight_key(
        {
            "id": callsign,
            "runway": runway,
            "icao24": icao24,
            "landing_time_utc": landing_time,
        },
        0,
    )
    record = {
        "flight_key": key,
        "icao24": icao24,
        "callsign": callsign,
        "outcome": "assigned",
        "runway": runway,
        "landing_time_utc": landing_time,
        "landing_sample_index": landing_sample_index,
        "start_time_utc": "1970-01-01T00:00:00Z",
        "duration_s": float(samples[-1][0]),
        "max_sample_gap_s": 1.0,
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "assignment": {"outcome": "assigned", "runway": runway},
        "observed_threshold_event": {
            "status": "estimated",
            "runway": runway,
            "runway_data_fingerprint": runway_data_fingerprint(_runway(runway)),
        },
        "samples": samples,
    }
    relative = f"assigned/{runway}/{key}.json"
    path = paths.tracks / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return {
        "flight_key": key,
        "file": relative,
        "outcome": "assigned",
        "runway": runway,
        "icao24": icao24,
        "callsign": callsign,
        "landing_time_utc": landing_time,
        "landing_sample_index": landing_sample_index,
        "event_status": "estimated",
    }


def _write_source_manifest(paths: HarvestPaths, rows: list[dict]) -> None:
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "airport": paths.code,
                "altitude_source": ALTITUDE_SOURCE,
                "altitude_datum": ALTITUDE_DATUM,
                "counts": {
                    "assigned": len(rows),
                    "ambiguous": 0,
                    "unassignable": 0,
                    "not_landing": 0,
                },
                "total": len(rows),
                "records": rows,
            }
        ),
        encoding="utf-8",
    )


def _approach_samples() -> list[list[float]]:
    # Three samples outside the 25 km ring, then final entry and threshold crossing.
    # The last sample is rollout after the landing anchor and must not reach modeling.
    return [
        [0.0, 0.30, 0.0, 1000.0],
        [1.0, 0.29, 0.0, 950.0],
        [2.0, 0.28, 0.0, 900.0],
        [3.0, 0.20, 0.0, 700.0],
        [4.0, 0.10, 0.0, 400.0],
        [5.0, 0.01, 0.0, 100.0],
        [6.0, 0.00, 0.0, 25.0],
        [7.0, -0.01, 0.0, 10.0],
    ]


def test_anchor_index_consumes_the_stored_assignment_anchor_without_refitting():
    first_pass = [
        [float(i), -0.045 + i * 0.005, 0.0, 300.0 - i * 30.0]
        for i in range(10)
    ]
    final_pass = [
        [float(20 + i), -0.045 + i * 0.005, 0.0, 300.0 - i * 30.0]
        for i in range(10)
    ]
    track = {
        "flight_key": "TWOPASS_18_abc123_19700101T000009Z",
        "landing_sample_index": 9,
        "observed_threshold_event": {
            "status": "estimated",
            "runway": "18",
            "runway_data_fingerprint": runway_data_fingerprint(_runway("18")),
        },
        "samples": [*first_pass, [10.0, 0.10, 0.0, 500.0], *final_pass],
    }

    assert _anchor_index(track, _runway("18")) == 9


def test_anchor_index_directs_legacy_tracks_to_local_reclassification():
    track = {
        "flight_key": "LEGACY",
        "landing_sample_index": 1,
        "samples": [[0.0, 0.1, 0.0, 100.0], [1.0, 0.0, 0.0, 25.0]],
    }
    with pytest.raises(ValueError, match="reclassify-existing"):
        _anchor_index(track, _runway("18"))


def test_arrival_manifest_crops_final_entry_to_landing_anchor_and_excludes_non_model_data(
    tmp_path,
):
    paths = HarvestPaths(tmp_path, "KAAA")
    rows = [
        _source_track(
            paths,
            callsign="ARR1",
            icao24="aaa001",
            runway="18",
            samples=_approach_samples(),
            landing_sample_index=6,
        ),
        _source_track(
            paths,
            callsign="LOCAL1",
            icao24="aaa002",
            runway="18",
            samples=[
                [0.0, 0.01, 0.0, 30.0],
                [1.0, 0.02, 0.0, 80.0],
                [2.0, 0.01, 0.0, 40.0],
                [3.0, 0.00, 0.0, 25.0],
            ],
            landing_sample_index=3,
        ),
        _source_track(
            paths,
            callsign="NOTCH",
            icao24="aaa003",
            runway="19",
            samples=_approach_samples(),
            landing_sample_index=6,
        ),
        _source_track(
            paths,
            callsign="NOGLIDE",
            icao24="aaa004",
            runway="20",
            samples=_approach_samples(),
            landing_sample_index=6,
        ),
    ]
    _write_source_manifest(paths, rows)

    manifest = write_arrival_records(_airport(), paths)

    assert manifest["schema_version"] == "harvest-arrivals-v3-track-slices"
    assert manifest["counts"] == {
        "source_total": 4,
        "assigned": 4,
        "included": 1,
        "local_circuit": 1,
        "no_published_tch": 1,
        "no_published_glidepath": 1,
    }
    [flight] = load_arrival_flights(paths.airport)
    assert flight["id"] == "ARR1"
    assert [row[0] for row in flight["waypoints"]] == [0.0, 1.0, 2.0, 3.0]
    assert flight["waypoints"][0][1] == pytest.approx(0.20)
    assert flight["waypoints"][-1][1] == pytest.approx(0.0)
    assert flight["arrival_truncated"] is True
    assert flight["cut_samples"] == 3
    assert flight["runway_target"]["threshold_crossing_height_m"] == 15.0
    assert flight["runway_target"]["published_glidepath_deg"] == 3.0
    assert not (paths.airport / "arrivals" / "records").exists()
    assert manifest["records"][0]["source_file"].startswith("assigned/18/")
    assert manifest["records"][0]["first_sample_index"] == 3
    assert manifest["records"][0]["last_sample_index"] == 6


def test_loader_uses_only_manifest_roster_and_rejects_duplicate_flight_keys(tmp_path):
    paths = HarvestPaths(tmp_path, "KAAA")
    row = _source_track(
        paths,
        callsign="ARR1",
        icao24="aaa001",
        runway="18",
        samples=_approach_samples(),
        landing_sample_index=6,
    )
    _write_source_manifest(paths, [row])
    write_arrival_records(_airport(), paths)

    assert [flight["id"] for flight in load_arrival_flights(paths.airport)] == ["ARR1"]

    manifest_path = paths.airport / "arrivals" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"].append(dict(manifest["records"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate flight_key"):
        load_arrival_flights(manifest_path)


def test_loader_rejects_duplicate_identity_in_source_manifest(tmp_path):
    paths = HarvestPaths(tmp_path, "KAAA")
    row = _source_track(
        paths,
        callsign="ARR1",
        icao24="aaa001",
        runway="18",
        samples=_approach_samples(),
        landing_sample_index=6,
    )
    _write_source_manifest(paths, [row])
    write_arrival_records(_airport(), paths)

    source = json.loads(paths.manifest.read_text(encoding="utf-8"))
    source["records"].append(dict(source["records"][0]))
    paths.manifest.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest.*duplicate flight_key"):
        load_arrival_flights(paths.airport)


def test_loader_rejects_a_source_track_changed_after_arrival_manifest(tmp_path):
    paths = HarvestPaths(tmp_path, "KAAA")
    row = _source_track(
        paths,
        callsign="ARR1",
        icao24="aaa001",
        runway="18",
        samples=_approach_samples(),
        landing_sample_index=6,
    )
    _write_source_manifest(paths, [row])
    write_arrival_records(_airport(), paths)
    source_path = paths.tracks / row["file"]
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical source track.*changed"):
        load_arrival_flights(paths.airport)


def test_loader_rejects_a_legacy_flight_array_instead_of_treating_it_as_a_manifest(tmp_path):
    legacy = tmp_path / "KAAA_18_landings.json"
    legacy.write_text(json.dumps([{"id": "OLD", "waypoints": []}]), encoding="utf-8")

    with pytest.raises(ValueError, match="arrival manifest"):
        load_arrival_flights(legacy)
