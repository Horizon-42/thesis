import json
from pathlib import Path

import pytest

from aeroviz_backend.observed_trajectories import ObservedTrajectoryBackend
from flight_scenarios.identity import flight_key
from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
    threshold_frame_fingerprint,
    threshold_frame_snapshot,
)
from trajectory_data_process.harvest.arrivals import write_arrival_records
from trajectory_data_process.harvest.store import (
    ALTITUDE_DATUM,
    ALTITUDE_SOURCE,
    TRACK_SCHEMA_VERSION,
    HarvestPaths,
)


def _write_harvest(root: Path) -> None:
    tracks = root / "KAAA" / "tracks"
    rows = []
    for index, runway in enumerate(("01", "01", "19", "19")):
        source = {
            "id": f"TEST{index}",
            "callsign": f"TEST{index}",
            "icao24": f"abc00{index}",
            "runway": runway,
            "landing_time_utc": f"2026-07-01T00:00:0{index}Z",
        }
        key = flight_key(source, index)
        relative = f"assigned/{runway}/{key}.json"
        record = {
            "flight_key": key,
            "icao24": source["icao24"],
            "callsign": source["callsign"],
            "outcome": "assigned",
            "runway": runway,
            "landing_time_utc": source["landing_time_utc"],
            "altitude_source": "opensky_history_geoaltitude_m",
            "altitude_datum": "hae",
            "samples": [
                [0.0, -78.0 + index * 0.01, 35.0, 200.0],
                [1.0, -77.999 + index * 0.01, 35.0, 190.0],
            ],
        }
        path = tracks / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        rows.append(
            {
                "flight_key": key,
                "file": relative,
                "outcome": "assigned",
                "runway": runway,
            }
        )
    tracks.joinpath("manifest.json").write_text(
        json.dumps({"airport": "KAAA", "records": rows}),
        encoding="utf-8",
    )


def _write_evaluation(root: Path, harvest_root: Path) -> list[str]:
    manifest = json.loads(
        (harvest_root / "KAAA" / "tracks" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    keys = [row["flight_key"] for row in manifest["records"]]
    report = {
        "subject": "observed",
        "total": 4,
        "verdict_counts": {"pass": 2, "fail": 1, "indeterminate": 1},
        "observed": {"event_estimated_rate": 0.75},
        "lateral_m": {"mean": 12.5},
        "vertical_m": {"mean_abs": 4.5},
        "trajectories": [
            {"flight_key": keys[0], "verdict": "pass"},
            {"flight_key": keys[1], "verdict": "pass"},
            {"flight_key": keys[2], "verdict": "fail"},
            {"flight_key": keys[3], "verdict": "indeterminate"},
        ],
    }
    path = root / "KAAA" / "comparison" / "observed" / "evaluation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return keys


# ── arrival-window fixture: a real harvest, so the real arrival writer produces the slice ──

# Three samples outside the 25 km terminal ring, then the final entry run to the threshold.
# The arrival window must therefore begin at index 3 (0.20 deg of longitude at the equator
# is ~22 km), and the full window at index 0.
_RING_ENTRY_INDEX = 3
_APPROACH_SAMPLES = [
    [0.0, 0.30, 0.0, 1000.0],
    [1.0, 0.29, 0.0, 950.0],
    [2.0, 0.28, 0.0, 900.0],
    [3.0, 0.20, 0.0, 700.0],
    [4.0, 0.10, 0.0, 400.0],
    [5.0, 0.01, 0.0, 100.0],
    [6.0, 0.00, 0.0, 25.0],
]


def _arrival_runway(ident: str = "18") -> Runway:
    return Runway(
        airport="KAAA",
        ident=ident,
        lat=0.0,
        lon=0.0,
        elevation_hae_m=10.0,
        elevation_msl_m=10.0,
        course_deg=90.0,
        hae_minus_msl_m=0.0,
        threshold_crossing_height_m=15.0,
        published_glidepath_deg=3.0,
        width_m=45.72,
        lpv_course_width_m=106.75,
        runway_source_cycle="2026-08-06",
        procedure_source_cycle="2026-08-06",
        position_source="faa_cifp_path_point",
        vertical_source="faa_cifp_path_point",
    )


def _write_harvest_with_arrivals(root: Path) -> tuple[HarvestPaths, str]:
    """A harvest whose single assigned track has both windows, and its arrival manifest."""
    paths = HarvestPaths(root, "KAAA")
    runway = _arrival_runway()
    landing_time = "1970-01-01T00:00:06Z"
    key = flight_key(
        {
            "id": "ARR1",
            "runway": runway.ident,
            "icao24": "aaa001",
            "landing_time_utc": landing_time,
        },
        0,
    )
    relative = f"assigned/{runway.ident}/{key}.json"
    track = {
        "flight_key": key,
        "icao24": "aaa001",
        "callsign": "ARR1",
        "outcome": "assigned",
        "runway": runway.ident,
        "landing_time_utc": landing_time,
        "landing_sample_index": 6,
        "start_time_utc": "1970-01-01T00:00:00Z",
        "duration_s": float(_APPROACH_SAMPLES[-1][0]),
        "max_sample_gap_s": 1.0,
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "assignment": {"outcome": "assigned", "runway": runway.ident},
        "observed_threshold_event": {
            "schema_version": "runway-threshold-event-v1",
            "runway": runway.ident,
            "threshold_frame_snapshot": threshold_frame_snapshot(runway),
            "threshold_frame_fingerprint": threshold_frame_fingerprint(runway),
            "status": "estimated",
            "method": "direct_linear_bracket",
            "observability": "within_observed_support",
            "event_time_s": 1.0,
            "threshold_crossing_lat": runway.lat,
            "threshold_crossing_lon": runway.lon,
            "threshold_crossing_altitude_m": runway.elevation_hae_m + 15.0,
            "altitude_datum": "hae",
            "signed_cross_track_m": 0.0,
            "source_sample_range": [0, 1],
            "interpolation_fraction": 1.0,
            "extrapolation_distance_m": 0.0,
            "uncertainty": {"status": "uncalibrated"},
        },
        "samples": _APPROACH_SAMPLES,
    }
    path = paths.tracks / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(track), encoding="utf-8")
    paths.manifest.write_text(
        json.dumps(
            {
                "schema_version": TRACK_SCHEMA_VERSION,
                "airport": "KAAA",
                "altitude_source": ALTITUDE_SOURCE,
                "altitude_datum": ALTITUDE_DATUM,
                "counts": {
                    "assigned": 1,
                    "ambiguous": 0,
                    "unassignable": 0,
                    "not_landing": 0,
                },
                "total": 1,
                "source_integrity_complete": True,
                "records": [
                    {
                        "flight_key": key,
                        "file": relative,
                        "outcome": "assigned",
                        "runway": runway.ident,
                        "icao24": "aaa001",
                        "callsign": "ARR1",
                        "landing_time_utc": landing_time,
                        "landing_sample_index": 6,
                        "event_status": "estimated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    airport = Airport(
        code="KAAA",
        lat=0.0,
        lon=0.0,
        elevation_msl_m=10.0,
        runways=(runway,),
    )
    write_arrival_records(airport, paths)
    return paths, key


def _samples_of(packet: dict) -> list[tuple[float, float, float, float]]:
    degrees = packet["position"]["cartographicDegrees"]
    return [tuple(degrees[i:i + 4]) for i in range(0, len(degrees), 4)]


def test_arrival_window_serves_the_model_slice_rebased_to_terminal_entry(tmp_path):
    """The comparison reference must share the modeling time origin, not the track's."""
    paths, key = _write_harvest_with_arrivals(tmp_path / "harvest")
    backend = ObservedTrajectoryBackend(harvest_root=paths.root)

    full = backend.query("KAAA", flight_keys=[key])
    arrival = backend.query("KAAA", flight_keys=[key], window="arrival")

    assert full["trackWindow"] == "full"
    assert arrival["trackWindow"] == "arrival"

    full_samples = _samples_of(full["czml"][1])
    arrival_samples = _samples_of(arrival["czml"][1])

    # The full window starts at first reception; the arrival window at the ring entry,
    # rebased to t=0 — the same physical sample, a different time label. Rendering the
    # first against records built on the second is the whole misalignment.
    assert full_samples[0][0] == 0.0
    assert full_samples[0][1] == pytest.approx(_APPROACH_SAMPLES[0][1])
    assert arrival_samples[0][0] == 0.0
    assert arrival_samples[0][1] == pytest.approx(
        _APPROACH_SAMPLES[_RING_ENTRY_INDEX][1]
    )
    assert [sample[0] for sample in arrival_samples] == [
        row[0] - _APPROACH_SAMPLES[_RING_ENTRY_INDEX][0]
        for row in _APPROACH_SAMPLES[_RING_ENTRY_INDEX:]
    ]
    # Read-time slicing only: the stored track keeps every sample it was harvested with.
    stored = json.loads(
        (paths.tracks / f"assigned/18/{key}.json").read_text(encoding="utf-8")
    )
    assert stored["samples"] == _APPROACH_SAMPLES


def test_arrival_window_rosters_from_the_arrival_manifest_not_the_track_manifest(tmp_path):
    """A window's roster comes from the manifest that defines that window."""
    paths, key = _write_harvest_with_arrivals(tmp_path / "harvest")
    # An assigned track that is not on the arrival roster — the real harvest excludes such
    # tracks for a local circuit or a missing published TCH/glidepath; here it is simply
    # added after the arrival manifest was written. It must still be offered in the full
    # window, and must be refused (not silently mis-sliced) in the arrival window.
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    extra = dict(manifest["records"][0])
    extra["flight_key"] = "NOMODEL_18_aaa009_19700101T000006Z"
    extra["file"] = "assigned/18/NOMODEL.json"
    extra["callsign"] = "NOMODEL"
    extra["icao24"] = "aaa009"
    track = json.loads(
        (paths.tracks / manifest["records"][0]["file"]).read_text(encoding="utf-8")
    )
    track["flight_key"] = extra["flight_key"]
    track["callsign"] = "NOMODEL"
    track["icao24"] = "aaa009"
    (paths.tracks / extra["file"]).write_text(json.dumps(track), encoding="utf-8")
    manifest["records"].append(extra)
    manifest["counts"]["assigned"] = 2
    manifest["total"] = 2
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    backend = ObservedTrajectoryBackend(harvest_root=paths.root)

    full_ids = [packet["id"] for packet in backend.query("KAAA", limit=0)["czml"][1:]]
    arrival_ids = [
        packet["id"]
        for packet in backend.query("KAAA", limit=0, window="arrival")["czml"][1:]
    ]

    assert set(full_ids) == {key, extra["flight_key"]}
    assert arrival_ids == [key]
    with pytest.raises(ValueError, match="not found"):
        backend.query("KAAA", flight_keys=[extra["flight_key"]], window="arrival")


def test_query_rejects_an_unknown_track_window(tmp_path):
    backend = ObservedTrajectoryBackend(harvest_root=tmp_path)

    with pytest.raises(ValueError, match="window must be one of"):
        backend.query("KAAA", window="approach")


def test_query_filters_before_loading_and_returns_playable_czml(tmp_path):
    harvest_root = tmp_path / "harvest"
    _write_harvest(harvest_root)
    backend = ObservedTrajectoryBackend(harvest_root=harvest_root)

    response = backend.query("kaaa", runway="01", limit=1, seed=7)
    packets = response["czml"]

    assert response["schemaVersion"] == "observed-trajectories-v2"
    assert response["trackWindow"] == "full"
    assert packets[0]["id"] == "document"
    assert len(packets) == 2
    assert packets[1]["properties"]["runway"] == "01"
    assert packets[1]["id"].startswith("TEST")
    assert len(packets[1]["position"]["cartographicDegrees"]) == 8


def test_query_sampling_is_stable_for_the_same_seed(tmp_path):
    harvest_root = tmp_path / "harvest"
    _write_harvest(harvest_root)
    backend = ObservedTrajectoryBackend(harvest_root=harvest_root)

    first = backend.query("KAAA", limit=2, seed=42)
    second = backend.query("KAAA", limit=2, seed=42)

    assert [packet["id"] for packet in first["czml"][1:]] == [
        packet["id"] for packet in second["czml"][1:]
    ]


def test_query_returns_exact_requested_flight_keys_in_request_order(tmp_path):
    harvest_root = tmp_path / "harvest"
    _write_harvest(harvest_root)
    manifest = json.loads(
        (harvest_root / "KAAA" / "tracks" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    keys = [row["flight_key"] for row in manifest["records"]]
    backend = ObservedTrajectoryBackend(harvest_root=harvest_root)

    response = backend.query(
        "KAAA",
        flight_keys=[keys[3], keys[0]],
        # Exact selection is independent of the ordinary random-sample limit.
        limit=1,
        seed=999,
    )

    assert [packet["id"] for packet in response["czml"][1:]] == [
        keys[3],
        keys[0],
    ]


def test_query_rejects_missing_or_duplicate_exact_flight_keys(tmp_path):
    harvest_root = tmp_path / "harvest"
    _write_harvest(harvest_root)
    manifest = json.loads(
        (harvest_root / "KAAA" / "tracks" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    key = manifest["records"][0]["flight_key"]
    backend = ObservedTrajectoryBackend(harvest_root=harvest_root)

    with pytest.raises(ValueError, match="not found"):
        backend.query("KAAA", flight_keys=[key, "missing-flight"])
    with pytest.raises(ValueError, match="duplicate"):
        backend.query("KAAA", flight_keys=[key, key])


def test_query_filters_by_verdict_before_sampling_and_returns_metadata(tmp_path):
    harvest_root = tmp_path / "harvest"
    evaluation_root = tmp_path / "airports"
    _write_harvest(harvest_root)
    keys = _write_evaluation(evaluation_root, harvest_root)
    backend = ObservedTrajectoryBackend(
        harvest_root=harvest_root,
        evaluation_root=evaluation_root,
    )

    response = backend.query("KAAA", limit=2, verdict="fail", seed=0)

    # There is only one failed flight. A limit of two therefore returns that flight
    # instead of sampling two records globally and filtering the result afterwards.
    assert [packet["id"] for packet in response["czml"][1:]] == [keys[2]]
    assert response["verdicts"] == {
        "counts": {"pass": 2, "fail": 1, "undecided": 1},
        "byFlightId": {keys[2]: "fail"},
        "matched": 4,
        "total": 4,
    }
    assert response["evaluation"] == {
        "total": 4,
        "verdict_counts": {"pass": 2, "fail": 1, "indeterminate": 1},
        "observed": {"event_estimated_rate": 0.75},
        "lateral_m": {"mean": 12.5},
        "vertical_m": {"mean_abs": 4.5},
    }


def test_zero_limit_returns_all_only_within_the_safe_response_cap(tmp_path):
    harvest_root = tmp_path / "harvest"
    _write_harvest(harvest_root)
    backend = ObservedTrajectoryBackend(
        harvest_root=harvest_root,
        max_trajectories=3,
    )

    runway_packets = backend.query("KAAA", runway="01", limit=0)["czml"]
    assert len(runway_packets) == 3

    with pytest.raises(ValueError, match="exceeds the safe response maximum"):
        backend.query("KAAA", limit=0)


def test_query_rejects_invalid_airport_and_excessive_positive_limit(tmp_path):
    backend = ObservedTrajectoryBackend(
        harvest_root=tmp_path,
        max_trajectories=1000,
    )

    with pytest.raises(ValueError, match="airport"):
        backend.query("../KRDU", limit=200)
    with pytest.raises(ValueError, match="limit must be"):
        backend.query("KRDU", limit=1001)

    with pytest.raises(ValueError, match="verdict"):
        backend.query("KRDU", verdict="failed")
