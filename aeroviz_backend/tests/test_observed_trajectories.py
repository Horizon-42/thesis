import json
from pathlib import Path

import pytest

from aeroviz_backend.observed_trajectories import ObservedTrajectoryBackend
from flight_scenarios.identity import flight_key


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


def test_query_filters_before_loading_and_returns_playable_czml(tmp_path):
    harvest_root = tmp_path / "harvest"
    _write_harvest(harvest_root)
    backend = ObservedTrajectoryBackend(harvest_root=harvest_root)

    response = backend.query("kaaa", runway="01", limit=1, seed=7)
    packets = response["czml"]

    assert response["schemaVersion"] == "observed-trajectories-v1"
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
