"""Observed CZML has one physical entity store, with runway metadata as a view."""

from __future__ import annotations

import json

from flight_scenarios.identity import flight_key
from trajectory_data_process.harvest.czml import _extrapolated_waypoints, render_observed_czml
from trajectory_data_process.harvest.store import ALTITUDE_DATUM, ALTITUDE_SOURCE, HarvestPaths


def test_render_writes_one_canonical_czml_and_removes_legacy_partitions(tmp_path):
    paths = HarvestPaths(tmp_path / "harvest", "KAAA")
    rows = []
    for index, runway in enumerate(("01", "19")):
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
            "altitude_source": ALTITUDE_SOURCE,
            "altitude_datum": ALTITUDE_DATUM,
            "samples": [
                [0.0, -78.0, 35.0, 200.0],
                [1.0, -77.999, 35.0, 190.0],
            ],
        }
        path = paths.tracks / relative
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
    paths.manifest.write_text(
        json.dumps({"records": rows}), encoding="utf-8"
    )

    frontend = tmp_path / "frontend"
    legacy = frontend / "airports" / "KAAA" / "landings" / "KAAA_01.czml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    old_work = paths.approach / "_czml_input"
    old_work.mkdir(parents=True)
    (old_work / "stale.json").write_text("[]", encoding="utf-8")

    rendered = render_observed_czml(paths, frontend_data_root=frontend)

    assert rendered.runway_counts == {"01": 1, "19": 1}
    assert not legacy.exists()
    assert not old_work.exists()
    packets = json.loads(rendered.combined_czml.read_text(encoding="utf-8"))
    assert len(packets) == 3
    assert {packet["properties"]["runway"] for packet in packets[1:]} == {"01", "19"}
    manifest = json.loads(rendered.manifest.read_text(encoding="utf-8"))
    assert {row["file"] for row in manifest["runways"]} == {"trajectories.czml"}


def test_extrapolated_tail_uses_the_serialized_event_and_fit_source_range():
    track = {
        "flight_key": "TEST_01_abc_20260812T000000Z",
        "runway": "01",
        "landing_sample_index": 3,
        "samples": [
            [0.0, -78.0, 35.0, 200.0],
            [1.0, -78.0, 35.0005, 190.0],
            [2.0, -78.0, 35.0010, 180.0],
            [3.0, -78.0, 35.0015, 170.0],
        ],
        "observed_threshold_event": {
            "schema_version": "observed-threshold-event-v4",
            "status": "estimated",
            "method": "final_segment_window_ensemble",
            "method_version": 4,
            "runway": "01",
            "component_source_sample_ranges": {
                "lateral": [0, 2], "vertical": [0, 2],
            },
            "extrapolation_m": 300.0,
            "threshold_crossing_lon": -78.0,
            "threshold_crossing_lat": 35.0037,
            "threshold_crossing_altitude_m": 150.0,
        },
    }

    tail = _extrapolated_waypoints(track)

    assert tail[0][0] == 2.0
    assert tail[0][2] == 35.001
    assert tail[1][1:] == [-78.0, 35.0037, 150.0]


def test_direct_lateral_threshold_event_does_not_add_an_extrapolated_czml_tail():
    track = {
        "flight_key": "DIRECT_01_abc_20260812T000000Z",
        "runway": "01",
        "landing_sample_index": 2,
        "samples": [
            [0.0, -78.0, 35.0, 200.0],
            [1.0, -78.0, 35.0005, 190.0],
            [2.0, -78.0, 35.0010, 180.0],
        ],
        "observed_threshold_event": {
            "schema_version": "observed-threshold-event-v4",
            "status": "estimated",
            "method": "direct_lateral_fitted_vertical",
            "method_version": 4,
            "runway": "01",
            "component_source_sample_ranges": {
                "lateral": [1, 2], "vertical": [0, 1],
            },
            "lateral_extrapolation_m": 0.0,
            "vertical_extrapolation_m": 300.0,
            "extrapolation_m": 300.0,
        },
    }

    assert _extrapolated_waypoints(track) is None
