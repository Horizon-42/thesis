"""ADS-B sidecar lookup for no-download threshold-event regeneration."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trajectory_data_process.harvest.adsb_metadata import SidecarStateMetadata


def _write_partition(tmp_path, rows):
    airport = tmp_path / "KAAA"
    state = airport / "state_vectors"
    state.mkdir(parents=True)
    path = state / "part.parquet"
    frame = pd.DataFrame(rows)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"aeroviz_adsb_backfill"] = json.dumps({
        "source_query_digest": "part",
        "start_utc": "2026-08-01T00:00:00Z",
        "stop_utc": "2026-08-01T01:00:00Z",
    }).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)
    (airport / "manifest.json").write_text(json.dumps({
        "airport": "KAAA",
        "schema": "aeroviz.adsb-metadata-backfill.v1",
        "source_contract": {
            "state_vector_key": ["time", "icao24"],
            "reported_velocity_unit": "m/s",
            "lastposupdate_unit": "unix_seconds",
            "lastcontact_unit": "unix_seconds",
        },
        "partitions": {
            "state_vectors": [{"file": "state_vectors/part.parquet"}],
        },
    }), encoding="utf-8")
    return airport


def test_lookup_returns_reported_speed_and_real_position_update_time(tmp_path):
    timestamp = pd.Timestamp("2026-08-01T00:10:00Z")
    _write_partition(tmp_path, [{
        "time": timestamp,
        "icao24": "ABC123",
        "velocity": 72.5,
        "lastposupdate": timestamp.timestamp() - 0.3,
        "lastcontact": timestamp.timestamp() - 0.1,
    }])

    metadata = SidecarStateMetadata(tmp_path, "KAAA").lookup(
        "abc123", timestamp.timestamp()
    )

    assert metadata is not None
    assert metadata.reported_ground_speed_m_s == 72.5
    assert metadata.last_position_update_s == timestamp.timestamp() - 0.3
    assert metadata.last_contact_s == timestamp.timestamp() - 0.1


def test_conflicting_duplicate_state_rows_are_not_guessed(tmp_path):
    timestamp = pd.Timestamp("2026-08-01T00:10:00Z")
    _write_partition(tmp_path, [
        {
            "time": timestamp,
            "icao24": "abc123",
            "velocity": 70.0,
            "lastposupdate": timestamp.timestamp() - 0.3,
            "lastcontact": timestamp.timestamp() - 0.1,
        },
        {
            "time": timestamp,
            "icao24": "abc123",
            "velocity": 90.0,
            "lastposupdate": timestamp.timestamp() - 0.3,
            "lastcontact": timestamp.timestamp() - 0.1,
        },
    ])

    assert SidecarStateMetadata(tmp_path, "KAAA").lookup(
        "abc123", timestamp.timestamp()
    ) is None


def test_duplicate_aware_backfill_contract_is_supported(tmp_path):
    timestamp = pd.Timestamp("2026-08-01T00:10:00Z")
    airport = _write_partition(tmp_path, [{
        "time": timestamp,
        "icao24": "abc123",
        "velocity": 72.5,
        "lastposupdate": timestamp.timestamp() - 0.3,
        "lastcontact": timestamp.timestamp() - 0.1,
    }])
    manifest_path = airport / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_contract"].update({
        "state_vector_key": None,
        "state_vector_row_alignment": "preserved source-cache row order",
        "state_vector_identity": (
            "full source state-vector row plus occurrence among identical rows"
        ),
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert SidecarStateMetadata(tmp_path, "KAAA").lookup(
        "abc123", timestamp.timestamp()
    ) is not None
