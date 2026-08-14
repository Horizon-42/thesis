"""Safety and data-contract tests for the additive ADS-B metadata backfill."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from trajectory_data_process import backfill_adsb_metadata as backfill


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path) -> tuple[backfill.AirportPlan, Path, Path]:
    harvest_root = tmp_path / "outputs" / "harvest"
    tracks = harvest_root / "KAAA" / "tracks"
    tracks.mkdir(parents=True)
    manifest = tracks / "manifest.json"
    manifest.write_text(
        json.dumps({"airport": "KAAA", "records": [], "provenance": {}}),
        encoding="utf-8",
    )

    cache_root = tmp_path / "opensky-cache"
    cache_root.mkdir()
    source_cache = cache_root / "source-query.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-05-01T00:00:01Z", "2026-05-01T00:00:02Z"]
            ),
            "icao24": ["abc123", "abc123"],
            "lat": [35.5, 35.6],
            "lon": [-78.5, -78.4],
            "velocity": [70.0, 71.5],
            "heading": [180.0, 181.0],
            "vertrate": [-2.0, -1.5],
            "callsign": ["TEST1   ", "TEST1   "],
            "onground": [False, False],
            "baroaltitude": [1000.0, 990.0],
            "geoaltitude": [1020.0, 1010.0],
        }
    ).to_parquet(source_cache, index=False)

    chunk = backfill.QueryChunk(
        airport="KAAA",
        start_utc="2026-05-01T00:00:00Z",
        stop_utc="2026-05-01T06:00:00Z",
        bounds=(-79.0, 35.0, -78.0, 36.0),
        source_query_digest="source-query",
        source_cache_path=source_cache,
    )
    plan = backfill.AirportPlan(
        airport="KAAA",
        harvest_root=harvest_root,
        source_manifest=manifest,
        source_manifest_sha256=_sha256(manifest),
        chunks=(chunk,),
    )
    return plan, harvest_root, cache_root


class _Provider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_state_freshness(
        self, chunk: backfill.QueryChunk, source_stop: pd.Timestamp
    ) -> pd.DataFrame:
        self.calls.append("state")
        assert source_stop == pd.Timestamp("2026-05-01T00:00:02Z")
        return pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2026-05-01T00:00:01Z", "2026-05-01T00:00:02Z"]
                ),
                "icao24": ["abc123", "abc123"],
                "lat": [35.5, 35.6],
                "lon": [-78.5, -78.4],
                "velocity": [70.0, 71.5],
                "heading": [180.0, 181.0],
                "vertrate": [-2.0, -1.5],
                "callsign": ["TEST1", "TEST1"],
                "onground": [False, False],
                "baroaltitude": [1000.0, 990.0],
                "geoaltitude": [1020.0, 1010.0],
                "lastposupdate": [1777593600.8, 1777593601.8],
                "lastcontact": [1777593600.9, 1777593601.9],
            }
        )

    def fetch_operational_status(
        self,
        chunk: backfill.QueryChunk,
        windows: tuple[backfill.AircraftWindow, ...],
    ) -> pd.DataFrame:
        self.calls.append("status")
        assert len(windows) == 1
        assert windows[0].icao24 == "abc123"
        assert windows[0].first_seen == pd.Timestamp("2026-05-01T00:00:01Z")
        assert windows[0].last_seen == pd.Timestamp("2026-05-01T00:00:02Z")
        return pd.DataFrame(
            {
                "mintime": pd.to_datetime(["2026-05-01T00:00:01Z"]),
                "maxtime": pd.to_datetime(["2026-05-01T00:00:01.5Z"]),
                "icao24": ["abc123"],
                "rawmsg": ["8daabbccddeeff"],
                "version": [2],
                "geometricverticalaccuracy": [2],
            }
        )


class _ForbiddenProvider:
    def fetch_state_freshness(
        self, chunk: backfill.QueryChunk, source_stop: pd.Timestamp
    ) -> pd.DataFrame:
        raise AssertionError("dry-run/resume must not query OpenSky")

    def fetch_operational_status(
        self,
        chunk: backfill.QueryChunk,
        windows: tuple[backfill.AircraftWindow, ...],
    ) -> pd.DataFrame:
        raise AssertionError("dry-run/resume must not query OpenSky")


class _InterruptDuringStatusProvider(_Provider):
    def fetch_operational_status(
        self,
        chunk: backfill.QueryChunk,
        windows: tuple[backfill.AircraftWindow, ...],
    ) -> pd.DataFrame:
        self.calls.append("status")
        raise KeyboardInterrupt


class _TransientStateProvider(_Provider):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.state_attempts = 0

    def fetch_state_freshness(
        self, chunk: backfill.QueryChunk, source_stop: pd.Timestamp
    ) -> pd.DataFrame:
        self.state_attempts += 1
        if self.state_attempts <= self.failures:
            raise httpx.ConnectTimeout("temporary OpenSky token timeout")
        return super().fetch_state_freshness(chunk, source_stop)


def test_dry_run_neither_writes_nor_calls_network_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    before = {
        plan.source_manifest: _sha256(plan.source_manifest),
        plan.chunks[0].source_cache_path: _sha256(
            plan.chunks[0].source_cache_path
        ),
    }

    result = backfill.execute_airport(
        plan,
        output_root=output_root,
        cache_root=cache_root,
        provider=_ForbiddenProvider(),
        apply=False,
    )

    assert result.mode == "dry-run"
    assert not output_root.exists()
    assert {path: _sha256(path) for path in before} == before


def test_apply_writes_only_additive_sidecars_and_preserves_source_bytes(
    tmp_path: Path,
) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    before = {
        plan.source_manifest: _sha256(plan.source_manifest),
        plan.chunks[0].source_cache_path: _sha256(
            plan.chunks[0].source_cache_path
        ),
    }
    provider = _Provider()

    result = backfill.execute_airport(
        plan,
        output_root=output_root,
        cache_root=cache_root,
        provider=provider,
        apply=True,
        available_bytes=10**12,
    )

    assert result.mode == "applied"
    assert provider.calls == ["state", "status"]
    assert {path: _sha256(path) for path in before} == before

    airport_output = output_root / "KAAA"
    state = pd.read_parquet(airport_output / "state_vectors" / "source-query.parquet")
    status = pd.read_parquet(
        airport_output / "operational_status" / "source-query.parquet"
    )
    manifest = json.loads((airport_output / "manifest.json").read_text())

    assert list(state.columns) == [
        "time",
        "icao24",
        "velocity",
        "lastposupdate",
        "lastcontact",
    ]
    assert state["velocity"].tolist() == [70.0, 71.5]
    assert status.loc[0, "geometricverticalaccuracy"] == 2
    assert manifest["source_manifest"]["sha256"] == before[plan.source_manifest]
    assert manifest["tables"]["state_vectors"]["rows"] == 2
    assert manifest["tables"]["state_vectors"]["nonunique_time_icao_rows"] == 0
    assert manifest["source_contract"]["state_vector_key"] == ["time", "icao24"]
    assert manifest["tables"]["operational_status"]["gva_available_rows"] == 1


def test_completed_run_is_reused_without_network_or_overwrite(tmp_path: Path) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    first = backfill.execute_airport(
        plan,
        output_root=output_root,
        cache_root=cache_root,
        provider=_Provider(),
        apply=True,
        available_bytes=10**12,
    )
    state_path = output_root / "KAAA" / "state_vectors" / "source-query.parquet"
    status_path = output_root / "KAAA" / "operational_status" / "source-query.parquet"
    before = {_path: _sha256(_path) for _path in (state_path, status_path)}

    second = backfill.execute_airport(
        plan,
        output_root=output_root,
        cache_root=cache_root,
        provider=_ForbiddenProvider(),
        apply=True,
        available_bytes=10**12,
    )

    assert first.mode == "applied"
    assert second.mode == "already-complete"
    assert {_path: _sha256(_path) for _path in before} == before


def test_interrupted_status_query_reuses_completed_state_partition(
    tmp_path: Path,
) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    interrupted = _InterruptDuringStatusProvider()

    with pytest.raises(KeyboardInterrupt):
        backfill.execute_airport(
            plan,
            output_root=output_root,
            cache_root=cache_root,
            provider=interrupted,
            apply=True,
            available_bytes=10**12,
        )

    state_path = output_root / "KAAA/state_vectors/source-query.parquet"
    status_path = output_root / "KAAA/operational_status/source-query.parquet"
    manifest_path = output_root / "KAAA/manifest.json"
    state_hash = _sha256(state_path)
    assert interrupted.calls == ["state", "status"]
    assert not status_path.exists()
    assert not manifest_path.exists()

    resumed = _Provider()
    result = backfill.execute_airport(
        plan,
        output_root=output_root,
        cache_root=cache_root,
        provider=resumed,
        apply=True,
        available_bytes=10**12,
    )

    assert result.mode == "applied"
    assert resumed.calls == ["status"]
    assert _sha256(state_path) == state_hash
    assert status_path.is_file()
    assert manifest_path.is_file()


def test_transient_opensky_timeout_retries_same_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    provider = _TransientStateProvider(failures=2)
    delays: list[float] = []
    monkeypatch.setattr(backfill, "NETWORK_RETRY_DELAYS_SECONDS", (1.0, 2.0))
    monkeypatch.setattr(backfill, "_sleep", delays.append)

    result = backfill.execute_airport(
        plan,
        output_root=output_root,
        cache_root=cache_root,
        provider=provider,
        apply=True,
        available_bytes=10**12,
    )

    assert result.mode == "applied"
    assert provider.state_attempts == 3
    assert provider.calls == ["state", "status"]
    assert delays == [1.0, 2.0]
    assert (output_root / "KAAA/manifest.json").is_file()


def test_persistent_opensky_timeout_stops_after_bounded_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    provider = _TransientStateProvider(failures=10)
    delays: list[float] = []
    monkeypatch.setattr(backfill, "NETWORK_RETRY_DELAYS_SECONDS", (1.0, 2.0))
    monkeypatch.setattr(backfill, "_sleep", delays.append)

    with pytest.raises(httpx.ConnectTimeout, match="token timeout"):
        backfill.execute_airport(
            plan,
            output_root=output_root,
            cache_root=cache_root,
            provider=provider,
            apply=True,
            available_bytes=10**12,
        )

    assert provider.state_attempts == 3
    assert provider.calls == []
    assert delays == [1.0, 2.0]
    assert not (output_root / "KAAA/manifest.json").exists()


def test_non_network_query_error_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _harvest_root, cache_root = _plan(tmp_path)
    output_root = tmp_path / "outputs" / "adsb-metadata"
    provider = _TransientStateProvider(failures=0)
    attempts = 0
    delays: list[float] = []

    def invalid_state(
        _chunk: backfill.QueryChunk, _source_stop: pd.Timestamp
    ) -> pd.DataFrame:
        nonlocal attempts
        attempts += 1
        raise ValueError("historical rows changed")

    provider.fetch_state_freshness = invalid_state  # type: ignore[method-assign]
    monkeypatch.setattr(backfill, "_sleep", delays.append)

    with pytest.raises(ValueError, match="historical rows changed"):
        backfill.execute_airport(
            plan,
            output_root=output_root,
            cache_root=cache_root,
            provider=provider,
            apply=True,
            available_bytes=10**12,
        )

    assert attempts == 1
    assert delays == []


def test_refuses_output_that_overlaps_protected_inputs(tmp_path: Path) -> None:
    plan, harvest_root, cache_root = _plan(tmp_path)

    with pytest.raises(ValueError, match="overlaps protected input"):
        backfill.validate_disjoint_roots(
            harvest_root=harvest_root,
            cache_root=cache_root,
            output_root=harvest_root / "metadata",
        )

    with pytest.raises(ValueError, match="overlaps protected input"):
        backfill.validate_disjoint_roots(
            harvest_root=harvest_root,
            cache_root=cache_root,
            output_root=cache_root,
        )


def test_state_join_rejects_a_changed_historical_row_set() -> None:
    velocity = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-05-01T00:00:01Z"]),
            "icao24": ["abc123"],
            "lat": [35.5],
            "lon": [-78.5],
            "velocity": [70.0],
            "heading": [180.0],
            "vertrate": [-2.0],
            "callsign": ["TEST1"],
            "onground": [False],
            "baroaltitude": [1000.0],
            "geoaltitude": [1020.0],
        }
    )
    freshness = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-05-01T00:00:02Z"]),
            "icao24": ["abc123"],
            "lat": [35.5],
            "lon": [-78.5],
            "velocity": [70.0],
            "heading": [180.0],
            "vertrate": [-2.0],
            "callsign": ["TEST1"],
            "onground": [False],
            "baroaltitude": [1000.0],
            "geoaltitude": [1020.0],
            "lastposupdate": [1777593601.8],
            "lastcontact": [1777593601.9],
        }
    )

    with pytest.raises(ValueError, match="row identities differ"):
        backfill.merge_state_metadata(velocity, freshness)


def test_state_join_accepts_raw_unix_seconds_from_literal_query() -> None:
    velocity = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-05-01T00:00:01Z"]),
            "icao24": ["abc123"],
            "lat": [35.5],
            "lon": [-78.5],
            "velocity": [70.0],
            "heading": [180.0],
            "vertrate": [-2.0],
            "callsign": ["TEST1   "],
            "onground": [False],
            "baroaltitude": [1000.0],
            "geoaltitude": [1020.0],
        }
    )
    freshness = pd.DataFrame(
        {
            "time": [1777593601],
            "icao24": ["abc123"],
            "lat": [35.5],
            "lon": [-78.5],
            "velocity": [70.0],
            "heading": [180.0],
            "vertrate": [-2.0],
            "callsign": ["TEST1"],
            "onground": [False],
            "baroaltitude": [1000.0],
            "geoaltitude": [1020.0],
            "lastposupdate": [1777593600.8],
            "lastcontact": [1777593600.9],
        }
    )

    merged = backfill.merge_state_metadata(velocity, freshness)

    assert merged["time"].tolist() == [pd.Timestamp("2026-05-01T00:00:01Z")]
    assert str(merged["time"].dtype) == "datetime64[ms, UTC]"
    assert merged["lastcontact"].tolist() == [1777593600.9]


def test_state_join_preserves_conflicting_and_exact_duplicate_source_rows(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-05-01T00:00:01Z",
                    "2026-05-01T00:00:01Z",
                    "2026-05-01T00:00:02Z",
                    "2026-05-01T00:00:02Z",
                ]
            ),
            "icao24": ["ABC123", "abc123", "abc123", "abc123"],
            "lat": [35.5, 35.6, 35.7, 35.7],
            "lon": [-78.5, -78.4, -78.3, -78.3],
            "velocity": [70.0, 71.0, 72.0, 72.0],
            "heading": [180.0, 181.0, 182.0, 182.0],
            "vertrate": [-2.0, -1.0, 0.0, 0.0],
            "callsign": ["TEST1   ", "TEST1   ", "TEST1   ", "TEST1   "],
            "onground": [False, False, False, False],
            "baroaltitude": [1000.0, 990.0, 980.0, 980.0],
            "geoaltitude": [1020.0, 1010.0, 1000.0, 1000.0],
        }
    )
    enriched = source.iloc[[1, 0, 3, 2]].copy().reset_index(drop=True)
    enriched["callsign"] = enriched["callsign"].str.strip()
    enriched["lastposupdate"] = [11.0, 10.0, 21.0, 20.0]
    enriched["lastcontact"] = [11.5, 10.5, 21.5, 20.5]

    merged = backfill.merge_state_metadata(source, enriched)

    assert len(merged) == len(source)
    assert merged["velocity"].tolist() == [70.0, 71.0, 72.0, 72.0]
    assert merged["lastcontact"].tolist() == [10.5, 11.5, 20.5, 21.5]

    partition = tmp_path / "state_vectors" / "source-query.parquet"
    partition.parent.mkdir()
    merged.to_parquet(partition, index=False)
    summary = backfill._partition_summary(partition, merged)
    plan, _harvest_root, _cache_root = _plan(tmp_path / "manifest")
    manifest = backfill._build_output_manifest(plan, [summary], [])
    assert manifest["tables"]["state_vectors"]["nonunique_time_icao_rows"] == 4
    assert manifest["source_contract"]["state_vector_key"] is None


def test_state_query_submits_literal_sql_without_legacy_prepare() -> None:
    from sqlalchemy import literal, select

    calls: list[object] = []

    class FakeClient:
        def history(self, **kwargs: object) -> pd.DataFrame:
            statement = select(literal("abc123").label("icao24"))
            return self.query(statement, cached=True)

        def query(self, statement: object, **kwargs: object) -> pd.DataFrame:
            calls.append(statement)
            return pd.DataFrame(
                {
                    "time": [1777593601],
                    "icao24": ["abc123"],
                    "lat": [35.5],
                    "lon": [-78.5],
                    "velocity": [70.0],
                    "heading": [180.0],
                    "vertrate": [-2.0],
                    "callsign": ["TEST1"],
                    "onground": [False],
                    "baroaltitude": [1000.0],
                    "geoaltitude": [1020.0],
                    "lastposupdate": [1777593600.8],
                    "lastcontact": [1777593600.9],
                }
            )

    provider = object.__new__(backfill.OpenSkyMetadataProvider)
    provider._client = FakeClient()
    chunk = backfill.QueryChunk(
        airport="KAAA",
        start_utc="2026-05-01T00:00:00Z",
        stop_utc="2026-05-01T06:00:00Z",
        bounds=(-79.0, 35.0, -78.0, 36.0),
        source_query_digest="source-query",
        source_cache_path=Path("unused"),
    )

    result = provider.fetch_state_freshness(
        chunk, pd.Timestamp("2026-05-01T00:00:01Z")
    )

    assert len(calls) == 1
    assert isinstance(calls[0], str)
    assert "abc123" in calls[0]
    assert result["lastcontact"].tolist() == [1777593600.9]


def test_operational_status_is_limited_to_source_aircraft_time_windows() -> None:
    state = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-05-01T00:00:10Z",
                    "2026-05-01T00:00:20Z",
                    "2026-05-01T00:01:00Z",
                    "2026-05-01T00:01:10Z",
                ]
            ),
            "icao24": ["ABC123", "abc123", "def456", "def456"],
        }
    )
    status = pd.DataFrame(
        {
            # OpenSky raw tables return Unix seconds, unlike state-vector time.
            "mintime": [
                1777593609.0,
                1777593615.0,
                1777593621.0,
                1777593665.0,
                1777593665.0,
            ],
            "icao24": ["abc123", "ABC123", "abc123", "def456", "fff999"],
            "rawmsg": ["before", "inside-a", "after", "inside-b", "unknown"],
        }
    )

    filtered = backfill.filter_operational_status_to_state_windows(status, state)

    assert filtered["rawmsg"].tolist() == ["inside-a", "inside-b"]
    assert filtered["icao24"].tolist() == ["ABC123", "def456"]
    assert filtered["mintime"].tolist() == [1777593615.0, 1777593665.0]
    assert pd.api.types.is_numeric_dtype(filtered["mintime"].dtype)


def test_operational_status_query_pushes_source_windows_without_state_join(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    class FakeClient:
        def query(self, statement: object, **kwargs: object) -> pd.DataFrame:
            calls.append(statement)
            return backfill._empty_operational_status()

    provider = backfill.OpenSkyMetadataProvider()
    provider._client = FakeClient()
    plan, _harvest_root, _cache_root = _plan(tmp_path)
    windows = (
        backfill.AircraftWindow(
            icao24="abc123",
            first_seen=pd.Timestamp("2026-05-01T00:00:01Z"),
            last_seen=pd.Timestamp("2026-05-01T00:00:02Z"),
        ),
        backfill.AircraftWindow(
            icao24="def456",
            first_seen=pd.Timestamp("2026-05-01T00:01:00Z"),
            last_seen=pd.Timestamp("2026-05-01T00:02:00Z"),
        ),
    )

    result = provider.fetch_operational_status(plan.chunks[0], windows)

    assert len(calls) == 1
    assert isinstance(calls[0], str)
    sql = calls[0].lower()
    assert "operational_status_data4" in sql
    assert "state_vectors_data4" not in sql
    assert "abc123" in sql
    assert "def456" in sql
    assert tuple(result.columns) == backfill.OPERATIONAL_STATUS_OUTPUT_COLUMNS


def test_exclusive_partition_writer_never_replaces_existing_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "partition.parquet"
    target.write_bytes(b"existing-user-data")

    with pytest.raises(FileExistsError):
        backfill.write_parquet_exclusive(
            pd.DataFrame({"value": [1]}),
            target,
            metadata={"schema": backfill.SCHEMA_ID},
        )

    assert target.read_bytes() == b"existing-user-data"


def test_apply_runtime_installs_server_side_query_cancellation(monkeypatch) -> None:
    installed: list[bool] = []
    monkeypatch.setattr(
        backfill,
        "install_query_cancel_on_interrupt",
        lambda: installed.append(True),
    )

    backfill.prepare_opensky_runtime(apply=True)

    assert installed == [True]


def test_dry_run_does_not_install_network_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        backfill,
        "install_query_cancel_on_interrupt",
        lambda: (_ for _ in ()).throw(AssertionError("must not install")),
    )

    backfill.prepare_opensky_runtime(apply=False)
