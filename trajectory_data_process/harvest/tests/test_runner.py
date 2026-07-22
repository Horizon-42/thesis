"""Regression tests for the backward harvest scan's stopping rules."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trajectory_data_process.harvest import runner as runner_module
from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.runner import HarvestPlan, harvest_airport
from trajectory_data_process.harvest.store import HarvestPaths


def _airport() -> Airport:
    runway = Runway(
        airport="KAAA",
        ident="18",
        lat=0.0,
        lon=0.0,
        elevation_msl_m=0.0,
        course_deg=180.0,
        geoid_undulation_m=0.0,
        threshold_crossing_height_m=None,
        published_glidepath_deg=None,
    )
    return Airport(
        code="KAAA",
        lat=0.0,
        lon=0.0,
        elevation_msl_m=0.0,
        runways=(runway,),
    )


def test_idle_runway_gives_up_after_four_days_of_backward_scan(tmp_path: Path) -> None:
    stop = datetime(2026, 7, 22, tzinfo=timezone.utc)
    calls: list[dict[str, Any]] = []

    def empty_fetch(**kwargs: Any) -> None:
        calls.append(kwargs)
        return None

    result = harvest_airport(
        _airport(),
        HarvestPaths(root=tmp_path, code="KAAA"),
        HarvestPlan(
            target_per_runway=1,
            start=stop,
            chunk_hours=6.0,
            max_lookback_days=30.0,
            dry_give_up_days=4.0,
        ),
        fetch=empty_fetch,
        log=lambda _message: None,
    )

    assert result.chunks_fetched == 16
    assert result.scanned_from == stop - timedelta(days=4)
    assert result.manifest["provenance"]["given_up"] == ["18"]
    assert len(calls) == 16


def test_dry_window_restarts_when_runway_count_last_increases(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop = datetime(2026, 7, 22, tzinfo=timezone.utc)
    frames = iter(
        [
            pd.DataFrame(
                [
                    {
                        "time": stop.timestamp() - 1,
                        "icao24": "abc123",
                        "lat": 0.0,
                        "lon": 0.0,
                        "callsign": "TEST1",
                        "onground": False,
                        "geoaltitude": 100.0,
                    }
                ]
            )
        ]
    )

    # One landing is found in the first chunk and no older landing is found after it.
    # The runway still needs a second landing, so its four-day dry window starts at the
    # first chunk's older boundary rather than at the harvest's fixed start.
    monkeypatch.setattr(
        runner_module,
        "classify_tracks",
        lambda *_args, **_kwargs: [SimpleNamespace(runway="18")],
    )
    monkeypatch.setattr(
        runner_module,
        "write_tracks",
        lambda _classified, _paths, *, provenance: {"provenance": provenance},
    )

    result = harvest_airport(
        _airport(),
        HarvestPaths(root=tmp_path, code="KAAA"),
        HarvestPlan(
            target_per_runway=2,
            start=stop,
            chunk_hours=6.0,
            max_lookback_days=30.0,
            dry_give_up_days=4.0,
        ),
        fetch=lambda **_kwargs: next(frames, None),
        log=lambda _message: None,
    )

    assert result.chunks_fetched == 17
    assert result.scanned_from == stop - timedelta(days=4, hours=6)
    assert result.manifest["provenance"]["given_up"] == ["18"]


def test_harvest_reconstructs_only_one_aircraft_at_a_time(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop = datetime(2026, 7, 22, tzinfo=timezone.utc)
    reconstructed_aircraft: list[set[str]] = []
    frame = pd.DataFrame(
        [
            {
                "time": stop.timestamp() - 1,
                "icao24": icao24,
                "lat": 0.0,
                "lon": 0.0,
                "callsign": icao24.upper(),
                "onground": False,
                "geoaltitude": 100.0,
            }
            for icao24 in ("abc123", "def456")
        ]
    )

    def record_reconstruction(rows: list[dict[str, Any]], **_kwargs: Any) -> list[Any]:
        reconstructed_aircraft.append({str(row["icao24"]) for row in rows})
        return []

    monkeypatch.setattr(runner_module, "reconstruct_tracks", record_reconstruction)

    result = harvest_airport(
        _airport(),
        HarvestPaths(root=tmp_path, code="KAAA"),
        HarvestPlan(
            target_per_runway=1,
            start=stop,
            chunk_hours=6.0,
            max_lookback_days=30.0,
            dry_give_up_days=0.25,
        ),
        fetch=lambda **_kwargs: frame,
        log=lambda _message: None,
    )

    assert result.chunks_fetched == 1
    assert reconstructed_aircraft
    assert all(len(aircraft) == 1 for aircraft in reconstructed_aircraft)


def test_interrupted_harvest_resumes_at_the_next_unfinished_chunk(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop = datetime(2026, 7, 22, tzinfo=timezone.utc)
    paths = HarvestPaths(root=tmp_path, code="KAAA")
    plan = HarvestPlan(
        target_per_runway=2,
        start=stop,
        chunk_hours=6.0,
        max_lookback_days=30.0,
        dry_give_up_days=0.5,
    )
    first_frame = pd.DataFrame(
        [
            {
                "time": stop.timestamp() - 1,
                "icao24": "abc123",
                "lat": 0.0,
                "lon": 0.0,
                "callsign": "TEST1",
                "onground": False,
                "geoaltitude": 100.0,
            }
        ]
    )
    monkeypatch.setattr(
        runner_module,
        "classify_tracks",
        lambda *_args, **_kwargs: [SimpleNamespace(runway="18")],
    )
    monkeypatch.setattr(
        runner_module,
        "write_tracks",
        lambda _classified, _paths, *, provenance: {"provenance": provenance},
    )

    first_calls: list[dict[str, Any]] = []

    def interrupted_fetch(**kwargs: Any) -> pd.DataFrame:
        first_calls.append(kwargs)
        if len(first_calls) == 1:
            return first_frame
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        harvest_airport(
            _airport(), paths, plan, fetch=interrupted_fetch, log=lambda _message: None
        )

    assert paths.checkpoint_state.exists()
    assert paths.checkpoint_db.exists()

    resumed_calls: list[dict[str, Any]] = []
    messages: list[str] = []

    def resumed_fetch(**kwargs: Any) -> None:
        resumed_calls.append(kwargs)
        return None

    result = harvest_airport(
        _airport(), paths, plan, fetch=resumed_fetch, log=messages.append
    )

    assert resumed_calls[0]["start"] == stop - timedelta(hours=12)
    assert resumed_calls[0]["stop"] == stop - timedelta(hours=6)
    assert result.chunks_fetched == 3
    assert any("resuming checkpoint" in message for message in messages)
    assert not paths.checkpoint.exists()


def test_transient_network_failure_retries_the_same_chunk(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop = datetime(2026, 7, 22, tzinfo=timezone.utc)
    calls: list[dict[str, Any]] = []
    delays: list[float] = []
    messages: list[str] = []

    def unstable_fetch(**kwargs: Any) -> None:
        calls.append(kwargs)
        if len(calls) == 1:
            try:
                raise socket.gaierror(-2, "Name or service not known")
            except socket.gaierror as cause:
                # httpx.ConnectError wraps the resolver failure this way.
                raise RuntimeError("OpenSky connection failed") from cause
        return None

    monkeypatch.setattr(runner_module, "_sleep", delays.append, raising=False)

    result = harvest_airport(
        _airport(),
        HarvestPaths(root=tmp_path, code="KAAA"),
        HarvestPlan(
            target_per_runway=1,
            start=stop,
            chunk_hours=6.0,
            max_lookback_days=30.0,
            dry_give_up_days=0.25,
        ),
        fetch=unstable_fetch,
        log=messages.append,
    )

    assert len(calls) == 2
    assert calls[0]["start"] == calls[1]["start"]
    assert calls[0]["stop"] == calls[1]["stop"]
    assert result.chunks_fetched == 1
    assert delays == [5.0]
    assert any("retrying same chunk" in message for message in messages)


def test_persistent_network_failure_keeps_checkpoint_at_current_chunk(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop = datetime(2026, 7, 22, tzinfo=timezone.utc)
    paths = HarvestPaths(root=tmp_path, code="KAAA")
    calls = 0
    delays: list[float] = []
    messages: list[str] = []

    def offline_fetch(**_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(runner_module, "_sleep", delays.append)

    with pytest.raises(socket.gaierror, match="Name or service not known"):
        harvest_airport(
            _airport(),
            paths,
            HarvestPlan(target_per_runway=1, start=stop),
            fetch=offline_fetch,
            log=messages.append,
        )

    checkpoint = json.loads(paths.checkpoint_state.read_text(encoding="utf-8"))
    assert calls == 7
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
    assert checkpoint["chunks_fetched"] == 0
    assert checkpoint["cursor_utc"] == stop.isoformat()
    assert any("checkpoint retained" in message for message in messages)
