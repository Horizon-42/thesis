"""Regression tests for the backward harvest scan's stopping rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        fetch=lambda **_kwargs: None,
        log=lambda _message: None,
    )

    assert result.chunks_fetched == 17
    assert result.scanned_from == stop - timedelta(days=4, hours=6)
    assert result.manifest["provenance"]["given_up"] == ["18"]
