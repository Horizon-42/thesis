"""UTC timestamps as the harvest writes them — each format defined once. `landing_time_utc` is part of `flight_key`
(`id_runway_icao24_landingTime`), so a second copy of its format is an identity risk, not a style point."""

from __future__ import annotations

from datetime import datetime, timezone


def iso_utc(time_s: float) -> str:
    """Whole seconds, ``2026-09-10T19:19:32Z``: landing times (part of `flight_key`) and when a file was written."""
    return datetime.fromtimestamp(time_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_utc_ms(time_s: float) -> str:
    """Milliseconds, ``2026-09-10T19:19:32.125Z``: a track's first reception."""
    return datetime.fromtimestamp(time_s, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_iso_utc() -> str:
    """This moment, as `iso_utc`."""
    return iso_utc(datetime.now(tz=timezone.utc).timestamp())


def parse_iso_utc_s(text: str) -> float:
    """Epoch seconds of either stamp above (``…Z``), millisecond precision kept."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
