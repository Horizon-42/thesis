"""OpenSky history database access through the `traffic` package.

This module is the preferred training-data source when OpenSky database access
is available. It avoids REST `/tracks/all` and returns state-vector rows that
can include both barometric and geometric altitude.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .dataset_store import partition_path, write_jsonl_records
except ImportError:  # pragma: no cover - supports direct script execution.
    from dataset_store import partition_path, write_jsonl_records


HISTORY_COLUMNS = (
    "time",
    "icao24",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "callsign",
    "onground",
    "baroaltitude",
    "geoaltitude",
    "estdepartureairport",
    "estarrivalairport",
)


def require_traffic_opensky() -> Any:
    """Import and return traffic.data.opensky with a focused error message."""
    try:
        from traffic.data import opensky  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The `traffic` package is required for OpenSky history DB mode. "
            "Install and configure traffic/OpenSky DB access before using "
            "--training-source history-db."
        ) from e
    return opensky


def _traffic_result_to_dataframe(result: Any) -> pd.DataFrame:
    """Normalize traffic's return value to a pandas DataFrame."""
    if result is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    if isinstance(result, pd.DataFrame):
        return result.copy()
    data = getattr(result, "data", None)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise RuntimeError(f"Unsupported traffic history result type: {type(result)!r}")


def fetch_history_dataframe(
    *,
    start: datetime,
    stop: datetime,
    airport: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    selected_columns: tuple[str, ...] = HISTORY_COLUMNS,
    cached: bool = True,
) -> pd.DataFrame:
    """Fetch OpenSky history rows through traffic.data.opensky.history."""
    opensky = require_traffic_opensky()
    kwargs: dict[str, Any] = {
        "start": start,
        "stop": stop,
        "selected_columns": selected_columns,
        "cached": cached,
    }
    if airport:
        kwargs["airport"] = airport.upper()
    if bounds:
        kwargs["bounds"] = bounds
    return _traffic_result_to_dataframe(opensky.history(**kwargs))


def normalize_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a predictable, raw-value-preserving history dataframe view."""
    out = df.copy()
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True)
        out = out.sort_values(["icao24", "time"], kind="stable")
    return out.reset_index(drop=True)


def write_history_rows(
    output_root: Path,
    *,
    airport: str,
    df: pd.DataFrame,
    fetched_at: datetime,
) -> Path:
    """Write raw history DB rows as JSONL in a fetch-time partition."""
    partition = partition_path(
        output_root,
        "history_rows",
        airport=airport,
        timestamp=fetched_at.astimezone(timezone.utc),
    )
    path = partition / "rows.jsonl"
    records = _json_records(normalize_history_dataframe(df))
    write_jsonl_records(path, records)
    return path


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records without changing column names."""
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")
