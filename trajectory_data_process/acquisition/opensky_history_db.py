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

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory_data_process.datasets.dataset_store import partition_path, write_jsonl_records


STATE_VECTOR_COLUMNS = (
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
)

AIRPORT_HISTORY_COLUMNS = (
    *STATE_VECTOR_COLUMNS,
    "FlightsData4.estdepartureairport",
    "FlightsData4.estarrivalairport",
)

HISTORY_COLUMNS = (
    *STATE_VECTOR_COLUMNS,
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
    selected_columns: tuple[str, ...] = AIRPORT_HISTORY_COLUMNS,
    cached: bool = True,
) -> pd.DataFrame:
    """Fetch OpenSky history rows through traffic.data.opensky.history.

    ``bounds`` follows traffic/OpenSky DB order: west, south, east, north.
    Columns from the joined flights table must be prefixed as ``FlightsData4.*``
    in the request, while traffic usually returns the unqualified field name.
    """
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
    """Return a predictable derived view for analysis.

    This is deliberately not used by ``write_history_rows``. The stored history
    rows should preserve the dataframe returned by traffic as closely as JSON
    serialization allows; sorting, aliasing, and duplicate removal belong only
    to downstream analysis.
    """
    out = df.copy()
    out = out.rename(columns={column: _canonical_history_column(column) for column in out.columns})
    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True)
    if "icao24" in out.columns:
        out["icao24"] = out["icao24"].astype(str).str.lower().str.strip()
    sort_columns = [column for column in ("icao24", "time") if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns, kind="stable")
    dedupe_columns = [column for column in ("icao24", "time", "lat", "lon") if column in out.columns]
    if len(dedupe_columns) >= 2:
        out = out.drop_duplicates(subset=dedupe_columns, keep="first")
    return out.reset_index(drop=True)


def _canonical_history_column(column: Any) -> str:
    """Map traffic/pyopensky column labels to the internal lowercase names."""
    text = str(column).strip().strip('"')
    lowered = text.lower()
    if "." in lowered:
        lowered = lowered.rsplit(".", 1)[-1]
    return lowered


def write_history_rows(
    output_root: Path,
    *,
    airport: str,
    df: pd.DataFrame,
    fetched_at: datetime,
    query_name: str = "rows",
) -> Path:
    """Write history DB rows as JSONL in a fetch-time partition.

    This function does not sort, filter, rename, or deduplicate rows. It only
    performs JSON-safe serialization of values returned by traffic.
    """
    partition = partition_path(
        output_root,
        "history_rows",
        airport=airport,
        timestamp=fetched_at.astimezone(timezone.utc),
    )
    safe_query_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in query_name).strip("_")
    path = partition / f"{safe_query_name or 'rows'}.jsonl"
    records = _json_records(df)
    write_jsonl_records(path, records)
    return path


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records without changing column names."""
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return records


def _json_safe_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-serializable values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is not None:
            ts = ts.tz_convert(timezone.utc)
            return ts.isoformat().replace("+00:00", "Z")
        return ts.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
