"""OpenSky history database access through the `traffic` package.

This is the package's single data source. ``traffic.data.opensky.history`` returns
state-vector rows that include both barometric and geometric altitude, so no
secondary altitude join is needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


# State-vector columns requested from the history database. ``geoaltitude`` is
# mandatory: every downloaded trajectory must carry geometric altitude.
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

# When querying by airport, the joined flights table supplies estimated
# departure/arrival airports and must be prefixed as ``FlightsData4.*``.
AIRPORT_HISTORY_COLUMNS = (
    *STATE_VECTOR_COLUMNS,
    "FlightsData4.estdepartureairport",
    "FlightsData4.estarrivalairport",
)


def require_traffic_opensky() -> Any:
    """Import and return ``traffic.data.opensky`` with a focused error message."""
    try:
        from traffic.data import opensky  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The `traffic` package is required for OpenSky history-DB access. "
            "Install traffic and configure OpenSky database access first: "
            "https://traffic-viz.github.io/data_sources/opensky_db.html"
        ) from e
    return opensky


def fetch_history_dataframe(
    *,
    start: datetime,
    stop: datetime,
    airport: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    selected_columns: tuple[str, ...] = AIRPORT_HISTORY_COLUMNS,
    cached: bool = True,
) -> pd.DataFrame:
    """Fetch OpenSky history rows. ``bounds`` order is west, south, east, north."""
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
    try:
        return _as_dataframe(opensky.history(**kwargs))
    except Exception as e:  # noqa: BLE001 - turn DB driver errors into actionable guidance.
        if "PERMISSION_DENIED" in str(e) or "Access Denied" in str(e):
            raise RuntimeError(
                "OpenSky history query was denied (PERMISSION_DENIED). The account is "
                "authenticated but not authorized for the historical Trino database. "
                "Request historical-data access from OpenSky, then configure credentials "
                "in ~/.config/pyopensky/settings.conf or via the OPENSKY_USERNAME / "
                "OPENSKY_PASSWORD environment variables (note: OPENSKY_USER is not read)."
            ) from e
        raise


def _as_dataframe(result: Any) -> pd.DataFrame:
    """Normalize traffic's return value (Traffic | DataFrame | None) to a frame."""
    if result is None:
        return pd.DataFrame(columns=[_bare(c) for c in STATE_VECTOR_COLUMNS])
    if isinstance(result, pd.DataFrame):
        return result.copy()
    data = getattr(result, "data", None)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise RuntimeError(f"Unsupported traffic history result type: {type(result)!r}")


def history_rows_as_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize history rows to JSON-safe records, preserving column names."""
    return [
        {str(k): _json_safe(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _bare(column: str) -> str:
    return column.rsplit(".", 1)[-1] if "." in column else column


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert(timezone.utc) if value.tzinfo else value
        return ts.isoformat().replace("+00:00", "Z")
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
