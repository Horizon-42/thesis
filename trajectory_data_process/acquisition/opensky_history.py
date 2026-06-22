"""OpenSky history database access through the `traffic` package.

This is the package's single data source. ``traffic.data.opensky.history`` returns
state-vector rows that include both barometric and geometric altitude, so no
secondary altitude join is needed.
"""

from __future__ import annotations

import signal
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

# ``traffic`` returns altitudes in feet (its aviation-unit convention), whereas the
# rest of this package works in metres. Convert these columns on the way out.
FEET_TO_M = 0.3048
ALTITUDE_COLUMNS = ("altitude", "geoaltitude", "baroaltitude")


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


# ── Cancel in-flight queries on interrupt ─────────────────────────────────────
# pyopensky runs each history query through a Trino cursor but never exposes it.
# If the process is killed mid-query the query lingers on the server and keeps
# occupying the account's small query quota (2 running + 2 queued). We track the
# active Trino cursors via a class-level hook so a Ctrl-C can cancel them
# explicitly (Trino cursor.cancel() sends a DELETE that stops the query at once).
_active_cursors: set[Any] = set()
_cursor_tracking_installed = False


def _ensure_cursor_tracking() -> None:
    global _cursor_tracking_installed
    if _cursor_tracking_installed:
        return
    try:
        import trino.dbapi as trino_dbapi  # type: ignore
    except ModuleNotFoundError:
        return
    original_execute = trino_dbapi.Cursor.execute

    def tracked_execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        _active_cursors.add(self)
        return original_execute(self, *args, **kwargs)

    trino_dbapi.Cursor.execute = tracked_execute  # type: ignore[method-assign]
    _cursor_tracking_installed = True


def cancel_active_queries() -> int:
    """Cancel any in-flight Trino queries; returns how many were cancelled."""
    cancelled = 0
    for cursor in list(_active_cursors):
        try:
            cursor.cancel()
            cancelled += 1
        except Exception:  # noqa: BLE001 - best effort; a finished cursor is fine.
            pass
    _active_cursors.clear()
    return cancelled


def install_query_cancel_on_interrupt() -> None:
    """Install a SIGINT/SIGTERM handler that cancels the running query first.

    Call once from a CLI ``main()`` so interrupting a long download does not leave
    queries holding the account's Trino quota. Must run on the main thread.
    """
    _ensure_cursor_tracking()

    def handler(signum: int, _frame: Any) -> None:
        n = cancel_active_queries()
        if n:
            print(f"\n[opensky] cancelled {n} running query(ies) on interrupt", flush=True)
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # e.g. not on the main thread
            pass


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
        return _altitudes_to_metres(_as_dataframe(opensky.history(**kwargs)))
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
    finally:
        # The query is done (or raised); drop finished cursors so a later interrupt
        # between queries does not try to cancel them.
        _active_cursors.clear()


def _altitudes_to_metres(df: pd.DataFrame) -> pd.DataFrame:
    """Convert traffic's feet-based altitude columns to metres in place."""
    for column in ALTITUDE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce") * FEET_TO_M
    return df


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
