"""OpenSky history database access, through `pyopensky` directly.

This is the package's single data source. ``Trino.history`` returns state-vector rows
that include both barometric and geometric altitude, so no secondary altitude join is
needed.

WHY NOT `traffic`
-----------------
This module used ``traffic.data.opensky``, which is a thin wrapper over exactly this
pyopensky client. It is now unusable: ``traffic`` monkey-patches a pandas INTERNAL class
at import time (``pandas.core.internals.blocks.DatetimeTZBlock``, to fix interpolation of
timezone-aware columns), and pandas 3.0 removed that class. The failure is an ImportError
during ``from traffic.data import opensky`` -- before any credential is read, so it looks
like an auth problem and is not one. traffic 2.13 is the latest release and no version
supports pandas 3.

This is not environment drift: ``.env-backup/aeroviz-pip-freeze.txt`` records
``pandas==3.0.3`` with ``traffic==2.13``, i.e. the pairing has never worked here. Going
direct removes the dependency rather than pinning the whole thesis env (casadi, torch and
the geospatial stack all live in it) to an old pandas.

UNITS -- THE TRAP IN THIS SWAP
------------------------------
``traffic`` converted OpenSky's metres to FEET (its aviation convention,
``_format_history``: ``df.altitude / 0.3048``), and this module converted them back.
pyopensky does NO conversion, so its altitudes are already metres and the return contract
is unchanged -- but the old back-conversion had to go with the wrapper. Applying it to
pyopensky output would divide every altitude by 3.28, which does not crash: it silently
turns a 3 deg approach into a 9.9 deg one.

Column names are unaffected: ``trajectory.canonical_column`` already maps BOTH traffic's
names and pyopensky's raw ones onto the same canonical set.
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

# The harvest also needs the source position clock. ``time`` is merely the state-row
# clock and may advance while OpenSky repeats an older position. Keeping these separate
# prevents a stale final position from being misread as an impossible one-second jump.
HARVEST_STATE_VECTOR_COLUMNS = (
    *STATE_VECTOR_COLUMNS,
    "lastposupdate",
    "lastcontact",
)

# When querying by airport, the joined flights table supplies estimated
# departure/arrival airports and must be prefixed as ``FlightsData4.*``. These columns
# are ONLY available with ``airport=`` -- see the check in fetch_history_dataframe.
_JOINED_TABLE = "FlightsData4."
AIRPORT_HISTORY_COLUMNS = (
    *STATE_VECTOR_COLUMNS,
    "FlightsData4.estdepartureairport",
    "FlightsData4.estarrivalairport",
)


def require_opensky_history() -> Any:
    """Build the pyopensky Trino client, with a focused error message."""
    try:
        from pyopensky.trino import Trino  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The `pyopensky` package is required for OpenSky history-DB access. "
            "Install pyopensky and configure historical access first: "
            "https://open-aviation.github.io/pyopensky/"
        ) from e
    return Trino()


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
    """Fetch OpenSky history rows in METRES. ``bounds`` is west, south, east, north.

    Altitudes come back as OpenSky stores them -- metres -- and are NOT converted here.
    See the module docstring: the old feet round-trip existed only because ``traffic``
    imposed nautical units, and re-applying it would scale every altitude by 3.28.
    """
    joined = [c for c in selected_columns if c.startswith(_JOINED_TABLE)]
    if joined and not airport:
        # The estimated departure/arrival airports live in FlightsData4, which is only
        # joined in when the query is BY airport. Asking for them off a bbox query makes
        # pyopensky fail deep in its statement builder with an unrelated-looking
        # NameError, so it is caught here where the cause is obvious.
        raise ValueError(
            f"{', '.join(joined)} require airport=; a bounds-only query does not join "
            f"{_JOINED_TABLE.rstrip('.')}. Pass STATE_VECTOR_COLUMNS for a bbox query."
        )
    opensky = require_opensky_history()
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
        return _strip_table_prefixes(_as_dataframe(opensky.history(**kwargs)))
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


def _strip_table_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``FlightsData4.`` qualifier joined columns come back carrying.

    ``estdepartureairport`` / ``estarrivalairport`` are requested table-qualified (the
    Trino join requires it) and are returned that way. Consumers that go through
    ``trajectory.normalize_history_dataframe`` would have the prefix stripped for them,
    but the harvest reads the frame's records directly, so it is normalised once here --
    at the boundary, rather than in each reader.
    """
    return df.rename(columns={c: _bare(c) for c in df.columns})


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
