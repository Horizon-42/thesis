#!/usr/bin/env python
"""Add missing OpenSky metadata without modifying harvested trajectories.

The existing ``tracks/`` trees are source artifacts for this operation and are opened
read-only.  Results are written to a disjoint sidecar dataset:

    trajectory_data_process/outputs/adsb-metadata/<ICAO>/
        state_vectors/<original-query-digest>.parquet
        operational_status/<original-query-digest>.parquet
        manifest.json

``state_vectors`` preserves one row per historical state-vector row and contains
``time``, ``icao24``, reported ground ``velocity``, ``lastposupdate`` and
``lastcontact``.  ``operational_status`` preserves the decoded operational-status
messages, including ``geometricverticalaccuracy`` when OpenSky has it.

Safety properties:

* dry-run is the default and performs no network request or write;
* output must be disjoint from both the harvest and OpenSky cache trees;
* existing output files are validated and reused, never replaced;
* each Parquet partition is created atomically with embedded provenance;
* the source manifest and original velocity cache files are hashed before and after;
* pyopensky cache expiration is disabled before pyopensky is imported;
* missing original velocity cache files abort the run instead of redownloading them.

Usage::

    # Read-only preflight for one airport
    conda run -n aeroviz python trajectory_data_process/backfill_adsb_metadata.py \
        --airport KMSY

    # Perform the additive backfill; safe to resume after interruption
    conda run -n aeroviz python trajectory_data_process/backfill_adsb_metadata.py \
        --airport KMSY --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep as _sleep
from typing import Any, Callable, Iterator, Protocol, Sequence
from uuid import uuid4

# pyopensky.config purges expired query caches when imported unless this is set.
# This must happen before any lazy pyopensky import below.
os.environ.setdefault("OPENSKY_CACHE_NO_EXPIRE", "1")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.acquisition.opensky_history import (
    STATE_VECTOR_COLUMNS,
    install_query_cancel_on_interrupt,
)
from trajectory_data_process.geo import bounds_from_radius_km

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARVEST_ROOT = REPO_ROOT / "trajectory_data_process/outputs/harvest"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "trajectory_data_process/outputs/adsb-metadata"
DEFAULT_CONFIG = REPO_ROOT / "trajectory_data_process/config/runway_thresholds.json"

SCHEMA_ID = "aeroviz.adsb-metadata-backfill.v1"
# Match the established harvest retry policy: retry only transient transport
# failures. Six waits span 195 seconds, bridging brief OpenSky/auth outages while
# leaving authentication, query, schema, and data-integrity failures visible.
NETWORK_RETRY_DELAYS_SECONDS = (5.0, 10.0, 20.0, 40.0, 60.0, 60.0)
_TRANSIENT_TRANSPORT_ERROR_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "ConnectionTimeout",
    "NetworkError",
    "PoolTimeout",
    "ProxyError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TransportError",
    "WriteTimeout",
}
_TRANSIENT_MESSAGE_MARKERS = (
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname provided",
    "connection reset by peer",
    "connection refused",
    "connection timed out",
    "connect timeout",
    "read timeout",
    "network is unreachable",
    "no route to host",
    "server disconnected",
    "remote end closed connection",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)
_TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
STATE_FRESHNESS_COLUMNS = (
    *STATE_VECTOR_COLUMNS,
    "lastposupdate",
    "lastcontact",
)
STATE_OUTPUT_COLUMNS = (
    "time",
    "icao24",
    "velocity",
    "lastposupdate",
    "lastcontact",
)
OPERATIONAL_STATUS_EXTRA_COLUMNS = (
    "sensors",
    "maxtime",
    "msgcount",
    "subtypecode",
    "unknowncapcode",
    "unknownopcode",
    "hasoperationaltcas",
    "has1090esin",
    "supportsairreferencedvelocity",
    "haslowtxpower",
    "supportstargetstatereport",
    "supportstargetchangereport",
    "hasuatin",
    "nacv",
    "nicsupplementc",
    "hastcasresolutionadvisory",
    "hasactiveidentswitch",
    "usessingleantenna",
    "systemdesignassurance",
    "gpsantennaoffset",
    "airplanelength",
    "airplanewidth",
    "version",
    "nicsupplementa",
    "positionnac",
    "geometricverticalaccuracy",
    "sourceintegritylevel",
    "barometricaltitudeintegritycode",
    "trackheadinginfo",
    "horizontalreferencedirection",
    "hour",
)
OPERATIONAL_STATUS_OUTPUT_COLUMNS = (
    "sensors",
    "rawmsg",
    "mintime",
    "maxtime",
    "msgcount",
    "icao24",
    *OPERATIONAL_STATUS_EXTRA_COLUMNS[3:-1],
    "hour",
)

GIB = 1024**3


@dataclass(frozen=True)
class QueryChunk:
    airport: str
    start_utc: str
    stop_utc: str
    bounds: tuple[float, float, float, float]
    source_query_digest: str
    source_cache_path: Path

    @property
    def start(self) -> datetime:
        return _parse_utc(self.start_utc)

    @property
    def stop(self) -> datetime:
        return _parse_utc(self.stop_utc)


@dataclass(frozen=True)
class AirportPlan:
    airport: str
    harvest_root: Path
    source_manifest: Path
    source_manifest_sha256: str
    chunks: tuple[QueryChunk, ...]


@dataclass(frozen=True)
class AircraftWindow:
    icao24: str
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp


@dataclass(frozen=True)
class ExecutionResult:
    airport: str
    mode: str
    chunks: int
    source_cache_bytes: int
    required_free_bytes: int


class MetadataProvider(Protocol):
    def fetch_state_freshness(
        self, chunk: QueryChunk, source_stop: pd.Timestamp | None
    ) -> pd.DataFrame: ...

    def fetch_operational_status(
        self, chunk: QueryChunk, windows: tuple[AircraftWindow, ...]
    ) -> pd.DataFrame: ...


class OpenSkyMetadataProvider:
    """Historical OpenSky reader; every request uses additive query caching."""

    def __init__(self) -> None:
        from pyopensky.schema import OperationalStatusData4
        from pyopensky.trino import Trino

        self._client = Trino()
        self._operational_table = OperationalStatusData4

    def fetch_state_freshness(
        self, chunk: QueryChunk, source_stop: pd.Timestamp | None
    ) -> pd.DataFrame:
        if source_stop is None:
            return pd.DataFrame(columns=STATE_FRESHNESS_COLUMNS)
        result = _call_with_literal_queries(
            self._client,
            lambda: self._client.history(
                start=chunk.start,
                stop=source_stop.to_pydatetime(),
                bounds=chunk.bounds,
                selected_columns=STATE_FRESHNESS_COLUMNS,
                cached=True,
            ),
        )
        if result is None:
            return pd.DataFrame(columns=STATE_FRESHNESS_COLUMNS)
        frame = _as_dataframe(result)
        missing = set(STATE_FRESHNESS_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"freshness query lacks columns {sorted(missing)}")
        return frame.loc[:, list(STATE_FRESHNESS_COLUMNS)]

    def fetch_operational_status(
        self, chunk: QueryChunk, windows: tuple[AircraftWindow, ...]
    ) -> pd.DataFrame:
        if not windows:
            return _empty_operational_status()
        from sqlalchemy import and_, or_, select

        table = self._operational_table
        aircraft_conditions = tuple(
            and_(
                table.icao24 == window.icao24,
                table.mintime >= window.first_seen,
                table.mintime <= window.last_seen,
            )
            for window in windows
        )
        start = pd.Timestamp(chunk.start)
        stop = pd.Timestamp(chunk.stop)
        statement = select(table).where(
            table.rawmsg.is_not(None),
            table.mintime >= start,
            table.mintime <= stop,
            table.hour >= start.floor("1h"),
            table.hour < stop.ceil("1h"),
            or_(*aircraft_conditions),
        )
        # pyOpenSky forces legacy prepared statements.  With hundreds of bound
        # values, OpenSky can spend minutes in PREPARE before the SELECT starts.
        # SQLAlchemy quotes these typed literals; submitting the compiled SQL as a
        # string lets the driver execute one statement with no PREPARE/DEALLOCATE.
        query_sql = _literal_sql(statement)
        result = self._client.query(query_sql, cached=True)
        if result is None:
            return _empty_operational_status()
        frame = _as_dataframe(result)
        missing = set(OPERATIONAL_STATUS_OUTPUT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"operational-status query lacks columns {sorted(missing)}"
            )
        frame["hour"] = _utc_timestamps(frame["hour"]).astype(
            pd.ArrowDtype(pa.timestamp("s", tz="UTC"))
        )
        for column in ("rawmsg", "icao24"):
            frame[column] = frame[column].astype("string")
        return frame.loc[:, list(OPERATIONAL_STATUS_OUTPUT_COLUMNS)]


def validate_disjoint_roots(
    *, harvest_root: Path, cache_root: Path, output_root: Path
) -> None:
    """Refuse any layout where output could shadow or contain a protected input."""
    output = output_root.resolve()
    for protected in (harvest_root.resolve(), cache_root.resolve()):
        if _contains(output, protected) or _contains(protected, output):
            raise ValueError(
                f"output {output} overlaps protected input {protected}; choose a "
                "disjoint --output directory"
            )


def merge_state_metadata(
    source_frame: pd.DataFrame, freshness_frame: pd.DataFrame
) -> pd.DataFrame:
    """Attach freshness while preserving the protected source row multiset.

    OpenSky can contain more than one state-vector row for the same aircraft and
    second.  The full state row plus its occurrence within identical rows is therefore
    used only as a validation/join identity; output remains in source-cache row order.
    """
    source_identity = _normalized_state_identity(source_frame, label="cached state")
    fresh_identity = _normalized_state_identity(
        freshness_frame, label="freshness query"
    )
    fresh_missing = {"lastposupdate", "lastcontact"} - set(freshness_frame.columns)
    if fresh_missing:
        raise ValueError(
            f"freshness query lacks required columns {sorted(fresh_missing)}"
        )

    identity = list(STATE_VECTOR_COLUMNS)
    occurrence = "_identity_occurrence"
    source_rows = source_identity.copy()
    source_rows["_source_order"] = range(len(source_rows))
    source_rows[occurrence] = source_rows.groupby(
        identity, dropna=False, sort=False
    ).cumcount()

    fresh_rows = fresh_identity.copy()
    for column in ("lastposupdate", "lastcontact"):
        fresh_rows[column] = pd.to_numeric(
            freshness_frame[column], errors="coerce"
        ).reset_index(drop=True)
    fresh_rows = fresh_rows.sort_values(
        [*identity, "lastposupdate", "lastcontact"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    fresh_rows[occurrence] = fresh_rows.groupby(
        identity, dropna=False, sort=False
    ).cumcount()

    keys = [*identity, occurrence]
    joined = source_rows.merge(
        fresh_rows.loc[:, [*keys, "lastposupdate", "lastcontact"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    counts = joined["_merge"].value_counts()
    source_only = int(counts.get("left_only", 0))
    query_only = int(counts.get("right_only", 0))
    if source_only or query_only:
        raise ValueError(
            "state-vector row identities differ between the preserved source cache "
            f"and freshness query ({source_only} cached-only, {query_only} new-only); "
            "refusing to combine different historical snapshots"
        )
    joined = joined.sort_values("_source_order", kind="stable").reset_index(drop=True)

    output = source_frame.loc[:, ["time", "icao24", "velocity"]].copy()
    output["time"] = _utc_timestamps(output["time"]).astype("datetime64[ms, UTC]")
    output["icao24"] = (
        output["icao24"].astype("string").str.lower().str.strip()
    )
    output = output.reset_index(drop=True)
    output["lastposupdate"] = joined["lastposupdate"].reset_index(drop=True)
    output["lastcontact"] = joined["lastcontact"].reset_index(drop=True)
    return output.loc[:, list(STATE_OUTPUT_COLUMNS)]


def _normalized_state_identity(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = set(STATE_VECTOR_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks required columns {sorted(missing)}")
    result = frame.loc[:, list(STATE_VECTOR_COLUMNS)].copy().reset_index(drop=True)
    result["time"] = _utc_timestamps(result["time"]).astype(
        "datetime64[ms, UTC]"
    )
    result["icao24"] = (
        result["icao24"].astype("string").str.lower().str.strip()
    )
    result["callsign"] = result["callsign"].astype("string").str.strip()
    result["onground"] = result["onground"].astype("boolean")
    for column in (
        "lat",
        "lon",
        "velocity",
        "heading",
        "vertrate",
        "baroaltitude",
        "geoaltitude",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "Float64"
        )
    return result


def source_snapshot_stop(
    velocity_frame: pd.DataFrame, chunk: QueryChunk
) -> pd.Timestamp | None:
    """Return the last state time actually present in the protected source cache.

    A harvest window can end at the download time while OpenSky's historical store is
    still several hours behind.  Repeating the planned window later would then add rows
    which were not part of the harvested snapshot.  The cache remains the authoritative
    row roster, so freshness queries must not extend beyond its observed tail.
    """
    if "time" not in velocity_frame.columns:
        raise ValueError("cached velocity lacks required column 'time'")
    times = _utc_timestamps(velocity_frame["time"]).dropna()
    if times.empty:
        return None
    actual = pd.Timestamp(times.max())
    planned = pd.Timestamp(chunk.stop)
    if actual > planned:
        raise ValueError(
            f"cached velocity contains time {actual.isoformat()} beyond planned "
            f"chunk stop {planned.isoformat()}"
        )
    return actual


def state_aircraft_windows(frame: pd.DataFrame) -> tuple[AircraftWindow, ...]:
    """Return deterministic in-bounds time windows for the partition's aircraft."""
    required = {"time", "icao24"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"state partition lacks required columns {sorted(missing)}")
    state = frame.loc[:, ["time", "icao24"]].copy()
    state["time"] = pd.to_datetime(state["time"], utc=True)
    state["icao24"] = state["icao24"].astype("string").str.lower().str.strip()
    state = state.dropna(subset=["time", "icao24"])
    grouped = (
        state.groupby("icao24", as_index=False, sort=True)["time"]
        .agg(first_seen="min", last_seen="max")
        .reset_index(drop=True)
    )
    return tuple(
        AircraftWindow(
            icao24=str(row.icao24),
            first_seen=pd.Timestamp(row.first_seen),
            last_seen=pd.Timestamp(row.last_seen),
        )
        for row in grouped.itertuples(index=False)
    )


def filter_operational_status_to_state_windows(
    status_frame: pd.DataFrame, state_frame: pd.DataFrame
) -> pd.DataFrame:
    """Keep status messages while each aircraft was inside the source query bounds.

    The former bounds-based pyOpenSky query derived the same per-aircraft first/last
    state-vector times with a server-side aggregate and join.  Applying those windows
    locally preserves that result boundary while letting Trino query the operational
    status table directly by ICAO24.
    """
    state_required = {"time", "icao24"}
    status_required = {"mintime", "icao24"}
    missing_state = state_required - set(state_frame.columns)
    missing_status = status_required - set(status_frame.columns)
    if missing_state:
        raise ValueError(
            f"state partition lacks required columns {sorted(missing_state)}"
        )
    if missing_status:
        raise ValueError(
            f"operational-status query lacks required columns {sorted(missing_status)}"
        )
    if status_frame.empty:
        return status_frame.copy()

    state = state_frame.loc[:, ["time", "icao24"]].copy()
    state["time"] = pd.to_datetime(state["time"], utc=True)
    state["icao24"] = state["icao24"].astype("string").str.lower().str.strip()
    state = state.dropna(subset=["time", "icao24"])
    windows = (
        state.groupby("icao24", as_index=False, sort=True)["time"]
        .agg(first_seen="min", last_seen="max")
        .reset_index(drop=True)
    )

    status_icao24 = (
        status_frame["icao24"].astype("string").str.lower().str.strip()
    )
    status_time = _utc_timestamps(status_frame["mintime"])
    first_seen = status_icao24.map(windows.set_index("icao24")["first_seen"])
    last_seen = status_icao24.map(windows.set_index("icao24")["last_seen"])
    keep = status_time.between(first_seen, last_seen)
    return status_frame.loc[keep.fillna(False)].copy().reset_index(drop=True)


def _utc_timestamps(values: pd.Series) -> pd.Series:
    """Normalize OpenSky timestamps, whose raw tables use Unix seconds."""
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_datetime(values, unit="s", utc=True)
    return pd.to_datetime(values, utc=True)


def write_parquet_exclusive(
    frame: pd.DataFrame,
    target: Path,
    *,
    metadata: dict[str, Any],
) -> None:
    """Atomically create one Parquet file; never replace an existing pathname."""
    if target.exists():
        raise FileExistsError(f"refusing to replace existing output {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"aeroviz_adsb_backfill"] = encoded
    try:
        pq.write_table(table.replace_schema_metadata(schema_metadata), temporary)
        # A hard link is an atomic create-if-absent operation.  Unlike os.replace(), it
        # cannot overwrite a file created by another process between our checks.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def execute_airport(
    plan: AirportPlan,
    *,
    output_root: Path,
    cache_root: Path,
    provider: MetadataProvider,
    apply: bool,
    available_bytes: int | None = None,
) -> ExecutionResult:
    """Preflight or execute one airport's additive, resumable backfill."""
    validate_disjoint_roots(
        harvest_root=plan.harvest_root,
        cache_root=cache_root,
        output_root=output_root,
    )
    _validate_plan_inputs(plan, cache_root=cache_root)
    source_bytes = sum(
        chunk.source_cache_path.stat().st_size
        for chunk in _unique_chunks(plan.chunks)
    )
    required = _required_free_bytes(source_bytes)
    airport_output = output_root / plan.airport
    manifest_path = airport_output / "manifest.json"

    if manifest_path.exists():
        _validate_completed_manifest(manifest_path, plan)
        return ExecutionResult(
            plan.airport, "already-complete", len(plan.chunks), source_bytes, required
        )

    if not apply:
        return ExecutionResult(
            plan.airport, "dry-run", len(plan.chunks), source_bytes, required
        )

    free = available_bytes
    if free is None:
        free = shutil.disk_usage(_nearest_existing(output_root)).free
    if free < required:
        raise RuntimeError(
            f"{plan.airport}: insufficient free space: {_human_bytes(free)} available, "
            f"{_human_bytes(required)} required by the conservative preflight"
        )

    original_hashes = {
        chunk.source_cache_path: _sha256(chunk.source_cache_path)
        for chunk in _unique_chunks(plan.chunks)
    }
    state_partitions: list[dict[str, Any]] = []
    status_partitions: list[dict[str, Any]] = []

    for number, chunk in enumerate(plan.chunks, start=1):
        cache_sha = original_hashes[chunk.source_cache_path]
        common = _partition_metadata(chunk, cache_sha=cache_sha)
        state_path = (
            airport_output
            / "state_vectors"
            / f"{chunk.source_query_digest}.parquet"
        )
        status_path = (
            airport_output
            / "operational_status"
            / f"{chunk.source_query_digest}.parquet"
        )

        state_meta = {**common, "table": "state_vectors"}
        if state_path.exists():
            _validate_existing_partition(state_path, state_meta)
            state = pd.read_parquet(state_path)
        else:
            source_state = pd.read_parquet(
                chunk.source_cache_path,
                columns=list(STATE_VECTOR_COLUMNS),
            )
            freshness = _call_opensky_with_retries(
                lambda: provider.fetch_state_freshness(
                    chunk, source_snapshot_stop(source_state, chunk)
                ),
                airport=plan.airport,
                operation="state-vector freshness",
            )
            state = merge_state_metadata(source_state, freshness)
            write_parquet_exclusive(state, state_path, metadata=state_meta)

        status_meta = {**common, "table": "operational_status"}
        if status_path.exists():
            _validate_existing_partition(status_path, status_meta)
            status = pd.read_parquet(status_path)
        else:
            status = _call_opensky_with_retries(
                lambda: provider.fetch_operational_status(
                    chunk, state_aircraft_windows(state)
                ),
                airport=plan.airport,
                operation="operational status",
            )
            if status is None:
                status = _empty_operational_status()
            status = filter_operational_status_to_state_windows(status, state)
            write_parquet_exclusive(status, status_path, metadata=status_meta)

        state_partitions.append(_partition_summary(state_path, state))
        status_partitions.append(_partition_summary(status_path, status))
        print(
            f"[adsb-metadata] {plan.airport}: {number}/{len(plan.chunks)} "
            f"state={len(state):,}, operational_status={len(status):,}",
            flush=True,
        )

    _verify_unchanged(plan.source_manifest, plan.source_manifest_sha256)
    for path, digest in original_hashes.items():
        _verify_unchanged(path, digest)

    manifest = _build_output_manifest(plan, state_partitions, status_partitions)
    _write_json_exclusive(manifest_path, manifest)
    return ExecutionResult(
        plan.airport, "applied", len(plan.chunks), source_bytes, required
    )


def build_airport_plan(
    airport: str,
    *,
    harvest_root: Path,
    cache_root: Path,
    config_file: Path,
    chunk_hours: float = 6.0,
) -> AirportPlan:
    """Reconstruct original bbox queries and bind them to existing cache files."""
    code = airport.upper()
    source_manifest = harvest_root / code / "tracks" / "manifest.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if payload.get("airport") != code:
        raise ValueError(
            f"{source_manifest}: airport is {payload.get('airport')!r}, expected {code!r}"
        )
    config = json.loads(config_file.read_text(encoding="utf-8"))
    try:
        airport_config = config["airports"][code]
        latitude = float(airport_config["lat"])
        longitude = float(airport_config["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{config_file}: missing valid coordinates for {code}") from exc

    leaves = _download_provenance_leaves(payload.get("provenance"))
    chunks_by_digest: dict[str, QueryChunk] = {}
    for provenance in leaves:
        radius_km = float(provenance["radius_km"])
        bounds = bounds_from_radius_km(latitude, longitude, radius_km)
        for start, stop in _source_windows(provenance, chunk_hours=chunk_hours):
            digest = _history_query_digest(start, stop, bounds)
            cache_path = _locate_cache_file(cache_root, digest)
            chunk = QueryChunk(
                airport=code,
                start_utc=_iso_utc(start),
                stop_utc=_iso_utc(stop),
                bounds=bounds,
                source_query_digest=digest,
                source_cache_path=cache_path,
            )
            previous = chunks_by_digest.setdefault(digest, chunk)
            if previous != chunk:
                raise ValueError(f"query digest collision while planning {code}: {digest}")

    chunks = tuple(
        sorted(
            chunks_by_digest.values(),
            key=lambda item: (item.start_utc, item.stop_utc, item.source_query_digest),
        )
    )
    if not chunks:
        raise ValueError(f"{source_manifest}: no original download windows found")
    return AirportPlan(
        airport=code,
        harvest_root=harvest_root,
        source_manifest=source_manifest,
        source_manifest_sha256=_sha256(source_manifest),
        chunks=chunks,
    )


def _download_provenance_leaves(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("harvest manifest has no usable provenance object")
    required = {"scanned_from_utc", "scanned_to_utc", "chunks_fetched", "radius_km"}
    if required <= value.keys():
        return [value]
    merge = value.get("merge")
    sources = merge.get("sources") if isinstance(merge, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            "harvest provenance contains neither a download scan nor merge sources"
        )
    leaves: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("invalid merge source provenance")
        leaves.extend(_download_provenance_leaves(source.get("provenance")))
    return leaves


def _source_windows(
    provenance: dict[str, Any], *, chunk_hours: float
) -> list[tuple[datetime, datetime]]:
    if chunk_hours <= 0:
        raise ValueError("chunk_hours must be positive")
    floor = _parse_utc(str(provenance["scanned_from_utc"]))
    cursor = _parse_utc(str(provenance["scanned_to_utc"]))
    count = int(provenance["chunks_fetched"])
    if count <= 0 or floor >= cursor:
        raise ValueError("invalid download scan range in harvest provenance")
    result: list[tuple[datetime, datetime]] = []
    step = timedelta(hours=chunk_hours)
    for _ in range(count):
        start = max(cursor - step, floor)
        if start >= cursor:
            raise ValueError("download provenance has more chunks than its time span")
        result.append((start, cursor))
        cursor = start
    if cursor != floor:
        raise ValueError(
            "--chunk-hours does not reconstruct the recorded scan exactly; refusing to "
            "guess cache identities"
        )
    return result


def _history_query_digest(
    start: datetime,
    stop: datetime,
    bounds: tuple[float, float, float, float],
) -> str:
    """Compile pyopensky's query without connecting and return its cache digest."""
    from pyopensky.trino import Trino

    captured: list[str] = []
    client = Trino()

    def capture(query: Any, cached: bool = True, compress: bool = False) -> pd.DataFrame:
        statement = query.compile()
        query_text = f"{statement}\n{statement.params}"
        captured.append(hashlib.md5(query_text.encode("utf8")).hexdigest())
        return pd.DataFrame(columns=STATE_VECTOR_COLUMNS)

    client.query = capture  # type: ignore[method-assign]
    client.history(
        start=start,
        stop=stop,
        bounds=bounds,
        selected_columns=STATE_VECTOR_COLUMNS,
        cached=True,
    )
    if len(captured) != 1:
        raise RuntimeError("could not reconstruct exactly one OpenSky history query")
    return captured[0]


def _locate_cache_file(cache_root: Path, digest: str) -> Path:
    plain = cache_root / f"{digest}.parquet"
    compressed = cache_root / f"{digest}.parquet.gz"
    matches = [path for path in (plain, compressed) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one preserved cache file for query {digest}, found "
            f"{len(matches)}; refusing to redownload original trajectory data"
        )
    return matches[0]


def _validate_plan_inputs(plan: AirportPlan, *, cache_root: Path) -> None:
    _verify_unchanged(plan.source_manifest, plan.source_manifest_sha256)
    seen: set[str] = set()
    for chunk in plan.chunks:
        if chunk.airport != plan.airport:
            raise ValueError("query chunk airport differs from its plan")
        if chunk.source_query_digest in seen:
            raise ValueError(f"duplicate query chunk {chunk.source_query_digest}")
        seen.add(chunk.source_query_digest)
        if not chunk.source_cache_path.is_file():
            raise FileNotFoundError(chunk.source_cache_path)
        try:
            chunk.source_cache_path.resolve().relative_to(cache_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"source cache file is outside --cache-root: {chunk.source_cache_path}"
            ) from exc
        schema = pq.read_schema(chunk.source_cache_path)
        missing = {"time", "icao24", "velocity"} - set(schema.names)
        if missing:
            raise ValueError(
                f"{chunk.source_cache_path}: cached state query lacks {sorted(missing)}"
            )


def _validate_completed_manifest(path: Path, plan: AirportPlan) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing output manifest is unreadable: {path}") from exc
    expected = {
        "schema": SCHEMA_ID,
        "airport": plan.airport,
    }
    actual = {key: payload.get(key) for key in expected}
    source = payload.get("source_manifest")
    if actual != expected or not isinstance(source, dict) or source.get(
        "sha256"
    ) != plan.source_manifest_sha256:
        raise ValueError(
            f"existing output manifest does not match the current source: {path}; "
            "refusing to replace it"
        )
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError(f"existing output manifest has no partition roster: {path}")
    expected_digests = {chunk.source_query_digest for chunk in plan.chunks}
    for table in ("state_vectors", "operational_status"):
        rows = partitions.get(table)
        if not isinstance(rows, list):
            raise ValueError(f"existing manifest has no {table} partition list")
        actual_digests = {row.get("source_query_digest") for row in rows}
        if actual_digests != expected_digests:
            raise ValueError(
                f"existing {table} partition roster differs from the current source"
            )
        for row in rows:
            partition = path.parent / str(row["file"])
            if not partition.is_file() or _sha256(partition) != row.get("sha256"):
                raise ValueError(f"existing output partition failed validation: {partition}")


def _partition_metadata(chunk: QueryChunk, *, cache_sha: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_ID,
        "airport": chunk.airport,
        "start_utc": chunk.start_utc,
        "stop_utc": chunk.stop_utc,
        "bounds": list(chunk.bounds),
        "source_query_digest": chunk.source_query_digest,
        "source_cache_sha256": cache_sha,
    }


def _validate_existing_partition(path: Path, expected: dict[str, Any]) -> None:
    try:
        metadata = pq.read_schema(path).metadata or {}
        actual = json.loads(metadata[b"aeroviz_adsb_backfill"].decode("utf-8"))
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"existing partition lacks valid backfill provenance: {path}; refusing "
            "to replace it"
        ) from exc
    if actual != expected:
        raise ValueError(
            f"existing partition provenance does not match this run: {path}; "
            "refusing to replace it"
        )


def _partition_summary(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "file": str(path.parent.name + "/" + path.name),
        "source_query_digest": path.stem,
        "rows": int(len(frame)),
        "sha256": _sha256(path),
    }
    if path.parent.name == "state_vectors":
        summary.update(
            {
                "_velocity_available": _available_rows(frame, "velocity"),
                "_lastpos_available": _available_rows(frame, "lastposupdate"),
                "_lastcontact_available": _available_rows(frame, "lastcontact"),
                "_nonunique_time_icao_rows": int(
                    frame.duplicated(["time", "icao24"], keep=False).sum()
                ),
            }
        )
    elif path.parent.name == "operational_status":
        summary["_gva_reported"] = _available_rows(
            frame, "geometricverticalaccuracy"
        )
        summary["_gva_available"] = _usable_gva_rows(frame)
    return summary


def _build_output_manifest(
    plan: AirportPlan,
    state_partitions: list[dict[str, Any]],
    status_partitions: list[dict[str, Any]],
) -> dict[str, Any]:
    state_rows = sum(int(item["rows"]) for item in state_partitions)
    status_rows = sum(int(item["rows"]) for item in status_partitions)
    velocity_available = sum(
        int(item.get("_velocity_available", 0)) for item in state_partitions
    )
    lastpos_available = sum(
        int(item.get("_lastpos_available", 0)) for item in state_partitions
    )
    lastcontact_available = sum(
        int(item.get("_lastcontact_available", 0)) for item in state_partitions
    )
    nonunique_time_icao_rows = sum(
        int(item.get("_nonunique_time_icao_rows", 0))
        for item in state_partitions
    )
    gva_available = sum(
        int(item.get("_gva_available", 0)) for item in status_partitions
    )
    gva_reported = sum(
        int(item.get("_gva_reported", 0)) for item in status_partitions
    )
    public_state_partitions = [_public_summary(item) for item in state_partitions]
    public_status_partitions = [_public_summary(item) for item in status_partitions]
    return {
        "schema": SCHEMA_ID,
        "airport": plan.airport,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source_manifest": {
            "path": str(plan.source_manifest.resolve()),
            "sha256": plan.source_manifest_sha256,
        },
        "source_contract": {
            "harvest_is_read_only": True,
            "original_query_cache_files_are_read_only": True,
            "new_opensky_query_cache_files_may_be_created": True,
            "state_vector_key": (
                ["time", "icao24"] if nonunique_time_icao_rows == 0 else None
            ),
            "state_vector_row_alignment": "preserved source-cache row order",
            "state_vector_identity": (
                "full source state-vector row plus occurrence among identical rows"
            ),
            "reported_velocity_unit": "m/s",
            "lastposupdate_unit": "unix_seconds",
            "lastcontact_unit": "unix_seconds",
            "gva_field": "geometricverticalaccuracy",
            "gva_semantics": "broadcast category; not a per-sample measured error",
        },
        "tables": {
            "state_vectors": {
                "rows": state_rows,
                "velocity_available_rows": velocity_available,
                "lastposupdate_available_rows": lastpos_available,
                "lastcontact_available_rows": lastcontact_available,
                "nonunique_time_icao_rows": nonunique_time_icao_rows,
            },
            "operational_status": {
                "rows": status_rows,
                "gva_reported_rows": gva_reported,
                "gva_available_rows": gva_available,
            },
        },
        "partitions": {
            "state_vectors": public_state_partitions,
            "operational_status": public_status_partitions,
        },
    }


def _available_rows(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(frame[column].notna().sum())


def _usable_gva_rows(frame: pd.DataFrame) -> int:
    if "geometricverticalaccuracy" not in frame.columns:
        return 0
    # Code 0 means that geometric vertical accuracy is unknown. Preserve it in the
    # Parquet row, but do not advertise it as an available accuracy category.
    values = pd.to_numeric(frame["geometricverticalaccuracy"], errors="coerce")
    return int(values.gt(0).sum())


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_state_rows(
    frame: pd.DataFrame, *, required: Sequence[str], label: str
) -> pd.DataFrame:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks required columns {sorted(missing)}")
    result = frame.loc[:, list(required)].copy()
    result["time"] = _utc_timestamps(result["time"])
    result["icao24"] = result["icao24"].astype("string").str.lower().str.strip()
    result = result.drop_duplicates()
    if result.duplicated(["time", "icao24"], keep=False).any():
        raise ValueError(f"{label} has conflicting duplicate (time, icao24) rows")
    return result


def _empty_operational_status() -> pd.DataFrame:
    return pd.DataFrame(columns=OPERATIONAL_STATUS_OUTPUT_COLUMNS)


def _as_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result.copy()
    data = getattr(result, "data", None)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    raise TypeError(f"OpenSky returned unsupported type {type(result)!r}")


def _literal_sql(statement: Any) -> str:
    if isinstance(statement, str):
        return statement
    return str(statement.compile(compile_kwargs={"literal_binds": True}))


def _call_with_literal_queries(client: Any, operation: Any) -> Any:
    """Run a pyOpenSky builder while forcing its generated SELECT to plain SQL."""
    original_query = client.query

    def query_without_prepare(
        statement: Any, cached: bool = True, compress: bool = False
    ) -> pd.DataFrame:
        return original_query(
            _literal_sql(statement), cached=cached, compress=compress
        )

    client.query = query_without_prepare
    try:
        return operation()
    finally:
        client.query = original_query


def _call_opensky_with_retries(
    operation_call: Callable[[], Any], *, airport: str, operation: str
) -> Any:
    """Retry one side-effect-free OpenSky read after transient network failures."""
    total_attempts = len(NETWORK_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return operation_call()
        except Exception as error:  # noqa: BLE001 - narrowed by the predicate.
            if not _is_transient_network_error(error):
                raise
            if attempt == total_attempts:
                print(
                    f"[adsb-metadata] {airport}: OpenSky network failure during "
                    f"{operation} persisted after {total_attempts} attempts; "
                    f"completed partitions retained ({_error_summary(error)})",
                    flush=True,
                )
                raise
            delay = NETWORK_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                f"[adsb-metadata] {airport}: temporary OpenSky network failure "
                f"during {operation} ({_error_summary(error)}); retrying same "
                f"partition in {delay:g}s (attempt {attempt + 1}/{total_attempts})",
                flush=True,
            )
            _sleep(delay)
    raise AssertionError("unreachable")


def _is_transient_network_error(error: BaseException) -> bool:
    for item in _exception_chain(error):
        if isinstance(item, (ConnectionError, TimeoutError, socket.gaierror)):
            return True
        error_type = type(item)
        module = error_type.__module__.lower()
        if (
            module.startswith(("httpx", "httpcore", "requests", "urllib3"))
            and error_type.__name__ in _TRANSIENT_TRANSPORT_ERROR_NAMES
        ):
            return True
        response = getattr(item, "response", None)
        status = getattr(response, "status_code", None)
        if status in _TRANSIENT_HTTP_STATUSES:
            return True
        message = str(item).lower()
        if any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS):
            return True
    return False


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        linked = current.__cause__ or current.__context__
        if linked is None:
            candidate = getattr(current, "orig", None)
            linked = candidate if isinstance(candidate, BaseException) else None
        current = linked


def _error_summary(error: BaseException) -> str:
    root = error
    for item in _exception_chain(error):
        root = item
    message = " ".join(str(root).split())
    if len(message) > 200:
        message = f"{message[:197]}..."
    return f"{type(root).__name__}: {message or 'no details'}"


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _unique_chunks(chunks: Sequence[QueryChunk]) -> tuple[QueryChunk, ...]:
    result: dict[Path, QueryChunk] = {}
    for chunk in chunks:
        result.setdefault(chunk.source_cache_path, chunk)
    return tuple(result.values())


def _required_free_bytes(source_cache_bytes: int) -> int:
    # Allows room for the new pyopensky query caches and durable sidecar partitions.
    return 2 * source_cache_bytes + GIB


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_unchanged(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"protected input changed during backfill: {path}; expected "
            f"{expected_sha256}, got {actual}"
        )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def _default_cache_root() -> Path:
    from pyopensky.config import cache_path

    return Path(cache_path)


def prepare_opensky_runtime(*, apply: bool) -> None:
    """Ensure an interrupted CLI cancels its Trino query instead of leaking quota."""
    if apply:
        install_query_cancel_on_interrupt()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add ADS-B freshness, velocity and operational-status sidecars"
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--airport", action="append", help="ICAO code; repeat as needed")
    scope.add_argument("--all-airports", action="store_true")
    parser.add_argument("--harvest-root", type=Path, default=DEFAULT_HARVEST_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-hours", type=float, default=6.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform additive network queries and writes (default is read-only dry-run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_opensky_runtime(apply=args.apply)
    cache_root = args.cache_root or _default_cache_root()
    if args.all_airports:
        airports = sorted(
            path.parent.parent.name
            for path in args.harvest_root.glob("*/tracks/manifest.json")
        )
    else:
        airports = sorted({str(code).upper() for code in args.airport})
    if not airports:
        raise SystemExit("no airport harvest manifests found")

    plans = [
        build_airport_plan(
            code,
            harvest_root=args.harvest_root,
            cache_root=cache_root,
            config_file=args.config,
            chunk_hours=args.chunk_hours,
        )
        for code in airports
    ]
    validate_disjoint_roots(
        harvest_root=args.harvest_root,
        cache_root=cache_root,
        output_root=args.output,
    )
    unique_source_paths = {
        chunk.source_cache_path
        for plan in plans
        for chunk in plan.chunks
    }
    total_source_bytes = sum(path.stat().st_size for path in unique_source_paths)
    total_required = _required_free_bytes(total_source_bytes)
    total_free = shutil.disk_usage(_nearest_existing(args.output)).free
    if args.apply:
        if total_free < total_required:
            raise SystemExit(
                "insufficient space for the complete requested backfill: "
                f"{_human_bytes(total_free)} available, "
                f"{_human_bytes(total_required)} conservatively required"
            )
    provider: MetadataProvider
    provider = OpenSkyMetadataProvider() if args.apply else _DryRunProvider()
    for plan in plans:
        result = execute_airport(
            plan,
            output_root=args.output,
            cache_root=cache_root,
            provider=provider,
            apply=args.apply,
        )
        print(
            f"[adsb-metadata] {result.airport}: {result.mode}; "
            f"chunks={result.chunks}, preserved-cache={_human_bytes(result.source_cache_bytes)}, "
            f"required-free={_human_bytes(result.required_free_bytes)}"
        )
    if not args.apply:
        print(
            "[adsb-metadata] dry-run only: no network request and no file write; "
            f"all-airport preserved-cache={_human_bytes(total_source_bytes)}, "
            f"required-free={_human_bytes(total_required)}, "
            f"available={_human_bytes(total_free)}"
        )
    return 0


class _DryRunProvider:
    def fetch_state_freshness(
        self, chunk: QueryChunk, source_stop: pd.Timestamp | None
    ) -> pd.DataFrame:
        raise AssertionError("dry-run must not query OpenSky")

    def fetch_operational_status(
        self, chunk: QueryChunk, windows: tuple[AircraftWindow, ...]
    ) -> pd.DataFrame:
        raise AssertionError("dry-run must not query OpenSky")


if __name__ == "__main__":
    raise SystemExit(main())
