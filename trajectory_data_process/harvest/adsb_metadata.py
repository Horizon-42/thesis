"""Read protected ADS-B freshness sidecars for no-download reclassification."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


SIDECAR_SCHEMA = "aeroviz.adsb-metadata-backfill.v1"
_PARQUET_AUDIT_KEY = b"aeroviz_adsb_backfill"


@dataclass(frozen=True)
class AdsbStateMetadata:
    """Source telemetry associated with one OpenSky state-vector row."""

    reported_ground_speed_m_s: float | None
    last_position_update_s: float | None
    last_contact_s: float | None


@dataclass(frozen=True)
class _Partition:
    path: Path
    start_s: float
    stop_s: float


class SidecarStateMetadata:
    """Exact ``(icao24, state-row time)`` lookup over backfilled Parquet files.

    Partitions are loaded lazily and cached two at a time. Reclassification orders all
    stored outcomes chronologically, so every partition is normally read once without
    loading the multi-gigabyte airport sidecar into memory at once.
    """

    def __init__(self, root: Path, airport: str, *, cache_partitions: int = 2) -> None:
        if cache_partitions < 1:
            raise ValueError("cache_partitions must be at least one")
        self.airport = airport.upper()
        self.base = root / self.airport
        manifest_path = self.base / "manifest.json"
        self._partitions = _load_catalog(self.base, self.airport)
        self.provenance = {
            "schema": SIDECAR_SCHEMA,
            "airport": self.airport,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        self._starts = [partition.start_s for partition in self._partitions]
        running_stop = -math.inf
        self._prefix_max_stops = []
        for partition in self._partitions:
            running_stop = max(running_stop, partition.stop_s)
            self._prefix_max_stops.append(running_stop)
        self._cache_limit = cache_partitions
        self._cache: OrderedDict[
            Path, dict[tuple[str, int], AdsbStateMetadata | None]
        ] = OrderedDict()

    def lookup(self, icao24: str, state_time_s: float) -> AdsbStateMetadata | None:
        """Return one unambiguous exact row; never guess across duplicates."""
        if not math.isfinite(state_time_s):
            return None
        key = (icao24.lower().strip(), round(state_time_s * 1000.0))
        matches: list[AdsbStateMetadata] = []
        for partition in self._containing(state_time_s):
            value = self._rows(partition).get(key)
            if value is not None:
                matches.append(value)
        if not matches:
            return None
        first = matches[0]
        return first if all(value == first for value in matches[1:]) else None

    def _containing(self, state_time_s: float) -> list[_Partition]:
        stop = bisect.bisect_right(self._starts, state_time_s)
        # Adjacent OpenSky chunks can share a boundary. In practice no timestamp is
        # covered by more than two partitions; walking backwards also handles a short
        # overlapping snapshot without assuming exact chunk length.
        output: list[_Partition] = []
        for index in range(stop - 1, -1, -1):
            if self._prefix_max_stops[index] < state_time_s:
                break
            partition = self._partitions[index]
            if partition.stop_s >= state_time_s:
                output.append(partition)
        return output

    def _rows(
        self, partition: _Partition
    ) -> dict[tuple[str, int], AdsbStateMetadata | None]:
        cached = self._cache.pop(partition.path, None)
        if cached is not None:
            self._cache[partition.path] = cached
            return cached
        frame = pd.read_parquet(
            partition.path,
            columns=["time", "icao24", "velocity", "lastposupdate", "lastcontact"],
        )
        # Parquet readers may preserve the source as ms, us, or ns. Normalize before
        # taking the integer view; assuming pandas' unit here causes exact joins to miss.
        time_ms = pd.to_datetime(frame["time"], utc=True).astype(
            "datetime64[ms, UTC]"
        ).astype("int64")
        aircraft = frame["icao24"].astype(str).str.lower().str.strip()
        rows: dict[tuple[str, int], AdsbStateMetadata | None] = {}
        for index, source in enumerate(
            frame[["velocity", "lastposupdate", "lastcontact"]].itertuples(
                index=False, name=None
            )
        ):
            key = (aircraft.iat[index], int(time_ms.iat[index]))
            value = AdsbStateMetadata(*(_finite(item) for item in source))
            previous = rows.get(key, value)
            rows[key] = value if previous == value else None
        self._cache[partition.path] = rows
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return rows


def _load_catalog(base: Path, airport: str) -> list[_Partition]:
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run backfill_adsb_metadata.py before "
            "--reclassify-existing"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("airport") != airport or manifest.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"{manifest_path}: unsupported airport or sidecar schema")
    contract = manifest.get("source_contract") or {}
    expected_units = {
        "reported_velocity_unit": "m/s",
        "lastposupdate_unit": "unix_seconds",
        "lastcontact_unit": "unix_seconds",
    }
    units_match = all(
        contract.get(key) == value for key, value in expected_units.items()
    )
    # Early completed airports declared (time, icao24) as the lookup key. Later
    # manifests correctly record that OpenSky can contain duplicate rows and instead
    # declare the full-row-plus-occurrence merge identity. This reader is safe for
    # both: an exact duplicate metadata value is accepted; a conflicting duplicate is
    # returned as unavailable.
    identity_matches = contract.get("state_vector_key") == ["time", "icao24"] or (
        contract.get("state_vector_key") is None
        and contract.get("state_vector_row_alignment")
        == "preserved source-cache row order"
        and contract.get("state_vector_identity")
        == "full source state-vector row plus occurrence among identical rows"
    )
    if not units_match or not identity_matches:
        raise ValueError(f"{manifest_path}: incompatible state-vector source contract")
    roster = (manifest.get("partitions") or {}).get("state_vectors")
    if not isinstance(roster, list) or not roster:
        raise ValueError(f"{manifest_path}: no state-vector partitions")

    resolved_base = base.resolve()
    output: list[_Partition] = []
    for index, row in enumerate(roster):
        relative = row.get("file") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise ValueError(f"{manifest_path}: state partition {index} lacks file")
        path = (base / relative).resolve()
        if not path.is_relative_to(resolved_base) or not path.is_file():
            raise ValueError(f"{manifest_path}: invalid state partition {relative!r}")
        metadata = pq.read_metadata(path).metadata or {}
        try:
            audit = json.loads(metadata[_PARQUET_AUDIT_KEY].decode())
            start_s = pd.Timestamp(audit["start_utc"]).timestamp()
            stop_s = pd.Timestamp(audit["stop_utc"]).timestamp()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: invalid ADS-B backfill audit metadata") from error
        output.append(_Partition(path, start_s, stop_s))
    return sorted(output, key=lambda partition: (partition.start_s, partition.stop_s))


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
