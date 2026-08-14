"""Build source-timed tracks in a new staging root without touching source data."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from final_approach import LandingScreen

from trajectory_data_process.harvest.adsb_metadata import SidecarStateMetadata
from trajectory_data_process.harvest.airports import Airport, runway_data_fingerprint
from trajectory_data_process.harvest.classify import ClassifiedTrack, classify_track
from trajectory_data_process.harvest.reclassify import (
    _reclassification_order,
    _stored_track,
    _strict_json,
    _validate_source_manifest,
    source_timed_track_from_metadata,
)
from trajectory_data_process.harvest.store import (
    TRACK_SCHEMA_VERSION,
    HarvestPaths,
    read_manifest,
    write_tracks,
)
from trajectory_data_process.harvest.tracks import (
    SOURCE_INTEGRITY_SCHEMA,
    SourceIntegrity,
    Track,
)


DEFAULT_REBUILD_BATCH_TRACKS = 512
OUTPUT_SPACE_MULTIPLIER = 3
MIN_FREE_AFTER_REBUILD_BYTES = 2 * 1024**3


def rebuild_fresh_tracks(
    airport: Airport,
    source: HarvestPaths,
    destination: HarvestPaths,
    *,
    metadata: SidecarStateMetadata,
    batch_tracks: int = DEFAULT_REBUILD_BATCH_TRACKS,
    screen: LandingScreen = LandingScreen(),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Rebuild one airport into an absent destination, preserving the source bitwise.

    The source manifest and every rostered record are opened read-only.  Output is
    first written under a temporary directory in the destination filesystem and is
    renamed into place only after the source roster's size/mtime fingerprint is
    unchanged.
    """
    if batch_tracks < 1:
        raise ValueError("batch_tracks must be at least one")
    _validate_distinct_roots(source, destination)
    source_manifest = read_manifest(source)
    _validate_source_manifest(source_manifest, source)
    if (
        source_manifest.get("schema_version") == TRACK_SCHEMA_VERSION
        and source_manifest.get("source_integrity_complete") is True
    ):
        raise ValueError(
            f"{source.manifest} is already source-timed; use --reclassify-existing "
            "for a runway/CIFP change or --evaluate-only for downstream views"
        )
    before = _source_fingerprint(source, source_manifest)
    source_record_bytes = _source_record_bytes(source, source_manifest)
    space = _require_output_space(destination, source_record_bytes)
    source_manifest_sha256 = hashlib.sha256(source.manifest.read_bytes()).hexdigest()
    audit: dict[str, Any] = {
        "schema_version": SOURCE_INTEGRITY_SCHEMA,
        "source_total": int(source_manifest["total"]),
        "source_counts": dict(source_manifest["counts"]),
        "output_total": 0,
        "excluded_total": 0,
        "excluded": [],
        "totals": {},
    }
    provenance = dict(source_manifest.get("provenance") or {})
    provenance["freshness_rebuild"] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "network_access": False,
        "source_root": str(source.airport.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_roster_stat_sha256": before,
        "adsb_metadata": dict(metadata.provenance),
        "batch_tracks": batch_tracks,
        "space_preflight": space,
        "runway_data_fingerprints": {
            runway.ident: runway_data_fingerprint(runway)
            for runway in airport.runways
        },
    }

    destination.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.code}-freshness-", dir=destination.root
    ) as temporary:
        staged = HarvestPaths(Path(temporary), destination.code)
        manifest = write_tracks(
            _classified_batches(
                source_manifest,
                airport,
                source,
                metadata,
                audit,
                batch_tracks=batch_tracks,
                screen=screen,
                log=log,
            ),
            staged,
            provenance=provenance,
            source_integrity=audit,
        )
        audit["output_total"] = int(manifest["total"])
        audit["excluded_total"] = len(audit["excluded"])
        if audit["output_total"] + audit["excluded_total"] != audit["source_total"]:
            raise RuntimeError(
                "freshness rebuild output plus exclusions does not equal source total"
            )
        provenance["freshness_rebuild"]["completed_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        manifest["source_integrity"] = audit
        staged.manifest.write_text(
            json.dumps(manifest, indent=1, allow_nan=False), encoding="utf-8"
        )
        after = _source_fingerprint(source, source_manifest)
        if after != before:
            raise RuntimeError(
                "source harvest changed while the freshness rebuild was running; "
                "staged output was not committed"
            )
        staged.airport.replace(destination.airport)
    return manifest


def _classified_batches(
    source_manifest: dict[str, Any],
    airport: Airport,
    paths: HarvestPaths,
    metadata: SidecarStateMetadata,
    audit: dict[str, Any],
    *,
    batch_tracks: int,
    screen: LandingScreen,
    log: Callable[[str], None],
) -> Iterator[ClassifiedTrack]:
    records = sorted(source_manifest["records"], key=_reclassification_order)
    totals: Counter[str] = Counter()
    seen: set[str] = set()
    for offset in range(0, len(records), batch_tracks):
        batch = records[offset : offset + batch_tracks]
        loaded: list[tuple[dict[str, Any], Track]] = []
        queries: list[tuple[str, float]] = []
        lengths: list[int] = []
        for index, row in enumerate(batch, start=offset):
            if not isinstance(row, dict) or not isinstance(row.get("file"), str):
                raise ValueError(f"{paths.manifest}: record {index} lacks file")
            key = row.get("flight_key")
            if not isinstance(key, str) or key in seen:
                raise ValueError(
                    f"{paths.manifest}: duplicate or invalid flight_key {key!r}"
                )
            seen.add(key)
            record_path = (paths.tracks / row["file"]).resolve()
            if not record_path.is_relative_to(paths.tracks.resolve()):
                raise ValueError(
                    f"{paths.manifest}: record {index} escapes tracks directory"
                )
            record = _strict_json(record_path)
            if record.get("flight_key") != key:
                raise ValueError(f"{record_path}: flight_key disagrees with manifest")
            track = _stored_track(record, record_path)
            loaded.append((row, track))
            lengths.append(len(track.samples))
            queries.extend((track.icao24, sample.time_s) for sample in track.samples)

        resolved = metadata.lookup_many(queries)
        cursor = 0
        for (row, track), length in zip(loaded, lengths):
            values = resolved[cursor : cursor + length]
            cursor += length
            rebuilt, integrity = source_timed_track_from_metadata(track, values)
            _accumulate_integrity(totals, integrity)
            if rebuilt is None:
                audit["excluded"].append(
                    {
                        "source_flight_key": row["flight_key"],
                        "source_outcome": row.get("outcome"),
                        "reason": "final fresh position block has fewer than two samples",
                        "source_integrity": integrity.to_dict(),
                    }
                )
                continue
            yield classify_track(rebuilt, airport, screen=screen)
        log(
            f"[freshness] {airport.code}: "
            f"{min(offset + len(batch), len(records)):,}/{len(records):,} "
            f"source tracks"
        )
    audit["totals"] = dict(sorted(totals.items()))


def _accumulate_integrity(totals: Counter[str], integrity: SourceIntegrity) -> None:
    for item in fields(integrity):
        value = getattr(integrity, item.name)
        if isinstance(value, int):
            totals[item.name] += value


def _validate_distinct_roots(source: HarvestPaths, destination: HarvestPaths) -> None:
    source_root = source.root.resolve()
    destination_root = destination.root.resolve()
    source_airport = source.airport.resolve()
    destination_airport = destination.airport.resolve()
    if source_airport == destination_airport:
        raise ValueError("freshness rebuild destination must differ from the source")
    if destination_airport.is_relative_to(source_airport) or source_airport.is_relative_to(
        destination_airport
    ):
        raise ValueError("freshness rebuild source and destination may not be nested")
    if destination_root.is_relative_to(source_root) or source_root.is_relative_to(
        destination_root
    ):
        raise ValueError(
            "freshness rebuild source and destination roots may not be nested"
        )
    if destination.airport.exists():
        raise FileExistsError(
            f"{destination.airport} already exists; choose a new staging output root"
        )


def _source_fingerprint(paths: HarvestPaths, manifest: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    manifest_stat = paths.manifest.stat()
    digest.update(
        f"manifest\0{manifest_stat.st_size}\0{manifest_stat.st_mtime_ns}\n".encode()
    )
    for index, row in enumerate(manifest["records"]):
        relative = row.get("file") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise ValueError(f"{paths.manifest}: record {index} lacks file")
        path = (paths.tracks / relative).resolve()
        if not path.is_relative_to(paths.tracks.resolve()):
            raise ValueError(f"{paths.manifest}: record {index} escapes tracks directory")
        stat = path.stat()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _source_record_bytes(paths: HarvestPaths, manifest: dict[str, Any]) -> int:
    total = 0
    for index, row in enumerate(manifest["records"]):
        relative = row.get("file") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise ValueError(f"{paths.manifest}: record {index} lacks file")
        path = (paths.tracks / relative).resolve()
        if not path.is_relative_to(paths.tracks.resolve()):
            raise ValueError(f"{paths.manifest}: record {index} escapes tracks directory")
        total += path.stat().st_size
    return total


def _require_output_space(
    destination: HarvestPaths, source_record_bytes: int
) -> dict[str, int]:
    destination.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.root).free
    estimated = max(1024**3, source_record_bytes * OUTPUT_SPACE_MULTIPLIER)
    required = estimated + MIN_FREE_AFTER_REBUILD_BYTES
    if free < required:
        raise OSError(
            f"insufficient staging space at {destination.root}: "
            f"{free / 1024**3:.1f} GiB free, require at least "
            f"{required / 1024**3:.1f} GiB (estimated output plus 2 GiB reserve)"
        )
    return {
        "source_record_bytes": source_record_bytes,
        "estimated_output_bytes": estimated,
        "free_before_bytes": free,
        "required_with_reserve_bytes": required,
    }
