"""Transactionally merge complete harvest manifests without downloading ADS-B again."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from trajectory_data_process.harvest.airports import Airport
from trajectory_data_process.harvest.reclassify import reclassify_stored_tracks
from trajectory_data_process.harvest.store import (
    ALTITUDE_DATUM,
    ALTITUDE_SOURCE,
    HarvestPaths,
)

_BUCKETS = ("assigned", "ambiguous", "unassignable", "not_landing")


def merge_stored_tracks(
    destination: HarvestPaths,
    additional_sources: list[HarvestPaths],
    *,
    airport: Airport,
) -> dict[str, Any]:
    """Merge destination plus additional source manifests through a staged hard-link tree.

    Every source manifest and rostered JSON record is validated, reclassified against
    the current airport data, checked for current-key collisions, and serialized before
    the destination is changed. Duplicate identities or relative record paths are
    rejected rather than guessed away. Because all current harvest roots share one
    filesystem, hard links keep the raw staging tree small.
    """
    if not additional_sources:
        raise ValueError("at least one additional harvest source is required")

    sources = [destination, *additional_sources]
    _reject_repeated_manifests(sources)
    validated = [_validate_source(source, destination.code) for source in sources]
    records = _combined_roster(validated)
    provenance = {
        "merge": {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "network_access": False,
            "method": "validated_same_filesystem_hard_links_then_reclassification",
            "sources": [item["audit"] for item in validated],
        }
    }

    destination.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.code}-merge-", dir=destination.root
    ) as temporary:
        staged = HarvestPaths(Path(temporary), destination.code)
        _stage_records(staged, validated)
        _write_manifest(staged, records, provenance)
        # Reclassification is part of the same transaction. Parsing, current
        # assignment, current-key uniqueness, and serialization must all pass
        # while the canonical tree and its derived views are still untouched.
        manifest = reclassify_stored_tracks(airport, staged)
        _replace_tracks_directory(staged.tracks, destination.tracks)

    _invalidate_local_views(destination)
    return manifest


def _reject_repeated_manifests(sources: list[HarvestPaths]) -> None:
    seen: set[Path] = set()
    for source in sources:
        manifest = source.manifest.resolve()
        if manifest in seen:
            raise ValueError(f"harvest manifest was supplied more than once: {manifest}")
        seen.add(manifest)


def _validate_source(source: HarvestPaths, airport: str) -> dict[str, Any]:
    manifest = _strict_json(source.manifest)
    if manifest.get("airport") != airport:
        raise ValueError(
            f"{source.manifest}: airport {manifest.get('airport')!r} does not match {airport}"
        )
    if manifest.get("altitude_source") != ALTITUDE_SOURCE \
            or manifest.get("altitude_datum") != ALTITUDE_DATUM:
        raise ValueError(
            f"{source.manifest}: stored tracks must use "
            f"{ALTITUDE_SOURCE}/{ALTITUDE_DATUM}"
        )

    rows = manifest.get("records")
    if not isinstance(rows, list) or manifest.get("total") != len(rows):
        raise ValueError(f"{source.manifest}: invalid records roster")

    counts = {bucket: 0 for bucket in _BUCKETS}
    per_runway: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    local_keys: set[str] = set()
    local_paths: set[str] = set()
    for index, row in enumerate(rows):
        entry = _validate_row(source, row, index)
        key = entry["row"]["flight_key"]
        relative = entry["row"]["file"]
        if key in local_keys:
            raise ValueError(f"{source.manifest}: duplicate flight_key {key!r}")
        if relative in local_paths:
            raise ValueError(f"{source.manifest}: duplicate record path {relative!r}")
        local_keys.add(key)
        local_paths.add(relative)
        outcome = entry["row"]["outcome"]
        counts[outcome] += 1
        runway = entry["row"].get("runway")
        if runway is not None:
            per_runway[runway] = per_runway.get(runway, 0) + 1
        entries.append(entry)

    if manifest.get("counts") != counts:
        raise ValueError(
            f"{source.manifest}: counts disagree with records roster "
            f"({manifest.get('counts')!r} != {counts!r})"
        )
    if manifest.get("per_runway") != dict(sorted(per_runway.items())):
        raise ValueError(f"{source.manifest}: per_runway disagrees with records roster")

    return {
        "paths": source,
        "entries": entries,
        "audit": {
            "manifest": str(source.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(source.manifest.read_bytes()).hexdigest(),
            "written_utc": manifest.get("written_utc"),
            "total": len(rows),
            "provenance": manifest.get("provenance"),
        },
    }


def _validate_row(source: HarvestPaths, row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{source.manifest}: record {index} must be an object")
    key = row.get("flight_key")
    relative_value = row.get("file")
    outcome = row.get("outcome")
    if not isinstance(key, str) or not key:
        raise ValueError(f"{source.manifest}: record {index} lacks flight_key")
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError(f"{source.manifest}: record {index} lacks file")
    if outcome not in _BUCKETS:
        raise ValueError(f"{source.manifest}: record {index} has invalid outcome {outcome!r}")
    if row.get("runway") is not None and not isinstance(row.get("runway"), str):
        raise ValueError(f"{source.manifest}: record {index} has invalid runway")

    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"{source.manifest}: record {index} escapes tracks directory: {relative_value!r}"
        )
    record_path = source.tracks / relative
    try:
        resolved = record_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{source.manifest}: missing rostered record {record_path}") from error
    if not resolved.is_relative_to(source.tracks.resolve()):
        raise ValueError(
            f"{source.manifest}: record {index} escapes tracks directory: {relative_value!r}"
        )
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError(f"{source.manifest}: rostered record must be a regular file: {record_path}")

    record = _strict_json(record_path)
    for field in ("flight_key", "outcome", "runway", "icao24", "callsign",
                  "landing_time_utc", "landing_sample_index"):
        if record.get(field) != row.get(field):
            raise ValueError(
                f"{record_path}: {field} disagrees with source manifest record {index}"
            )
    if record.get("altitude_source") != ALTITUDE_SOURCE \
            or record.get("altitude_datum") != ALTITUDE_DATUM:
        raise ValueError(f"{record_path}: unsupported altitude source/datum")
    _validate_samples(record.get("samples"), record_path)

    copied_row = dict(row)
    copied_row["file"] = relative.as_posix()
    return {"row": copied_row, "source_path": record_path}


def _validate_samples(value: Any, path: Path) -> None:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{path}: at least two stored samples are required")
    previous = -math.inf
    for index, sample in enumerate(value):
        if not isinstance(sample, list) or len(sample) != 4:
            raise ValueError(f"{path}: samples[{index}] must be [t, lon, lat, alt]")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in sample
        ):
            raise ValueError(f"{path}: samples[{index}] must contain finite numbers")
        time_offset = float(sample[0])
        if time_offset < previous:
            raise ValueError(f"{path}: sample times must be nondecreasing")
        previous = time_offset


def _combined_roster(validated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys: set[str] = set()
    paths: set[str] = set()
    for source in validated:
        for entry in source["entries"]:
            row = entry["row"]
            key = row["flight_key"]
            relative = row["file"]
            if key in keys:
                raise ValueError(f"merged harvest has duplicate flight_key {key!r}")
            if relative in paths:
                raise ValueError(f"merged harvest has duplicate record path {relative!r}")
            keys.add(key)
            paths.add(relative)
            records.append(row)
    return records


def _stage_records(staged: HarvestPaths, validated: list[dict[str, Any]]) -> None:
    for source in validated:
        for entry in source["entries"]:
            destination = staged.tracks / entry["row"]["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(entry["source_path"], destination, follow_symlinks=False)
            except OSError as error:
                raise OSError(
                    f"could not hard-link {entry['source_path']} into staged merge; "
                    "harvest roots must be on the same filesystem"
                ) from error


def _write_manifest(
    staged: HarvestPaths,
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    counts = {bucket: 0 for bucket in _BUCKETS}
    per_runway: dict[str, int] = {}
    for row in records:
        counts[row["outcome"]] += 1
        runway = row.get("runway")
        if runway is not None:
            per_runway[runway] = per_runway.get(runway, 0) + 1
    manifest = {
        "airport": staged.code,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "counts": counts,
        "per_runway": dict(sorted(per_runway.items())),
        "total": len(records),
        "provenance": provenance,
        "records": records,
    }
    staged.manifest.parent.mkdir(parents=True, exist_ok=True)
    staged.manifest.write_text(
        json.dumps(manifest, indent=1, allow_nan=False), encoding="utf-8"
    )
    return manifest


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"{path}: non-standard JSON numeric constant {token!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _replace_tracks_directory(staged: Path, destination: Path) -> None:
    backup = destination.parent / f".tracks-before-merge-{uuid4().hex}"
    destination.replace(backup)
    try:
        staged.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def _invalidate_local_views(paths: HarvestPaths) -> None:
    for directory in (paths.airport / "arrivals", paths.approach):
        if directory.exists():
            shutil.rmtree(directory)
