"""Re-run assignment from stored HAE samples without downloading ADS-B again."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from final_approach import LandingScreen

from trajectory_data_process.harvest.airports import (
    Airport,
    runway_data_fingerprint,
)
from trajectory_data_process.harvest.classify import ClassifiedTrack, classify_track
from trajectory_data_process.harvest.store import (
    ALTITUDE_DATUM,
    ALTITUDE_SOURCE,
    HarvestPaths,
    read_manifest,
    write_tracks,
)
from trajectory_data_process.harvest.tracks import Sample, Track


def reclassify_stored_tracks(
    airport: Airport,
    paths: HarvestPaths,
    *,
    screen: LandingScreen = LandingScreen(),
) -> dict[str, Any]:
    """Reclassify every rostered track, staging all output before the swap.

    The existing ``tracks/`` directory remains intact until every source record has
    parsed, classified, and serialized successfully in a sibling temporary directory.
    No acquisition module is imported or called.
    """
    source = read_manifest(paths)
    _validate_source_manifest(source, paths)
    source_manifest_sha256 = hashlib.sha256(paths.manifest.read_bytes()).hexdigest()
    provenance = dict(source.get("provenance") or {})
    provenance["reclassification"] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "network_access": False,
        "source_manifest_sha256": source_manifest_sha256,
        "runway_data_fingerprints": {
            runway.ident: runway_data_fingerprint(runway)
            for runway in airport.runways
        },
    }

    paths.root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{paths.code}-reclassify-", dir=paths.root
    ) as temporary:
        staged = HarvestPaths(Path(temporary), paths.code)
        manifest = write_tracks(
            _classified_records(source, airport, paths, screen),
            staged,
            provenance=provenance,
        )
        if manifest["total"] != source["total"]:
            raise ValueError(
                f"reclassification produced {manifest['total']} records from "
                f"{source['total']} source records"
            )
        _replace_tracks_directory(staged.tracks, paths.tracks)
    return manifest


def _validate_source_manifest(source: dict[str, Any], paths: HarvestPaths) -> None:
    if source.get("airport") != paths.code:
        raise ValueError(
            f"{paths.manifest}: airport {source.get('airport')!r} does not match {paths.code}"
        )
    if source.get("altitude_source") != ALTITUDE_SOURCE \
            or source.get("altitude_datum") != ALTITUDE_DATUM:
        raise ValueError(
            f"{paths.manifest}: stored tracks must use {ALTITUDE_SOURCE}/{ALTITUDE_DATUM}"
        )
    records = source.get("records")
    if not isinstance(records, list) or source.get("total") != len(records):
        raise ValueError(f"{paths.manifest}: invalid records roster")


def _classified_records(
    source: dict[str, Any],
    airport: Airport,
    paths: HarvestPaths,
    screen: LandingScreen,
) -> Iterator[ClassifiedTrack]:
    seen: set[str] = set()
    for index, row in enumerate(source["records"]):
        if not isinstance(row, dict) or not isinstance(row.get("file"), str):
            raise ValueError(f"{paths.manifest}: record {index} lacks file")
        key = row.get("flight_key")
        if not isinstance(key, str) or key in seen:
            raise ValueError(f"{paths.manifest}: duplicate or invalid flight_key {key!r}")
        seen.add(key)
        record_path = (paths.tracks / row["file"]).resolve()
        if not record_path.is_relative_to(paths.tracks.resolve()):
            raise ValueError(f"{paths.manifest}: record {index} escapes tracks directory")
        record = _strict_json(record_path)
        if record.get("flight_key") != key:
            raise ValueError(f"{record_path}: flight_key disagrees with manifest")
        yield classify_track(_stored_track(record, record_path), airport, screen=screen)


def _stored_track(record: dict[str, Any], path: Path) -> Track:
    if record.get("altitude_source") != ALTITUDE_SOURCE \
            or record.get("altitude_datum") != ALTITUDE_DATUM:
        raise ValueError(f"{path}: unsupported altitude source/datum")
    start_value = record.get("start_time_utc")
    if not isinstance(start_value, str):
        raise ValueError(f"{path}: start_time_utc is required")
    try:
        parsed = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{path}: invalid start_time_utc {start_value!r}") from error
    start_s = parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    rows = record.get("samples")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{path}: at least two stored samples are required")
    samples: list[Sample] = []
    previous_time = -math.inf
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{path}: samples[{index}] must be [t, lon, lat, alt]")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in row):
            raise ValueError(f"{path}: samples[{index}] must contain finite numbers")
        offset, lon, lat, altitude = map(float, row)
        time_s = start_s + offset
        if time_s < previous_time:
            raise ValueError(f"{path}: sample times must be nondecreasing")
        previous_time = time_s
        samples.append(Sample(time_s, lat, lon, altitude, False))
    icao24 = record.get("icao24")
    callsign = record.get("callsign")
    if not isinstance(icao24, str) or (callsign is not None and not isinstance(callsign, str)):
        raise ValueError(f"{path}: invalid icao24/callsign")
    return Track(icao24, callsign, tuple(samples))


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {token!r}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: track record must be an object")
    return value


def _replace_tracks_directory(staged: Path, destination: Path) -> None:
    backup = destination.parent / f".tracks-before-reclassify-{uuid4().hex}"
    destination.replace(backup)
    try:
        staged.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)
