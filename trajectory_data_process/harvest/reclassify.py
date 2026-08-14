"""Re-run assignment from stored HAE samples without downloading ADS-B again."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4

from final_approach import LandingScreen

from trajectory_data_process.harvest.airports import (
    Airport,
    runway_data_fingerprint,
)
from trajectory_data_process.harvest.adsb_metadata import AdsbStateMetadata
from trajectory_data_process.harvest.classify import ClassifiedTrack, classify_track
from trajectory_data_process.harvest.store import (
    ALTITUDE_DATUM,
    ALTITUDE_SOURCE,
    HarvestPaths,
    read_manifest,
    write_tracks,
)
from trajectory_data_process.harvest.tracks import (
    Sample,
    SourceIntegrity,
    Track,
    source_integrity_from_dict,
    source_timed_final_block,
)
from trajectory_data_process.harvest.threshold_event import StateMetadataLookup


_FLIGHT_KEY_TIME = re.compile(r"_(\d{8}T\d{6}Z)$")
DEFAULT_RECLASSIFY_BATCH_TRACKS = 512
StateMetadataBatchLookup = Callable[
    [list[tuple[str, float]]], list[AdsbStateMetadata | None]
]


def reclassify_stored_tracks(
    airport: Airport,
    paths: HarvestPaths,
    *,
    metadata_lookup: StateMetadataLookup,
    metadata_provenance: dict[str, Any],
    metadata_lookup_many: StateMetadataBatchLookup | None = None,
    batch_tracks: int = DEFAULT_RECLASSIFY_BATCH_TRACKS,
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
    freshness_provenance = provenance.get("freshness_rebuild")
    if (
        isinstance(freshness_provenance, dict)
        and "source_integrity" in freshness_provenance
    ):
        freshness_provenance = dict(freshness_provenance)
        freshness_provenance.pop("source_integrity")
        provenance["freshness_rebuild"] = freshness_provenance
    provenance["reclassification"] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "network_access": False,
        "source_manifest_sha256": source_manifest_sha256,
        "adsb_metadata": dict(metadata_provenance),
        "batch_tracks": batch_tracks,
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
            _classified_records(
                source,
                airport,
                paths,
                screen,
                metadata_lookup=metadata_lookup,
                metadata_lookup_many=metadata_lookup_many,
                batch_tracks=batch_tracks,
            ),
            staged,
            provenance=provenance,
            source_integrity=(
                source["source_integrity"]
                if isinstance(source.get("source_integrity"), dict)
                else None
            ),
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
    *,
    metadata_lookup: StateMetadataLookup | None,
    metadata_lookup_many: StateMetadataBatchLookup | None,
    batch_tracks: int,
) -> Iterator[ClassifiedTrack]:
    if batch_tracks < 1:
        raise ValueError("batch_tracks must be at least one")
    seen: set[str] = set()
    # Sidecar partitions are chronological. Every outcome must follow flight time:
    # tracks previously rejected by an older classifier may now reach metadata lookup,
    # so callsign ordering would repeatedly evict and reload the same Parquet partitions.
    ordered = sorted(source["records"], key=_reclassification_order)
    for offset in range(0, len(ordered), batch_tracks):
        loaded: list[tuple[Path, Track, int]] = []
        queries: list[tuple[str, float]] = []
        for index, row in enumerate(
            ordered[offset : offset + batch_tracks], start=offset
        ):
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
            stored = _stored_track(record, record_path)
            query_count = 0
            if stored.source_integrity is None:
                query_count = len(stored.samples)
                queries.extend(
                    (stored.icao24, sample.time_s) for sample in stored.samples
                )
            loaded.append((record_path, stored, query_count))

        if metadata_lookup_many is not None:
            resolved = metadata_lookup_many(queries)
        else:
            resolved = [
                metadata_lookup(icao24, time_s)
                if metadata_lookup is not None
                else None
                for icao24, time_s in queries
            ]
        if len(resolved) != len(queries):
            raise ValueError("ADS-B batch lookup returned the wrong result count")
        cursor = 0
        for record_path, stored, query_count in loaded:
            if stored.source_integrity is not None:
                fresh = stored
            else:
                fresh, _integrity = source_timed_track_from_metadata(
                    stored, resolved[cursor : cursor + query_count]
                )
                cursor += query_count
            if fresh is None:
                raise ValueError(
                    f"{record_path}: final fresh position block has fewer than two "
                    "samples; use --rebuild-fresh-from with a new staging output to "
                    "retain the exclusion in the source denominator"
                )
            yield classify_track(fresh, airport, screen=screen)
        if cursor != len(resolved):
            raise AssertionError("ADS-B batch results were not consumed exactly once")


def _reclassification_order(row: Any) -> tuple[str, str]:
    if not isinstance(row, dict):
        return ("99999999T999999Z", "")
    key = row.get("flight_key")
    if not isinstance(key, str):
        return ("99999999T999999Z", "")
    match = _FLIGHT_KEY_TIME.search(key)
    return (
        match.group(1) if match is not None else "99999999T999999Z",
        key,
    )


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
    integrity_value = record.get("source_integrity")
    integrity = (
        source_integrity_from_dict(integrity_value)
        if integrity_value is not None
        else None
    )
    speeds_value = record.get("reported_ground_speeds_m_s")
    if integrity is not None and (
        not isinstance(speeds_value, list) or len(speeds_value) != len(rows)
    ):
        raise ValueError(
            f"{path}: source-timed track requires one reported speed per sample"
        )
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
        speed = None
        if integrity is not None:
            assert isinstance(speeds_value, list)
            raw_speed = speeds_value[index]
            if raw_speed is not None:
                if (
                    isinstance(raw_speed, bool)
                    or not isinstance(raw_speed, (int, float))
                    or not math.isfinite(float(raw_speed))
                ):
                    raise ValueError(
                        f"{path}: reported_ground_speeds_m_s[{index}] must be "
                        "finite or null"
                    )
                speed = float(raw_speed)
        samples.append(
            Sample(
                time_s,
                lat,
                lon,
                altitude,
                False,
                speed,
                time_s if integrity is not None else None,
                None,
            )
        )
    icao24 = record.get("icao24")
    callsign = record.get("callsign")
    if not isinstance(icao24, str) or (callsign is not None and not isinstance(callsign, str)):
        raise ValueError(f"{path}: invalid icao24/callsign")
    if integrity is not None and integrity.retained_rows != len(samples):
        raise ValueError(f"{path}: source_integrity retained_rows disagrees with samples")
    return Track(icao24, callsign, tuple(samples), integrity)


def source_timed_track_from_metadata(
    track: Track, metadata: Sequence[AdsbStateMetadata | None]
) -> tuple[Track | None, SourceIntegrity]:
    """Build one clean final block from already aligned sidecar results."""
    if len(metadata) != len(track.samples):
        raise ValueError("ADS-B metadata result count does not match track samples")
    enriched: list[Sample] = []
    for sample, state in zip(track.samples, metadata):
        enriched.append(
            Sample(
                sample.time_s,
                sample.lat,
                sample.lon,
                sample.alt_hae_m,
                sample.on_ground,
                state.reported_ground_speed_m_s if state else None,
                state.last_position_update_s if state else None,
                state.last_contact_s if state else None,
            )
        )
    samples, integrity = source_timed_final_block(enriched)
    if len(samples) < 2:
        return None, integrity
    return Track(track.icao24, track.callsign, tuple(samples), integrity), integrity


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
