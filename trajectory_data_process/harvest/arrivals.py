"""Build the one model-ready arrival view from the authoritative harvest roster.

``tracks/`` is the complete measured harvest and keeps all four outcomes.  Modeling
consumers need a narrower, explicit contract: one assigned runway, a published LPV
threshold target, and the final terminal-entry-to-threshold segment.  This module builds
that view through a second manifest; consumers never glob either directory.

The record geometry remains HAE.  Cesium needs HAE, while the modeling plane converts it
to MSL at ``flight_scenarios.datum``.  Moving that conversion here would silently apply
the geoid correction twice to one of the two consumers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trajectory_data_process.arrival_segment import ENTRY_RADIUS_KM, truncate_flights
from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
)
from trajectory_data_process.harvest.czml import czml_input_flight, verify_identity
from trajectory_data_process.harvest.store import (
    HarvestPaths,
    read_manifest,
    require_source_timed_manifest,
)
from trajectory_data_process.harvest.threshold_event import require_current_threshold_event

ARRIVALS_DIR = "arrivals"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = "harvest-arrivals-v4-source-timed-track-slices"


def arrival_manifest_path(paths: HarvestPaths) -> Path:
    return paths.airport / ARRIVALS_DIR / MANIFEST_NAME


def write_arrival_records(
    airport: Airport,
    paths: HarvestPaths,
    *,
    entry_radius_km: float = ENTRY_RADIUS_KM,
) -> dict[str, Any]:
    """Write the model-ready arrival records and their roster.

    Only ``assigned`` source tracks can enter. Tracks without a published per-runway TCH
    or glidepath remain in ``tracks/`` and the observed CZML, but are excluded here because
    a scenario/TS target would otherwise fall back to an invented vertical reference.
    Local circuits are also excluded after the terminal-entry cut.
    """
    source = read_manifest(paths)
    require_source_timed_manifest(source, path=paths.manifest)
    root = paths.airport / ARRIVALS_DIR
    _clear(root)
    root.mkdir(parents=True, exist_ok=True)

    roster: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    runway_targets: dict[str, dict[str, Any]] = {}
    assigned = 0

    for row in source["records"]:
        if row["outcome"] != "assigned":
            continue
        assigned += 1
        source_bytes = (paths.tracks / row["file"]).read_bytes()
        track = json.loads(source_bytes)
        runway = airport.runway(row["runway"])

        if runway.threshold_crossing_height_m is None:
            excluded.append(
                {
                    "flight_key": row["flight_key"],
                    "outcome": "no_published_tch",
                    "runway": runway.ident,
                    "reason": f"runway {runway.ident} publishes no LPV TCH",
                }
            )
            continue
        if runway.published_glidepath_deg is None:
            excluded.append(
                {
                    "flight_key": row["flight_key"],
                    "outcome": "no_published_glidepath",
                    "runway": runway.ident,
                    "reason": f"runway {runway.ident} publishes no LPV glidepath",
                }
            )
            continue

        anchor = _anchor_index(track, runway)
        flight = czml_input_flight(track)
        # The measured track may include rollout/taxi.  The supervised arrival ends at
        # the sample that defined landing_time_utc, never after it.
        flight["waypoints"] = track["samples"][: anchor + 1]
        flight["arr_airport"] = airport.code
        flight["runway_target"] = _runway_target(runway)
        verify_identity(flight, track["flight_key"])

        arrivals, locals_ = truncate_flights(
            [flight], airport.lat, airport.lon, entry_radius_km=entry_radius_km
        )
        if locals_:
            excluded.append(
                {
                    "flight_key": row["flight_key"],
                    "outcome": "local_circuit",
                    "runway": runway.ident,
                    "reason": "track starts at the field and never leaves the terminal-entry ring",
                }
            )
            continue

        arrival = arrivals[0]
        first_sample_index = int(arrival["cut_samples"])
        last_sample_index = anchor
        runway_targets.setdefault(runway.ident, _runway_target(runway))
        roster.append(
            {
                "flight_key": row["flight_key"],
                # The waypoint payload lives only in tracks/. This manifest is a view:
                # it names the inclusive source slice and the metadata derived from it.
                "source_file": row["file"],
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "first_sample_index": first_sample_index,
                "last_sample_index": last_sample_index,
                "runway": runway.ident,
                "icao24": row["icao24"],
                "callsign": row["callsign"],
                "landing_time_utc": row["landing_time_utc"],
                "entry_time_utc": arrival.get("entry_time_utc"),
                "samples": len(arrival["waypoints"]),
                "arrival_truncated": bool(arrival["arrival_truncated"]),
                "cut_samples": int(arrival["cut_samples"]),
                "arrival_duration_s": float(arrival["arrival_duration_s"]),
            }
        )

    counts: dict[str, int] = {
        "source_total": int(source["total"]),
        "assigned": assigned,
        "included": len(roster),
    }
    for item in excluded:
        key = item["outcome"]
        counts[key] = counts.get(key, 0) + 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "airport": airport.code,
        "source_manifest": "../tracks/manifest.json",
        "source_counts": source["counts"],
        "entry_radius_km": entry_radius_km,
        "altitude_source": source["altitude_source"],
        "altitude_datum": source["altitude_datum"],
        "counts": counts,
        "runway_targets": runway_targets,
        "excluded": excluded,
        "records": roster,
    }
    path = arrival_manifest_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=1, allow_nan=False), encoding="utf-8"
    )
    return manifest


def load_arrival_flights(
    path: str | Path,
    *,
    include_flight_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load model-ready arrivals strictly through ``arrivals/manifest.json``.

    When ``include_flight_keys`` is supplied, excluded source-track files are never opened.
    This lets train-only diagnostics lock a split from roster metadata without reading
    validation/test trajectory values.
    """
    manifest_path = resolve_arrival_manifest(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(
            f"{manifest_path} is not an arrival manifest object; legacy flight-array "
            "inputs are no longer supported"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path} has schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if manifest.get("altitude_source") != "opensky_history_geoaltitude_m":
        raise ValueError(
            f"{manifest_path} has unsupported altitude_source "
            f"{manifest.get('altitude_source')!r}; perform a full re-harvest"
        )
    source_manifest_value = manifest.get("source_manifest")
    if not isinstance(source_manifest_value, str):
        raise ValueError(f"{manifest_path} lacks source_manifest")
    source_manifest_path = (manifest_path.parent / source_manifest_value).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_root = source_manifest_path.parent
    source_records = source_manifest.get("records")
    if not isinstance(source_records, list):
        raise ValueError(f"{source_manifest_path} lacks a records roster")
    source_rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(source_records):
        if not isinstance(row, dict) or not isinstance(row.get("flight_key"), str):
            raise ValueError(
                f"{source_manifest_path}: source manifest record {index} lacks flight_key"
            )
        key = row["flight_key"]
        if key in source_rows:
            raise ValueError(
                f"{source_manifest_path}: source manifest lists duplicate flight_key {key!r}"
            )
        source_rows[key] = row
    runway_targets = manifest.get("runway_targets")
    if not isinstance(runway_targets, dict):
        raise ValueError(f"{manifest_path} lacks runway_targets")
    flights: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in manifest.get("records", []):
        key = row["flight_key"]
        if key in seen:
            raise ValueError(f"{manifest_path} lists duplicate flight_key {key!r}")
        seen.add(key)
        if include_flight_keys is not None and key not in include_flight_keys:
            continue
        source_row = source_rows.get(key)
        if source_row is None:
            raise ValueError(
                f"{manifest_path} references flight_key {key!r} absent from "
                f"{source_manifest_path}"
            )
        if source_row.get("file") != row.get("source_file"):
            raise ValueError(
                f"{manifest_path}: flight {key!r} source_file disagrees with "
                f"{source_manifest_path}"
            )
        source_path = source_root / source_row["file"]
        source_bytes = source_path.read_bytes()
        source_sha256 = row.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or hashlib.sha256(source_bytes).hexdigest() != source_sha256
        ):
            raise ValueError(
                f"{manifest_path}: canonical source track for flight {key!r} changed at "
                f"{source_path}; regenerate arrivals from tracks/manifest.json"
            )
        track = json.loads(source_bytes)
        first = row.get("first_sample_index")
        last = row.get("last_sample_index")
        if (
            not isinstance(first, int)
            or not isinstance(last, int)
            or first < 0
            or last < first
            or last >= len(track.get("samples", []))
        ):
            raise ValueError(
                f"{manifest_path}: flight {key!r} has invalid source sample slice "
                f"[{first!r}, {last!r}]"
            )
        waypoints = track["samples"][first:last + 1]
        t0 = waypoints[0][0]
        flight = czml_input_flight(track)
        flight["waypoints"] = [
            [sample[0] - t0, sample[1], sample[2], sample[3]]
            for sample in waypoints
        ]
        flight["arr_airport"] = manifest["airport"]
        flight["runway_target"] = runway_targets.get(row["runway"])
        flight["arrival_truncated"] = bool(row["arrival_truncated"])
        flight["cut_samples"] = int(row["cut_samples"])
        flight["arrival_duration_s"] = float(row["arrival_duration_s"])
        flight["entry_time_utc"] = row.get("entry_time_utc")
        verify_identity(flight, key)
        _validate_runway_target(flight, manifest_path)
        flights.append(flight)
    if include_flight_keys is not None:
        missing = include_flight_keys - seen
        if missing:
            raise ValueError(
                f"{manifest_path} does not roster requested flight_key {min(missing)!r}"
            )
    return flights


def resolve_arrival_manifest(path: str | Path) -> Path:
    """Resolve an airport root, arrivals directory, or explicit manifest path."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    direct = candidate / MANIFEST_NAME
    nested = candidate / ARRIVALS_DIR / MANIFEST_NAME
    if nested.exists():
        return nested
    if direct.exists():
        return direct
    raise FileNotFoundError(
        f"no arrival manifest at {nested} or {direct}; run a full harvest (or "
        "--evaluate-only on an existing tracks manifest) first"
    )


def _anchor_index(track: dict[str, Any], runway: Runway) -> int:
    samples = track.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"track {track.get('flight_key')!r} has no samples")
    event = track.get("observed_threshold_event")
    if not isinstance(event, dict):
        raise ValueError(
            f"track {track.get('flight_key')!r} lacks the serialized threshold event "
            f"for runway {runway.ident!r}; run --reclassify-existing"
        )
    require_current_threshold_event(event, runway)
    stored = track.get("landing_sample_index")
    if isinstance(stored, int) and not isinstance(stored, bool) and 0 <= stored < len(samples):
        return stored
    raise ValueError(
        f"track {track.get('flight_key')!r} has no valid stored landing_sample_index; "
        "run --reclassify-existing"
    )


def _runway_target(runway: Runway) -> dict[str, Any]:
    return {
        "lat": runway.lat,
        "lon": runway.lon,
        "elevation_msl_m": runway.elevation_msl_m,
        "elevation_hae_m": runway.elevation_hae_m,
        "hae_minus_msl_m": runway.hae_minus_msl_m,
        "course_deg": runway.course_deg,
        "threshold_crossing_height_m": runway.threshold_crossing_height_m,
        "published_glidepath_deg": runway.published_glidepath_deg,
        "position_source": runway.position_source,
        "vertical_source": runway.vertical_source,
    }


def _validate_runway_target(flight: dict[str, Any], path: Path) -> None:
    target = flight.get("runway_target") or {}
    required = (
        "lat", "lon", "elevation_hae_m", "elevation_msl_m", "hae_minus_msl_m",
        "course_deg", "threshold_crossing_height_m", "published_glidepath_deg",
        "position_source", "vertical_source",
    )
    missing = [key for key in required if target.get(key) is None]
    if missing:
        raise ValueError(
            f"{path}: flight {flight.get('id')!r} lacks runway_target fields "
            f"{missing}; perform a full re-harvest"
        )
    if abs(
        float(target["elevation_hae_m"])
        - float(target["elevation_msl_m"])
        - float(target["hae_minus_msl_m"])
    ) > 1e-6:
        raise ValueError(f"{path}: flight {flight.get('id')!r} has inconsistent runway datum")


def _clear(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*.json"), reverse=True):
        path.unlink()
    for path in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
        path.rmdir()
