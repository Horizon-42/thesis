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

import json
from pathlib import Path
from typing import Any

from final_approach import TrackPoint

from trajectory_data_process.arrival_segment import ENTRY_RADIUS_KM, truncate_flights
from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.czml import czml_input_flight, verify_identity
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

ARRIVALS_DIR = "arrivals"
RECORDS_DIR = "records"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = "harvest-arrivals-v1"


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
    root = paths.airport / ARRIVALS_DIR
    records_dir = root / RECORDS_DIR
    _clear(root)
    records_dir.mkdir(parents=True, exist_ok=True)

    roster: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    assigned = 0

    for row in source["records"]:
        if row["outcome"] != "assigned":
            continue
        assigned += 1
        track = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
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
        record_path = records_dir / f"{row['flight_key']}.json"
        record_path.write_text(json.dumps(arrival, indent=1), encoding="utf-8")
        roster.append(
            {
                "flight_key": row["flight_key"],
                "file": str(record_path.relative_to(root)),
                "runway": runway.ident,
                "icao24": row["icao24"],
                "callsign": row["callsign"],
                "landing_time_utc": row["landing_time_utc"],
                "entry_time_utc": arrival.get("entry_time_utc"),
                "samples": len(arrival["waypoints"]),
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
        "excluded": excluded,
        "records": roster,
    }
    path = arrival_manifest_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_arrival_flights(path: str | Path) -> list[dict[str, Any]]:
    """Load model-ready arrivals strictly through ``arrivals/manifest.json``."""
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
    root = manifest_path.parent
    flights: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in manifest.get("records", []):
        key = row["flight_key"]
        if key in seen:
            raise ValueError(f"{manifest_path} lists duplicate flight_key {key!r}")
        seen.add(key)
        flight = json.loads((root / row["file"]).read_text(encoding="utf-8"))
        verify_identity(flight, key)
        flights.append(flight)
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
    stored = track.get("landing_sample_index")
    if isinstance(stored, int) and 0 <= stored < len(track["samples"]):
        return stored
    frame = runway.frame("hae")
    return min(
        range(len(track["samples"])),
        key=lambda index: frame.distance_m(
            TrackPoint(
                lat=float(track["samples"][index][2]),
                lon=float(track["samples"][index][1]),
                alt_m=float(track["samples"][index][3]),
            )
        ),
    )


def _runway_target(runway: Runway) -> dict[str, Any]:
    return {
        "lat": runway.lat,
        "lon": runway.lon,
        "elevation_msl_m": runway.elevation_msl_m,
        "course_deg": runway.course_deg,
        "threshold_crossing_height_m": runway.threshold_crossing_height_m,
        "published_glidepath_deg": runway.published_glidepath_deg,
        "position_source": runway.position_source,
    }


def _clear(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*.json"), reverse=True):
        path.unlink()
    for path in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
        path.rmdir()
