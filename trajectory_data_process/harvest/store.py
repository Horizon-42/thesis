"""The harvest's on-disk layout: source samples and policy-free derived events.

    outputs/harvest/<ICAO>/
        tracks/                     <- Reconstructed HAE samples plus assignment output.
            assigned/<RWY>/<flight_key>.json
            ambiguous/<flight_key>.json
            unassignable/<flight_key>.json
            not_landing/<flight_key>.json
            manifest.json
        arrivals/                   <- Model-ready final-arrival views of assigned tracks.

WHY THE SPLIT IS PHYSICAL AND NOT COSMETIC
------------------------------------------
The sample array remains the sensor reconstruction: ellipsoidal altitude exactly as
broadcast, with no extrapolated point inserted. Runway assignment also serializes one
clearly labelled ``observed_threshold_event`` beside those samples. That event is
policy-free: a valid measured threshold bracket is interpolated directly in 3D; only a
right-censored pass uses the one robust fit that already selected its runway. Consumers
must not mistake an inferred event for a measured sample or refit the samples
independently. A changed estimator method therefore requires local reclassification of
these regenerable records from the stored samples; it does not require another OpenSky
download.

WHY ALL FOUR BUCKETS ARE WRITTEN
--------------------------------
Rejected tracks are not waste, they are the denominator. ``unassignable`` (the receiver
lost it) and ``not_landing`` (it never landed here) answer different questions from
``assigned``, and a rate computed without them is a rate computed over a selection.
The manifest carries the counts so no consumer has to glob a directory to find out --
globbing is how an earlier stage silently counted orphans from a previous run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from trajectory_data_process.harvest.classify import ClassifiedTrack

TRACKS_DIR = "tracks"
APPROACH_DIR = "approach"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_DIR = ".download-checkpoint"
TRACK_SCHEMA_VERSION = "harvest-tracks-v2-source-timing"

# What the harvest records about its own altitudes, so a consumer can never guess.
ALTITUDE_SOURCE = "opensky_history_geoaltitude_m"
ALTITUDE_DATUM = "hae"

_BUCKETS = ("assigned", "ambiguous", "unassignable", "not_landing")


@dataclass(frozen=True)
class HarvestPaths:
    """Where one airport's harvest lives."""

    root: Path
    code: str

    @property
    def airport(self) -> Path:
        return self.root / self.code

    @property
    def tracks(self) -> Path:
        return self.airport / TRACKS_DIR

    @property
    def approach(self) -> Path:
        return self.airport / APPROACH_DIR

    @property
    def manifest(self) -> Path:
        return self.tracks / MANIFEST_NAME

    @property
    def checkpoint(self) -> Path:
        return self.airport / CHECKPOINT_DIR

    @property
    def checkpoint_state(self) -> Path:
        return self.checkpoint / "state.json"

    @property
    def checkpoint_db(self) -> Path:
        return self.checkpoint / "history.sqlite"

    def bucket(self, outcome: str, runway: str | None = None) -> Path:
        return self.tracks / outcome / runway if runway else self.tracks / outcome

    def record(self, classified: ClassifiedTrack) -> Path:
        return self.bucket(classified.outcome, classified.runway) / f"{classified.flight_key}.json"


def track_record(classified: ClassifiedTrack) -> dict[str, Any]:
    """One measured track, serialised.

    Sample times are rebased to the track's own start (``start_time_utc`` carries the
    absolute reference) and follow the project's waypoint order
    ``[t_offset_s, lon, lat, alt_m]``.

    The assignment's ``scores`` ride along on EVERY outcome, including the rejections,
    so a disputed runway can be audited from the record alone -- without re-running the
    fit or, worse, re-downloading.
    """
    track = classified.track
    t0 = track.start_s
    return {
        "flight_key": classified.flight_key,
        "icao24": track.icao24,
        "callsign": track.callsign,
        "outcome": classified.outcome,
        "runway": classified.runway,
        "landing_time_utc": classified.landing_time_utc,
        "landing_sample_index": classified.landing_sample_index,
        "start_time_utc": _iso_precise(track.start_s),
        "duration_s": round(track.end_s - track.start_s, 3),
        "max_sample_gap_s": round(track.max_gap_s, 3),
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "source_integrity": (
            track.source_integrity.to_dict()
            if track.source_integrity is not None
            else None
        ),
        "reported_ground_speeds_m_s": [
            _round(sample.reported_ground_speed_m_s)
            for sample in track.samples
        ],
        "assignment": {
            "outcome": classified.outcome,
            "runway": classified.runway,
            "scores_m": {k: round(v, 2) for k, v in sorted(classified.assignment.scores.items())},
            "margin_m": _round(classified.assignment.margin_m),
            "reason": classified.assignment.reason,
        },
        "observed_threshold_event": classified.observed_threshold_event,
        "samples": [
            [round(s.time_s - t0, 3), round(s.lon, 6), round(s.lat, 6), round(s.alt_hae_m, 1)]
            for s in track.samples
        ],
    }


def write_tracks(
    classified: Iterable[ClassifiedTrack],
    paths: HarvestPaths,
    *,
    provenance: dict[str, Any],
    source_integrity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write every bucket plus the manifest; return the manifest.

    Existing bucket contents are cleared first. A harvest is a complete statement about
    one airport, and leaving a previous run's records behind is how a roster ends up
    describing flights that a later run no longer believes in.
    """
    _clear(paths.tracks)
    roster: list[dict[str, Any]] = []
    counts: dict[str, int] = {b: 0 for b in _BUCKETS}
    per_runway: dict[str, int] = {}
    seen_flight_keys: set[str] = set()
    source_integrity_complete = True

    for item in classified:
        if item.flight_key in seen_flight_keys:
            raise ValueError(
                f"duplicate flight_key {item.flight_key!r} produced during "
                "classification/reclassification; multiple source tracks resolve to "
                "the same current flight identity"
            )
        seen_flight_keys.add(item.flight_key)
        path = paths.record(item)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = track_record(item)
        source_integrity_complete = (
            source_integrity_complete and record["source_integrity"] is not None
        )
        path.write_text(
            json.dumps(record, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
        if item.runway:
            per_runway[item.runway] = per_runway.get(item.runway, 0) + 1
        roster.append(
            {
                "flight_key": item.flight_key,
                "file": str(path.relative_to(paths.tracks)),
                "outcome": item.outcome,
                "runway": item.runway,
                "icao24": item.track.icao24,
                "callsign": item.track.callsign,
                "landing_time_utc": item.landing_time_utc,
                "landing_sample_index": item.landing_sample_index,
                "event_status": item.observed_threshold_event.get("status"),
            }
        )

    manifest = {
        "schema_version": TRACK_SCHEMA_VERSION,
        "airport": paths.code,
        "written_utc": _iso(datetime.now(tz=timezone.utc).timestamp()),
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "counts": counts,
        "per_runway": dict(sorted(per_runway.items())),
        "total": len(roster),
        "source_integrity_complete": source_integrity_complete,
        "provenance": provenance,
        "records": roster,
    }
    if source_integrity is not None:
        manifest["source_integrity"] = source_integrity
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(
        json.dumps(manifest, indent=1, allow_nan=False), encoding="utf-8"
    )
    return manifest


def read_manifest(paths: HarvestPaths) -> dict[str, Any]:
    """Load the roster. Raises if absent -- there is no glob fallback, deliberately."""
    if not paths.manifest.exists():
        raise FileNotFoundError(
            f"{paths.manifest} is missing; a harvest directory is read through its manifest, "
            "never by globbing (globbing counts orphans from earlier runs)."
        )
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def require_source_timed_manifest(
    manifest: dict[str, Any], *, path: Path | None = None
) -> None:
    """Reject derived tracks that predate source freshness reconstruction."""
    if (
        manifest.get("schema_version") != TRACK_SCHEMA_VERSION
        or manifest.get("source_integrity_complete") is not True
    ):
        location = f"{path}: " if path is not None else ""
        raise ValueError(
            f"{location}tracks are not {TRACK_SCHEMA_VERSION}; rebuild them into a "
            "new staging root with --rebuild-fresh-from before creating arrivals "
            "or evaluation"
        )


def iter_records(paths: HarvestPaths, *, outcome: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield stored track records via the manifest roster, optionally one bucket."""
    for row in read_manifest(paths)["records"]:
        if outcome is not None and row["outcome"] != outcome:
            continue
        yield json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))


def _clear(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*.json"), reverse=True):
        path.unlink()
    for path in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True):
        path.rmdir()


def _iso(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_precise(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
