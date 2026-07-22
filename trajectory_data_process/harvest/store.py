"""The harvest's on-disk layout: measured tracks and derived fits, kept apart.

    outputs/harvest/<ICAO>/
        tracks/                     <- MEASURED. Reconstructed samples, HAE, unfitted.
            assigned/<RWY>/<flight_key>.json
            ambiguous/<flight_key>.json
            unassignable/<flight_key>.json
            not_landing/<flight_key>.json
            manifest.json
        approach/                   <- DERIVED. Fits, MSL, written by the arrival stage.
            fits/<flight_key>.json
            summary.json

WHY THE SPLIT IS PHYSICAL AND NOT COSMETIC
------------------------------------------
``tracks/`` is what the sensors said: ellipsoidal altitudes exactly as broadcast, no
model, no extrapolation, no datum conversion. ``approach/`` is what a fit INFERRED:
MSL, a straight-line model, a crossing the receivers never saw. Mixing them in one
directory invites the mistake of reading an inferred crossing as a measurement -- and
the whole reason this pipeline exists is that the last measured sample and the actual
crossing are 325 m apart.

It also means the expensive half can be recomputed. Changing the fit window, the
established criteria, or the TCH source rewrites ``approach/`` only; ``tracks/`` is
re-derived solely by re-downloading.

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
        "start_time_utc": _iso(track.start_s),
        "duration_s": round(track.end_s - track.start_s, 3),
        "max_sample_gap_s": round(track.max_gap_s, 3),
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "assignment": {
            "outcome": classified.outcome,
            "runway": classified.runway,
            "scores_m": {k: round(v, 2) for k, v in sorted(classified.assignment.scores.items())},
            "margin_m": _round(classified.assignment.margin_m),
            "reason": classified.assignment.reason,
        },
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

    for item in classified:
        path = paths.record(item)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = track_record(item)
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
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
            }
        )

    manifest = {
        "airport": paths.code,
        "written_utc": _iso(datetime.now(tz=timezone.utc).timestamp()),
        "altitude_source": ALTITUDE_SOURCE,
        "altitude_datum": ALTITUDE_DATUM,
        "counts": counts,
        "per_runway": dict(sorted(per_runway.items())),
        "total": len(roster),
        "provenance": provenance,
        "records": roster,
    }
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def read_manifest(paths: HarvestPaths) -> dict[str, Any]:
    """Load the roster. Raises if absent -- there is no glob fallback, deliberately."""
    if not paths.manifest.exists():
        raise FileNotFoundError(
            f"{paths.manifest} is missing; a harvest directory is read through its manifest, "
            "never by globbing (globbing counts orphans from earlier runs)."
        )
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


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


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
