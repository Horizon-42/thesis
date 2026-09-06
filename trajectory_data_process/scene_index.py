"""A time index over one airport's stored tracks: who was airborne when.

The scene data plane (scene design §五 P2) asks, for an ego flight at its anchor time
t₀, which other tracks of the same harvest have samples in the window before t₀. The
tracks roster (``tracks/manifest.json``) carries no times, and the times live inside
66,942 track files (1.2 GB at KRDU), so this module reads them ONCE into a small index
— per record: ``flight_key``, outcome, runway, icao24, the track's first-sample UTC
(``start_time_utc``, milliseconds), its last-sample UTC (start + ``duration_s``), the
sample count, the landing UTC (assigned tracks only) and the file — and caches it
beside the manifest as ``tracks/scene_index.json`` under a contract: the schema string,
the manifest's SHA-256 and its record count. A cache whose contract does not match the
manifest on disk is rebuilt, never trusted; the index follows the roster and never
globs (repo invariant).

Queries: ``airborne_at(t0, window_s)`` — every entry with a sample inside
``[t0 − window_s, t0]`` (the entry's span overlaps the window; the caller then reads the
samples and applies t ≤ t0 to each) — and ``landings_before(t0)``, the assigned
landings already on the ground at t0, by runway. Both read the PAST only; a landing
time is future information for an aircraft still airborne at t0 and is exposed by
``future_label`` alone (see ``flight_scenarios.scene_context``).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from trajectory_data_process.harvest.store import HarvestPaths, read_manifest, require_source_timed_manifest

SCENE_INDEX_SCHEMA = "scene-index-v1"
INDEX_NAME = "scene_index.json"
OUTCOME_ASSIGNED = "assigned"       # mirror of harvest.classify's outcome name (a string in the roster)


def parse_utc_s(text: str) -> float:
    """Epoch seconds of an ISO-8601 UTC stamp (``…Z``), millisecond precision kept."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


@dataclass(frozen=True)
class IndexEntry:
    flight_key: str
    file: str
    outcome: str
    runway: str | None
    icao24: str
    callsign: str | None
    start_utc_s: float
    end_utc_s: float
    n_samples: int
    landing_utc_s: float | None


@dataclass(frozen=True)
class SceneIndex:
    airport: str
    manifest_sha256: str
    entries: tuple[IndexEntry, ...]

    def __post_init__(self) -> None:
        starts = np.array([e.start_utc_s for e in self.entries], dtype=np.float64)
        ends = np.array([e.end_utc_s for e in self.entries], dtype=np.float64)
        landings = sorted(
            (e.landing_utc_s, e.runway, i) for i, e in enumerate(self.entries)
            if e.outcome == OUTCOME_ASSIGNED and e.landing_utc_s is not None
        )
        object.__setattr__(self, "_starts", starts)
        object.__setattr__(self, "_ends", ends)
        object.__setattr__(self, "_by_key", {e.flight_key: i for i, e in enumerate(self.entries)})
        object.__setattr__(self, "_landing_times", [x[0] for x in landings])
        object.__setattr__(self, "_landings", landings)

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, flight_key: str) -> IndexEntry:
        return self.entries[self._by_key[flight_key]]

    def airborne_at(self, t0_utc_s: float, window_s: float) -> list[IndexEntry]:
        """Entries whose sampled span overlaps ``[t0 − window_s, t0]`` (a sample in the
        window is not guaranteed for a sparse track: the caller filters the samples)."""
        mask = (self._starts <= t0_utc_s) & (self._ends >= t0_utc_s - window_s)
        return [self.entries[i] for i in np.flatnonzero(mask)]

    def landings_before(self, t0_utc_s: float, *, since_s: float | None = None) -> list[tuple[float, str | None, IndexEntry]]:
        """Assigned landings with ``landing_utc_s ≤ t0`` (and ``> t0 − since_s`` when
        given), oldest first, as ``(landing_utc_s, runway, entry)``."""
        hi = bisect_right(self._landing_times, t0_utc_s)
        lo = 0 if since_s is None else bisect_right(self._landing_times, t0_utc_s - since_s)
        return [(t, runway, self.entries[i]) for t, runway, i in self._landings[lo:hi]]

    def to_payload(self, record_count: int) -> dict[str, Any]:
        return {
            "schema": SCENE_INDEX_SCHEMA, "airport": self.airport,
            "manifest_sha256": self.manifest_sha256, "record_count": record_count,
            "entries": [asdict(e) for e in self.entries],
        }


def _manifest_sha256(paths: HarvestPaths) -> str:
    return hashlib.sha256(paths.manifest.read_bytes()).hexdigest()


def _iter_track_times(paths: HarvestPaths, rows: list[dict[str, Any]]) -> Iterator[IndexEntry]:
    for row in rows:
        track = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
        start = parse_utc_s(track["start_time_utc"])
        landing = row.get("landing_time_utc")
        yield IndexEntry(
            flight_key=row["flight_key"], file=row["file"], outcome=row["outcome"],
            runway=row.get("runway"), icao24=row["icao24"], callsign=row.get("callsign"),
            start_utc_s=start, end_utc_s=start + float(track["duration_s"]),
            n_samples=len(track["samples"]),
            landing_utc_s=parse_utc_s(landing) if isinstance(landing, str) else None,
        )


def build_scene_index(paths: HarvestPaths, *, verbose: bool = True) -> SceneIndex:
    """Read every rostered track's timing once and write the cache."""
    manifest = read_manifest(paths)
    require_source_timed_manifest(manifest, path=paths.manifest)
    rows = manifest["records"]
    if verbose:
        print(f"  scene index: reading {len(rows)} track files under {paths.tracks}")
    index = SceneIndex(str(manifest["airport"]).strip().upper(), _manifest_sha256(paths), tuple(_iter_track_times(paths, rows)))
    (paths.tracks / INDEX_NAME).write_text(json.dumps(index.to_payload(len(rows))), encoding="utf-8")
    return index


def load_scene_index(paths: HarvestPaths, *, verbose: bool = True) -> SceneIndex:
    """The cached index when its contract matches the manifest on disk, else a rebuild."""
    cache = paths.tracks / INDEX_NAME
    manifest_sha = _manifest_sha256(paths)
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        record_count = len(read_manifest(paths)["records"])
        if (payload.get("schema") == SCENE_INDEX_SCHEMA and payload.get("manifest_sha256") == manifest_sha
                and payload.get("record_count") == record_count == len(payload.get("entries", []))):
            return SceneIndex(str(payload["airport"]), manifest_sha, tuple(IndexEntry(**e) for e in payload["entries"]))
        if verbose:
            print(f"  scene index: {cache} does not match the manifest (schema / sha / count); rebuilding")
    return build_scene_index(paths, verbose=verbose)
