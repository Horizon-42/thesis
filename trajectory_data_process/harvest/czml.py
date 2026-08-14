"""Render the harvested tracks as the viewer's observed layer.

Closes the loop the verdict colouring needs: the tracks on screen and the verdicts
painted onto them must come from ONE harvest. While the observed CZML was still built
by the old pipeline, any flight the new assignment moved to a different runway got a
different ``flight_key`` and simply failed to match — the colour join silently found
nothing for it.

WHAT IS WRITTEN (the layout the frontend already reads)::

    public/data/airports/<ICAO>/
        trajectories.czml               canonical entities for every runway
        landings/index.json             runway counts + filters into that one file

ONLY THE ``assigned`` BUCKET IS RENDERED. ``not_landing`` is not an approach to this
airport; ``ambiguous`` and ``unassignable`` have no runway to file under, and drawing
them would put tracks on the map that no verdict can ever refer to. They stay in
``tracks/`` where the manifest counts them.

ALTITUDES STAY HAE. Cesium consumes CZML positions as heights above the WGS84
ellipsoid (``src/types/czml.d.ts``), and the stored tracks are HAE as broadcast, so
this stage converts nothing. The MSL conversion belongs to ``observed.py``, which
feeds the modeling plane — converting here would push the viewer 33 m off.

RENDERING IS DELEGATED, not reimplemented: the CZML is built by
``aeroviz-4d/python/generate_czml.py`` exactly as the old pipeline built it, so entity
ids, packet shape and clock handling cannot drift between the two. It is invoked as a
subprocess because that tree is standalone frontend tooling and must not be imported
from the modeling side.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flight_scenarios.identity import flight_key
from geokit import haversine_m

from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATOR = REPO_ROOT / "aeroviz-4d" / "python" / "generate_czml.py"
DEFAULT_FRONTEND_DATA = REPO_ROOT / "aeroviz-4d" / "public" / "data"


@dataclass(frozen=True)
class RenderedObserved:
    """What the render produced."""

    combined_czml: Path
    runway_counts: dict[str, int]
    manifest: Path
    flights: int


def czml_input_flight(track: dict[str, Any]) -> dict[str, Any]:
    """One stored track in the czml-input shape ``generate_czml`` consumes.

    ``id`` is written as the same truncated callsign the stored ``flight_key`` was
    derived from, because ``generate_czml`` re-derives the entity id from these fields.
    ``verify_identity`` checks that the round trip lands on the stored key rather than
    trusting it: a mismatch here would produce a viewer whose entity ids no verdict can
    address, and it would look like an empty report rather than an error.
    """
    return {
        "id": (track["callsign"] or track["icao24"]).replace(" ", "")[:16],
        "callsign": track["callsign"],
        "type": "UNK",
        "icao24": track["icao24"],
        "dep_airport": None,
        "arr_airport": None,
        "runway": track["runway"],
        "landing_time_utc": track["landing_time_utc"],
        "altitude_source": track["altitude_source"],
        "waypoints": track["samples"],
    }


def verify_identity(flight: dict[str, Any], expected_key: str) -> None:
    """Fail loudly when the rendered entity id would not match the stored flight key."""
    derived = flight_key(flight, index=0)
    if derived != expected_key:
        raise ValueError(
            f"entity id would be {derived!r} but the track is stored as {expected_key!r}; "
            "the verdict join is keyed on this and would silently match nothing"
        )


def observed_czml_flights(
    paths: HarvestPaths,
    rows: Iterable[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield renderer-ready flight dictionaries for selected manifest rows.

    This is the shared measured/derived join used by both the canonical batch
    publisher and the HTTP trajectory sampler. Keeping it here guarantees that
    an API-selected flight has the same identity, HAE positions, and optional
    fitted threshold extension as the full ``trajectories.czml`` artifact.
    """
    source_rows = read_manifest(paths)["records"] if rows is None else rows
    for row in source_rows:
        if row["outcome"] != "assigned":
            continue
        track = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
        flight = czml_input_flight(track)
        verify_identity(flight, track["flight_key"])
        segment = _extrapolated_waypoints(track)
        if segment is not None:
            # Kept separate from measured waypoints: this is an inferred tail.
            flight["extrapolated_waypoints"] = segment
        yield flight


def render_observed_czml(
    paths: HarvestPaths,
    *,
    frontend_data_root: Path = DEFAULT_FRONTEND_DATA,
    generator: Path = DEFAULT_GENERATOR,
    multiplier: int | None = None,
) -> RenderedObserved:
    """Write one canonical observed CZML plus its runway-filter manifest."""
    airport_dir = frontend_data_root / "airports" / paths.code
    landings_dir = airport_dir / "landings"
    landings_dir.mkdir(parents=True, exist_ok=True)

    combined = airport_dir / "trajectories.czml"
    runway_counts: dict[str, int] = {}
    flights = 0
    max_offset = 0.0
    with tempfile.TemporaryDirectory(prefix=f"{paths.code}-observed-czml-") as work:
        source = Path(work) / "flights.jsonl"
        with source.open("w", encoding="utf-8") as output:
            for flight in observed_czml_flights(paths):
                output.write(
                    json.dumps(
                        flight,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ) + "\n"
                )
                max_offset = max(
                    max_offset,
                    max(
                        (
                            float(waypoint[0])
                            for key in ("waypoints", "extrapolated_waypoints")
                            for waypoint in flight.get(key, [])
                        ),
                        default=0.0,
                    ),
                )
                runway = str(flight["runway"])
                runway_counts[runway] = runway_counts.get(runway, 0) + 1
                flights += 1

        if flights == 0:
            raise ValueError(
                f"{paths.code}: no assigned tracks to render — harvest first, or check "
                "the manifest's counts"
            )
        _generate(
            generator,
            paths.code,
            source,
            combined,
            multiplier,
            input_jsonl=True,
            max_offset=max_offset,
        )

    # Remove the superseded physical partitions and the old persistent renderer inputs.
    # The runway selector now filters entities in the canonical airport-wide file.
    for old in landings_dir.glob("*.czml"):
        old.unlink()
    _remove_legacy_workdir(paths.approach / "_czml_input")

    manifest_rows = [
        {"runway": runway, "file": "trajectories.czml", "count": count}
        for runway, count in sorted(runway_counts.items())
    ]

    manifest_path = landings_dir / "index.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schemaVersion": "observed-landings-v2-canonical",
                "airport": paths.code,
                "combined": "trajectories.czml",
                "runways": sorted(manifest_rows, key=lambda r: r["runway"]),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return RenderedObserved(
        combined_czml=combined,
        runway_counts=runway_counts,
        manifest=manifest_path,
        flights=flights,
    )


def _extrapolated_waypoints(track: dict[str, Any]) -> list[list[float]] | None:
    """Render the stored threshold estimate; never refit the measured track."""
    samples = track.get("samples")
    event = track.get("observed_threshold_event")
    if not isinstance(samples, list) or len(samples) < 2 or not isinstance(event, dict):
        return None
    if event.get("status") != "estimated" or event.get("runway") != track.get("runway"):
        return None
    if event.get("schema_version") != "observed-threshold-event-v6" \
            or event.get("method_version") != 6:
        raise ValueError(
            f"track {track.get('flight_key')!r} has an obsolete threshold event; "
            "run --reclassify-existing"
        )
    if event.get("method") != "final_segment_robust_fit":
        raise ValueError(
            f"track {track.get('flight_key')!r} has unsupported threshold-event method "
            f"{event.get('method')!r}"
        )
    if isinstance(event.get("threshold_bracket"), dict):
        # The measured track already contains both sides of its selected physical
        # crossing.  The bracket anchors the fit but is not itself the event estimate;
        # drawing an extra fitted tail would still duplicate measured geometry.
        return None
    fit_range = event.get("source_sample_range")
    anchor = track.get("landing_sample_index")
    if (
        not isinstance(fit_range, list) or len(fit_range) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in fit_range)
        or not isinstance(anchor, int) or isinstance(anchor, bool)
        or not (0 <= fit_range[0] <= fit_range[1] <= anchor < len(samples))
    ):
        raise ValueError(
            f"track {track.get('flight_key')!r} has invalid event source range or "
            "landing_sample_index; run --reclassify-existing"
        )
    start_index = fit_range[1]
    start = samples[start_index]
    previous = samples[start_index - 1] if start_index > 0 else samples[start_index]
    dt = float(start[0]) - float(previous[0])
    distance = haversine_m(float(previous[2]), float(previous[1]), float(start[2]), float(start[1]))
    speed_ms = distance / dt if dt > 0.0 and distance > 0.0 else 70.0
    extrapolation = float(event["extrapolation_m"])
    end_t = float(start[0]) + extrapolation / speed_ms
    crossing = [
        round(float(event["threshold_crossing_lon"]), 7),
        round(float(event["threshold_crossing_lat"]), 7),
        round(float(event["threshold_crossing_altitude_m"]), 2),
    ]
    return [
        [round(float(start[0]), 3), round(float(start[1]), 7),
         round(float(start[2]), 7), round(float(start[3]), 2)],
        [round(end_t, 3), *crossing],
    ]


def _generate(
    generator: Path,
    code: str,
    source: Path,
    output: Path,
    multiplier: int | None,
    *,
    input_jsonl: bool = False,
    max_offset: float | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(generator),
        "--airport", code,
        "--input-jsonl" if input_jsonl else "--input", str(source),
        "--output", str(output),
    ]
    if multiplier is not None:
        cmd += ["--multiplier", str(multiplier)]
    if max_offset is not None:
        cmd += ["--max-offset", str(max_offset)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _remove_legacy_workdir(directory: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    directory.rmdir()
