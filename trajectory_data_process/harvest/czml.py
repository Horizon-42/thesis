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
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from final_approach import RunwayFrame, TrackPoint, fit_final_segment
from flight_scenarios.identity import flight_key
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATOR = REPO_ROOT / "aeroviz-4d" / "python" / "generate_czml.py"
DEFAULT_FRONTEND_DATA = REPO_ROOT / "aeroviz-4d" / "public" / "data"
EVALUATION_REPORT_NAME = "evaluation_report.json"
EVALUATION_RECORDS_DIR = "records"


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
    extrapolated_records = _extrapolated_record_paths(paths)

    combined = airport_dir / "trajectories.czml"
    runway_counts: dict[str, int] = {}
    flights = 0
    max_offset = 0.0
    with tempfile.TemporaryDirectory(prefix=f"{paths.code}-observed-czml-") as work:
        source = Path(work) / "flights.jsonl"
        with source.open("w", encoding="utf-8") as output:
            for row in read_manifest(paths)["records"]:
                if row["outcome"] != "assigned":
                    continue
                track = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
                flight = czml_input_flight(track)
                verify_identity(flight, track["flight_key"])
                record_path = extrapolated_records.get(track["flight_key"])
                if record_path is not None:
                    segment = _extrapolated_waypoints(record_path)
                    if segment is not None:
                        # Kept separate from measured waypoints: this is an inferred tail.
                        flight["extrapolated_waypoints"] = segment
                output.write(
                    json.dumps(flight, ensure_ascii=False, separators=(",", ":")) + "\n"
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
                runway = str(row["runway"])
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
        ),
        encoding="utf-8",
    )
    return RenderedObserved(
        combined_czml=combined,
        runway_counts=runway_counts,
        manifest=manifest_path,
        flights=flights,
    )


def _extrapolated_record_paths(paths: HarvestPaths) -> dict[str, Path]:
    """Evaluation records whose verdict actually used an extrapolated crossing.

    The report is the authority for whether ``final_approach`` accepted the segment as
    established. Re-fitting every assigned track without this guard would draw an inferred
    tail even for flights the evaluation explicitly classified as ``not_established``.
    """
    report_path = paths.approach / EVALUATION_REPORT_NAME
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records_dir = paths.approach / EVALUATION_RECORDS_DIR
    return {
        row["flight_key"]: records_dir / Path(row["file"]).name
        for row in report.get("trajectories", [])
        if row.get("extrapolated") is True
        and isinstance(row.get("flight_key"), str)
        and isinstance(row.get("file"), str)
    }


def _extrapolated_waypoints(record_path: Path) -> list[list[float]] | None:
    """The fitted final-approach line from its last fit sample to the threshold.

    Evaluation records are MSL. The returned CZML-input points are converted back to HAE,
    the altitude datum Cesium expects, using the record's stamped geoid undulation. Both
    endpoints lie on the same two OLS lines used by ``evaluation.arrival``; no independent
    visual-only extrapolator is introduced here.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    states = record.get("states", [])
    target = record.get("target_state") or {}
    source = record.get("source") or {}
    course = source.get("runway_course_deg")
    if len(states) < 2 or course is None:
        return None

    frame = RunwayFrame(
        ident=str(source.get("runway", "?")),
        lat=float(target["lat"]),
        lon=float(target["lon"]),
        elevation_m=float(target["alt"]),
        course_deg=float(course),
    )
    points = [TrackPoint(float(s["lat"]), float(s["lon"]), float(s["alt"])) for s in states]
    fit = fit_final_segment(points, frame)
    if fit is None:
        return None

    projected = frame.project_all(points)
    start_index = min(
        range(len(projected)),
        key=lambda index: abs(projected[index].along_m - fit.nearest_sample_along_m),
    )
    start_t = float(states[start_index]["t"])
    speed_ms = float(states[start_index].get("V") or 0.0)
    if not math.isfinite(speed_ms) or speed_ms <= 1.0:
        speed_ms = 70.0
    end_t = start_t + fit.extrapolation_m / speed_ms

    if source.get("hae_minus_msl_m") is None:
        raise ValueError("record lacks CIFP hae_minus_msl_m; regenerate legacy artifact")
    geoid_m = float(source["hae_minus_msl_m"])
    start_along = fit.nearest_sample_along_m
    start = _fitted_point(
        frame,
        along_m=start_along,
        cross_m=fit.cross.intercept + fit.cross.slope * start_along,
        height_m=fit.height.intercept + fit.height.slope * start_along,
        geoid_m=geoid_m,
    )
    crossing = _fitted_point(
        frame,
        along_m=0.0,
        cross_m=fit.cross_at_threshold_m,
        height_m=fit.height_at_threshold_m,
        geoid_m=geoid_m,
    )
    return [
        [round(start_t, 3), *start],
        [round(end_t, 3), *crossing],
    ]


def _fitted_point(
    frame: RunwayFrame,
    *,
    along_m: float,
    cross_m: float,
    height_m: float,
    geoid_m: float,
) -> list[float]:
    """Inverse of ``RunwayFrame.project`` for one fitted point, returned as lon/lat/HAE."""
    course = math.radians(frame.course_deg)
    east_hat = math.sin(course)
    north_hat = math.cos(course)
    east_m = along_m * east_hat + cross_m * north_hat
    north_m = along_m * north_hat - cross_m * east_hat
    lon = frame.lon + east_m / metres_per_deg_lon(frame.lat)
    lat = frame.lat + north_m / METRES_PER_DEG_LAT
    altitude_hae_m = frame.elevation_m + height_m + geoid_m
    return [round(lon, 7), round(lat, 7), round(altitude_hae_m, 2)]


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
