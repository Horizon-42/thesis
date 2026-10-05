"""Reference evaluation records: each scenario's OBSERVED track in the evaluation contract.

One reference record per scenario, beside the optimizer's records, so the evaluation compares the
optimized and the observed flight on the same target; the observed tracks live once in a store
shared by the target datasets (``optimization-references-v3-shared-tracks``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flight_scenarios import (
    FlightScenario,
    flight_key,
    load_model_arrivals,
    state_samples_from_track,
)
from evaluation_export import (
    OBSERVED_TRACKS_DIR,
    OBSERVED_TRACK_SUFFIX,
    REFERENCE_CACHE_SCHEMA,
    REFERENCE_EVAL_SUFFIX as _REFERENCE_EVAL_SUFFIX,
    REFERENCES_DIR,
    file_sha256 as _file_sha256,
    observed_track_document,
    observed_track_path as _observed_track_path,
    reference_evaluation_record,
)
from scenario_batch import observed_track_filename, reference_filename, scenario_filename


# The reference cache contract (REFERENCE_CACHE_SCHEMA) and its integrity primitives
# (file_sha256, observed_track_path) are single-sourced in evaluation_export.py — imported
# above, shared with the pipeline runner's reuse validator.


def _cached_track_matches(reference_path: Path, expected_sha256: str) -> bool:
    track = _observed_track_path(reference_path)
    return track.is_file() and _file_sha256(track) == expected_sha256


def _sweep_observed_tracks(tracks_dir: Path) -> None:
    """Drop track files no sibling reference directory quotes any more.

    The store is shared by every target dataset under the same anchor, so it must NOT be
    swept against one dataset's roster — that would delete the other's tracks. The keep set
    is the union over all sibling ``*_reference_eval.json`` names, which is exactly what a
    reader can still resolve.
    """
    if not tracks_dir.is_dir():
        return
    keep = {
        path.name.removesuffix(_REFERENCE_EVAL_SUFFIX) + OBSERVED_TRACK_SUFFIX
        for sibling in tracks_dir.parent.iterdir()
        if sibling.is_dir() and sibling != tracks_dir
        for path in sibling.glob(f"*{_REFERENCE_EVAL_SUFFIX}")
    }
    stale = [
        path for path in sorted(tracks_dir.glob(f"*{OBSERVED_TRACK_SUFFIX}"))
        if path.name not in keep
    ]
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} unreferenced observed track(s) from {tracks_dir}")


def write_reference_records(
    scenarios: list[FlightScenario],
    observed_tracks: str | Path | list[dict[str, Any]],
    *,
    output_dir: str | Path,
    references_dir: str = REFERENCES_DIR,
    source_signature: dict[str, Any] | None = None,
) -> list[Path]:
    """One reference eval record per scenario, from its OBSERVED track.

    ``observed_tracks`` is the harvest arrival manifest the scenarios came from.
    Each scenario's flight is looked up in it by its ``flight_key``
    (``flight_scenarios.identity``) — the same key the output filenames are. The track becomes a reference record in the
    evaluation contract (per-sample kinematics via
    ``flight_scenarios.state_samples_from_track``, EMPTY controls, the SAME target
    the optimizer flies to), written under ``<output_dir>/<references_dir>/`` and
    named by the scenario's identity — so the batch can point every eval record at
    its reference deterministically (``reference_file``). Missing flights raise:
    references and solves must come from the same dataset.
    """
    out = (Path(output_dir) / references_dir).resolve()
    expected = [
        out / reference_filename(scenario_filename(scenario, index))
        for index, scenario in enumerate(scenarios)
    ]
    expected_cache_rows = [
        {
            "file": path.name,
            "identity": {
                "flight_key": flight_key(scenario.source, index),
                **{
                    key: scenario.source.get(key)
                    for key in ("id", "runway", "icao24", "landing_time_utc")
                },
            },
        }
        for index, (path, scenario) in enumerate(zip(expected, scenarios))
    ]
    cache_manifest = out / "manifest.json"
    if source_signature is not None and cache_manifest.exists():
        try:
            cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        cached_rows = cached.get("records")
        cache_matches = (
            cached.get("schema_version") == REFERENCE_CACHE_SCHEMA
            and cached.get("source_signature") == source_signature
            and isinstance(cached_rows, list)
            and len(cached_rows) == len(expected_cache_rows)
        )
        if cache_matches:
            for path, expected_row, cached_row in zip(
                expected, expected_cache_rows, cached_rows
            ):
                if (
                    not isinstance(cached_row, dict)
                    or cached_row.get("file") != expected_row["file"]
                    or cached_row.get("identity") != expected_row["identity"]
                    or not isinstance(cached_row.get("sha256"), str)
                    or not isinstance(cached_row.get("track_sha256"), str)
                    or not path.is_file()
                    or _file_sha256(path) != cached_row["sha256"]
                    or not _cached_track_matches(path, cached_row["track_sha256"])
                ):
                    cache_matches = False
                    break
        if cache_matches:
            print(f"✓ reusing {len(expected)} canonical reference record(s) -> {out}")
            return expected

    # Through the SAME loader the scenarios came from: it converts the observed altitudes
    # from ellipsoidal (HAE) to MSL. Reading the file directly here would put the reference
    # record ~30 m below the scenario built from the identical track.
    flights = load_model_arrivals(observed_tracks)
    by_key = {flight_key(f, 0): f for f in flights}
    out.mkdir(parents=True, exist_ok=True)
    # Fresh reference set = fresh directory: references from an earlier run over a
    # different flight set would otherwise accumulate (same stale-record class as
    # _clear_stale_records — dormant, but unbounded growth).
    stale = sorted(out.glob(f"*{_REFERENCE_EVAL_SUFFIX}"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} reference record(s) from a previous run in {out}")
    # The observed track store is a SIBLING of the per-target reference dirs, so the
    # fitted-ADS-B and runway datasets — which reference the same flights and differ only
    # in target_state — quote one copy of each track instead of two.
    tracks_dir = out.parent / OBSERVED_TRACKS_DIR
    tracks_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    track_paths: list[Path] = []
    for index, scenario in enumerate(scenarios):
        src = scenario.source
        if scenario.target is None:
            raise ValueError(
                f"scenario {src.get('id')!r} has no target state; build scenarios with "
                "flight_scenarios (its build_scenario populates target) first."
            )
        key = flight_key(src, index)
        flight = by_key.get(key)
        if flight is None:
            raise ValueError(f"no observed flight in the reference-tracks file for flight_key {key!r}")
        timed_states = state_samples_from_track(
            flight["waypoints"], mass_kg=scenario.initial.m,
            window_s=float(src["window_s"]),
        )
        states_name = scenario_filename(scenario, index)
        track_path = tracks_dir / observed_track_filename(states_name)
        track_path.write_text(
            json.dumps(observed_track_document(timed_states),
                       separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        track_paths.append(track_path)
        record = reference_evaluation_record(
            scenario.initial, scenario.target, timed_states, src, subject="observed",
            track_ref=f"../{OBSERVED_TRACKS_DIR}/{track_path.name}",
        )
        path = out / reference_filename(states_name)
        path.write_text(
            json.dumps(record, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        written.append(path)
    if source_signature is not None:
        records = [
            {**row, "sha256": _file_sha256(path), "track_sha256": _file_sha256(track)}
            for row, path, track in zip(expected_cache_rows, written, track_paths)
        ]
        manifest_payload = {
            "schema_version": REFERENCE_CACHE_SCHEMA,
            "source_signature": source_signature,
            "records": records,
        }
        temporary = cache_manifest.with_name(f".{cache_manifest.name}.tmp")
        temporary.write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8",
        )
        temporary.replace(cache_manifest)
    _sweep_observed_tracks(tracks_dir)
    print(f"✓ wrote {len(written)} reference record(s) -> {out}")
    print(f"  observed tracks -> {tracks_dir} ({len(track_paths)} shared)")
    return written


def remove_legacy_category_references(output_dir: str | Path) -> None:
    """Drop only the old per-category reference copies after moving to a sibling anchor."""
    legacy = Path(output_dir) / REFERENCES_DIR
    if not legacy.is_dir():
        return
    removed = 0
    for path in legacy.glob(f"*{_REFERENCE_EVAL_SUFFIX}"):
        path.unlink()
        removed += 1
    (legacy / "manifest.json").unlink(missing_ok=True)
    try:
        legacy.rmdir()
    except OSError:
        return
    if removed:
        print(f"… removed {removed} superseded per-category reference copy/copies -> {legacy}")
