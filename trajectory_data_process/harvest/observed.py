"""Harvested tracks -> observed evaluation records. The measured/derived boundary.

This is the one place the harvest crosses into the modeling plane, and it does exactly
three things, none of which belongs upstream of it:

  1. **Datum.** Track altitudes are HAE as broadcast; every gate, threshold elevation and
     CIFP altitude is MSL. ``H_MSL = h_HAE - N``. Skipping this scored real completed
     airline landings at 1.8% on the gates.
  2. **Velocity.** ``V / psi / gamma`` come from ``flight_scenarios.start_state``'s
     least-squares fit, imported rather than re-derived. That fit projects through the
     true tangent scales; a hand-rolled flat-chart version overstates the north component
     by 0.33%, which is a bug this project has already paid for once.
  3. **Target.** The published per-runway TCH from the CIFP, never a flat assumption. A
     runway with no LPV procedure has no TCH and is SKIPPED, loudly -- it cannot be
     judged against LPV gates at all.

Output lands in ``approach/``, apart from ``tracks/``, because this is an evaluation
view: it converts datum, derives kinematics, supplies the benchmark target, and carries
the already serialized threshold estimate into the evaluator.

The samples arrive through ``store.read_track_view``, so a state vector reporting an
unreachable altitude never reaches the least-squares velocity fit — one 20 km needle
inside the fit window moves ``V``/``gamma`` for the whole record. The batch summary
reports how many altitudes that filter replaced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from flight_scenarios.datum import MSL_ALTITUDE_SOURCE
from flight_scenarios.start_state import state_samples_from_track

from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
)
from trajectory_data_process.harvest.altitude_filter import (
    DEFAULT_POLICY,
    FILTER_SCHEMA_VERSION,
)
from trajectory_data_process.harvest.store import (
    HarvestPaths,
    read_manifest,
    read_track_view,
    require_source_timed_manifest,
)
from trajectory_data_process.harvest.threshold_event import require_current_threshold_event

# Nominal mass for the state samples. It reaches no gate -- the regulation checks are
# positional -- and an observed track carries no mass information, so inventing a
# per-aircraft value here would be false precision rather than accuracy.
NOMINAL_MASS_KG = 60_000.0

RECORDS_DIR = "records"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "evaluation_report.json"


@dataclass(frozen=True)
class SkippedTrack:
    flight_key: str
    reason: str


def observed_record(
    track: dict[str, Any], runway: Runway, *, mass_kg: float = NOMINAL_MASS_KG
) -> dict[str, Any]:
    """One stored track as an ``evaluation.records`` record, in MSL.

    Evaluation consumes the threshold event already produced by runway assignment;
    it does not refit these state samples.
    """
    if runway.threshold_crossing_height_m is None:
        raise ValueError(f"{runway.airport} {runway.ident} publishes no LPV TCH")
    event = track.get("observed_threshold_event")
    if not isinstance(event, dict):
        raise ValueError(
            f"track {track.get('flight_key')!r} lacks a threshold event "
            f"for runway {runway.ident}; run --reclassify-existing"
        )
    require_current_threshold_event(event, runway)

    # H_MSL = h_HAE - N, applied once, here.
    waypoints = [
        [t, lon, lat, alt_hae - runway.hae_minus_msl_m]
        for t, lon, lat, alt_hae in track["samples"]
    ]
    samples = state_samples_from_track(waypoints, mass_kg=mass_kg)
    states = [
        {
            "t": t,
            "lat": s.latitude,
            "lon": s.longitude,
            "alt": s.altitude,
            "V": s.V,
            "psi": s.psi,
            "gamma": s.gamma,
            "m": s.m,
        }
        for t, s in samples
    ]
    target = {
        "lat": runway.lat,
        "lon": runway.lon,
        "alt": runway.target_altitude("msl"),
        "V": states[-1]["V"],
        "psi": states[-1]["psi"],
        "gamma": states[-1]["gamma"],
        "m": mass_kg,
    }
    return {
        "source": {
            "id": track["callsign"] or track["icao24"],
            "subject": "observed",
            "arr_airport": runway.airport,
            "flight_key": track["flight_key"],
            "icao24": track["icao24"],
            "runway": runway.ident,
            "runway_course_deg": runway.course_deg,
            "threshold_crossing_height_m": runway.threshold_crossing_height_m,
            "published_glidepath_deg": runway.published_glidepath_deg,
            "landing_time_utc": track["landing_time_utc"],
            "altitude_source": MSL_ALTITUDE_SOURCE,
            "hae_minus_msl_m": runway.hae_minus_msl_m,
            "vertical_source": runway.vertical_source,
            "source_integrity": track.get("source_integrity"),
            # Copy policy-free producer output verbatim.  Benchmark selection and
            # limits remain evaluation-owned.
            "observed_threshold_event": event,
        },
        "initial_state": {k: v for k, v in states[0].items() if k != "t"},
        "target_state": target,
        "final_time_s": states[-1]["t"],
        "states": states,
        "controls": [],
    }


def write_observed_records(
    airport: Airport, paths: HarvestPaths, *, mass_kg: float = NOMINAL_MASS_KG
) -> dict[str, Any]:
    """Build observed records for every assigned track; return the summary roster.

    Tracks on runways with no published LPV TCH are skipped and LISTED -- a bounded
    coverage that is stated in the output rather than silently shrinking the batch.
    """
    source = read_manifest(paths)
    require_source_timed_manifest(source, path=paths.manifest)
    availability = source_event_availability(source)
    records_dir = paths.approach / RECORDS_DIR
    _clear(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)

    roster: list[dict[str, Any]] = []
    skipped: list[SkippedTrack] = []

    repaired_samples = 0
    repaired_records = 0

    for row in source["records"]:
        if row["outcome"] != "assigned":
            continue
        track = read_track_view(paths, row["file"])
        runway = airport.runway(row["runway"])
        if runway.threshold_crossing_height_m is None:
            skipped.append(
                SkippedTrack(row["flight_key"], f"runway {runway.ident} publishes no LPV TCH")
            )
            continue
        record = observed_record(track, runway, mass_kg=mass_kg)
        repaired = int(track["altitude_filter"]["outlier_count"])
        repaired_samples += repaired
        repaired_records += bool(repaired)
        name = f"{row['flight_key']}_eval.json"
        (records_dir / name).write_text(
            json.dumps(record, separators=(",", ":"), allow_nan=False), encoding="utf-8"
        )
        roster.append(
            {
                "flight_key": row["flight_key"],
                "eval_file": f"{RECORDS_DIR}/{name}",
                "runway": runway.ident,
                "icao24": row["icao24"],
                "landing_time_utc": row["landing_time_utc"],
            }
        )

    summary = {
        "airport": airport.code,
        "subject": "observed",
        "source_total": int(source["total"]),
        "source_counts": source["counts"],
        "event_availability": availability,
        "altitude_source": MSL_ALTITUDE_SOURCE,
        # A gate verdict is a claim about what was flown, so the batch states how much of
        # its altitude data was a repaired view rather than a raw reading.
        "altitude_filter": {
            "schema_version": FILTER_SCHEMA_VERSION,
            "policy": DEFAULT_POLICY.to_dict(),
            "repaired_samples": repaired_samples,
            "repaired_records": repaired_records,
        },
        "mass_kg": mass_kg,
        "total": len(roster),
        "skipped": [{"flight_key": s.flight_key, "reason": s.reason} for s in skipped],
        "results": roster,
    }
    (paths.approach / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8"
    )
    return summary


def source_event_availability(source: dict[str, Any]) -> dict[str, Any]:
    """Availability over arrival candidates, before evaluation-record filtering.

    ``not_landing`` tracks are outside the population: classification determined that
    they are not arrivals at this airport. Assigned, ambiguous, and unassignable tracks
    are candidate arrivals and therefore all belong in the denominator.
    """
    records = source.get("records")
    if not isinstance(records, list) or source.get("total") != len(records):
        raise ValueError("track manifest has an invalid records roster")
    candidates = []
    excluded_not_landing = 0
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"track manifest record {index} must be an object")
        outcome = row.get("outcome")
        if outcome == "not_landing":
            excluded_not_landing += 1
            continue
        if outcome not in ("assigned", "ambiguous", "unassignable"):
            raise ValueError(f"track manifest record {index} has invalid outcome {outcome!r}")
        status = row.get("event_status")
        if status not in ("estimated", "unavailable"):
            raise ValueError(
                f"track manifest record {index} lacks event_status; run "
                "--reclassify-existing"
            )
        candidates.append(status)
    estimated = sum(status == "estimated" for status in candidates)
    integrity_excluded_candidates = 0
    integrity = source.get("source_integrity")
    if integrity is not None:
        if not isinstance(integrity, dict) or not isinstance(
            integrity.get("excluded"), list
        ):
            raise ValueError("track manifest has invalid source_integrity exclusions")
        for index, excluded in enumerate(integrity["excluded"]):
            if not isinstance(excluded, dict):
                raise ValueError(
                    f"source_integrity exclusion {index} must be an object"
                )
            outcome = excluded.get("source_outcome")
            if outcome == "not_landing":
                excluded_not_landing += 1
            elif outcome in ("assigned", "ambiguous", "unassignable"):
                integrity_excluded_candidates += 1
            else:
                raise ValueError(
                    f"source_integrity exclusion {index} has invalid source_outcome "
                    f"{outcome!r}"
                )
    denominator = len(candidates) + integrity_excluded_candidates
    return {
        "denominator": "arrival_candidates_excluding_not_landing",
        "event_denominator": denominator,
        "event_estimated": estimated,
        "event_unavailable": denominator - estimated,
        "event_estimated_rate": estimated / denominator if denominator else 0.0,
        "excluded_not_landing": excluded_not_landing,
        "source_integrity_excluded_candidates": integrity_excluded_candidates,
    }


def iter_observed_records(paths: HarvestPaths) -> Iterator[Any]:
    """Yield observed ``TrajectoryRecord``s one at a time, via the batch roster.

    An airport such as KRDU has thousands of approach records and hundreds of
    megabytes of state samples.  Keeping this boundary lazy lets evaluation release
    each record before the next file is decoded.
    """
    from evaluation.records import record_from_dict

    summary = json.loads((paths.approach / SUMMARY_NAME).read_text(encoding="utf-8"))
    for row in summary["results"]:
        path = paths.approach / row["eval_file"]
        yield record_from_dict(json.loads(path.read_text(encoding="utf-8")), path=path)


def load_observed_records(paths: HarvestPaths) -> list[Any]:
    """Materialize the observed batch; prefer :func:`iter_observed_records` for evaluation."""
    return list(iter_observed_records(paths))


def _clear(directory: Path) -> None:
    if directory.exists():
        for path in directory.glob("*.json"):
            path.unlink()
