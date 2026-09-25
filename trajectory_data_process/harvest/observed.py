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
  3. **Target.** The published per-runway TCH from the CIFP, never a flat assumption:
     the LPV Path Point's, or the LNAV/VNAV runway leg's. A runway with neither (KRDU 14)
     has no TCH and is SKIPPED, loudly -- it has no published path to judge against.

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
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from flight_scenarios.build import resolve_airframe
from flight_scenarios.scenario import aircraft_dynamics_source
from flight_scenarios.crossing_span import CROSSING_SPAN_KEY, crossing_span_from_event
from flight_scenarios.datum import MSL_ALTITUDE_SOURCE, flight_to_msl
from flight_scenarios.start_state import state_samples_from_track

from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
)
from trajectory_data_process.harvest.arrivals import runway_target
from trajectory_data_process.harvest.altitude_filter import (
    DEFAULT_POLICY,
    FILTER_SCHEMA_VERSION,
)
from trajectory_data_process.harvest.store import (
    OUTCOME_NOT_LANDING,
    OUTCOMES,
    HarvestPaths,
    integrity_audits,
    read_manifest,
    read_track_view,
    require_source_timed_manifest,
)
from trajectory_data_process.harvest.staging import RECORDS_PREVIOUS_PREFIX, RECORDS_STAGING_PREFIX
from trajectory_data_process.harvest.threshold_event import require_current_threshold_event

# Fallback mass for the state samples when the airframe's type has no OpenAP/preset
# dynamics, or cannot be resolved from its icao24 at all. A resolved record carries
# its ICAO type (``aircraft_type``, the SAME identity chain the scenarios use) — the
# key the baseline's speed gate looks its PUBLISHED approach-speed window up by —
# and its type's landing mass when the dynamics exist, this nominal mass otherwise
# (``mass_source`` says which; the gate never reads the mass of an observed record).
# An UNRESOLVED identity gets no ``aircraft_type`` and grades speed-indeterminate,
# loudly: judging an unknown airframe against an invented window would be false
# precision.
NOMINAL_MASS_KG = 60_000.0
# The provider an observed record's airframe is resolved with, and its mass labelled by: one value for both.
AIRCRAFT_PROVIDER = "auto"

RECORDS_DIR = "records"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "evaluation_report.json"


@dataclass(frozen=True)
class SkippedTrack:
    flight_key: str
    reason: str


def observed_record(
    track: dict[str, Any], runway: Runway, *, mass_kg: float | None = None
) -> dict[str, Any]:
    """One stored track as an ``evaluation.records`` record, in MSL.

    Evaluation consumes the threshold event already produced by runway assignment;
    it does not refit these state samples. The ICAO type comes from the flight's own
    resolved identity (icao24 → identity, the same chain the scenarios use) and is
    the key the speed gate looks the published window up by; the mass is the type's
    OpenAP landing mass when those dynamics exist, else ``NOMINAL_MASS_KG``. An
    unresolvable identity falls back to the nominal mass with no ``aircraft_type``
    and grades speed-indeterminate. An explicit ``mass_kg`` bypasses resolution
    (tests, synthetic tracks) and writes no type.
    """
    if runway.threshold_crossing_height_m is None:
        raise ValueError(
            f"{runway.airport} {runway.ident} publishes no vertically guided RNAV approach"
        )
    aircraft_type: str | None = None
    mass_source = "explicit"
    if mass_kg is None:
        resolved = resolve_airframe(track.get("icao24"), aircraft_provider=AIRCRAFT_PROVIDER)
        if resolved is not None:
            mass_kg, aircraft_type = resolved
        if mass_kg is None:
            mass_kg, mass_source = NOMINAL_MASS_KG, "nominal"
        else:
            # Where the type's own landing mass came from: a preset, the performance index's
            # own-parameter row, or OpenAP (resolve_airframe resolves with the default provider).
            mass_source = aircraft_dynamics_source(aircraft_type, provider=AIRCRAFT_PROVIDER)
    event = track.get("observed_threshold_event")
    if not isinstance(event, dict):
        raise ValueError(
            f"track {track.get('flight_key')!r} lacks a threshold event "
            f"for runway {runway.ident}; run --reclassify-existing"
        )
    require_current_threshold_event(event, runway)

    # H_MSL = h_HAE - N, applied once, here, by the seam's own conversion (the runway's CIFP
    # offset, keyed on the track's altitude_source: an unknown or legacy tag is refused).
    waypoints = flight_to_msl({
        "id": track["flight_key"],
        "altitude_source": track["altitude_source"],
        "runway_target": runway_target(runway),
        "waypoints": track["samples"],
    })["waypoints"]
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
    # Target kinematics and final_time_s come from the last MEASURED row; the
    # crossing-span rows appended below are inferred data and must not move them.
    target = {
        "lat": runway.lat,
        "lon": runway.lon,
        "alt": runway.target_altitude("msl"),
        "V": states[-1]["V"],
        "psi": states[-1]["psi"],
        "gamma": states[-1]["gamma"],
        "m": mass_kg,
    }
    final_time_s = states[-1]["t"]
    # The record says WHERE its crossing lives (`crossing_span`), so evaluation
    # grades the states through one shared interpolation instead of re-reading the
    # event: a direct bracket marks its sample pair, a censored fit appends the one
    # inferred crossing row. `state_samples_from_track` yields one state per stored
    # sample, which is the 1:1 alignment the event's sample indices rely on.
    crossing_span = None
    if event.get("status") == "estimated":
        crossing_span, appended_rows = crossing_span_from_event(
            event, states, hae_minus_msl_m=runway.hae_minus_msl_m
        )
        states = states + appended_rows
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
            "tch_source": runway.tch_source,
            "baro_vnav_minima": runway.baro_vnav_minima,
            # The resolved airframe's ICAO type designator: the key the speed gate
            # looks its published approach speed up by (evaluation.speed_gate
            # TYPECODE_KEYS) — present whenever the identity resolved, dynamics or
            # not (see NOMINAL_MASS_KG above); mass_source says what the states' m is.
            **({"aircraft_type": aircraft_type} if aircraft_type is not None else {}),
            "mass_source": mass_source,
            "source_integrity": track.get("source_integrity"),
            # Copy policy-free producer output verbatim.  Benchmark selection and
            # limits remain evaluation-owned.
            "observed_threshold_event": event,
            **({CROSSING_SPAN_KEY: crossing_span} if crossing_span is not None else {}),
        },
        "initial_state": {k: v for k, v in states[0].items() if k != "t"},
        "target_state": target,
        "final_time_s": final_time_s,
        "states": states,
        "controls": [],
    }


def write_observed_records(
    airport: Airport, paths: HarvestPaths, *, mass_kg: float | None = None
) -> dict[str, Any]:
    """Build observed records for every assigned track; return the summary roster.

    Tracks on runways with no published vertical path are skipped and LISTED -- a bounded
    coverage that is stated in the output rather than silently shrinking the batch.

    Records are written into a staging directory and swapped in only once every one of
    them was built, so a failure leaves the previous ``records/`` + ``summary.json`` pair
    intact rather than a summary pointing at deleted records.
    """
    source = read_manifest(paths)
    require_source_timed_manifest(source, path=paths.manifest)
    availability = source_event_availability(source)
    records_dir = paths.approach / RECORDS_DIR
    staging = paths.approach / f"{RECORDS_STAGING_PREFIX}{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        summary = _write_observed_batch(
            airport, paths, source, availability, staging, mass_kg=mass_kg
        )
    except BaseException:
        shutil.rmtree(staging)
        raise
    summary_path = paths.approach / SUMMARY_NAME
    staged_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    staged_summary.write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    previous = paths.approach / f"{RECORDS_PREVIOUS_PREFIX}{uuid4().hex}"
    if records_dir.exists():
        records_dir.replace(previous)
    staging.replace(records_dir)
    staged_summary.replace(summary_path)
    if previous.exists():
        shutil.rmtree(previous)
    return summary


def _write_observed_batch(
    airport: Airport,
    paths: HarvestPaths,
    source: dict[str, Any],
    availability: dict[str, Any],
    records_dir: Path,
    *,
    mass_kg: float | None,
) -> dict[str, Any]:
    """Write every observed record into ``records_dir``; return the summary roster."""
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
                SkippedTrack(
                    row["flight_key"],
                    f"runway {runway.ident} publishes no vertically guided RNAV approach "
                    "(no LPV Path Point, no LNAV/VNAV final leg)",
                )
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
    return summary


def source_event_availability(source: dict[str, Any]) -> dict[str, Any]:
    """Availability over arrival candidates, before evaluation-record filtering.

    ``not_landing`` tracks are outside the population: classification determined that
    they are not arrivals at this airport. Assigned, ambiguous, and unassignable tracks
    are candidate arrivals and therefore all belong in the denominator, and so do the
    candidates a freshness rebuild excluded (``store.integrity_audits``). A roster built
    from harvests that did not all carry an exclusion audit says how many did not
    (``sources_without_integrity_audit``): its denominator covers the audited ones only.
    """
    statuses, excluded_outcomes, unaudited = _availability_inputs(source)
    candidates = [status for outcome, status in statuses if outcome != OUTCOME_NOT_LANDING]
    estimated = sum(status == "estimated" for status in candidates)
    excluded_not_landing = (
        sum(outcome == OUTCOME_NOT_LANDING for outcome, _ in statuses)
        + sum(outcome == OUTCOME_NOT_LANDING for outcome in excluded_outcomes)
    )
    integrity_excluded_candidates = sum(outcome != OUTCOME_NOT_LANDING for outcome in excluded_outcomes)
    denominator = len(candidates) + integrity_excluded_candidates
    return {
        "denominator": "arrival_candidates_excluding_not_landing",
        "event_denominator": denominator,
        "event_estimated": estimated,
        "event_unavailable": denominator - estimated,
        "event_estimated_rate": estimated / denominator if denominator else 0.0,
        "excluded_not_landing": excluded_not_landing,
        "source_integrity_excluded_candidates": integrity_excluded_candidates,
        "sources_without_integrity_audit": unaudited,
    }


# Arrival candidates: every roster outcome but ``not_landing``.
CANDIDATE_OUTCOMES = tuple(outcome for outcome in OUTCOMES if outcome != OUTCOME_NOT_LANDING)


def _availability_inputs(
    source: dict[str, Any],
) -> tuple[list[tuple[str, str | None]], list[str], int]:
    """The tracks roster read once, at this boundary: each record's ``(outcome, event_status)``,
    the outcome of every candidate an exclusion audit removed, and how many harvests carried no
    audit. The roster is a FILE (stale or hand-edited), so its shape is checked here and the
    derivation above only counts."""
    records = source.get("records")
    if not isinstance(records, list) or source.get("total") != len(records):
        raise ValueError("track manifest has an invalid records roster")
    statuses: list[tuple[str, str | None]] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"track manifest record {index} must be an object")
        outcome = row.get("outcome")
        if outcome == OUTCOME_NOT_LANDING:
            statuses.append((outcome, None))
            continue
        if outcome not in CANDIDATE_OUTCOMES:
            raise ValueError(f"track manifest record {index} has invalid outcome {outcome!r}")
        status = row.get("event_status")
        if status not in ("estimated", "unavailable"):
            raise ValueError(
                f"track manifest record {index} lacks event_status; run "
                "--reclassify-existing"
            )
        statuses.append((outcome, status))
    audits, unaudited = integrity_audits(source)
    excluded_outcomes: list[str] = []
    for integrity in audits:
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
            if outcome != OUTCOME_NOT_LANDING and outcome not in CANDIDATE_OUTCOMES:
                raise ValueError(
                    f"source_integrity exclusion {index} has invalid source_outcome "
                    f"{outcome!r}"
                )
            excluded_outcomes.append(outcome)
    return statuses, excluded_outcomes, unaudited



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
