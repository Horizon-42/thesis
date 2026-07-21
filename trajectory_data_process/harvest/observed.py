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

Output lands in ``approach/``, apart from ``tracks/``, because everything here is
inferred: an MSL altitude that was never measured, a velocity that was never broadcast,
and downstream a crossing the receivers never saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from flight_scenarios.datum import MSL_ALTITUDE_SOURCE
from flight_scenarios.start_state import state_samples_from_track

from trajectory_data_process.harvest.airports import Airport, Runway
from trajectory_data_process.harvest.store import HarvestPaths, read_manifest

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

    ``source.subject = "observed"`` is what makes ``evaluation.arrival`` measure this at
    its fitted threshold crossing rather than at ``states[-1]`` -- see that module for
    why the two are 325 m apart. ``source.runway_course_deg`` is stamped because
    ``evaluation`` is geokit+stdlib only and cannot read a runway config.
    """
    if runway.threshold_crossing_height_m is None:
        raise ValueError(f"{runway.airport} {runway.ident} publishes no LPV TCH")

    # H_MSL = h_HAE - N, applied once, here.
    waypoints = [
        [t, lon, lat, alt_hae - runway.geoid_undulation_m]
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
            "flight_key": track["flight_key"],
            "icao24": track["icao24"],
            "runway": runway.ident,
            "runway_course_deg": runway.course_deg,
            "threshold_crossing_height_m": runway.threshold_crossing_height_m,
            "published_glidepath_deg": runway.published_glidepath_deg,
            "landing_time_utc": track["landing_time_utc"],
            "altitude_source": MSL_ALTITUDE_SOURCE,
            "geoid_undulation_m": runway.geoid_undulation_m,
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
    records_dir = paths.approach / RECORDS_DIR
    _clear(records_dir)
    records_dir.mkdir(parents=True, exist_ok=True)

    roster: list[dict[str, Any]] = []
    skipped: list[SkippedTrack] = []

    for row in read_manifest(paths)["records"]:
        if row["outcome"] != "assigned":
            continue
        track = json.loads((paths.tracks / row["file"]).read_text(encoding="utf-8"))
        runway = airport.runway(row["runway"])
        if runway.threshold_crossing_height_m is None:
            skipped.append(
                SkippedTrack(row["flight_key"], f"runway {runway.ident} publishes no LPV TCH")
            )
            continue
        record = observed_record(track, runway, mass_kg=mass_kg)
        name = f"{row['flight_key']}_eval.json"
        (records_dir / name).write_text(json.dumps(record, indent=1), encoding="utf-8")
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
        "altitude_source": MSL_ALTITUDE_SOURCE,
        "mass_kg": mass_kg,
        "total": len(roster),
        "skipped": [{"flight_key": s.flight_key, "reason": s.reason} for s in skipped],
        "results": roster,
    }
    (paths.approach / SUMMARY_NAME).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_observed_records(paths: HarvestPaths) -> list[Any]:
    """Read the observed batch back as ``TrajectoryRecord``s, via its roster."""
    from evaluation.records import record_from_dict

    summary = json.loads((paths.approach / SUMMARY_NAME).read_text(encoding="utf-8"))
    records = []
    for row in summary["results"]:
        path = paths.approach / row["eval_file"]
        records.append(record_from_dict(json.loads(path.read_text(encoding="utf-8")), path=path))
    return records


def _clear(directory: Path) -> None:
    if directory.exists():
        for path in directory.glob("*.json"):
            path.unlink()
