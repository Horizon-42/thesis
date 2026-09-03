#!/usr/bin/env python
"""Runway-hypothesis expansion: one threshold-anchored forecast per candidate runway.

The airport-frame ablation (``4dTrajectory/ts_transformer/docs/
2026-09-03_airport_frame_ablation_results.md``) showed that the chart's threshold anchor IS
the model's runway knowledge: take it away and the deterministic predictor averages
across each parallel pair. The complementary question is what that knowledge is WORTH —
how much of the baseline's error is runway misassignment, and whether the choice can be
made outside the predictor. This script answers it without training anything:

    for every validation flight, for every runway with a published CIFP target
        clone the flight dict with that runway's target        (the data source is untouched)
        build the series in THAT threshold's chart              (same build_series as train)
        forecast with the trained checkpoint                    (same forecast as predict)
        map the forecast back to world coordinates and score it against the observed
        track in the TRUE runway's chart                        (same metrics as predict)

and then evaluates selection rules over the K hypotheses per flight — the assigned label
(which must reproduce the baseline's numbers exactly), the oracle (min over K, an upper
bound), and causal rules that never read the future: the active configuration from
co-temporal landings, a course gate, and the forecast's own self-consistency (how close
it gets to the runway it was told about).

Development scope: the checkpoint's validation split only; the co-temporal context pool
is the development roster (train + validation) and only landings BEFORE the ego flight's
terminal-ring entry time.

    python run_ts_runway_hypotheses.py --checkpoint <ckpt> --airport KRDU \
        --output-dir 4dTrajectory/outputs/KRDU/experiments/runway_hypotheses_20260903/A_seed1337
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
for path in (TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from approach_difficulty import approach_difficulty  # noqa: E402
from channels import IDX, channels_from_states, states_from_channels  # noqa: E402
from coordinate_frames import COORDINATE_FRAME_AIRPORT_ENU  # noqa: E402
from dataset import (  # noqa: E402
    FlightSeries,
    arrival_data_provenance,
    build_series,
    load_flight_dicts,
    require_matching_data_provenance,
)
from export import observed_series_metrics  # noqa: E402
from forecast import Forecast, default_anchor, forecast_approaches  # noqa: E402
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon  # noqa: E402
from lateral_eligibility import default_lateral_pass_roster_path  # noqa: E402
from models import resolve_device  # noqa: E402
from train import load_checkpoint  # noqa: E402

SCHEMA = "ts-runway-hypotheses-v2-mirror-control"
# A pseudo-candidate per flight: the assigned threshold mirrored to the far side of its
# parallel sibling's offset (same separation, same course). An oracle that gains as much
# from this fake alternative as from the real sibling is picking the luckiest of K noisy
# forecasts, not using runway knowledge.
MIRROR = "MIRROR"
STRAIGHT_TORTUOSITY = 1.05
# A candidate whose inbound course is more than this far from the aircraft's track at the
# anchor is not being flown to; the parallel sibling stays inside the gate by construction.
COURSE_GATE_DEG = 90.0


def identity(source: dict[str, Any]) -> str:
    """Runway-independent flight identity (the runway is what the hypotheses vary)."""
    return f"{source.get('id')}_{source.get('icao24')}_{source.get('landing_time_utc')}"


def _wrap_deg(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def reproject(forecast: Forecast, from_series: FlightSeries, to_series: FlightSeries) -> Forecast:
    """Express a forecast made in one threshold's chart in another threshold's chart."""
    mass_kg = float(from_series.scenario.initial.m)
    offsets = np.cumsum(forecast.sample_durations_s)
    geodetic = states_from_channels(offsets, forecast.values, from_series.frame, mass_kg=mass_kg)
    _times, values = channels_from_states(geodetic, to_series.frame)
    return replace(forecast, values=values)


def track_course_deg(series: FlightSeries, anchor: int) -> float:
    """Aircraft track over the ground at the anchor, math-ENU degrees."""
    velocity = series.values[anchor]
    east, north = series.frame.to_world_horizontal(
        float(velocity[IDX["edot"]]), float(velocity[IDX["ndot"]])
    )
    return math.degrees(math.atan2(north, east))


def hypothesis_row(
    forecast: Forecast, series: FlightSeries, truth: FlightSeries, *, points: int
) -> dict[str, Any]:
    """Score one (flight, candidate runway) forecast against the observed track."""
    metrics = observed_series_metrics(truth, reproject(forecast, series, truth), points=points)
    last = forecast.values[-1]
    # In the candidate's own chart the runway is the origin: how close did the forecast get?
    closest_m = float(math.hypot(last[IDX["e"]], last[IDX["n"]]))
    # The endpoint against the TRUE runway's centreline, in the true chart.
    mass_kg = float(series.scenario.initial.m)
    end_state = states_from_channels(
        np.asarray([float(forecast.final_time_s)]), forecast.values[-1:], series.frame,
        mass_kg=mass_kg,
    )
    _t, end_true = channels_from_states(end_state, truth.frame)
    psi = float(truth.scenario.target.psi)
    east, north = truth.frame.to_world_horizontal(float(end_true[0, 0]), float(end_true[0, 1]))
    course = float(series.scenario.target.psi)
    return {
        "ade_m": float(metrics["ade_m"]),
        "fde_m": float(metrics["fde_m"]),
        "arrival_endpoint_error_m": float(metrics["arrival_endpoint_error_m"]),
        "final_time_error_s": float(metrics["final_time_error_s"]),
        "predicted_final_time_s": float(forecast.predicted_final_time_s),
        "horizon_capped": bool(forecast.horizon_capped),
        "closest_approach_m": closest_m,
        "endpoint_cross_track_true_m": east * math.sin(psi) - north * math.cos(psi),
        "endpoint_along_track_true_m": east * math.cos(psi) + north * math.sin(psi),
        "course_deg": math.degrees(course),
        "course_delta_deg": abs(_wrap_deg(
            track_course_deg(truth, forecast.anchor) - math.degrees(course)
        )),
    }


def parallel_sibling(runway: str, targets: dict[str, dict[str, Any]]) -> str | None:
    """The nearest other threshold with the same inbound course (within 30 deg)."""
    own = targets[runway]
    same = [
        other for other, target in targets.items()
        if other != runway and abs(_wrap_deg(target["course_deg"] - own["course_deg"])) <= 30.0
    ]
    if not same:
        return None
    return min(same, key=lambda other: math.hypot(
        (targets[other]["lon"] - own["lon"]) * metres_per_deg_lon(own["lat"]),
        (targets[other]["lat"] - own["lat"]) * METRES_PER_DEG_LAT,
    ))


def mirror_target(own: dict[str, Any], sibling: dict[str, Any]) -> dict[str, Any]:
    """The assigned threshold displaced by the sibling's offset, in the opposite direction."""
    d_lat = sibling["lat"] - own["lat"]
    d_lon = sibling["lon"] - own["lon"]
    return {**own, "lat": own["lat"] - d_lat, "lon": own["lon"] - d_lon}


def active_configuration(
    records: list[dict[str, Any]], development_keys: set[str], airport: str,
    *, window: timedelta,
) -> "callable":
    """Most-used runway among development landings in the window before an entry time."""
    landings = sorted(
        (_parse_utc(row["landing_time_utc"]), row["runway"])
        for row in records
        if f"{airport}:{row['flight_key']}" in development_keys
    )
    times = [item[0] for item in landings]

    def lookup(entry_time_utc: str) -> tuple[str | None, int]:
        entry = _parse_utc(entry_time_utc)
        lo = np.searchsorted(times, entry - window, side="left")
        hi = np.searchsorted(times, entry, side="left")
        recent = Counter(runway for _t, runway in landings[lo:hi])
        if not recent:
            return None, 0
        return recent.most_common(1)[0][0], sum(recent.values())

    return lookup


def select(
    rows: dict[str, dict[str, Any]], candidates: list[str], *, assigned: str,
    config_runway: str | None,
) -> dict[str, str]:
    """Every selection rule's pick for one flight."""
    available = [r for r in candidates if r in rows and r != MIRROR]
    gated = [r for r in available if rows[r]["course_delta_deg"] <= COURSE_GATE_DEG] or available
    # The parallel sibling(s) of the assigned runway: same inbound course within 30 deg.
    same_direction = [
        r for r in available
        if abs(_wrap_deg(rows[r]["course_deg"] - rows[assigned]["course_deg"])) <= 30.0
    ]
    mirror_pool = [assigned] + ([MIRROR] if MIRROR in rows else [])
    by_closest = lambda pool: min(pool, key=lambda r: rows[r]["closest_approach_m"])  # noqa: E731
    picks = {
        "assigned": assigned,
        "oracle_fde": min(available, key=lambda r: rows[r]["fde_m"]),
        "oracle_ade": min(available, key=lambda r: rows[r]["ade_m"]),
        "oracle_same_direction": min(same_direction, key=lambda r: rows[r]["fde_m"]),
        "oracle_mirror_control": min(mirror_pool, key=lambda r: rows[r]["fde_m"]),
        "self_consistency": by_closest(available),
        "course_gate_then_self": by_closest(gated),
        "active_config": config_runway if config_runway in available else by_closest(gated),
        "active_config_then_gate": (
            config_runway if config_runway in gated else by_closest(gated)
        ),
    }
    return picks


def summarise(flights: list[dict[str, Any]], selectors: list[str]) -> dict[str, Any]:
    tort = np.array([f["difficulty"]["route_tortuosity"] for f in flights])
    established = np.array([bool(f["difficulty"]["established_at_anchor"]) for f in flights])
    strata = {
        "all": np.ones(len(flights), dtype=bool),
        "straight-in": tort < STRAIGHT_TORTUOSITY,
        "vectored": (tort >= STRAIGHT_TORTUOSITY) & ~established,
    }
    out: dict[str, Any] = {}
    for stratum, mask in strata.items():
        block: dict[str, Any] = {"n": int(mask.sum()), "selectors": {}}
        for selector in selectors:
            ade = np.array([
                f["hypotheses"][f["picks"][selector]]["ade_m"] for f in flights
            ])[mask]
            fde = np.array([
                f["hypotheses"][f["picks"][selector]]["fde_m"] for f in flights
            ])[mask]
            hit = np.array([f["picks"][selector] == f["assigned"] for f in flights])[mask]
            block["selectors"][selector] = {
                "ade_mean": float(ade.mean()), "ade_median": float(np.median(ade)),
                "fde_mean": float(fde.mean()), "fde_median": float(np.median(fde)),
                "runway_accuracy": float(hit.mean()),
            }
        out[stratum] = block
    # Per assigned runway: the oracle gap tells where the misassignment cost lives.
    by_runway: dict[str, Any] = {}
    for runway in sorted({f["assigned"] for f in flights}):
        group = [f for f in flights if f["assigned"] == runway]
        by_runway[runway] = {
            "n": len(group),
            **{
                selector: {
                    "fde_median": float(np.median([
                        f["hypotheses"][f["picks"][selector]]["fde_m"] for f in group
                    ])),
                    "runway_accuracy": float(np.mean([
                        f["picks"][selector] == runway for f in group
                    ])),
                }
                for selector in selectors
            },
        }
    out["by_assigned_runway"] = by_runway
    return out


def print_summary(summary: dict[str, Any], selectors: list[str]) -> None:
    for stratum in ("all", "straight-in", "vectored"):
        block = summary[stratum]
        print(f"\n### {stratum} — n = {block['n']}\n")
        print("| selector | ADE mean | ADE med | FDE mean | FDE med | runway acc |")
        print("|---|---:|---:|---:|---:|---:|")
        for selector in selectors:
            s = block["selectors"][selector]
            print(f"| {selector} | {s['ade_mean']:.0f} | {s['ade_median']:.0f} | "
                  f"{s['fde_mean']:.0f} | {s['fde_median']:.0f} | {s['runway_accuracy'] * 100:.1f}% |")
    print("\n### Per assigned runway — FDE median (m) / runway accuracy\n")
    print("| runway | n | " + " | ".join(selectors) + " |")
    print("|---|---:|" + "---:|" * len(selectors))
    for runway, block in summary["by_assigned_runway"].items():
        cells = [
            f"{block[s]['fde_median']:.0f} / {block[s]['runway_accuracy'] * 100:.0f}%"
            for s in selectors
        ]
        print(f"| {runway} | {block['n']} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--airport", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-window-min", type=float, default=30.0,
                        help="co-temporal landings window before the ego's entry time")
    args = parser.parse_args(argv)

    airport = args.airport.upper()
    manifest_path = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    roster_path = default_lateral_pass_roster_path(manifest_path)
    model, config, normalizer, payload = load_checkpoint(args.checkpoint)
    if config.coordinate_frame == COORDINATE_FRAME_AIRPORT_ENU:
        parser.error("runway hypotheses need a threshold-anchored checkpoint (enu / runway-aligned)")
    provenance = arrival_data_provenance(manifest_path, eligibility_rosters=[roster_path])
    require_matching_data_provenance(payload, provenance, allow_subset=True)
    device = resolve_device(args.device)
    model = model.to(device)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = sorted(manifest["runway_targets"])
    val_keys = [k for k in payload["split"]["val"] if k.startswith(f"{airport}:")]
    development_keys = set(payload["split"]["train"]) | set(payload["split"]["val"])
    raw_flights = load_flight_dicts(manifest_path, include_flight_keys=set(val_keys))
    print(f"{airport}: {len(raw_flights)} validation flights × {len(candidates)} runway "
          f"hypotheses {candidates}; outer-test stays closed")
    config_lookup = active_configuration(
        manifest["records"], development_keys, airport,
        window=timedelta(minutes=args.context_window_min),
    )

    anchor = default_anchor(config)
    points = config.validation_common_grid_points
    # Build + forecast once per candidate; every clone keeps the ORIGINAL dict untouched.
    per_candidate: dict[str, dict[str, tuple[FlightSeries, Forecast]]] = {}
    targets = manifest["runway_targets"]
    mirrors = {
        runway: mirror_target(targets[runway], targets[sibling])
        for runway in candidates
        if (sibling := parallel_sibling(runway, targets)) is not None
    }
    for runway in [*candidates, MIRROR]:
        if runway == MIRROR:
            clones = [
                {**flight, "runway": f"{flight['runway']}{MIRROR}",
                 "runway_target": mirrors[flight["runway"]]}
                for flight in raw_flights if flight["runway"] in mirrors
            ]
        else:
            clones = [
                {**flight, "runway": runway, "runway_target": targets[runway]}
                for flight in raw_flights
            ]
        series, report = build_series(clones, config, airport=airport,
                                      aircraft_type=config.aircraft_type)
        print(f"  {runway}: {report.format().splitlines()[0]}")
        forecasts = forecast_approaches(model, series, config, normalizer, device=device)
        per_candidate[runway] = {
            identity(s.scenario.source): (s, f) for s, f in zip(series, forecasts, strict=True)
        }

    selectors = [
        "assigned", "oracle_fde", "oracle_ade", "oracle_same_direction",
        "oracle_mirror_control", "self_consistency", "course_gate_then_self", "active_config",
        "active_config_then_gate",
    ]
    flights: list[dict[str, Any]] = []
    missing_context = 0
    for flight in raw_flights:
        key = identity(flight)
        assigned = flight["runway"]
        if key not in per_candidate[assigned]:
            continue  # unbuildable under its own runway: not in the baseline either
        truth, _ = per_candidate[assigned][key]
        rows: dict[str, dict[str, Any]] = {}
        for runway, table in per_candidate.items():
            if key in table:  # a hypothesis can be unbuildable (track cut too short)
                series, forecast = table[key]
                rows[runway] = hypothesis_row(forecast, series, truth, points=points)
        config_runway, context_count = config_lookup(flight["entry_time_utc"])
        missing_context += context_count == 0
        picks = select(rows, candidates, assigned=assigned, config_runway=config_runway)
        difficulty = approach_difficulty(truth, anchor).to_dict()
        flights.append({
            "identity": key,
            "flight_key": flight.get("flight_key"),
            "assigned": assigned,
            "entry_time_utc": flight["entry_time_utc"],
            "active_config": {"runway": config_runway, "landings": context_count},
            "difficulty": difficulty,
            "hypotheses": rows,
            "picks": picks,
        })

    summary = summarise(flights, selectors)
    baseline = summary["all"]["selectors"]["assigned"]
    print(f"\n{airport}: {len(flights)} flights scored; {missing_context} without a landing in "
          f"the {args.context_window_min:g} min context window (course gate used instead)")
    print(f"assigned-runway baseline: ADE {baseline['ade_mean']:.1f} m  FDE {baseline['fde_mean']:.1f} m "
          "(must equal the checkpoint's own validation prediction)")
    print_summary(summary, selectors)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hypotheses.json").write_text(json.dumps({
        "schema_version": SCHEMA,
        "airport": airport,
        "checkpoint": str(args.checkpoint),
        "config": config.to_dict(),
        "candidates": candidates,
        "mirror_pseudo_candidates": mirrors,
        "selectors": selectors,
        "course_gate_deg": COURSE_GATE_DEG,
        "context_window_min": args.context_window_min,
        "context_pool": "development roster (train + validation), landings before entry",
        "flights_without_context": missing_context,
        "summary": summary,
        "flights": flights,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'hypotheses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
