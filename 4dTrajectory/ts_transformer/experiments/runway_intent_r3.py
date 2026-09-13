"""Runway-intent R3: multi-runway joint assignment and the arrival scheduler, flown by the plan experts.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §17. The flights are R2b's (every flight on ``day_a``'s
validation days that the retrained expert could fly), read from R2b's artifact and never recomputed:
the runway head's belief ``r11_lift`` at the expert's anchor (L-1) and, under each candidate runway,
the expert's ETA — the anchor's wall-clock time plus the time the head predicted there. The
scheduler (`inference.runway_schedule`) gives every flight a (runway, landing time), first come
first served by ETA, each runway scored ``log p - lambda * delay`` under the FAA arrival separation
(`faa_separation`: radar and CWT wake minima on one runway or a pair closer than 2500 ft, the
dependent-parallel diagonal, nothing between independent parallels or other directions). The
plan is then FLOWN: every flight's series is built on its scheduled runway and the expert flies it
in lockstep with the scheduled time as its assignment (`strategy.Assignment`, the v5.4 time
closure), cut at its threshold crossing.

Two schedules are read: the plan's (FCFS by ETA, flown and gated) and a CAUSAL one (each flight
placed when its anchor is reached, against the slots already frozen — the FCFS order uses ETAs
that later-anchored flights only have minutes after an earlier one's anchor; plan-only). Against
both: the independent forecast (each flight on its head's top runway at its own ETA, unscheduled)
and the truth (the landed runway, the landing time on the same clock). Traffic is incomplete —
the outer-test hash and the flights the expert could not fly are missing from the stream — so
the busy hours are counted on this roster and the visible share is printed beside them.

    python run_ts.py runway_intent_r3 --airport KRDU \\
        --r2 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2b_20260913/KRDU/runway_intent_r2.json \\
        --checkpoint 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2b_20260913/KRDU_day_a_expert/checkpoint.pt \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r3_20260914/KRDU
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from aircraft.identity import get_default_identity_resolver
from geokit import FT_M, METRES_PER_DEG_LAT, NM_M, metres_per_deg_lon
from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import PREDICTION_PLAN, TSConfig
from ts_transformer.data.channels import IDX, states_from_channels
from ts_transformer.data.data_provenance import arrival_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.data.lateral_eligibility import default_lateral_pass_roster_path
from ts_transformer.data.runway_context import airport_reference, operational_day, parse_utc
from ts_transformer.data.splits import split_name_for_dataset_id
from ts_transformer.experiments.runway_hypotheses import HARVEST_ROOT, hypothesis_row, identity
from ts_transformer.experiments.runway_intent_r1 import day_folds
from ts_transformer.geometry.final_approach_geometry import (
    alignment_cosine,
    hard_on_final,
    position_direction,
    runway_axes,
)
from ts_transformer.inference.forecast import Forecast, cut_at_threshold_crossing, default_anchor
from ts_transformer.inference.runway_schedule import (
    FAA_REDUCED_RADAR_NM,
    Arrival,
    Separation,
    Slot,
    faa_separation,
    fcfs_by_eta,
    schedule,
    violations,
    wake_category,
)
from ts_transformer.outputs.plan.guidance.timing import TIME_TOLERANCE_S
from ts_transformer.outputs.plan.strategy import Assignment, SkeletonCache, rolled_predictions_lockstep
from ts_transformer.training.train import load_checkpoint

SCHEMA = "ts-runway-intent-r3-v1"
HEAD = "r11_lift"                 # R2b's runway head (runway_intent_r2.HEAD)
BUSY_PER_HOUR = 10                # plan §17.3: an hour with >= 10 roster landings is busy
FLOWN_TOLERANCE_S = 10.0          # plan §17.3 gate 2: a flown gap counts as kept at >= S - 10 s
FINAL_SPEED_WINDOW_M = 2.0 * NM_M  # the approach speed is read over the last 2 NM before the threshold
STRATA = ("all", "busy", "quiet")
EXPERT_FAILS_M = 1000.0           # an unassigned forecast ending further than this from the true threshold: the expert's own failure
CLOSE_PAIR_MINIMA = 2.0           # an order is contestable when the true gap is under two minima
#: Plan-only readings under other rule sets the text allows (docs/literature/arrival_separation §7):
#: the reduced 2.5 NM of 5-5-4 j (authorization unverified at every airport), and the visual-approach
#: reading of 7-4-4 c (no minimum between two parallel runways; each keeps its own).
VARIANTS = {
    "radar_2.5nm": {"radar_nm": FAA_REDUCED_RADAR_NM},
    "visual_parallels": {"visual_parallels": True},
}


@dataclass(frozen=True)
class Flown:
    """One flight flown on its scheduled runway at its scheduled time."""

    landed: bool
    time_s: float                  # wall clock of the threshold crossing (the rollout's end if it never crossed)
    metrics: dict[str, Any]        # `hypothesis_row` against the observed track, in the true runway's chart
    closure: dict[str, float]      # X at the first ask, the speed factor, the stretch
    wall_s: np.ndarray             # [N] wall clock of every row
    east_m: np.ndarray             # [N] about the airport reference
    north_m: np.ndarray
    on_final: np.ndarray           # [N] the on-final gate, in the scheduled runway's axes


def wall_s(stamp: str) -> float:
    return datetime.fromisoformat(stamp).timestamp()


def endpoint_error_m(row: dict[str, Any]) -> float:
    """How far a forecast ENDS from the true threshold (`hypothesis_row`'s end point in the true
    runway's axes). The row's ``fde_m`` is the displacement at the TRUTH's landing time, so it scores a
    forecast that lands later than the truth by the path it has left — an assigned flight lands at its
    assigned time — and cannot compare an assigned flight with an unassigned one; this can."""
    return float(math.hypot(row["endpoint_along_track_true_m"], row["endpoint_cross_track_true_m"]))


def approach_speed_mps(flights: Sequence[dict[str, Any]], targets: dict[str, dict[str, Any]]) -> tuple[float, int]:
    """The median ground speed over the last `FINAL_SPEED_WINDOW_M` before the landed threshold — the
    contiguous run of rows inside the window that ends at the track's closest approach to it, so rows
    on the runway past it are never read — and how many flights it was read from (a flight whose track
    starts inside the window is skipped: where it entered is unknown)."""
    speeds = []
    for flight in flights:
        target = targets[flight["runway"]]
        lat0, lon0 = float(target["lat"]), float(target["lon"])
        rows = np.asarray([(float(w[0]), float(w[1]), float(w[2])) for w in flight["waypoints"]])
        distance = np.hypot((rows[:, 1] - lon0) * metres_per_deg_lon(lat0), (rows[:, 2] - lat0) * METRES_PER_DEG_LAT)
        end = int(np.argmin(distance))
        if distance[end] > FINAL_SPEED_WINDOW_M:
            continue
        first = end
        while first > 0 and distance[first - 1] <= FINAL_SPEED_WINDOW_M:
            first -= 1
        if first == 0 or rows[end, 0] <= rows[first, 0]:
            continue
        speeds.append((distance[first] - distance[end]) / (rows[end, 0] - rows[first, 0]))
    return float(np.median(speeds)), len(speeds)


def training_day_flights(manifest_path: Path, airport: str, config: TSConfig) -> list[dict[str, Any]]:
    """Every non-test-hash arrival on ``day_a``'s TRAINING days — what the approach speed is read from."""
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    keys = {
        f"{airport}:{r['flight_key']}" for r in records
        if split_name_for_dataset_id(f"{airport}:{r['flight_key']}", config) != "test"
        and day_folds(operational_day(parse_utc(r["landing_time_utc"])), config)["a"] == "train"
    }
    return load_flight_dicts(manifest_path, include_flight_keys=keys, verbose=False)


def visible_landings(tracks_path: Path, airport: str, config: TSConfig, hours: set[tuple[str, int]]) -> Counter:
    """Non-test-hash assigned landings per (date, UTC hour) in the given hours — the traffic this
    roster is a part of (the outer-test hash is never read)."""
    counts: Counter = Counter()
    for row in json.loads(tracks_path.read_text(encoding="utf-8"))["records"]:
        if row.get("outcome") != "assigned":
            continue
        if split_name_for_dataset_id(f"{airport}:{row['flight_key']}", config) == "test":
            continue
        t = parse_utc(row["landing_time_utc"])
        hour = (t.date().isoformat(), t.hour)
        if hour in hours:
            counts[hour] += 1
    return counts


def on_final_rows(forecast: Forecast, series) -> np.ndarray:
    """Every row's `on-final` gate in the series' runway axes (the rule `cut_at_threshold_crossing` reads)."""
    psi = torch.tensor([float(series.scenario.target.psi)], dtype=torch.float64)
    e = torch.from_numpy(np.ascontiguousarray(forecast.values[:, IDX["e"]] - series.target_chart[0]))[None]
    n = torch.from_numpy(np.ascontiguousarray(forecast.values[:, IDX["n"]] - series.target_chart[1]))[None]
    d, xt = runway_axes(e, n, psi)
    anchor = series.values[forecast.anchor]
    step_e, step_n = position_direction(
        e, n,
        torch.tensor([float(anchor[IDX["e"]] - series.target_chart[0])], dtype=torch.float64),
        torch.tensor([float(anchor[IDX["n"]] - series.target_chart[1])], dtype=torch.float64),
    )
    return hard_on_final(d, xt, alignment_cosine(step_e, step_n, psi))[0].numpy()


def fly(
    model, config: TSConfig, normalizer, device: torch.device, flights: dict[str, dict[str, Any]],
    slots: dict[str, Slot], truth_runway: dict[str, str], anchor_wall: dict[str, float],
    targets: dict[str, dict[str, Any]], airport: str,
) -> dict[str, Flown]:
    """Every flight flown on its scheduled runway with its scheduled time as the assignment."""
    anchor = default_anchor(config)
    points = config.validation_common_grid_points
    ref_lon, ref_lat = airport_reference(targets)
    needed: dict[str, set[str]] = defaultdict(set)
    for key, slot in slots.items():
        needed[slot.runway].add(key)
        needed[truth_runway[key]].add(key)
    series_on: dict[str, dict[str, Any]] = {}
    for runway, keys in sorted(needed.items()):
        clones = [{**flights[k], "runway": runway, "runway_target": targets[runway]} for k in sorted(keys)]
        built, _report = build_series(clones, config, airport=airport, aircraft_type=config.aircraft_type)
        series_on[runway] = {identity(x.scenario.source): x for x in built}
    out: dict[str, Flown] = {}
    skeletons = SkeletonCache()
    for runway in sorted({slot.runway for slot in slots.values()}):
        keys = sorted(k for k, slot in slots.items() if slot.runway == runway)
        group = [series_on[runway][identity(flights[k])] for k in keys]
        assignments = [
            Assignment(arrival_time_s=float(x.times[anchor]) + (slots[k].time_s - anchor_wall[k]))
            for k, x in zip(keys, group, strict=True)
        ]
        rolled = rolled_predictions_lockstep(
            model, group, config, normalizer, [anchor] * len(group), device,
            [skeletons.for_series(x) for x in group], assignments=assignments,
        )
        print(f"  flew {len(group)} on {runway}", flush=True)
        for key, x, flight in zip(keys, group, rolled, strict=True):
            cut = cut_at_threshold_crossing(flight.forecast, x)
            truth = series_on[truth_runway[key]][identity(flights[key])]
            hook = cut.command_hook_diagnostics
            offsets = np.cumsum(cut.sample_durations_s)
            states = states_from_channels(offsets, cut.values, x.frame, mass_kg=float(x.scenario.initial.m))
            lat = np.asarray([s.latitude for _, s in states])
            lon = np.asarray([s.longitude for _, s in states])
            out[key] = Flown(
                landed=bool(cut.truncated_at_threshold),
                time_s=anchor_wall[key] + float(cut.final_time_s),
                metrics=hypothesis_row(cut, x, truth, points=points),
                closure={name: float(hook[name]) for name in (
                    "planUnabsorbedFirstS", "planUnabsorbedLastS", "planSpeedFactorFirst", "planStretchM",
                )},
                wall_s=anchor_wall[key] + offsets,
                east_m=(lon - ref_lon) * metres_per_deg_lon(ref_lat),
                north_m=(lat - ref_lat) * METRES_PER_DEG_LAT,
                on_final=on_final_rows(cut, x),
            )
    return out


def one_runway_pairs(slots: Sequence[Slot], separation: Separation) -> list[tuple[Slot, Slot]]:
    """Consecutive landings within each group separated as one runway, in order on the approach clock
    (`Separation.approach_s`: two staggered thresholds' own times do not say who was ahead)."""
    ordered = sorted(slots, key=lambda s: (separation.approach_s(s), s.key))
    last: dict[str, Slot] = {}
    pairs = []
    for s in ordered:
        leader = max((l for r, l in last.items() if separation.one_runway(r, s.runway)), key=separation.approach_s, default=None)
        if leader is not None:
            pairs.append((leader, s))
        last[s.runway] = s
    return pairs


def kept_share(slots: Sequence[Slot], separation: Separation, tolerance_s: float, *, close_only: bool = False) -> tuple[float, int]:
    """Of the consecutive one-runway pairs — every one, or (``close_only``) those under
    `CLOSE_PAIR_MINIMA` minima apart, the ones a minimum can bind — the share at least their minimum
    less ``tolerance_s`` apart on the approach clock, and how many pairs that is."""
    kept = []
    for leader, follower in one_runway_pairs(slots, separation):
        need = separation.gap_s(leader.runway, leader.category, follower.runway, follower.category)
        gap = separation.approach_s(follower) - separation.approach_s(leader)
        if close_only and gap >= CLOSE_PAIR_MINIMA * need:
            continue
        kept.append(gap >= need - tolerance_s)
    return (float(np.mean(kept)) if kept else math.nan), len(kept)


def final_separation(flown: dict[str, Flown], slots: Sequence[Slot], separation: Separation) -> dict[str, Any]:
    """For consecutive flown one-runway pairs, the closest the two came while BOTH were on the final,
    against the pair's distance minimum."""
    margins, without = [], 0
    for leader, follower in one_runway_pairs(slots, separation):
        a, b = flown[leader.key], flown[follower.key]
        rows = (b.wall_s >= a.wall_s[0]) & (b.wall_s <= a.wall_s[-1]) & b.on_final
        if not rows.any():
            without += 1
            continue
        t = b.wall_s[rows]
        leader_on = np.interp(t, a.wall_s, a.on_final.astype(float)) >= 0.5
        if not leader_on.any():
            without += 1
            continue
        gap = np.hypot(b.east_m[rows] - np.interp(t, a.wall_s, a.east_m), b.north_m[rows] - np.interp(t, a.wall_s, a.north_m))
        need = separation.distance_nm(leader.runway, leader.category, follower.runway, follower.category)
        margins.append(float(np.min(gap[leader_on])) / NM_M - need)
    m = np.asarray(margins)
    return {
        "pairs": len(margins), "pairs_never_both_on_final": without,
        "margin_nm_p5": float(np.percentile(m, 5)) if len(m) else math.nan,
        "margin_nm_p50": float(np.median(m)) if len(m) else math.nan,
        "share_kept": float(np.mean(m >= 0.0)) if len(m) else math.nan,
        "share_within_half_nm": float(np.mean(m >= -0.5)) if len(m) else math.nan,
    }


def _abs_quantiles(values: Sequence[float]) -> dict[str, float]:
    v = np.abs(np.asarray(values, dtype=float))
    return {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)), "mean": float(np.mean(v))} if len(v) else \
        {"p50": math.nan, "p90": math.nan, "mean": math.nan}


def order_agreement(
    rows: Sequence[dict[str, Any]], separation: Separation, field: str, *, close_only: bool = False,
) -> tuple[float, int]:
    """Of consecutive TRUE landings within a group landed as one runway, the share ``field`` orders the
    same way — every such pair, or (``close_only``) the ones under `CLOSE_PAIR_MINIMA` minima apart,
    whose order is contestable (a pair four minutes apart is ordered by any forecast)."""
    truth = [Slot(r["flight_key"], r["truth"]["runway"], r["truth"]["time_s"], r["truth"]["time_s"], r["category"]) for r in rows]
    by_key = {r["flight_key"]: r for r in rows}

    def clock(key: str) -> float:
        landing = by_key[key][field]
        return separation.approach_time_s(landing["runway"], landing["time_s"])

    same = [
        clock(l.key) < clock(f.key)
        for l, f in one_runway_pairs(truth, separation)
        if not close_only or separation.approach_s(f) - separation.approach_s(l)
        < CLOSE_PAIR_MINIMA * separation.gap_s(l.runway, l.category, f.runway, f.category)
    ]
    return (float(np.mean(same)) if same else math.nan), len(same)


def variant_reading(rows: list[dict[str, Any]], slots: dict[str, Slot], separation: Separation) -> dict[str, Any]:
    """A plan-only schedule under another rule set: its assignment and timing against the independent
    forecast, and how the truth and the unscheduled forecast sit under those rules."""
    truth = [Slot(r["flight_key"], r["truth"]["runway"], r["truth"]["time_s"], r["truth"]["time_s"], r["category"]) for r in rows]
    independent = [Slot(r["flight_key"], r["independent"]["runway"], r["independent"]["time_s"], r["independent"]["time_s"],
                        r["category"]) for r in rows]
    out: dict[str, Any] = {
        "plan_violations": len(violations(list(slots.values()), separation)),
        "independent_violations": len(violations(independent, separation)),
        "truth_violations_10s": len(violations(truth, separation, tolerance_s=FLOWN_TOLERANCE_S)),
        "truth_one_runway_kept_share": kept_share(truth, separation, FLOWN_TOLERANCE_S),
        "truth_one_runway_kept_share_close": kept_share(truth, separation, FLOWN_TOLERANCE_S, close_only=True),
    }
    for stratum in STRATA:
        members = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
        if not members:
            continue
        moved = [r for r in members
                 if slots[r["flight_key"]].delay_s > 1.0 or slots[r["flight_key"]].runway != r["independent"]["runway"]]
        out[stratum] = {
            "runway_accuracy": float(np.mean([slots[r["flight_key"]].runway == r["truth"]["runway"] for r in members])),
            "time_error_s": _abs_quantiles([slots[r["flight_key"]].time_s - r["truth"]["time_s"] for r in members]),
            "delayed_share": float(np.mean([slots[r["flight_key"]].delay_s > 1.0 for r in members])),
            "moved": len(moved),
            "moved_time_error_s": _abs_quantiles([slots[r["flight_key"]].time_s - r["truth"]["time_s"] for r in moved]),
            "moved_independent_time_error_s": _abs_quantiles([r["independent"]["time_s"] - r["truth"]["time_s"] for r in moved]),
        }
    return out


def summarise(rows: list[dict[str, Any]], separation: Separation, flown: dict[str, Flown] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in STRATA:
        members = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
        if not members:
            continue
        cell: dict[str, Any] = {"flights": len(members)}
        for name in ("independent", "scheduled", "causal"):
            cell[name] = {
                "runway_accuracy": float(np.mean([r[name]["runway"] == r["truth"]["runway"] for r in members])),
                "time_error_s": _abs_quantiles([r[name]["time_s"] - r["truth"]["time_s"] for r in members]),
                "order_agreement": order_agreement(members, separation, name)[0],
                "order_agreement_close": order_agreement(members, separation, name, close_only=True)[0],
            }
        for name in ("scheduled", "causal"):
            delays = np.asarray([r[name]["delay_s"] for r in members])
            cell[name]["delay_s"] = {"p50": float(np.median(delays)), "p90": float(np.percentile(delays, 90)),
                                     "delayed_share": float(np.mean(delays > 1.0))}
        cell["order_pairs"] = order_agreement(members, separation, "scheduled")[1]
        cell["order_pairs_close"] = order_agreement(members, separation, "scheduled", close_only=True)[1]
        # the flights the schedule moved (delayed, or put on another runway than the head's top): the
        # only ones whose time differs from the independent forecast's, read paired
        moved = [r for r in members if r["scheduled"]["delay_s"] > 1.0 or r["scheduled"]["runway"] != r["independent"]["runway"]]
        cell["moved"] = {
            "flights": len(moved),
            "scheduled_time_error_s": _abs_quantiles([r["scheduled"]["time_s"] - r["truth"]["time_s"] for r in moved]),
            "independent_time_error_s": _abs_quantiles([r["independent"]["time_s"] - r["truth"]["time_s"] for r in moved]),
            "scheduled_closer_share": float(np.mean([
                abs(r["scheduled"]["time_s"] - r["truth"]["time_s"]) < abs(r["independent"]["time_s"] - r["truth"]["time_s"])
                for r in moved])) if moved else math.nan,
            "runway_changed": sum(r["scheduled"]["runway"] != r["independent"]["runway"] for r in moved),
            "runway_changed_to_truth": sum(r["scheduled"]["runway"] != r["independent"]["runway"]
                                           and r["scheduled"]["runway"] == r["truth"]["runway"] for r in moved),
        }
        if flown is not None:
            landed = [r for r in members if r["flown"]["landed"]]
            cell["flown"] = {
                "landed_share": len(landed) / len(members),
                "time_error_s": _abs_quantiles([r["flown"]["time_s"] - r["truth"]["time_s"] for r in landed]),
                # the same flights' scheduled and independent errors, so the three are paired
                "scheduled_time_error_s_same_flights": _abs_quantiles([r["scheduled"]["time_s"] - r["truth"]["time_s"] for r in landed]),
                "independent_time_error_s_same_flights": _abs_quantiles([r["independent"]["time_s"] - r["truth"]["time_s"] for r in landed]),
                "delivery_error_s": _abs_quantiles([r["flown"]["time_s"] - r["scheduled"]["time_s"] for r in landed]),
                # where the forecast ENDS against the true threshold, flown vs the same flight unassigned
                "endpoint_error_m_median": float(np.median([r["flown"]["endpoint_error_m"] for r in members])),
                "unassigned_endpoint_error_m_median": float(np.median([r["scheduled"]["unassigned_endpoint_error_m"] for r in members])),
                "endpoint_error_m_median_landed": float(np.median([r["flown"]["endpoint_error_m"] for r in landed])),
                "unassigned_endpoint_error_m_median_landed": float(np.median([r["scheduled"]["unassigned_endpoint_error_m"] for r in landed])),
                # the time-aligned displacement at the truth's landing time: see `endpoint_error_m`
                "fde_m_median": float(np.median([r["flown"]["fde_m"] for r in members])),
                "unassigned_fde_m_median": float(np.median([r["scheduled"]["unassigned_fde_m"] for r in members])),
                "unabsorbed_first_s_p50": float(np.median([r["flown"]["closure"]["planUnabsorbedFirstS"] for r in members])),
                "absorbed_share": float(np.mean([abs(r["flown"]["closure"]["planUnabsorbedFirstS"]) <= TIME_TOLERANCE_S for r in members])),
                "speed_factor_first_p50": float(np.median([r["flown"]["closure"]["planSpeedFactorFirst"] for r in members])),
                "stretched_share": float(np.mean([r["flown"]["closure"]["planStretchM"] > 0.0 for r in members])),
            }
        out[stratum] = cell
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--airport", required=True)
    parser.add_argument("--r2", required=True, help="R2b's runway_intent_r2.json (--roster day-val) for this airport")
    parser.add_argument("--checkpoint", required=True, help="the plan expert R2b flew (trained on day_a's training days)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--delay-weight-per-s", type=float, default=1.0 / 60.0,
                        help="lambda: nats of runway log probability one second of delay costs (plan §17.4: 1/60)")
    parser.add_argument("--min-probability", type=float, default=0.01,
                        help="epsilon: a runway below it is never tried unless it is the head's top one")
    parser.add_argument("--no-fly", action="store_true", help="read the schedules only; do not fly them")
    args = parser.parse_args(argv)

    airport = args.airport.upper()
    r2 = json.loads(Path(args.r2).read_text(encoding="utf-8"))
    if r2["airport"] != airport or r2["evaluation"]["roster"] != "day-val":
        parser.error(f"--r2 must be {airport}'s R2b artifact (roster day-val); it is {r2['airport']} / {r2['evaluation']['roster']}")
    config = TSConfig()
    manifest_path = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    targets = json.loads(manifest_path.read_text(encoding="utf-8"))["runway_targets"]

    # every quantity the rules read that the data sets is measured on day_a's TRAINING days
    speed_mps, speed_flights = approach_speed_mps(training_day_flights(manifest_path, airport, config), targets)
    separation = faa_separation(targets, speed_mps=speed_mps)
    print(f"{airport}: approach speed {speed_mps:.1f} m/s (median over the last 2 NM, {speed_flights} training-day flights); "
          f"3 NM = {separation.gap_s('x', '', 'x', ''):.1f} s")
    for pair, relation in sorted(separation.relations.items(), key=lambda kv: sorted(kv[0])):
        a, b = sorted(pair)
        print(f"  {a}/{b}: {relation}, spacing {separation.spacing_nm[pair] * NM_M / FT_M:.0f} ft, "
              f"threshold stagger {abs(separation.along_nm[a] - separation.along_nm[b]):.2f} NM")

    records = {r["flight_key"]: r for r in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]}
    resolver = get_default_identity_resolver()
    rows: list[dict[str, Any]] = []
    arrivals: list[Arrival] = []
    known_at: dict[str, float] = {}
    for f in r2["flights"]:
        key = f["flight_key"]
        anchor = wall_s(f["anchor_time_utc"])
        hyp = f["hypotheses"]
        probs = f["probabilities"][HEAD]
        etas = {r: anchor + float(row["predicted_final_time_s"]) for r, row in hyp.items()}
        logp = {r: math.log(max(float(probs[r]), 1e-300)) for r in hyp}
        truth_row = hyp[f["assigned"]]
        truth_time = anchor + float(truth_row["predicted_final_time_s"]) - float(truth_row["final_time_error_s"])
        rec = records[key]
        typecode = resolver.resolve(declared_type=rec.get("type"), icao24=rec.get("icao24")).typecode
        category = wake_category(typecode)
        top = max(hyp, key=lambda r: (probs[r], r))
        arrivals.append(Arrival(key, etas, logp, category))
        known_at[key] = anchor
        rows.append({
            "flight_key": key, "typecode": typecode, "category": category, "anchor_s": anchor,
            "truth": {"runway": f["assigned"], "time_s": truth_time},
            "independent": {"runway": top, "time_s": etas[top]},
            "etas_s": etas, "probabilities": {r: float(probs[r]) for r in hyp},
        })

    plan = {s.key: s for s in schedule(fcfs_by_eta(arrivals, args.min_probability), separation,
                                       delay_weight_per_s=args.delay_weight_per_s, min_probability=args.min_probability)}
    causal = {s.key: s for s in schedule(sorted(arrivals, key=lambda a: (known_at[a.key], a.key)), separation,
                                         delay_weight_per_s=args.delay_weight_per_s, min_probability=args.min_probability)}
    hour_of = {}
    for r in rows:
        landed = datetime.fromtimestamp(r["truth"]["time_s"], tz=timezone.utc)
        hour_of[r["flight_key"]] = (landed.date().isoformat(), landed.hour)
    per_hour = Counter(hour_of.values())
    r2_by_key = {f["flight_key"]: f for f in r2["flights"]}
    for r in rows:
        key = r["flight_key"]
        for name, table in (("scheduled", plan), ("causal", causal)):
            s = table[key]
            r[name] = {"runway": s.runway, "time_s": s.time_s, "eta_s": s.eta_s, "delay_s": s.delay_s}
        r["scheduled"]["unassigned_fde_m"] = float(r2_by_key[key]["hypotheses"][plan[key].runway]["fde_m"])
        r["scheduled"]["unassigned_endpoint_error_m"] = endpoint_error_m(r2_by_key[key]["hypotheses"][plan[key].runway])
        r["hour_landings"] = per_hour[hour_of[key]]
        r["stratum"] = "busy" if per_hour[hour_of[key]] >= BUSY_PER_HOUR else "quiet"

    truth_slots = [Slot(r["flight_key"], r["truth"]["runway"], r["truth"]["time_s"], r["truth"]["time_s"], r["category"]) for r in rows]
    independent_slots = [Slot(r["flight_key"], r["independent"]["runway"], r["independent"]["time_s"], r["independent"]["time_s"], r["category"]) for r in rows]
    variants = {}
    for name, options in VARIANTS.items():
        rules = faa_separation(targets, speed_mps=speed_mps, **options)
        table = {s.key: s for s in schedule(fcfs_by_eta(arrivals, args.min_probability), rules, delay_weight_per_s=args.delay_weight_per_s,
                                            min_probability=args.min_probability)}
        variants[name] = {"options": options, **variant_reading(rows, table, rules)}
    visible = visible_landings(HARVEST_ROOT / airport / "tracks" / "manifest.json", airport, config, set(per_hour))
    busy_hours = {h for h, n in per_hour.items() if n >= BUSY_PER_HOUR}
    checks: dict[str, Any] = {
        "plan_violations": len(violations(list(plan.values()), separation)),
        "causal_violations": len(violations(list(causal.values()), separation)),
        "independent_violations": len(violations(independent_slots, separation)),
        "truth_violations_10s": len(violations(truth_slots, separation, tolerance_s=FLOWN_TOLERANCE_S)),
        "truth_one_runway_kept_share": kept_share(truth_slots, separation, FLOWN_TOLERANCE_S),
        "truth_one_runway_kept_share_close": kept_share(truth_slots, separation, FLOWN_TOLERANCE_S, close_only=True),
        "roster_share_of_visible_landings": sum(per_hour.values()) / max(sum(visible.values()), 1),
        "roster_share_of_visible_landings_busy": sum(per_hour[h] for h in busy_hours) / max(sum(visible[h] for h in busy_hours), 1),
        "busy_hours": len(busy_hours), "hours": len(per_hour),
    }

    flown = None
    if not args.no_fly:
        model, ckpt_config, normalizer, payload = load_checkpoint(args.checkpoint)
        if ckpt_config.prediction_output != PREDICTION_PLAN:
            parser.error("R3 flies the PLAN expert")
        if str(Path(args.checkpoint).resolve()) != str(Path(r2["checkpoint"]).resolve()):
            parser.error(f"--checkpoint must be the expert R2b flew ({r2['checkpoint']})")
        provenance = arrival_data_provenance(manifest_path, eligibility_rosters=[default_lateral_pass_roster_path(manifest_path)])
        require_matching_data_provenance(payload, provenance, allow_subset=True)
        device = resolve_device(args.device)
        model = model.to(device)
        loaded = load_flight_dicts(manifest_path, include_flight_keys={f"{airport}:{r['flight_key']}" for r in rows}, verbose=False)
        by_identity = {identity(f): f for f in loaded}
        flights = {f["flight_key"]: by_identity[f["identity"]] for f in r2["flights"]}
        flown = fly(model, ckpt_config, normalizer, device, flights, plan,
                    {r["flight_key"]: r["truth"]["runway"] for r in rows}, {r["flight_key"]: r["anchor_s"] for r in rows},
                    targets, airport)
        for r in rows:
            x = flown[r["flight_key"]]
            r["flown"] = {"landed": x.landed, "time_s": x.time_s, "fde_m": x.metrics["fde_m"], "ade_m": x.metrics["ade_m"],
                          "endpoint_error_m": endpoint_error_m(x.metrics), "closure": x.closure}
        flown_slots = [Slot(k, plan[k].runway, x.time_s, plan[k].eta_s, plan[k].category) for k, x in flown.items() if x.landed]
        # a flight that never crossed is out of every flown pair: counted here, never read as kept
        checks["flown_unlanded"] = sum(not x.landed for x in flown.values())
        # ...split by whether the expert fails the flight without an assignment too (R2b's forecast on
        # the same runway ends > EXPERT_FAILS_M from the true threshold): the rest the assignment broke
        checks["flown_unlanded_expert_fails_unassigned"] = sum(
            not r["flown"]["landed"] and r["scheduled"]["unassigned_endpoint_error_m"] > EXPERT_FAILS_M for r in rows)
        checks["flown_violations_10s"] = len(violations(flown_slots, separation, tolerance_s=FLOWN_TOLERANCE_S))
        checks["flown_one_runway_kept_share"] = kept_share(flown_slots, separation, FLOWN_TOLERANCE_S)
        checks["flown_one_runway_kept_share_close"] = kept_share(flown_slots, separation, FLOWN_TOLERANCE_S, close_only=True)
        checks["flown_final_separation"] = final_separation(flown, flown_slots, separation)

    summary = summarise(rows, separation, flown)
    print(json.dumps({"checks": checks, "summary": summary, "variants": variants}, indent=1))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runway_intent_r3.json").write_text(json.dumps({
        "schema_version": SCHEMA,
        "airport": airport,
        "r2": str(args.r2),
        "checkpoint": str(args.checkpoint),
        "flown": flown is not None,
        "scheduler": {
            "order": ("FCFS by the earliest ETA over the eligible runways (the plan; NOT causal: it reads ETAs some flights "
                      "only have after an earlier flight's anchor); causal: each flight placed at its anchor against the "
                      "frozen slots"),
            "delay_weight_per_s": args.delay_weight_per_s, "min_probability": args.min_probability,
            "runway_head": HEAD, "eta": "the anchor's wall clock + the expert head's predicted time under each runway",
        },
        "separation": {
            "rules": "faa_separation (FAA JO 7110.65BB; docs/literature/arrival_separation/README.md)",
            "approach_speed_mps": speed_mps, "approach_speed_flights": speed_flights,
            "same_nm": separation.same_nm,
            "relations": {"/".join(sorted(p)): rel for p, rel in separation.relations.items()},
            "spacing_ft": {"/".join(sorted(p)): s * NM_M / FT_M for p, s in separation.spacing_nm.items()},
            "diagonal_nm": {"/".join(sorted(p)): d for p, d in separation.diagonal_nm.items()},
            "along_nm": dict(separation.along_nm),
            "not_modelled": [
                "intersecting / converging runways (3-10-4): read as unrelated — KMSY 02/20 and 11/29 are never separated",
                "runway occupancy (3-10-3 a): shorter than the radar minimum's time",
                "JO 7110.308E's 1.0 NM diagonal at KSTL 12/30 (its current use there is unverified)",
            ],
        },
        "strata": {"busy_per_hour": BUSY_PER_HOUR, "clock": "the truth landing's UTC hour, counted on this roster"},
        "checks": checks,
        "summary": summary,
        "variants": variants,
        "flights": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'runway_intent_r3.json'}: {len(rows)} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
