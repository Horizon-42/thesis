"""Runway-intent R2a: the runway head's choice flown end to end — the plan expert under every candidate runway, paired against the known runway and the causal rules.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §16. The runway heads are R1.1b's ``r11_lift`` and
R1's, trained exactly as there (`runway_intent_r1.build_samples`, the partition's training days,
ring anchors) and asked ONCE, at the plan expert's own anchor (L-1), from the last raw track point at
or before it. The expert is the plan checkpoint, flown under each candidate by R0b's mechanism
(`runway_hypotheses`: the flight dict cloned with that runway's target, the series built in its chart,
the same `forecast_approaches` — the rolled lockstep cut at the threshold — scored in the true
runway's chart). Every pick is scored against the known runway on the same flights.

The flights are the ones NEITHER model trained on: the plan checkpoint's per-flight validation split,
on the operating days that are validation days of a day partition (``day_a`` first) — each flight is
read by that partition's runway head and rules. The expert did see other flights of the same days
(it trains on the per-flight split): the paired comparison is unaffected, the absolute error level is
not — R2b retrains the experts on the day split.

    python run_ts.py runway_intent_r2 --airport KRDU --checkpoint <plan checkpoint.pt> \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2_20260913/KRDU
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import PREDICTION_PLAN
from ts_transformer.data.approach_difficulty import approach_difficulty
from ts_transformer.data.data_provenance import arrival_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series
from ts_transformer.data.lateral_eligibility import default_lateral_pass_roster_path
from ts_transformer.data.runway_context import RULES, operational_day, parse_utc
from ts_transformer.data.runway_features import (
    airline,
    anchor_features,
    candidate_rows,
    minutes_since_each,
    track_course_at,
)
from ts_transformer.experiments.runway_hypotheses import (
    HARVEST_ROOT,
    hypothesis_row,
    identity,
    print_summary,
    skeleton_error,
    summarise,
)
from ts_transformer.experiments.runway_intent_r1 import (
    HGB_SETTINGS,
    RunwaySamples,
    add_sample_arguments,
    build_samples,
    day_folds,
)
from ts_transformer.experiments.runway_intent_r11 import (
    HEAD_SETTINGS,
    ListwiseBooster,
    airline_shares,
    head_table,
    prior_share,
)
from ts_transformer.inference.forecast import default_anchor, forecast_approaches
from ts_transformer.training.train import load_checkpoint

SCHEMA = "ts-runway-intent-r2-v1"
PARTS = ("day_a", "day_b")
HEAD = "r11_lift"
SELECTORS = ("assigned", HEAD, "r1", *RULES, "oracle_fde")


def evaluation_flights(s: RunwaySamples, val_keys: set[str]) -> dict[str, str]:
    """flight key -> the day partition whose validation day it lands on (``day_a`` first), for
    every flight of the expert's validation split; flights on neither partition's validation days
    are out (one of the models has seen them or their day)."""
    out: dict[str, str] = {}
    for key, flight in s.usable.items():
        if key not in val_keys:
            continue
        folds = day_folds(operational_day(parse_utc(flight["landing_time_utc"])), s.config)
        if folds["a"] == "val":
            out[key] = "day_a"
        elif folds["b"] == "val":
            out[key] = "day_b"
    return out


def anchor_waypoint(flight: dict[str, Any], seconds_after_first: float) -> int:
    """The last raw track point at or before the expert's anchor (nothing after it is read)."""
    waypoints = flight["waypoints"]
    start = float(waypoints[0][0])
    index = 0
    for i, row in enumerate(waypoints):
        if float(row[0]) - start <= seconds_after_first + 1e-9:
            index = i
        else:
            break
    return index


def train_heads(s: RunwaySamples, part: str) -> tuple[ListwiseBooster, HistGradientBoostingClassifier]:
    """R1.1b's ``r11_lift`` and R1's head on the partition's training samples — as they were trained."""
    train = s.folds[part] == "train"
    prior = prior_share(s.contexts[part].majority_counts, s.candidates)
    share, base = airline_shares(s.operators, s.y, s.days, s.keys, train, len(s.candidates))
    table = candidate_rows(s.space, s.X, group_of=s.group_of, prior_share=prior,
                           b1_pick=s.picks[part]["B1_active_config"], minutes_since=s.minutes_since,
                           airline_share=share)
    lift = ListwiseBooster(**HEAD_SETTINGS).fit(head_table(table, HEAD, base)[train], s.y[train])
    r1 = HistGradientBoostingClassifier(**HGB_SETTINGS).fit(s.X[train], s.y[train])
    return lift, r1


def query(
    s: RunwaySamples, part: str, queries: list[tuple[str, int, Any]],
    lift: ListwiseBooster, r1: HistGradientBoostingClassifier,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Both heads' probabilities ``[m, C]`` and the partition's rule picks for (flight key, raw
    waypoint index, anchor time) queries — the same causal features the heads were trained on."""
    index_of = {r: i for i, r in enumerate(s.candidates)}
    context, rules = s.pool.rules, s.contexts[part]
    flat, since, b1, picks = [], [], [], []
    for key, index, t in queries:
        flight = s.usable[key]
        sector = s.pool.sectors[key]
        flat.append(anchor_features(s.space, context, flight, index, sector=sector, anchor_time=t))
        since.append(minutes_since_each(context, t, s.candidates))
        rule_picks = rules.picks(t, sector=sector, track_course_deg=track_course_at(flight["waypoints"], index))
        b1.append(index_of[rule_picks["B1_active_config"].runway])
        picks.append(rule_picks)
    flat_arr = np.vstack(flat)
    m = len(queries)
    # the queries join the airline count's roster as non-training rows: read, never counted
    share, base = airline_shares(
        np.concatenate([s.operators, [airline(s.usable[k]) for k, _, _ in queries]]),
        np.concatenate([s.y, np.zeros(m, dtype=s.y.dtype)]),
        np.concatenate([s.days, [operational_day(parse_utc(s.usable[k]["landing_time_utc"])) for k, _, _ in queries]]),
        np.concatenate([s.keys, [f"query:{k}" for k, _, _ in queries]]),
        np.concatenate([s.folds[part] == "train", np.zeros(m, dtype=bool)]),
        len(s.candidates),
    )
    prior = prior_share(rules.majority_counts, s.candidates)
    table = candidate_rows(s.space, flat_arr, group_of=s.group_of, prior_share=prior, b1_pick=np.asarray(b1),
                           minutes_since=np.asarray(since, dtype=np.float64), airline_share=share[-m:])
    p_lift = lift.predict_proba(head_table(table, HEAD, base[-m:]))
    p_r1 = np.zeros((m, len(s.candidates)))
    p_r1[:, r1.classes_] = r1.predict_proba(flat_arr)
    return p_lift, p_r1, picks


def _top(prob: dict[str, float], available: set[str]) -> str:
    return max(sorted(available), key=lambda r: prob[r])


def _expected(prob: dict[str, float], rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The belief-weighted error over the candidates that have a forecast (renormalised)."""
    total = sum(prob[r] for r in rows)
    return {metric: float(sum(prob[r] * rows[r][metric] for r in rows) / total) for metric in ("fde_m", "ade_m")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_sample_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="the plan expert (a threshold-anchored plan checkpoint)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    s = build_samples(args)
    airport = s.airport
    model, config, normalizer, payload = load_checkpoint(args.checkpoint)
    if config.prediction_output != PREDICTION_PLAN:
        parser.error("R2a flies the PLAN expert")
    manifest_path = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    provenance = arrival_data_provenance(manifest_path, eligibility_rosters=[default_lateral_pass_roster_path(manifest_path)])
    require_matching_data_provenance(payload, provenance, allow_subset=True)
    device = resolve_device(args.device)
    model = model.to(device)
    targets = json.loads(manifest_path.read_text(encoding="utf-8"))["runway_targets"]

    val_keys = {k.split(":", 1)[1] for k in payload["split"]["val"] if k.startswith(f"{airport}:")}
    partition_of = evaluation_flights(s, val_keys)
    flights = {key: s.usable[key] for key in sorted(partition_of)}
    print(f"{airport}: {len(flights)} evaluation flights (expert validation x a partition's validation days; "
          f"day_a {sum(p == 'day_a' for p in partition_of.values())}, day_b {sum(p == 'day_b' for p in partition_of.values())})")

    unflyable = {}
    for runway in s.candidates:
        error = skeleton_error(list(flights.values()), runway, targets[runway], config, airport)
        if error is not None:
            unflyable[runway] = error
    candidates = [r for r in s.candidates if r not in unflyable]
    anchor = default_anchor(config)
    points = config.validation_common_grid_points
    per_candidate: dict[str, dict[str, Any]] = {}
    for runway in candidates:
        clones = [{**flight, "runway": runway, "runway_target": targets[runway]} for flight in flights.values()]
        series, report = build_series(clones, config, airport=airport, aircraft_type=config.aircraft_type)
        print(f"  {runway}: {report.format().splitlines()[0]}")
        forecasts = forecast_approaches(model, series, config, normalizer, device=device)
        per_candidate[runway] = {identity(x.scenario.source): (x, f) for x, f in zip(series, forecasts, strict=True)}

    # the anchor each flight is read at, from its own runway's series (the expert's L-1)
    scored: dict[str, dict[str, Any]] = {}
    for key, flight in flights.items():
        ident = identity(flight)
        assigned = flight["runway"]
        if assigned in unflyable or ident not in per_candidate[assigned]:
            continue
        truth, _ = per_candidate[assigned][ident]
        rows = {r: hypothesis_row(table[ident][1], table[ident][0], truth, points=points)
                for r, table in per_candidate.items() if ident in table}
        seconds = float(truth.times[anchor])
        t = parse_utc(flight["entry_time_utc"]) + timedelta(seconds=float(flight["waypoints"][0][0]) + seconds)
        scored[key] = {"flight": flight, "truth": truth, "rows": rows, "anchor_time": t,
                       "index": anchor_waypoint(flight, seconds)}

    out_flights: list[dict[str, Any]] = []
    for part in PARTS:
        keys = [k for k in scored if partition_of[k] == part]
        if not keys:
            continue
        lift, r1 = train_heads(s, part)
        p_lift, p_r1, rule_picks = query(s, part, [(k, scored[k]["index"], scored[k]["anchor_time"]) for k in keys], lift, r1)
        for i, key in enumerate(keys):
            item = scored[key]
            rows, flight = item["rows"], item["flight"]
            available = set(rows)
            probs = {HEAD: dict(zip(s.candidates, p_lift[i].tolist())), "r1": dict(zip(s.candidates, p_r1[i].tolist()))}
            picks = {"assigned": flight["runway"], HEAD: _top(probs[HEAD], available), "r1": _top(probs["r1"], available),
                     **{rule: pick.runway for rule, pick in rule_picks[i].items()},
                     "oracle_fde": min(sorted(available), key=lambda r: rows[r]["fde_m"])}
            out_flights.append({
                "identity": identity(flight), "flight_key": key, "assigned": flight["runway"], "partition": part,
                "anchor_time_utc": item["anchor_time"].isoformat(), "anchor_waypoint_index": item["index"],
                "difficulty": approach_difficulty(item["truth"], anchor).to_dict(),
                "hypotheses": rows, "probabilities": probs, "picks": picks,
                "expected": {head: _expected(probs[head], rows) for head in (HEAD, "r1")},
                "context_fallback": {rule: pick.fallback for rule, pick in rule_picks[i].items()},
            })

    summary = summarise(out_flights, list(SELECTORS))
    print_summary(summary, list(SELECTORS))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runway_intent_r2.json").write_text(json.dumps({
        "schema_version": SCHEMA,
        "airport": airport,
        "checkpoint": str(args.checkpoint),
        "candidates": candidates,
        "unflyable_candidates": unflyable,
        "selectors": list(SELECTORS),
        "runway_head": {"name": HEAD, "settings": HEAD_SETTINGS, "reference": "r1", "reference_settings": HGB_SETTINGS,
                        "asked_at": "the expert's anchor (L-1), from the last raw track point at or before it"},
        "evaluation": {
            "rule": "the expert's per-flight validation split x the operating days that are a day partition's "
                    "validation days (day_a first); each flight read by that partition's head and rules",
            "flights": len(out_flights),
            "by_partition": {part: sum(f["partition"] == part for f in out_flights) for part in PARTS},
            "unscored": len(flights) - len(out_flights),
        },
        "summary": summary,
        "flights": out_flights,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'runway_intent_r2.json'}: {len(out_flights)} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
