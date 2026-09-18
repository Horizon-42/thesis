"""Runway-intent R2c (part 2): the lock rules end to end — the plan expert re-anchored at every ask and flown to the runway each rule holds there.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §3.4 / §16. The heads are asked at the expert's
anchor and every `ASK_S` after it (R2's query), on R2a's roster (flights neither model trained on,
each read by its partition's head); every lock rule of `runway_intent_r2c.lock_picks` names a runway
at every ask (``first`` = the pick at the expert's anchor). A forecast is fixed by the flight, the
anchor and the runway, so the expert is flown only where some rule holds a runway other than the
known one — under every runway a rule holds there, and the known runway: an ask where every rule
holds the known runway costs every rule nothing. Two readings, named apart: each rule's cost against
the known runway averaged over ALL asks (zeros included — an ask where every rule agrees on a wrong
runway costs them all alike), and the difference BETWEEN rules on the asks where they differ.

    python run_ts.py runway_intent_r2c_e2e --airport KRDU --checkpoint <plan checkpoint.pt> \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2c_20260913/KRDU_e2e
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.data.data_provenance import arrival_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series
from ts_transformer.data.lateral_eligibility import default_lateral_pass_roster_path
from ts_transformer.data.runway_context import parse_utc
from ts_transformer.experiments.runway_hypotheses import HARVEST_ROOT, hypothesis_row, identity
from ts_transformer.experiments.runway_intent_r1 import add_sample_arguments, build_samples
from ts_transformer.experiments.runway_intent_r2 import anchor_waypoint, evaluation_flights, query, train_heads
from ts_transformer.experiments.runway_intent_r2c import ASK_S, lock_picks
from ts_transformer.inference.forecast import default_anchor, forecast_approaches
from ts_transformer.training.train import load_checkpoint

SCHEMA = "ts-runway-intent-r2c-e2e-v1"
#: an ask needs at least this much observed track after it to be scored
MIN_FUTURE_S = 60.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_sample_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    s = build_samples(args)
    airport = s.airport
    model, config, normalizer, payload = load_checkpoint(args.checkpoint)
    manifest_path = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    require_matching_data_provenance(
        payload, arrival_data_provenance(manifest_path, eligibility_rosters=[default_lateral_pass_roster_path(manifest_path)]),
        allow_subset=True)
    device = resolve_device(args.device)
    model = model.to(device)
    targets = json.loads(manifest_path.read_text(encoding="utf-8"))["runway_targets"]
    val_keys = {k.split(":", 1)[1] for k in payload["split"]["val"] if k.startswith(f"{airport}:")}
    partition_of = evaluation_flights(s, val_keys)
    flights = {key: s.usable[key] for key in sorted(partition_of)}
    step = int(round(ASK_S / config.dt_s))
    first = default_anchor(config)
    min_future = int(round(MIN_FUTURE_S / config.dt_s))

    # every flight's series under every candidate (the same track, charted at each threshold)
    series: dict[str, dict[str, Any]] = {}
    for runway in s.candidates:
        clones = [{**flight, "runway": runway, "runway_target": targets[runway]} for flight in flights.values()]
        built, _report = build_series(clones, config, airport=airport, aircraft_type=config.aircraft_type)
        series[runway] = {identity(x.scenario.source): x for x in built}

    # the asks: the expert's anchor and every ASK_S after it, while MIN_FUTURE_S of track is left
    plan: dict[str, list[int]] = {}
    for key, flight in flights.items():
        ident = identity(flight)
        truth = series[flight["runway"]].get(ident)
        if truth is None:
            continue
        plan[key] = list(range(first, len(truth.times) - min_future, step))

    # the heads at every ask, by partition
    picks: dict[str, dict[str, list[int]]] = {}
    peaks: dict[str, list[float]] = {}
    index_of = {r: i for i, r in enumerate(s.candidates)}
    for part in ("day_a", "day_b"):
        keys = [k for k in plan if partition_of[k] == part and plan[k]]
        if not keys:
            continue
        lift, r1 = train_heads(s, part)
        queries = []
        for key in keys:
            flight = s.usable[key]
            truth = series[flight["runway"]][identity(flight)]
            entry = parse_utc(flight["entry_time_utc"]) + timedelta(seconds=float(flight["waypoints"][0][0]))
            for a in plan[key]:
                seconds = float(truth.times[a])
                queries.append((key, anchor_waypoint(flight, seconds), entry + timedelta(seconds=seconds)))
        p_lift, _p_r1, _rules = query(s, part, queries, lift, r1)
        row = 0
        for key in keys:
            n = len(plan[key])
            tops = [int(i) for i in p_lift[row:row + n].argmax(axis=1)]
            peaks[key] = [float(v) for v in p_lift[row:row + n].max(axis=1)]
            picks[key] = lock_picks(tops, peaks[key])
            row += n

    # forecasts wherever some rule holds a runway other than the known one, grouped by anchor
    rules = list(next(iter(picks.values())).keys()) if picks else []
    needed: dict[int, set[tuple[str, str]]] = defaultdict(set)
    every_ask: list[tuple[str, int]] = []
    for key, by_rule in picks.items():
        known = s.usable[key]["runway"]
        for j, a in enumerate(plan[key]):
            every_ask.append((key, j))
            chosen = {s.candidates[by_rule[rule][j]] for rule in rules}
            if chosen != {known}:
                for runway in chosen | {known}:
                    needed[a].add((key, runway))
    errors: dict[tuple[str, int, str], dict[str, float]] = {}
    for a, pairs in sorted(needed.items()):
        # every candidate's series starts at the same first point on the same grid, so anchor ``a`` is
        # one instant in all of them; one cut short of it (charted at another threshold) has no forecast there
        items = [(key, runway, series[runway][identity(s.usable[key])]) for key, runway in sorted(pairs)
                 if identity(s.usable[key]) in series[runway]
                 and a < len(series[runway][identity(s.usable[key])].times) - 1]
        if not items:
            continue
        forecasts = forecast_approaches(model, [x for _, _, x in items], config, normalizer, anchor=a, device=device)
        for (key, runway, x), forecast in zip(items, forecasts, strict=True):
            truth = series[s.usable[key]["runway"]][identity(s.usable[key])]
            errors[(key, a, runway)] = hypothesis_row(forecast, x, truth, points=config.validation_common_grid_points)

    def fde(key: str, j: int, runway: str) -> float | None:
        return errors.get((key, plan[key][j], runway), {}).get("fde_m")

    # an ask is scored once every forecast it needs exists, for every rule alike
    delta: dict[str, list[float]] = {rule: [] for rule in rules}
    right: dict[str, list[bool]] = {rule: [] for rule in rules}
    between: dict[str, list[float]] = {rule: [] for rule in rules} | {"known": []}
    differing: list[tuple[str, int]] = []
    dropped = needing = 0
    for key, j in every_ask:
        known = s.usable[key]["runway"]
        held = {rule: s.candidates[picks[key][rule][j]] for rule in rules}
        if set(held.values()) == {known}:
            for rule in rules:
                delta[rule].append(0.0)
                right[rule].append(True)
            continue
        needing += 1
        base = fde(key, j, known)
        values = {rule: fde(key, j, runway) for rule, runway in held.items()}
        if base is None or any(v is None for v in values.values()):
            dropped += 1
            continue
        for rule in rules:
            delta[rule].append(values[rule] - base)
            right[rule].append(held[rule] == known)
        if len(set(held.values())) > 1:
            differing.append((key, j))
            between["known"].append(base)
            for rule in rules:
                between[rule].append(values[rule])
    summary = {
        "flights": len(plan), "asks": len(every_ask), "asks_needing_a_forecast": needing,
        "asks_dropped": dropped, "asks_scored": len(every_ask) - dropped,
        "flights_where_rules_differ": len({k for k, _ in differing}), "asks_where_rules_differ": len(differing),
        "end_to_end_over_all_asks": {
            rule: {"fde_delta_vs_known_mean": float(np.mean(delta[rule])) if delta[rule] else None,
                   "runway_accuracy": float(np.mean(right[rule])) if right[rule] else None}
            for rule in rules},
        "between_rules_on_differing_asks": {
            "fde_mean": {name: float(np.mean(v)) if v else None for name, v in between.items()}},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runway_intent_r2c_e2e.json").write_text(json.dumps({
        "schema_version": SCHEMA, "airport": airport, "checkpoint": str(args.checkpoint),
        "ask_every_s": ASK_S, "min_future_s": MIN_FUTURE_S, "rules": rules, "summary": summary,
        "differing": [{"flight_key": k, "ask": j, "anchor": plan[k][j],
                       "runways": {r: s.candidates[picks[k][r][j]] for r in rules}, "known": s.usable[k]["runway"],
                       "fde_m": {r: fde(k, j, s.candidates[picks[k][r][j]]) for r in rules},
                       "known_fde_m": fde(k, j, s.usable[k]["runway"])}
                      for k, j in differing],
    }, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
