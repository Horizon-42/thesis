"""Runway-intent R2c (part 1): the runway belief re-asked every 30 s along the approach — how it evolves, how often its top pick flips, and what each lock rule would fly toward.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §3.4 / §16. The heads (R1.1b's ``r11_lift``, R1's for
reference) and the rules are day_a's, trained as in R2 (`runway_intent_r2.train_heads`), and asked on
every flight of day_a's validation days at the arrival-slice entry and every `ASK_S` after it until
the landing — from the last raw track point at or before each ask, the same causal query as R2
(`runway_intent_r2.query`). The remaining flown path at each ask only BINS the reading (the anytime
grid); it is never a feature. Part 2 (whether a flip costs trajectory error, and a lock rule's
end-to-end effect) needs the experts re-anchored at each ask and follows R2b.

Lock rules read here, each the runway a forecast made at an ask would fly toward:
``never`` (the current top pick), ``first`` (the first ask's pick, kept — here the slice entry; in
part 2 the expert's own anchor, where its asks start), ``p<τ>`` (the current pick until the first ask
whose top probability reaches τ, then kept).

    python run_ts.py runway_intent_r2c --airport KRDU \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2c_20260913/KRDU
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.data.runway_context import parse_utc
from ts_transformer.data.runway_features import remaining_path_m
from ts_transformer.experiments.runway_intent_r1 import add_sample_arguments, build_samples
from ts_transformer.experiments.runway_intent_r2 import HEAD, day_validation_flights, query, train_heads

SCHEMA = "ts-runway-intent-r2c-v1"
ASK_S = 30.0
LOCK_THRESHOLDS = (0.8, 0.9, 0.95)
#: remaining flown path (km) the anytime curve is binned by, far to near
BINS_KM = (40.0, 30.0, 20.0, 15.0, 10.0, 6.0, 3.0, 0.0)
EPS = 1e-6


def asks(flight: dict[str, Any]) -> list[tuple[int, float, float]]:
    """(raw waypoint index, seconds after the first point, remaining flown path m) at the entry and
    every `ASK_S` after it, while the aircraft is still short of its last point (the landing)."""
    waypoints = flight["waypoints"]
    start, end = float(waypoints[0][0]), float(waypoints[-1][0])
    remaining = remaining_path_m(waypoints)
    out, index, t = [], 0, 0.0
    while start + t < end:
        while index + 1 < len(waypoints) and float(waypoints[index + 1][0]) - start <= t + 1e-9:
            index += 1
        out.append((index, t, remaining[index]))
        t += ASK_S
    return out


def lock_picks(tops: list[int], peaks: list[float]) -> dict[str, list[int]]:
    """Each lock rule's runway at every ask of one flight."""
    rules: dict[str, list[int]] = {"never": list(tops), "first": [tops[0]] * len(tops)}
    for tau in LOCK_THRESHOLDS:
        locked: int | None = None
        seq = []
        for top, peak in zip(tops, peaks):
            if locked is None and peak >= tau:
                locked = top
            seq.append(top if locked is None else locked)
        rules[f"p{tau:g}"] = seq
    return rules


def flips(seq: list[int]) -> int:
    return sum(a != b for a, b in zip(seq, seq[1:]))


def bin_of(remaining_m: float) -> str:
    km = remaining_m / 1000.0
    for hi, lo in zip((float("inf"),) + BINS_KM[:-1], BINS_KM):
        if km >= lo:
            return f">{lo:g}km" if hi == float("inf") else f"{lo:g}-{hi:g}km"
    return f"<{BINS_KM[-1]:g}km"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_sample_arguments(parser)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    s = build_samples(args)
    part = "day_a"
    roster = sorted(day_validation_flights(s, part))
    lift, r1 = train_heads(s, part)
    queries, owner = [], []
    for key in roster:
        flight = s.usable[key]
        entry = parse_utc(flight["entry_time_utc"]) + timedelta(seconds=float(flight["waypoints"][0][0]))
        for index, t, remaining in asks(flight):
            queries.append((key, index, entry + timedelta(seconds=t)))
            owner.append((key, t, remaining))
    p_lift, p_r1, rule_picks = query(s, part, queries, lift, r1)
    index_of = {r: i for i, r in enumerate(s.candidates)}
    truth_of = {key: index_of[s.usable[key]["runway"]] for key in roster}

    per_flight: dict[str, dict[str, Any]] = {}
    for i, (key, t, remaining) in enumerate(owner):
        cell = per_flight.setdefault(key, {"t": [], "remaining_m": [], HEAD: [], "r1": [], "B1": [], "B3": [],
                                           "peak": [], "p_true": [], "p_true_r1": []})
        cell["t"].append(t)
        cell["remaining_m"].append(remaining)
        cell[HEAD].append(int(p_lift[i].argmax()))
        cell["r1"].append(int(p_r1[i].argmax()))
        cell["B1"].append(index_of[rule_picks[i]["B1_active_config"].runway])
        cell["B3"].append(index_of[rule_picks[i]["B3_same_sector_last"].runway])
        cell["peak"].append(float(p_lift[i].max()))
        cell["p_true"].append(float(p_lift[i][truth_of[key]]))
        cell["p_true_r1"].append(float(p_r1[i][truth_of[key]]))

    # the anytime curve: every ask, binned by the remaining flown path
    curve: dict[str, dict[str, Any]] = {}
    for key, cell in per_flight.items():
        truth = truth_of[key]
        for j, remaining in enumerate(cell["remaining_m"]):
            b = curve.setdefault(bin_of(remaining), {"asks": 0, "flights": set(),
                                                     **{f"{h}_exact": 0 for h in (HEAD, "r1", "B1", "B3")},
                                                     f"{HEAD}_nll": 0.0, "r1_nll": 0.0})
            b["asks"] += 1
            b["flights"].add(key)
            for h in (HEAD, "r1", "B1", "B3"):
                b[f"{h}_exact"] += cell[h][j] == truth
            b[f"{HEAD}_nll"] -= float(np.log(max(cell["p_true"][j], EPS)))
            b["r1_nll"] -= float(np.log(max(cell["p_true_r1"][j], EPS)))
    order = [bin_of(km * 1000.0 + 1.0) for km in (1e9,) + BINS_KM[:-1]] + [bin_of(0.0)]
    anytime = {}
    for name in dict.fromkeys(order):
        b = curve.get(name)
        if not b:
            continue
        anytime[name] = {
            "asks": b["asks"], "flights": len(b["flights"]),
            **{f"{h}_exact": b[f"{h}_exact"] / b["asks"] for h in (HEAD, "r1", "B1", "B3")},
            f"{HEAD}_nll": b[f"{HEAD}_nll"] / b["asks"], "r1_nll": b["r1_nll"] / b["asks"],
        }

    # flips and the lock rules
    flip_stats: dict[str, Any] = {}
    for h in (HEAD, "r1", "B1"):
        counts = [flips(cell[h]) for cell in per_flight.values()]
        first_right = [cell[h][0] == truth_of[k] for k, cell in per_flight.items()]
        last_right = [cell[h][-1] == truth_of[k] for k, cell in per_flight.items()]
        toward = away = 0
        for k, cell in per_flight.items():
            for a, b in zip(cell[h], cell[h][1:]):
                if a != b:
                    toward += b == truth_of[k]
                    away += a == truth_of[k]
        flip_stats[h] = {
            "flights": len(counts), "flights_with_a_flip": int(sum(c > 0 for c in counts)),
            "flips_per_flight": float(np.mean(counts)), "flips_max": int(max(counts)),
            "first_ask_right": float(np.mean(first_right)), "last_ask_right": float(np.mean(last_right)),
            "flips_toward_the_truth": toward, "flips_away_from_it": away,
        }
    lock_curve: dict[str, dict[str, float]] = {}
    lock_totals: dict[str, dict[str, float]] = {}
    for key, cell in per_flight.items():
        rules = lock_picks(cell[HEAD], cell["peak"])
        for rule, seq in rules.items():
            tot = lock_totals.setdefault(rule, {"asks": 0, "right": 0, "flips": 0})
            tot["asks"] += len(seq)
            tot["right"] += sum(x == truth_of[key] for x in seq)
            tot["flips"] += flips(seq)
            for x, remaining in zip(seq, cell["remaining_m"]):
                c = lock_curve.setdefault(rule, {}).setdefault(bin_of(remaining), [0, 0])
                c[0] += 1
                c[1] += x == truth_of[key]
    locks = {rule: {"accuracy_over_asks": tot["right"] / tot["asks"], "flips_per_flight": tot["flips"] / len(per_flight),
                    "by_bin": {b: lock_curve[rule][b][1] / lock_curve[rule][b][0] for b in dict.fromkeys(order)
                               if b in lock_curve[rule]}}
             for rule, tot in lock_totals.items()}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": SCHEMA, "airport": s.airport, "candidates": s.candidates, "partition": part,
        "ask_every_s": ASK_S, "lock_thresholds": list(LOCK_THRESHOLDS), "bins_km": list(BINS_KM),
        "flights": len(per_flight), "asks": len(queries),
        "anytime": anytime, "flips": flip_stats, "locks": locks,
        "per_flight": {k: {"t": c["t"], "remaining_m": c["remaining_m"],
                           HEAD: [s.candidates[i] for i in c[HEAD]], "r1": [s.candidates[i] for i in c["r1"]],
                           "B1": [s.candidates[i] for i in c["B1"]], "peak": c["peak"], "p_true": c["p_true"],
                           "truth": s.candidates[truth_of[k]]}
                       for k, c in per_flight.items()},
    }
    (out / "runway_intent_r2c.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    lines = [f"{s.airport}: {len(per_flight)} day_a validation-day flights, {len(queries)} asks every {ASK_S:g} s"]
    lines.append("  anytime (exact %): " + "; ".join(
        f"{b} n={c['asks']} {HEAD} {100 * c[f'{HEAD}_exact']:.1f} r1 {100 * c['r1_exact']:.1f} "
        f"B1 {100 * c['B1_exact']:.1f} B3 {100 * c['B3_exact']:.1f}" for b, c in anytime.items()))
    for h, f in flip_stats.items():
        lines.append(f"  flips {h}: {f['flights_with_a_flip']}/{f['flights']} flights flip, {f['flips_per_flight']:.2f} per flight "
                     f"(max {f['flips_max']}), toward/away {f['flips_toward_the_truth']}/{f['flips_away_from_it']}, "
                     f"first ask {100 * f['first_ask_right']:.1f}% -> last {100 * f['last_ask_right']:.1f}%")
    for rule, c in locks.items():
        lines.append(f"  lock {rule}: {100 * c['accuracy_over_asks']:.1f}% of asks right, {c['flips_per_flight']:.2f} flips/flight")
    text = "\n".join(lines)
    (out / "runway_intent_r2c.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
