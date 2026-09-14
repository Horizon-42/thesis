"""Runway-intent R3.2 diagnosis: where the time closure loses an assigned arrival time between asks.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §18.2. Re-flies R3's plan (each flight's scheduled runway
and time from R3's artifact, the R2b expert that flew it) under three lockstep settings — the baseline
(R3's own: 30 s asks, adopt a new order at once), an order hold that never adopts a new order (re-lays
only at a phase change or a drift), and 10 s asks — and reads the delivery (flown − scheduled) for the
flights the schedule delayed and for the rest. On the baseline it reads the per-step closure records of
the flights that land > 10 s early: where X (assigned remaining − closed time; > 0 early) first passes the
tolerance, what the order was there, and X across the switch from the last instruction leg to the
closing — the jump that located the defect (an instruction leg's route is timed to the threshold through
the `LEG_EXTENSION_M` placeholder past its fix).

    python run_ts.py runway_intent_r32_diagnosis --airport KSMF --airport KSTL \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r32_20260914/diagnosis
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.experiments.runway_hypotheses import HARVEST_ROOT, identity
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.inference.forecast import cut_at_threshold_crossing, default_anchor
from ts_transformer.outputs.plan.guidance.timing import TIME_TOLERANCE_S
from ts_transformer.outputs.plan.strategy import Assignment, SkeletonCache, rolled_predictions_lockstep
from ts_transformer.training.train import load_checkpoint

SCHEMA = "ts-runway-intent-r32-diagnosis-v1"
EXPERIMENTS = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments"
VARIANTS = {"baseline": {"hold_asks": 1, "step_s": 30.0}, "never_adopt": {"hold_asks": 10 ** 6, "step_s": 30.0},
            "asks_10s": {"hold_asks": 1, "step_s": 10.0}}
DELIVERED_S = 10.0
DELAYED_S = 1.0


def fly(airport: str, r3: dict[str, Any], r2: dict[str, Any], checkpoint: Path, device: str) -> dict[str, dict[str, Any]]:
    """Every R3 flight re-flown on its scheduled runway at its scheduled time under each lockstep setting."""
    model, config, normalizer, _payload = load_checkpoint(str(checkpoint))
    model = model.to(resolve_device(device))
    manifest = HARVEST_ROOT / airport / "arrivals" / "manifest.json"
    targets = json.loads(manifest.read_text(encoding="utf-8"))["runway_targets"]
    rows = {r["flight_key"]: r for r in r3["flights"]}
    loaded = load_flight_dicts(manifest, include_flight_keys={f"{airport}:{k}" for k in rows}, verbose=False)
    by_identity = {identity(f): f for f in loaded}
    flights = {f["flight_key"]: by_identity[f["identity"]] for f in r2["flights"]}
    anchor = default_anchor(config)
    out: dict[str, dict[str, Any]] = {name: {} for name in VARIANTS}
    for runway in sorted({r["scheduled"]["runway"] for r in rows.values()}):
        keys = sorted(k for k, r in rows.items() if r["scheduled"]["runway"] == runway)
        built, _report = build_series([{**flights[k], "runway": runway, "runway_target": targets[runway]} for k in keys],
                                      config, airport=airport, aircraft_type=config.aircraft_type)
        series = {identity(x.scenario.source): x for x in built}
        group = [series[identity(flights[k])] for k in keys]
        skeletons = SkeletonCache()
        assignments = [Assignment(arrival_time_s=float(x.times[anchor]) + (rows[k]["scheduled"]["time_s"] - rows[k]["anchor_s"]))
                       for k, x in zip(keys, group, strict=True)]
        for name, options in VARIANTS.items():
            started = time.perf_counter()
            rolled = rolled_predictions_lockstep(model, group, config, normalizer, [anchor] * len(group),
                                                 next(model.parameters()).device, [skeletons.for_series(x) for x in group],
                                                 assignments=assignments, **options)
            for k, x, flight in zip(keys, group, rolled, strict=True):
                cut = cut_at_threshold_crossing(flight.forecast, x)
                steps = [rec for rec in flight.orders if "closure" in rec]
                out[name][k] = {
                    "landed": bool(cut.truncated_at_threshold),
                    "delivery_s": rows[k]["anchor_s"] + float(cut.final_time_s) - rows[k]["scheduled"]["time_s"],
                    "delay_s": rows[k]["scheduled"]["delay_s"],
                    "x_s": [rec["closure"]["unabsorbed_s"] for rec in steps],
                    "speed_factor": [rec["closure"]["speed_factor"] for rec in steps],
                    "stretch_laid_m": [rec["closure"]["stretch_m"] for rec in steps],
                    "closing": [rec["instruction"] is None for rec in steps],
                    "remaining_m": [rec["remaining_m"] for rec in steps],
                }
            print(f"  {airport} {runway} {name}: {len(group)} flights in {time.perf_counter() - started:.0f} s", flush=True)
    return out


def delivery(flown: dict[str, dict[str, Any]]) -> dict[str, Any]:
    landed = [v for v in flown.values() if v["landed"]]
    out: dict[str, Any] = {"unlanded": len(flown) - len(landed)}
    for name, members in (("delayed", [v for v in landed if v["delay_s"] > DELAYED_S]),
                          ("undelayed", [v for v in landed if v["delay_s"] <= DELAYED_S])):
        d = np.array([v["delivery_s"] for v in members])
        out[name] = {"n": len(members), "within_10s": float(np.mean(np.abs(d) <= DELIVERED_S)) if len(d) else float("nan"),
                     "p50_s": float(np.median(d)) if len(d) else float("nan"),
                     "early_share": float(np.mean(d < -DELIVERED_S)) if len(d) else float("nan")}
    return out


def early_flights(flown: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The flights landing > 10 s early: where X first passes the tolerance, and X across the switch
    from the last instruction leg to the closing (against the flights delivered within 10 s)."""
    out: dict[str, Any] = {}
    for label, pick in (("early", lambda v: v["landed"] and v["delivery_s"] < -DELIVERED_S),
                        ("delivered", lambda v: v["landed"] and abs(v["delivery_s"]) <= DELIVERED_S)):
        members = [v for v in flown.values() if pick(v)]
        before, jump, x_last, first_closing, first_km, first_factor = [], [], [], [], [], []
        for v in members:
            x = np.array(v["x_s"])
            closing = np.array(v["closing"])
            x_last.append(float(x[-1]))
            over = np.nonzero(x > TIME_TOLERANCE_S)[0]
            if len(over):
                first_closing.append(bool(closing[over[0]]))
                first_km.append(v["remaining_m"][over[0]] / 1000.0)
                first_factor.append(v["speed_factor"][over[0]])
            switch = [k for k in range(1, len(closing)) if closing[k] and not closing[k - 1]]
            if switch:
                before.append(float(x[switch[0] - 1]))
                jump.append(float(x[switch[0]] - x[switch[0] - 1]))
        out[label] = {
            "flights": len(members), "with_a_switch": len(jump),
            "x_before_switch_p50_s": float(np.median(before)) if before else float("nan"),
            "switch_jump_p50_s": float(np.median(jump)) if jump else float("nan"),
            "switch_jump_p90_s": float(np.percentile(jump, 90)) if jump else float("nan"),
            "x_last_p50_s": float(np.median(x_last)) if x_last else float("nan"),
            "first_over_tolerance_on_closing": float(np.mean(first_closing)) if first_closing else float("nan"),
            "first_over_tolerance_remaining_km_p50": float(np.median(first_km)) if first_km else float("nan"),
            "first_over_tolerance_speed_factor_p50": float(np.median(first_factor)) if first_factor else float("nan"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--airport", action="append", required=True)
    parser.add_argument("--r3-dir", default=str(EXPERIMENTS / "runway_intent_r3_20260914"))
    parser.add_argument("--r2b-dir", default=str(EXPERIMENTS / "runway_intent_r2b_20260913"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {"schema_version": SCHEMA, "variants": VARIANTS, "airports": {}}
    lines = []
    for airport in (a.upper() for a in args.airport):
        r3 = json.loads((Path(args.r3_dir) / airport / "runway_intent_r3.json").read_text(encoding="utf-8"))
        r2 = json.loads((Path(args.r2b_dir) / airport / "runway_intent_r2.json").read_text(encoding="utf-8"))
        flown = fly(airport, r3, r2, Path(args.r2b_dir) / f"{airport}_day_a_expert" / "checkpoint.pt", args.device)
        reading = {"delivery": {name: delivery(flown[name]) for name in VARIANTS}, "baseline_early": early_flights(flown["baseline"])}
        document["airports"][airport] = reading
        (out / f"{airport}_flights.json").write_text(json.dumps(flown), encoding="utf-8")
        lines.append(f"== {airport}")
        for name, d in reading["delivery"].items():
            lines.append(f"  {name:12s} undelayed {100 * d['undelayed']['within_10s']:.1f}% (n={d['undelayed']['n']}), "
                         f"delayed {100 * d['delayed']['within_10s']:.1f}% (n={d['delayed']['n']}) within 10 s; unlanded {d['unlanded']}")
        for label, e in reading["baseline_early"].items():
            lines.append(f"  baseline {label:9s} n={e['flights']}: X before the switch p50 {e['x_before_switch_p50_s']:+.1f} s, "
                         f"jump p50 {e['switch_jump_p50_s']:+.1f} (p90 {e['switch_jump_p90_s']:+.1f}), X last p50 {e['x_last_p50_s']:+.1f}; "
                         f"first X > tol on the closing {100 * e['first_over_tolerance_on_closing']:.0f}% at "
                         f"{e['first_over_tolerance_remaining_km_p50']:.1f} km, speed factor {e['first_over_tolerance_speed_factor_p50']:.2f}")
    text = "\n".join(lines)
    print(text)
    (out / "runway_intent_r32_diagnosis.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    (out / "runway_intent_r32_diagnosis.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
