#!/usr/bin/env python
"""Summarize a compare_plants.py run: evaluation verdicts per plant, paired transitions, trajectory differences."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
PLANTS = ("point-mass", "first-order-lag")
meta = json.loads((out / "meta.json").read_text())
rows = json.loads((out / "rows.json").read_text())
pairs = json.loads((out / "pairs.json").read_text())
airports = sorted({r["airport"] for r in rows["point-mass"]})

# evaluation rows per plant, keyed by flight_key
ev: dict[str, dict[str, dict]] = {p: {} for p in PLANTS}
reports = {}
for p in PLANTS:
    for a in airports:
        rep = json.loads((out / p / "records" / a / "evaluation_report.json").read_text())
        reports[(p, a)] = rep
        for t in rep["trajectories"]:
            ev[p][t["flight_key"]] = t
outcome = {p: {r["flight_key"]: r["outcome"] for r in rows[p]} for p in PLANTS}
stratum = {r["flight_key"]: r["stratum"] for r in rows["point-mass"]}
airport_of = {r["flight_key"]: r["airport"] for r in rows["point-mass"]}
keys = sorted(outcome["point-mass"])


def q(a, qs=(50, 90, 95, 100)):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    return {f"p{x}": float(np.percentile(a, x)) for x in qs} | {"mean": float(a.mean()), "n": int(len(a))} if len(a) else {}


def fmt(d):
    return "  ".join(f"{k} {v:8.2f}" if k != "n" else f"n={v}" for k, v in d.items())


lines = []
P = lines.append
P(f"# point-mass vs first-order-lag — {len(keys)} closed-loop flights ({meta['split']}, {meta['per_airport']}/airport, seed {meta['seed']})")
P(f"lag time constants (thrust, bank, load) s = {meta['plants']['first-order-lag']['time_constants_s_thrust_bank_load']}")
P("")
P("## 1. Outcome of the executor flight (judge, vocabulary §5.8)")
for p in PLANTS:
    c = Counter(outcome[p].values())
    P(f"- {p:16s} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
P("")
P("## 2. `evaluation` (terminal-approach-evaluation-v9) over the flights it graded")
P("| plant | flights | pass | fail | indet. | lateral pass | vertical pass | speed pass |")
P("|---|---|---|---|---|---|---|---|")
for p in PLANTS:
    t = [ev[p][k] for k in keys if k in ev[p]]
    v = Counter(x["verdict"] for x in t)
    g = lambda f: Counter(x[f] for x in t)
    P(f"| {p} | {len(t)} | {v['pass']} | {v['fail']} | {v['indeterminate']} | "
      f"{g('lateral_result')['pass']} | {g('vertical_result')['pass']} | {g('speed_result')['pass']} |")
P("")
P("Terminal error at the threshold plane (graded flights only):")
for p in PLANTS:
    t = [ev[p][k] for k in keys if k in ev[p]]
    P(f"- {p:16s} |cross-track| m: {fmt(q([abs(x['cross_track_m']) if x.get('cross_track_m') is not None else None for x in t]))}")
    P(f"{'':20s}vertical m (signed): {fmt(q([x.get('vertical_m') for x in t], (5, 50, 95)))}")
    P(f"{'':20s}final_time_s: {fmt(q([x.get('final_time_s') for x in t], (50, 95)))}")
    P(f"{'':20s}crossing speed m/s: {fmt(q([x.get('crossing_speed_ms') for x in t], (5, 50, 95)))}")
P("")
P("## 3. Paired by flight (same words, same start)")
confusion = Counter()
for k in keys:
    a, b = ev["point-mass"].get(k), ev["first-order-lag"].get(k)
    confusion[(a["verdict"] if a else "no record", b["verdict"] if b else "no record")] += 1
P("evaluation verdict, point-mass → lag:")
for (a, b), n in sorted(confusion.items(), key=lambda x: -x[1]):
    P(f"  {a:12s} → {b:12s} {n}")
trans = Counter((outcome["point-mass"][k], outcome["first-order-lag"][k]) for k in keys)
P("flight outcome, point-mass → lag:")
for (a, b), n in sorted(trans.items(), key=lambda x: -x[1]):
    P(f"  {a:18s} → {b:24s} {n}")
both = [k for k in keys if all(k in ev[p] and all(ev[p][k].get(f) is not None for f in ("cross_track_m","vertical_m","final_time_s","crossing_speed_ms")) for p in PLANTS)]
d_cross = [abs(ev["first-order-lag"][k]["cross_track_m"]) - abs(ev["point-mass"][k]["cross_track_m"]) for k in both]
d_vert = [ev["first-order-lag"][k]["vertical_m"] - ev["point-mass"][k]["vertical_m"] for k in both]
d_time = [ev["first-order-lag"][k]["final_time_s"] - ev["point-mass"][k]["final_time_s"] for k in both]
d_speed = [ev["first-order-lag"][k]["crossing_speed_ms"] - ev["point-mass"][k]["crossing_speed_ms"] for k in both]
P(f"lag − point-mass on {len(both)} flights both graded: ")
P(f"  Δ|cross-track| m   {fmt(q(d_cross, (5, 50, 95)))}")
P(f"  Δvertical m        {fmt(q(d_vert, (5, 50, 95)))}")
P(f"  Δfinal_time s      {fmt(q(d_time, (5, 50, 95)))}")
P(f"  Δcrossing speed    {fmt(q(d_speed, (5, 50, 95)))}")
P("")
P("## 4. The two flown tracks against each other")
P("Path geometry is TIME-FREE (each lag point against the point-mass polyline: its horizontal distance to it and its height")
P("minus the point-mass height at the projection); timing is the difference of the flights' end times (lag − point-mass).")
P("| group | n | horizontal offset, max over the flight m (p50 / p90 / max) | mean offset m (p50 / p90) | |height diff| max m (p50 / p90 / max) | end-time lag s (p50 / p90 / max) |")
P("|---|---|---|---|---|---|")
landed_both = [x for x in pairs if x["outcome_point_mass"] == "landed" and x["outcome_lag"] == "landed"]
P(f"Flights that landed under BOTH plants: {len(landed_both)} of {len(pairs)}; the other {len(pairs) - len(landed_both)} "
  f"are listed in section 3 (a lag flight that does not land leaves the path, so its offsets measure that, not the lag).")
groups = defaultdict(list)
for x in landed_both:
    for g in ("all", x["airport"], x["stratum"]):
        groups[g].append(x)
for g, xs in sorted(groups.items(), key=lambda kv: (kv[0] != "all", kv[0])):
    f = lambda vals, qs=(50, 90, 100): " / ".join(f"{np.percentile(vals, s):.1f}" for s in qs)
    lag = [x["end_s_lag"] - x["end_s_point_mass"] for x in xs]
    P(f"| {g} | {len(xs)} | {f([x['path_max_offset_m'] for x in xs])} | {f([x['path_mean_offset_m'] for x in xs], (50, 90))} | "
      f"{f([x['path_max_height_m'] for x in xs])} | {f(lag)} |")
Path(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
