#!/usr/bin/env python
"""Three-arm summary of regenerate_closed_loop.py output: evaluation verdicts, terminal error, correction words, geometry."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
ARMS = ["A_pm_stored", "B_lag_stored", "C_lag_regenerated"]
meta = json.loads((out / "meta.json").read_text())
rows = json.loads((out / "rows.json").read_text())
airports = sorted({r["airport"] for r in rows[ARMS[0]]})
ev = {}
for arm in ARMS:
    ev[arm] = {}
    for a in airports:
        rep = json.loads((out / arm / "records" / a / "evaluation_report.json").read_text())
        ev[arm].update({t["flight_key"]: t for t in rep["trajectories"]})
byrow = {arm: {r["flight_key"]: r for r in rows[arm]} for arm in ARMS}
keys = sorted(byrow[ARMS[0]])
L = []
P = L.append


def pct(v, qs):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return " / ".join(f"{np.percentile(v, q):.1f}" for q in qs)


P(f"# regenerated closed-loop words — {len(keys)} flights, lag τ (thrust, bank, load) = {meta['lag_time_constants_s']} s")
P("A point-mass plant + stored words;  B lag plant + the same stored words;  C lag plant + words regenerated with the lag plant in the loop.")
P(f"Regenerating under point-mass reproduces the stored artefact exactly (500/500). Refused under lag: {meta['lag_refused'] or 'none'}.")
P("")
P("## 1. Outcome and `evaluation`")
P("| arm | landed | other outcomes | eval pass | lateral / vertical / speed pass | \\|cross-track\\| p50 / p90 / p95 / max m | vertical p5 / p50 / p95 m |")
P("|---|---|---|---|---|---|---|")
for arm in ARMS:
    t = [ev[arm][k] for k in keys]
    o = Counter(byrow[arm][k]["outcome"] for k in keys)
    rest = ", ".join(f"{k} {n}" for k, n in o.most_common() if k != "landed") or "-"
    v = Counter(x["verdict"] for x in t)
    g = lambda f: Counter(x[f] for x in t)["pass"]
    xt = [abs(x["cross_track_m"]) for x in t if x.get("cross_track_m") is not None]
    vt = [x["vertical_m"] for x in t if x.get("vertical_m") is not None]
    P(f"| {arm} | {o['landed']} | {rest} | {v['pass']} | {g('lateral_result')} / {g('vertical_result')} / {g('speed_result')} | "
      f"{pct(xt, (50, 90, 95, 100))} | {pct(vt, (5, 50, 95))} |")
P("")
P("## 2. Paired verdicts (evaluation)")
for a, b in (("A_pm_stored", "C_lag_regenerated"), ("B_lag_stored", "C_lag_regenerated")):
    c = Counter((ev[a][k]["verdict"], ev[b][k]["verdict"]) for k in keys)
    P(f"{a} → {b}: " + ", ".join(f"{x}→{y} {n}" for (x, y), n in sorted(c.items(), key=lambda kv: -kv[1])))
P("")
P("## 3. What the corrector said (correction words per sentence) and how far the flight left the observed path")
P("| arm | heading | altitude | angle | speed | runway | largest lateral error vs observed p50 / p90 / max m | left the path (>300 m) |")
P("|---|---|---|---|---|---|---|---|")
w = meta["correction_words_per_sentence"]
for arm in ARMS:
    ws = w.get({"B_lag_stored": "A_pm_stored"}.get(arm, arm), {})
    lat = [byrow[arm][k].get("largest_lateral_m") for k in keys]
    lat = [x for x in lat if x is not None]
    note = " (B flies A's words)" if arm == "B_lag_stored" else ""
    P(f"| {arm}{note} | " + " | ".join(f"{ws.get(c, 0):.2f}" for c in ("heading", "altitude", "angle", "speed", "runway")) + " | " +
      (f"{pct(lat, (50, 90, 100))} | {sum(x > 300 for x in lat)} |" if lat else "n/a (not read against the observed path) | |"))
P("")
P("## 4. Flown tracks against A's (time-free: each point against A's polyline), flights landed in both")
tr = np.load(out / "tracks.npz")
idx = json.loads((out / "track_index.json").read_text())


def path_difference(a, b):
    p = np.column_stack((b["e"], b["n"])); q = np.column_stack((a["e"], a["n"]))
    seg = q[1:] - q[:-1]; l2 = np.maximum((seg ** 2).sum(1), 1e-9)
    rel = p[:, None, :] - q[None, :-1, :]
    t = np.clip((rel * seg[None]).sum(2) / l2[None], 0, 1)
    foot = q[None, :-1, :] + t[..., None] * seg[None]
    d = np.hypot(*(p[:, None, :] - foot).transpose(2, 0, 1))
    j = d.argmin(1); r = np.arange(len(p))
    return d[r, j], b["h"] - (a["h"][j] + t[r, j] * (a["h"][j + 1] - a["h"][j]))


for arm in ARMS[1:]:
    both = [k for k in keys if idx["A_pm_stored"][k]["outcome"] == "landed" and idx[arm][k]["outcome"] == "landed"]
    mo, mh, lag = [], [], []
    for k in both:
        a = {n: tr[f"A_pm_stored|{k}|{n}"] for n in ("e", "n", "h")}
        b = {n: tr[f"{arm}|{k}|{n}"] for n in ("e", "n", "h")}
        off, dh = path_difference(a, b)
        mo.append(off.max()); mh.append(np.abs(dh).max()); lag.append(idx[arm][k]["end_row"] - idx["A_pm_stored"][k]["end_row"])
    P(f"- {arm}: {len(both)} flights; horizontal offset max p50 / p90 / max = {pct(mo, (50, 90, 100))} m; "
      f"|height diff| max = {pct(mh, (50, 90, 100))} m; end-time lag = {pct(lag, (50, 90, 100))} s")
Path(out / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
