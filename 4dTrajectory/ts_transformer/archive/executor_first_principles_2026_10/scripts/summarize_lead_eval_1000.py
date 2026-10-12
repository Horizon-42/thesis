#!/usr/bin/env python
"""Tables of `lead_eval_1000` results: per column, per value, per row interval — evaluation pass on all flights and on
the flights whose OBSERVED track passes (observed grades from the harvest's own evaluation reports, given as JSON files
{flight_key: {verdict, ...}}), the paired difference to the base (net, z), the gate failures, the column's corrections
per sentence, its envelope on the observed tracks and its error around its observed words (p50 at 0..+4 rows, share
beyond tolerance at +1..+3)."""
import glob
import json
import math
import sys
from pathlib import Path

results = json.loads(Path(sys.argv[1]).read_text())
observed = {}
for f in sys.argv[2:]:
    observed.update(json.loads(Path(f).read_text()))
cell = {(r["arm"], r["row_interval_s"]): r for r in results}
good = {k for k, v in observed.items() if v["verdict"] == "pass"}


def paired(a, b, keys):
    x = sum(a[k]["verdict"] == "pass" and b[k]["verdict"] != "pass" for k in keys)
    y = sum(b[k]["verdict"] == "pass" and a[k]["verdict"] != "pass" for k in keys)
    return y - x, (y - x) / math.sqrt(x + y) if x + y else 0.0


for column, base_value in (("heading", 4.0), ("angle", 0.0), ("speed", 0.0)):
    print(f"\n## {column} lead (base: heading 4, angle 0, speed 0)")
    for interval in (2.0, 4.0):
        print(f"\n### Δ {interval:g} s\n")
        print("| lead s | pass all | net (z) | pass obs-pass | net (z) | lat / vert / speed fails (all) | corr/sent | envelope | err p50 0/+1/+2/+3/+4 | beyond tol +1/+2/+3 % |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        base = cell[("base", interval)]
        for value in (0.0, 2.0, 4.0, 6.0, 8.0):
            r = base if value == base_value else cell[(f"{column}{value:g}", interval)]
            v, bv = r["verdicts"], base["verdicts"]
            keys = v.keys() & bv.keys()
            gk = keys & good
            n_all, z_all = paired(bv, v, keys)
            n_g, z_g = paired(bv, v, gk)
            f = r["fails"]
            env = r["envelopes"][column]
            share = env.get("row_share", env.get("share"))
            aw = r["after_words"][column]
            p50 = " / ".join("—" if aw[str(o)] is None else f"{aw[str(o)]['p50']:.1f}" for o in (0, 1, 2, 3, 4))
            bt = " / ".join("—" if aw[str(o)] is None else f"{100 * aw[str(o)]['beyond_tolerance']:.0f}" for o in (1, 2, 3))
            corr = r["corrections_per_sentence"].get(column, 0.0)
            mark = " (base)" if value == base_value else ""
            print(f"| {value:g}{mark} | {r['eval_pass']}/{r['graded']} | {n_all:+d} ({z_all:.1f}) | "
                  f"{sum(v[k]['verdict'] == 'pass' for k in gk)}/{len(gk)} | {n_g:+d} ({z_g:.1f}) | "
                  f"{f.get('lateral', 0)} / {f.get('vertical', 0)} / {f.get('speed', 0)} | {corr:.1f} | "
                  f"{100 * share:.1f} % | {p50} | {bt} |")
