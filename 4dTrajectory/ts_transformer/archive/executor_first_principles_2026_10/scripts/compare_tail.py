#!/usr/bin/env python
"""Paired comparison of `lead_eval_tail` A and B with `lead_eval_1000_combo`'s combo_a4_s4 (no tail rule): pass counts,
fails by gate, the flights that flip, z of the paired difference; also on the observed-pass flights only."""
import glob
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
import archive_paths   # noqa: E402  (README §2)
RESULTS = archive_paths.RESULTS
SCRATCH = RESULTS / "inputs/observed_1000"   # was the session's scratch area


def cells(path, arm=None):
    return {r["row_interval_s"]: r for r in json.loads(Path(path).read_text()) if arm is None or r["arm"] == arm}


def observed_pass():
    keys = {}
    for f in glob.glob(str(SCRATCH / "obs1000_*.json")):
        keys.update(json.loads(Path(f).read_text()))
    return {k for k, v in keys.items() if (v["verdict"] if isinstance(v, dict) else v) == "pass"}


def compare(base, arm, only=None):
    keys = sorted(set(base["verdicts"]) & set(arm["verdicts"]) & (only if only is not None else set(base["verdicts"])))
    b = [base["verdicts"][k]["verdict"] == "pass" for k in keys]
    a = [arm["verdicts"][k]["verdict"] == "pass" for k in keys]
    gained, lost = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    z = (gained - lost) / math.sqrt(gained + lost) if gained + lost else 0.0
    fails = lambda c: dict(Counter(g for k in keys if c["verdicts"][k]["verdict"] == "fail" for g in c["verdicts"][k]["violations"]))
    return {"flights": len(keys), "base_pass": sum(b), "arm_pass": sum(a), "gained": gained, "lost": lost,
            "z": round(z, 2), "base_fails": fails(base), "arm_fails": fails(arm)}


def main():
    base = cells(RESULTS / "lead_eval_1000_combo/results.json", "combo_a4_s4")
    obs = observed_pass()
    for rule in ("A", "B"):
        arm = cells(RESULTS / f"lead_eval_tail_{rule}/results.json")
        counts = json.loads((RESULTS / f"lead_eval_tail_{rule}/rule_counts.json").read_text())
        print(f"== rule {rule}: {counts}")
        for d in (2.0, 4.0):
            print(f"  d{d:g} all      ", compare(base[d], arm[d]))
            print(f"  d{d:g} obs-pass ", compare(base[d], arm[d], obs))


main()
