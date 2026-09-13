"""Runway-intent R2c readout: the runway belief along the approach, its flips, and the lock rules — alone and end to end — at the five airports.

Reads `runway_intent_r2c.json` (``<ICAO>/``) and `runway_intent_r2c_e2e.json` (``<ICAO>_e2e/``) from the
campaign folder and writes `readout.md` / `readout.json` there.

    python run_ts.py runway_intent_r2c_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2c_20260913]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r2c_20260913"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
HEAD = "r11_lift"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args(argv)
    out: Path = args.campaign_dir
    belief = {a: json.loads((out / a / "runway_intent_r2c.json").read_text(encoding="utf-8"))
              for a in AIRPORTS if (out / a / "runway_intent_r2c.json").is_file()}
    e2e = {a: json.loads((out / f"{a}_e2e" / "runway_intent_r2c_e2e.json").read_text(encoding="utf-8"))
           for a in AIRPORTS if (out / f"{a}_e2e" / "runway_intent_r2c_e2e.json").is_file()}

    lines = ["### The belief along the approach — exact accuracy (%) by remaining flown path: r11_lift / B1", ""]
    bins = list(dict.fromkeys(b for doc in belief.values() for b in doc["anytime"]))
    lines += ["| airport | flights | " + " | ".join(bins) + " |", "|---|---:|" + "---|" * len(bins)]
    for a, doc in belief.items():
        cells = [f"{pct(doc['anytime'][b][f'{HEAD}_exact'])} / {pct(doc['anytime'][b]['B1_exact'])}" if b in doc["anytime"] else "—"
                 for b in bins]
        lines.append(f"| {a} | {doc['flights']} | " + " | ".join(cells) + " |")

    lines += ["", "### Top-pick flips (per flight, over the whole approach)", "",
              "| airport | head | flights with a flip | flips / flight | toward / away from the truth | first ask right → last ask right |",
              "|---|---|---:|---:|---|---|"]
    for a, doc in belief.items():
        for h, f in doc["flips"].items():
            lines.append(f"| {a} | {h} | {f['flights_with_a_flip']} / {f['flights']} | {f['flips_per_flight']:.2f} | "
                         f"{f['flips_toward_the_truth']} / {f['flips_away_from_it']} | "
                         f"{pct(f['first_ask_right'])} → {pct(f['last_ask_right'])} |")

    rules = list(dict.fromkeys(r for doc in belief.values() for r in doc["locks"]))
    lines += ["", "### Lock rules on the runway alone — share of asks whose held runway is right (%), flips / flight", "",
              "| airport | " + " | ".join(rules) + " |", "|---|" + "---|" * len(rules)]
    for a, doc in belief.items():
        lines.append(f"| {a} | " + " | ".join(
            f"{pct(doc['locks'][r]['accuracy_over_asks'])} ({doc['locks'][r]['flips_per_flight']:.2f})" for r in rules) + " |")

    lines += ["", "### Lock rules end to end — on the asks where two rules differ: the expert's FDE change against the known runway (m, mean) and runway accuracy (%)", "",
              "| airport | flights / asks where rules differ | " + " | ".join(rules) + " |", "|---|---|" + "---|" * len(rules)]
    for a, doc in e2e.items():
        s = doc["summary"]
        cells = [f"{s['fde_delta_vs_known_mean'][r]:+.0f} ({pct(s['runway_accuracy_on_those_asks'][r])})"
                 if s["fde_delta_vs_known_mean"].get(r) is not None else "—" for r in rules]
        lines.append(f"| {a} | {s['flights_where_rules_differ']} / {s['asks_where_rules_differ']} of {s['asks']} | " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    (out / "readout.md").write_text(text + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps({
        "belief": {a: {k: d[k] for k in ("flights", "asks", "anytime", "flips", "locks")} for a, d in belief.items()},
        "end_to_end": {a: d["summary"] for a, d in e2e.items()},
    }, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
