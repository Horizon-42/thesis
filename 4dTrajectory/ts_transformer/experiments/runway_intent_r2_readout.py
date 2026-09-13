"""Runway-intent R2a readout: the runway head's choice against the known runway and the causal rules, end to end, and the gates of plan §16.

Every comparison is on each airport's COMMON flights — the ones where the known runway, both heads'
picks and every rule's pick all have a forecast — so two rules are never read on different flights.
Written to the campaign folder as `readout.md` / `readout.json`.

    python run_ts.py runway_intent_r2_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2_20260913]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.data.approach_difficulty import STRAIGHT_TORTUOSITY
from ts_transformer.data.runway_context import RULES
from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r2_20260913"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
HEAD = "r11_lift"
COMPARED = ("assigned", HEAD, "r1", *RULES)
#: §16.1 gates: strict at the airports whose parallel side is the problem, no regression elsewhere.
STRICT = ("KRDU", "KSMF", "KSTL")
TOLERANT = ("KSJC", "KMSY")
TOLERANCE_M = 25.0
SHORT = {"assigned": "known", HEAD: "r11_lift", "r1": "R1", "B0_majority": "B0", "B1_active_config": "B1",
         "B2_active_config_gated": "B2", "B3_same_sector_last": "B3", "B4_wind": "B4"}


def common(flights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in flights if all(f["picks"][s] in f["hypotheses"] for s in COMPARED)]


def stats(flights: list[dict[str, Any]]) -> dict[str, Any]:
    """Per selector on these flights: runway accuracy and the paired error against the known runway."""
    base_fde = np.array([f["hypotheses"][f["assigned"]]["fde_m"] for f in flights])
    base_ade = np.array([f["hypotheses"][f["assigned"]]["ade_m"] for f in flights])
    out: dict[str, Any] = {"n": len(flights), "known_fde_mean": float(base_fde.mean()) if flights else None,
                           "known_ade_mean": float(base_ade.mean()) if flights else None, "selectors": {}}
    for s in COMPARED:
        fde = np.array([f["hypotheses"][f["picks"][s]]["fde_m"] for f in flights])
        ade = np.array([f["hypotheses"][f["picks"][s]]["ade_m"] for f in flights])
        out["selectors"][s] = {
            "runway_accuracy": float(np.mean([f["picks"][s] == f["assigned"] for f in flights])),
            "fde_mean": float(fde.mean()), "fde_delta_mean": float((fde - base_fde).mean()),
            "fde_delta_median": float(np.median(fde - base_fde)), "ade_delta_mean": float((ade - base_ade).mean()),
        }
    for head in (HEAD, "r1"):
        fde = np.array([f["expected"][head]["fde_m"] for f in flights])
        ade = np.array([f["expected"][head]["ade_m"] for f in flights])
        out[f"expected_{head}"] = {"fde_delta_mean": float((fde - base_fde).mean()),
                                   "ade_delta_mean": float((ade - base_ade).mean())}
    wrong = [f for f in flights if f["picks"][HEAD] != f["assigned"]]
    out["head_wrong"] = {
        "n": len(wrong),
        "fde_delta_mean": float(np.mean([f["hypotheses"][f["picks"][HEAD]]["fde_m"] - f["hypotheses"][f["assigned"]]["fde_m"]
                                         for f in wrong])) if wrong else None,
        "same_direction": sum(_same_direction(f) for f in wrong),
    }
    return out


def _same_direction(f: dict[str, Any]) -> bool:
    a, b = f["hypotheses"][f["assigned"]]["course_deg"], f["hypotheses"][f["picks"][HEAD]]["course_deg"]
    return abs((a - b + 180.0) % 360.0 - 180.0) <= 30.0


def strata(flights: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    straight = [f for f in flights if f["difficulty"]["route_tortuosity"] < STRAIGHT_TORTUOSITY]
    vectored = [f for f in flights if f["difficulty"]["route_tortuosity"] >= STRAIGHT_TORTUOSITY
                and not f["difficulty"]["established_at_anchor"]]
    return {"all": flights, "straight-in": straight, "vectored": vectored}


def table(cells: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| airport | n | known FDE mean (m) | " + " | ".join(f"{SHORT[s]}: acc / ΔFDE" for s in COMPARED[1:])
             + f" | E[Δ] {HEAD} |", "|---|---:|---:|" + "---|" * (len(COMPARED) - 1) + "---:|"]
    for name, c in cells.items():
        if not c["n"]:
            lines.append(f"| {name} | 0 | | " + " | " * (len(COMPARED) - 1) + " |")
            continue
        parts = [f"{100 * c['selectors'][s]['runway_accuracy']:.1f} % / {c['selectors'][s]['fde_delta_mean']:+.0f}"
                 for s in COMPARED[1:]]
        lines.append(f"| {name} | {c['n']} | {c['known_fde_mean']:.0f} | " + " | ".join(parts)
                     + f" | {c[f'expected_{HEAD}']['fde_delta_mean']:+.0f} |")
    return lines


def gates(cells: dict[str, dict[str, Any]], pooled: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for airport in STRICT + TOLERANT:
        c = cells.get(airport)
        if not c or not c["n"]:
            out[airport] = None
            continue
        head = c["selectors"][HEAD]["fde_delta_mean"]
        best_rule = min(RULES, key=lambda r: c["selectors"][r]["fde_delta_mean"])
        rule = c["selectors"][best_rule]["fde_delta_mean"]
        margin = 0.0 if airport in STRICT else TOLERANCE_M
        out[airport] = {"head": head, "best_rule": best_rule, "rule": rule,
                        "pass": head < rule if airport in STRICT else head <= rule + margin}
    out["pooled"] = {"head": pooled["selectors"][HEAD]["fde_delta_mean"],
                     "b1": pooled["selectors"]["B1_active_config"]["fde_delta_mean"],
                     "pass": pooled["selectors"][HEAD]["fde_delta_mean"] < pooled["selectors"]["B1_active_config"]["fde_delta_mean"]}
    out["all_pass"] = all(v is not None and v["pass"] for v in out.values() if isinstance(v, dict)) and \
        all(a in cells for a in AIRPORTS)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args(argv)
    out: Path = args.campaign_dir
    docs = {a: json.loads((out / a / "runway_intent_r2.json").read_text(encoding="utf-8"))
            for a in AIRPORTS if (out / a / "runway_intent_r2.json").is_file()}
    flights = {a: common(d["flights"]) for a, d in docs.items()}
    lines = ["Common flights (every compared pick has a forecast): "
             + ", ".join(f"{a} {len(f)} of {len(docs[a]['flights'])}" for a, f in flights.items()), ""]

    combined: dict[str, Any] = {"strata": {}}
    for stratum in ("all", "straight-in", "vectored"):
        cells = {a: stats(strata(f)[stratum]) for a, f in flights.items()}
        pooled = stats([x for f in flights.values() for x in strata(f)[stratum]])
        combined["strata"][stratum] = {"airports": cells, "pooled": pooled}
        lines += [f"### {stratum} — runway accuracy and the paired FDE change against the known runway (m, mean)", ""]
        lines += table({**cells, "pooled": pooled}) + [""]

    all_cells = combined["strata"]["all"]["airports"]
    lines += ["### Where the head is wrong (all flights)", "",
              "| airport | wrong picks | of which the same direction (a parallel) | mean FDE change on them (m) | median ΔFDE over all (m) | ΔADE mean (m) |",
              "|---|---:|---:|---:|---:|---:|"]
    for a, c in all_cells.items():
        w = c["head_wrong"]
        lines.append(f"| {a} | {w['n']} | {w['same_direction']} | "
                     f"{'—' if w['fde_delta_mean'] is None else format(w['fde_delta_mean'], '+.0f')} | "
                     f"{c['selectors'][HEAD]['fde_delta_median']:+.0f} | {c['selectors'][HEAD]['ade_delta_mean']:+.0f} |")

    lines += ["", "### By partition (all flights): head accuracy / ΔFDE, B1 accuracy / ΔFDE", "",
              "| airport | day_a | day_b |", "|---|---|---|"]
    for a, f in flights.items():
        cells = []
        for part in ("day_a", "day_b"):
            c = stats([x for x in f if x["partition"] == part])
            cells.append("—" if not c["n"] else
                         f"n {c['n']}: {100 * c['selectors'][HEAD]['runway_accuracy']:.1f} % / {c['selectors'][HEAD]['fde_delta_mean']:+.0f}; "
                         f"B1 {100 * c['selectors']['B1_active_config']['runway_accuracy']:.1f} % / {c['selectors']['B1_active_config']['fde_delta_mean']:+.0f}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")

    verdict = gates(all_cells, combined["strata"]["all"]["pooled"])
    combined["gates"] = verdict
    lines += ["", "### §16.1 gates (all flights, paired FDE change against the known runway)", "",
              "| gate | head ΔFDE (m) | reference ΔFDE (m) | verdict |", "|---|---:|---:|---|"]
    for a in STRICT + TOLERANT:
        g = verdict[a]
        if g is None:
            lines.append(f"| {a} | missing | | FAIL |")
            continue
        rule = f"{g['rule']:+.0f} ({SHORT[g['best_rule']]}{'' if a in STRICT else ' + 25 m'})"
        lines.append(f"| {a}: head {'<' if a in STRICT else '≤'} best rule | {g['head']:+.0f} | {rule} | "
                     f"{'pass' if g['pass'] else 'FAIL'} |")
    p = verdict["pooled"]
    lines.append(f"| pooled: head < B1 | {p['head']:+.0f} | {p['b1']:+.0f} | {'pass' if p['pass'] else 'FAIL'} |")
    lines.append(f"\nEvery gate: {'PASS' if verdict['all_pass'] else 'not all pass'}")

    text = "\n".join(lines)
    (out / "readout.md").write_text(text + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
