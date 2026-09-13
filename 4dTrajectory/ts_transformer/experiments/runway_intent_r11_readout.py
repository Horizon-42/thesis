"""Runway-intent R1.1 readout: the candidate-symmetric head against R1's on the same samples, and the gates of plan §14.

Written to the campaign folder as `readout.md` / `readout.json`. The §11.4 gates are applied by
`runway_intent_r1_readout.gates` itself (one definition), to the R1.1 head.

    python run_ts.py runway_intent_r11_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r11_20260913]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.runway_intent_r1_readout import best, gate_table, gates, pct
from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r11_20260913"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
PARTITIONS = ("day_a", "day_b")
#: §14 gate (i): KSJC's two 30L-closure days (both in day_a's validation fold) and the margin to B1.
CLOSURE_DAYS = {"KSJC": ("2026-07-20", "2026-07-21")}
CLOSURE_MARGIN = 0.02
#: §14 gate (ii): the share of R1's side gain R1.1 must keep, at these airports.
KEEP_SHARE = 0.9
KEEP_AIRPORTS = ("KRDU", "KSMF")
#: §14 gate (iii) G5: the entry anchor's exact accuracy against the best rule, every airport.
ENTRY_MARGIN = 0.01
DAY_GAP = 0.05


def head_table(docs: dict[str, Any], anchor: str) -> list[str]:
    lines = [
        "| airport | partition | exact: r11 / r1 / best rule | side: r11 / r1 / best rule | minority: r11 / r1 / best rule | NLL: r11 / r1 / B1-prob | ECE: r11 / r1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for airport, doc in docs.items():
        for part in PARTITIONS:
            v = doc["models"][part]["validation"]
            a, b = v["r11"][anchor], v["r1"][anchor]
            ma, mb, rules = a["model"], b["model"], a["rules"]
            lines.append(
                f"| {airport} | {part} | {pct(ma['exact'])} / {pct(mb['exact'])} / {pct(best(rules, 'exact'))} | "
                f"{pct(ma['side_given_direction'])} / {pct(mb['side_given_direction'])} / {pct(best(rules, 'side_given_direction'))} | "
                f"{pct(ma['minority_accuracy'])} / {pct(mb['minority_accuracy'])} / {pct(best(rules, 'minority_accuracy'))} | "
                f"{ma['nll']:.3f} / {mb['nll']:.3f} / {a['b1_prob_nll']:.3f} | {ma['ece']:.3f} / {mb['ece']:.3f} |"
            )
    return lines


def day_label(day: str, cell: dict[str, Any]) -> str:
    return f"{day} ({cell['flights']} flights, {cell['top_runway']} {100 * cell['top_share']:.0f} %)"


def trailing_days_table(docs: dict[str, Any]) -> list[str]:
    lines = ["| airport | partition | day | r11 | r1 | B1 |", "|---|---|---|---:|---:|---:|"]
    for airport, doc in docs.items():
        for part in PARTITIONS:
            for day, c in doc["validation_by_day"][part].items():
                e = c["exact"]
                if e["B1_active_config"] - min(e["r11"], e["r1"]) >= DAY_GAP:
                    lines.append(f"| {airport} | {part} | {day_label(day, c)} | {pct(e['r11'])} | {pct(e['r1'])} | "
                                 f"{pct(e['B1_active_config'])} |")
    return lines


def leakage_table(docs: dict[str, Any]) -> list[str]:
    lines = [
        "| airport | paired flights | r11: day_a / flight | r1: day_a / flight | B1 |",
        "|---|---:|---|---|---:|",
    ]
    for airport, doc in docs.items():
        leak = doc["leakage"]
        cells = [f"{pct(leak[h]['day_a']['all']['model']['exact'])} / {pct(leak[h]['flight']['all']['model']['exact'])}"
                 for h in ("r11", "r1")]
        lines.append(f"| {airport} | {doc['split']['paired_leakage_flights']} | {cells[0]} | {cells[1]} | "
                     f"{pct(leak['r11']['day_a']['all']['rules']['B1_active_config']['exact'])} |")
    return lines


def leakage_by_day_table(docs: dict[str, Any]) -> list[str]:
    lines = ["| airport | head | day (paired flights) | day_a | flight | B1 |", "|---|---|---|---:|---:|---:|"]
    for airport, doc in docs.items():
        for head in ("r11", "r1"):
            for day, c in doc["leakage_by_day"][head].items():
                e = c["exact"]
                if abs(e["flight"] - e["day_a"]) >= DAY_GAP:
                    lines.append(f"| {airport} | {head} | {day_label(day, c)} | {pct(e['day_a'])} | {pct(e['flight'])} | "
                                 f"{pct(e['B1_active_config'])} |")
    return lines


def importance_table(docs: dict[str, Any]) -> list[str]:
    groups = sorted({g for doc in docs.values() for g in doc["importance_day_a"]})
    lines = ["| airport | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for airport, doc in docs.items():
        cells = []
        for g in groups:
            c = doc["importance_day_a"].get(g)
            side = "n/a" if c is None or c["side_drop"] is None else f"{100 * c['side_drop']:+.1f}"
            cells.append("—" if c is None else f"{100 * c['exact_drop']:+.1f} / {side}")
        lines.append(f"| {airport} | " + " | ".join(cells) + " |")
    return lines


def r11_gates(docs: dict[str, Any]) -> dict[str, Any]:
    closure: dict[str, Any] = {}
    for airport, days in CLOSURE_DAYS.items():
        cells = docs[airport]["validation_by_day"]["day_a"] if airport in docs else {}
        for day in days:
            c = cells.get(day)
            closure[f"{airport}/{day}"] = None if c is None else {
                "r11": c["exact"]["r11"], "r1": c["exact"]["r1"], "b1": c["exact"]["B1_active_config"],
                "pass": c["exact"]["r11"] >= c["exact"]["B1_active_config"] - CLOSURE_MARGIN,
            }
    keep: dict[str, Any] = {}
    for airport in KEEP_AIRPORTS:
        if airport not in docs:
            continue
        for part in PARTITIONS:
            v = docs[airport]["models"][part]["validation"]
            rule_side = best(v["r11"]["all"]["rules"], "side_given_direction")
            gain_r11 = 100 * (v["r11"]["all"]["model"]["side_given_direction"] - rule_side)
            gain_r1 = 100 * (v["r1"]["all"]["model"]["side_given_direction"] - rule_side)
            keep[f"{airport}/{part}"] = {"gain_r11": gain_r11, "gain_r1": gain_r1,
                                         "pass": gain_r11 >= KEEP_SHARE * gain_r1}
    entry: dict[str, Any] = {}
    for airport, doc in docs.items():
        for part in PARTITIONS:
            v = doc["models"][part]["validation"]["r11"]["entry"]
            rule = best(v["rules"], "exact")
            entry[f"{airport}/{part}"] = {"r11": v["model"]["exact"], "best_rule": rule,
                                          "pass": v["model"]["exact"] >= rule - ENTRY_MARGIN}
    # §11.4's gates, by R1's own function, on the R1.1 head
    as_r1 = {airport: {"models": {part: {"validation": doc["models"][part]["validation"]["r11"]}
                                  for part in PARTITIONS}}
             for airport, doc in docs.items()}
    return {"closure_days": closure, "keep_side_gain": keep, "g5_entry": entry, "section_11_4": gates(as_r1, "full")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args(argv)
    out: Path = args.campaign_dir
    docs = {a: json.loads((out / a / "runway_intent_r11.json").read_text(encoding="utf-8"))
            for a in AIRPORTS if (out / a / "runway_intent_r11.json").is_file()}

    check = {a: d["r1_reference_check"].get("identical") for a, d in docs.items()}
    lines = [f"R1's head retrained on the same samples matches R1's artifact: {check}", ""]
    lines += ["### R1.1 (r11) against R1 (r1), all anchors pooled", ""] + head_table(docs, "all")
    lines += ["", "### At the arrival-slice entry", ""] + head_table(docs, "entry")
    lines += ["", "### Days a head trails B1 by ≥ 5 points of exact accuracy (each day model's validation days)", ""]
    lines += trailing_days_table(docs)
    lines += ["", "### Leakage — paired flights, the day-blocked model against the per-flight one", ""] + leakage_table(docs)
    lines += ["", "### Leakage by day — days where the two differ by ≥ 5 points", ""] + leakage_by_day_table(docs)
    lines += ["", "### What R1.1 decides on — day_a, accuracy lost when a feature group is permuted (exact / side, points)", ""]
    lines += importance_table(docs)

    verdict = r11_gates(docs)
    yn = lambda flag: "pass" if flag else "FAIL"  # noqa: E731
    lines += ["", "### §14 gate (i) — KSJC's closure days: R1.1 ≥ B1 − 2 points, each day", "",
              "| day | r11 | r1 | B1 | verdict |", "|---|---:|---:|---:|---|"]
    for key, c in verdict["closure_days"].items():
        lines.append(f"| {key} | missing | | | FAIL |" if c is None else
                     f"| {key} | {pct(c['r11'])} | {pct(c['r1'])} | {pct(c['b1'])} | {yn(c['pass'])} |")
    lines += ["", "### §14 gate (ii) — KRDU / KSMF keep ≥ 90 % of R1's side gain (points over the best rule, pooled)", "",
              "| airport / partition | R1.1 gain | R1 gain | verdict |", "|---|---:|---:|---|"]
    for key, c in verdict["keep_side_gain"].items():
        lines.append(f"| {key} | {c['gain_r11']:+.2f} | {c['gain_r1']:+.2f} | {yn(c['pass'])} |")
    lines += ["", "### §14 gate (iii) G5 — entry anchor: R1.1 exact ≥ best rule − 1 point, every airport", "",
              "| airport / partition | R1.1 | best rule | verdict |", "|---|---:|---:|---|"]
    for key, c in verdict["g5_entry"].items():
        lines.append(f"| {key} | {pct(c['r11'])} | {pct(c['best_rule'])} | {yn(c['pass'])} |")
    lines += ["", "### §14 gate (iii) — §11.4's gates on R1.1 (all anchors pooled)", ""]
    lines += gate_table(verdict["section_11_4"])

    combined = {"gates": verdict, "r1_reference_check": check}
    text = "\n".join(lines)
    (out / "readout.md").write_text(text + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
