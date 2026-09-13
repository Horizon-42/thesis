"""Runway-intent R1.1 / R1.1b readout: the candidate-symmetric heads against R1's on the same samples, and the gates of plan §14.

Reads whichever heads a campaign's documents hold — R1.1's ``r11`` (schema v1) and R1.1b's
``r11_noid`` / ``r11_lift`` (v2) — and applies §14's gates to each; the §11.4 gates by
`runway_intent_r1_readout.gates` itself (one definition). R1.1b's pick (§15.3): ``r11_lift`` if
it passes every gate, else ``r11_noid`` if it does, else neither. Written to the campaign folder
as `readout.md` / `readout.json`.

    python run_ts.py runway_intent_r11_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r11b_20260913]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.runway_intent_r1_readout import best, gate_table, gates, pct
from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r11b_20260913"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
PARTITIONS = ("day_a", "day_b")
HEAD_ORDER = ("r1", "r11", "r11_noid", "r11_lift")
#: §15.3: the head R1.1b carries forward, in order of preference, if it passes every gate.
PICK_ORDER = ("r11_lift", "r11_noid")
#: §14 gate (i): KSJC's two 30L-closure days (both in day_a's validation fold) and the margin to B1.
CLOSURE_DAYS = {"KSJC": ("2026-07-20", "2026-07-21")}
CLOSURE_MARGIN = 0.02
#: §14 gate (ii): the share of R1's side gain a symmetric head must keep, at these airports.
KEEP_SHARE = 0.9
KEEP_AIRPORTS = ("KRDU", "KSMF")
#: §14 gate (iii) G5: the entry anchor's exact accuracy against the best rule, every airport.
ENTRY_MARGIN = 0.01
DAY_GAP = 0.05


def heads_in(docs: dict[str, Any]) -> list[str]:
    present = set.intersection(*(set(d["models"]["day_a"]["validation"]) for d in docs.values()))
    return [head for head in HEAD_ORDER if head in present]


def symmetric(heads: list[str]) -> list[str]:
    return [head for head in heads if head != "r1"]


def importance_of(doc: dict[str, Any]) -> dict[str, Any]:
    """v2 keys the importance by head; v1 held R1.1's alone."""
    return doc["importance_day_a"] if doc["schema_version"].endswith("-v2") else {"r11": doc["importance_day_a"]}


def head_table(docs: dict[str, Any], heads: list[str], anchor: str) -> list[str]:
    lines = ["| airport | partition | head | exact | side given direction | minority | NLL | ECE |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for airport, doc in docs.items():
        for part in PARTITIONS:
            v = doc["models"][part]["validation"]
            for head in heads:
                m = v[head][anchor]["model"]
                lines.append(f"| {airport} | {part} | {head} | {pct(m['exact'])} | {pct(m['side_given_direction'])} | "
                             f"{pct(m['minority_accuracy'])} | {m['nll']:.3f} | {m['ece']:.3f} |")
            rules = v[heads[0]][anchor]["rules"]
            lines.append(f"| {airport} | {part} | best rule (B1-prob NLL) | {pct(best(rules, 'exact'))} | "
                         f"{pct(best(rules, 'side_given_direction'))} | {pct(best(rules, 'minority_accuracy'))} | "
                         f"{v[heads[0]][anchor]['b1_prob_nll']:.3f} | |")
    return lines


def day_label(day: str, cell: dict[str, Any]) -> str:
    return f"{day} ({cell['flights']} flights, {cell['top_runway']} {100 * cell['top_share']:.0f} %)"


def trailing_days_table(docs: dict[str, Any], heads: list[str]) -> list[str]:
    lines = ["| airport | partition | day | " + " | ".join(heads) + " | B1 |",
             "|---|---|---|" + "---:|" * (len(heads) + 1)]
    for airport, doc in docs.items():
        for part in PARTITIONS:
            for day, c in doc["validation_by_day"][part].items():
                e = c["exact"]
                if e["B1_active_config"] - min(e[h] for h in heads) >= DAY_GAP:
                    lines.append(f"| {airport} | {part} | {day_label(day, c)} | "
                                 + " | ".join(pct(e[h]) for h in heads) + f" | {pct(e['B1_active_config'])} |")
    return lines


def leakage_table(docs: dict[str, Any], heads: list[str]) -> list[str]:
    lines = ["| airport | paired flights | " + " | ".join(f"{h}: day_a / flight" for h in heads) + " | B1 |",
             "|---|---:|" + "---|" * len(heads) + "---:|"]
    for airport, doc in docs.items():
        leak = doc["leakage"]
        cells = [f"{pct(leak[h]['day_a']['all']['model']['exact'])} / {pct(leak[h]['flight']['all']['model']['exact'])}"
                 for h in heads]
        lines.append(f"| {airport} | {doc['split']['paired_leakage_flights']} | " + " | ".join(cells)
                     + f" | {pct(leak[heads[0]]['day_a']['all']['rules']['B1_active_config']['exact'])} |")
    return lines


def leakage_by_day_table(docs: dict[str, Any], heads: list[str]) -> list[str]:
    lines = ["| airport | head | day (paired flights) | day_a | flight | B1 |", "|---|---|---|---:|---:|---:|"]
    for airport, doc in docs.items():
        for head in heads:
            for day, c in doc["leakage_by_day"][head].items():
                e = c["exact"]
                if abs(e["flight"] - e["day_a"]) >= DAY_GAP:
                    lines.append(f"| {airport} | {head} | {day_label(day, c)} | {pct(e['day_a'])} | {pct(e['flight'])} | "
                                 f"{pct(e['B1_active_config'])} |")
    return lines


def importance_table(docs: dict[str, Any], heads: list[str]) -> list[str]:
    groups = sorted({g for doc in docs.values() for imp in importance_of(doc).values() for g in imp})
    lines = ["| airport | head | " + " | ".join(groups) + " |", "|---|---|" + "---|" * len(groups)]
    for airport, doc in docs.items():
        for head in symmetric(heads):
            imp = importance_of(doc)[head]
            cells = []
            for g in groups:
                c = imp.get(g)
                side = "n/a" if c is None or c["side_drop"] is None else f"{100 * c['side_drop']:+.1f}"
                cells.append("—" if c is None else f"{100 * c['exact_drop']:+.1f} / {side}")
            lines.append(f"| {airport} | {head} | " + " | ".join(cells) + " |")
    return lines


def head_gates(docs: dict[str, Any], head: str) -> dict[str, Any]:
    closure: dict[str, Any] = {}
    for airport, days in CLOSURE_DAYS.items():
        cells = docs[airport]["validation_by_day"]["day_a"] if airport in docs else {}
        for day in days:
            c = cells.get(day)
            closure[f"{airport}/{day}"] = None if c is None else {
                "head": c["exact"][head], "r1": c["exact"]["r1"], "b1": c["exact"]["B1_active_config"],
                "pass": c["exact"][head] >= c["exact"]["B1_active_config"] - CLOSURE_MARGIN,
            }
    keep: dict[str, Any] = {}
    for airport in KEEP_AIRPORTS:
        if airport not in docs:
            continue
        for part in PARTITIONS:
            v = docs[airport]["models"][part]["validation"]
            rule_side = best(v[head]["all"]["rules"], "side_given_direction")
            gain = 100 * (v[head]["all"]["model"]["side_given_direction"] - rule_side)
            gain_r1 = 100 * (v["r1"]["all"]["model"]["side_given_direction"] - rule_side)
            keep[f"{airport}/{part}"] = {"gain": gain, "gain_r1": gain_r1, "pass": gain >= KEEP_SHARE * gain_r1}
    entry: dict[str, Any] = {}
    for airport, doc in docs.items():
        for part in PARTITIONS:
            v = doc["models"][part]["validation"][head]["entry"]
            rule = best(v["rules"], "exact")
            entry[f"{airport}/{part}"] = {"head": v["model"]["exact"], "best_rule": rule,
                                          "pass": v["model"]["exact"] >= rule - ENTRY_MARGIN}
    as_r1 = {airport: {"models": {part: {"validation": doc["models"][part]["validation"][head]}
                                  for part in PARTITIONS}}
             for airport, doc in docs.items()}
    section = gates(as_r1, "full")
    flags = [c is not None and c["pass"] for c in closure.values()] + [c["pass"] for c in keep.values()]
    flags += [c["pass"] for c in entry.values()] + [not section["veto_side_irreducible"]]
    flags += [flag for cell in section["cells"].values() for key, flag in cell.items() if key.startswith("g")]
    return {"closure_days": closure, "keep_side_gain": keep, "g5_entry": entry, "section_11_4": section,
            "all_pass": all(flags)}


def gate_lines(head: str, verdict: dict[str, Any]) -> list[str]:
    def yn(flag: bool) -> str:
        return "pass" if flag else "FAIL"

    lines = [f"### Gates for `{head}` — every gate: {'PASS' if verdict['all_pass'] else 'not all pass'}", "",
             "(i) KSJC's closure days: head ≥ B1 − 2 points, each day", "",
             "| day | head | r1 | B1 | verdict |", "|---|---:|---:|---:|---|"]
    for key, c in verdict["closure_days"].items():
        lines.append(f"| {key} | missing | | | FAIL |" if c is None else
                     f"| {key} | {pct(c['head'])} | {pct(c['r1'])} | {pct(c['b1'])} | {yn(c['pass'])} |")
    lines += ["", "(ii) KRDU / KSMF keep ≥ 90 % of R1's side gain (points over the best rule, pooled)", "",
              "| airport / partition | head gain | R1 gain | verdict |", "|---|---:|---:|---|"]
    for key, c in verdict["keep_side_gain"].items():
        lines.append(f"| {key} | {c['gain']:+.2f} | {c['gain_r1']:+.2f} | {yn(c['pass'])} |")
    lines += ["", "(iii) G5 — entry anchor: head exact ≥ best rule − 1 point", "",
              "| airport / partition | head | best rule | verdict |", "|---|---:|---:|---|"]
    for key, c in verdict["g5_entry"].items():
        lines.append(f"| {key} | {pct(c['head'])} | {pct(c['best_rule'])} | {yn(c['pass'])} |")
    lines += ["", "(iii) §11.4's gates (all anchors pooled)", ""] + gate_table(verdict["section_11_4"])
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args(argv)
    out: Path = args.campaign_dir
    docs = {a: json.loads((out / a / "runway_intent_r11.json").read_text(encoding="utf-8"))
            for a in AIRPORTS if (out / a / "runway_intent_r11.json").is_file()}
    heads = heads_in(docs)

    check = {a: d["r1_reference_check"].get("identical") for a, d in docs.items()}
    lines = [f"Heads: {', '.join(heads)}. R1's head retrained on the same samples matches R1's artifact: {check}", ""]
    lines += ["### All anchors pooled", ""] + head_table(docs, heads, "all")
    lines += ["", "### At the arrival-slice entry", ""] + head_table(docs, heads, "entry")
    lines += ["", "### Days a head trails B1 by ≥ 5 points of exact accuracy (each day model's validation days)", ""]
    lines += trailing_days_table(docs, heads)
    lines += ["", "### Leakage — paired flights, the day-blocked model against the per-flight one", ""]
    lines += leakage_table(docs, heads)
    lines += ["", "### Leakage by day — days where the two differ by ≥ 5 points", ""] + leakage_by_day_table(docs, heads)
    lines += ["", "### What each symmetric head decides on — day_a, accuracy lost when a feature group is permuted (exact / side, points)", ""]
    lines += importance_table(docs, heads)

    verdicts = {head: head_gates(docs, head) for head in symmetric(heads)}
    for head, verdict in verdicts.items():
        lines += [""] + gate_lines(head, verdict)
    picked = next((head for head in PICK_ORDER if head in verdicts and verdicts[head]["all_pass"]), None)
    if any(head in verdicts for head in PICK_ORDER):
        lines += ["", f"R1.1b's pick (§15.3: r11_lift if every gate passes, else r11_noid, else neither): {picked or 'neither'}"]

    combined = {"heads": heads, "gates": verdicts, "r1_reference_check": check, "r11b_pick": picked}
    text = "\n".join(lines)
    (out / "readout.md").write_text(text + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
