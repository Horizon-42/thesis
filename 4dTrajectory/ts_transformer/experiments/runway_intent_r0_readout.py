"""Runway-intent R0 readout: the five airports' R0a (`runway_intent_r0.json`) and R0b (`hypotheses.json`) in one table set.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §11. R0b is re-summarised here with the PAIRED
`runway_hypotheses.summarise` (every selector's FDE change against the assigned runway on the same
flights), from the per-flight records each run stored, so every airport is read with one
definition whichever code wrote its `hypotheses.json`. Written to the campaign folder as
`readout.md` / `readout.json`.

    python run_ts.py runway_intent_r0_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r0_20260913]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ts_transformer.experiments.runway_hypotheses import summarise as summarise_hypotheses
from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r0_20260913"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
RULES = ["B0_majority", "B1_active_config", "B2_active_config_gated", "B3_same_sector_last", "B4_wind"]
SHORT = {"B0_majority": "B0", "B1_active_config": "B1", "B2_active_config_gated": "B2",
         "B3_same_sector_last": "B3", "B4_wind": "B4"}
SELECTORS = ["B0_majority", "B1_active_config", "B3_same_sector_last", "B4_wind", "oracle_same_direction"]
ALONG = ["entry", "30km", "20km", "15km", "10km", "6km", "3km"]


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def r0a_tables(out: Path, combined: dict[str, Any]) -> list[str]:
    lines = [
        "### R0a — exact accuracy (%) by rule, at the slice entry and at 10 km remaining", "",
        "| airport | n | anchor | B0 | B1 | B2 | B3 | B4 | direction (B1) | side given direction (B1 / B3, n) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for airport in AIRPORTS:
        path = out / airport / "runway_intent_r0.json"
        if not path.is_file():
            lines.append(f"| {airport} | — | missing | | | | | | | |")
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        combined[airport] = {"r0a": {k: doc[k] for k in ("validation_flights", "direction_groups", "summary",
                                                         "per_runway", "direction_flips", "context_pool")}}
        for anchor in ("entry", "10km"):
            cell = doc["summary"].get(anchor)
            if not cell:
                continue
            rules = cell["rules"]
            b1, b3 = rules["B1_active_config"], rules["B3_same_sector_last"]
            side = (f"{pct(b1['side_given_direction'])} / {pct(b3['side_given_direction'])} ({b1['side_flights']})"
                    if b1["side_given_direction"] is not None else "n/a (no parallel)")
            lines.append(f"| {airport} | {cell['flights']} | {anchor} | "
                         + " | ".join(pct(rules[r]["exact"]) for r in RULES)
                         + f" | {pct(b1['direction'])} | {side} |")

    lines += ["", "### R0a — B1 / B3 exact accuracy (%) along the approach (coverage = flights with that much path left)", ""]
    lines += ["| airport | " + " | ".join(ALONG) + " |", "|---|" + "---:|" * len(ALONG)]
    for airport in AIRPORTS:
        doc = combined.get(airport, {}).get("r0a")
        if not doc:
            continue
        cells = []
        for b in ALONG:
            cell = doc["summary"].get(b)
            if not cell:
                cells.append("—")
                continue
            r = cell["rules"]
            cells.append(f"{pct(r['B1_active_config']['exact'])} / {pct(r['B3_same_sector_last']['exact'])} "
                         f"({100 * cell['flights'] / doc['validation_flights']:.0f}%)")
        lines.append(f"| {airport} | " + " | ".join(cells) + " |")

    lines += ["", "### R0a — per landing runway, exact accuracy (%) at the slice entry (B1 / B3)", ""]
    for airport in AIRPORTS:
        doc = combined.get(airport, {}).get("r0a")
        if not doc:
            continue
        parts = [f"{runway} (n={cell['flights']}) {pct(cell['B1_active_config'])} / {pct(cell['B3_same_sector_last'])}"
                 for runway, cell in doc["per_runway"].get("entry", {}).items()]
        lines.append(f"- **{airport}**: " + "; ".join(parts))

    lines += ["", "### R0a — landing-direction flips (15-min bins, >=2 landings, new direction held 2 bins)", ""]
    lines += ["| airport | days | flips | days with a flip | changes across a >3 h gap | day-blocked val: days / flips | day-blocked test: days / flips |",
              "|---|---:|---:|---:|---:|---|---|"]
    for airport in AIRPORTS:
        doc = combined.get(airport, {}).get("r0a")
        if not doc:
            continue
        flips = doc["direction_flips"]
        folds, per_day = flips["day_blocked_folds"], flips["per_day"]
        lines.append(
            f"| {airport} | {flips['days']} | {sum(per_day.values())} | {sum(v > 0 for v in per_day.values())} | "
            f"{sum(flips.get('across_gap_per_day', {}).values())} | "
            f"{folds.get('val', {}).get('days', 0)} / {folds.get('val', {}).get('flips', 0)} | "
            f"{folds.get('test', {}).get('days', 0)} / {folds.get('test', {}).get('flips', 0)} |"
        )
    return lines


def r0b_tables(out: Path, combined: dict[str, Any]) -> list[str]:
    lines = [
        "", "### R0b — what a wrong runway costs the plan head (its L-1 anchor), PAIRED on the same flights", "",
        "Each cell: runway accuracy % / mean FDE change vs the assigned runway on the flights that rule was scored on (m), n.", "",
        "| airport | stratum | n | assigned FDE mean (m) | " + " | ".join(SHORT.get(s, "same-dir oracle") for s in SELECTORS) + " |",
        "|---|---|---:|---:|" + "---|" * len(SELECTORS),
    ]
    for airport in AIRPORTS:
        path = out / airport / "hypotheses.json"
        if not path.is_file():
            lines.append(f"| {airport} | missing | | |" + " |" * len(SELECTORS))
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        summary = summarise_hypotheses(doc["flights"], doc["selectors"])
        combined.setdefault(airport, {})["r0b"] = {
            "summary_paired": summary, "unflyable_candidates": doc.get("unflyable_candidates"),
            "candidates": doc["candidates"], "context_rules": doc.get("context_rules"),
            "flights_scored": len(doc["flights"]),
        }
        for stratum_key, block in summary.items():
            if stratum_key == "by_assigned_runway":
                continue
            s = block["selectors"]
            cells = [f"{pct(s[sel]['runway_accuracy'])} / {s[sel]['fde_delta_mean']:+.0f} ({s[sel]['n']})"
                     for sel in SELECTORS]
            lines.append(f"| {airport} | {stratum_key.split(' (')[0]} | {block['n']} | "
                         f"{s['assigned']['fde_mean']:.0f} | " + " | ".join(cells) + " |")
        if doc.get("unflyable_candidates"):
            lines.append(f"| {airport} | unflyable on the plan path: {', '.join(doc['unflyable_candidates'])} | | |"
                         + " |" * len(SELECTORS))

    lines += ["", "### R0b — per assigned runway: FDE median (m) under the assigned runway / under B1 (B1 runway accuracy %)", ""]
    for airport in AIRPORTS:
        r0b = combined.get(airport, {}).get("r0b")
        if not r0b:
            continue
        per = r0b["summary_paired"]["by_assigned_runway"]
        parts = [f"{runway} (n={cell['n']}) {cell['assigned']['fde_median']:.0f} / {cell['B1_active_config']['fde_median']:.0f} "
                 f"({pct(cell['B1_active_config']['runway_accuracy'])})" for runway, cell in per.items()]
        lines.append(f"- **{airport}**: " + "; ".join(parts))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args(argv)
    out: Path = args.campaign_dir
    combined: dict[str, Any] = {}
    lines = r0a_tables(out, combined) + r0b_tables(out, combined)
    text = "\n".join(lines)
    (out / "readout.md").write_text(text + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
