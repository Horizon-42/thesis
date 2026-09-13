"""Runway-intent R1 readout: the five airports' `runway_intent_r1.json` in one table set, and the §11.4 gates applied mechanically.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §11.4 (the gates) and §13 (the results). Written to
the campaign folder as `readout.md` / `readout.json`.

    python run_ts.py runway_intent_r1_readout \\
        [--campaign-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r1_20260913]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

DEFAULT_CAMPAIGN = REPO_ROOT / "4dTrajectory" / "outputs" / "POOLED" / "experiments" / "runway_intent_r1_20260913"
AIRPORTS = ["KRDU", "KSJC", "KSTL", "KSMF", "KMSY"]
GATED = {"KRDU": 10.0, "KSMF": 10.0, "KSTL": 0.0}      # §11.4 gate 1: side gain (points) required
MINORITY_GATED = ("KRDU", "KSMF")                       # §11.4 gate 2
PARTITIONS = ("day_a", "day_b")
ALONG = ["entry", "r20km", "r15km", "r10km", "r6km"]
DAY_GAP = 0.05        # a day is listed where two pick arrays differ by at least this much exact accuracy


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def best(rules: dict[str, Any], key: str) -> float | None:
    values = [cell[key] for cell in rules.values() if cell.get(key) is not None]
    return max(values) if values else None


def model_table(docs: dict[str, Any], anchor: str) -> list[str]:
    lines = [
        "| airport | model | val flights | exact (best rule) | direction (B1) | side given direction (best rule) | minority (best rule) | NLL (B1-prob) | ECE |",
        "|---|---|---:|---|---|---|---|---|---:|",
    ]
    for airport, doc in docs.items():
        for name in PARTITIONS:
            v = doc["models"][name]["validation"][anchor]
            m, r = v["model"], v["rules"]
            lines.append(
                f"| {airport} | {name} | {doc['models'][name]['val_flights']} | "
                f"{pct(m['exact'])} ({pct(best(r, 'exact'))}) | {pct(m['direction'])} ({pct(r['B1_active_config']['direction'])}) | "
                f"{pct(m['side_given_direction'])} ({pct(best(r, 'side_given_direction'))}) | "
                f"{pct(m['minority_accuracy'])} ({pct(best(r, 'minority_accuracy'))}) | "
                f"{m['nll']:.3f} ({v['b1_prob_nll']:.3f}) | {m['ece']:.3f} |"
            )
    return lines


def along_table(docs: dict[str, Any]) -> list[str]:
    lines = ["| airport | partition | " + " | ".join(ALONG) + " |", "|---|---|" + "---|" * len(ALONG)]
    for airport, doc in docs.items():
        if airport not in GATED:
            continue
        for part in PARTITIONS:
            v = doc["models"][part]["validation"]
            cells = []
            for anchor in ALONG:
                if anchor not in v:
                    cells.append("—")
                    continue
                m, r = v[anchor]["model"], v[anchor]["rules"]
                cells.append(f"{pct(m['side_given_direction'])} / {pct(best(r, 'side_given_direction'))}")
            lines.append(f"| {airport} | {part} | " + " | ".join(cells) + " |")
    return lines


def leakage_table(docs: dict[str, Any]) -> list[str]:
    lines = [
        "| airport | flights | day_a exact / side / NLL | flight exact / side / NLL | B1 exact on the same flights |",
        "|---|---:|---|---|---:|",
    ]
    for airport, doc in docs.items():
        a, f = doc["leakage"]["day_a"]["all"], doc["leakage"]["flight"]["all"]
        lines.append(
            f"| {airport} | {doc['split']['paired_leakage_flights']} | "
            f"{pct(a['model']['exact'])} / {pct(a['model']['side_given_direction'])} / {a['model']['nll']:.3f} | "
            f"{pct(f['model']['exact'])} / {pct(f['model']['side_given_direction'])} / {f['model']['nll']:.3f} | "
            f"{pct(a['rules']['B1_active_config']['exact'])} |"
        )
    return lines


def day_label(day: str, cell: dict[str, Any]) -> str:
    return f"{day} ({cell['flights']} flights, {cell['top_runway']} {100 * cell['top_share']:.0f} %)"


def per_day_table(docs: dict[str, Any]) -> list[str]:
    lines = [
        "| airport | partition | val days | model median (min) | B1 median (min) | days the model trails B1 by ≥ 5 points: model / B1 |",
        "|---|---|---:|---|---|---|",
    ]
    for airport, doc in docs.items():
        for part in PARTITIONS:
            cells = doc["validation_by_day"][part]
            model = [c["exact"]["model"] for c in cells.values()]
            b1 = [c["exact"]["B1_active_config"] for c in cells.values()]
            trailing = [
                f"{day_label(day, c)} {pct(c['exact']['model'])} / {pct(c['exact']['B1_active_config'])}"
                for day, c in cells.items() if c["exact"]["B1_active_config"] - c["exact"]["model"] >= DAY_GAP
            ]
            lines.append(
                f"| {airport} | {part} | {len(cells)} | {pct(float(np.median(model)))} ({pct(min(model))}) | "
                f"{pct(float(np.median(b1)))} ({pct(min(b1))}) | {'; '.join(trailing) or 'none'} |"
            )
    return lines


def leakage_by_day_table(docs: dict[str, Any]) -> list[str]:
    lines = [
        "| airport | day (paired flights, busiest runway) | day_a | flight | B1 |",
        "|---|---|---:|---:|---:|",
    ]
    for airport, doc in docs.items():
        rows = [
            f"| {airport} | {day_label(day, c)} | {pct(c['exact']['day_a'])} | {pct(c['exact']['flight'])} | "
            f"{pct(c['exact']['B1_active_config'])} |"
            for day, c in doc["leakage_by_day"].items()
            if abs(c["exact"]["flight"] - c["exact"]["day_a"]) >= DAY_GAP
        ]
        lines += rows or [f"| {airport} | none | | | |"]
    return lines


def nowx_table(docs: dict[str, Any]) -> list[str]:
    lines = [
        "| airport | partition | exact full → nowx | side given direction full → nowx | NLL full → nowx |",
        "|---|---|---|---|---|",
    ]
    for airport, doc in docs.items():
        for part in PARTITIONS:
            full = doc["models"][part]["validation"]["all"]["model"]
            nowx = doc["models"][f"{part}_nowx"]["validation"]["all"]["model"]
            lines.append(f"| {airport} | {part} | {pct(full['exact'])} → {pct(nowx['exact'])} | "
                         f"{pct(full['side_given_direction'])} → {pct(nowx['side_given_direction'])} | "
                         f"{full['nll']:.3f} → {nowx['nll']:.3f} |")
    return lines


def importance_table(docs: dict[str, Any]) -> list[str]:
    groups = sorted({g for doc in docs.values() for g in doc["importance_day_a"]})
    lines = ["| airport | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for airport, doc in docs.items():
        imp = doc["importance_day_a"]
        cells = []
        for g in groups:
            c = imp.get(g)
            side = "n/a" if c is None or c["side_drop"] is None else f"{100 * c['side_drop']:+.1f}"
            cells.append("—" if c is None else f"{100 * c['exact_drop']:+.1f} / {side}")
        lines.append(f"| {airport} | " + " | ".join(cells) + " |")
    return lines


def gates(docs: dict[str, Any], variant: str) -> dict[str, Any]:
    verdict: dict[str, Any] = {}
    for airport, need in GATED.items():
        doc = docs.get(airport)
        if not doc:
            continue
        for part in PARTITIONS:
            name = part if variant == "full" else f"{part}_nowx"
            v = doc["models"][name]["validation"]["all"]
            m, r = v["model"], v["rules"]
            gain = 100 * (m["side_given_direction"] - best(r, "side_given_direction"))
            cell = {
                "side_gain_points": gain,
                "g1_side": gain >= need,
                "g3_nll": m["nll"] < v["b1_prob_nll"],
                "g3_ece": m["ece"] <= 0.05,
                "g4_direction": m["direction"] >= r["B1_active_config"]["direction"],
            }
            if airport in MINORITY_GATED:
                cell["g2_minority"] = m["minority_accuracy"] >= 0.60
            verdict[f"{airport}/{part}"] = cell
    veto = all(verdict[f"{a}/{p}"]["side_gain_points"] < 5.0
               for a in MINORITY_GATED for p in PARTITIONS if f"{a}/{p}" in verdict)
    return {"cells": verdict, "veto_side_irreducible": veto}


def gate_table(result: dict[str, Any]) -> list[str]:
    def yn(flag: bool) -> str:
        return "pass" if flag else "FAIL"

    lines = [
        "| airport / partition | side gain (points) | G1 side | G2 minority ≥ 60 % | G3 NLL < B1-prob | G3 ECE ≤ 0.05 | G4 direction ≥ B1 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for key, c in result["cells"].items():
        lines.append(f"| {key} | {c['side_gain_points']:+.2f} | {yn(c['g1_side'])} | "
                     f"{yn(c['g2_minority']) if 'g2_minority' in c else '—'} | {yn(c['g3_nll'])} | "
                     f"{yn(c['g3_ece'])} | {yn(c['g4_direction'])} |")
    lines.append(f"\nVeto (side gain < 5 points at KRDU and KSMF on both partitions): "
                 f"{'FIRES' if result['veto_side_irreducible'] else 'does not fire'}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args(argv)
    out: Path = args.campaign_dir
    docs = {a: json.loads((out / a / "runway_intent_r1.json").read_text(encoding="utf-8"))
            for a in AIRPORTS if (out / a / "runway_intent_r1.json").is_file()}

    lines = ["### R1 — the full model on its validation days, all anchors pooled (the gates read this)", ""]
    lines += model_table(docs, "all")
    lines += ["", "### R1 — at the arrival-slice entry (the hardest anchor)", ""]
    lines += model_table(docs, "entry")
    lines += ["", "### Along the approach — side given direction, model / best rule (%), at the entry and at each ring (first crossing)", ""]
    lines += along_table(docs)
    lines += ["", "### Per operating day — exact accuracy (%), all anchors, on each day model's validation days", ""]
    lines += per_day_table(docs)
    lines += ["", "### Leakage — the same flights (validation under BOTH splits), day-blocked model vs per-flight model", ""]
    lines += leakage_table(docs)
    lines += ["", "### Leakage by day — the days where the two models differ by ≥ 5 points of exact accuracy on the paired flights", ""]
    lines += leakage_by_day_table(docs)
    lines += ["", "### Secondary — without the day-level features (wind, time of day), all anchors pooled", ""]
    lines += nowx_table(docs)
    lines += ["", "### What the head decides on — day_a, accuracy lost when a feature group is permuted (exact / side, points)", ""]
    lines += importance_table(docs)

    combined: dict[str, Any] = {}
    for variant in ("full", "nowx"):
        result = gates(docs, variant)
        combined[f"gates_{variant}"] = result
        label = "the full model — the pre-registered reading" if variant == "full" else "secondary: without wind / time of day"
        lines += ["", f"### §11.4 gates — {label}", ""]
        lines += gate_table(result)

    combined["airports"] = {
        a: {k: d[k] for k in ("schema_version", "split", "minority_runways", "direction_groups",
                              "validation_by_day", "leakage_by_day")}
        for a, d in docs.items()
    }
    text = "\n".join(lines)
    (out / "readout.md").write_text(text + "\n", encoding="utf-8")
    (out / "readout.json").write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
