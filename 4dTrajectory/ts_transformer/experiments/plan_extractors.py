"""Plan-and-guidance step 1: the eight plan parameters over a checkpoint's cohort — their distributions, the airport-median baseline, and how many joins are published transitions.

The design (`docs/2026-09-09_plan_and_guidance_design.md` §3, §9 step 1) predicts five
operating parameters and assigns or samples three route parameters. Before anything is
trained, this readout says what the observed tracks actually do: per stratum, the p10 /
p50 / p90 of every parameter, how well the stratum's MEDIAN predicts each one (the §7 veto —
a plan head that does no better than this baseline is not learning), how often a value
sits outside the range the procedure and the aircraft allow, and how many observed joins
lie on a published transition rather than a radar vector (§8 risk 2).

The cohort is a checkpoint's own split, rebuilt the way the anytime curve rebuilds it
(`experiments.anytime_curve.load_arm` / `cohort_series`), so the numbers sit on the same
flights the design's gate numbers were measured on; the anchor is the checkpoint's own
fixed anchor (L−1). No model is run.

    python run_ts.py plan_extractors --checkpoint native32=<run>/checkpoint.pt --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ts_transformer.config import default_anchor
from ts_transformer.data.approach_difficulty import (
    STRATA_COVARIATES,
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_SHORT,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    strata_masks,
)
from ts_transformer.experiments.anytime_curve import Grid, cohort_series, load_arm, parse_arms
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.plan.extractors import PLAN_PARAMETERS, extract_plan
from ts_transformer.outputs.plan.skeleton import runway_skeleton

INSTRUMENT = "the plan extractors"
SCHEMA = "ts-plan-extractors-v1"
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ESTABLISHED)


def _p(values: np.ndarray, quantile: float) -> float:
    return float(np.percentile(values, quantile))


def parameter_summary(values: list[float | int | None], ranges: list[dict]) -> dict:
    """One parameter over one stratum: quantiles, the median baseline, the range shares."""
    defined = np.array([v for v in values if v is not None], dtype=np.float64)
    if defined.size == 0:
        return {"n": 0, "undefined": len(values)}
    median = float(np.median(defined))
    below = above = 0
    bounded = 0
    for value, bounds in zip(values, ranges, strict=True):
        if value is None:
            continue
        low, high = bounds
        if low is not None or high is not None:
            bounded += 1
        if low is not None and value < low:
            below += 1
        if high is not None and value > high:
            above += 1
    return {
        "n": int(defined.size),
        "undefined": int(len(values) - defined.size),
        "p10": _p(defined, 10), "p50": median, "p90": _p(defined, 90),
        "mean": float(np.mean(defined)),
        # the veto's reference: predicting the stratum's own median
        "median_baseline_mae": float(np.mean(np.abs(defined - median))),
        "bounded": bounded, "below_range": below, "above_range": above,
    }


def summarize(rows: list[dict]) -> dict:
    """Per stratum, per parameter; plus the route statistics the design's §3b/§8 ask for."""
    covariates = {row["dataset_id"]: row["difficulty"] for row in rows}
    masks = strata_masks(covariates, [row["dataset_id"] for row in rows])
    out: dict = {}
    for stratum in STRATA:
        mask = masks[stratum]
        members = [row for row, keep in zip(rows, mask, strict=True) if keep]
        block: dict = {"flights": len(members), "parameters": {}}
        for name in PLAN_PARAMETERS:
            block["parameters"][name] = parameter_summary(
                [row["labels"][name] for row in members],
                [tuple(row["labels"]["ranges"][name]) for row in members],
            )
        joins = [row["labels"] for row in members if row["labels"]["d_join_m"] is not None]
        classified = [lab for lab in joins if lab["join_on_transition"] is not None]
        block["route"] = {
            "no_join": len(members) - len(joins),
            "join_at_anchor": sum(1 for row in members if row["labels"]["join_at_anchor"]),
            "v_mid_from_anchor": sum(1 for row in members if row["labels"]["v_mid_from_anchor"]),
            "side_left": sum(1 for lab in joins if lab["side"] == -1),
            "side_straight": sum(1 for lab in joins if lab["side"] == 0),
            "side_right": sum(1 for lab in joins if lab["side"] == 1),
            "join_classified": len(classified),
            "join_on_transition": sum(1 for lab in classified if lab["join_on_transition"]),
        }
        out[stratum] = block
    return out


def _cell(summary: dict, digits: int) -> str:
    if summary.get("n", 0) == 0:
        return "n/a"
    return f"{summary['p50']:.{digits}f} [{summary['p10']:.{digits}f}, {summary['p90']:.{digits}f}]"


def format_tables(summary: dict) -> str:
    """The readout as text: the values, the median baseline, the route statistics."""
    digits = {"T_s": 0, "V_mid_mps": 1, "d_decel_m": 0, "V_final_mps": 1,
              "h_capture_m": 0, "d_join_m": 0, "side": 0, "L_pre_m": 0}
    strata = [s for s in STRATA if summary[s]["flights"]]
    lines = ["parameter p50 [p10, p90] per stratum (n = " + ", ".join(
        f"{STRATUM_SHORT[s]} {summary[s]['flights']}" for s in strata) + ")"]
    head = f"{'parameter':<14}" + "".join(f"{STRATUM_SHORT[s]:>30}" for s in strata)
    lines.append(head)
    for name in PLAN_PARAMETERS:
        lines.append(f"{name:<14}" + "".join(
            f"{_cell(summary[s]['parameters'][name], digits[name]):>30}" for s in strata))
    lines.append("")
    lines.append("median-baseline MAE (the §7 veto's reference) / undefined / outside the coded range")
    lines.append(head)
    for name in PLAN_PARAMETERS:
        cells = []
        for s in strata:
            p = summary[s]["parameters"][name]
            if p.get("n", 0) == 0:
                cells.append(f"{'n/a':>30}")
                continue
            outside = p["below_range"] + p["above_range"]
            cells.append(f"{p['median_baseline_mae']:.{max(digits[name], 1)}f} / {p['undefined']} / {outside}".rjust(30))
        lines.append(f"{name:<14}" + "".join(cells))
    lines.append("")
    lines.append("route: joins on a published transition, sides, joins before the anchor (censored), no join, V_mid from the anchor")
    for s in strata:
        r = summary[s]["route"]
        share = (r["join_on_transition"] / r["join_classified"]) if r["join_classified"] else float("nan")
        lines.append(
            f"  {STRATUM_SHORT[s]:<12} on-transition {r['join_on_transition']}/{r['join_classified']} "
            f"({100 * share:.1f} %)   side L/0/R {r['side_left']}/{r['side_straight']}/{r['side_right']}   "
            f"at-anchor {r['join_at_anchor']}   no join {r['no_join']}   V_mid@anchor {r['v_mid_from_anchor']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                        help="the checkpoint whose split is the cohort (one)")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the split (a smoke test)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    if len(arms) != 1:
        parser.error("exactly one --checkpoint names the cohort")
    (label, path), = arms.items()
    grid = Grid(split=args.split, bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    arm = load_arm(label, path, grid, torch.device("cpu"), instrument=INSTRUMENT)
    series = cohort_series(arm, grid)
    anchor = default_anchor(arm.config)
    skeletons: dict[tuple[str, str], object] = {}
    rows: list[dict] = []
    for item in series:
        key = (item.airport, str(item.scenario.source.get("runway")))
        if key not in skeletons:
            skeletons[key] = runway_skeleton(item)
        labels = extract_plan(item, anchor, skeletons[key])
        rows.append({
            "dataset_id": item.dataset_id,
            "flight_id": item.flight_id,
            "runway": key[1],
            "anchor": anchor,
            "difficulty": approach_difficulty(item, anchor).to_dict(),
            "labels": labels.to_dict(),
        })
    summary = summarize(rows)
    text = format_tables(summary)
    print(text, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out / "plan_extractors.json", {
        "schema_version": SCHEMA,
        "generated_at": utc_now(),
        "instrument": INSTRUMENT,
        "checkpoint": {"label": label, "path": str(path)},
        "split": args.split, "limit": args.limit, "flights": len(rows),
        "anchor": anchor, "anchor_rule": "the checkpoint's fixed anchor (L-1)",
        "strata_covariates": list(STRATA_COVARIATES),
        "parameters": list(PLAN_PARAMETERS),
        "skeletons": {f"{a} {r}": getattr(s, "procedure_uid") for (a, r), s in skeletons.items()},
        "summary": summary,
        "rows": rows,
    })
    (args.out / "plan_extractors.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'plan_extractors.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
