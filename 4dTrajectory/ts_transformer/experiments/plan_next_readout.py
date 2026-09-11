"""Plan-and-guidance step 3a (design v5): the lead a single instruction must predict.

At each anchor — the one row N seconds after the slice starts (a shorter window's anchor)
and D random anchors per flight, uniform in remaining path over the admissible rows — read
the plan's next instruction off the observed track: whether the flight is already
established (censored), whether the next thing is the join, and otherwise where the next
fly-by fix lies (ahead and across, in runway axes about the anchor; along the path; in
time) and how fast the aircraft is there. Per stratum (fixed at L−1, as every readout
does) and per anchor kind, with the trivial baseline a head must beat: the stratum's
median next fix and the majority class of "next is the join".

    python run_ts.py plan_next_readout --checkpoint native32=… [--draws 5] [--anchor-s 60] --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ts_transformer.config import PREDICTION_PLAN, default_anchor
from ts_transformer.data.anchor_strata import remaining_path_uniform_offset
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_SHORT,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    remaining_path_profile_m,
    strata_masks,
)
from ts_transformer.data.dataset import truth_duration_s
from ts_transformer.experiments.anytime_curve import Grid, cohort_series, load_arm, parse_arms
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.dynamics.context import ANCHOR_CONTROL_SAMPLES
from ts_transformer.inference.forecast import history_at_anchor
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.labels import TARGETS, targets_from_labels
from ts_transformer.outputs.plan.model import prediction_rows
from ts_transformer.outputs.plan.skeleton import runway_skeleton

INSTRUMENT = "the next-instruction readout"
SCHEMA = "ts-plan-next-readout-v2"   # v2 (2026-09-11): the head's single-step prediction beside the truth
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ESTABLISHED)
#: A random anchor needs this much truth after it (the oracle's slack).
MIN_FUTURE_S = 60.0
ANCHOR_FIXED = "fixed"
ANCHOR_RANDOM = "random"


def random_anchors(series, draws: int, seed: int, index: int) -> list[int]:
    """``draws`` anchors uniform in remaining path over the flight's admissible rows."""
    profile = remaining_path_profile_m(series)
    first = ANCHOR_CONTROL_SAMPLES - 1
    admissible = [row for row in range(first, series.n_samples) if truth_duration_s(series, row) >= MIN_FUTURE_S]
    if not admissible:
        return []
    remaining = profile[admissible]
    rng = np.random.default_rng([seed, index])
    return [int(admissible[remaining_path_uniform_offset(remaining, float(u))]) for u in rng.random(draws)]


def predicted_row(arm, series, anchor: int, skeleton) -> dict:
    """The plan head's single-step reading at ``anchor``: its target vector in physical
    units (`labels.TARGETS` order) and its next-is-join probability."""
    x = history_at_anchor(series, arm.config, arm.normalizer, anchor).astype(np.float32)
    with torch.no_grad():
        prediction = arm.model(torch.from_numpy(x[None]))
    values, probability = prediction_rows(prediction)
    return {"predicted": {name: float(v) for name, v in zip(TARGETS, values[0], strict=True)},
            "predicted_next_is_join_probability": float(probability[0])}


def next_instruction_row(series, anchor: int, skeleton, kind: str, arm=None) -> dict:
    labels = extract_plan(series, anchor, skeleton)
    row = {
        "anchor": int(anchor), "anchor_kind": kind,
        "remaining_at_anchor_m": labels.remaining_path_at_anchor_m,
        "speed_at_anchor_mps": labels.ground_speed_at_anchor_mps,
        "censored": bool(labels.join_at_anchor),
        "next_is_join": bool(not labels.join_at_anchor and not labels.waypoints),
        "fixes_ahead": 0 if not labels.waypoints else len(labels.waypoints),
        "fixes_dropped": int(labels.waypoints_dropped),
        "lead_to_join_m": None if labels.L_pre_m is None else float(labels.L_pre_m),
    }
    if labels.waypoints:
        fix = labels.waypoints[0]
        d_anchor, xt_anchor = skeleton.axes(series.values[anchor : anchor + 1, 0], series.values[anchor : anchor + 1, 1])
        d_fix, xt_fix = skeleton.axes(np.array([fix[0]]), np.array([fix[1]]))
        remaining_fix, speed_fix = labels.waypoint_speeds[0]
        row.update({
            "next_ahead_m": float(d_anchor[0] - d_fix[0]),
            "next_across_m": float(xt_fix[0] - xt_anchor[0]),
            "next_lead_path_m": float(labels.remaining_path_at_anchor_m - remaining_fix),
            "next_lead_time_s": float(labels.waypoint_times_s[0]),
            "next_speed_mps": float(speed_fix),
        })
    # the head reads the anchors it has a full lookback at; the truth reading covers every
    # admissible anchor (the head's cells are over the anchors it read)
    if arm is not None and arm.config.prediction_output == PREDICTION_PLAN and anchor >= arm.config.seq_len - 1:
        targets = targets_from_labels(labels, series, anchor, skeleton)
        row["targets"] = {name: float(v) for name, v in zip(TARGETS, targets.values, strict=True)}
        row["targets_valid"] = {name: bool(v) for name, v in zip(TARGETS, targets.valid, strict=True)}
        row.update(predicted_row(arm, series, anchor, skeleton))
    return row


def _p(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")


def summarize(rows: list[dict]) -> dict:
    covariates = {r["dataset_id"]: r["difficulty"] for r in rows}
    ids = [r["dataset_id"] for r in rows]
    masks = strata_masks(covariates, ids)
    out: dict = {}
    for stratum in STRATA:
        for kind in (ANCHOR_FIXED, ANCHOR_RANDOM):
            members = [r for r, keep in zip(rows, masks[stratum], strict=True) if keep and r["anchor_kind"] == kind]
            if not members:
                continue
            open_ = [r for r in members if not r["censored"]]
            with_fix = [r for r in open_ if not r["next_is_join"]]
            cell: dict = {
                "anchors": len(members),
                "remaining_at_anchor_p50_m": _p([r["remaining_at_anchor_m"] for r in members], 50),
                "censored_share": float(np.mean([r["censored"] for r in members])),
                "next_is_join_share_of_open": float(np.mean([r["next_is_join"] for r in open_])) if open_ else float("nan"),
                "fixes_ahead_mean_of_open": float(np.mean([r["fixes_ahead"] for r in open_])) if open_ else float("nan"),
                "lead_to_join_p50_m": _p([r["lead_to_join_m"] for r in open_ if r["lead_to_join_m"] is not None], 50),
            }
            if with_fix:
                ahead = np.array([r["next_ahead_m"] for r in with_fix])
                across = np.array([r["next_across_m"] for r in with_fix])
                med = np.array([np.median(ahead), np.median(across)])
                baseline_error = np.hypot(ahead - med[0], across - med[1])
                majority = max(
                    float(np.mean([r["next_is_join"] for r in open_])),
                    1.0 - float(np.mean([r["next_is_join"] for r in open_])),
                )
                predicted = [r for r in with_fix if "predicted" in r]
                if predicted:
                    fix_error = np.array([
                        np.hypot(r["predicted"]["next_ahead_m"] - r["next_ahead_m"],
                                 r["predicted"]["next_across_m"] - r["next_across_m"])
                        for r in predicted
                    ])
                    join_rows = [r for r in open_ if "predicted_next_is_join_probability" in r]
                    cell.update({
                        "head_fix_error_p50_m": _p(fix_error, 50), "head_fix_error_p90_m": _p(fix_error, 90),
                        "head_speed_error_p50_mps": _p([abs(r["predicted"]["next_speed_mps"] - r["next_speed_mps"]) for r in predicted], 50),
                        "head_next_remaining_error_p50_m": _p([abs(r["predicted"]["next_remaining_m"] - r["targets"]["next_remaining_m"]) for r in predicted], 50),
                        "head_next_is_join_accuracy": float(np.mean([
                            (r["predicted_next_is_join_probability"] >= 0.5) == bool(r["next_is_join"]) for r in join_rows
                        ])) if join_rows else float("nan"),
                    })
                eta = [r for r in members if "predicted" in r]
                if eta:
                    cell.update({
                        "head_T_error_mae_s": float(np.mean([abs(r["predicted"]["T_s"] - r["targets"]["T_s"]) for r in eta])),
                        "head_remaining_error_p50_m": _p([abs(r["predicted"]["remaining_m"] - r["targets"]["remaining_m"]) for r in eta], 50),
                    })
                cell.update({
                    "next_lead_path_p10_m": _p([r["next_lead_path_m"] for r in with_fix], 10),
                    "next_lead_path_p50_m": _p([r["next_lead_path_m"] for r in with_fix], 50),
                    "next_lead_path_p90_m": _p([r["next_lead_path_m"] for r in with_fix], 90),
                    "next_lead_time_p10_s": _p([r["next_lead_time_s"] for r in with_fix], 10),
                    "next_lead_time_p50_s": _p([r["next_lead_time_s"] for r in with_fix], 50),
                    "next_lead_time_p90_s": _p([r["next_lead_time_s"] for r in with_fix], 90),
                    "next_ahead_p50_m": float(med[0]), "next_across_p50_m": float(med[1]),
                    "next_across_abs_p50_m": _p(np.abs(across), 50),
                    "next_speed_p50_mps": _p([r["next_speed_mps"] for r in with_fix], 50),
                    "speed_change_to_next_p50_mps": _p([r["next_speed_mps"] - r["speed_at_anchor_mps"] for r in with_fix], 50),
                    "median_baseline_fix_error_p50_m": _p(baseline_error, 50),
                    "median_baseline_fix_error_p90_m": _p(baseline_error, 90),
                    "next_is_join_majority_accuracy": majority,
                })
            out[f"{stratum} | {kind}"] = cell
    return out


def format_table(summary: dict) -> str:
    metrics = [
        ("anchors", "anchors", 0), ("remaining_at_anchor_p50_m", "remaining at anchor p50 (m)", 0),
        ("censored_share", "censored (established)", 3), ("next_is_join_share_of_open", "next is the join (of open)", 3),
        ("fixes_ahead_mean_of_open", "fixes ahead (mean, open)", 2), ("lead_to_join_p50_m", "lead to join p50 (m)", 0),
        ("next_lead_path_p10_m", "next fix: lead path p10 (m)", 0), ("next_lead_path_p50_m", "next fix: lead path p50 (m)", 0),
        ("next_lead_path_p90_m", "next fix: lead path p90 (m)", 0),
        ("next_lead_time_p10_s", "next fix: lead time p10 (s)", 0), ("next_lead_time_p50_s", "next fix: lead time p50 (s)", 0),
        ("next_lead_time_p90_s", "next fix: lead time p90 (s)", 0),
        ("next_ahead_p50_m", "next fix: ahead p50 (m)", 0), ("next_across_abs_p50_m", "next fix: |across| p50 (m)", 0),
        ("next_speed_p50_mps", "next fix: speed p50 (m/s)", 1), ("speed_change_to_next_p50_mps", "speed change to it p50 (m/s)", 1),
        ("median_baseline_fix_error_p50_m", "median-baseline fix error p50 (m)", 0),
        ("median_baseline_fix_error_p90_m", "median-baseline fix error p90 (m)", 0),
        ("next_is_join_majority_accuracy", "next-is-join majority accuracy", 3),
        # the head's own reading, where the checkpoint is a plan one
        ("head_fix_error_p50_m", "HEAD next fix error p50 (m)", 0), ("head_fix_error_p90_m", "HEAD next fix error p90 (m)", 0),
        ("head_speed_error_p50_mps", "HEAD next speed error p50 (m/s)", 1),
        ("head_next_remaining_error_p50_m", "HEAD next remaining error p50 (m)", 0),
        ("head_next_is_join_accuracy", "HEAD next-is-join accuracy", 3),
        ("head_T_error_mae_s", "HEAD T MAE (s)", 1), ("head_remaining_error_p50_m", "HEAD remaining error p50 (m)", 0),
    ]
    cells = list(summary)
    # the JSON keys are the full stratum labels; the column heads their short names
    heads = {c: f"{STRATUM_SHORT[c.split(' | ')[0]]} | {c.split(' | ')[1]}" for c in cells}
    width = max(len(h) for h in heads.values()) if cells else 10
    lines = ["next-instruction readout — " + "; ".join(cells), ""]
    lines.append(f"{'metric':<36}" + "".join(f"{heads[c]:>{width + 2}}" for c in cells))
    for key, label, digits in metrics:
        row = f"{label:<36}"
        for c in cells:
            value = summary[c].get(key)
            row += f"{'—' if value is None else format(value, f'.{digits}f'):>{width + 2}}"
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the split (a smoke test)")
    parser.add_argument("--draws", type=int, default=5, help="random anchors per flight, uniform in remaining path")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--anchor-s", type=float, default=60.0, help="the fixed anchor: this many seconds after the slice starts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    if len(arms) != 1:
        parser.error("exactly one --checkpoint names the cohort")
    (label, path), = arms.items()
    grid = Grid(split=args.split, bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    arm = load_arm(label, path, grid, torch.device("cpu"), instrument=INSTRUMENT)
    series = cohort_series(arm, grid)
    l1 = default_anchor(arm.config)
    fixed_row = int(round(args.anchor_s / arm.config.dt_s))
    skeletons: dict[tuple[str, str], object] = {}
    rows: list[dict] = []
    for index, item in enumerate(series):
        key = (item.airport, str(item.scenario.source.get("runway")))
        if key not in skeletons:
            skeletons[key] = runway_skeleton(item)
        sk = skeletons[key]
        difficulty = approach_difficulty(item, l1).to_dict()
        anchors = []
        if fixed_row < item.n_samples and truth_duration_s(item, fixed_row) >= MIN_FUTURE_S:
            anchors.append((fixed_row, ANCHOR_FIXED))
        anchors.extend((row, ANCHOR_RANDOM) for row in random_anchors(item, args.draws, args.seed, index))
        for row, kind in anchors:
            rows.append({"dataset_id": item.dataset_id, "flight_id": item.flight_id, "difficulty": difficulty,
                         **next_instruction_row(item, row, sk, kind, arm)})
        if (index + 1) % 200 == 0:
            print(f"  read {index + 1}/{len(series)}", flush=True)
    summary = summarize(rows)
    text = format_table(summary)
    print(text, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out / "plan_next_readout.json", {
        "schema_version": SCHEMA, "generated_at": utc_now(), "instrument": INSTRUMENT,
        "checkpoint": {"label": label, "path": str(path)}, "split": args.split, "limit": args.limit,
        "draws": args.draws, "seed": args.seed, "anchor_s": args.anchor_s, "min_future_s": MIN_FUTURE_S,
        "flights": len(series), "rows": rows, "summary": summary,
    })
    (args.out / "plan_next_readout.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'plan_next_readout.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
