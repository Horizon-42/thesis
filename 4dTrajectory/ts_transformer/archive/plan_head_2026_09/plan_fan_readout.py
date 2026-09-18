"""Plan-and-guidance step 3(g): the fan over the next fix — a fan head's K components read at the anchor, and flown one step deep as lockstep members against the top-1 and a displaced control.

The design (`docs/2026-09-09_plan_and_guidance_design.md` §9 step 3(g), §3b) makes the
next instruction a small readable set — "the next vector is one of these" — rather than a
committed point. This runner reads a `plan_fan_components` ≥ 2 checkpoint twice:

- **single step**, at the anchor: against the truth's next fix (where the track has one),
  the top-weight component's fix error, the NEAREST component's, whether the truth lies
  inside any component's 2σ box (ahead and across), and the components' usage (how often
  each is the top weight — a dead component shows as 0);
- **rolled**: the top-1 flight (`rolled_predictions_lockstep`, the deployed prediction),
  every component as a member whose FIRST order is that component and every later order
  the top-1's (the fan one step deep, `lockstep_model_policy(first_component=)`), and a
  DISPLACED control fan of the same K — the top-1's first fix moved `--control-radius-m`
  in the flight's RUNWAY AXES at K evenly spaced bearings from "toward the runway" (the
  schedule coordinate moved with it), a fan with the same K and spread and no learned
  structure — scored as the latent fan is (`latent_fan_readout`, `geometry_cell`): per
  stratum the top-1 ADE, minADE over the top-1 AND the members (the top-1 is a member of
  BOTH sets — the top component flies the top-1's flight again, and the control is read
  the same way), the truth's chamfer to the NEAREST member against its chamfer to the
  top-1 with the share of flights the nearest member beats it on, and the share where
  the top-1 or any member is established. Only flights whose FIRST order flies a fix are
  fanned: where the head says "no fix ahead" (or the aircraft is on the final) every
  member is the top-1 again, so those flights are counted (`head fix flown`) and not
  rolled 2K more times.

    python run_ts.py plan_fan_readout --checkpoint fan4=<run>/checkpoint.pt --anchor-s 60 --out <dir>

A fan always holds something nearer the truth than its own top-1, so the fan's columns
are read against the displaced control's, never alone.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ts_transformer.config import PREDICTION_PLAN, default_anchor
from ts_transformer.data.approach_difficulty import STRATUM_SHORT, approach_difficulty, strata_masks
from ts_transformer.data.dataset import truth_duration_s
from ts_transformer.data.target_conditioning import conditioned_history
from ts_transformer.data.dataset import series_conditioning
from ts_transformer.experiments.anytime_curve import Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.plan_oracle import HORIZON_SLACK_S, STRATA
from ts_transformer.experiments.quantile_fan_readout import geometry_cell
from ts_transformer.experiments.support import forecast_geometry
from ts_transformer.geometry.dubins import chart_from_axes_np
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import cut_at_threshold_crossing
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.dynamics.context import ANCHOR_CONTROL_SAMPLES
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import rolled_history
from ts_transformer.outputs.plan.labels import INSTRUCTION, TARGETS, PlanOrder, targets_from_labels
from ts_transformer.outputs.plan.model import fan_rows
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton, runway_skeleton
from ts_transformer.outputs.plan.strategy import rolled_predictions_lockstep

INSTRUMENT = "the plan fan readout"
SCHEMA = "ts-plan-fan-readout-v1"
#: The control fan's displacement: §12.4's median-baseline next-fix error on the vectored
#: stratum at the 60 s anchor (5.1 km) — the spread a fan with no learned structure has.
DEFAULT_CONTROL_RADIUS_M = 5_000.0
#: A truth fix inside this many σ of a component, ahead AND across, is covered by it.
COVERAGE_SIGMAS = 2.0


def _p(values, quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if len(values) else float("nan")


def displaced(order: PlanOrder, radius_m: float, bearing_rad: float, skeleton: RunwaySkeleton) -> PlanOrder:
    """The order with its fix moved ``radius_m`` in the runway axes — ``bearing_rad`` = 0
    toward the runway along the course, π/2 to its right — and the fix's schedule
    coordinate moved with it (a fix nearer the runway has that much less path to go), so
    a control member is the same kind of object as a fan component. An order with no fix
    is returned as it is."""
    if order.instruction is None:
        return order
    ahead, across = radius_m * math.cos(bearing_rad), radius_m * math.sin(bearing_rad)
    d, xt = skeleton.axes(np.array([order.instruction.fix_e]), np.array([order.instruction.fix_n]))
    e, n = chart_from_axes_np(np.array([float(d[0]) - ahead]), np.array([float(xt[0]) + across]), skeleton.course_rad)
    fix = replace(
        order.instruction,
        fix_e=float(e[0]) + skeleton.target_e, fix_n=float(n[0]) + skeleton.target_n,
        remaining_m=max(order.instruction.remaining_m - ahead, 0.0),
    )
    return replace(order, instruction=fix)


def single_step(fan: np.ndarray, weights: np.ndarray, sigma: np.ndarray, truth: np.ndarray, has_fix: bool) -> dict:
    """One flight's single-step reading: the top-1 and nearest-component fix errors (m), the
    2σ coverage, the top weight — against the truth's next fix in runway axes, or None."""
    ahead, across = INSTRUCTION.index("next_ahead_m"), INSTRUCTION.index("next_across_m")
    i_ahead, i_across = TARGETS.index("next_ahead_m"), TARGETS.index("next_across_m")
    top = int(weights.argmax())
    row = {"top_component": top, "top_weight": float(weights[top]), "has_truth_fix": bool(has_fix)}
    if not has_fix:
        return row
    error = np.hypot(fan[:, i_ahead] - truth[i_ahead], fan[:, i_across] - truth[i_across])
    inside = (
        (np.abs(fan[:, i_ahead] - truth[i_ahead]) <= COVERAGE_SIGMAS * sigma[:, ahead])
        & (np.abs(fan[:, i_across] - truth[i_across]) <= COVERAGE_SIGMAS * sigma[:, across])
    )
    row.update({
        "top_fix_error_m": float(error[top]), "nearest_fix_error_m": float(error.min()),
        "nearest_component": int(error.argmin()), "covered_2sigma": bool(inside.any()),
    })
    return row


def member_scores(item, flights) -> list[dict]:
    """Each rolled flight cut at its threshold crossing: ADE, chamfer, established."""
    out = []
    for flight in flights:
        cut = cut_at_threshold_crossing(flight.forecast, item)
        metrics = observed_series_metrics(item, replace(cut, predicted_final_time_s=cut.final_time_s))
        out.append({
            "ade_m": metrics["ade_m"], "chamfer_m": forecast_geometry(item, cut)["chamfer_m"],
            "established": bool(cut.truncated_at_threshold),
        })
    return out


def summarize(rows: list[dict], k: int) -> dict:
    covariates = {row["dataset_id"]: row["difficulty"] for row in rows}
    masks = strata_masks(covariates, [row["dataset_id"] for row in rows])
    out: dict = {}
    for stratum in STRATA:
        members = [row for row, keep in zip(rows, masks[stratum], strict=True) if keep]
        if not members:
            out[stratum] = {"flights": 0}
            continue
        with_fix = [row for row in members if row["single"]["has_truth_fix"]]
        fanned = [row for row in members if row["fan"] is not None]
        top_ade = [row["top"]["ade_m"] for row in members]
        top_chamfer = [row["top"]["chamfer_m"] for row in members]

        def fan_block(key: str) -> dict:
            # the fan's flights only: the top-1 is a member of the set (minADE over the
            # top-1 and the members, as `latent_fan_readout` reads minADE_K), the chamfer
            # cell the shared `geometry_cell` (nearest MEMBER against the top-1, strict)
            top = np.array([row["top"]["ade_m"] for row in fanned])
            min_ade = np.minimum(top, np.array([min(m["ade_m"] for m in row[key]) for row in fanned])) if fanned else np.zeros(0)
            reference = np.array([row["top"]["chamfer_m"] for row in fanned])
            nearest = np.array([min(m["chamfer_m"] for m in row[key]) for row in fanned])
            return {
                "flights": len(fanned),
                "min_ade_mean_m": float(np.mean(min_ade)) if fanned else float("nan"), "min_ade_p50_m": _p(min_ade, 50),
                **geometry_cell(reference, nearest, reference_key="top_chamfer_p50_m"),
                "any_established_share": float(np.mean([
                    row["top"]["established"] or any(m["established"] for m in row[key]) for row in fanned
                ])) if fanned else float("nan"),
            }

        out[stratum] = {
            "flights": len(members), "flights_with_truth_fix": len(with_fix),
            "head_fix_flown_share": float(np.mean([row["fan"] is not None for row in members])),
            "top_ade_mean_m": float(np.mean(top_ade)), "top_ade_p50_m": _p(top_ade, 50),
            "top_chamfer_p50_m": _p(top_chamfer, 50),
            "top_established_share": float(np.mean([row["top"]["established"] for row in members])),
            "fan": fan_block("fan"), "control": fan_block("control"),
            "single_step": {
                "top_fix_error_p50_m": _p([row["single"]["top_fix_error_m"] for row in with_fix], 50),
                "nearest_fix_error_p50_m": _p([row["single"]["nearest_fix_error_m"] for row in with_fix], 50),
                "covered_2sigma_share": float(np.mean([row["single"]["covered_2sigma"] for row in with_fix])) if with_fix else float("nan"),
                "top_weight_mean": float(np.mean([row["single"]["top_weight"] for row in members])),
                "usage": [float(np.mean([row["single"]["top_component"] == c for row in members])) for c in range(k)],
            },
        }
    return out


def format_table(summary: dict, k: int) -> str:
    strata = [s for s in STRATA if summary[s]["flights"]]
    lines = [f"the fan over the next fix (K = {k}), one step deep; the top-1 is a member of both sets; n = " + ", ".join(
        f"{STRATUM_SHORT[s]} {summary[s]['flights']}" for s in strata)]
    lines.append(f"{'metric':<34}" + "".join(f"{STRATUM_SHORT[s]:>14}" for s in strata))
    def num(value, digits: int) -> str:
        # a stratum with no fanned flight (the established one: no fix ahead) has no fan cell
        return "n/a" if value is None or value != value else f"{value:.{digits}f}"

    rows = [
        ("top-1 ADE mean / p50", lambda b: f"{num(b['top_ade_mean_m'], 0)} / {num(b['top_ade_p50_m'], 0)}"),
        ("top-1 chamfer p50", lambda b: num(b['top_chamfer_p50_m'], 0)),
        ("top-1 established", lambda b: num(b['top_established_share'], 3)),
        ("head fix flown (fanned)", lambda b: f"{num(b['head_fix_flown_share'], 3)} ({b['fan']['flights']})"),
        ("fanned: top-1 chamfer p50", lambda b: num(b['fan']['top_chamfer_p50_m'], 0)),
        ("fan minADE_K mean / p50", lambda b: f"{num(b['fan']['min_ade_mean_m'], 0)} / {num(b['fan']['min_ade_p50_m'], 0)}"),
        ("control minADE_K mean / p50", lambda b: f"{num(b['control']['min_ade_mean_m'], 0)} / {num(b['control']['min_ade_p50_m'], 0)}"),
        ("fan nearest chamfer p50", lambda b: num(b['fan']['chamfer_nearest_p50_m'], 0)),
        ("control nearest chamfer p50", lambda b: num(b['control']['chamfer_nearest_p50_m'], 0)),
        ("fan nearest beats top-1", lambda b: num(b['fan']['nearest_better_share'], 3)),
        ("control nearest beats top-1", lambda b: num(b['control']['nearest_better_share'], 3)),
        ("fan any established", lambda b: num(b['fan']['any_established_share'], 3)),
        ("control any established", lambda b: num(b['control']['any_established_share'], 3)),
        ("flights with a truth fix", lambda b: f"{b['flights_with_truth_fix']}"),
        ("single: top-1 fix error p50", lambda b: num(b['single_step']['top_fix_error_p50_m'], 0)),
        ("single: nearest fix error p50", lambda b: num(b['single_step']['nearest_fix_error_p50_m'], 0)),
        ("single: truth inside 2σ", lambda b: num(b['single_step']['covered_2sigma_share'], 3)),
        ("single: top weight mean", lambda b: num(b['single_step']['top_weight_mean'], 3)),
    ]
    for label, cell in rows:
        lines.append(f"{label:<34}" + "".join(f"{cell(summary[s]):>14}" for s in strata))
    # the usage per component (how often each is the top weight): one line per component
    for component in range(k):
        lines.append(f"{f'single: usage of component {component}':<34}" + "".join(
            f"{summary[s]['single_step']['usage'][component]:>14.3f}" for s in strata
        ))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--anchor-s", type=float, default=0.0,
                        help="the one row this many seconds after the slice starts (0 = L-1), as plan_oracle")
    parser.add_argument("--control-radius-m", type=float, default=DEFAULT_CONTROL_RADIUS_M,
                        help="the displaced control fan's radius about the top-1's first fix")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    if len(arms) != 1:
        parser.error("exactly one --checkpoint names the fan head")
    (label, path), = arms.items()
    grid = Grid(split=args.split, bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    arm = load_arm(label, path, grid, torch.device("cpu"), instrument=INSTRUMENT)
    k = arm.config.plan_fan_components
    if arm.config.prediction_output != PREDICTION_PLAN or k < 2:
        parser.error(f"{label} is not a fan head (prediction_output={arm.config.prediction_output!r}, plan_fan_components={k})")
    series = cohort_series(arm, grid)
    if args.anchor_s > 0.0:
        row = int(round(args.anchor_s / arm.config.dt_s))
        if row < ANCHOR_CONTROL_SAMPLES - 1:
            parser.error(f"--anchor-s needs at least {(ANCHOR_CONTROL_SAMPLES - 1) * arm.config.dt_s:g} s of track")
        anchors = {
            item.dataset_id: (row if row < item.n_samples and truth_duration_s(item, row) >= HORIZON_SLACK_S else None)
            for item in series
        }
        without = sum(1 for a in anchors.values() if a is None)
        series = [item for item in series if anchors[item.dataset_id] is not None]
        print(f"  anchor at {args.anchor_s:g} s (row {row}): {len(series)} flights have one, {without} do not", flush=True)
    else:
        anchors = {item.dataset_id: default_anchor(arm.config) for item in series}
        without = 0
    device = torch.device("cpu")
    bearings = [2.0 * math.pi * j / k for j in range(k)]
    skeletons: dict[tuple[str, str], object] = {}
    rows: list[dict] = []
    started = time.perf_counter()
    arm.model.eval()
    for start in range(0, len(series), args.batch_size):
        chunk = series[start:start + args.batch_size]
        chunk_skeletons = []
        for item in chunk:
            key = (item.airport, str(item.scenario.source.get("runway")))
            if key not in skeletons:
                skeletons[key] = runway_skeleton(item)
            chunk_skeletons.append(skeletons[key])
        chunk_anchors = [int(anchors[item.dataset_id]) for item in chunk]
        # the single step: the fan at the anchor window against the truth's next fix
        windows = np.stack([
            conditioned_history(
                arm.normalizer.encode(rolled_history(item, a, [], arm.config)),
                series_conditioning(item, arm.config, arm.normalizer, anchor=a),
            )
            for item, a in zip(chunk, chunk_anchors, strict=True)
        ]).astype(np.float32)
        with torch.no_grad():
            prediction = arm.model(torch.from_numpy(windows).to(device))
        fan, weights, sigma = fan_rows(prediction)
        singles = []
        for i, (item, a, sk) in enumerate(zip(chunk, chunk_anchors, chunk_skeletons, strict=True)):
            targets = targets_from_labels(extract_plan(item, a, sk), item, a, sk)
            has_fix = not targets.next_is_join and bool(targets.valid[TARGETS.index("next_ahead_m")])
            singles.append(single_step(fan[i], weights[i], sigma[i], targets.values.astype(np.float64), has_fix))
        # the rolled flights: the top-1 for every flight; the K members and the K displaced
        # controls for the flights whose first order FLEW a fix (elsewhere every member is
        # the top-1 again)
        top = rolled_predictions_lockstep(arm.model, chunk, arm.config, arm.normalizer, chunk_anchors, device, chunk_skeletons)
        fanned = [i for i in range(len(chunk)) if top[i].orders[0]["flown_fix"] is not None]
        sub = (
            arm.model, [chunk[i] for i in fanned], arm.config, arm.normalizer, [chunk_anchors[i] for i in fanned], device,
            [chunk_skeletons[i] for i in fanned],
        )
        members = [rolled_predictions_lockstep(*sub, first_component=c) for c in range(k)] if fanned else []
        controls = [
            rolled_predictions_lockstep(
                *sub, first_order=lambda o, s, b=b: displaced(o, args.control_radius_m, b, s.skeleton),
            )
            for b in bearings
        ] if fanned else []
        position = {i: j for j, i in enumerate(fanned)}
        for i, (item, a, single) in enumerate(zip(chunk, chunk_anchors, singles, strict=True)):
            j = position.get(i)
            rows.append({
                "dataset_id": item.dataset_id, "flight_id": item.flight_id, "anchor": a,
                "difficulty": approach_difficulty(item, default_anchor(arm.config)).to_dict(),
                "single": single,
                "top": member_scores(item, [top[i]])[0],
                "fan": None if j is None else member_scores(item, [m[j] for m in members]),
                "control": None if j is None else member_scores(item, [c[j] for c in controls]),
            })
        print(f"  flown {len(rows)}/{len(series)} (fanned {len(fanned)} of {len(chunk)} × 2 × {k})", flush=True)
    wall_s = time.perf_counter() - started
    summary = summarize(rows, k)
    # the rows are the measurement: written first, so a fault in the table never loses them
    # (2026-09-12: a 40-minute run died formatting a stratum with no fanned flight)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out / "plan_fan_readout.json", {
        "schema_version": SCHEMA, "generated_at": utc_now(), "instrument": INSTRUMENT,
        "checkpoint": {"label": label, "path": str(path)}, "split": args.split, "limit": args.limit,
        "flights": len(rows), "anchor_s": args.anchor_s, "flights_without_anchor": without,
        "fan_components": k, "control_radius_m": args.control_radius_m, "coverage_sigmas": COVERAGE_SIGMAS,
        "wall_s": wall_s, "summary": summary, "rows": rows,
    })
    text = format_table(summary, k)
    text += f"\nflown in {wall_s:.0f} s ({wall_s / max(len(rows), 1):.2f} s per flight; a fanned flight is {1 + 2 * k} rolled flights)\n"
    print(text, flush=True)
    (args.out / "plan_fan_readout.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'plan_fan_readout.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
