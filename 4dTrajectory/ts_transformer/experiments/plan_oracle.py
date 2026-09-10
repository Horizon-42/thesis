"""Plan-and-guidance step 2: the oracle ceiling — the TRUE plans flown through the guidance layer on the skeleton, scored as a prediction and as a reference.

The design (`docs/2026-09-09_plan_and_guidance_design.md` §9 step 2, §7) asks, before any
head is trained, what the parametrisation can do at best: extract every flight's own eight
plan parameters from its observed track (`outputs/plan/extractors.py`), lay the route
(`outputs/plan/guidance/route.py`), fly it with the guidance controller on the shared
point-mass rollout (`outputs/plan/forecast.py`), and score the result twice — as a
PREDICTION against the observed track (ADE, FDE, chamfer, Fréchet, the arrival-time
error, per stratum) and as a REFERENCE against the procedure and the envelope (fully
flyable, established on the final at the threshold, corridor and glidepath violations on
the final). If the ceiling is worse than today's model as a prediction, the
parametrisation is too coarse (the §7 veto) and must be revised before anything is trained.

The cohort is a checkpoint's own split, rebuilt the way the anytime curve rebuilds it, at
the checkpoint's fixed anchor (L−1); the checkpoint's model is not run. Records are cut at
the first threshold crossing on the final (`forecast.cut_at_threshold_crossing`), as every
hooked-arm readout is.

    python run_ts.py plan_oracle --checkpoint native32=<run>/checkpoint.pt --out <dir>
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ts_transformer.config import default_anchor
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    strata_masks,
)
from ts_transformer.data.channels import IDX
from ts_transformer.experiments.anytime_curve import Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.support import forecast_geometry
from ts_transformer.geometry.final_approach_geometry import (
    corridor_violations,
    hard_on_final,
    position_direction,
    alignment_cosine,
    runway_axes,
)
from ts_transformer.geometry.flyability import flyability_summary, required_controls
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import cut_at_threshold_crossing
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import fly_plans
from ts_transformer.outputs.plan.guidance.route import KIND_DOWNWIND, KIND_STRETCHED
from ts_transformer.outputs.plan.skeleton import runway_skeleton

INSTRUMENT = "the plan oracle"
SCHEMA = "ts-plan-oracle-v1"
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ESTABLISHED)
_SHORT = {
    STRATUM_ALL: "all", STRATUM_STRAIGHT_IN: "straight-in", STRATUM_VECTORED: "vectored",
    STRATUM_ESTABLISHED: "established",
}
#: The rollout horizon past the plan's T: a flight that arrives late is measured as late.
HORIZON_SLACK_S = 30.0
HORIZON_SLACK_FRACTION = 0.1
#: The design's §7 "today's best" column, re-read off the artifacts on 2026-09-10 (its
#: provenance note): what the ceiling is compared against, per stratum where it exists.
TODAYS_BEST = {
    "vectored_ade_mean_m": 2870.2,       # L1_native32, vectored
    "straight_in_chamfer_p50_m": 109.0,  # L1_native32, straight-in
    "pooled_final_time_mae_s": 25.9,     # L1_native32
    "fully_flyable_share": 0.46,         # L3.e-r / L3.f-r stack, true time, cut at the threshold
    "established_share": 0.70,           # L3.e-r +60 s
}


def _p(values, quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def reference_verdicts(series, forecast, skeleton) -> dict[str, float | bool]:
    """The reference reading of one forecast: flyable, established, inside the corridor and
    the glidepath window on the final."""
    geodetic = forecast.geodetic_values
    rows = [
        {"t": float(t), "lat": float(g[0]), "lon": float(g[1]), "alt": float(g[2]),
         "V": float(g[3]), "psi": float(g[4]), "gamma": float(g[5]), "m": float(g[6])}
        for t, g in zip(np.cumsum(forecast.sample_durations_s), geodetic, strict=True)
    ]
    controls = required_controls(rows, series.scenario.aircraft, aero=series.scenario.aero)
    summary = flyability_summary(controls, aircraft_code=str(series.scenario.aircraft.code))
    values = np.asarray(forecast.values, dtype=np.float64)
    e = torch.from_numpy(values[:, IDX["e"]])[None]
    n = torch.from_numpy(values[:, IDX["n"]])[None]
    u = torch.from_numpy(values[:, IDX["u"]])[None]
    psi = torch.tensor([skeleton.course_rad], dtype=torch.float64)
    target = torch.from_numpy(np.asarray(series.target_chart, dtype=np.float64))
    d, xt = runway_axes(e - target[0], n - target[1], psi)
    anchor_e = torch.tensor([float(series.values[forecast.anchor, IDX["e"]])], dtype=torch.float64)
    anchor_n = torch.tensor([float(series.values[forecast.anchor, IDX["n"]])], dtype=torch.float64)
    v_e, v_n = position_direction(e, n, anchor_e, anchor_n)
    cos_align = alignment_cosine(v_e, v_n, psi)
    # the reference is graded on the final BEFORE the threshold: past it the glidepath is
    # the ground and the record is already cut there
    on_final = (hard_on_final(d, xt, cos_align) & (d > 0.0))[0].numpy()
    lateral, vertical = corridor_violations(
        d, xt, u - target[2], torch.tensor([skeleton.glidepath_tan], dtype=torch.float64)
    )
    # the corridor binds from the JOIN on, as the optimizer's constraint rows do: from the
    # first on-final row inside the design corridor (the join point on the centreline),
    # never on the intercept before it — a tangent turn onto the final passes the gate's 30°
    # alignment while still a quarter of a turn radius wide of the centreline
    inside = on_final & (lateral[0].numpy() <= 0.0)
    entered = np.cumsum(inside) > 0
    graded = on_final & entered
    lateral = lateral[0].numpy()[graded]
    vertical = vertical[0].numpy()[graded]
    return {
        "fully_flyable": bool(summary["fully_flyable"]),
        "violations": dict(summary["violations"]),
        "soft_violations": dict(summary["soft"]),
        "established": bool(forecast.truncated_at_threshold),
        "on_final_rows": int(on_final.sum()),
        "graded_rows": int(graded.sum()),
        "lateral_violation": bool(lateral.size and lateral.max() > 0.0),
        "lateral_violation_max_m": float(lateral.max()) if lateral.size else 0.0,
        "glidepath_violation": bool(vertical.size and vertical.max() > 0.0),
        "glidepath_violation_max_m": float(vertical.max()) if vertical.size else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    covariates = {row["dataset_id"]: row["difficulty"] for row in rows}
    masks = strata_masks(covariates, [row["dataset_id"] for row in rows])
    out: dict = {}
    for stratum in STRATA:
        members = [row for row, keep in zip(rows, masks[stratum], strict=True) if keep]
        if not members:
            out[stratum] = {"flights": 0}
            continue
        ade = [row["prediction"]["ade_m"] for row in members]
        fde = [row["prediction"]["fde_m"] for row in members]
        dt = [row["prediction"]["final_time_error_s"] for row in members]
        out[stratum] = {
            "flights": len(members),
            "ade_mean_m": float(np.mean(ade)), "ade_p50_m": _p(ade, 50), "ade_p95_m": _p(ade, 95),
            "fde_p50_m": _p(fde, 50),
            "chamfer_p50_m": _p([row["geometry"]["chamfer_m"] for row in members], 50),
            "frechet_p50_m": _p([row["geometry"]["frechet_m"] for row in members], 50),
            "abs_final_time_error_p50_s": _p(np.abs(dt), 50),
            "abs_final_time_error_p80_s": _p(np.abs(dt), 80),
            "final_time_error_mae_s": float(np.mean(np.abs(dt))),
            "route_time_minus_T_p50_s": _p([row["route"]["route_time_s"] - row["T_s"] for row in members], 50),
            "fully_flyable_share": float(np.mean([row["reference"]["fully_flyable"] for row in members])),
            "established_share": float(np.mean([row["reference"]["established"] for row in members])),
            "lateral_violation_share": float(np.mean([row["reference"]["lateral_violation"] for row in members])),
            "glidepath_violation_share": float(np.mean([row["reference"]["glidepath_violation"] for row in members])),
            "route_shortfall_p50_m": _p([row["route"]["shortfall_m"] for row in members], 50),
            "route_intercept_abs_p50_deg": _p([abs(row["route"]["intercept_deg"]) for row in members], 50),
            "route_stretched_share": float(np.mean([row["route"]["kind"] in (KIND_DOWNWIND, KIND_STRETCHED) for row in members])),
            "hook_bank_capped_share": float(np.mean([row["hook"]["planBankCappedSteps"] for row in members])),
            "hook_thrust_saturated_share": float(np.mean([row["hook"]["planThrustSaturatedSteps"] for row in members])),
            "hook_thrust_idle_share": float(np.mean([row["hook"]["planThrustIdleSteps"] for row in members])),
            "hook_load_clamped_share": float(np.mean([row["hook"]["planLoadClampedSteps"] for row in members])),
            "hook_route_cross_track_p50_m": _p([row["hook"]["planRouteCrossTrackM"] for row in members], 50),
            # the corridor barrier composed on the final: where its gate was open, and where
            # it clamped the tracker's bank
            "barrier_gated_share": float(np.mean([row["hook"]["gatedSteps"] for row in members])),
            "barrier_clamped_share": float(np.mean([row["hook"]["clampedSteps"] for row in members])),
            "capture_height_clamped_share": float(np.mean([row["hook"]["planCaptureHeightClamped"] for row in members])),
        }
    return out


def format_table(summary: dict) -> str:
    strata = [s for s in STRATA if summary[s]["flights"]]
    metrics = [
        ("ade_mean_m", "ADE mean", 0), ("ade_p50_m", "ADE p50", 0), ("ade_p95_m", "ADE p95", 0),
        ("fde_p50_m", "FDE p50", 0), ("chamfer_p50_m", "chamfer p50", 0), ("frechet_p50_m", "Fréchet p50", 0),
        ("abs_final_time_error_p50_s", "|dt| p50 s", 1), ("abs_final_time_error_p80_s", "|dt| p80 s", 1),
        ("final_time_error_mae_s", "dt MAE s", 1), ("route_time_minus_T_p50_s", "route t − T p50 s", 1),
        ("fully_flyable_share", "fully flyable", 3), ("established_share", "established", 3),
        ("lateral_violation_share", "lateral viol.", 3), ("glidepath_violation_share", "glidepath viol.", 3),
        ("route_shortfall_p50_m", "route shortfall p50", 0), ("route_stretched_share", "route stretched", 3),
        ("route_intercept_abs_p50_deg", "route |intercept| p50", 1),
        ("hook_bank_capped_share", "bank capped", 3), ("hook_thrust_saturated_share", "thrust saturated", 3),
        ("hook_thrust_idle_share", "thrust idle", 3),
        ("hook_load_clamped_share", "load clamped", 3), ("hook_route_cross_track_p50_m", "route xt p50", 0),
        ("barrier_gated_share", "barrier gated", 3), ("barrier_clamped_share", "barrier clamped", 3),
        ("capture_height_clamped_share", "capture h clamped", 3),
    ]
    lines = ["oracle ceiling (true plans through the guidance), n = " + ", ".join(
        f"{_SHORT[s]} {summary[s]['flights']}" for s in strata)]
    lines.append(f"{'metric':<22}" + "".join(f"{_SHORT[s]:>14}" for s in strata))
    for key, label, digits in metrics:
        lines.append(f"{label:<22}" + "".join(f"{summary[s][key]:>14.{digits}f}" for s in strata))
    lines.append("")
    lines.append("today's best (design §7, provenance-checked): " + ", ".join(
        f"{k} {v}" for k, v in TODAYS_BEST.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=32)
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
    for start in range(0, len(series), args.batch_size):
        chunk = series[start:start + args.batch_size]
        chunk_skeletons = []
        for item in chunk:
            key = (item.airport, str(item.scenario.source.get("runway")))
            if key not in skeletons:
                skeletons[key] = runway_skeleton(item)
            chunk_skeletons.append(skeletons[key])
        labels = [extract_plan(item, anchor, sk) for item, sk in zip(chunk, chunk_skeletons, strict=True)]
        # the rollout runs PAST the plan's T so a late arrival is measured as late rather
        # than cut off unestablished; the metrics' clock is the truth's regardless
        horizons = [lab.T_s + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION * lab.T_s) for lab in labels]
        forecasts, routes, times = fly_plans(
            chunk, anchor, labels, chunk_skeletons, arm.config, durations_s=horizons,
        )
        for item, sk, lab, forecast, route, route_time in zip(
            chunk, chunk_skeletons, labels, forecasts, routes, times, strict=True
        ):
            cut = cut_at_threshold_crossing(forecast, item)
            # the oracle's arrival time is where the rollout crossed the threshold (the
            # rollout's end where it never did) — the plan's T is the truth, not a prediction
            cut = replace(cut, predicted_final_time_s=cut.final_time_s)
            metrics = observed_series_metrics(item, cut)
            rows.append({
                "dataset_id": item.dataset_id,
                "flight_id": item.flight_id,
                "difficulty": approach_difficulty(item, anchor).to_dict(),
                "T_s": lab.T_s,
                "labels": lab.to_dict(),
                "route": {
                    "kind": route.kind, "length_m": route.length_m,
                    "pre_final_m": route.pre_final_m, "requested_pre_final_m": route.requested_pre_final_m,
                    "shortfall_m": route.shortfall_m, "stretch_offset_m": route.stretch_offset_m,
                    "intercept_deg": math.degrees(route.intercept_rad),
                    "route_time_s": route_time,
                },
                "prediction": {
                    key: metrics[key] for key in (
                        "ade_m", "fde_m", "horizontal_ade_m", "arrival_endpoint_error_m",
                        "final_time_error_s", "true_final_time_s", "coverage_ratio",
                    )
                },
                "geometry": forecast_geometry(item, cut),
                "reference": reference_verdicts(item, cut, sk),
                "hook": cut.command_hook_diagnostics,
                "cut_final_time_s": cut.final_time_s,
            })
        print(f"  flown {len(rows)}/{len(series)}", flush=True)
    summary = summarize(rows)
    text = format_table(summary)
    print(text, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out / "plan_oracle.json", {
        "schema_version": SCHEMA, "generated_at": utc_now(), "instrument": INSTRUMENT,
        "checkpoint": {"label": label, "path": str(path)},
        "split": args.split, "limit": args.limit, "flights": len(rows), "anchor": anchor,
        "todays_best": TODAYS_BEST, "summary": summary, "rows": rows,
    })
    (args.out / "plan_oracle.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'plan_oracle.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
