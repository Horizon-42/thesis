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
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ts_transformer.config import PREDICTION_PLAN, default_anchor
from ts_transformer.data.anchor_grid import bin_anchor
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_SHORT,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    strata_masks,
)
from ts_transformer.data.approach_difficulty import remaining_path_profile_m
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import truth_duration_s
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
from ts_transformer.outputs.dynamics.context import ANCHOR_CONTROL_SAMPLES
from ts_transformer.outputs.plan.extractors import MAX_WAYPOINTS, extract_plan
from ts_transformer.outputs.plan.forecast import (
    KIND_ROLLED,
    ROUTE_PLAN,
    ROUTES,
    fly_plans,
    LOCKSTEP_S,
    ORDER_HOLD_ASKS,
    fly_lockstep_truth,
    fly_rolling,
)
from ts_transformer.outputs.plan.labels import truth_instructions
from ts_transformer.outputs.plan.rolled import POLICY_MODEL as ROLLED_POLICY_MODEL, POLICY_TRUTH as ROLLED_POLICY_TRUTH
from ts_transformer.outputs.plan.guidance.timing import TIME_TOLERANCE_S
from ts_transformer.outputs.plan.strategy import Assignment, rolled_prediction, rolled_predictions_lockstep
from ts_transformer.outputs.plan.guidance.route import KIND_DOWNWIND, KIND_STRETCHED, KIND_WAYPOINTS
from ts_transformer.outputs.plan.skeleton import runway_skeleton

INSTRUMENT = "the plan oracle"
#: Before the FAF the coded floor applies (the optimizer's `prefaf_floor_m` rows), with
#: this much below it tolerated — the observed evaluation's altitude tolerance.
FLOOR_TOLERANCE_M = 30.0
#: `--policy model`: the plan HEAD's orders (a plan checkpoint) rolled through the same
#: guidance and graded by the same instrument — the rolled prediction beside its ceiling.
POLICY_TRUTH = ROLLED_POLICY_TRUTH
POLICY_MODEL = ROLLED_POLICY_MODEL
#: `--rolling`: one instruction per leg, the head asked at each fix (`fly_legs`, design §12.4),
#: or receding-horizon lockstep — asked every `LOCKSTEP_S`, the group stepped together (v5.1).
ROLLING_LEG = "leg"
ROLLING_LOCKSTEP = "lockstep"
#: `--route next`: the truth's instructions flown one at a time (design v5), re-anchored
#: at each fix — the rolled form of the waypoints oracle.
ROUTE_NEXT = "next"
#: `--assign-time truth` / `--assign-join truth` (v5.4, §9 step 4): the truth's arrival time
#: (plus `--assign-time-offset-s`) and/or the truth's join are the scheduler's assignment to
#: the head's rolled flight — the oracle form, which READS THE FUTURE and says so in the
#: artifact (`assignment`); never a prediction result.
ASSIGN_NONE = "none"
ASSIGN_TRUTH = "truth"
ASSIGNMENTS = (ASSIGN_NONE, ASSIGN_TRUTH)
SCHEMA = "ts-plan-oracle-v2"   # v2 (2026-09-11): glidepath verdict inside the FAF only, floor before it, the truth graded alongside
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ESTABLISHED)
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


def held_share(hook: dict) -> float:
    """The share of a lockstep flight's steps the order hold kept the order in force on
    (v5.3); every lockstep flight takes at least one step."""
    return hook["planHeldSteps"] / hook["planSteps"]


def corridor_verdicts(series, values: np.ndarray, anchor: int, skeleton) -> dict[str, float | bool]:
    """The corridor reading of one track from ``anchor``: on the final, inside the design
    corridor from the join on, in the glidepath window inside the FAF, above the coded floor
    before it. Read on a forecast AND on the truth's own future rows, so a flown share is
    read against the truth's under the same rule (2026-09-11: the truth fails the pre-FAF
    floor on 14/48 smoke flights — vectored aircraft are assigned altitudes below the coded
    IF floor — and the glidepath window on 3/48)."""
    values = np.asarray(values, dtype=np.float64)
    e = torch.from_numpy(values[:, IDX["e"]])[None]
    n = torch.from_numpy(values[:, IDX["n"]])[None]
    u = torch.from_numpy(values[:, IDX["u"]])[None]
    psi = torch.tensor([skeleton.course_rad], dtype=torch.float64)
    target = torch.from_numpy(np.asarray(series.target_chart, dtype=np.float64))
    d, xt = runway_axes(e - target[0], n - target[1], psi)
    anchor_e = torch.tensor([float(series.values[anchor, IDX["e"]])], dtype=torch.float64)
    anchor_n = torch.tensor([float(series.values[anchor, IDX["n"]])], dtype=torch.float64)
    v_e, v_n = position_direction(e, n, anchor_e, anchor_n)
    cos_align = alignment_cosine(v_e, v_n, psi)
    # graded on the final BEFORE the threshold: past it the glidepath is the ground and the
    # record is already cut there
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
    # the vertical verdict as the optimizer's rows read it: the glidepath window binds
    # inside the FAF; before it, on the final, the coded floor at the next fix — a real
    # approach holds its platform altitude below the glidepath until the intercept from
    # below, and graded from the join that read as a violation on the truth itself
    d_rows = d[0].numpy()
    inside_faf = graded & (d_rows <= skeleton.faf.d)
    before_faf = graded & (d_rows > skeleton.faf.d)
    vertical = vertical[0].numpy()[inside_faf]
    altitude = u[0].numpy()[before_faf] + skeleton.aim_altitude_m
    floors = np.array([
        -np.inf if (floor := skeleton.floor_altitude_m(float(dd))) is None else floor
        for dd in d_rows[before_faf]
    ], dtype=np.float64)
    floor_excess = np.clip(floors - FLOOR_TOLERANCE_M - altitude, 0.0, None) if altitude.size else np.zeros(0)
    return {
        "on_final_rows": int(on_final.sum()),
        "graded_rows": int(graded.sum()),
        "lateral_violation": bool(lateral.size and lateral.max() > 0.0),
        "lateral_violation_max_m": float(lateral.max()) if lateral.size else 0.0,
        "glidepath_violation": bool(vertical.size and vertical.max() > 0.0),
        "glidepath_violation_max_m": float(vertical.max()) if vertical.size else 0.0,
        "floor_violation": bool(floor_excess.size and floor_excess.max() > 0.0),
        "floor_violation_max_m": float(floor_excess.max()) if floor_excess.size else 0.0,
    }


def reference_verdicts(series, forecast, skeleton) -> dict[str, float | bool]:
    """The reference reading of one forecast: flyable, established, and the corridor
    verdicts of `corridor_verdicts`."""
    geodetic = forecast.geodetic_values
    rows = [
        {"t": float(t), "lat": float(g[0]), "lon": float(g[1]), "alt": float(g[2]),
         "V": float(g[3]), "psi": float(g[4]), "gamma": float(g[5]), "m": float(g[6])}
        for t, g in zip(np.cumsum(forecast.sample_durations_s), geodetic, strict=True)
    ]
    controls = required_controls(rows, series.scenario.aircraft, aero=series.scenario.aero)
    summary = flyability_summary(controls, aircraft_code=str(series.scenario.aircraft.code))
    return {
        "fully_flyable": bool(summary["fully_flyable"]),
        "violations": dict(summary["violations"]),
        "soft_violations": dict(summary["soft"]),
        "established": bool(forecast.truncated_at_threshold),
        **corridor_verdicts(series, forecast.values, forecast.anchor, skeleton),
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
            "floor_violation_share": float(np.mean([row["reference"]["floor_violation"] for row in members])),
            # the truth's own rows under the same rule
            "lateral_violation_share_truth": float(np.mean([row["truth"]["lateral_violation"] for row in members])),
            "glidepath_violation_share_truth": float(np.mean([row["truth"]["glidepath_violation"] for row in members])),
            "floor_violation_share_truth": float(np.mean([row["truth"]["floor_violation"] for row in members])),
            "route_shortfall_p50_m": _p([row["route"]["shortfall_m"] for row in members], 50),
            "route_intercept_abs_p50_deg": _p([abs(row["route"]["intercept_deg"]) for row in members], 50),
            "route_stretched_share": float(np.mean([row["route"]["kind"] in (KIND_DOWNWIND, KIND_STRETCHED) for row in members])),
            "route_waypoints_share": float(np.mean([row["route"]["kind"] in (KIND_WAYPOINTS, KIND_ROLLED) for row in members])),
            "rolled_legs_mean": float(np.mean([row["route"].get("legs", 0) for row in members])),
            "instructions_skipped_share": float(np.mean([row["route"].get("instructions_skipped", 0) > 0 for row in members])),
            "turns_incomplete_share": float(np.mean([row["route"].get("turns_incomplete", 0) > 0 for row in members])),
            "rolled_capped_share": float(np.mean([bool(row["route"].get("capped_by")) for row in members])),
            # the head's own arrival-time prediction, where the flight was the head's
            "eta_head_mae_s": float(np.mean([
                abs(row["eta_predicted_s"] - row["prediction"]["true_final_time_s"]) for row in members
                if row.get("eta_predicted_s") is not None
            ])) if any(row.get("eta_predicted_s") is not None for row in members) else float("nan"),
            "waypoints_mean": float(np.mean([len(row["labels"]["waypoints"] or ()) for row in members])),
            "waypoints_dropped_share": float(np.mean([row["labels"]["waypoints_dropped"] > 0 for row in members])),
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
            # the order hold (v5.3): the share of a flight's steps the policy's change was
            # held on, and the material changes adopted per flight — lockstep flights only
            # (the whole-path and leg forms carry no `planSteps`)
            "orders_held_share": float(np.mean([held_share(row["hook"]) for row in members]))
            if all("planHeldSteps" in row["hook"] for row in members) else float("nan"),
            # the assignment's closure (v5.4): X at the first ask, the share it did not
            # absorb, the speed factor, the stretch — assigned flights only
            **_assignment_cells(members),
            "order_changes_per_flight": float(np.mean([row["hook"]["planOrderChanges"] for row in members]))
            if all("planOrderChanges" in row["hook"] for row in members) else float("nan"),
        }
    return out


def _assignment_cells(members: list[dict]) -> dict[str, float]:
    """The time closure's readout over the flights that were ASSIGNED a time (`planAssigned`);
    NaN where none was — the whole-path, leg and unassigned lockstep forms."""
    assigned = [row for row in members if row["hook"].get("planAssigned")]
    if not assigned:
        return {
            "assigned_share": float(np.mean([bool(row["hook"].get("planAssigned")) for row in members])),
            "unabsorbed_first_p50_s": float("nan"), "unabsorbed_first_p90_s": float("nan"),
            "unabsorbed_share": float("nan"), "unabsorbed_step_p50_s": float("nan"),
            "speed_factor_first_p50": float("nan"), "stretched_share": float("nan"),
            "stretch_dropped_share": float("nan"), "final_time_error_absorbed_p50_s": float("nan"),
        }
    first = np.array([row["hook"]["planUnabsorbedFirstS"] for row in assigned])
    absorbed = [row for row in assigned if abs(row["hook"]["planUnabsorbedFirstS"]) <= TIME_TOLERANCE_S]
    return {
        "assigned_share": float(len(assigned) / len(members)),
        "unabsorbed_first_p50_s": _p(first, 50), "unabsorbed_first_p90_s": _p(np.abs(first), 90),
        "unabsorbed_share": float(np.mean(np.abs(first) > TIME_TOLERANCE_S)),
        "unabsorbed_step_p50_s": _p([row["hook"]["planUnabsorbedStepP50S"] for row in assigned], 50),
        "speed_factor_first_p50": _p([row["hook"]["planSpeedFactorFirst"] for row in assigned], 50),
        "stretched_share": float(np.mean([row["hook"]["planStretchM"] > 0.0 for row in assigned])),
        "stretch_dropped_share": float(np.mean([row["hook"]["planStretchDrops"] > 0.0 for row in assigned])),
        # the arrival-time error where the plan said it could meet the time: the closure's own check
        "final_time_error_absorbed_p50_s": _p([row["prediction"]["final_time_error_s"] for row in absorbed], 50)
        if absorbed else float("nan"),
    }


def format_table(summary: dict) -> str:
    strata = [s for s in STRATA if summary[s]["flights"]]
    metrics = [
        ("ade_mean_m", "ADE mean", 0), ("ade_p50_m", "ADE p50", 0), ("ade_p95_m", "ADE p95", 0),
        ("fde_p50_m", "FDE p50", 0), ("chamfer_p50_m", "chamfer p50", 0), ("frechet_p50_m", "Fréchet p50", 0),
        ("abs_final_time_error_p50_s", "|dt| p50 s", 1), ("abs_final_time_error_p80_s", "|dt| p80 s", 1),
        ("final_time_error_mae_s", "dt MAE s", 1), ("route_time_minus_T_p50_s", "route t − T p50 s", 1),
        ("fully_flyable_share", "fully flyable", 3), ("established_share", "established", 3),
        ("lateral_violation_share", "lateral viol.", 3), ("lateral_violation_share_truth", "  … the truth", 3),
        ("glidepath_violation_share", "glidepath viol. (in FAF)", 3),
        ("glidepath_violation_share_truth", "  … the truth", 3),
        ("floor_violation_share", "floor viol. (pre-FAF)", 3), ("floor_violation_share_truth", "  … the truth", 3),
        ("route_shortfall_p50_m", "route shortfall p50", 0), ("route_stretched_share", "route stretched", 3),
        ("route_waypoints_share", "route via waypoints", 3), ("waypoints_mean", "waypoints per plan", 2),
        ("rolled_legs_mean", "legs flown (rolled)", 2), ("instructions_skipped_share", "instruction skipped", 3),
        ("turns_incomplete_share", "turn not completed", 3), ("rolled_capped_share", "rolled flight capped", 3),
        ("eta_head_mae_s", "ETA MAE s (the head's T)", 1),
        ("waypoints_dropped_share", "waypoints dropped", 3),
        ("route_intercept_abs_p50_deg", "route |intercept| p50", 1),
        ("hook_bank_capped_share", "bank capped", 3), ("hook_thrust_saturated_share", "thrust saturated", 3),
        ("hook_thrust_idle_share", "thrust idle", 3),
        ("hook_load_clamped_share", "load clamped", 3), ("hook_route_cross_track_p50_m", "route xt p50", 0),
        ("barrier_gated_share", "barrier gated", 3), ("barrier_clamped_share", "barrier clamped", 3),
        ("capture_height_clamped_share", "capture h clamped", 3),
        ("orders_held_share", "orders held (of steps)", 3), ("order_changes_per_flight", "order changes / flight", 2),
        ("assigned_share", "assigned a time", 3), ("unabsorbed_first_p50_s", "X first ask p50 s", 1),
        ("unabsorbed_first_p90_s", "|X| first ask p90 s", 1), ("unabsorbed_share", "|X| > tol share", 3),
        ("unabsorbed_step_p50_s", "X over asks p50 s", 1),
        ("speed_factor_first_p50", "speed factor p50", 3), ("stretched_share", "path stretched", 3),
        ("stretch_dropped_share", "stretch dropped", 3),
        ("final_time_error_absorbed_p50_s", "dt p50 where absorbed s", 1),
    ]
    lines = ["oracle ceiling (true plans through the guidance), n = " + ", ".join(
        f"{STRATUM_SHORT[s]} {summary[s]['flights']}" for s in strata)]
    lines.append(f"{'metric':<26}" + "".join(f"{STRATUM_SHORT[s]:>14}" for s in strata))
    for key, label, digits in metrics:
        lines.append(f"{label:<26}" + "".join(f"{summary[s][key]:>14.{digits}f}" for s in strata))
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
    parser.add_argument("--policy", choices=(POLICY_TRUTH, POLICY_MODEL), default=POLICY_TRUTH,
                        help="whose plan is flown: the truth's own labels (the ceiling), or the plan "
                             "checkpoint's orders rolled leg by leg (the prediction; needs --route next)")
    parser.add_argument("--lockstep-s", type=float, default=LOCKSTEP_S,
                        help="under --rolling lockstep: the re-ask period in seconds (the step length axis)")
    parser.add_argument("--hold-asks", type=int, default=ORDER_HOLD_ASKS,
                        help="under --rolling lockstep: a material change of order is adopted only once given on this "
                             "many consecutive asks (v5.3; 1 = adopt at once, the v5.1/v5.2 behaviour and the default)")
    parser.add_argument("--hold-flips-only", action="store_true",
                        help="hold only a fix <-> none flip; a moved fix is adopted at once (needs --hold-asks > 1)")
    parser.add_argument("--assign-time", choices=ASSIGNMENTS, default=ASSIGN_NONE,
                        help="v5.4: assign the head's rolled flight an arrival time — the TRUTH's (+ the offset), "
                             "the oracle form that reads the future (needs --policy model --rolling lockstep)")
    parser.add_argument("--assign-time-offset-s", type=float, default=0.0,
                        help="the scheduler's counterfactual: the assigned time is the truth's plus this")
    parser.add_argument("--assign-join", choices=ASSIGNMENTS, default=ASSIGN_NONE,
                        help="v5.4: assign the head's rolled flight the TRUTH's join distance-to-go (where the truth "
                             "has one; needs --policy model --rolling lockstep)")
    parser.add_argument("--rolling", choices=(ROLLING_LEG, ROLLING_LOCKSTEP), default=ROLLING_LOCKSTEP,
                        help="under --route next: one leg per instruction (the head asked at each fix), or the "
                             "receding-horizon lockstep (asked every 30 s, the batch stepped together; v5.1)")
    parser.add_argument("--route", choices=(*ROUTES, ROUTE_NEXT), default=ROUTE_PLAN,
                        help="lay the route from the three route parameters, from those plus the plan's waypoints, "
                             "or fly the waypoints one instruction at a time, re-anchored at each (design v5)")
    parser.add_argument("--max-waypoints", type=int, default=MAX_WAYPOINTS,
                        help="the fixed K of the waypoints representation")
    parser.add_argument("--anchor-km", type=float, default=0.0,
                        help="plan from each flight's own sample nearest this REMAINING PATH (0 = the L-1 anchor; "
                             "the anytime grid's coordinate — later along a vectored track than L-1). The oracle "
                             "needs no lookback, so only the control-inversion rows are required")
    parser.add_argument("--anchor-s", type=float, default=0.0,
                        help="plan from the one row this many seconds after the slice starts, every flight (0 = L-1): "
                             "the anchor a shorter lookback window would give")
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
    if args.anchor_km > 0.0 and args.anchor_s > 0.0:
        parser.error("--anchor-km and --anchor-s are two anchors; give one")
    if args.policy == POLICY_MODEL:
        if args.route != ROUTE_NEXT:
            parser.error("--policy model rolls the head's orders: it needs --route next")
        if arm.config.prediction_output != PREDICTION_PLAN:
            parser.error(f"--policy model needs a plan checkpoint; {label} predicts {arm.config.prediction_output!r}")
    assigning = args.assign_time != ASSIGN_NONE or args.assign_join != ASSIGN_NONE
    if assigning and (args.policy != POLICY_MODEL or args.route != ROUTE_NEXT or args.rolling != ROLLING_LOCKSTEP):
        parser.error("an assignment is flown by the head's lockstep: --policy model --route next --rolling lockstep")
    if args.assign_time_offset_s and args.assign_time == ASSIGN_NONE:
        parser.error("--assign-time-offset-s offsets an assigned time; give --assign-time truth")
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
    elif args.anchor_km > 0.0:
        # the strata stay the L-1 ones (fixed, as the anytime curve fixes them); the plan
        # is read from, and flown from, each flight's own sample nearest the bin
        anchors = {
            item.dataset_id: bin_anchor(
                item, remaining_path_profile_m(item), 1000.0 * args.anchor_km,
                seq_len=ANCHOR_CONTROL_SAMPLES + 1, min_future_s=HORIZON_SLACK_S,
            )
            for item in series
        }
        without = sum(1 for a in anchors.values() if a is None)
        series = [item for item in series if anchors[item.dataset_id] is not None]
        print(f"  anchor at {args.anchor_km:g} km: {len(series)} flights have one, {without} do not", flush=True)
    else:
        anchors = {item.dataset_id: anchor for item in series}
        without = 0
    skeletons: dict[tuple[str, str], object] = {}
    rows: list[dict] = []
    started = time.perf_counter()
    for start in range(0, len(series), args.batch_size):
        chunk = series[start:start + args.batch_size]
        chunk_skeletons = []
        for item in chunk:
            key = (item.airport, str(item.scenario.source.get("runway")))
            if key not in skeletons:
                skeletons[key] = runway_skeleton(item)
            chunk_skeletons.append(skeletons[key])
        chunk_anchors = [int(anchors[item.dataset_id]) for item in chunk]
        labels = [
            extract_plan(item, a, sk, max_waypoints=args.max_waypoints)
            for item, a, sk in zip(chunk, chunk_anchors, chunk_skeletons, strict=True)
        ]
        # the rollout runs PAST the plan's T so a late arrival is measured as late rather
        # than cut off unestablished; the metrics' clock is the truth's regardless
        horizons = [lab.T_s + max(HORIZON_SLACK_S, HORIZON_SLACK_FRACTION * lab.T_s) for lab in labels]
        if args.route == ROUTE_NEXT and args.policy == POLICY_MODEL and args.rolling == ROLLING_LOCKSTEP:
            # the assignment: the truth's arrival time on the series clock (+ the offset) and
            # its join where the track has one (a flight established at the anchor has none)
            assignments = [
                Assignment(
                    arrival_time_s=(float(item.times[a]) + lab.T_s + args.assign_time_offset_s)
                    if args.assign_time == ASSIGN_TRUTH else None,
                    d_join_m=lab.d_join_m if args.assign_join == ASSIGN_TRUTH else None,
                )
                for item, a, lab in zip(chunk, chunk_anchors, labels, strict=True)
            ] if assigning else None
            rolled = rolled_predictions_lockstep(
                arm.model, chunk, arm.config, arm.normalizer, chunk_anchors, torch.device("cpu"), chunk_skeletons,
                step_s=args.lockstep_s, hold_asks=args.hold_asks, hold_flips_only=args.hold_flips_only,
                assignments=assignments,
            )
            forecasts = [r.forecast for r in rolled]
            routes = [r.routes[-1] for r in rolled]
            times = [r.route_time_s for r in rolled]
        elif args.route == ROUTE_NEXT and args.policy == POLICY_MODEL:
            rolled = [
                rolled_prediction(arm.model, item, arm.config, arm.normalizer, a, torch.device("cpu"), sk)
                for item, a, sk in zip(chunk, chunk_anchors, chunk_skeletons, strict=True)
            ]
            forecasts = [r.forecast for r in rolled]
            routes = [r.routes[-1] for r in rolled]
            times = [r.route_time_s for r in rolled]
        elif args.route == ROUTE_NEXT and args.rolling == ROLLING_LOCKSTEP:
            rolled = fly_lockstep_truth(
                chunk, chunk_anchors, labels, chunk_skeletons, arm.config, horizons_s=horizons, step_s=args.lockstep_s,
                hold_asks=args.hold_asks, hold_flips_only=args.hold_flips_only,
            )
            forecasts = [r.forecast for r in rolled]
            routes = [r.routes[-1] for r in rolled]
            times = [r.route_time_s for r in rolled]
        elif args.route == ROUTE_NEXT:
            rolled = [
                fly_rolling(item, a, lab, sk, arm.config, instructions=truth_instructions(lab, sk), horizon_s=h)
                for item, a, lab, sk, h in zip(chunk, chunk_anchors, labels, chunk_skeletons, horizons, strict=True)
            ]
            forecasts = [r.forecast for r in rolled]
            routes = [r.routes[-1] for r in rolled]
            times = [r.route_time_s for r in rolled]
        else:
            rolled = [None] * len(chunk)
            forecasts, routes, times = fly_plans(
                chunk, chunk_anchors, labels, chunk_skeletons, arm.config, durations_s=horizons, route=args.route,
            )
        for item, a, sk, lab, forecast, route, route_time, flight in zip(
            chunk, chunk_anchors, chunk_skeletons, labels, forecasts, routes, times, rolled, strict=True
        ):
            cut = cut_at_threshold_crossing(forecast, item)
            # the arrival time is where the rollout crossed the threshold (the rollout's
            # end where it never did): under the truth's plan its T is the truth, not a
            # prediction; under the head's the flown arrival is what the rolled flight
            # delivers, and the head's own T is kept beside it (`eta_predicted_s`)
            eta_predicted_s = float(forecast.predicted_final_time_s) if args.policy == POLICY_MODEL else None
            cut = replace(cut, predicted_final_time_s=cut.final_time_s)
            metrics = observed_series_metrics(item, cut)
            rows.append({
                "dataset_id": item.dataset_id,
                "flight_id": item.flight_id,
                "difficulty": approach_difficulty(item, anchor).to_dict(),
                "anchor": a,
                "T_s": lab.T_s,
                "labels": lab.to_dict(),
                "route": {
                    "kind": route.kind if flight is None else KIND_ROLLED,
                    "length_m": route.length_m if flight is None else float(sum(flight.leg_lengths_m[:-1]) + route.length_m),
                    "pre_final_m": route.pre_final_m if flight is None else flight.pre_final_m,
                    "requested_pre_final_m": route.requested_pre_final_m if flight is None else float(lab.L_pre_m or 0.0),
                    "shortfall_m": route.shortfall_m if flight is None else float(lab.L_pre_m or 0.0) - flight.pre_final_m,
                    "stretch_offset_m": route.stretch_offset_m,
                    "intercept_deg": math.degrees(route.intercept_rad),
                    "route_time_s": route_time,
                    **({} if flight is None else {
                        "legs": len(flight.routes), "instructions_flown": flight.instructions_flown,
                        "instructions_skipped": flight.instructions_skipped,
                        "leg_kinds": [leg.kind for leg in flight.routes],
                        "turns_incomplete": flight.turns_incomplete, "capped_by": flight.capped_by,
                        "steps": len(flight.orders),
                    }),
                },
                "prediction": {
                    key: metrics[key] for key in (
                        "ade_m", "fde_m", "horizontal_ade_m", "arrival_endpoint_error_m",
                        "final_time_error_s", "true_final_time_s", "coverage_ratio",
                    )
                },
                "geometry": forecast_geometry(item, cut),
                "reference": reference_verdicts(item, cut, sk),
                "truth": corridor_verdicts(item, item.values[a + 1:], a, sk),
                "hook": cut.command_hook_diagnostics,
                "cut_final_time_s": cut.final_time_s,
                "policy": args.policy,
                "eta_predicted_s": eta_predicted_s,
                "orders": None if flight is None else flight.orders,
            })
        print(f"  flown {len(rows)}/{len(series)}", flush=True)
    wall_s = time.perf_counter() - started
    summary = summarize(rows)
    text = format_table(summary)
    text += f"\nflown in {wall_s:.0f} s ({wall_s / max(len(rows), 1):.2f} s per flight)\n"
    print(text, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out / "plan_oracle.json", {
        "schema_version": SCHEMA, "generated_at": utc_now(), "instrument": INSTRUMENT,
        "checkpoint": {"label": label, "path": str(path)},
        "split": args.split, "limit": args.limit, "flights": len(rows), "anchor": anchor,
        "route": args.route, "policy": args.policy, "rolling": args.rolling, "lockstep_s": args.lockstep_s,
        "hold_asks": args.hold_asks, "hold_flips_only": args.hold_flips_only, "wall_s": wall_s,
        "assignment": {"time": args.assign_time, "offset_s": args.assign_time_offset_s, "join": args.assign_join},
        "max_waypoints": args.max_waypoints, "anchor_km": args.anchor_km,
        "anchor_s": args.anchor_s,
        "flights_without_anchor": without,
        "todays_best": TODAYS_BEST, "summary": summary, "rows": rows,
    })
    (args.out / "plan_oracle.txt").write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'plan_oracle.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
