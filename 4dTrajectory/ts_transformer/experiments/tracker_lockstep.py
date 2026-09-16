#!/usr/bin/env python
"""The plan-given tracker in lockstep (two-tier T1, protocol C): a control checkpoint re-asked every Δ.

Two-tier feasibility (`docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §10.3): the control
model is told the plan it flies — the truth's (`plan_conditioning=truth-next`, the oracle) — and the
arrival time (`cta_conditioning=given`), and asked again every ``--step-s`` seconds on its OWN flown
rows. Gate G1 reads this runner's records against the rule guidance flying the same plan
(`plan_oracle --route next --policy truth`, plan-and-guidance design §12.5: vectored 1847 m /
straight-in 283 m at L−1)::

    python run_ts.py tracker_lockstep --checkpoint T1a=<ckpt> --out <dir> --step-s 30 --write-records

Per flight from the fixed anchor ``a0`` (`default_anchor`), one ask at a time for the whole cohort
(ask k is at anchor ``a0 + k·Δ/dt`` for every flight still flying):

* the history is the observed track to ``a0`` continued by the rows flown so far
  (`inference.receding.rolled_series` — the model's ordinary predict path on that series);
* the plan comes from ONE `TruthExpert` per flight, built at ``a0`` and read in time order at each
  ask's pose — at ``a0`` exactly the plan head's label (`targets_from_labels`), afterwards the
  truth's policy at the flown pose (the instruction in force; the truth's time-to-go from its
  nearest row plus the way back to it). The arrival time is the truth duration at ``a0`` (what
  training hands a ``cta=given`` checkpoint) and that expert time-to-go afterwards. The pose's
  heading is the chart velocity's and its speed the extractors' physical ground speed
  (`extractors.ground_speeds`) — the definitions the label reads;
* the forecast is flown for Δ, unless it crosses the threshold ON THE FINAL inside Δ
  (`cut_at_threshold_crossing`: the flight ends there) or ends within Δ plus the checkpoint's
  `random_train_anchor_min_future_s` — the next ask would hand it an arrival time below every
  anchor it trained at, so this forecast is kept whole (cut at its crossing) instead; a flight
  still flying at ``--cap-factor`` × its truth duration from ``a0`` is ended and counted capped.
  An ask whose arrival time is below that floor anyway is counted per flight (``asks_below_floor``).

Variants (``--variants``), each scored per stratum (`strata_fixed_at_anchor` at ``a0``) and, with
``--write-records``, its own predict-shaped record directory under ``records/<label>/<variant>/``:
``receding`` (the plan as the checkpoint is conditioned), ``receding-no-plan`` (the ABSENT token on
the same weights — what the plan buys; refused for a checkpoint without a plan token) and
``one-shot`` (one ask at ``a0``, kept whole to its crossing on the final — what re-asking buys,
over the same span the receding records are cut to). `python -m evaluation`
(flyability, established) and `run_ts.py lead_time_error` read the records unchanged; their
summaries carry a ``lockstep`` block, and the publisher refuses them without a category variant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import (
    CONTROL_HOOK_OFF,
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_OFF,
    DURATION_HEAD_POINT,
    PLAN_CONDITIONING_OFF,
    PREDICTION_CONTROL,
    default_anchor,
)
from ts_transformer.data.anchor_grid import strata_fixed_at_anchor
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import FlightSeries, truth_duration_s
from ts_transformer.experiments.anytime_curve import FORBIDDEN_SPLIT, Arm, Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.support import REPO_ROOT, forecast_geometry
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast, concatenate, cut_at_threshold_crossing
from ts_transformer.inference.receding import ROW_TOLERANCE_S, cut_at_lead, displacement_at, rolled_series
from ts_transformer.io_utils import file_sha256
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import PLAN_TOKEN_KEY, plan_token
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.plan.extractors import PlanLabels, extract_plan, ground_speeds
from ts_transformer.outputs.plan.labels import TruthExpert, targets_at
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton, SkeletonCache

RESULT_SCHEMA = "ts-tracker-lockstep-v1"
#: The block each record directory's summary carries — the publisher's
#: `VARIANT_RECORD_BLOCKS` mirrors this name.
RECORDS_BLOCK = "lockstep"
RECORDS_SCHEMA = "ts-lockstep-records-v1"
RECORDS_DIR = "records"
INSTRUMENT = "the plan-given lockstep"
DEFAULT_STEP_S = 30.0
DEFAULT_CAP_FACTOR = 1.5
VARIANT_RECEDING = "receding"
VARIANT_NO_PLAN = "receding-no-plan"
VARIANT_ONE_SHOT = "one-shot"
VARIANTS = (VARIANT_RECEDING, VARIANT_NO_PLAN, VARIANT_ONE_SHOT)
#: How a flight's lockstep ended.
ENDED_CROSSED = "crossed"          # crossed the threshold on the final inside a step
ENDED_FORECAST = "forecast-end"    # kept whole: the next ask would fall below the training floor
ENDED_CAPPED = "capped"            # still flying at the cap
ENDED_ONE_SHOT = "one-shot"
LEADS_S = (30.0, 60.0, 120.0, 180.0, 300.0)


@dataclass(frozen=True)
class LockstepPlan:
    split: str
    step_s: float
    cap_factor: float
    variants: tuple[str, ...]
    limit: int
    batch_size: int | None
    write_records: bool


@dataclass
class FlightRun:
    """One flight being stepped: its truth expert, the legs flown, how it ended."""

    series: FlightSeries
    labels: PlanLabels
    skeleton: RunwaySkeleton
    expert: TruthExpert
    cap_s: float
    legs: list[Forecast] = field(default_factory=list)
    asks: int = 0
    asks_below_floor: int = 0
    ended: str | None = None
    truncated: bool = False

    @property
    def flown_s(self) -> float:
        return float(sum(np.sum(leg.sample_durations_s) for leg in self.legs))


# ── one ask ────────────────────────────────────────────────────────────────────

def ask_row(run: FlightRun, history: FlightSeries, anchor: int, config, *, first: bool, with_plan: bool) -> dict[str, np.ndarray]:
    """The dynamics row of one ask: the anchor state of ``history`` and, as the checkpoint
    reads them, the truth expert's arrival time and plan at that pose."""
    row = dynamics_arrays(history, anchor)
    values = np.asarray(history.values[anchor], dtype=np.float64)
    e, n = float(values[IDX["e"]]), float(values[IDX["n"]])
    heading = math.atan2(float(values[IDX["ndot"]]), float(values[IDX["edot"]]))
    speed = float(ground_speeds(history, slice(anchor, anchor + 1))[0])
    try:
        instruction, operating = run.expert.order_at(e, n, heading, speed)
    except ValueError as exc:  # name the flight: the expert's own message names only the path
        raise ValueError(f"{run.series.dataset_id} at anchor {anchor}: {exc}") from exc
    if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
        # at a0 the CTA training hands the checkpoint (the truth duration); afterwards the
        # expert's time-to-go at the flown pose — at a0 the two differ by the expert's
        # nearest-row reading (0.24 s on the synthetic fixture)
        arrival_s = truth_duration_s(run.series, anchor) if first else float(operating.T_s)
        run.asks_below_floor += int(arrival_s < config.random_train_anchor_min_future_s)
        row["cta_s"] = np.array(arrival_s, dtype=np.float64)
    if config.plan_conditioning != PLAN_CONDITIONING_OFF:
        targets = targets_at(operating, instruction, e, n, run.skeleton) if with_plan else None
        row[PLAN_TOKEN_KEY] = plan_token(targets)
    return row


def fly_variant(arm: Arm, series: list[FlightSeries], variant: str, plan: LockstepPlan, device: torch.device,
                batch_size: int) -> list[FlightRun]:
    config = arm.config
    a0 = default_anchor(config)
    step_rows = int(round(plan.step_s / config.dt_s))
    skeletons = SkeletonCache()
    runs = []
    for item in series:
        skeleton = skeletons.for_series(item)
        labels = extract_plan(item, a0, skeleton)
        runs.append(FlightRun(
            series=item, labels=labels, skeleton=skeleton, expert=TruthExpert(labels, item, a0, skeleton),
            cap_s=plan.cap_factor * truth_duration_s(item, a0),
        ))
    ask = 0
    while True:
        active = [run for run in runs if run.ended is None]
        if not active:
            break
        anchor = a0 + ask * step_rows
        histories = [
            run.series if ask == 0 else rolled_series(run.series, a0, concatenate(run.legs, a0, 0.0), anchor, config.dt_s)
            for run in active
        ]
        rows = [
            ask_row(run, history, anchor, config, first=ask == 0, with_plan=variant != VARIANT_NO_PLAN)
            for run, history in zip(active, histories, strict=True)
        ]
        forecasts: list[Forecast] = []
        for start in range(0, len(active), batch_size):
            chunk = rows[start : start + batch_size]
            dynamics = {name: torch.from_numpy(np.stack([row[name] for row in chunk])).to(device) for name in chunk[0]}
            forecasts.extend(forecast_control_batch(
                arm.model, histories[start : start + batch_size], config, arm.normalizer, anchor, device,
                dynamics=dynamics,
            ))
        # an ask whose forecast leaves less than this after the step is the last: the next
        # would hand the checkpoint an arrival time below every anchor it trained at
        last_ask_s = plan.step_s + config.random_train_anchor_min_future_s + ROW_TOLERANCE_S
        for run, history, forecast in zip(active, histories, forecasts, strict=True):
            run.asks += 1
            crossed = cut_at_threshold_crossing(forecast, history)
            if variant == VARIANT_ONE_SHOT:
                run.legs.append(crossed)
                run.ended, run.truncated = ENDED_ONE_SHOT, crossed.truncated_at_threshold
                continue
            if crossed.truncated_at_threshold and crossed.final_time_s <= plan.step_s + ROW_TOLERANCE_S:
                run.legs.append(crossed)
                run.ended, run.truncated = ENDED_CROSSED, True
            elif forecast.final_time_s <= last_ask_s:
                run.legs.append(crossed)
                run.ended, run.truncated = ENDED_FORECAST, crossed.truncated_at_threshold
            else:
                run.legs.append(cut_at_lead(forecast, plan.step_s))
                if run.flown_s >= run.cap_s - ROW_TOLERANCE_S:
                    run.ended = ENDED_CAPPED
        print(f"    {variant} ask {ask}: {len(active)} flights at anchor {anchor}, "
              f"{sum(1 for run in runs if run.ended is None)} continue", flush=True)
        ask += 1
    return runs


def whole_forecast(run: FlightRun, a0: int) -> Forecast:
    """The flight's legs as one forecast from ``a0``: its predicted end is the last ask's."""
    last = run.legs[-1]
    before = float(sum(np.sum(leg.sample_durations_s) for leg in run.legs[:-1]))
    whole = concatenate(run.legs, a0, before + float(last.predicted_final_time_s))
    return replace(whole, truncated_at_threshold=run.truncated, horizon_capped=run.ended == ENDED_CAPPED)


# ── the readout ────────────────────────────────────────────────────────────────

def flight_row(run: FlightRun, forecast: Forecast, a0: int, points: int) -> tuple[dict, dict]:
    metrics = observed_series_metrics(run.series, forecast, points=points)
    geometry = forecast_geometry(run.series, forecast)
    origin = float(run.series.times[a0])
    row = {
        "asks": run.asks, "asks_below_floor": run.asks_below_floor, "ended": run.ended,
        "truncated_at_threshold": run.truncated,
        "ade_m": float(metrics["ade_m"]), "fde_m": float(metrics["fde_m"]),
        "final_time_error_s": float(metrics["final_time_error_s"]),
        "chamfer_m": float(geometry["chamfer_m"]), "frechet_m": float(geometry["frechet_m"]),
        "at": {f"{lead:g}": displacement_at(run.series, forecast, a0, origin + lead) for lead in LEADS_S},
    }
    return row, metrics


def _p50(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def stratum_block(rows: dict[str, dict], keys: list[str]) -> dict:
    cells = [rows[key] for key in keys]
    return {
        "n": len(cells),
        "ade_mean_m": float(np.mean([c["ade_m"] for c in cells])) if cells else None,
        "ade_p50_m": _p50([c["ade_m"] for c in cells]),
        "fde_p50_m": _p50([c["fde_m"] for c in cells]),
        "chamfer_p50_m": _p50([c["chamfer_m"] for c in cells]),
        "frechet_p50_m": _p50([c["frechet_m"] for c in cells]),
        "abs_dt_p50_s": _p50([abs(c["final_time_error_s"]) for c in cells]),
        "asks_mean": float(np.mean([c["asks"] for c in cells])) if cells else None,
        "flights_with_an_ask_below_floor": sum(1 for c in cells if c["asks_below_floor"]),
        "truncated_at_threshold": sum(1 for c in cells if c["truncated_at_threshold"]),
        "ended": {name: sum(1 for c in cells if c["ended"] == name) for name in sorted({c["ended"] for c in cells})},
        "at_lead_p50_m": {
            lead: {"n": len(v), "p50": _p50(v)}
            for lead in (f"{h:g}" for h in LEADS_S)
            for v in [[c["at"][lead] for c in cells if c["at"][lead] is not None]]
        },
    }


def _fmt(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    plan = payload["plan"]
    lines = [
        f"Plan-given lockstep (protocol C: the truth's plan and arrival time READ): re-asked every {plan['step_s']:g} s, "
        f"cap {plan['cap_factor']:g}× truth duration, split {plan['split']}"
        + (f", limit {plan['limit']}" if plan["limit"] else ""),
        "ADE/FDE on the whole record (export's accounting), every variant cut at its crossing on the final; "
        "chamfer / Fréchet time-free; disp p50 at leads from a0; below-floor = flights asked with an arrival "
        "time under the checkpoint's training future floor.",
    ]
    for label, arm in payload["checkpoints"].items():
        lines.append("")
        lines.append(f"── {label} — {arm['checkpoint']} ({arm['flights']} flights, a0={arm['anchor']})")
        for variant, block in arm["variants"].items():
            for stratum, cell in block["strata"].items():
                leads = " ".join(f"{k}s {_fmt(v['p50'])}" for k, v in cell["at_lead_p50_m"].items())
                lines.append(
                    f"   {variant:<17s} {stratum[:34]:<34s} n={cell['n']:>4d} ADE {_fmt(cell['ade_mean_m']):>5}/"
                    f"{_fmt(cell['ade_p50_m']):>5} FDE50 {_fmt(cell['fde_p50_m']):>5} cham50 {_fmt(cell['chamfer_p50_m']):>5} "
                    f"Fr50 {_fmt(cell['frechet_p50_m']):>5} |dt|50 {_fmt(cell['abs_dt_p50_s'], 1):>5} asks {_fmt(cell['asks_mean'], 1)} "
                    f"ended {cell['ended']} cut {cell['truncated_at_threshold']} below-floor {cell['flights_with_an_ask_below_floor']} | {leads}"
                )
    return "\n".join(lines) + "\n"


def check_arm(arm: Arm, plan: LockstepPlan) -> None:
    config = arm.config
    if config.prediction_output != PREDICTION_CONTROL:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} flies a CONTROL checkpoint, "
                         f"this one predicts {config.prediction_output!r}")
    if config.control_command_hook != CONTROL_HOOK_OFF:
        raise SystemExit(f"{arm.label} ({arm.path}): a hooked checkpoint rewrites the schedule; "
                         f"{INSTRUMENT} flies the network's own")
    if config.latent_dim or config.duration_head != DURATION_HEAD_POINT:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} is defined on the deterministic point head")
    if config.cta_conditioning not in (CTA_CONDITIONING_OFF, CTA_CONDITIONING_GIVEN):
        raise SystemExit(f"{arm.label} ({arm.path}): cta_conditioning={config.cta_conditioning!r} names no "
                         f"arrival time {INSTRUMENT} can hand it")
    if VARIANT_NO_PLAN in plan.variants and config.plan_conditioning == PLAN_CONDITIONING_OFF:
        raise SystemExit(f"{arm.label} ({arm.path}): {VARIANT_NO_PLAN!r} drops a plan token this checkpoint "
                         "does not read")
    for name, unit in (("dt_s", config.dt_s), ("control_rollout_integrator_dt_s", config.control_rollout_integrator_dt_s)):
        steps = plan.step_s / unit
        if abs(steps - round(steps)) > 1e-9:
            raise SystemExit(f"{arm.label}: --step-s {plan.step_s:g} is not a whole multiple of {name} {unit:g}")


def measure_arm(arm: Arm, series: list[FlightSeries], plan: LockstepPlan, device: torch.device,
                records_root: Path | None, campaign: str) -> dict:
    started = time.time()
    config = arm.config
    a0 = default_anchor(config)
    batch_size = plan.batch_size or config.batch_size
    keys = [item.dataset_id for item in series]
    masks = strata_fixed_at_anchor(series, keys, anchor=a0)
    print(f"{arm.label}: {len(series)} flights, a0 {a0}, every {plan.step_s:g} s, variants {', '.join(plan.variants)}", flush=True)
    variants = {}
    record_dirs = {}
    for variant in plan.variants:
        runs = fly_variant(arm, series, variant, plan, device, batch_size)
        rows: dict[str, dict] = {}
        records, metrics = [], []
        for index, run in enumerate(runs):
            forecast = whole_forecast(run, a0)
            row, flight_metrics = flight_row(run, forecast, a0, config.validation_common_grid_points)
            rows[run.series.dataset_id] = row
            if records_root is not None:
                records.append(build_prediction_record(
                    run.series, forecast, index=index, model_name=config.model, horizon_mode=config.horizon_mode,
                    split=plan.split,
                ))
                metrics.append(flight_metrics)
        strata = {
            stratum: stratum_block(rows, [key for key, keep in zip(keys, masks[stratum], strict=True) if keep])
            for stratum in (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
        }
        variants[variant] = {"strata": strata, "flights": rows}
        if records_root is not None:
            directory = records_root / arm.label / variant
            write_batch(
                records, output_dir=directory, config_dict=config.to_dict(), flight_metrics=metrics,
                checkpoint=str(arm.path), split=plan.split,
                extra_summary={RECORDS_BLOCK: {
                    "schema": RECORDS_SCHEMA, "campaign": campaign, "label": arm.label, "variant": variant,
                    "step_s": plan.step_s, "cap_factor": plan.cap_factor, "anchor": a0,
                    "plan_conditioning": config.plan_conditioning, "cta_conditioning": config.cta_conditioning,
                    "reads_the_future": "the truth's plan and arrival time at every ask (protocol C)",
                    "split": plan.split, "limit": plan.limit,
                    "split_flights": len(arm.payload["split"][plan.split]), "records": len(records),
                }},
            )
            record_dirs[variant] = str(directory.relative_to(records_root.parent))
        vectored = strata[STRATUM_VECTORED]
        print(f"  {variant}: vectored ADE mean {_fmt(vectored['ade_mean_m'])} m (n {vectored['n']}), "
              f"straight-in {_fmt(strata[STRATUM_STRAIGHT_IN]['ade_mean_m'])} m", flush=True)
    return {
        "checkpoint": str(arm.path), "checkpoint_sha256": file_sha256(arm.path), "anchor": a0,
        "flights": len(series), "split_flights": len(arm.payload["split"][plan.split]),
        "variants": variants, "record_dirs": record_dirs, "seconds": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--split", choices=("val", "train", FORBIDDEN_SPLIT), default="val")
    parser.add_argument("--step-s", type=float, default=DEFAULT_STEP_S,
                        help=f"the re-ask period (default {DEFAULT_STEP_S:g}, the plan lockstep's)")
    parser.add_argument("--cap-factor", type=float, default=DEFAULT_CAP_FACTOR,
                        help=f"end a flight still flying at this × its truth duration (default {DEFAULT_CAP_FACTOR:g})")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"comma-separated subset of {VARIANTS} (default: all)")
    parser.add_argument("--limit", type=int, default=0, help="first N flights of the split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--write-records", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def parse_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> LockstepPlan:
    if args.split == FORBIDDEN_SPLIT:
        parser.error(f"the {FORBIDDEN_SPLIT} split is sealed")
    if not math.isfinite(args.step_s) or args.step_s <= 0.0:
        parser.error("--step-s must be positive")
    if not math.isfinite(args.cap_factor) or args.cap_factor < 1.0:
        parser.error("--cap-factor must be at least 1 (the truth's own duration)")
    variants = tuple(token.strip() for token in args.variants.split(",") if token.strip())
    unknown = [name for name in variants if name not in VARIANTS]
    if not variants or unknown or len(set(variants)) != len(variants):
        parser.error(f"--variants takes a subset of {VARIANTS} without repeats, got {args.variants!r}")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return LockstepPlan(split=args.split, step_s=float(args.step_s), cap_factor=float(args.cap_factor),
                        variants=variants, limit=int(args.limit), batch_size=args.batch_size,
                        write_records=bool(args.write_records))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    plan = parse_plan(parser, args)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the lockstep readout is an immutable artifact")
    device = resolve_device(args.device)
    grid = Grid(split=plan.split, bins_m=(), min_future_s=0.0, batch_size=plan.batch_size, limit=plan.limit)
    loaded = [
        load_arm(label, path, grid, device, instrument=INSTRUMENT, refuse_cta_given=False, refuse_plan=False)
        for label, path in arms.items()
    ]
    for arm in loaded:
        check_arm(arm, plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    try:
        records_root = staged / RECORDS_DIR if plan.write_records else None
        payload = {
            "schema": RESULT_SCHEMA,
            "plan": {"split": plan.split, "step_s": plan.step_s, "cap_factor": plan.cap_factor,
                     "variants": list(plan.variants), "limit": plan.limit, "write_records": plan.write_records},
            "reads_the_future": "the truth's plan (TruthExpert) and arrival time at every ask: protocol C, an oracle",
            "strata_anchor": "a0 = default_anchor (strata_fixed_at_anchor)",
            "leads_s": list(LEADS_S),
            "device": str(device),
            "checkpoints": {
                arm.label: measure_arm(arm, cohort_series(arm, grid), plan, device, records_root, out.name)
                for arm in loaded
            },
        }
        (staged / "tracker_lockstep.json").write_text(json.dumps(payload, indent=1))
        text = render(payload)
        (staged / "tracker_lockstep.txt").write_text(text)
        staged.chmod(0o755)
        staged.rename(out)
    except BaseException:
        print(f"\nremoving the staged artifact {staged} — the run did not complete", file=sys.stderr, flush=True)
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print()
    print(text, end="")
    print(f"wrote {out / 'tracker_lockstep.txt'} and {out / 'tracker_lockstep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
