#!/usr/bin/env python
"""The short-horizon readout (two-tier v2 §3, evaluation 1): a control checkpoint's error inside its
first Δ seconds, from REAL histories re-anchored along the approach, with and without its plan.

`docs/2026-09-17_two_tier_plan_v2.zh.md` §3. A fixed-horizon control head (`control_horizon_s`) told
the truth's coarse waypoints is flying the truth's plan — its error inside Δ is e_track (protocol
C, an oracle). The SAME weights with the token absent read the history alone — protocol B, the
number a whole-approach checkpoint (native32) is compared on. This runner reads both off one
checkpoint and pairs them, flight by flight and anchor by anchor, against a reference::

    python run_ts.py short_horizon_readout --checkpoint L1_pa=<ckpt> [--checkpoint ...] \\
        --reference native32=<ckpt> --out <dir> [--bins-km 12,8,6] [--horizon-s 60] [--leads-s 30,60] \\
        [--split val] [--limit N] [--batch-size B] [--write-records]

**Anchor sets.** Each arm's own fixed anchor (`default_anchor`: L−1, or the run's `anchor_floor_index`)
and the remaining-path bins of `data/anchor_grid.py` (`anchors_for_bin`, the closest admissible
sample, never before the fixed anchor, with at least Δ of truth after it — the horizon's own future
floor, so every reading spans [0, Δ]). The arms of one artifact must share their fixed anchor, so
a flight's bin anchor is the same sample under every arm and the pairing is anchor for anchor.
**Strata are fixed at the fixed anchor** (`strata_fixed_at_anchor`, the anytime curve's rule).

**Variants.** A plan-reading checkpoint is read twice: ``truth-plan`` (the truth's token at the
anchor, exactly the training row's — reads the future) and ``no-plan`` (the ABSENT token). A
plan-free checkpoint is read once, as ``no-plan``. A whole-approach forecast is CUT at Δ
(`cut_at_lead`); one that ends before Δ (a predicted duration under the horizon) is ABSENT at
that anchor and counted, never held or scored short. **So is a flight whose OBSERVED track ends
before Δ after the anchor** (``truth_shorter_than_horizon``): the anchor sets admit on the
supervision rows — the track closed to the threshold, `truth_duration_s` — while the readings
are against the observed rows (`mean_displacement_to`, `lead_time_error`'s accounting), and the
two ends differ by the fitted tail (KRDU at the fixed anchor 59: 3 of native32's 1404 val flights).

**Numbers**, per arm · variant · anchor set · stratum: the displacement at each lead (p50 / mean)
and the mean displacement over [0, Δ] on the 1 s grid (`mean_displacement_to`, `lead_time_error`'s
accounting), over the flights present; then PAIRED over the (flight, anchor set) pairs both sides
hold — every arm's ``no-plan`` against the reference's (what the short head knows from the
history that the whole-approach head does not), and ``truth-plan`` against ``no-plan`` (what the
plan buys). The per-flight rows stay in the artifact.

``--write-records`` writes every scored forecast as a predict-shaped record directory under
``records/<arm>/<variant>/<anchor set>/`` (the shape `predict` writes, so the publisher takes it),
with a ``short_horizon`` summary block saying which subset and which protocol it is.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from ts_transformer.config import PLAN_CONDITIONING_OFF, PREDICTION_CONTROL, default_anchor
from ts_transformer.data.anchor_grid import (
    anchors_for_bin,
    bin_label,
    difficulty_at_anchor,
    remaining_path_profiles,
)
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.experiments.anytime_curve import FORBIDDEN_SPLIT, Arm, Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast
from ts_transformer.inference.receding import ROW_TOLERANCE_S, cut_at_lead, displacement_at, mean_displacement_to
from ts_transformer.io_utils import file_sha256
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import PLAN_TOKEN_KEY, plan_token_width, training_plan_token
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.plan.skeleton import SkeletonCache

RESULT_SCHEMA = "ts-short-horizon-readout-v1"
RECORDS_BLOCK = "short_horizon"
RECORDS_SCHEMA = "ts-short-horizon-records-v1"
RECORDS_DIR = "records"
INSTRUMENT = "the short-horizon readout"
DEFAULT_HORIZON_S = 60.0
DEFAULT_LEADS_S = (30.0, 60.0)
DEFAULT_BINS_KM = (12.0, 8.0, 6.0)
#: The fixed anchor's set, beside the bins (`bin_label` names those).
FIXED_SET = "fixed"
VARIANT_PLAN = "truth-plan"
VARIANT_NO_PLAN = "no-plan"
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
READS_THE_FUTURE = {
    VARIANT_PLAN: "the truth's plan token at the anchor (protocol C: the tracker under the truth's plan, e_track)",
    VARIANT_NO_PLAN: "nothing but the landed runway (protocol B: the history alone)",
}


@dataclass(frozen=True)
class ReadoutPlan:
    split: str
    horizon_s: float
    leads_s: tuple[float, ...]
    bins_m: tuple[float, ...]
    limit: int
    batch_size: int | None
    write_records: bool


def variants_of(arm: Arm) -> tuple[str, ...]:
    return (VARIANT_PLAN, VARIANT_NO_PLAN) if arm.config.plan_conditioning != PLAN_CONDITIONING_OFF else (VARIANT_NO_PLAN,)


def check_arm(arm: Arm, plan: ReadoutPlan, *, reference: bool) -> None:
    config = arm.config
    if config.prediction_output != PREDICTION_CONTROL:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} reads a CONTROL checkpoint, this one predicts "
                         f"{config.prediction_output!r}")
    if config.latent_dim:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} is defined on the deterministic decode")
    if reference and config.plan_conditioning != PLAN_CONDITIONING_OFF:
        raise SystemExit(f"--reference {arm.label} ({arm.path}) reads a plan token; the reference is the plan-free "
                         "whole-approach head every no-plan reading is paired against")
    if config.control_horizon_s and abs(config.control_horizon_s - plan.horizon_s) > ROW_TOLERANCE_S:
        raise SystemExit(f"{arm.label} ({arm.path}): a {config.control_horizon_s:g} s fixed horizon against a "
                         f"--horizon-s {plan.horizon_s:g} readout; the readout spans the head's own horizon")
    for lead in plan.leads_s:
        steps = lead / config.control_rollout_integrator_dt_s
        if abs(steps - round(steps)) > 1e-9:
            raise SystemExit(f"{arm.label}: lead {lead:g} s is not on the {config.control_rollout_integrator_dt_s:g} s "
                             "rollout grid")


def check_arms_share_the_anchor(arms: list[Arm]) -> None:
    anchors = {arm.label: default_anchor(arm.config) for arm in arms}
    if len(set(anchors.values())) != 1:
        raise SystemExit(f"the arms' fixed anchors differ {anchors}; the pairing is anchor for anchor, so a "
                         "shorter lookback needs the same anchor_floor_index as the reference")


def ask_rows(arm: Arm, batch: list[FlightSeries], anchor: int, variant: str, skeletons: SkeletonCache,
             device: torch.device) -> dict[str, torch.Tensor]:
    """The dynamics batch of one anchor group: the anchor's own physical context and, on a
    plan-reading checkpoint, the truth's token (`truth-plan`) or the absent one (`no-plan`)."""
    config = arm.config
    rows = [
        dynamics_arrays(item, anchor, parameterization=config.control_thrust_parameterization,
                        condition_features=config.control_condition_features)
        for item in batch
    ]
    if config.plan_conditioning != PLAN_CONDITIONING_OFF:
        for row, item in zip(rows, batch, strict=True):
            row[PLAN_TOKEN_KEY] = (
                training_plan_token(item, anchor, config, skeletons) if variant == VARIANT_PLAN
                else np.zeros(plan_token_width(config), dtype=np.float32)
            )
    return {name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device) for name in rows[0]}


def within_horizon(forecast: Forecast, horizon_s: float) -> Forecast | None:
    """The forecast's first ``horizon_s``, or None when it ends before them (a predicted duration
    under the horizon): absent, never held or scored short."""
    if forecast.final_time_s < horizon_s - ROW_TOLERANCE_S:
        return None
    return forecast if forecast.final_time_s < horizon_s + ROW_TOLERANCE_S else cut_at_lead(forecast, horizon_s)


def anchor_sets(arm: Arm, series: list[FlightSeries], profiles, plan: ReadoutPlan) -> dict[str, dict[int, int]]:
    """``{set label: {flight index: anchor}}`` — the fixed anchor for every flight the horizon fits
    after (the cohort's admission rule, `window_anchors`, guarantees it), then each bin."""
    config = arm.config
    fixed = default_anchor(config)
    sets = {FIXED_SET: {index: fixed for index in range(len(series))}}
    for target_m in plan.bins_m:
        sets[bin_label(target_m)] = anchors_for_bin(
            series, profiles, target_m, seq_len=config.seq_len, min_future_s=plan.horizon_s,
            minimum_anchor_index=fixed,
        )
    return sets


def observed_reaches(item: FlightSeries, anchor: int, horizon_s: float) -> bool:
    """Whether the OBSERVED track (what the readings are taken against) reaches ``horizon_s``
    after ``anchor``; the anchor sets admit on the supervision rows, which run on to the threshold."""
    return float(item.times[-1] - item.times[anchor]) >= horizon_s - ROW_TOLERANCE_S


def measure_variant(arm: Arm, series: list[FlightSeries], anchors: dict[int, int], variant: str, plan: ReadoutPlan,
                    device: torch.device, batch_size: int, skeletons: SkeletonCache, *, build_records: bool,
                    ) -> tuple[dict[str, dict], list[tuple[int, object, dict]], int, int]:
    """Every flight of one anchor set, forecast from its anchor and scored inside the horizon.
    Returns the per-flight rows by key, the record pairs (index, record, metrics), the count of
    forecasts that ended before the horizon (absent) and the count of flights whose observed
    track does (absent too, never scored against a truth that is not there)."""
    config = arm.config
    groups: dict[int, list[int]] = {}
    truth_short = 0
    for index, anchor in anchors.items():
        if not observed_reaches(series[index], anchor, plan.horizon_s):
            truth_short += 1
            continue
        groups.setdefault(anchor, []).append(index)
    rows: dict[str, dict] = {}
    pairs: list[tuple[int, object, dict]] = []
    short = 0
    for anchor in sorted(groups):
        members = groups[anchor]
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            batch = [series[index] for index in chunk]
            forecasts = forecast_control_batch(
                arm.model, batch, config, arm.normalizer, anchor, device,
                dynamics=ask_rows(arm, batch, anchor, variant, skeletons, device),
            )
            for index, item, forecast in zip(chunk, batch, forecasts, strict=True):
                inside = within_horizon(forecast, plan.horizon_s)
                if inside is None:
                    short += 1
                    continue
                origin = float(item.times[anchor])
                rows[item.dataset_id] = {
                    "anchor_index": anchor,
                    "ade_m": mean_displacement_to(item, inside, anchor, plan.horizon_s),
                    "at": {f"{lead:g}": displacement_at(item, inside, anchor, origin + lead) for lead in plan.leads_s},
                    "predicted_final_time_s": float(forecast.predicted_final_time_s),
                }
                if build_records:
                    pairs.append((index, build_prediction_record(
                        item, inside, index=index, model_name=config.model, horizon_mode=config.horizon_mode,
                        split=plan.split,
                    ), observed_series_metrics(item, inside, points=config.validation_common_grid_points)))
    return rows, pairs, short, truth_short


def _p50(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def stratum_cells(rows: dict[str, dict], masks: dict[str, np.ndarray], keys: list[str], leads: tuple[float, ...]) -> dict:
    out = {}
    for stratum in STRATA:
        members = [key for key, keep in zip(keys, masks[stratum], strict=True) if keep and key in rows]
        cells = [rows[key] for key in members]
        out[stratum] = {
            "n": len(cells),
            "ade_mean_m": _mean([c["ade_m"] for c in cells]),
            "ade_p50_m": _p50([c["ade_m"] for c in cells]),
            "at_lead_m": {
                f"{lead:g}": {"n": len(v), "p50": _p50(v), "mean": _mean(v)}
                for lead in leads
                for v in [[c["at"][f"{lead:g}"] for c in cells if c["at"][f"{lead:g}"] is not None]]
            },
        }
    return out


def paired(arm_rows: dict[str, dict], base_rows: dict[str, dict], masks: dict[str, np.ndarray], keys: list[str]) -> dict:
    """ΔADE = arm − base per flight over the flights BOTH hold at this anchor set, per stratum."""
    common = [key for key, keep in zip(keys, masks[STRATUM_ALL], strict=True) if keep and key in arm_rows and key in base_rows]
    out = {}
    for stratum in STRATA:
        inside = {key for key, keep in zip(keys, masks[stratum], strict=True) if keep}
        delta = np.array([arm_rows[key]["ade_m"] - base_rows[key]["ade_m"] for key in common if key in inside])
        out[stratum] = {
            "n": int(delta.size),
            "delta_mean_m": float(delta.mean()) if delta.size else None,
            "delta_p50_m": float(np.median(delta)) if delta.size else None,
            "arm_better_share": float(np.mean(delta < 0.0)) if delta.size else None,
        }
    return out


def records_block(arm: Arm, variant: str, set_label: str, plan: ReadoutPlan, *, campaign: str, measured: int,
                  records: int, short: int, truth_short: int) -> dict:
    return {
        "schema": RECORDS_SCHEMA, "campaign": campaign, "label": arm.label, "variant": variant,
        "anchor_set": set_label, "horizon_s": plan.horizon_s, "leads_s": list(plan.leads_s),
        "fixed_anchor": default_anchor(arm.config),
        "plan_conditioning": arm.config.plan_conditioning, "control_horizon_s": arm.config.control_horizon_s,
        "reads_the_future": READS_THE_FUTURE[variant],
        "split": plan.split, "limit": plan.limit, "split_flights": len(arm.payload["split"][plan.split]),
        "measured_flights": measured, "records": records, "forecasts_shorter_than_horizon": short,
        "truth_shorter_than_horizon": truth_short,
    }


def measure_arm(arm: Arm, series: list[FlightSeries], plan: ReadoutPlan, device: torch.device,
                records_root: Path | None, campaign: str) -> dict:
    started = time.time()
    config = arm.config
    fixed = default_anchor(config)
    keys = [item.dataset_id for item in series]
    difficulty = difficulty_at_anchor(series, keys, anchor=fixed)
    masks = strata_masks(difficulty, keys)
    profiles = remaining_path_profiles(series)
    sets = anchor_sets(arm, series, profiles, plan)
    batch_size = plan.batch_size or config.batch_size
    skeletons = SkeletonCache()
    print(f"{arm.label}: {len(series)} flights, fixed anchor {fixed}, horizon {plan.horizon_s:g} s, "
          f"variants {', '.join(variants_of(arm))}, sets {', '.join(sets)}", flush=True)
    variants: dict[str, dict] = {}
    record_dirs: dict[str, dict[str, str]] = {}
    for variant in variants_of(arm):
        variants[variant] = {"sets": {}}
        for set_label, anchors in sets.items():
            rows, pairs, short, truth_short = measure_variant(
                arm, series, anchors, variant, plan, device, batch_size, skeletons,
                build_records=records_root is not None,
            )
            variants[variant]["sets"][set_label] = {
                "anchored_flights": len(anchors), "forecasts_shorter_than_horizon": short,
                "truth_shorter_than_horizon": truth_short,
                "strata": stratum_cells(rows, masks, keys, plan.leads_s), "flights": rows,
            }
            if records_root is not None and pairs:
                ordered = sorted(pairs, key=lambda pair: pair[0])
                directory = records_root / arm.label / variant / set_label
                write_batch(
                    [record for _, record, _ in ordered], output_dir=directory, config_dict=config.to_dict(),
                    flight_metrics=[metrics for _, _, metrics in ordered], checkpoint=str(arm.path), split=plan.split,
                    extra_summary={RECORDS_BLOCK: records_block(
                        arm, variant, set_label, plan, campaign=campaign, measured=len(series),
                        records=len(ordered), short=short, truth_short=truth_short,
                    )},
                )
                record_dirs.setdefault(variant, {})[set_label] = str(directory.relative_to(records_root.parent))
            cell = variants[variant]["sets"][set_label]["strata"]
            print(f"  {variant:<10s} {set_label:<6s} n={cell[STRATUM_ALL]['n']:>4d} ADE[0,{plan.horizon_s:g}] mean "
                  f"{_fmt(cell[STRATUM_ALL]['ade_mean_m'])} | vectored {_fmt(cell[STRATUM_VECTORED]['ade_mean_m'])} "
                  f"(n {cell[STRATUM_VECTORED]['n']}) | short {short} | truth short {truth_short}", flush=True)
    return {
        "checkpoint": str(arm.path), "checkpoint_sha256": file_sha256(arm.path), "fixed_anchor": fixed,
        "plan_conditioning": config.plan_conditioning, "control_horizon_s": config.control_horizon_s,
        "seq_len": config.seq_len, "flights": len(series), "split_flights": len(arm.payload["split"][plan.split]),
        "masks": {stratum: [key for key, keep in zip(keys, mask, strict=True) if keep] for stratum, mask in masks.items()},
        "variants": variants, "record_dirs": record_dirs, "seconds": time.time() - started,
    }


def pair_arms(arms: dict[str, dict], reference: str) -> dict:
    """Every arm's ``no-plan`` against the reference's, and its ``truth-plan`` against its own
    ``no-plan``, per anchor set and stratum, over the flights both hold there."""
    base = arms[reference]
    out: dict[str, dict] = {}
    for label, arm in arms.items():
        keys = sorted(arm["masks"][STRATUM_ALL])
        masks = {stratum: np.array([key in set(arm["masks"][stratum]) for key in keys]) for stratum in STRATA}
        readings: dict[str, dict] = {}
        if label != reference:
            readings[f"{VARIANT_NO_PLAN} − reference {VARIANT_NO_PLAN}"] = {
                set_label: paired(arm["variants"][VARIANT_NO_PLAN]["sets"][set_label]["flights"],
                                  base["variants"][VARIANT_NO_PLAN]["sets"][set_label]["flights"], masks, keys)
                for set_label in arm["variants"][VARIANT_NO_PLAN]["sets"]
            }
        if VARIANT_PLAN in arm["variants"]:
            readings[f"{VARIANT_PLAN} − {VARIANT_NO_PLAN}"] = {
                set_label: paired(arm["variants"][VARIANT_PLAN]["sets"][set_label]["flights"],
                                  arm["variants"][VARIANT_NO_PLAN]["sets"][set_label]["flights"], masks, keys)
                for set_label in arm["variants"][VARIANT_PLAN]["sets"]
            }
        out[label] = readings
    return out


def _fmt(value, digits: int = 0) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    plan = payload["plan"]
    lines = [
        f"Short-horizon readout: ADE[0,{plan['horizon_s']:g}] and the displacement at {', '.join(f'{h:g}' for h in plan['leads_s'])} s, "
        f"from the fixed anchor and the remaining-path bins {', '.join(plan['bins'])} (≥ {plan['horizon_s']:g} s of truth after each), "
        f"split {plan['split']}" + (f", limit {plan['limit']}" if plan["limit"] else "") + f"; reference {payload['reference']}.",
        f"{VARIANT_PLAN} = the truth's token at the anchor (reads the future, e_track); {VARIANT_NO_PLAN} = the absent token "
        "(the history alone). A whole-approach forecast is cut at the horizon; one shorter than it is absent, never scored.",
    ]
    for label, arm in payload["arms"].items():
        lines.append("")
        lines.append(f"── {label} — {arm['checkpoint']} (plan={arm['plan_conditioning']}, horizon={arm['control_horizon_s']:g}, "
                     f"L={arm['seq_len']}, fixed anchor {arm['fixed_anchor']}, {arm['flights']} flights)")
        for variant, block in arm["variants"].items():
            for set_label, cell in block["sets"].items():
                for stratum in STRATA:
                    s = cell["strata"][stratum]
                    leads = " ".join(f"e({k}) {_fmt(v['p50'])}/{_fmt(v['mean'])}" for k, v in s["at_lead_m"].items())
                    lines.append(f"   {variant:<10s} {set_label:<6s} {stratum[:30]:<30s} n={s['n']:>4d} "
                                 f"ADE {_fmt(s['ade_mean_m']):>5}/{_fmt(s['ade_p50_m']):>5} | {leads}"
                                 + (f" | short {cell['forecasts_shorter_than_horizon']} truth-short {cell['truth_shorter_than_horizon']}"
                                    if stratum == STRATUM_ALL else ""))
    lines.append("")
    lines.append("paired (mean / p50 ΔADE, arm-better share; over the flights both hold at the anchor set):")
    for label, readings in payload["paired"].items():
        for name, sets in readings.items():
            for set_label, strata in sets.items():
                for stratum, cell in strata.items():
                    lines.append(f"   {label:<12s} {name:<36s} {set_label:<6s} {stratum[:30]:<30s} n={cell['n']:>4d} "
                                 f"{_fmt(cell['delta_mean_m']):>6} / {_fmt(cell['delta_p50_m']):>6}  better {_fmt(cell['arm_better_share'], 3)}")
    return "\n".join(lines) + "\n"


def measure(arms: list[Arm], reference: str, series_by_arm: dict[str, list[FlightSeries]], plan: ReadoutPlan,
            device: torch.device, records_root: Path | None, campaign: str) -> dict:
    """The whole artifact from loaded arms — what `main` writes, and what a test drives directly."""
    check_arms_share_the_anchor(arms)
    measured = {arm.label: measure_arm(arm, series_by_arm[arm.label], plan, device, records_root, campaign) for arm in arms}
    return {
        "schema": RESULT_SCHEMA,
        "plan": {"split": plan.split, "horizon_s": plan.horizon_s, "leads_s": list(plan.leads_s),
                 "bins": [FIXED_SET, *(bin_label(value) for value in plan.bins_m)], "bins_m": list(plan.bins_m),
                 "limit": plan.limit, "write_records": plan.write_records},
        "reference": reference,
        "reads_the_future": READS_THE_FUTURE,
        "strata_anchor": "the fixed anchor (strata_fixed_at_anchor)",
        "device": str(device),
        "arms": measured,
        "paired": pair_arms(measured, reference),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH",
                        help="an arm to read (repeatable); a plan-reading one is read with and without its token")
    parser.add_argument("--reference", required=True, metavar="LABEL=PATH",
                        help="the plan-free whole-approach checkpoint every no-plan reading is paired against")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--split", choices=("val", "train", FORBIDDEN_SPLIT), default="val")
    parser.add_argument("--horizon-s", type=float, default=DEFAULT_HORIZON_S)
    parser.add_argument("--leads-s", default=",".join(f"{h:g}" for h in DEFAULT_LEADS_S))
    parser.add_argument("--bins-km", default=",".join(f"{b:g}" for b in DEFAULT_BINS_KM),
                        help="remaining-path bins beside the fixed anchor (empty string: the fixed anchor only)")
    parser.add_argument("--limit", type=int, default=0, help="first N flights of the split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--write-records", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser


def parse_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ReadoutPlan:
    if args.split == FORBIDDEN_SPLIT:
        parser.error(f"the {FORBIDDEN_SPLIT} split is sealed")
    if not math.isfinite(args.horizon_s) or args.horizon_s <= 0.0:
        parser.error("--horizon-s must be positive")
    leads = tuple(float(x) for x in args.leads_s.split(",") if x.strip())
    if not leads or any(lead <= 0.0 or lead > args.horizon_s + ROW_TOLERANCE_S for lead in leads):
        parser.error("--leads-s are positive seconds inside the horizon")
    bins = tuple(float(x) * 1000.0 for x in args.bins_km.split(",") if x.strip())
    if any(value <= 0.0 for value in bins) or len(set(bins)) != len(bins):
        parser.error("--bins-km are distinct positive kilometres")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return ReadoutPlan(split=args.split, horizon_s=float(args.horizon_s), leads_s=leads, bins_m=bins,
                       limit=int(args.limit), batch_size=args.batch_size, write_records=bool(args.write_records))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    [(reference_label, reference_path)] = parse_arms(parser, [args.reference]).items()
    if reference_label in arms:
        parser.error(f"--reference label {reference_label!r} is also a --checkpoint label")
    plan = parse_plan(parser, args)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the readout is an immutable artifact")
    device = resolve_device(args.device)
    grid = Grid(split=plan.split, bins_m=(), min_future_s=0.0, batch_size=plan.batch_size, limit=plan.limit)
    loaded = [load_arm(reference_label, reference_path, grid, device, instrument=INSTRUMENT)]
    check_arm(loaded[0], plan, reference=True)
    for label, path in arms.items():
        arm = load_arm(label, path, grid, device, instrument=INSTRUMENT, refuse_plan=False)
        check_arm(arm, plan, reference=False)
        loaded.append(arm)
    check_arms_share_the_anchor(loaded)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    try:
        series_by_arm = {arm.label: cohort_series(arm, grid) for arm in loaded}
        payload = measure(loaded, reference_label, series_by_arm, plan, device,
                          staged / RECORDS_DIR if plan.write_records else None, out.name)
        (staged / "short_horizon_readout.json").write_text(json.dumps(payload, indent=1))
        text = render(payload)
        (staged / "short_horizon_readout.txt").write_text(text)
        staged.chmod(0o755)
        staged.rename(out)
    except BaseException:
        print(f"\nremoving the staged artifact {staged} — the run did not complete", file=sys.stderr, flush=True)
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print()
    print(text, end="")
    print(f"wrote {out / 'short_horizon_readout.txt'} and {out / 'short_horizon_readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
