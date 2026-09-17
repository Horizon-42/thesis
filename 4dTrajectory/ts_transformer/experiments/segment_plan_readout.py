#!/usr/bin/env python
"""The segment-plan readout (two-tier v2 §4, the L2 evaluation): a plan head's waypoint error at
each lead, from real histories re-anchored along the approach, paired against whole-approach
references and a constant-velocity extrapolation.

`docs/2026-09-17_two_tier_plan_v2.zh.md` §4. The plan layer draws the next M coarse waypoints
in one shot; the number the design asks for is the displacement between the drawn path and the
truth at 60 / 120 / 180 / 300 s after the anchor, per stratum, against (i) a constant-velocity
extrapolation of the anchor state, (ii) native32's own rollout at the same lead and (iii) the
state path's forecast — flight by flight over the same anchors::

    python run_ts.py segment_plan_readout --checkpoint A_s1337=<ckpt> [--checkpoint ...] \\
        --reference native32=<ckpt> --reference state=<ckpt> --out <dir> \\
        [--leads-s 60,120,180,300] [--bins-km 12,8,6] [--split val] [--limit N] [--batch-size B] \\
        [--write-records]

**One anchor for every checkpoint.** The readout's fixed anchor is the LATEST of the checkpoints'
own fixed anchors (`default_anchor`: L−1, or a run's `anchor_floor_index`), so every checkpoint has
a complete lookback there and the pairing is anchor for anchor; each checkpoint's own anchor is
stated in the artifact. The remaining-path bins (`data/anchor_grid.py`, `anchors_for_bin`) take
the longest lookback of the set, never before the fixed anchor, with the first lead of truth after
them. **Strata are fixed at the fixed anchor** (`strata_fixed_at_anchor`).

**Each checkpoint is read on ITS OWN split** (`cohort_series`: the flights its checkpoint names,
built under its own config — a control head's aircraft filter is not a state head's), and the
pairing is over the flights BOTH sides hold, the count stated per cell: native32 is an
openap-direct cohort, the state arm and a plan head are all-aircraft ones, and the same flight is
the same geometry under every build. The constant-velocity reference is read over the union.

**Readings past a forecast's end HOLD its last row** (`displacement_at(hold_forecast_end=True)`):
a plan that says it arrived is at the threshold, and reading it there at a later lead measures
that claim — the S1 readout's "absent, never scored" rule would let an early arrival escape. Only
the truth's end makes a lead absent — the OBSERVED track's end, the truth `displacement_at` reads
(the plan block and the bins' future floor read the supervision rows closed to the threshold,
`truth_duration_s`; the two ends differ by the fitted tail, a median 6 s at KRDU) — and the held
count is stated per lead, in the paired cells and in the gate's criterion lines too. A reference
whose horizon cannot reach the last lead by construction (a ``window`` head) is refused.

The fixed set admits a flight only where the common anchor exists in its observed track with the
first lead of truth after it (a reference's own build may hold a flight one sample shorter than
that); the bins go through `anchors_for_bin`. Both counts are stated (``anchored_flights``).

**Numbers**, per checkpoint · anchor set · stratum: the displacement at each lead (n, p50, mean,
held); for a segment-plan checkpoint also its plan block (`outputs.segment_plan.readout`: the ADE
inside the plan's span, the per-segment errors, the arrival confusion, the arrival-time error).
Then PAIRED, every checkpoint against every reference (the constant-velocity extrapolation
included) per set · stratum · lead over the flights BOTH hold: each side's p50 over those flights,
their difference (what gate L2 reads: `delta_of_p50_m`), the p50 of the per-flight difference
and the arm-better share. The per-flight rows stay in the artifact.

``--write-records`` writes every checkpoint's forecasts — the arms' AND the references' (the
constant-velocity rows are not records) — as predict-shaped record directories under
``records/<label>/<anchor set>/`` with a ``segment_plan`` summary block naming the subset.
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
from ts_transformer.config import (
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PLAN_CONDITIONING_OFF,
    PLAN_WAYPOINT_SEGMENT_S,
    PREDICTION_SEGMENT_PLAN,
    PREDICTION_STATE,
    default_anchor,
)
from ts_transformer.data.anchor_grid import anchors_for_bin, bin_label, remaining_path_profiles, strata_fixed_at_anchor
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.experiments.anytime_curve import FORBIDDEN_SPLIT, Arm, Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast, forecast_approaches
from ts_transformer.inference.receding import ROW_TOLERANCE_S, displacement_at
from ts_transformer.io_utils import file_sha256
from ts_transformer.outputs.base import ForecastOptions
from ts_transformer.outputs.segment_plan.readout import plan_reading, summarize_readings
from ts_transformer.outputs.segment_plan.strategy import decode_series, forecast_from_plan

RESULT_SCHEMA = "ts-segment-plan-readout-v1"
RECORDS_BLOCK = "segment_plan"
RECORDS_SCHEMA = "ts-segment-plan-records-v1"
RECORDS_DIR = "records"
INSTRUMENT = "the segment-plan readout"
DEFAULT_LEADS_S = (60.0, 120.0, 180.0, 300.0)
DEFAULT_BINS_KM = (12.0, 8.0, 6.0)
FIXED_SET = "fixed"
#: The built-in reference: the anchor state carried on at its own velocity.
CONSTANT_VELOCITY = "constant-velocity"
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED)


@dataclass(frozen=True)
class ReadoutPlan:
    split: str
    leads_s: tuple[float, ...]
    bins_m: tuple[float, ...]
    limit: int
    batch_size: int | None
    write_records: bool


def check_checkpoint(arm: Arm, *, reference: bool, leads_s: tuple[float, ...]) -> None:
    config = arm.config
    if config.latent_dim:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} is defined on the deterministic decode")
    if config.plan_conditioning != PLAN_CONDITIONING_OFF:
        raise SystemExit(f"{arm.label} ({arm.path}): a plan-reading checkpoint reads the truth's plan; {INSTRUMENT} "
                         "measures the heads that draw one")
    if config.control_horizon_s:
        raise SystemExit(f"{arm.label} ({arm.path}): a {config.control_horizon_s:g} s fixed-horizon head ends before the "
                         f"leads; {INSTRUMENT} reads whole-approach forecasts and plans (`short_horizon_readout` reads it)")
    if reference and config.prediction_output == PREDICTION_SEGMENT_PLAN:
        raise SystemExit(f"--reference {arm.label} ({arm.path}) is a segment-plan head; a reference is a whole-approach "
                         "forecast the plan is paired against")
    if config.horizon_mode == HORIZON_WINDOW and config.pred_len * config.dt_s < max(leads_s) - ROW_TOLERANCE_S:
        raise SystemExit(f"{arm.label} ({arm.path}): a {config.pred_len * config.dt_s:g} s window forecast ends before the "
                         f"{max(leads_s):g} s lead on EVERY flight; held there it would be read at its end, not predicted")


def common_anchor(arms: list[Arm]) -> int:
    """The latest of the checkpoints' fixed anchors — every checkpoint has a full lookback there."""
    return max(default_anchor(arm.config) for arm in arms)


def constant_velocity_forecast(series: FlightSeries, anchor: int, leads: tuple[float, ...], channels: int) -> Forecast:
    """The anchor state carried on at its own chart velocity, one row per lead."""
    state = np.asarray(series.values[anchor], dtype=np.float64)
    offsets = np.asarray(leads, dtype=np.float64)
    values = np.zeros((len(offsets), channels), dtype=np.float64)
    values[:, list(POSITION_IDX)] = state[list(POSITION_IDX)][None, :] + offsets[:, None] * state[list(VELOCITY_IDX)][None, :]
    values[:, list(VELOCITY_IDX)] = state[list(VELOCITY_IDX)][None, :]
    durations = np.diff(np.concatenate([[0.0], offsets]))
    return Forecast(
        times=float(series.times[anchor]) + offsets, values=values, normalized_progress=offsets / offsets[-1],
        anchor=anchor, final_time_s=float(offsets[-1]), predicted_final_time_s=float(offsets[-1]),
        horizon_mode=HORIZON_NORMALIZED, passes=1, truncated_at_threshold=False, horizon_capped=True,
        sample_durations_s=durations, segment_durations_s=durations, prediction_output=PREDICTION_STATE,
    )


def anchor_sets(series: list[FlightSeries], plan: ReadoutPlan, *, fixed: int, seq_len: int) -> dict[str, dict[int, int]]:
    """``{set label: {flight index: anchor}}`` — the common fixed anchor for every flight whose
    observed track holds it with the first lead after it, then each bin at the longest lookback
    of the checkpoints, with the first lead of truth after it (`anchors_for_bin`)."""
    profiles = remaining_path_profiles(series)
    sets = {FIXED_SET: {
        index: fixed for index, item in enumerate(series)
        if fixed < item.n_samples and float(item.times[-1] - item.times[fixed]) >= min(plan.leads_s) - ROW_TOLERANCE_S
    }}
    for target_m in plan.bins_m:
        sets[bin_label(target_m)] = anchors_for_bin(
            series, profiles, target_m, seq_len=seq_len, min_future_s=min(plan.leads_s), minimum_anchor_index=fixed,
        )
    return sets


def lead_row(item: FlightSeries, forecast: Forecast, anchor: int, leads: tuple[float, ...]) -> dict:
    origin = float(item.times[anchor])
    end = float(forecast.times[-1])
    at, held = {}, {}
    for lead in leads:
        key = f"{lead:g}"
        at[key] = displacement_at(item, forecast, anchor, origin + lead, hold_forecast_end=True)
        # `displacement_at`'s own rule for "past the forecast's end", so held and absent agree
        held[key] = bool(at[key] is not None and origin + lead > end + ROW_TOLERANCE_S)
    return {"anchor_index": anchor, "at": at, "held": held, "forecast_end_s": end - origin,
            "predicted_final_time_s": float(forecast.predicted_final_time_s)}


def measure_set(arm: Arm | None, series: list[FlightSeries], anchors: dict[int, int], plan: ReadoutPlan,
                device: torch.device, batch_size: int, channels: int, *, build_records: bool,
                ) -> tuple[dict[str, dict], list[tuple[int, object, dict]], list[tuple[str, dict]]]:
    """Every flight of one anchor set forecast from its anchor (``arm`` None = the constant-velocity
    reference) and read at the leads. Returns the rows by key, the record pairs and, for a
    segment-plan head, the per-flight plan readings (`plan_reading`) by key."""
    groups: dict[int, list[int]] = {}
    for index, anchor in anchors.items():
        groups.setdefault(anchor, []).append(index)
    rows: dict[str, dict] = {}
    pairs: list[tuple[int, object, dict]] = []
    readings: list[tuple[str, dict]] = []
    for anchor in sorted(groups):
        members = groups[anchor]
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            batch = [series[index] for index in chunk]
            if arm is None:
                forecasts = [constant_velocity_forecast(item, anchor, plan.leads_s, channels) for item in batch]
            elif arm.config.prediction_output == PREDICTION_SEGMENT_PLAN:
                # ONE forward: the plans are decoded once, read by the plan block and laid as the forecast
                span_s = float(arm.config.segment_plan_segments) * PLAN_WAYPOINT_SEGMENT_S
                plans = decode_series(arm.model, batch, arm.config, arm.normalizer, anchor, device)
                readings.extend((item.dataset_id, plan_reading(decoded, item, anchor, span_s))
                                for item, decoded in zip(batch, plans, strict=True))
                forecasts = [forecast_from_plan(item, decoded, anchor, arm.config) for item, decoded in zip(batch, plans, strict=True)]
            else:
                forecasts = forecast_approaches(arm.model, batch, arm.config, arm.normalizer, anchor=anchor, device=device,
                                                options=ForecastOptions())
            for index, item, forecast in zip(chunk, batch, forecasts, strict=True):
                rows[item.dataset_id] = lead_row(item, forecast, anchor, plan.leads_s)
                if build_records and arm is not None:
                    pairs.append((index, build_prediction_record(
                        item, forecast, index=index, model_name=arm.config.model, horizon_mode=arm.config.horizon_mode,
                        split=plan.split,
                    ), observed_series_metrics(item, forecast, points=arm.config.validation_common_grid_points)))
    return rows, pairs, readings


def _p50(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def stratum_cells(rows: dict[str, dict], masks: dict[str, np.ndarray], keys: list[str], leads: tuple[float, ...]) -> dict:
    out = {}
    for stratum in STRATA:
        cells = [rows[key] for key, keep in zip(keys, masks[stratum], strict=True) if keep and key in rows]
        out[stratum] = {"n": len(cells), "at_lead_m": {}}
        for lead in leads:
            key = f"{lead:g}"
            values = [c["at"][key] for c in cells if c["at"][key] is not None]
            out[stratum]["at_lead_m"][key] = {
                "n": len(values), "p50": _p50(values), "mean": _mean(values),
                "held": int(sum(1 for c in cells if c["held"][key])),
            }
    return out


def plan_cells(readings: dict[str, dict], masks: dict[str, np.ndarray], keys: list[str], config) -> dict:
    out = {}
    for stratum in STRATA:
        members = [readings[key] for key, keep in zip(keys, masks[stratum], strict=True) if keep and key in readings]
        out[stratum] = summarize_readings(members, config) if members else None
    return out


def paired(arm_rows: dict[str, dict], base_rows: dict[str, dict], masks: dict[str, np.ndarray], keys: list[str],
           leads: tuple[float, ...]) -> dict:
    """Per stratum and lead, over the flights BOTH hold there: each side's p50, their difference,
    the p50 of the per-flight difference, the arm-better share."""
    out = {}
    for stratum in STRATA:
        common = [key for key, keep in zip(keys, masks[stratum], strict=True) if keep and key in arm_rows and key in base_rows]
        out[stratum] = {}
        for lead in leads:
            k = f"{lead:g}"
            both = [key for key in common if arm_rows[key]["at"][k] is not None and base_rows[key]["at"][k] is not None]
            a = np.array([arm_rows[key]["at"][k] for key in both])
            b = np.array([base_rows[key]["at"][k] for key in both])
            out[stratum][k] = {
                "n": int(a.size),
                "arm_p50_m": _p50(a), "reference_p50_m": _p50(b),
                "delta_of_p50_m": float(np.median(a) - np.median(b)) if a.size else None,
                "delta_p50_m": _p50(a - b),
                "arm_better_share": float(np.mean(a < b)) if a.size else None,
                # how many of each side's readings are its last row held past its end
                "arm_held": int(sum(1 for key in both if arm_rows[key]["held"][k])),
                "reference_held": int(sum(1 for key in both if base_rows[key]["held"][k])),
            }
    return out


def records_block(arm: Arm, set_label: str, plan: ReadoutPlan, *, campaign: str, anchor: int, measured: int,
                  records: int) -> dict:
    return {
        "schema": RECORDS_SCHEMA, "campaign": campaign, "label": arm.label, "anchor_set": set_label,
        "leads_s": list(plan.leads_s), "fixed_anchor": anchor, "own_fixed_anchor": default_anchor(arm.config),
        "prediction_output": arm.config.prediction_output, "split": plan.split, "limit": plan.limit,
        "split_flights": len(arm.payload["split"][plan.split]), "measured_flights": measured, "records": records,
    }


def measure_checkpoint(arm: Arm | None, label: str, series: list[FlightSeries], sets: dict[str, dict[int, int]],
                       masks: dict[str, np.ndarray], keys: list[str], plan: ReadoutPlan, device: torch.device,
                       batch_size: int, channels: int, records_root: Path | None, campaign: str, anchor: int) -> dict:
    started = time.time()
    is_plan = arm is not None and arm.config.prediction_output == PREDICTION_SEGMENT_PLAN
    out_sets: dict[str, dict] = {}
    record_dirs: dict[str, str] = {}
    for set_label, anchors in sets.items():
        rows, pairs, readings = measure_set(arm, series, anchors, plan, device, batch_size, channels,
                                            build_records=records_root is not None)
        out_sets[set_label] = {
            "anchored_flights": len(anchors), "strata": stratum_cells(rows, masks, keys, plan.leads_s), "flights": rows,
            **({"plan": plan_cells(dict(readings), masks, keys, arm.config)} if is_plan else {}),
        }
        if records_root is not None and pairs:
            ordered = sorted(pairs, key=lambda pair: pair[0])
            directory = records_root / label / set_label
            write_batch(
                [record for _, record, _ in ordered], output_dir=directory, config_dict=arm.config.to_dict(),
                flight_metrics=[metrics for _, _, metrics in ordered], checkpoint=str(arm.path), split=plan.split,
                extra_summary={RECORDS_BLOCK: records_block(arm, set_label, plan, campaign=campaign, anchor=anchor,
                                                            measured=len(series), records=len(ordered))},
            )
            record_dirs[set_label] = str(directory.relative_to(records_root.parent))
        cell = out_sets[set_label]["strata"]
        leads = " ".join(f"e({k}) {_fmt(v['p50'])}" for k, v in cell[STRATUM_VECTORED]["at_lead_m"].items())
        print(f"  {label:<14s} {set_label:<6s} n={cell[STRATUM_ALL]['n']:>4d} vectored (n {cell[STRATUM_VECTORED]['n']}) "
              f"p50 {leads}", flush=True)
    return {
        "checkpoint": None if arm is None else str(arm.path),
        "checkpoint_sha256": None if arm is None else file_sha256(arm.path),
        "prediction_output": CONSTANT_VELOCITY if arm is None else arm.config.prediction_output,
        "own_fixed_anchor": None if arm is None else default_anchor(arm.config),
        "seq_len": None if arm is None else arm.config.seq_len,
        "segment_plan_segments": arm.config.segment_plan_segments if is_plan else None,
        "sets": out_sets, "record_dirs": record_dirs, "seconds": time.time() - started,
    }


def pair_all(measured: dict[str, dict], arm_labels: list[str], reference_labels: list[str],
             cohorts: dict[str, "Cohort"], leads: tuple[float, ...]) -> dict:
    """Every arm against every reference over the flights both hold, in the ARM's strata."""
    out: dict[str, dict] = {}
    for label in arm_labels:
        out[label] = {}
        for reference in reference_labels:
            out[label][reference] = {
                set_label: paired(measured[label]["sets"][set_label]["flights"],
                                  measured[reference]["sets"][set_label]["flights"],
                                  cohorts[label].masks, cohorts[label].keys, leads)
                for set_label in measured[label]["sets"]
            }
    return out


@dataclass(frozen=True)
class Cohort:
    """One checkpoint's flights at the readout's anchor: the series, their keys, the strata masks
    fixed at the anchor, and the anchor sets."""

    series: list[FlightSeries]
    keys: list[str]
    masks: dict[str, np.ndarray]
    sets: dict[str, dict[int, int]]


def cohort_at(series: list[FlightSeries], plan: ReadoutPlan, *, anchor: int, seq_len: int) -> Cohort:
    """The cohort's flights that HOLD the common anchor (a reference's own build may admit a flight
    one sample short of it), stratified there."""
    series = [item for item in series if anchor < item.n_samples]
    keys = [item.dataset_id for item in series]
    return Cohort(series, keys, strata_fixed_at_anchor(series, keys, anchor=anchor),
                  anchor_sets(series, plan, fixed=anchor, seq_len=seq_len))


def _fmt(value, digits: int = 0) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    plan = payload["plan"]
    lines = [
        f"Segment-plan readout: the displacement at {', '.join(f'{h:g}' for h in plan['leads_s'])} s from the common fixed "
        f"anchor {payload['anchor']} and the remaining-path bins {', '.join(plan['bins'])}, split {plan['split']}"
        + (f", limit {plan['limit']}" if plan["limit"] else "") + f"; references {', '.join(payload['references'])}.",
        "A forecast's last row is HELD past its end (a plan that arrived is at the threshold); a lead is absent only "
        "where the truth has landed. 'held' counts the readings past the forecast's own end. Each checkpoint is read on its "
        "own split; the pairing is over the flights both hold (n per cell). Cohorts: "
        + ", ".join(f"{label} {c['flights']}" for label, c in payload["cohorts"].items()) + ".",
    ]
    for label, block in payload["checkpoints"].items():
        lines.append("")
        lines.append(f"── {label} — {block['checkpoint'] or CONSTANT_VELOCITY} ({block['prediction_output']}"
                     + (f", own fixed anchor {block['own_fixed_anchor']}, L={block['seq_len']}" if block["checkpoint"] else "")
                     + (f", M={block['segment_plan_segments']}" if block["segment_plan_segments"] else "") + ")")
        for set_label, cell in block["sets"].items():
            for stratum in STRATA:
                s = cell["strata"][stratum]
                leads = " ".join(f"e({k}) {_fmt(v['p50'])}/{_fmt(v['mean'])} n{v['n']}" + (f" h{v['held']}" if v["held"] else "")
                                 for k, v in s["at_lead_m"].items())
                lines.append(f"   {set_label:<6s} {stratum[:30]:<30s} n={s['n']:>4d} | {leads}")
            if cell.get("plan"):
                for stratum in STRATA:
                    p = cell["plan"][stratum]
                    if p is None:
                        continue
                    c = p["arrival_confusion"]
                    lines.append(f"   {set_label:<6s} {stratum[:30]:<30s} plan: covered ADE {_fmt(p['covered_ade_m'])} over "
                                 f"{_fmt(p['covered_s_mean'])} s | arrives plan/truth {p['plan_arrives_share']:.2f}/"
                                 f"{p['truth_arrives_share']:.2f} (both {c['both']}, plan-only {c['plan_only']}, "
                                 f"truth-only {c['truth_only']}) | |dT| {_fmt(p['arrival_time_error_s']['mean_abs'], 1)} s "
                                 f"n{p['arrival_time_error_s']['n']}")
    lines.append("")
    lines.append("paired (arm p50 − reference p50 over the flights both hold; then p50 of the per-flight difference; arm-better share):")
    for label, references in payload["paired"].items():
        for reference, sets in references.items():
            for set_label, strata in sets.items():
                for stratum, cells in strata.items():
                    parts = " ".join(f"{k}s {_fmt(v['arm_p50_m'])}−{_fmt(v['reference_p50_m'])}={_fmt(v['delta_of_p50_m'])} "
                                     f"(Δp50 {_fmt(v['delta_p50_m'])}, better {_fmt(v['arm_better_share'], 2)}, n{v['n']}"
                                     + (f", held {v['arm_held']}/{v['reference_held']}" if v["arm_held"] or v["reference_held"] else "")
                                     + ")"
                                     for k, v in cells.items())
                    lines.append(f"   {label:<12s} − {reference:<17s} {set_label:<6s} {stratum[:30]:<30s} {parts}")
    return "\n".join(lines) + "\n"


def measure(arms: list[Arm], references: list[Arm], series_by_label: dict[str, list[FlightSeries]], plan: ReadoutPlan,
            device: torch.device, records_root: Path | None, campaign: str) -> dict:
    """The whole artifact from loaded checkpoints, each over its own cohort — what `main` writes,
    and what a test drives directly."""
    everything = [*references, *arms]
    anchor = common_anchor(everything)
    seq_len = max(arm.config.seq_len for arm in everything)
    cohorts = {arm.label: cohort_at(series_by_label[arm.label], plan, anchor=anchor, seq_len=seq_len) for arm in everything}
    union: dict[str, FlightSeries] = {}
    for arm in everything:
        for item in series_by_label[arm.label]:
            union.setdefault(item.dataset_id, item)
    cohorts[CONSTANT_VELOCITY] = cohort_at(list(union.values()), plan, anchor=anchor, seq_len=seq_len)
    channels = len(everything[0].config.channels)
    batch_size = plan.batch_size or min(arm.config.batch_size for arm in everything)
    print(f"common fixed anchor {anchor}, leads {plan.leads_s}, sets {', '.join(cohorts[CONSTANT_VELOCITY].sets)}; cohorts "
          + ", ".join(f"{label} {len(c.series)}" for label, c in cohorts.items()), flush=True)
    measured: dict[str, dict] = {}
    for arm in everything:
        c = cohorts[arm.label]
        measured[arm.label] = measure_checkpoint(arm, arm.label, c.series, c.sets, c.masks, c.keys, plan, device, batch_size,
                                                 channels, records_root, campaign, anchor)
    c = cohorts[CONSTANT_VELOCITY]
    measured[CONSTANT_VELOCITY] = measure_checkpoint(None, CONSTANT_VELOCITY, c.series, c.sets, c.masks, c.keys, plan, device,
                                                     batch_size, channels, None, campaign, anchor)
    reference_labels = [*(arm.label for arm in references), CONSTANT_VELOCITY]
    return {
        "schema": RESULT_SCHEMA,
        "plan": {"split": plan.split, "leads_s": list(plan.leads_s),
                 "bins": [FIXED_SET, *(bin_label(value) for value in plan.bins_m)], "bins_m": list(plan.bins_m),
                 "limit": plan.limit, "write_records": plan.write_records},
        "anchor": anchor,
        "arms": [arm.label for arm in arms],
        "references": reference_labels,
        "strata_anchor": "the common fixed anchor (strata_fixed_at_anchor)",
        "cohorts": {
            label: {"flights": len(c.series),
                    "masks": {stratum: [key for key, keep in zip(c.keys, mask, strict=True) if keep] for stratum, mask in c.masks.items()}}
            for label, c in cohorts.items()
        },
        "device": str(device),
        "checkpoints": measured,
        "paired": pair_all(measured, [arm.label for arm in arms], reference_labels, cohorts, plan.leads_s),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH",
                        help="a plan head to read (repeatable)")
    parser.add_argument("--reference", action="append", required=True, metavar="LABEL=PATH",
                        help="a whole-approach checkpoint the arms are paired against (repeatable; the "
                             "constant-velocity extrapolation is always included)")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--split", choices=("val", "train", FORBIDDEN_SPLIT), default="val")
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
    leads = tuple(float(x) for x in args.leads_s.split(",") if x.strip())
    if not leads or any(lead <= 0.0 for lead in leads) or list(leads) != sorted(set(leads)):
        parser.error("--leads-s are distinct positive seconds in increasing order")
    bins = tuple(float(x) * 1000.0 for x in args.bins_km.split(",") if x.strip())
    if any(value <= 0.0 for value in bins) or len(set(bins)) != len(bins):
        parser.error("--bins-km are distinct positive kilometres")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return ReadoutPlan(split=args.split, leads_s=leads, bins_m=bins, limit=int(args.limit), batch_size=args.batch_size,
                       write_records=bool(args.write_records))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    references = parse_arms(parser, args.reference)
    clash = set(arms) & set(references)
    if clash or CONSTANT_VELOCITY in arms or CONSTANT_VELOCITY in references:
        parser.error(f"a label is both an arm and a reference, or is the built-in {CONSTANT_VELOCITY!r}: {sorted(clash)}")
    plan = parse_plan(parser, args)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the readout is an immutable artifact")
    device = resolve_device(args.device)
    grid = Grid(split=plan.split, bins_m=(), min_future_s=0.0, batch_size=plan.batch_size, limit=plan.limit)
    loaded_references = []
    for label, path in references.items():
        arm = load_arm(label, path, grid, device, instrument=INSTRUMENT)
        check_checkpoint(arm, reference=True, leads_s=plan.leads_s)
        loaded_references.append(arm)
    loaded_arms = []
    for label, path in arms.items():
        arm = load_arm(label, path, grid, device, instrument=INSTRUMENT)
        check_checkpoint(arm, reference=False, leads_s=plan.leads_s)
        loaded_arms.append(arm)
    everything = [*loaded_references, *loaded_arms]
    # each checkpoint on its OWN split; the pairing is over the flights both hold (counts in every cell)
    cohorts = {arm.label: cohort_series(arm, grid) for arm in everything}
    ids = {label: {item.dataset_id for item in series} for label, series in cohorts.items()}
    for arm in loaded_arms:
        for reference in loaded_references:
            common = len(ids[arm.label] & ids[reference.label])
            print(f"  {arm.label} ∩ {reference.label}: {common} of {len(ids[arm.label])} / {len(ids[reference.label])} flights",
                  flush=True)
            if not common:
                raise SystemExit(f"{arm.label} and {reference.label} share no {plan.split!r} flight; nothing to pair")
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    try:
        payload = measure(loaded_arms, loaded_references, cohorts, plan, device,
                          staged / RECORDS_DIR if plan.write_records else None, out.name)
        (staged / "segment_plan_readout.json").write_text(json.dumps(payload, indent=1))
        text = render(payload)
        (staged / "segment_plan_readout.txt").write_text(text)
        staged.chmod(0o755)
        staged.rename(out)
    except BaseException:
        print(f"\nremoving the staged artifact {staged} — the run did not complete", file=sys.stderr, flush=True)
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print(text, end="")
    print(f"wrote {out / 'segment_plan_readout.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
