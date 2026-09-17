#!/usr/bin/env python
"""The plan-given tracker in lockstep (two-tier T1 / T2): a control checkpoint re-asked every Δ.

Two-tier feasibility (`docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §10.3): the control
model is told the plan it flies — the truth's (`plan_conditioning=truth-next`, the oracle) — and the
arrival time (`cta_conditioning=given`), and asked again every ``--step-s`` seconds on its OWN flown
rows. Gate G1 reads this runner's records against the rule guidance flying the same plan
(`plan_oracle --route next --policy truth`, plan-and-guidance design §12.5: vectored 1847 m /
straight-in 283 m at L−1)::

    python run_ts.py tracker_lockstep --checkpoint T1a=<ckpt> --out <dir> --step-s 30 --write-records
    python run_ts.py tracker_lockstep --checkpoint T1a=<ckpt> --plan-head <plan ckpt> --out <dir> ...   # T2
    python run_ts.py tracker_lockstep --checkpoint L1_pa_s1337=<ckpt> --plan-head <segment-plan ckpt> \
        --anchor-floor-index 60 --out <dir> ...                                                        # two-tier E2E (S3)

WHERE THE PLAN COMES FROM is the protocol (`TruthPlans` / `TruthWaypoints` / `HeadPlans` /
`SegmentHeadWaypoints`): by default the TRUTH's (T1 / two-tier L1, protocol C, an oracle); with
``--plan-head`` a head's OWN prediction on the same rolled history at every ask (protocol A) — a plan
head's next instruction (T2, `order_from_prediction` → `targets_at`) for a ``truth-next`` tracker, or a
SEGMENT-PLAN head's next coarse waypoints (two-tier v2 §5, the E2E lockstep: `decode_series` → the
head's waypoints relative to the flown row, its own arrival time as the first ask's horizon — the
plan's span where it draws no arrival) for a ``waypoints`` tracker; a head of the other kind is
refused. Under the segment head every ask also records **e_plan** — the head's waypoints against the
truth's at the same flown time and position (`truth_waypoints`, the reading that never enters an
input) — beside the step error, so the design's e_plan and e_track are read from one run.
``--anchor-floor-index N`` reads every tracker at anchor N instead of its own fixed anchor (a head
whose lookback does not fit before the tracker's anchor — the L2 head's 61 samples against the L1
arms' 59 — needs it; stated in the artifact, refused below the tracker's own anchor). One rule is shared by both after the first ask (`on_final_capture`):
the capture height is undefined where the aircraft is ON the final at the ask (`labels.on_final_pose`)
— the observable reading of the training label's `join_at_anchor`, which a `TruthExpert` built at
``a0`` would otherwise keep from ``a0`` for the whole flight. The truth's first ask is the label
itself (future-read `join_at_anchor`); measured at a0 on 100 KRDU val flights the two rules differ on
1 flight. A head is refused on any flight it trained on, and on a series contract
(dt, chart, channels, airports, a lookback longer than the tracker's anchor) other than the tracker's.

Per flight from the fixed anchor ``a0`` (`default_anchor`), one ask at a time for the whole cohort
(ask k is at anchor ``a0 + k·Δ/dt`` for every flight still flying):

* the history is the observed track to ``a0`` continued by the rows flown so far
  (`inference.receding.rolled_series` — the model's ordinary predict path on that series);
* under the truth, the plan comes from ONE `TruthExpert` per flight, built at ``a0`` and read in
  time order at each ask's pose — at ``a0`` exactly the plan head's label (`targets_from_labels`),
  afterwards the truth's policy at the flown pose (the instruction in force; the truth's time-to-go
  from its nearest row plus the way back to it). The arrival time is the truth duration at ``a0``
  (what training hands a ``cta=given`` checkpoint) and that expert time-to-go afterwards. The
  pose's heading is the chart velocity's and its speed the extractors' physical ground speed
  (`extractors.ground_speeds`) — the definitions the label reads. Under a head, both come from the
  head's order at the pose (its arrival time clamped at `labels.T_MIN_S`, as the guidance flies it);
* the forecast is flown for Δ, unless it crosses the threshold ON THE FINAL inside Δ
  (`cut_at_threshold_crossing`: the flight ends there, ``crossed``). A flight ends ONLY by crossing or
  at its HORIZON, ``T₀ + max(30 s, 0.1·T₀)`` from ``a0`` (`plan_oracle.closing_horizon_s`, T₀ its
  FIRST ask's arrival time — the truth duration under the truth, the head's own under a head, so a
  head's run reads no future), cut there (``horizon``) — the budget the rule guidance's own rollout
  is flown to, so an on-time or slightly late arrival is scored as the guidance's is (review
  2026-09-16: ending at the arrival time left "established" to the sign of a rounding error on an
  on-time flight). Every ask hands a CTA of at least ``max(random_train_anchor_min_future_s, Δ)`` —
  an anchor the checkpoint trained at, and one whole step, so every leg is a whole step and the
  cohort stays in lockstep; an ask whose arrival time had to be raised to it is counted per flight
  (``asks_below_floor``). A checkpoint without a given CTA whose forecast is shorter than one step
  flies it whole and ends (``forecast-end``).

Variants (``--variants``), each scored per stratum (`strata_fixed_at_anchor` at ``a0``) and, with
``--write-records``, its own predict-shaped record directory under ``records/<label>/<variant>/``:
``receding`` (the plan as the checkpoint is conditioned), ``receding-no-plan`` (the ABSENT token on
the same weights — what the plan buys; refused for a checkpoint without a plan token) and
``one-shot`` (one ask at ``a0``, flown to its last whole step or its crossing, and closed from there
by the same asks as ``receding`` — what re-asking buys over the plan's own duration).

Each flight row carries the plan oracle's REFERENCE reading of the whole record
(`plan_oracle.reference_verdicts`: fully flyable, established = crossed the threshold on the final,
the corridor / glidepath / floor verdicts) and its difficulty covariates, so a gate that compares
the tracker with the rule guidance flying the same plan (`run_ts.py two_tier_gates`, design §10.7)
reads both with ONE definition, flight by flight. `run_ts.py lead_time_error` reads the records
unchanged; their summaries carry a ``lockstep`` block, and the publisher refuses them without a
category variant.
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
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_OFF,
    CTA_CONDITIONING_GIVEN,
    CTA_CONDITIONING_OFF,
    DURATION_HEAD_POINT,
    HOOK_SATURATIONS,
    HOOK_SATURATION_SOFT,
    PLAN_CONDITIONING_OFF,
    PLAN_CONDITIONING_WAYPOINTS,
    PREDICTION_CONTROL,
    PREDICTION_PLAN,
    PREDICTION_SEGMENT_PLAN,
    default_anchor,
    plan_waypoint_count,
)
from ts_transformer.data.anchor_grid import difficulty_at_anchor
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks
from ts_transformer.data.channels import IDX, POSITION_IDX
from ts_transformer.data.dataset import FlightSeries, truth_duration_s, window_anchors
from ts_transformer.experiments.anytime_curve import FORBIDDEN_SPLIT, Arm, Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.plan_oracle import closing_horizon_s, reference_verdicts
from ts_transformer.experiments.support import REPO_ROOT, forecast_geometry
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast, concatenate, cut_at_threshold_crossing, cut_rows, history_batch
from ts_transformer.inference.receding import ROW_TOLERANCE_S, cut_at_lead, displacement_at, rolled_series
from ts_transformer.io_utils import file_sha256
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import (
    PLAN_TOKEN_KEY, Waypoints, plan_token, plan_token_width, truth_waypoints, waypoint_token,
)
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.plan.extractors import extract_plan, ground_speeds
from ts_transformer.outputs.plan.labels import (
    Instruction, Operating, TruthExpert, on_final_pose, order_from_prediction, targets_at,
)
from ts_transformer.outputs.plan.model import prediction_rows
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton, SkeletonCache
from ts_transformer.outputs.segment_plan.strategy import decode_series

# v2 (2026-09-16): every flight row carries `difficulty` and the plan oracle's `reference`
# verdicts; every stratum block their shares. v3 (2026-09-17, two-tier v2 §3): every flight
# row carries `asks_e_m` — the displacement from the truth at the END of every leg flown (the
# per-ask step error, the drift reading) — and every stratum block its p50 by ask; the plan
# records the command hook the arms were flown under. v4 (2026-09-17, two-tier v2 §5): every
# flight row carries `asks_plan_e_m` (e_plan per ask under a segment-plan head, empty otherwise)
# and `horizon_from_plan_span` (the first ask's horizon came from the plan's span, not an arrival
# it drew); the plan records `anchor_floor_index`. The L1 campaign's artifacts are v3. 2026-09-18: each
# checkpoint block also carries `anchor_floor_excluded` (count / flight_keys — the split flights a floor
# override excluded, beside the block's `flights` and `split_flights`; optional for a reader, zero without a floor).
RESULT_SCHEMA = "ts-tracker-lockstep-v4"
RESULT_SCHEMA_V3 = "ts-tracker-lockstep-v3"   # what the gates still read: the L1 campaign's artifacts
#: The block each record directory's summary carries — the publisher's
#: `VARIANT_RECORD_BLOCKS` mirrors this name.
RECORDS_BLOCK = "lockstep"
RECORDS_SCHEMA = "ts-lockstep-records-v1"
RECORDS_DIR = "records"
INSTRUMENT = "the plan-given lockstep"
DEFAULT_STEP_S = 30.0
VARIANT_RECEDING = "receding"
VARIANT_NO_PLAN = "receding-no-plan"
VARIANT_ONE_SHOT = "one-shot"
VARIANTS = (VARIANT_RECEDING, VARIANT_NO_PLAN, VARIANT_ONE_SHOT)
#: How a flight's lockstep ended.
ENDED_CROSSED = "crossed"          # crossed the threshold on the final
ENDED_HORIZON = "horizon"          # still flying at its horizon, cut there
ENDED_FORECAST = "forecast-end"    # no given CTA and a forecast shorter than one step: flown whole
LEADS_S = (30.0, 60.0, 120.0, 180.0, 300.0)
HORIZON_RULE = "T0 + max(30 s, 0.1·T0), T0 = the first ask's arrival time (plan_oracle.closing_horizon_s)"


@dataclass(frozen=True)
class LockstepPlan:
    split: str
    step_s: float
    variants: tuple[str, ...]
    limit: int
    batch_size: int | None
    write_records: bool
    #: The command hook every arm is flown under (``hook/saturation``), or None: the network's
    #: own schedule. Two-tier v2 §3: the gate reads the hook-free run, the hooked one is the
    #: delivery-form reading beside it.
    command_hook: str | None = None
    #: ``--anchor-floor-index``: every tracker read at this anchor instead of its own (None = its own).
    anchor_floor_index: int | None = None


@dataclass
class FlightRun:
    """One flight being stepped: the legs flown, how it ended."""

    series: FlightSeries
    skeleton: RunwaySkeleton
    horizon_s: float = math.inf    # set at the first ask, from its arrival time
    next_ask: int = 0              # the ask index this flight is next asked at
    legs: list[Forecast] = field(default_factory=list)
    asks: int = 0
    asks_below_floor: int = 0
    cta_raised_max_s: float = 0.0  # the largest amount an ask's arrival time was raised by
    ended: str | None = None
    truncated: bool = False
    #: Per leg flown: the ask it came from, the seconds it spans and the displacement from the
    #: truth at its END (None past the truth's end) — the per-ask step error a drift reading
    #: lines up along the flight.
    asks_e: list[dict] = field(default_factory=list)
    #: Under a segment-plan head, per ask: the head's waypoints against the truth's at the same
    #: flown time and position — e_plan (two-tier v2 §5), per waypoint and pooled; empty otherwise.
    asks_plan_e: list[dict] = field(default_factory=list)
    #: Under a segment-plan head: whether the FIRST ask's plan drew an arrival (its time is then the
    #: horizon) or not (the plan's span is — a cap, and the flight that outlives it ends at the
    #: horizon un-established), and that segment's arrival probability. None under any other source.
    plan_arrives_at_a0: bool | None = None
    plan_arrival_probability_at_a0: float | None = None

    @property
    def flown_s(self) -> float:
        return float(sum(np.sum(leg.sample_durations_s) for leg in self.legs))


# ── where the plan comes from ───────────────────────────────────────────────────

@dataclass(frozen=True)
class AskPlan:
    """What one ask hands the checkpoint besides its history: the arrival time (the CTA of a
    ``cta=given`` checkpoint, and the first ask's horizon) and the plan TOKEN, already in the
    shape the checkpoint reads (`plan_token` / `waypoint_token`)."""

    arrival_s: float
    token: np.ndarray


def pose(history: FlightSeries, anchor: int) -> tuple[float, float, float, float]:
    """``(e, n, heading, ground speed)`` at the anchor: the chart velocity's heading and the
    extractors' physical ground speed — the definitions the plan label reads."""
    values = np.asarray(history.values[anchor], dtype=np.float64)
    return (
        float(values[IDX["e"]]), float(values[IDX["n"]]),
        math.atan2(float(values[IDX["ndot"]]), float(values[IDX["edot"]])),
        float(ground_speeds(history, slice(anchor, anchor + 1))[0]),
    )


def on_final_capture(operating: Operating, skeleton: RunwaySkeleton, e: float, n: float, heading: float) -> Operating:
    """The capture height undefined where the aircraft is on the final at this pose — what a
    label read at this anchor says of a joined flight, from what the pose itself shows."""
    return replace(operating, h_capture_m=None) if on_final_pose(skeleton, e, n, heading) else operating


class TruthPlans:
    """T1, protocol C: the truth's plan — one `TruthExpert` per flight, built at ``a0`` and read in
    time order — and the truth's arrival time (the truth duration at ``a0``, the expert's
    time-to-go afterwards)."""

    source = "truth"
    reads_the_future = "the truth's plan and arrival time at every ask (protocol C, an oracle)"

    def __init__(self, runs: list[FlightRun], a0: int, config=None) -> None:
        del config    # the instruction plan reads the skeleton, not the run's token shape
        self.experts = {
            id(run): TruthExpert(extract_plan(run.series, a0, run.skeleton), run.series, a0, run.skeleton)
            for run in runs
        }

    def at_ask(self, runs: list[FlightRun], histories: list[FlightSeries], anchor: int, *, first: bool) -> list[AskPlan]:
        out = []
        for run, history in zip(runs, histories, strict=True):
            e, n, heading, speed = pose(history, anchor)
            try:
                instruction, operating = self.experts[id(run)].order_at(e, n, heading, speed)
            except ValueError as exc:  # name the flight: the expert's own message names only the path
                raise ValueError(f"{run.series.dataset_id} at anchor {anchor}: {exc}") from exc
            # at a0 the CTA training hands a `cta=given` checkpoint (the truth duration) and the
            # label itself; afterwards the expert's time-to-go at the flown pose and the shared
            # on-final capture rule — at a0 the two times differ by the expert's nearest-row
            # reading (0.24 s on the synthetic fixture)
            arrival = truth_duration_s(run.series, anchor) if first else float(operating.T_s)
            if not first:
                operating = on_final_capture(operating, run.skeleton, e, n, heading)
            out.append(AskPlan(arrival_s=arrival, token=plan_token(targets_at(operating, instruction, e, n, run.skeleton))))
        return out


class HeadPlans:
    """T2, protocol A: a plan head's own order on the ask's history, at every ask — its target
    vector through `order_from_prediction` (every parameter clamped into its range, the arrival
    time at `T_MIN_S`), the same `Operating` / `Instruction` pair the truth hands over."""

    source = "head"
    reads_the_future = ("nothing but the landed runway (the threshold frame and the skeleton, as every ts "
                        "number): the plan and the arrival time are the plan head's own (protocol A)")

    def __init__(self, head: Arm, batch_size: int, device: torch.device, config) -> None:
        del config   # the tracker's; the instruction token has one shape
        self.head, self.batch_size, self.device = head, batch_size, device

    def at_ask(self, runs: list[FlightRun], histories: list[FlightSeries], anchor: int, *, first: bool) -> list[AskPlan]:
        del first
        config, normalizer = self.head.config, self.head.normalizer
        values, probability = [], []
        for start in range(0, len(histories), self.batch_size):
            windows = history_batch(histories[start : start + self.batch_size], config, normalizer, anchor)
            with torch.no_grad():
                chunk_values, chunk_probability = prediction_rows(self.head.model(torch.from_numpy(windows).to(self.device)))
            values.append(chunk_values)
            probability.append(chunk_probability)
        values, probability = np.concatenate(values), np.concatenate(probability)
        if not (np.isfinite(values).all() and np.isfinite(probability).all()):
            raise ValueError(f"the plan head {self.head.path} returned a non-finite prediction at anchor {anchor}")
        out = []
        for i, (run, history) in enumerate(zip(runs, histories, strict=True)):
            e, n, heading, _speed = pose(history, anchor)
            order = order_from_prediction(values[i], float(probability[i]), e, n, run.skeleton)
            operating = on_final_capture(Operating(
                T_s=order.T_s, V_mid_mps=order.V_mid_mps, d_decel_m=order.d_decel_m, V_final_mps=order.V_final_mps,
                h_capture_m=order.h_capture_m, d_join_m=order.d_join_m, remaining_m=order.remaining_m,
            ), run.skeleton, e, n, heading)
            out.append(AskPlan(arrival_s=float(order.T_s),
                               token=plan_token(targets_at(operating, order.instruction, e, n, run.skeleton))))
        return out


class TruthWaypoints:
    """Two-tier v2 §3, protocol C: the TRUTH's coarse plan seen from where the aircraft IS — its
    position every `PLAN_WAYPOINT_SEGMENT_S` after the ask's flown time, relative to the flown
    row (`plan_token.truth_waypoints`: the training row's own definition with the origin moved
    from the observed anchor to the flown state). Time-indexed, not pose-indexed: a tracker
    that has fallen behind is told where the truth IS at the next coarse instants, so the
    plan carries the schedule as well as the path. The first ask's arrival time is the truth
    duration at ``a0``, read for the horizon only — a fixed-horizon checkpoint takes no CTA."""

    source = "truth-waypoints"
    reads_the_future = ("the truth's position at the next coarse waypoints from every ask's flown time and "
                        "position, and its duration at a0 for the horizon (protocol C, an oracle)")

    def __init__(self, runs: list[FlightRun], a0: int, config) -> None:
        del runs
        self.a0, self.count = a0, plan_waypoint_count(config)

    def at_ask(self, runs: list[FlightRun], histories: list[FlightSeries], anchor: int, *, first: bool) -> list[AskPlan]:
        del first
        out = []
        for run, history in zip(runs, histories, strict=True):
            # the rolled series keeps the observed clock, so its anchor row IS the flown time
            origin = np.asarray(history.values[anchor], dtype=np.float64)[list(POSITION_IDX)]
            waypoints = truth_waypoints(run.series, float(history.times[anchor]), origin, self.count)
            out.append(AskPlan(arrival_s=truth_duration_s(run.series, self.a0),
                               token=waypoint_token(waypoints, self.count)))
        return out


class SegmentHeadWaypoints:
    """Two-tier v2 §5 (S3), protocol A: the SEGMENT-PLAN head's own coarse plan on the ask's rolled
    history — its waypoints every `PLAN_WAYPOINT_SEGMENT_S` after the ask relative to the flown row
    (the token shape `TruthWaypoints` hands over; a waypoint past the head's own arrival is
    invalid, as the truth's past its end), and its own arrival time as the first ask's horizon
    (the plan's span where it draws no arrival: the head sees no landing inside it). Beside the
    token every ask records e_plan — the head's waypoints against `truth_waypoints` at the same
    flown time and position, per waypoint and pooled over the waypoints both hold — a reading
    that enters no input."""

    source = "segment-head"
    reads_the_future = ("nothing but the landed runway: the waypoints and the arrival time are the segment-plan "
                        "head's own on the flown history (protocol A); the e_plan reading beside each ask compares "
                        "them with the truth's waypoints and enters no input")

    def __init__(self, head: Arm, batch_size: int, device: torch.device, config) -> None:
        self.head, self.batch_size, self.device, self.count = head, batch_size, device, plan_waypoint_count(config)
        if int(head.config.segment_plan_segments) < self.count:
            raise SystemExit(f"--plan-head {head.path}: draws {head.config.segment_plan_segments} segments, the tracker "
                             f"reads {self.count} waypoints")

    def at_ask(self, runs: list[FlightRun], histories: list[FlightSeries], anchor: int, *, first: bool) -> list[AskPlan]:
        plans = decode_series(self.head.model, histories, self.head.config, self.head.normalizer, anchor, self.device,
                              batch_size=self.batch_size)
        out = []
        for run, history, plan in zip(runs, histories, plans, strict=True):
            origin_time = float(history.times[anchor])
            origin = np.asarray(history.values[anchor], dtype=np.float64)[list(POSITION_IDX)]
            lead_s = plan.times_s[: self.count]
            valid = np.ones(self.count, dtype=bool) if not plan.arrives else lead_s <= plan.arrival_time_s + ROW_TOLERANCE_S
            waypoints = Waypoints(deltas=(plan.waypoints[: self.count] - origin) * valid[:, None], lead_s=lead_s,
                                  valid=valid.astype(np.float64))
            truth = truth_waypoints(run.series, origin_time, origin, self.count)
            both = valid & (truth.valid > 0)
            per = np.linalg.norm(waypoints.deltas - truth.deltas, axis=1)
            # `run.next_ask` IS the lockstep ask index of this ask (the filter that made the run
            # active), so e_plan lines up with `asks_e_m` under one-shot legs too
            run.asks_plan_e.append({
                "ask": run.next_ask,
                "per_waypoint_m": [float(per[k]) if both[k] else None for k in range(self.count)],
                "e_m": float(per[both].mean()) if both.any() else None,
            })
            if first:
                run.plan_arrives_at_a0, run.plan_arrival_probability_at_a0 = plan.arrives, plan.arrival_probability
            out.append(AskPlan(arrival_s=float(plan.arrival_time_s if plan.arrives else plan.times_s[-1]),
                               token=waypoint_token(waypoints, self.count)))
        return out


def plan_source_class(head: Arm | None, config):
    """WHICH source flies the plan: a head when one is given (a segment-plan head for a
    ``waypoints`` tracker, a plan head for a ``truth-next`` one — the head must hand over the
    token shape the tracker reads), else the truth's in that shape (a plan-free checkpoint
    takes the instruction source for its arrival time alone)."""
    if head is not None:
        segment = head.config.prediction_output == PREDICTION_SEGMENT_PLAN
        waypoints = config.plan_conditioning == PLAN_CONDITIONING_WAYPOINTS
        if segment != waypoints:
            raise SystemExit(f"--plan-head {head.path} hands over {'waypoints' if segment else 'the next instruction'}; "
                             f"the tracker reads plan_conditioning={config.plan_conditioning!r}")
        return SegmentHeadWaypoints if segment else HeadPlans
    return TruthWaypoints if config.plan_conditioning == PLAN_CONDITIONING_WAYPOINTS else TruthPlans


# ── one ask ────────────────────────────────────────────────────────────────────

def ask_floor_s(config, step_s: float) -> float:
    """The smallest CTA an ask hands over: an anchor the checkpoint trained at, and one whole step."""
    return max(float(config.random_train_anchor_min_future_s), float(step_s))


def ask_row(run: FlightRun, history: FlightSeries, anchor: int, config, plan_at: AskPlan, *, with_plan: bool,
            floor_s: float) -> dict[str, np.ndarray]:
    """The dynamics row of one ask: the anchor state of ``history`` and, as the checkpoint reads
    them, the arrival time (raised to ``floor_s``, `ask_floor_s`) and the plan token at that pose."""
    row = dynamics_arrays(
        history, anchor, parameterization=config.control_thrust_parameterization,
        condition_features=config.control_condition_features,
    )
    if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
        run.asks_below_floor += int(plan_at.arrival_s < floor_s)
        run.cta_raised_max_s = max(run.cta_raised_max_s, floor_s - plan_at.arrival_s)
        row["cta_s"] = np.array(max(plan_at.arrival_s, floor_s), dtype=np.float64)
    if config.plan_conditioning != PLAN_CONDITIONING_OFF:
        # the source's token, or the ABSENT token of this checkpoint's width
        row[PLAN_TOKEN_KEY] = (
            plan_at.token if with_plan else np.zeros(plan_token_width(config), dtype=np.float32)
        )
    return row


def fly_variant(arm: Arm, series: list[FlightSeries], variant: str, plan: LockstepPlan, device: torch.device,
                batch_size: int, head: Arm | None = None) -> list[FlightRun]:
    config = arm.config
    a0 = default_anchor(config)
    step_rows = int(round(plan.step_s / config.dt_s))
    skeletons = SkeletonCache()
    runs = [FlightRun(series=item, skeleton=skeletons.for_series(item)) for item in series]
    source = plan_source_class(head, config)
    plans = source(head, batch_size, device, config) if head is not None else source(runs, a0, config)
    floor_s = ask_floor_s(config, plan.step_s)
    ask = 0
    while any(run.ended is None for run in runs):
        active = [run for run in runs if run.ended is None and run.next_ask == ask]
        if not active:          # every flight still flying is inside a one-shot leg
            ask += 1
            continue
        anchor = a0 + ask * step_rows
        histories = [
            run.series if ask == 0 else rolled_series(run.series, a0, concatenate(run.legs, a0, 0.0), anchor, config.dt_s)
            for run in active
        ]
        asked = plans.at_ask(active, histories, anchor, first=ask == 0)
        if ask == 0:
            for run, plan_at in zip(active, asked, strict=True):
                run.horizon_s = closing_horizon_s(plan_at.arrival_s)
        rows = [
            ask_row(run, history, anchor, config, plan_at, with_plan=variant != VARIANT_NO_PLAN, floor_s=floor_s)
            for run, history, plan_at in zip(active, histories, asked, strict=True)
        ]
        forecasts: list[Forecast] = []
        for start in range(0, len(active), batch_size):
            chunk = rows[start : start + batch_size]
            dynamics = {name: torch.from_numpy(np.stack([row[name] for row in chunk])).to(device) for name in chunk[0]}
            forecasts.extend(forecast_control_batch(
                arm.model, histories[start : start + batch_size], config, arm.normalizer, anchor, device,
                dynamics=dynamics,
            ))
        for run, history, forecast in zip(active, histories, forecasts, strict=True):
            steps = (int((forecast.final_time_s + ROW_TOLERANCE_S) // plan.step_s)
                     if variant == VARIANT_ONE_SHOT and ask == 0 else 1)
            fly_leg(run, history, forecast, steps=steps, step_s=plan.step_s, ask=ask, a0=a0)
        print(f"    {variant} ask {ask}: {len(active)} flights at anchor {anchor}, "
              f"{sum(1 for run in runs if run.ended is None)} continue", flush=True)
        ask += 1
    return runs


def fly_leg(run: FlightRun, history: FlightSeries, forecast: Forecast, *, steps: int, step_s: float, ask: int,
            a0: int) -> None:
    """Fly ``steps`` whole steps of this ask's ``forecast`` (a forecast shorter than one step — only
    without a given CTA or a fixed horizon — is flown whole and ends the flight), and end the flight
    where it crosses the threshold on the final or reaches its horizon, whichever comes first. The
    next ask is ``ask + steps``. ``run.asks`` counts the asks that flew a leg: an ask whose flight's
    horizon falls before the forecast's first row ends the flight there and flies nothing. Every leg
    flown records the displacement from the truth at its END (`run.asks_e`; None past the truth's
    end) — the step error of that ask, which a drift reading lines up along the flight."""
    crossed = cut_at_threshold_crossing(forecast, history)
    remaining_s = run.horizon_s - run.flown_s
    short = forecast.final_time_s < step_s - ROW_TOLERANCE_S
    span_s = forecast.final_time_s if short else steps * step_s
    if crossed.truncated_at_threshold and crossed.final_time_s <= min(span_s, remaining_s) + ROW_TOLERANCE_S:
        leg, run.ended, run.truncated = crossed, ENDED_CROSSED, True
    elif remaining_s <= span_s + ROW_TOLERANCE_S:
        offsets = np.cumsum(forecast.sample_durations_s)
        rows = int(np.searchsorted(offsets, remaining_s + ROW_TOLERANCE_S, side="right"))
        leg, run.ended = (cut_rows(forecast, rows) if rows else None), ENDED_HORIZON
    elif short:
        leg, run.ended = crossed, ENDED_FORECAST
    else:
        leg, run.next_ask = cut_at_lead(forecast, span_s), ask + steps
    if leg is not None:
        run.legs.append(leg)
        run.asks += 1
        # a0's observed row stands in as the leg's t=0 row; the reading is at the leg's END,
        # which lies on the leg's own rows, so a rolled ask reads its flown position there
        run.asks_e.append({
            "ask": ask, "lead_s": float(np.sum(leg.sample_durations_s)),
            "e_m": displacement_at(run.series, leg, a0, float(leg.times[-1])),
        })


def whole_forecast(run: FlightRun, a0: int) -> Forecast:
    """The flight's legs as one forecast from ``a0``: its predicted end is the last ask's."""
    last = run.legs[-1]
    before = float(sum(np.sum(leg.sample_durations_s) for leg in run.legs[:-1]))
    whole = concatenate(run.legs, a0, before + float(last.predicted_final_time_s))
    return replace(whole, truncated_at_threshold=run.truncated, horizon_capped=run.ended == ENDED_HORIZON)


# ── the readout ────────────────────────────────────────────────────────────────

#: The plan oracle's reference verdicts a stratum block reports as shares (`plan_oracle.summarize`'s).
REFERENCE_SHARES = ("fully_flyable", "established", "lateral_violation", "glidepath_violation", "floor_violation")


def flight_row(run: FlightRun, forecast: Forecast, a0: int, points: int, difficulty: dict) -> tuple[dict, dict]:
    metrics = observed_series_metrics(run.series, forecast, points=points)
    geometry = forecast_geometry(run.series, forecast)
    origin = float(run.series.times[a0])
    row = {
        "difficulty": difficulty,
        "reference": reference_verdicts(run.series, forecast, run.skeleton),
        "asks": run.asks, "asks_below_floor": run.asks_below_floor, "cta_raised_max_s": run.cta_raised_max_s,
        "ended": run.ended,
        "truncated_at_threshold": run.truncated,
        "ade_m": float(metrics["ade_m"]), "fde_m": float(metrics["fde_m"]),
        "final_time_error_s": float(metrics["final_time_error_s"]),
        "chamfer_m": float(geometry["chamfer_m"]), "frechet_m": float(geometry["frechet_m"]),
        "at": {f"{lead:g}": displacement_at(run.series, forecast, a0, origin + lead) for lead in LEADS_S},
        "asks_e_m": run.asks_e,
        "asks_plan_e_m": run.asks_plan_e,
        # the first ask's horizon came from the plan's SPAN (no arrival drawn) — a cap, stated
        "horizon_from_plan_span": run.plan_arrives_at_a0 is False,
        "plan_arrival_probability_at_a0": run.plan_arrival_probability_at_a0,
    }
    return row, metrics


def _p50(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def e_by_ask(cells: list[dict]) -> dict[str, dict]:
    """The per-ask step error pooled over ``cells``: ``{ask: {n, lead_s, p50}}`` over the legs whose
    end the truth still covers — the drift reading's raw curve (two-tier v2 §3)."""
    by_ask: dict[int, list[dict]] = {}
    for cell in cells:
        for entry in cell["asks_e_m"]:
            if entry["e_m"] is not None:
                by_ask.setdefault(int(entry["ask"]), []).append(entry)
    return {
        str(ask): {"n": len(entries), "lead_s": _p50([e["lead_s"] for e in entries]), "p50": _p50([e["e_m"] for e in entries])}
        for ask, entries in sorted(by_ask.items())
    }


def plan_e_by_ask(cells: list[dict]) -> dict[str, dict]:
    """e_plan pooled over ``cells`` per ask: ``{ask: {n, p50, per_waypoint_p50, per_waypoint_n}}`` —
    ``n`` the asks whose head and truth waypoints overlap anywhere, ``p50`` over their pooled
    ``e_m`` (a mean over 1 or 2 waypoints, so the per-waypoint curve is the honest reading), each
    waypoint's p50 over the ``per_waypoint_n`` asks both sides hold it. Empty under a source that
    draws no plan — and under the L1 campaign's v3 artifacts, whose rows carry no `asks_plan_e_m`
    (the gates read those through this function)."""
    by_ask: dict[int, list[dict]] = {}
    for cell in cells:
        for entry in cell.get("asks_plan_e_m", ()):
            if entry["e_m"] is not None:
                by_ask.setdefault(int(entry["ask"]), []).append(entry)
    out = {}
    for ask, entries in sorted(by_ask.items()):
        count = len(entries[0]["per_waypoint_m"])
        per = [[e["per_waypoint_m"][k] for e in entries if e["per_waypoint_m"][k] is not None] for k in range(count)]
        out[str(ask)] = {
            "n": len(entries), "p50": _p50([e["e_m"] for e in entries]),
            "per_waypoint_p50": [_p50(values) for values in per],
            "per_waypoint_n": [len(values) for values in per],
        }
    return out


def stratum_block(rows: dict[str, dict], keys: list[str]) -> dict:
    cells = [rows[key] for key in keys]
    return {
        "e_by_ask_p50_m": e_by_ask(cells),
        "plan_e_by_ask_p50_m": plan_e_by_ask(cells),
        "n": len(cells),
        "ade_mean_m": float(np.mean([c["ade_m"] for c in cells])) if cells else None,
        "ade_p50_m": _p50([c["ade_m"] for c in cells]),
        "fde_p50_m": _p50([c["fde_m"] for c in cells]),
        "chamfer_p50_m": _p50([c["chamfer_m"] for c in cells]),
        "frechet_p50_m": _p50([c["frechet_m"] for c in cells]),
        "abs_dt_p50_s": _p50([abs(c["final_time_error_s"]) for c in cells]),
        "asks_mean": float(np.mean([c["asks"] for c in cells])) if cells else None,
        "flights_with_an_ask_below_floor": sum(1 for c in cells if c["asks_below_floor"]),
        # how far the raise reached: a flight told 30 s at 29 s to go is not one told 30 s at 2 s
        "cta_raised_max_p50_s": _p50([c["cta_raised_max_s"] for c in cells if c["asks_below_floor"]]),
        "truncated_at_threshold": sum(1 for c in cells if c["truncated_at_threshold"]),
        # under a segment-plan head: flights whose first ask drew no arrival, so their horizon is the plan's span
        "horizon_from_plan_span": sum(1 for c in cells if c.get("horizon_from_plan_span")),
        "ended": {name: sum(1 for c in cells if c["ended"] == name) for name in sorted({c["ended"] for c in cells})},
        "at_lead_p50_m": {
            lead: {"n": len(v), "p50": _p50(v)}
            for lead in (f"{h:g}" for h in LEADS_S)
            for v in [[c["at"][lead] for c in cells if c["at"][lead] is not None]]
        },
        **{
            f"{name}_share": float(np.mean([c["reference"][name] for c in cells])) if cells else None
            for name in REFERENCE_SHARES
        },
    }


def _fmt(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    plan = payload["plan"]
    lines = [
        f"Plan-given lockstep, plan source {payload['plan_source']} — reads {payload['reads_the_future']}; "
        f"re-asked every {plan['step_s']:g} s, ended at a crossing on the final or at {plan['horizon']}, "
        f"split {plan['split']}"
        + (f", limit {plan['limit']}" if plan["limit"] else ""),
        "ADE/FDE on the whole record (export's accounting), every variant cut at its crossing on the final; "
        "chamfer / Fréchet time-free; disp p50 at leads from a0; below-floor = flights with an ask whose arrival "
        "time was raised to the ask floor max(the training future floor, one step).",
    ]
    if plan.get("command_hook"):
        lines.append(f"command hook {plan['command_hook']} on every arm (the delivery-form reading; a gate reads the hook-free run).")
    for label, arm in payload["checkpoints"].items():
        lines.append("")
        excluded = arm["anchor_floor_excluded"]["count"]
        note = (f"; {excluded} of {arm['split_flights']} split flights cannot host anchor {arm['anchor']} — excluded, counted"
                if excluded else "")
        lines.append(f"── {label} — {arm['checkpoint']} ({arm['flights']} flights, a0={arm['anchor']}{note})")
        for variant, block in arm["variants"].items():
            for stratum, cell in block["strata"].items():
                leads = " ".join(f"{k}s {_fmt(v['p50'])}" for k, v in cell["at_lead_p50_m"].items())
                lines.append(
                    f"   {variant:<17s} {stratum[:34]:<34s} n={cell['n']:>4d} ADE {_fmt(cell['ade_mean_m']):>5}/"
                    f"{_fmt(cell['ade_p50_m']):>5} FDE50 {_fmt(cell['fde_p50_m']):>5} cham50 {_fmt(cell['chamfer_p50_m']):>5} "
                    f"Fr50 {_fmt(cell['frechet_p50_m']):>5} |dt|50 {_fmt(cell['abs_dt_p50_s'], 1):>5} asks {_fmt(cell['asks_mean'], 1)} "
                    f"ended {cell['ended']} cut {cell['truncated_at_threshold']} below-floor {cell['flights_with_an_ask_below_floor']} "
                    f"flyable {_fmt(cell['fully_flyable_share'], 3)} established {_fmt(cell['established_share'], 3)} | {leads}"
                )
                steps = " ".join(f"k{ask} {_fmt(v['p50'])}({v['n']})" for ask, v in cell["e_by_ask_p50_m"].items())
                lines.append(f"   {'':<17s} {'step error p50 by ask':<34s} {steps}")
                if cell["plan_e_by_ask_p50_m"]:
                    plan_steps = " ".join(
                        f"k{ask} {_fmt(v['p50'])}[{'/'.join(f'{_fmt(w)}({m})' for w, m in zip(v['per_waypoint_p50'], v['per_waypoint_n']))}]({v['n']})"
                        for ask, v in cell["plan_e_by_ask_p50_m"].items()
                    )
                    lines.append(f"   {'':<17s} {'plan error p50 by ask [per waypoint(n)]':<34s} {plan_steps}"
                                 f" | horizon from the plan's span {cell['horizon_from_plan_span']}")
    return "\n".join(lines) + "\n"


def check_arm(arm: Arm, plan: LockstepPlan) -> None:
    config = arm.config
    if config.prediction_output != PREDICTION_CONTROL:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} flies a CONTROL checkpoint, "
                         f"this one predicts {config.prediction_output!r}")
    if config.control_command_hook != CONTROL_HOOK_OFF and plan.command_hook is None:
        raise SystemExit(f"{arm.label} ({arm.path}): trained under command hook {config.control_command_hook!r}; "
                         f"{INSTRUMENT} flies the network's own schedule unless --command-hook names one")
    if config.control_horizon_s and config.control_horizon_s < plan.step_s - ROW_TOLERANCE_S:
        raise SystemExit(f"{arm.label} ({arm.path}): a {config.control_horizon_s:g} s fixed horizon does not cover "
                         f"one {plan.step_s:g} s step; every leg would end the flight")
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
                records_root: Path | None, campaign: str, head: Arm | None, *, floor_excluded: int = 0) -> dict:
    started = time.time()
    config = arm.config
    a0 = default_anchor(config)
    batch_size = plan.batch_size or config.batch_size
    keys = [item.dataset_id for item in series]
    # the strata fixed at a0, their covariates kept per flight
    difficulty = difficulty_at_anchor(series, keys, anchor=a0)
    masks = strata_masks(difficulty, keys)
    print(f"{arm.label}: {len(series)} flights, a0 {a0}, every {plan.step_s:g} s, variants {', '.join(plan.variants)}", flush=True)
    variants = {}
    record_dirs = {}
    for variant in plan.variants:
        runs = fly_variant(arm, series, variant, plan, device, batch_size, head)
        rows: dict[str, dict] = {}
        records, metrics = [], []
        for index, run in enumerate(runs):
            forecast = whole_forecast(run, a0)
            row, flight_metrics = flight_row(
                run, forecast, a0, config.validation_common_grid_points, difficulty[run.series.dataset_id]
            )
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
                    "step_s": plan.step_s, "horizon": HORIZON_RULE, "ask_floor_s": ask_floor_s(config, plan.step_s),
                    "anchor": a0, "anchor_floor_index": plan.anchor_floor_index, "command_hook": plan.command_hook,
                    "plan_conditioning": config.plan_conditioning, "cta_conditioning": config.cta_conditioning,
                    "control_horizon_s": config.control_horizon_s,
                    "plan_source": plan_source(head, config),
                    "plan_head_sha256": None if head is None else file_sha256(head.path),
                    "reads_the_future": plan_source_class(head, config).reads_the_future,
                    "split": plan.split, "limit": plan.limit,
                    "split_flights": len(arm.payload["split"][plan.split]), "records": len(records),
                    "anchor_floor_excluded": floor_excluded,
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


def at_anchor_floor(loaded: list[Arm], floor: int | None) -> list[Arm]:
    """Every tracker read at anchor ``floor`` (`--anchor-floor-index`): its config's floor replaced,
    refused below the tracker's own fixed anchor — a tracker is read at or after the anchor it
    trained at, never before it."""
    if floor is None:
        return loaded
    for arm in loaded:
        if floor < default_anchor(arm.config):
            raise SystemExit(f"--anchor-floor-index {floor} is before {arm.label}'s own fixed anchor "
                             f"{default_anchor(arm.config)}; a tracker is read at or after the anchor it trained at")
    return [replace(arm, config=replace(arm.config, anchor_floor_index=floor)) for arm in loaded]


def floor_cohort(own: Arm, floored: Arm, grid: Grid, plan: LockstepPlan) -> tuple[list[FlightSeries], dict]:
    """The arm's cohort for this run and the coverage block the artifact carries. The whole split
    must rebuild at the arm's OWN anchor (`cohort_series`, as every readout). Under
    ``--anchor-floor-index`` a flight that rebuilds there but cannot host the floor with the
    tracker's horizon after it is EXCLUDED and counted (`anchor_floor_excluded`: count, keys) —
    the artifact and the gate say the coverage; a silent subset would be a different cohort.
    Only the floor excludes: a flight the data plane drops still refuses the run."""
    series = cohort_series(own, grid)
    if plan.anchor_floor_index is None:
        return series, {"count": 0, "flight_keys": []}
    kept = [item for item in series if len(window_anchors(item, floored.config)) > 0]
    missing = [item.dataset_id for item in series if len(window_anchors(item, floored.config)) == 0]
    if missing:
        print(f"{own.label}: {len(missing)} of {len(series)} split flights cannot host anchor "
              f"{plan.anchor_floor_index} with the horizon after it — excluded, counted (first {missing[0]!r})",
              flush=True)
    return kept, {"count": len(missing), "flight_keys": missing}


def plan_source(head: Arm | None, config) -> str:
    source = plan_source_class(head, config)
    return f"{source.source}:{head.path}" if head is not None else source.source


def load_head(path: Path, grid: Grid, device: torch.device) -> Arm:
    head = load_arm("plan-head", path if path.is_absolute() else REPO_ROOT / path, grid, device,
                    instrument=f"{INSTRUMENT}'s plan head")
    if head.config.prediction_output not in (PREDICTION_PLAN, PREDICTION_SEGMENT_PLAN):
        raise SystemExit(f"--plan-head {path}: predicts {head.config.prediction_output!r}, not a plan")
    return head


def check_head(head: Arm, arms: list[Arm], grid: Grid) -> None:
    """Before any track is read: the head must read the series each arm's cohort is built under
    (the same dt, chart and channels, a lookback that fits before the arm's anchor, its airports
    covered), and must not have trained on a flight it is asked about — the cohort is the arm's
    own split keys (`cohort_series` refuses a flight it cannot rebuild), compared by key."""
    trained = set(head.payload["split"]["train"])
    for arm in arms:
        for name in ("dt_s", "coordinate_frame", "channels"):
            if getattr(head.config, name) != getattr(arm.config, name):
                raise SystemExit(f"--plan-head {head.path}: {name}={getattr(head.config, name)!r}, the tracker "
                                 f"{arm.label} reads {getattr(arm.config, name)!r}")
        if head.config.seq_len - 1 > default_anchor(arm.config):
            raise SystemExit(f"--plan-head {head.path}: a {head.config.seq_len}-sample lookback does not fit before "
                             f"{arm.label}'s anchor {default_anchor(arm.config)}")
        foreign = sorted(set(arm.airports) - set(head.airports))
        if foreign:
            raise SystemExit(f"--plan-head {head.path}: trained on {sorted(head.airports)}, asked about {foreign}")
        keys = arm.payload["split"][grid.split]
        seen = [key for key in (keys[: grid.limit] if grid.limit else keys) if key in trained]
        if seen:
            raise SystemExit(
                f"--plan-head {head.path}: {len(seen)} flights {arm.label} is asked about are in the head's TRAIN "
                f"split (first: {seen[0]!r}); its plans there are in-sample"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--split", choices=("val", "train", FORBIDDEN_SPLIT), default="val")
    parser.add_argument("--step-s", type=float, default=DEFAULT_STEP_S,
                        help=f"the re-ask period (default {DEFAULT_STEP_S:g}, the plan lockstep's)")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"comma-separated subset of {VARIANTS} (default: all)")
    parser.add_argument("--limit", type=int, default=0, help="first N flights of the split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--write-records", action="store_true")
    parser.add_argument("--plan-head", type=Path, default=None, metavar="PATH",
                        help="a plan checkpoint (T2) or a segment-plan checkpoint (two-tier E2E) whose own prediction "
                             "on each ask's history is the plan and the arrival time (protocol A); default: the "
                             "truth's (protocol C)")
    parser.add_argument("--anchor-floor-index", type=int, default=None, metavar="N",
                        help="read every tracker at anchor N instead of its own fixed anchor (a head whose lookback "
                             "does not fit before it needs one); refused below the tracker's own anchor")
    parser.add_argument("--command-hook", choices=CONTROL_HOOKS_AVAILABLE, default=None,
                        help="fly every arm under this command hook (what it means in `predict`); the "
                             "delivery-form reading beside the hook-free gate run (two-tier v2 §3)")
    parser.add_argument("--hook-saturation", choices=HOOK_SATURATIONS, default=HOOK_SATURATION_SOFT,
                        help=f"the hook's saturation (default {HOOK_SATURATION_SOFT}); read with --command-hook only")
    parser.add_argument("--device", default="auto")
    return parser


def parse_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> LockstepPlan:
    if args.split == FORBIDDEN_SPLIT:
        parser.error(f"the {FORBIDDEN_SPLIT} split is sealed")
    if not math.isfinite(args.step_s) or args.step_s <= 0.0:
        parser.error("--step-s must be positive")
    variants = tuple(token.strip() for token in args.variants.split(",") if token.strip())
    unknown = [name for name in variants if name not in VARIANTS]
    if not variants or unknown or len(set(variants)) != len(variants):
        parser.error(f"--variants takes a subset of {VARIANTS} without repeats, got {args.variants!r}")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.anchor_floor_index is not None and args.anchor_floor_index < 0:
        parser.error("--anchor-floor-index is a sample index")
    return LockstepPlan(split=args.split, step_s=float(args.step_s),
                        variants=variants, limit=int(args.limit), batch_size=args.batch_size,
                        write_records=bool(args.write_records),
                        command_hook=None if args.command_hook is None else f"{args.command_hook}/{args.hook_saturation}",
                        anchor_floor_index=args.anchor_floor_index)


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
        load_arm(label, path, grid, device, instrument=INSTRUMENT, refuse_cta_given=False, refuse_plan=False,
                 command_hook=args.command_hook,
                 hook_saturation=None if args.command_hook is None else args.hook_saturation)
        for label, path in arms.items()
    ]
    own_loaded = loaded
    loaded = at_anchor_floor(loaded, plan.anchor_floor_index)
    for arm in loaded:
        check_arm(arm, plan)
    head = None if args.plan_head is None else load_head(args.plan_head, grid, device)
    if head is not None:
        check_head(head, loaded, grid)
    # one plan SOURCE per artifact: the gates key on it, so two token shapes cannot share one
    sources = {plan_source(head, arm.config) for arm in loaded}
    if len(sources) != 1:
        raise SystemExit(f"the arms read different plan sources {sorted(sources)}; one artifact holds one source")
    source_config = loaded[0].config
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    try:
        records_root = staged / RECORDS_DIR if plan.write_records else None
        checkpoints = {}
        for own, arm in zip(own_loaded, loaded, strict=True):
            series, excluded = floor_cohort(own, arm, grid, plan)
            checkpoints[arm.label] = {
                **measure_arm(arm, series, plan, device, records_root, out.name, head, floor_excluded=excluded["count"]),
                "anchor_floor_excluded": excluded,
            }
        payload = {
            "schema": RESULT_SCHEMA,
            "plan": {"split": plan.split, "step_s": plan.step_s, "horizon": HORIZON_RULE,
                     "variants": list(plan.variants), "limit": plan.limit, "write_records": plan.write_records,
                     "command_hook": plan.command_hook, "anchor_floor_index": plan.anchor_floor_index},
            "plan_source": plan_source(head, source_config),
            "plan_head_sha256": None if head is None else file_sha256(head.path),
            "reads_the_future": plan_source_class(head, source_config).reads_the_future,
            "strata_anchor": "a0 = default_anchor (strata_fixed_at_anchor)",
            "leads_s": list(LEADS_S),
            "device": str(device),
            "checkpoints": checkpoints,
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
