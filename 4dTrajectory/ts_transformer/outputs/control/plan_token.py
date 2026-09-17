"""The plan token: what the control decoder is told about the plan it flies, one fused token.

Two shapes, selected by ``plan_conditioning`` (`config.PLAN_CONDITIONINGS`), one key
(`PLAN_TOKEN_KEY`) and one width per shape (`plan_token_width`):

* ``truth-next`` (two-tier feasibility T1, `docs/2026-09-16_two_tier_transformer_feasibility.zh.md`
  §10.3): the plan path's target vector at the anchor (`outputs.plan.labels.TARGETS`: the
  operating parameters and the next instruction in runway axes about the anchor) — the one
  definition the plan head is trained on. ``[values / SCALE_VECTOR · valid, valid,
  next_is_join, present]``.
* ``waypoints`` (two-tier v2, `docs/2026-09-17_two_tier_plan_v2.zh.md` §3): the coarse plan's
  next K waypoints — the position every `PLAN_WAYPOINT_SEGMENT_S` after the ask, relative to
  the CURRENT position in the chart (Δe, Δn, Δu) with the seconds until it — K = the fixed
  horizon in coarse segments (`plan_waypoint_count`). ``[deltas / scale · valid (K×3),
  lead / scale · valid (K), valid (K), present]``; a waypoint the plan does not reach is 0
  with its valid bit 0.

Both read the TRUTH in training and in the protocol-C lockstep (`training_plan_token` here,
`tracker_lockstep`'s sources): the oracle a plan-given tracker is judged under, never a
prediction result, and the run name says which (``plan=truth-next`` / ``plan=waypoints``).
The ABSENT token (no plan, or the plan dropped in training) is all zeros — ``present`` 0 is
what separates "no plan" from a plan whose entries all happen to be 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ts_transformer.config import (
    PLAN_CONDITIONING_TRUTH_NEXT,
    PLAN_CONDITIONING_WAYPOINTS,
    PLAN_WAYPOINT_SEGMENT_S,
    TSConfig,
    plan_waypoint_count,
)
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.labels import SCALE_VECTOR, TARGETS, PlanTargets, targets_from_labels
from ts_transformer.outputs.plan.skeleton import SkeletonCache

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The dynamics-context key the token rides under (beside `condition` and `cta_s`).
PLAN_TOKEN_KEY = "plan_token"
#: The `truth-next` token: values, valid bits, the no-fix flag, the present flag.
PLAN_TOKEN_WIDTH = 2 * len(TARGETS) + 2
#: The `waypoints` token's scales — a minute of approach flying laterally, its descent
#: vertically, and the horizon's order in time — so every entry is O(1).
WAYPOINT_DELTA_SCALE_M = np.array([5000.0, 5000.0, 300.0], dtype=np.float32)
WAYPOINT_LEAD_SCALE_S = 60.0


def plan_token_width(config: TSConfig) -> int:
    """The token's width under this run's ``plan_conditioning`` (the decoder's input size)."""
    if config.plan_conditioning == PLAN_CONDITIONING_WAYPOINTS:
        return 5 * plan_waypoint_count(config) + 1
    return PLAN_TOKEN_WIDTH


# ── truth-next ───────────────────────────────────────────────────────────────

def plan_token(targets: PlanTargets | None) -> np.ndarray:
    """The token of one target vector; None is the absent token."""
    token = np.zeros(PLAN_TOKEN_WIDTH, dtype=np.float32)
    if targets is None:
        return token
    width = len(TARGETS)
    valid = np.asarray(targets.valid, dtype=np.float32)
    token[:width] = np.asarray(targets.values, dtype=np.float32) / SCALE_VECTOR * valid
    token[width : 2 * width] = valid
    token[2 * width] = float(targets.next_is_join)
    token[2 * width + 1] = 1.0
    return token


def truth_plan_token(series: FlightSeries, anchor: int, skeletons: SkeletonCache) -> np.ndarray:
    """The truth's plan read at an observed ``anchor`` — the plan head's label there."""
    skeleton = skeletons.for_series(series)
    return plan_token(targets_from_labels(extract_plan(series, anchor, skeleton), series, anchor, skeleton))


# ── waypoints ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Waypoints:
    """K coarse-plan waypoints relative to the current position: ``deltas`` ``[K, 3]`` (Δe, Δn,
    Δu, metres in the chart), ``lead_s`` ``[K]`` (seconds until each) and ``valid`` ``[K]``
    (0 where the plan ends before the point; its delta is then 0 too)."""

    deltas: np.ndarray
    lead_s: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.valid)
        if self.deltas.shape != (count, 3) or self.lead_s.shape != (count,):
            raise ValueError(
                f"waypoints are [K,3] deltas with [K] leads and [K] valid bits, got "
                f"{self.deltas.shape}, {self.lead_s.shape}, {self.valid.shape}"
            )


def waypoint_token(waypoints: Waypoints | None, count: int) -> np.ndarray:
    """The token of ``count`` waypoints; None is the absent token of that width."""
    token = np.zeros(5 * count + 1, dtype=np.float32)
    if waypoints is None:
        return token
    if len(waypoints.valid) != count:
        raise ValueError(f"the token carries {count} waypoints, got {len(waypoints.valid)}")
    valid = np.asarray(waypoints.valid, dtype=np.float32)
    token[: 3 * count] = (
        np.asarray(waypoints.deltas, dtype=np.float32) / WAYPOINT_DELTA_SCALE_M * valid[:, None]
    ).reshape(-1)
    token[3 * count : 4 * count] = np.asarray(waypoints.lead_s, dtype=np.float32) / WAYPOINT_LEAD_SCALE_S * valid
    token[4 * count : 5 * count] = valid
    token[5 * count] = 1.0
    return token


def truth_waypoints(
    series: FlightSeries, origin_time_s: float, origin_position: np.ndarray, count: int
) -> Waypoints:
    """The TRUTH's coarse plan seen from ``origin``: its position ``k · PLAN_WAYPOINT_SEGMENT_S``
    after ``origin_time_s`` for k = 1..count, relative to ``origin_position`` (the current
    (e, n, u) in the chart); invalid where the truth ends before the point. Reads the future.

    ``origin`` is the observed anchor row in training, the FLOWN position at the flown time in
    the lockstep — the one function both read, so a re-ask is told the same plan a training
    row would be at that instant."""
    lead_s = PLAN_WAYPOINT_SEGMENT_S * np.arange(1, count + 1, dtype=np.float64)
    times = origin_time_s + lead_s
    valid = times <= float(series.supervision_times[-1]) + 1e-9
    positions = np.column_stack([
        np.interp(times, series.supervision_times, series.supervision_values[:, channel])
        for channel in POSITION_IDX
    ])
    deltas = (positions - np.asarray(origin_position, dtype=np.float64)) * valid[:, None]
    return Waypoints(deltas=deltas, lead_s=lead_s, valid=valid.astype(np.float64))


def truth_waypoint_token(series: FlightSeries, anchor: int, config: TSConfig) -> np.ndarray:
    """The truth's waypoints read at an observed ``anchor``, relative to its own row."""
    count = plan_waypoint_count(config)
    origin = np.asarray(series.values[anchor], dtype=np.float64)[list(POSITION_IDX)]
    return waypoint_token(truth_waypoints(series, float(series.times[anchor]), origin, count), count)


# ── the one door a training row and a predict row go through ────────────────

_TRUTH_TOKENS = {
    PLAN_CONDITIONING_TRUTH_NEXT: lambda series, anchor, config, skeletons: truth_plan_token(series, anchor, skeletons),
    PLAN_CONDITIONING_WAYPOINTS: lambda series, anchor, config, skeletons: truth_waypoint_token(series, anchor, config),
}


def training_plan_token(
    series: FlightSeries, anchor: int, config: TSConfig, skeletons: SkeletonCache
) -> np.ndarray:
    """The TRUTH's token at an observed ``anchor`` under this run's ``plan_conditioning`` — what
    a training row and `predict` hand the decoder (both read the future)."""
    return _TRUTH_TOKENS[config.plan_conditioning](series, anchor, config, skeletons)


def probe_plan_token(config: TSConfig) -> np.ndarray:
    """A PRESENT token of this run's width with nothing defined in it — what the batch-size
    probe carries, built by the token builders so the probe cannot restate a width."""
    if config.plan_conditioning == PLAN_CONDITIONING_WAYPOINTS:
        count = plan_waypoint_count(config)
        return waypoint_token(Waypoints(
            deltas=np.zeros((count, 3)), lead_s=np.zeros(count), valid=np.zeros(count),
        ), count)
    return plan_token(PlanTargets(
        values=np.zeros(len(TARGETS), dtype=np.float32), valid=np.zeros(len(TARGETS), dtype=np.float32),
        next_is_join=False,
    ))


__all__ = [
    "PLAN_TOKEN_KEY", "PLAN_TOKEN_WIDTH", "WAYPOINT_DELTA_SCALE_M", "WAYPOINT_LEAD_SCALE_S",
    "Waypoints", "plan_token", "plan_token_width", "probe_plan_token", "training_plan_token",
    "truth_plan_token", "truth_waypoint_token", "truth_waypoints", "waypoint_token",
]
