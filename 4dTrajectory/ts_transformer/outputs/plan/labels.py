"""The plan head's targets (design v5 §3, §6): what one training sample supervises, and
what a prediction is flown as.

Per drawn anchor the extractors read the flight's plan labels; this module turns them into
the fixed-width target vector the head regresses (`TARGETS`: the operating parameters,
then the NEXT instruction), the validity mask (a censored parameter never enters the loss)
and the next-is-join flag — and turns a prediction back into a `PlanOrder`: the operating
parameters and the one instruction the guidance flies next.

The instruction is a VECTOR (`Instruction`): the fix where the turn happens, the heading to
fly after it, the speed and height to be at there, and the remaining path there. The truth's
own instructions (`truth_instructions`) are the oracle's and the supervision's alike, so
the head is trained on exactly what the oracle flew.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.geometry.dubins import chart_from_axes_np
from ts_transformer.outputs.plan.extractors import PlanLabels
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton

#: The operating parameters (§3a, plus the join distance and the remaining path — the
#: schedule's coordinate, which the guidance needs and a prediction must therefore carry).
OPERATING = ("T_s", "V_mid_mps", "d_decel_m", "V_final_mps", "h_capture_m", "d_join_m", "remaining_m")
#: The next instruction, in runway axes about the anchor: the fix (ahead along the course,
#: across it), the heading on after it relative to the course (as a unit vector), the
#: speed, the remaining path and the height at it.
INSTRUCTION = (
    "next_ahead_m", "next_across_m", "next_heading_cos", "next_heading_sin",
    "next_speed_mps", "next_remaining_m", "next_height_m",
)
TARGETS = OPERATING + INSTRUCTION
TIME_SCALE_S = 100.0
SPEED_SCALE_MPS = 50.0
DISTANCE_SCALE_M = 10_000.0
HEIGHT_SCALE_M = 500.0
#: The loss reads every target at its scale (an L1 residual in these units).
SCALES: dict[str, float] = {
    "T_s": TIME_SCALE_S,
    "V_mid_mps": SPEED_SCALE_MPS, "V_final_mps": SPEED_SCALE_MPS, "next_speed_mps": SPEED_SCALE_MPS,
    "d_decel_m": DISTANCE_SCALE_M, "d_join_m": DISTANCE_SCALE_M, "remaining_m": DISTANCE_SCALE_M,
    "next_ahead_m": DISTANCE_SCALE_M, "next_across_m": DISTANCE_SCALE_M, "next_remaining_m": DISTANCE_SCALE_M,
    "h_capture_m": HEIGHT_SCALE_M, "next_height_m": HEIGHT_SCALE_M,
    "next_heading_cos": 1.0, "next_heading_sin": 1.0,
}
SCALE_VECTOR = np.array([SCALES[name] for name in TARGETS], dtype=np.float32)
#: The ranges a prediction is clamped into before it is flown (design §3: the model never
#: plans an infeasible flight; every clamp is recorded).
T_MIN_S = 30.0
SPEED_MIN_MPS = 40.0
SPEED_MAX_MPS = 140.0

#: The context keys a plan batch carries (`PlanContext.row`).
CONTEXT_TARGETS = "plan_targets"
CONTEXT_VALID = "plan_valid"
CONTEXT_NEXT_IS_JOIN = "plan_next_is_join"
CONTEXT_SERIES = "plan_series_index"
PLAN_CONTEXT_KEYS = (CONTEXT_TARGETS, CONTEXT_VALID, CONTEXT_NEXT_IS_JOIN, CONTEXT_SERIES)

#: A last fix within this of the join IS the join (the turn onto the final): its heading
#: on is the course.
FIX_AT_JOIN_M = 2_000.0


@dataclass(frozen=True)
class Instruction:
    """One radar instruction as the plan carries it — a VECTOR: the fix where the turn
    happens, the heading to fly after it (ATC: "turn left heading 270"), the ground speed
    and the height to be at there ("reduce to 210", "descend to 4000"), and the remaining
    path there (the schedule's coordinate)."""

    fix_e: float
    fix_n: float
    heading_out_rad: float
    speed_mps: float
    remaining_m: float
    height_m: float


def join_point(skeleton: RunwaySkeleton, d_join: float) -> np.ndarray:
    """The join point in the chart: ``d_join`` of distance to go on the centreline."""
    e, n = chart_from_axes_np(np.array([d_join]), np.array([0.0]), skeleton.course_rad)
    return np.array([float(e[0]) + skeleton.target_e, float(n[0]) + skeleton.target_n])


def truth_instructions(labels: PlanLabels, skeleton: RunwaySkeleton) -> list[Instruction]:
    """The truth's own instructions from a label set: its fixes with their speeds, each
    heading on to the next fix — the last toward the join, or down the course where the
    last fix is the join itself (the turn onto the final)."""
    if not labels.waypoints:
        return []
    d_join = labels.remaining_path_at_anchor_m if labels.d_join_m is None else labels.d_join_m
    join = join_point(skeleton, max(float(d_join), 1.0))
    targets = [np.asarray(fix, dtype=np.float64) for fix in labels.waypoints[1:]] + [join]
    out = []
    for fix, ahead, (remaining, speed), height in zip(
        labels.waypoints, targets, labels.waypoint_speeds, labels.waypoint_heights_m, strict=True,
    ):
        heading = float(math.atan2(ahead[1] - fix[1], ahead[0] - fix[0]))
        if ahead is join and float(np.hypot(*(join - np.asarray(fix)))) < FIX_AT_JOIN_M:
            heading = float(skeleton.course_rad)
        out.append(Instruction(float(fix[0]), float(fix[1]), heading, float(speed), float(remaining), float(height)))
    return out


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PlanTargets:
    """One sample's supervision: the target vector (physical units, `TARGETS` order), which
    entries the track defines, and whether there is NO fix ahead — the join is next, or
    the flight is already on the final (`next_is_join`, defined on every sample: an
    established flight ordered a fix flies off the corridor to it, measured on the first
    smoke head, 2026-09-11)."""

    values: np.ndarray        # [P] float32
    valid: np.ndarray         # [P] float32, 1 where the target is defined
    next_is_join: bool

    def context(self, series_index: int) -> dict[str, np.ndarray]:
        return {
            CONTEXT_TARGETS: self.values.astype(np.float32),
            CONTEXT_VALID: self.valid.astype(np.float32),
            CONTEXT_NEXT_IS_JOIN: np.array(float(self.next_is_join), dtype=np.float32),
            CONTEXT_SERIES: np.array(int(series_index), dtype=np.int64),
        }


def targets_from_labels(labels: PlanLabels, series: FlightSeries, anchor: int, skeleton: RunwaySkeleton) -> PlanTargets:
    """The target vector of one label set read at ``anchor``. A None parameter is out of the
    loss (`valid` 0); the instruction group is defined only where the truth has a next fix;
    a flight established at the anchor supervises the operating parameters (its join
    distance being its own remaining path: the join is here) and the no-fix flag (§6)."""
    values = np.zeros(len(TARGETS), dtype=np.float32)
    valid = np.zeros(len(TARGETS), dtype=np.float32)

    def put(name: str, value: float | None) -> None:
        if value is None:
            return
        values[TARGETS.index(name)] = float(value)
        valid[TARGETS.index(name)] = 1.0

    put("T_s", labels.T_s)
    put("V_mid_mps", labels.V_mid_mps)
    put("d_decel_m", labels.d_decel_m)
    put("V_final_mps", labels.V_final_mps)
    put("h_capture_m", labels.h_capture_m)
    put("d_join_m", labels.d_join_m)
    put("remaining_m", labels.remaining_path_at_anchor_m)
    instructions = [] if labels.join_at_anchor else truth_instructions(labels, skeleton)
    if instructions:
        first = instructions[0]
        d_anchor, xt_anchor = skeleton.axes(series.values[anchor:anchor + 1, IDX["e"]], series.values[anchor:anchor + 1, IDX["n"]])
        d_fix, xt_fix = skeleton.axes(np.array([first.fix_e]), np.array([first.fix_n]))
        relative = wrap_angle(first.heading_out_rad - skeleton.course_rad)
        put("next_ahead_m", float(d_anchor[0] - d_fix[0]))
        put("next_across_m", float(xt_fix[0] - xt_anchor[0]))
        put("next_heading_cos", math.cos(relative))
        put("next_heading_sin", math.sin(relative))
        put("next_speed_mps", first.speed_mps)
        put("next_remaining_m", first.remaining_m)
        put("next_height_m", first.height_m)
    return PlanTargets(values=values, valid=valid, next_is_join=not instructions)


@dataclass(frozen=True)
class PlanOrder:
    """What one prediction tells the guidance to fly from an anchor: the operating
    parameters, the remaining path from here, and the next instruction (None: the leg runs
    onto the final, the join is next). ``clamped`` names the parameters moved into their
    range."""

    T_s: float
    V_mid_mps: float
    d_decel_m: float
    V_final_mps: float
    h_capture_m: float
    d_join_m: float
    remaining_m: float
    instruction: Instruction | None
    next_is_join_probability: float
    clamped: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "T_s": self.T_s, "V_mid_mps": self.V_mid_mps, "d_decel_m": self.d_decel_m,
            "V_final_mps": self.V_final_mps, "h_capture_m": self.h_capture_m, "d_join_m": self.d_join_m,
            "remaining_m": self.remaining_m, "next_is_join_probability": self.next_is_join_probability,
            "instruction": None if self.instruction is None else {
                "fix_e": self.instruction.fix_e, "fix_n": self.instruction.fix_n,
                "heading_out_rad": self.instruction.heading_out_rad, "speed_mps": self.instruction.speed_mps,
                "remaining_m": self.instruction.remaining_m, "height_m": self.instruction.height_m,
            },
            "clamped": list(self.clamped),
        }


def order_from_prediction(
    values: Sequence[float], next_is_join_probability: float, anchor_e: float, anchor_n: float,
    skeleton: RunwaySkeleton, *, next_is_join: bool | None = None,
) -> PlanOrder:
    """A predicted target vector (physical units, `TARGETS` order) as the order it flies:
    the fix placed in the chart from its runway-axes offsets about the anchor, the heading
    on from its unit vector and the course; every parameter clamped into its range (the
    speeds into the type's approach envelope, the distances into the remaining path). The
    instruction is None where the join is the next thing — by ``next_is_join`` when given,
    else by the predicted probability."""
    v = {name: float(x) for name, x in zip(TARGETS, values, strict=True)}
    clamped: list[str] = []

    def clamp(name: str, low: float, high: float) -> float:
        value = v[name]
        bounded = min(max(value, low), high)
        if bounded != value:
            clamped.append(name)
        return bounded

    remaining = clamp("remaining_m", 1.0, math.inf)
    d_join = clamp("d_join_m", 1.0, remaining)
    order_join = next_is_join_probability >= 0.5 if next_is_join is None else next_is_join
    instruction = None
    if not order_join:
        d_anchor, xt_anchor = skeleton.axes(np.array([anchor_e]), np.array([anchor_n]))
        e, n = chart_from_axes_np(
            np.array([float(d_anchor[0]) - v["next_ahead_m"]]), np.array([float(xt_anchor[0]) + v["next_across_m"]]),
            skeleton.course_rad,
        )
        heading = skeleton.course_rad + math.atan2(v["next_heading_sin"], v["next_heading_cos"])
        instruction = Instruction(
            fix_e=float(e[0]) + skeleton.target_e, fix_n=float(n[0]) + skeleton.target_n,
            heading_out_rad=float(wrap_angle(heading)),
            speed_mps=clamp("next_speed_mps", SPEED_MIN_MPS, SPEED_MAX_MPS),
            remaining_m=clamp("next_remaining_m", 0.0, remaining),
            height_m=v["next_height_m"],
        )
    return PlanOrder(
        T_s=clamp("T_s", T_MIN_S, math.inf),
        V_mid_mps=clamp("V_mid_mps", SPEED_MIN_MPS, SPEED_MAX_MPS),
        d_decel_m=clamp("d_decel_m", 0.0, remaining),
        V_final_mps=clamp("V_final_mps", SPEED_MIN_MPS, SPEED_MAX_MPS),
        h_capture_m=v["h_capture_m"],
        d_join_m=d_join, remaining_m=remaining,
        instruction=instruction, next_is_join_probability=float(next_is_join_probability),
        clamped=tuple(clamped),
    )
