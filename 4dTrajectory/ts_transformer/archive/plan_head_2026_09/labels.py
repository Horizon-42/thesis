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

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ts_transformer.data.approach_difficulty import remaining_path_profile_m
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.geometry.dubins import chart_from_axes_np
from ts_transformer.outputs.plan.extractors import PlanLabels
from ts_transformer.outputs.plan.guidance.route import INTERCEPT_MAX_RAD
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
#: The target contract stamped into a checkpoint and into a rolled-window table: the
#: target vector's composition — its names in order, digested — so a reordered or swapped
#: target refuses a stale checkpoint or table.
PLAN_TARGET_CONTRACT = "plan-v1-" + hashlib.sha1(",".join(TARGETS).encode()).hexdigest()[:8]
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

#: The context keys a plan batch carries (`PlanContext.row`). `plan_rolled` says whether
#: the sample's window is a ROLLED one (design v5.2: a lockstep flight's own window,
#: `outputs.plan.rolled`) rather than the observed track's at a drawn anchor — the epoch
#: record reports the share realised, and a reader of the loss can split it.
CONTEXT_TARGETS = "plan_targets"
CONTEXT_VALID = "plan_valid"
CONTEXT_NEXT_IS_JOIN = "plan_next_is_join"
CONTEXT_SERIES = "plan_series_index"
CONTEXT_ROLLED = "plan_rolled"
PLAN_CONTEXT_KEYS = (CONTEXT_TARGETS, CONTEXT_VALID, CONTEXT_NEXT_IS_JOIN, CONTEXT_SERIES, CONTEXT_ROLLED)

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


# ── the pose predicates the lockstep flies by and the labels are read by ─────────
#: A fix within this flight time of the aircraft is on top of it (skipped, never flown to);
#: a turn is done within this of the heading given; an aircraft within this of the
#: centreline, aligned inside the intercept limit, is ON the final. One definition —
#: `forecast` reads them from here for the flight, `TruthExpert` for the label.
LEG_MIN_S = 10.0
TURN_DONE_RAD = math.radians(5.0)
ON_FINAL_XT_M = 500.0


def fix_ahead(e: float, n: float, speed_mps: float, instruction: Instruction) -> bool:
    """Whether the fix is somewhere to fly to at all: not on top of the aircraft. A fix
    beside or behind it is a turn (a base turn puts the next fix 90° off the heading)."""
    return math.hypot(instruction.fix_e - e, instruction.fix_n - n) > LEG_MIN_S * speed_mps


def past_fix(e: float, n: float, instruction: Instruction) -> bool:
    """The aircraft is past the instruction's fix along the heading it gives."""
    u_e, u_n = math.cos(instruction.heading_out_rad), math.sin(instruction.heading_out_rad)
    return (e - instruction.fix_e) * u_e + (n - instruction.fix_n) * u_n >= 0.0


def turn_done(heading_rad: float, instruction: Instruction) -> bool:
    """The aircraft is on the heading the instruction gives, within `TURN_DONE_RAD`."""
    return abs(wrap_angle(heading_rad - instruction.heading_out_rad)) <= TURN_DONE_RAD


def on_final_pose(skeleton: RunwaySkeleton, e: float, n: float, heading_rad: float) -> bool:
    """Established on the final: within `ON_FINAL_XT_M` of the centreline, aligned with
    the course inside the intercept limit, before the threshold."""
    d, xt = skeleton.axes(np.array([e]), np.array([n]))
    return (
        float(d[0]) > 0.0 and abs(float(xt[0])) <= ON_FINAL_XT_M
        and abs(wrap_angle(heading_rad - skeleton.course_rad)) <= INTERCEPT_MAX_RAD
    )


class TruthExpert:
    """The truth's policy read at ANY pose: what it would order an aircraft there, and
    the operating parameters it flies from there.

    The queue is the truth's instructions in path order; the one in force is held until
    the aircraft has executed it — past its fix on its heading (`forecast.turn_done_row`'s
    rule read at one pose), or past it while on the final (`fly_lockstep`'s
    `behind_on_final`, the turn onto the final with its heading match still pending) —
    and one on top of the aircraft when its turn comes is skipped, as the oracle's policy
    skips it. The schedule coordinate and the arrival time are the truth's own at the
    NEAREST truth row of the leg in force (its rows between the previous fix and the
    current one, by remaining path), plus the way to that row: read at the anchor's own
    pose they are exactly the extractors' `remaining_path_at_anchor_m` and `T_s`, so the
    label at an observed anchor and at a rolled flight's step 0 are one and the same; at a
    learner's state off the truth's path they are the way back onto it and the truth's
    path from there — never the truth's values less what the learner happened to fly,
    which run out on a slow learner while it is still kilometres out. The lockstep's cap
    on instructions (`forecast.MAX_INSTRUCTION_LEGS`, a safety valve on a runaway head) is
    deliberately not mirrored: the label is the truth's instruction set, whole.
    """

    def __init__(self, labels: PlanLabels, series: FlightSeries, anchor: int, skeleton: RunwaySkeleton) -> None:
        self.labels = labels
        self.skeleton = skeleton
        self.queue = [] if labels.join_at_anchor else truth_instructions(labels, skeleton)
        self.current: Instruction | None = None
        self.skipped = 0
        truth = np.asarray(series.supervision_values[anchor:], dtype=np.float64)
        self._e, self._n = truth[:, IDX["e"]], truth[:, IDX["n"]]
        self._remaining_m = np.asarray(remaining_path_profile_m(series)[anchor:], dtype=np.float64)
        times = np.asarray(series.supervision_times[anchor:], dtype=np.float64)
        self._time_to_go_s = float(times[-1]) - times
        self._upper_m = float(self._remaining_m[0])    # the leg in force starts here (by remaining path)

    def order_at(self, e: float, n: float, heading_rad: float, speed_mps: float) -> tuple[Instruction | None, Operating]:
        if self.current is not None and past_fix(e, n, self.current) and (
            turn_done(heading_rad, self.current) or on_final_pose(self.skeleton, e, n, heading_rad)
        ):
            self._upper_m, self.current = float(self.current.remaining_m), None
        while self.current is None and self.queue:
            candidate = self.queue.pop(0)
            if fix_ahead(e, n, speed_mps, candidate):
                self.current = candidate
            else:
                self.skipped += 1
                self._upper_m = float(candidate.remaining_m)
        lower_m = 0.0 if self.current is None else float(self.current.remaining_m)
        leg = np.flatnonzero((self._remaining_m <= self._upper_m + 1e-6) & (self._remaining_m >= lower_m - 1e-6))
        if not leg.size:
            raise ValueError(f"the truth has no rows between remaining path {self._upper_m:.0f} and {lower_m:.0f} m")
        distance = np.hypot(self._e[leg] - e, self._n[leg] - n)
        k, way_m = int(leg[np.argmin(distance)]), float(distance.min())
        operating = Operating(
            T_s=float(self._time_to_go_s[k]) + way_m / max(float(speed_mps), 1.0),
            V_mid_mps=self.labels.V_mid_mps, d_decel_m=self.labels.d_decel_m, V_final_mps=self.labels.V_final_mps,
            h_capture_m=self.labels.h_capture_m, d_join_m=self.labels.d_join_m,
            remaining_m=float(self._remaining_m[k]) + way_m,
        )
        return self.current, operating


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

    def context(self, series_index: int, *, rolled: bool = False) -> dict[str, np.ndarray]:
        return {
            CONTEXT_TARGETS: self.values.astype(np.float32),
            CONTEXT_VALID: self.valid.astype(np.float32),
            CONTEXT_NEXT_IS_JOIN: np.array(float(self.next_is_join), dtype=np.float32),
            CONTEXT_SERIES: np.array(int(series_index), dtype=np.int64),
            CONTEXT_ROLLED: np.array(float(rolled), dtype=np.float32),
        }


@dataclass(frozen=True)
class Operating:
    """The operating group of one target vector (`OPERATING` order); None where the track
    does not define an entry (it is then out of the loss)."""

    T_s: float | None
    V_mid_mps: float | None
    d_decel_m: float | None
    V_final_mps: float | None
    h_capture_m: float | None
    d_join_m: float | None
    remaining_m: float | None


def operating_from_labels(labels: PlanLabels) -> Operating:
    """A label set's operating parameters as read at its own anchor."""
    return Operating(
        T_s=labels.T_s, V_mid_mps=labels.V_mid_mps, d_decel_m=labels.d_decel_m, V_final_mps=labels.V_final_mps,
        h_capture_m=labels.h_capture_m, d_join_m=labels.d_join_m, remaining_m=labels.remaining_path_at_anchor_m,
    )


def targets_at(
    operating: Operating, instruction: Instruction | None, e: float, n: float, skeleton: RunwaySkeleton,
) -> PlanTargets:
    """The target vector at a pose: the operating group as given, the instruction group
    read about ``(e, n)`` in runway axes where there is a fix ahead — the ONE definition,
    read at an observed anchor (`targets_from_labels`) and at a rolled flight's state
    (`outputs.plan.rolled`) alike. A None parameter is out of the loss (`valid` 0); the
    instruction group is defined only where there is a next fix; a flight with none ahead
    supervises the operating parameters and the no-fix flag (§6)."""
    values = np.zeros(len(TARGETS), dtype=np.float32)
    valid = np.zeros(len(TARGETS), dtype=np.float32)

    def put(name: str, value: float | None) -> None:
        if value is None:
            return
        values[TARGETS.index(name)] = float(value)
        valid[TARGETS.index(name)] = 1.0

    for name in OPERATING:
        put(name, getattr(operating, name))
    if instruction is not None:
        d_anchor, xt_anchor = skeleton.axes(np.array([e]), np.array([n]))
        d_fix, xt_fix = skeleton.axes(np.array([instruction.fix_e]), np.array([instruction.fix_n]))
        relative = wrap_angle(instruction.heading_out_rad - skeleton.course_rad)
        put("next_ahead_m", float(d_anchor[0] - d_fix[0]))
        put("next_across_m", float(xt_fix[0] - xt_anchor[0]))
        put("next_heading_cos", math.cos(relative))
        put("next_heading_sin", math.sin(relative))
        put("next_speed_mps", instruction.speed_mps)
        put("next_remaining_m", instruction.remaining_m)
        put("next_height_m", instruction.height_m)
    return PlanTargets(values=values, valid=valid, next_is_join=instruction is None)


def targets_from_labels(labels: PlanLabels, series: FlightSeries, anchor: int, skeleton: RunwaySkeleton) -> PlanTargets:
    """The target vector of one label set read at ``anchor``: the truth's policy at the
    anchor's own pose (`TruthExpert` — its first fix ahead, one on top of the aircraft
    skipped as the oracle skips it; the operating parameters as the extractors read them
    there); a flight established at the anchor has none ahead (its join distance being its
    own remaining path: the join is here)."""
    row = np.asarray(series.values[anchor], dtype=np.float64)
    e, n = float(row[IDX["e"]]), float(row[IDX["n"]])
    # the pose's heading is the chart velocity's direction (the chart is locally the ENU
    # frame), its speed the extractors' ground speed there
    heading = math.atan2(float(row[IDX["ndot"]]), float(row[IDX["edot"]]))
    instruction, operating = TruthExpert(labels, series, anchor, skeleton).order_at(
        e, n, heading, labels.ground_speed_at_anchor_mps,
    )
    return targets_at(operating, instruction, e, n, skeleton)


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
