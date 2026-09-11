"""Fly plans: a batch of plans through the guidance on the shared rollout, as `Forecast`s.

The plan path's forecast entry point, the shape `outputs.control.forecast.forecast_control_batch`
has: the per-flight physical context, the dense query grid, one rollout under the command
hook, one `Forecast` per flight with the schedule FLOWN in newtons and the hook's own
per-flight counts. Here the hook is the whole controller and the network's schedule is
zeros (the guidance overrides every command); the schedule's total is the plan's ``T``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
import torch

from ts_transformer.config import (
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_HOOK_OFF,
    CONTROL_RECIPE_CUSTOM,
    PREDICTION_CONTROL,
    TSConfig,
)
from ts_transformer.data.channels import IDX
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.geometry.final_approach_geometry import GLIDEPATH_ABOVE_M, GLIDEPATH_BELOW_M
from ts_transformer.inference.forecast import Forecast, cut_at_threshold_crossing
from ts_transformer.outputs.dynamics import rollout as control_rollout
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.dynamics.hooks import per_flight_hook_diagnostics
from ts_transformer.outputs.dynamics.rollout import padded_dense_queries
from ts_transformer.outputs.envelope import CONTROL_LOWER, CONTROL_UPPER, fraction_controls, physical_controls
from ts_transformer.outputs.plan.extractors import PlanLabels
from ts_transformer.outputs.plan.labels import Instruction, PlanOrder, join_point, truth_instructions, wrap_angle
from ts_transformer.outputs.plan.guidance.controller import PlanGuidance, PlanToFly, reference_height
from ts_transformer.outputs.plan.guidance.route import (
    CONVERGE_MIN_M,
    INTERCEPT_MAX_RAD,
    PATH_STEP_M,
    Route,
    build_route,
    route_time_s,
    route_turn_radius_m,
    speed_schedule_mps,
)
from ts_transformer.geometry.dubins import unit_vector
from ts_transformer.outputs.plan.skeleton import RunwaySkeleton

#: The capture height the guidance flies is inside the glidepath WINDOW at the join
#: (`final_approach_geometry`: −60 / +120 m about the glidepath, the optimizer's), by these
#: margins — a plan that captures the final above or below the window is clamped into it
#: and the clamp is counted (design §3: the model never plans an infeasible flight).
CAPTURE_ABOVE_GLIDEPATH_MAX_M = GLIDEPATH_ABOVE_M - 30.0
CAPTURE_BELOW_GLIDEPATH_MAX_M = GLIDEPATH_BELOW_M - 30.0

#: The guidance is a command hook, and only the lagged rollout runs hooks.
GUIDANCE_DYNAMICS = dict(
    prediction_output=PREDICTION_CONTROL,
    control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    control_command_hook=CONTROL_HOOK_OFF,
    control_recipe_name=CONTROL_RECIPE_CUSTOM,
)
#: The guidance is re-issued every hold; the tracker's dynamics (an L1 point 8 s ahead)
#: need a few commands per period, so the hold is sized at ~3 s: ``N = ceil(T / HOLD_S)``
#: inside ``[N_SEGMENTS_MIN, N_SEGMENTS_MAX]`` (a 7 s hold on a 470 s vectored arrival
#: oscillated ±300 m on the final, 2026-09-10).
HOLD_S = 3.0
N_SEGMENTS_MIN = 64
N_SEGMENTS_MAX = 256


def n_segments_for(horizon_s: float) -> int:
    return int(min(N_SEGMENTS_MAX, max(N_SEGMENTS_MIN, math.ceil(horizon_s / HOLD_S))))
#: The label a plan-guidance forecast carries as its hook name.
HOOK_NAME = "plan-guidance"
#: Which representation of the pre-final path the route is laid from: the three route
#: parameters (`d_join`, `side`, `L_pre` — design §3b as written), or those plus the
#: plan's fixed-K fly-by waypoints (`PlanLabels.waypoints`, 2026-09-11).
ROUTE_PLAN = "plan"
ROUTE_WAYPOINTS = "waypoints"
ROUTES = (ROUTE_PLAN, ROUTE_WAYPOINTS)


def guidance_config(config: TSConfig) -> TSConfig:
    """The rollout the guidance flies under: the lagged dynamics, no post-hoc hook."""
    return replace(config, **GUIDANCE_DYNAMICS)


def speed_points_for(labels: PlanLabels, route: str) -> tuple[tuple[float, float], ...]:
    """The speed schedule's points under ``route``: the anchor's and each fix's speed on
    the waypoints route, none on the plan's own law."""
    if route != ROUTE_WAYPOINTS or not labels.waypoint_speeds:
        return ()
    return ((labels.remaining_path_at_anchor_m, labels.ground_speed_at_anchor_mps), *labels.waypoint_speeds)


def plan_to_fly(labels: PlanLabels, anchor_height_m: float, skeleton: RunwaySkeleton, *,
                route: str = ROUTE_PLAN) -> tuple[PlanToFly, bool]:
    """The controller's parameters from a label set, and whether the capture height was
    clamped into the glidepath window at the join. A join before the anchor (a censored
    capture height) starts the height profile where the aircraft is, unclamped."""
    points = speed_points_for(labels, route)
    if labels.h_capture_m is None or labels.d_join_m is None:
        return PlanToFly(labels.V_mid_mps, labels.d_decel_m, labels.V_final_mps, anchor_height_m, points), False
    glidepath = max(labels.d_join_m, 0.0) * skeleton.glidepath_tan
    low = glidepath - CAPTURE_BELOW_GLIDEPATH_MAX_M
    high = glidepath + CAPTURE_ABOVE_GLIDEPATH_MAX_M
    clamped = min(max(labels.h_capture_m, low), high)
    return (
        PlanToFly(labels.V_mid_mps, labels.d_decel_m, labels.V_final_mps, clamped, points),
        clamped != labels.h_capture_m,
    )


def route_for(series: FlightSeries, anchor: int, labels: PlanLabels, skeleton: RunwaySkeleton,
              initial_state: np.ndarray, *, route: str = ROUTE_PLAN) -> Route:
    """The plan's route from the anchor pose (`dynamics_arrays`' initial state: the physical
    heading and speed there). A join before the anchor is the final from where it is."""
    row = series.values[anchor]
    d_join = labels.d_join_m if labels.d_join_m is not None else labels.remaining_path_at_anchor_m
    # the route's turns are sized at the speed the schedule has where they are flown
    points = speed_points_for(labels, route)

    def speed_at(remaining_m: float) -> float:
        return float(speed_schedule_mps(
            remaining_m, v_mid=labels.V_mid_mps, d_decel_m=labels.d_decel_m, v_final=labels.V_final_mps,
            points=points,
        ))

    # a censored side (the join before the anchor, or no offset wider than the side rule
    # reads) is `0`: the route then stretches, if it must, to the anchor's own side
    return build_route(
        float(row[IDX["e"]]), float(row[IDX["n"]]),
        float(initial_state[4]), float(initial_state[3]),
        d_join_m=d_join, side=0 if labels.side is None else labels.side,
        pre_final_m=0.0 if labels.L_pre_m is None else labels.L_pre_m,
        skeleton=skeleton, join_at_anchor=labels.join_at_anchor, speed_at=speed_at,
        waypoints=labels.waypoints if route == ROUTE_WAYPOINTS else None,
        waypoint_speeds=labels.waypoint_speeds if route == ROUTE_WAYPOINTS else None,
    )


def fly_plans(
    series: Sequence[FlightSeries],
    anchor: int | Sequence[int],
    labels: Sequence[PlanLabels],
    skeletons: Sequence[RunwaySkeleton],
    config: TSConfig,
    *,
    durations_s: Sequence[float] | None = None,
    device: torch.device | None = None,
    route: str = ROUTE_PLAN,
) -> tuple[list[Forecast], list[Route], list[float]]:
    """One rollout of every plan under the guidance; ``durations_s`` overrides each plan's
    ``T`` (the assigned arrival time); ``anchor`` is one index for the batch or one per
    flight. Returns the forecasts, the routes laid, and the time the speed schedule
    needs for each route (the time closure's reading)."""
    if not (len(series) == len(labels) == len(skeletons)):
        raise ValueError("one label set and one skeleton per flight")
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {route!r}")
    anchors = [int(anchor)] * len(series) if isinstance(anchor, (int, np.integer)) else [int(a) for a in anchor]
    if len(anchors) != len(series):
        raise ValueError("one anchor per flight")
    device = device or torch.device("cpu")
    config = guidance_config(config)
    rows = [dynamics_arrays(item, a) for item, a in zip(series, anchors, strict=True)]
    anchor_heights = [float(item.values[a, IDX["u"]]) for item, a in zip(series, anchors, strict=True)]
    routes = [
        route_for(item, a, lab, skeleton, row["initial_state"], route=route)
        for item, a, lab, skeleton, row in zip(series, anchors, labels, skeletons, rows, strict=True)
    ]
    flown = [plan_to_fly(lab, h, sk, route=route) for lab, h, sk in zip(labels, anchor_heights, skeletons, strict=True)]
    plans = [plan for plan, _clamped in flown]
    clamped = [clamped for _plan, clamped in flown]
    totals = [lab.T_s for lab in labels] if durations_s is None else [float(t) for t in durations_s]
    forecasts = fly_routes(
        rows, routes, plans, anchor_heights, config, totals,
        [float(item.times[a]) for item, a in zip(series, anchors, strict=True)], anchors,
        clamped=clamped, device=device,
    )
    times = [
        route_time_s(route, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps,
                     points=plan.speed_points)
        for route, plan in zip(routes, plans, strict=True)
    ]
    assert all(math.isfinite(t) for t in times)
    return forecasts, routes, times


def fly_routes(
    rows: Sequence[dict[str, np.ndarray]],
    routes: Sequence[Route],
    plans: Sequence[PlanToFly],
    anchor_heights_m: Sequence[float],
    config: TSConfig,
    totals_s: Sequence[float],
    start_times_s: Sequence[float],
    anchors: Sequence[int],
    *,
    clamped: Sequence[bool] | None = None,
    n_segments: int | None = None,
    progress: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> list[Forecast]:
    """One rollout of a batch already laid: each flight's dynamics row (`dynamics_arrays`'
    shape — a flown state re-anchored is one too, `rolling.advance_row`), its route, its
    plan, the height the profile starts from, the horizon and the wall-clock start. The
    guidance runs as the one command hook; ``config`` is already `guidance_config`'s."""
    device = device or torch.device("cpu")
    totals = np.asarray(list(totals_s), dtype=np.float64)
    if np.any(totals <= 0.0):
        raise ValueError("every plan needs a positive duration")
    dynamics = {
        name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device)
        for name in rows[0]
    }
    anchor_heights = np.asarray(list(anchor_heights_m), dtype=np.float64)
    # one segment count for the batch (`n_segments_for` the whole flight's; a lockstep
    # step's is `LOCKSTEP_SEGMENTS`), so the hold flown is each flight's own T / N
    n_segments = n_segments_for(float(totals.max())) if n_segments is None else int(n_segments)
    durations = np.repeat(totals[:, None] / n_segments, n_segments, axis=1)
    offsets, padded, valid = padded_dense_queries(durations, config.control_rollout_integrator_dt_s)
    hook = PlanGuidance(
        config, dynamics, list(routes), list(plans), anchor_heights,
        progress=None if progress is None else np.asarray(list(progress), dtype=np.int64),
    )
    zeros = torch.zeros((len(rows), n_segments, 3), dtype=torch.float64, device=device)
    with torch.no_grad():
        rollout = control_rollout.rollout_control_dense(
            zeros, torch.from_numpy(durations).to(device), dynamics,
            torch.from_numpy(padded), torch.from_numpy(valid), config, command_hook=hook,
        )
    query_channels = rollout.query_channels.detach().cpu().numpy().astype(np.float64)
    query_geodetic = rollout.query_geodetic_states.detach().cpu().numpy().astype(np.float64)
    controls = (
        physical_controls(rollout.controls, dynamics["max_thrust_n"]).detach().cpu().numpy().astype(np.float64)
    )
    diagnostics = per_flight_hook_diagnostics(hook)
    # one segment count for the batch, so the hold FLOWN is each flight's own T / N — at
    # most HOLD_S, shorter on a flight with less time than the batch's longest; the record
    # says which
    was_clamped = [False] * len(rows) if clamped is None else list(clamped)
    for row, flag, total in zip(diagnostics, was_clamped, totals, strict=True):
        row["planCaptureHeightClamped"] = float(flag)
        row["planHoldS"] = float(total / n_segments)
    forecasts: list[Forecast] = []
    for row, (start_s, a, row_offsets) in enumerate(zip(start_times_s, anchors, offsets, strict=True)):
        count = len(row_offsets)
        final_time_s = float(row_offsets[-1])
        forecasts.append(Forecast(
            times=float(start_s) + row_offsets,
            values=query_channels[row, :count],
            normalized_progress=row_offsets / final_time_s,
            anchor=int(a),
            final_time_s=final_time_s,
            predicted_final_time_s=float(totals[row]),
            horizon_mode=config.horizon_mode,
            passes=1,
            truncated_at_threshold=False,
            horizon_capped=False,
            controls=controls[row],
            sample_durations_s=np.diff(np.concatenate(([0.0], row_offsets))),
            segment_durations_s=durations[row],
            geodetic_values=query_geodetic[row, :count],
            prediction_output=PREDICTION_CONTROL,
            command_hook=HOOK_NAME,
            command_hook_diagnostics=diagnostics[row],
        ))
    return forecasts


# ── one instruction at a time (design v5 §4.2) ─────────────────────────────────
# Fly a plan ONE INSTRUCTION AT A TIME (design v5, §4.2): the next fly-by fix and the speed
# at it are the route for one leg; the guidance flies that leg on the shared rollout; the
# state it ends in is the next anchor, and the next instruction is asked for there; when
# there is none, the leg is the closing onto the join and the final. What a radar vector is
# to a cockpit — one at a time, executed within seconds, the next unknown until given.

# Each leg's ORDER comes from a callable (`LegOrders`), so the same flight serves the oracle
# (the truth's own instructions in path order, its own operating parameters —
# `fly_rolling`) and the plan head (its prediction at each anchor, read off the window of
# what was flown — `fly_rolling_orders`). A fix the aircraft is already on top of is not
# an instruction (`ahead_of`): the oracle offers its next one, the head's closing is flown.

#: The route kind a rolled flight reports: several legs, each a route of its own.
KIND_ROLLED = "rolled"
#: An instruction's leg is laid through its fix and on along the heading it gives, this
#: far — long enough for the corner at the fix to be rounded at the fix's speed (a 90°
#: corner at 110 m/s needs 3.5 km either side); only the part to the fix is flown.
LEG_EXTENSION_M = 8_000.0
#: A leg is flown for the time the schedule needs to its fix, at least this long (a fix
#: on top of the aircraft is passed over instead).
LEG_MIN_S = 10.0
#: A leg is flown until the AIRCRAFT is on the heading its instruction gives, within this,
#: past the fix (cut at the fix's first pass — mid-turn — the next leg started from a pose
#: the plan-route builder answered with a loop, 2026-09-11); the route's own turn end is
#: read with the same tolerance.
TURN_DONE_RAD = math.radians(5.0)
#: An instruction leg is rolled for the schedule's time to the route's turn end plus this,
#: so the aircraft (which lags the route) gets there; the rows past its turn are cut.
TURN_SETTLE_S = 30.0
#: An instruction whose fix is within this many turn radii of the aircraft (at the
#: instruction's speed), or beside or behind it, is flown as "turn to heading X" from
#: where the aircraft is rather than as a fly-by through the fix.
TURN_AT_FIX_RADII = 1.5
#: The "turn to heading X" form's onward point lies at least this many turn radii ahead:
#: nearer, the turn onto the heading cannot meet the leg before its corner and the builder
#: answers with a loop the other way (a 48° turn at 6 km: 240 s in a circle, 2026-09-11).
HEADING_FORM_RADII = 4.0
#: A closing laid from within this of the join that cannot head at it inside the
#: intercept limit is the intercept polyline onto the centreline (never the aligned-pose
#: turn-straight-turn, which loops from a pose 1–2 km short of the join).
CLOSING_INTERCEPT_M = 8_000.0
#: A heading on within this of the direction to the join points AT the join: the leg
#: gets no extension beyond its fix (the onward point overshot the join and the polyline
#: looped back to it — 32–44 km of closing path on the full L−1 oracle's worst flights).
EXTENSION_SKIP_RAD = math.radians(15.0)
#: The last leg (onto the join and down the final) runs past the plan's remaining time by
#: this, so a late arrival is measured as late rather than cut off short of the threshold
#: — the oracle's rule (`experiments.plan_oracle`).
CLOSING_SLACK_S = 30.0
CLOSING_SLACK_FRACTION = 0.1
#: A head that keeps issuing fixes is cut here: after this many instruction legs the
#: closing is flown whatever it says (the label set carries at most `MAX_WAYPOINTS`).
MAX_INSTRUCTION_LEGS = 6
#: A rolled prediction flies at most this multiple of its first order's budget (the
#: predicted arrival time plus the closing slack): the guard against a head whose legs
#: never reach the final — `rolled_time_cap_s`, the one expression for both forms.
ROLLED_TIME_CAP_FACTOR = 1.5


def closing_budget_s(arrival_time_s: float) -> float:
    """An order's time budget: its arrival time plus the closing slack."""
    return float(arrival_time_s) + max(CLOSING_SLACK_S, CLOSING_SLACK_FRACTION * float(arrival_time_s))


def rolled_time_cap_s(first_budget_s: float) -> float:
    return ROLLED_TIME_CAP_FACTOR * float(first_budget_s)


@dataclass(frozen=True)
class Anchor:
    """Where a leg starts: the chart state, the physical heading and speed, the remaining
    path, the wall-clock time and the dynamics row to roll from."""

    e: float
    n: float
    height_m: float
    heading_rad: float
    speed_mps: float
    remaining_m: float
    time_s: float
    row: dict[str, np.ndarray]


@dataclass(frozen=True)
class LegOrder:
    """What one leg is flown by: the operating parameters, the join distance, the remaining
    path from this anchor (None: keep the anchor's), the instruction (None: the closing),
    the time budget from here, and whether a skipped instruction is to be asked again
    (the oracle's queue) or means the closing (the head's answer stands)."""

    plan: PlanToFly
    d_join_m: float | None
    instruction: Instruction | None
    budget_s: float
    remaining_m: float | None = None
    retry_on_skip: bool = False
    capture_clamped: bool = False    # the capture height was clamped into the glidepath window
    arrival_time_s: float | None = None   # the head's own arrival time (the prediction a rolled flight reports)
    record: dict[str, object] | None = None


LegOrders = Callable[[Anchor, int, Sequence[Forecast]], LegOrder]


def first_anchor(series: FlightSeries, anchor: int, remaining_m: float) -> Anchor:
    row = dynamics_arrays(series, anchor)
    state = row["initial_state"]
    values = series.values[anchor]
    return Anchor(
        e=float(values[IDX["e"]]), n=float(values[IDX["n"]]), height_m=float(values[IDX["u"]]),
        heading_rad=float(state[4]), speed_mps=float(state[3]),
        remaining_m=float(remaining_m), time_s=float(series.times[anchor]), row=row,
    )


def advance_row(row: dict[str, np.ndarray], forecast: Forecast) -> dict[str, np.ndarray]:
    """The dynamics row re-anchored where ``forecast`` ends: its last geodetic state, and
    the last command flown as the actuators' state (the lagged actuators are within a
    time constant of it), clipped to the envelope box."""
    last_state = np.asarray(forecast.geodetic_values[-1], dtype=np.float64)
    controls = np.asarray(fraction_controls(np.asarray(forecast.controls[-1], dtype=np.float64), np.asarray(row["max_thrust_n"], dtype=np.float64)), dtype=np.float64)
    return {
        **row,
        "initial_state": last_state,
        "initial_controls": np.clip(controls, CONTROL_LOWER, CONTROL_UPPER).astype(np.float64),
    }


def next_anchor(previous: Anchor, forecast: Forecast, instruction: Instruction) -> Anchor:
    """The anchor a flown leg ends in. The remaining path is the INSTRUCTION's: the leg's
    own route runs on past the fix (the extension and the fallback beyond it), so its arc
    length at the aircraft would count path the plan never meant — read that way, the
    closing leg was asked to lay 20 km of pre-final path and flew a loop (2026-09-11)."""
    last = forecast.values[-1]
    state = forecast.geodetic_values[-1]
    return Anchor(
        e=float(last[IDX["e"]]), n=float(last[IDX["n"]]), height_m=float(last[IDX["u"]]),
        heading_rad=float(state[4]), speed_mps=float(state[3]),
        remaining_m=max(float(instruction.remaining_m), 0.0), time_s=float(forecast.times[-1]),
        row=advance_row(previous.row, forecast),
    )


def ahead_of(anchor: Anchor, instruction: Instruction) -> bool:
    """Whether the fix is somewhere to fly to at all: not on top of the aircraft. A fix
    beside or behind it is a turn (a base turn puts the next fix 90° off the heading), and
    the leg's own turn-straight-turn handles it."""
    return math.hypot(instruction.fix_e - anchor.e, instruction.fix_n - anchor.n) > LEG_MIN_S * anchor.speed_mps


def leg_route(anchor: Anchor, instruction: Instruction | None, d_join_m: float | None, skeleton: RunwaySkeleton,
              plan: PlanToFly, *, mid_flight: bool = False) -> tuple[Route, PlanToFly, int]:
    """The route and the schedule for one leg, and the route's OWN join index (its first
    point of the final leg, which the returned route's ``join_index`` overrides for an
    instruction leg — see below). ``mid_flight`` (the lockstep's re-lays) admits the
    "turn to heading X" forms for an instruction leg laid from inside its turn, with its
    fix behind, or where the fly-by would loop; the drawn replay and the leg form keep
    the fly-by through the fix.

    With an instruction: through its fix, the corner there rounded at the instruction's
    speed onto the heading it gives, that heading continued `LEG_EXTENSION_M` (none where
    it points at the join: the leg beyond the fix IS the join leg), then the join and the
    final beyond. The returned route's ``join_index`` is the END OF THE TURN at the fix
    (`turn_end_index`): the controller's pre-final descent ends there, at the
    instruction's height, and the leg is timed to it. Without an instruction — the
    closing — the polyline onto the join where the aircraft already heads at it inside
    the intercept limit (the corner there rounded onto the course, as the whole-path
    route ends), else the plan's route onto the join from where it is; the join never
    behind the aircraft (at or past it the converge rule of `build_route`). The schedule
    runs from the anchor's own speed to the instruction's, then the plan's deceleration."""
    d_join = anchor.remaining_m if d_join_m is None else float(d_join_m)
    d_join = min(d_join, anchor.remaining_m)
    points_closing = ((anchor.remaining_m, anchor.speed_mps),)
    onto_join = join_ahead = False
    if instruction is None:
        d0, _xt0 = skeleton.axes(np.array([anchor.e]), np.array([anchor.n]))
        join_ahead = float(d0[0]) - CONVERGE_MIN_M > d_join
        d_join = min(d_join, max(float(d0[0]) - CONVERGE_MIN_M, PATH_STEP_M))
        join = join_point(skeleton, max(d_join, 1.0))
        to_join = math.atan2(join[1] - anchor.n, join[0] - anchor.e)
        onto_join = join_ahead and abs(wrap_angle(anchor.heading_rad - to_join)) <= INTERCEPT_MAX_RAD
        intercept = None
        if join_ahead and not onto_join and float(d0[0]) - d_join <= CLOSING_INTERCEPT_M:
            # near the join but off its direction: intercept the centreline at the limit
            # angle, the corner there rounded onto the course (the aligned-pose
            # turn-straight-turn loops from here)
            d_intercept = float(d0[0]) - abs(float(_xt0[0])) / math.tan(INTERCEPT_MAX_RAD) - PATH_STEP_M
            if d_intercept > max(d_join, PATH_STEP_M):
                intercept = join_point(skeleton, d_intercept)
            else:
                d_join = max(float(d0[0]) - CONVERGE_MIN_M, PATH_STEP_M)
                join_ahead = False
        if onto_join:
            # the polyline's corner is the fly-by point BEFORE the join whose arc onto the
            # course ends at the join (the whole-path route's last corner), at the
            # schedule's radius there and the pose's angle to the course
            theta = abs(wrap_angle(anchor.heading_rad - skeleton.course_rad))
            tangent = route_turn_radius_m(speed_at_join := float(speed_schedule_mps(
                d_join, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps, points=points_closing,
            ))) * math.tan(0.5 * theta)
            corner = join_point(skeleton, d_join + tangent)
            onto_join = float(d0[0]) - PATH_STEP_M > d_join + tangent
            if not onto_join:
                # the corner onto the course would already have started: converge onto
                # the centreline from where the aircraft is (re-laid inside the corner,
                # the turn-straight-turn onto the aligned join pose looped)
                d_join = max(float(d0[0]) - CONVERGE_MIN_M, PATH_STEP_M)
                join_ahead = False
    points = ((anchor.remaining_m, anchor.speed_mps),) + (
        () if instruction is None else ((instruction.remaining_m, instruction.speed_mps),)
    )
    # the leg descends to the instruction's height by the end of its turn, the closing to
    # the plan's capture height at the join — and a closing that converges from at or
    # inside the join flies the height from where the aircraft is (re-laid every lockstep
    # from an established flight, the plan's capture height held it level at its anchor
    # height: every straight-in flight out of the glidepath window, 2026-09-11)
    leg_plan = replace(
        plan, speed_points=points,
        h_capture_m=(anchor.height_m if instruction is None and not join_ahead else plan.h_capture_m)
        if instruction is None else instruction.height_m,
    )

    def speed_at(remaining_m: float) -> float:
        return float(speed_schedule_mps(
            remaining_m, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps, points=points,
        ))

    common = dict(
        d_join_m=max(d_join, 1.0), side=0, pre_final_m=max(anchor.remaining_m - d_join, 0.0), skeleton=skeleton,
        speed_at=speed_at,
    )
    if instruction is None and onto_join:
        route = build_route(
            anchor.e, anchor.n, anchor.heading_rad, anchor.speed_mps, **common,
            waypoints=((float(corner[0]), float(corner[1])),), waypoint_speeds=((d_join + tangent, speed_at_join),),
        )
        return route, leg_plan, route.join_index
    if instruction is None and intercept is not None:
        route = build_route(
            anchor.e, anchor.n, anchor.heading_rad, anchor.speed_mps, **common,
            waypoints=((float(intercept[0]), float(intercept[1])),), waypoint_speeds=((d_intercept, speed_at(d_intercept)),),
        )
        return route, leg_plan, route.join_index
    if instruction is None:
        route = build_route(anchor.e, anchor.n, anchor.heading_rad, anchor.speed_mps, **common)
        return route, leg_plan, route.join_index
    fix = np.array([instruction.fix_e, instruction.fix_n])
    join = join_point(skeleton, max(d_join, 1.0))
    to_join = math.atan2(join[1] - fix[1], join[0] - fix[0])
    to_fix = fix - np.array([anchor.e, anchor.n])
    along_heading = float(to_fix[0] * math.cos(anchor.heading_rad) + to_fix[1] * math.sin(anchor.heading_rad))
    radius = route_turn_radius_m(instruction.speed_mps)
    if mid_flight and (float(np.hypot(*to_fix)) <= TURN_AT_FIX_RADII * radius or along_heading < -radius):
        # inside the turn at the fix, or the fix clearly behind (a base-turn fix is abeam
        # by construction and is flown to): the instruction is
        # "turn to heading X" — the polyline from the pose along the heading given (the
        # Dubins onto it is the turn), then the join; a fly-by through the fix from here
        # ran its first leg back to the fix and looped (2026-09-11, the lockstep's first
        # smoke: vectored ADE 4250 m)
        reach = max(LEG_EXTENSION_M, HEADING_FORM_RADII * radius)
        onward = np.array([anchor.e, anchor.n]) + reach * unit_vector(instruction.heading_out_rad)
        waypoints = ((float(onward[0]), float(onward[1])),)
        waypoint_speeds = ((instruction.remaining_m - reach, instruction.speed_mps),)
    elif abs(wrap_angle(instruction.heading_out_rad - to_join)) <= EXTENSION_SKIP_RAD:
        waypoints = ((float(fix[0]), float(fix[1])),)
        waypoint_speeds = ((instruction.remaining_m, instruction.speed_mps),)
    else:
        onward = fix + LEG_EXTENSION_M * unit_vector(instruction.heading_out_rad)
        waypoints = ((float(fix[0]), float(fix[1])), (float(onward[0]), float(onward[1])))
        waypoint_speeds = ((instruction.remaining_m, instruction.speed_mps),
                           (instruction.remaining_m - LEG_EXTENSION_M, instruction.speed_mps))
    route = build_route(
        anchor.e, anchor.n, anchor.heading_rad, anchor.speed_mps, **common,
        waypoints=waypoints, waypoint_speeds=waypoint_speeds,
    )
    turn_end = turn_end_index(route, instruction)
    if mid_flight and float(route.arc_m[turn_end]) > float(np.hypot(*to_fix)) + math.pi * radius + LEG_EXTENSION_M:
        # the fly-by from here loops before the fix (its arc to the turn's end runs more
        # than a half-circle past the straight distance): "turn to heading X" instead
        reach = max(LEG_EXTENSION_M, HEADING_FORM_RADII * radius)
        onward = np.array([anchor.e, anchor.n]) + reach * unit_vector(instruction.heading_out_rad)
        route = build_route(
            anchor.e, anchor.n, anchor.heading_rad, anchor.speed_mps, **common,
            waypoints=((float(onward[0]), float(onward[1])),),
            waypoint_speeds=((instruction.remaining_m - reach, instruction.speed_mps),),
        )
        turn_end = turn_end_index(route, instruction)
    return replace(route, join_index=min(turn_end + 1, len(route.points) - 1)), leg_plan, route.join_index


def first_pass_index(route: Route, fix_e: float, fix_n: float) -> int:
    """The route point where it FIRST passes the fix: the first local minimum of the
    distance to it that is also the running minimum. The route runs on past the fix to
    the join and can come back near it (the fallback beyond the leg loops where the
    leg's heading points away from the runway); the nearest point overall then lies on
    that loop, and a leg timed to it flew the loop (2026-09-11)."""
    distance = np.hypot(route.points[:, 0] - fix_e, route.points[:, 1] - fix_n)
    running_min = np.minimum.accumulate(distance)
    grows = np.concatenate((distance[1:] > distance[:-1], [True]))
    candidates = np.flatnonzero((distance <= running_min + 1e-9) & grows)
    return int(candidates[0]) if candidates.size else int(np.argmin(distance))


def turn_end_index(route: Route, instruction: Instruction) -> int:
    """The route point where the turn at the instruction's fix is DONE: the first point at
    or past the fix's first pass whose heading is within `TURN_DONE_RAD` of the heading
    on, before the route's own join (a fix onto the final ends its turn ON the join)."""
    start = first_pass_index(route, instruction.fix_e, instruction.fix_n)
    steps = np.diff(route.points, axis=0)
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    last = min(len(headings), max(route.join_index, start + 1))
    for index in range(start, last):
        if abs(wrap_angle(float(headings[index]) - instruction.heading_out_rad)) <= TURN_DONE_RAD:
            return index
    return start


def time_to_turn_end_s(route: Route, plan: PlanToFly) -> float:
    """The time the schedule needs along an instruction leg's route to the end of the turn
    at its fix (the route's ``join_index``, `leg_route`)."""
    arc = route.arc_m[: route.join_index]
    remaining = route.threshold_arc_m - arc
    speed = speed_schedule_mps(
        remaining, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps, points=plan.speed_points,
    )
    pace = 0.5 * (speed[1:] + speed[:-1])
    return float(np.sum(np.diff(arc) / np.maximum(pace, 1.0)))


def past_fix_abeam(anchor: Anchor, instruction: Instruction) -> bool:
    """The aircraft is past the instruction's fix along the heading it gives."""
    u_e, u_n = math.cos(instruction.heading_out_rad), math.sin(instruction.heading_out_rad)
    return (anchor.e - instruction.fix_e) * u_e + (anchor.n - instruction.fix_n) * u_n >= 0.0


def turn_done_row(forecast: Forecast, instruction: Instruction) -> int | None:
    """The first flown row at which the AIRCRAFT has executed the instruction: past the
    fix along the heading on, and on that heading within `TURN_DONE_RAD`; None where the
    leg ended before it got there. The route's own turn end is where the tracker should
    be; the tracker lags it (an L1 point 8 s ahead) — cut on the route's clock, the
    next leg started 60° off its heading 3 km short of the join and the plan-route
    builder answered with a 25 km loop (2026-09-11)."""
    u_e, u_n = math.cos(instruction.heading_out_rad), math.sin(instruction.heading_out_rad)
    past = (forecast.values[:, IDX["e"]] - instruction.fix_e) * u_e + (forecast.values[:, IDX["n"]] - instruction.fix_n) * u_n >= 0.0
    psi = np.asarray(forecast.geodetic_values[:, 4], dtype=np.float64)
    done = np.abs((psi - instruction.heading_out_rad + math.pi) % (2.0 * math.pi) - math.pi) <= TURN_DONE_RAD
    rows = np.flatnonzero(past & done)
    return int(rows[0]) if rows.size else None


def cut_rows(forecast: Forecast, count: int) -> Forecast:
    """The forecast's first ``count`` rows, its control segments cut to the one containing
    the new end and that one shortened to land on it — `inference.forecast.
    cut_at_threshold_crossing`'s clock rule, without its crossing rule."""
    if count >= len(forecast.times):
        return forecast
    sample_durations_s = forecast.sample_durations_s[:count]
    offsets = np.cumsum(sample_durations_s)
    final_time_s = float(offsets[-1])
    boundaries = np.cumsum(forecast.segment_durations_s)
    last = int(np.searchsorted(boundaries, final_time_s, side="left"))
    segment_durations_s = forecast.segment_durations_s[: last + 1].copy()
    segment_durations_s[-1] = final_time_s - (0.0 if last == 0 else boundaries[last - 1])
    # the hook's `steps` is the segments kept; its shares stay the rollout's (the cut
    # steps' own counts are not recoverable from a per-flight share)
    diagnostics = {**forecast.command_hook_diagnostics, "steps": float(last + 1)}
    return replace(
        forecast, times=forecast.times[:count], values=forecast.values[:count],
        normalized_progress=offsets / final_time_s, final_time_s=final_time_s,
        sample_durations_s=sample_durations_s, segment_durations_s=segment_durations_s,
        controls=forecast.controls[: last + 1], geodetic_values=forecast.geodetic_values[:count],
        command_hook_diagnostics=diagnostics,
    )


def concatenate(legs: Sequence[Forecast], anchor: int, predicted_final_time_s: float) -> Forecast:
    """The legs as one forecast: the rows run on (a leg's rows start one query step after
    its anchor, so nothing repeats); the hook counts are summed over the legs' steps."""
    first = legs[0]
    times = np.concatenate([leg.times for leg in legs])
    values = np.concatenate([leg.values for leg in legs])
    geodetic = np.concatenate([leg.geodetic_values for leg in legs])
    samples = np.concatenate([leg.sample_durations_s for leg in legs])
    segments = np.concatenate([leg.segment_durations_s for leg in legs])
    controls = np.concatenate([leg.controls for leg in legs])
    final_time_s = float(np.sum(samples))
    steps = [float(leg.command_hook_diagnostics["steps"]) for leg in legs]
    diagnostics: dict[str, float | str] = {"steps": float(sum(steps))}
    for key in legs[0].command_hook_diagnostics:
        if key == "steps":
            continue
        values_by_leg = [leg.command_hook_diagnostics[key] for leg in legs]
        if all(isinstance(v, (int, float)) for v in values_by_leg):
            diagnostics[key] = float(np.average(values_by_leg, weights=steps))
        else:
            diagnostics[key] = values_by_leg[0]
    diagnostics["rolledLegs"] = float(len(legs))
    return replace(
        first, times=times, values=values, geodetic_values=geodetic, sample_durations_s=samples,
        segment_durations_s=segments, controls=controls, normalized_progress=np.cumsum(samples) / max(final_time_s, 1e-9),
        final_time_s=final_time_s, predicted_final_time_s=float(predicted_final_time_s), anchor=int(anchor),
        command_hook_diagnostics=diagnostics,
    )


#: What ended a rolled flight before its closing closed naturally (`RolledFlight.capped_by`,
#: the record's `planCappedBy`): the instruction-leg cap, or the time cap.
CAPPED_BY_LEGS = "legs"
CAPPED_BY_TIME = "time"


@dataclass(frozen=True)
class RolledFlight:
    forecast: Forecast
    routes: list[Route]            # one per leg, the closing last
    instructions_flown: int
    instructions_skipped: int      # behind the aircraft when their turn came
    leg_lengths_m: list[float]     # the path each leg's rollout actually flew
    leg_durations_s: list[float]
    closing_time_s: float          # the schedule's time over the closing leg's route
    orders: list[dict[str, object]]   # what each leg was flown by, as its order recorded it
    turns_incomplete: int          # legs that ended before the aircraft was on the heading given
    capped_by: str | None          # `CAPPED_BY_LEGS` / `CAPPED_BY_TIME`, or None: closed naturally

    @property
    def pre_final_m(self) -> float:
        """The path flown before the closing leg plus the closing's own pre-final path."""
        return float(sum(self.leg_lengths_m[:-1]) + self.routes[-1].pre_final_m)

    @property
    def route_time_s(self) -> float:
        return float(sum(self.leg_durations_s[:-1]) + self.closing_time_s)


def fly_legs(
    series: FlightSeries,
    anchor: int,
    skeleton: RunwaySkeleton,
    config: TSConfig,
    *,
    orders: LegOrders,
    remaining_m: float,
    predicted_final_time_s: float,
    time_cap_s: float,
    device: torch.device | None = None,
) -> RolledFlight:
    """Fly ``series`` from ``anchor`` one leg at a time, each leg by the order ``orders``
    gives at its anchor (the anchor, the leg index, the legs flown so far). An instruction
    leg is rolled for the schedule's time to the end of its turn plus `TURN_SETTLE_S` and
    CUT where the aircraft has executed the instruction (`turn_done_row`); the flight ends
    with the closing onto the join and the final. ``time_cap_s`` bounds the whole flight
    and `MAX_INSTRUCTION_LEGS` the instruction legs; either cap is reported
    (``capped_by``, and the forecast's ``horizon_capped``)."""
    config = guidance_config(config)
    current = first_anchor(series, anchor, remaining_m)
    legs: list[Forecast] = []
    routes: list[Route] = []
    lengths: list[float] = []
    durations: list[float] = []
    records: list[dict[str, object]] = []
    flown = skipped = incomplete = 0
    capped_by: str | None = None
    leg_index = 0
    start_s = float(series.times[anchor])
    while True:
        order = orders(current, leg_index, legs)
        instruction = order.instruction
        while instruction is not None and not ahead_of(current, instruction):
            skipped += 1
            if not order.retry_on_skip:
                instruction = None
                break
            order = orders(current, leg_index, legs)
            instruction = order.instruction
        if instruction is not None and flown >= MAX_INSTRUCTION_LEGS:
            instruction, capped_by = None, CAPPED_BY_LEGS
        if order.remaining_m is not None:
            current = replace(current, remaining_m=max(float(order.remaining_m), 0.0))
        elapsed = current.time_s - start_s
        budget = min(float(order.budget_s), time_cap_s - elapsed)
        route, leg_plan, _route_join = leg_route(current, instruction, order.d_join_m, skeleton, order.plan)
        if instruction is None:
            duration = max(budget, LEG_MIN_S)
        else:
            duration = max(min(time_to_turn_end_s(route, leg_plan) + TURN_SETTLE_S, max(budget, LEG_MIN_S)), LEG_MIN_S)
        forecast, = fly_routes(
            [current.row], [route], [leg_plan], [current.height_m], config, [duration], [current.time_s], [anchor],
            clamped=[order.capture_clamped], device=device,
        )
        if instruction is not None:
            row = turn_done_row(forecast, instruction)
            if row is None:
                incomplete += 1
            else:
                forecast = cut_rows(forecast, row + 1)
            duration = float(forecast.final_time_s)
        legs.append(forecast)
        routes.append(route)
        lengths.append(float(np.sum(np.hypot(*np.diff(forecast.values[:, [IDX["e"], IDX["n"]]], axis=0).T))))
        durations.append(float(duration))
        records.append({} if order.record is None else dict(order.record))
        leg_index += 1
        if instruction is None:
            break
        if current.time_s + duration - start_s >= time_cap_s:
            capped_by = CAPPED_BY_TIME
            break
        flown += 1
        current = next_anchor(current, forecast, instruction)
    closing = route_time_s(
        routes[-1], v_mid=leg_plan.v_mid_mps, d_decel_m=leg_plan.d_decel_m, v_final=leg_plan.v_final_mps,
        points=leg_plan.speed_points,
    )
    whole = concatenate(legs, anchor, predicted_final_time_s)
    diagnostics = dict(whole.command_hook_diagnostics)
    diagnostics["planCappedBy"] = capped_by or ""
    diagnostics["planTurnsIncomplete"] = float(incomplete)
    whole = replace(whole, horizon_capped=capped_by is not None, command_hook_diagnostics=diagnostics)
    return RolledFlight(whole, routes, flown, skipped, lengths, durations, closing, records, incomplete, capped_by)


def fly_rolling(
    series: FlightSeries,
    anchor: int,
    labels: PlanLabels,
    skeleton: RunwaySkeleton,
    config: TSConfig,
    *,
    instructions: Sequence[Instruction],
    horizon_s: float | None = None,
    device: torch.device | None = None,
) -> RolledFlight:
    """The oracle: fly ``series`` from ``anchor`` by the truth's own label set, one
    instruction at a time — ``instructions`` in path order, each offered once, in turn (one
    the aircraft is on top of is skipped and the next offered). The flight ends with the
    closing onto the join and the final, flown for what is left of ``horizon_s`` (the
    plan's ``T`` plus the oracle's slack by default)."""
    plan, clamped = plan_to_fly(labels, float(series.values[anchor, IDX["u"]]), skeleton, route=ROUTE_WAYPOINTS)
    horizon = float(labels.T_s + max(CLOSING_SLACK_S, CLOSING_SLACK_FRACTION * labels.T_s)) if horizon_s is None else float(horizon_s)
    queue = list(instructions)
    start_s = float(series.times[anchor])

    def orders(current: Anchor, _leg: int, _legs: Sequence[Forecast]) -> LegOrder:
        instruction = queue.pop(0) if queue else None
        return LegOrder(
            plan=plan, d_join_m=labels.d_join_m, instruction=instruction,
            budget_s=horizon - (current.time_s - start_s), retry_on_skip=bool(queue), capture_clamped=clamped,
        )

    return fly_legs(
        series, anchor, skeleton, config, orders=orders, remaining_m=float(labels.remaining_path_at_anchor_m),
        predicted_final_time_s=horizon, time_cap_s=horizon, device=device,
    )


def rolled_history(series: FlightSeries, anchor: int, legs: Sequence[Forecast], config: TSConfig) -> np.ndarray:
    """The plan head's input window at the end of the legs flown so far: the last
    ``seq_len`` rows at ``dt_s`` of the observed track up to the anchor continued by the
    flown rows (each channel interpolated onto the window's own clock). An anchor without
    a full lookback is refused, as `inference.forecast.history_at_anchor` refuses it."""
    if anchor < config.seq_len - 1:
        raise ValueError(f"anchor {anchor} has no full lookback window (needs at least {config.seq_len - 1})")
    times = np.concatenate([np.asarray(series.times[: anchor + 1], dtype=np.float64)] + [leg.times for leg in legs])
    values = np.concatenate([np.asarray(series.values[: anchor + 1], dtype=np.float64)] + [leg.values for leg in legs])
    grid = float(times[-1]) - config.dt_s * np.arange(config.seq_len - 1, -1, -1, dtype=np.float64)
    return np.stack([np.interp(grid, times, values[:, c]) for c in range(values.shape[1])], axis=1)


def capture_height_for(order: PlanOrder, skeleton: RunwaySkeleton) -> tuple[float, bool]:
    """The capture height an order is flown at: inside the glidepath window at its join
    (`plan_to_fly`'s clamp), and whether the clamp moved it."""
    glidepath = max(order.d_join_m, 0.0) * skeleton.glidepath_tan
    clamped = min(max(order.h_capture_m, glidepath - CAPTURE_BELOW_GLIDEPATH_MAX_M), glidepath + CAPTURE_ABOVE_GLIDEPATH_MAX_M)
    return clamped, clamped != order.h_capture_m


def fly_rolling_orders(
    series: FlightSeries,
    anchor: int,
    skeleton: RunwaySkeleton,
    config: TSConfig,
    *,
    order_at: Callable[[Anchor, np.ndarray], PlanOrder],
    device: torch.device | None = None,
) -> RolledFlight:
    """The prediction: fly ``series`` from ``anchor`` by a head's orders, asked afresh at
    every anchor on the window of what was flown (`rolled_history`). ``order_at(anchor,
    window)`` is the head's reading; its first arrival time sets the flight's time cap."""
    window0 = rolled_history(series, anchor, [], config)
    first = order_at(first_anchor(series, anchor, 1.0), window0)

    def orders(current: Anchor, leg: int, legs: Sequence[Forecast]) -> LegOrder:
        # the first order is read once, before the first anchor's remaining path is known
        order = first if leg == 0 else order_at(current, rolled_history(series, anchor, legs, config))
        # a closing from at or inside the join flies the height from where the aircraft
        # is — `leg_route`'s rule, so the capture height here is the order's
        h_capture, clamped = capture_height_for(order, skeleton)
        plan = PlanToFly(order.V_mid_mps, order.d_decel_m, order.V_final_mps, h_capture, ())
        return LegOrder(
            plan=plan, d_join_m=order.d_join_m, instruction=order.instruction,
            budget_s=closing_budget_s(order.T_s),
            remaining_m=order.remaining_m, retry_on_skip=False, capture_clamped=clamped,
            record={**order.to_dict(), "leg": leg, "h_capture_flown_m": h_capture},
        )

    return fly_legs(
        series, anchor, skeleton, config, orders=orders, remaining_m=first.remaining_m,
        predicted_final_time_s=first.T_s, time_cap_s=rolled_time_cap_s(closing_budget_s(first.T_s)), device=device,
    )


# ── receding-horizon, lockstep rolling (design v5.1, §9 step 3(d)) ─────────────
# The unit of rolling is a TIME STEP, not an instruction. Every `LOCKSTEP_S` the policy is
# asked again from the aircraft's current pose and window — the oracle's policy holds the
# truth's next instruction until the aircraft has executed it, the head re-predicts it —
# the route to the instruction in force is re-laid from where the aircraft is, and the
# next step is flown. Every flight of a group is stepped together: one guidance rollout
# per step for the whole group (`LOCKSTEP_SEGMENTS` holds each), one policy call per step.
# Asked once per leg (`fly_legs`), the head's guess at the 60 s anchor for a turn 25–39 km
# away was committed for four minutes and its 2.6 km became 3781 m of rolled ADE; and one
# flight at a time, a full run took 90 min (2026-09-11).

#: The re-ask period, and the guidance holds one step is flown in (`HOLD_S` each).
LOCKSTEP_S = 30.0
LOCKSTEP_SEGMENTS = int(math.ceil(LOCKSTEP_S / HOLD_S))
#: The route in force is re-laid when the order changed materially — its fix (or the
#: closing's join) moved this far, its heading on this much, or fix ↔ none — or the
#: aircraft is this far off the route; otherwise it is trimmed and tracked on. Re-laid
#: every step from a mid-turn pose, the turn-straight-turn builder flipped its turn
#: direction step after step and the aircraft never executed the instruction.
RELAY_FIX_M = 1_000.0
RELAY_HEADING_RAD = math.radians(10.0)
RELAY_OFFSET_M = 1_000.0
#: An aircraft this close to the centreline and this aligned with the course is ON the
#: final: an instruction issued to it there is not flown (the head, re-asked on its own
#: flown rows, pulled nearly every straight-in flight off the final with one).
ON_FINAL_XT_M = 500.0


@dataclass
class FlightState:
    """One flight being stepped: where it is, what it has flown, the instruction in force."""

    series: FlightSeries
    anchor: int
    skeleton: RunwaySkeleton
    current: Anchor
    chunks: list[Forecast] = field(default_factory=list)
    instruction: Instruction | None = None
    #: the last step executed the instruction in force (past its fix, on its heading): the
    #: oracle's policy then offers the next; the head's re-predicts regardless
    instruction_executed: bool = False
    route: Route | None = None            # the route in force (whole; `progress` is where the aircraft is on it)
    progress: int = 0
    route_height_m: float = 0.0           # the height the route in force was laid from (the height law's anchor)
    plan: PlanToFly | None = None         # its schedule
    d_join_m: float | None = None         # the join it was laid to
    phase_routes: list[Route] = field(default_factory=list)    # the first route laid per phase
    phase_lengths_m: list[float] = field(default_factory=list)
    phase_durations_s: list[float] = field(default_factory=list)
    flown: int = 0
    skipped: int = 0
    incomplete: int = 0
    ignored_on_final: int = 0            # instructions not flown because the aircraft was on the final
    executed_on_final: bool = False      # this step's instruction was already behind an aircraft on the final
    records: list[dict[str, object]] = field(default_factory=list)
    steps: int = 0
    end_time_s: float = math.inf
    time_cap_s: float = math.inf
    cap_s: float | None = None            # the flight's time cap from its anchor (None: from its first budget)
    predicted_final_time_s: float = 0.0
    last_plan: PlanToFly | None = None
    capped_by: str | None = None
    done: bool = False

    @property
    def elapsed_s(self) -> float:
        return self.current.time_s - float(self.series.times[self.anchor])


#: A policy answers one step for a group of flights: one order per state, in order.
LockstepPolicy = Callable[[Sequence[FlightState]], Sequence[LegOrder]]


def route_progress(route: Route, start: int, e: float, n: float) -> int:
    """The route point the aircraft has reached: from ``start`` on, the NEAREST point to
    its position, as an absolute index — the guidance's own lookup (`PlanGuidance.
    _route_lookup`), so a step continues exactly where the guidance would. Not the
    first-pass rule: that followed a route that loops before its fix around the loop,
    step after step, where the guidance cuts across to the far side (2026-09-11)."""
    start = min(max(int(start), 0), len(route.points) - 1)
    distance = np.hypot(route.points[start:, 0] - e, route.points[start:, 1] - n)
    return start + int(np.argmin(distance))


def order_changed(state: FlightState, instruction: Instruction | None, d_join_m: float | None) -> bool:
    """Whether an order differs materially from the one the route in force was laid to."""
    if state.route is None or (instruction is None) != (state.instruction is None):
        return True
    if instruction is None:
        before, now = state.d_join_m, d_join_m
        return (before is None) != (now is None) or (before is not None and abs(float(now) - float(before)) > RELAY_FIX_M)
    held = state.instruction
    join_moved = (state.d_join_m is None) != (d_join_m is None) or (
        state.d_join_m is not None and abs(float(d_join_m) - float(state.d_join_m)) > RELAY_FIX_M
    )
    return (
        join_moved
        or math.hypot(instruction.fix_e - held.fix_e, instruction.fix_n - held.fix_n) > RELAY_FIX_M
        or abs(wrap_angle(instruction.heading_out_rad - held.heading_out_rad)) > RELAY_HEADING_RAD
    )


def on_final(state: FlightState) -> bool:
    """The aircraft is established on the final: within `ON_FINAL_XT_M` of the centreline,
    aligned with the course inside the intercept limit, before the threshold."""
    d0, xt0 = state.skeleton.axes(np.array([state.current.e]), np.array([state.current.n]))
    return (
        float(d0[0]) > 0.0 and abs(float(xt0[0])) <= ON_FINAL_XT_M
        and abs(wrap_angle(state.current.heading_rad - state.skeleton.course_rad)) <= INTERCEPT_MAX_RAD
    )


def off_route(state: FlightState) -> bool:
    """Whether the aircraft has drifted over `RELAY_OFFSET_M` from the route in force."""
    if state.route is None:
        return True
    ahead = state.route.points[state.progress:]
    distance = np.hypot(ahead[:, 0] - state.current.e, ahead[:, 1] - state.current.n)
    return float(distance.min()) > RELAY_OFFSET_M


def fly_lockstep(
    states: Sequence[FlightState],
    config: TSConfig,
    *,
    policy: LockstepPolicy,
    step_s: float = LOCKSTEP_S,
    time_caps_s: Sequence[float] | None = None,
    device: torch.device | None = None,
) -> list[RolledFlight]:
    """Step every flight of ``states`` together until each has closed: each step the
    policy's order per flight, the route to its instruction re-laid from its pose, one
    guidance rollout of ``step_s`` for the whole group. A flight ends when its budget
    (the order's, absolute) is spent; ``time_caps_s`` (from the anchor) bounds each flight,
    by default `ROLLED_TIME_CAP_FACTOR` × its first order's budget; an instruction still
    in force at the end is counted incomplete and the cap reported."""
    config = guidance_config(config)
    if time_caps_s is not None:
        for state, cap in zip(states, time_caps_s, strict=True):
            state.cap_s = float(cap)
    while True:
        active = [state for state in states if not state.done]
        if not active:
            break
        orders = list(policy(active))
        if len(orders) != len(active):
            raise ValueError("the policy must answer one order per active flight")
        rows, routes, plans, heights, durations, starts, anchors, clamped, progress = [], [], [], [], [], [], [], [], []
        stepped: list[tuple[FlightState, Instruction | None]] = []
        for state, order in zip(active, orders, strict=True):
            # the route is laid through the fix wherever it is (a fix on top of the
            # aircraft is passed over by `build_route`); an instruction is dropped only
            # once executed (below), never for being near
            instruction = order.instruction
            if instruction is not None and state.flown >= MAX_INSTRUCTION_LEGS:
                instruction, state.capped_by = None, CAPPED_BY_LEGS
            state.executed_on_final = False
            if instruction is not None and on_final(state):
                # on the final: a fix already behind the aircraft was executed (the turn
                # onto the final, its 5° heading match still pending); one ahead is not
                # flown (a spurious fix from the head on its own flown rows)
                if past_fix_abeam(state.current, instruction):
                    state.executed_on_final = True
                else:
                    state.ignored_on_final += 1
                instruction = None
            if order.remaining_m is not None:
                state.current = replace(state.current, remaining_m=max(float(order.remaining_m), 0.0))
            if state.steps == 0:
                state.time_cap_s = state.current.time_s + (rolled_time_cap_s(order.budget_s) if state.cap_s is None else state.cap_s)
                state.predicted_final_time_s = float(order.budget_s if order.arrival_time_s is None else order.arrival_time_s)
            # an order can shorten the budget, never extend it past the first order's
            state.end_time_s = min(state.end_time_s, state.time_cap_s, state.current.time_s + float(order.budget_s))
            # the residual of the budget is flown as it is, never floored past the end
            duration = max(min(float(step_s), state.end_time_s - state.current.time_s), 1e-3)
            new_phase = (
                not state.phase_routes or state.instruction_executed
                or (instruction is None) != (state.instruction is None)
            )
            relaid = new_phase or order_changed(state, instruction, order.d_join_m) or off_route(state)
            if relaid:
                route, leg_plan, _route_join = leg_route(
                    state.current, instruction, order.d_join_m, state.skeleton, order.plan, mid_flight=True,
                )
                state.route, state.plan, state.d_join_m, state.progress = route, leg_plan, order.d_join_m, 0
                state.route_height_m = state.current.height_m
            else:
                route, leg_plan = state.route, state.plan
            if new_phase:
                state.phase_routes.append(route)
                state.phase_lengths_m.append(0.0)
                state.phase_durations_s.append(0.0)
            elif relaid:
                # the phase's route is the one in force, re-laid mid-phase
                state.phase_routes[-1] = route
            state.instruction = instruction
            state.instruction_executed = False
            state.last_plan = leg_plan
            state.records.append({**(order.record or {}), "step": state.steps})
            rows.append(state.current.row)
            routes.append(route)
            plans.append(leg_plan)
            heights.append(state.route_height_m)
            durations.append(duration)
            starts.append(state.current.time_s)
            anchors.append(state.anchor)
            clamped.append(order.capture_clamped)
            progress.append(state.progress)
            stepped.append((state, instruction))
        chunks = fly_routes(
            rows, routes, plans, heights, config, durations, starts, anchors,
            clamped=clamped, n_segments=LOCKSTEP_SEGMENTS, progress=progress, device=device,
        )
        for (state, instruction), chunk in zip(stepped, chunks, strict=True):
            # a step that executes its instruction ends there: the next instruction is
            # laid from the turn's end, not up to a step later along the old route
            executed = False
            if instruction is not None:
                row = turn_done_row(chunk, instruction)
                if row is not None:
                    executed = True
                    if row > 0:
                        chunk = cut_rows(chunk, row + 1)
            state.chunks.append(chunk)
            flown_m = float(np.sum(np.hypot(*np.diff(chunk.values[:, [IDX["e"], IDX["n"]]], axis=0).T)))
            state.phase_lengths_m[-1] += flown_m
            state.phase_durations_s[-1] += float(chunk.final_time_s)
            state.steps += 1
            last = chunk.values[-1]
            geodetic = chunk.geodetic_values[-1]
            state.current = Anchor(
                e=float(last[IDX["e"]]), n=float(last[IDX["n"]]), height_m=float(last[IDX["u"]]),
                heading_rad=float(geodetic[4]), speed_mps=float(geodetic[3]),
                remaining_m=max(state.current.remaining_m - flown_m, 0.0), time_s=float(chunk.times[-1]),
                row=advance_row(state.current.row, chunk),
            )
            # the route in force continues from the point the aircraft reached on it
            state.progress = route_progress(state.route, state.progress, state.current.e, state.current.n)
            if state.executed_on_final:
                state.flown += 1
                state.instruction_executed = True
            if instruction is not None and executed:
                state.flown += 1
                state.instruction_executed = True
                # the schedule's coordinate re-synced to the instruction's own
                state.current = replace(state.current, remaining_m=max(float(instruction.remaining_m), 0.0))
            if state.current.time_s >= state.end_time_s - 1e-6 or cut_at_threshold_crossing(chunk, state.series).truncated_at_threshold:
                # the budget spent, or the threshold crossed on the final: the flight is over
                state.done = True
                if instruction is not None and not state.instruction_executed:
                    state.incomplete += 1
                    state.capped_by = CAPPED_BY_TIME
    out: list[RolledFlight] = []
    for state in states:
        plan = state.plan
        closing = route_time_s(
            state.route, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps,
            points=plan.speed_points,
        )
        whole = concatenate(state.chunks, state.anchor, state.predicted_final_time_s)
        diagnostics = dict(whole.command_hook_diagnostics)
        diagnostics["planCappedBy"] = state.capped_by or ""
        diagnostics["planTurnsIncomplete"] = float(state.incomplete)
        diagnostics["planSteps"] = float(state.steps)
        diagnostics["planIgnoredOnFinal"] = float(state.ignored_on_final)
        whole = replace(whole, horizon_capped=state.capped_by is not None, command_hook_diagnostics=diagnostics)
        out.append(RolledFlight(
            whole, list(state.phase_routes), state.flown, state.skipped, list(state.phase_lengths_m),
            list(state.phase_durations_s), closing, list(state.records), state.incomplete, state.capped_by,
        ))
    return out


def lockstep_states(
    series: Sequence[FlightSeries], anchors: Sequence[int], skeletons: Sequence[RunwaySkeleton],
    remaining_m: Sequence[float],
) -> list[FlightState]:
    return [
        FlightState(item, int(a), sk, first_anchor(item, int(a), float(r)))
        for item, a, sk, r in zip(series, anchors, skeletons, remaining_m, strict=True)
    ]


def truth_lockstep_policy(
    states: Sequence[FlightState], labels: Sequence[PlanLabels], horizons_s: Sequence[float],
) -> LockstepPolicy:
    """The oracle's policy: each flight's own instructions in path order, the one in force
    held until the aircraft has executed it (an instruction not ahead when its turn comes
    is skipped and the next offered); the operating parameters its own plan's."""
    by_state: dict[int, dict] = {}
    for state, lab, horizon in zip(states, labels, horizons_s, strict=True):
        plan, clamped = plan_to_fly(lab, state.current.height_m, state.skeleton, route=ROUTE_WAYPOINTS)
        by_state[id(state)] = {
            "queue": truth_instructions(lab, state.skeleton), "current": None, "plan": plan, "clamped": clamped,
            "d_join": lab.d_join_m, "end": float(state.series.times[state.anchor]) + float(horizon),
        }

    def policy(active: Sequence[FlightState]) -> list[LegOrder]:
        orders = []
        for state in active:
            own = by_state[id(state)]
            if own["current"] is None or state.instruction_executed:
                own["current"] = own["queue"].pop(0) if own["queue"] else None
                while own["current"] is not None and not ahead_of(state.current, own["current"]):
                    state.skipped += 1
                    own["current"] = own["queue"].pop(0) if own["queue"] else None
            orders.append(LegOrder(
                plan=own["plan"], d_join_m=own["d_join"], instruction=own["current"],
                budget_s=own["end"] - state.current.time_s, capture_clamped=own["clamped"],
            ))
        return orders

    return policy


def fly_lockstep_truth(
    series: Sequence[FlightSeries], anchors: Sequence[int], labels: Sequence[PlanLabels],
    skeletons: Sequence[RunwaySkeleton], config: TSConfig, *, horizons_s: Sequence[float],
    step_s: float = LOCKSTEP_S, device: torch.device | None = None,
) -> list[RolledFlight]:
    """The lockstep oracle: every flight's own instructions, re-laid every ``step_s`` from
    where the aircraft is, the group stepped together, each flown for its horizon."""
    states = lockstep_states(series, anchors, skeletons, [lab.remaining_path_at_anchor_m for lab in labels])
    policy = truth_lockstep_policy(states, labels, horizons_s)
    return fly_lockstep(states, config, policy=policy, step_s=step_s, time_caps_s=list(horizons_s), device=device)


def order_to_leg(order: PlanOrder, state: FlightState, step: int) -> LegOrder:
    """A head's order as the leg order one step flies: the capture height inside the
    glidepath window (the height from where the aircraft is when the closing starts at or
    inside the join), the budget the order's arrival time plus the closing slack."""
    h_capture, clamped = capture_height_for(order, state.skeleton)
    plan = PlanToFly(order.V_mid_mps, order.d_decel_m, order.V_final_mps, h_capture, ())
    return LegOrder(
        plan=plan, d_join_m=order.d_join_m, instruction=order.instruction,
        budget_s=closing_budget_s(order.T_s),
        remaining_m=order.remaining_m, capture_clamped=clamped, arrival_time_s=order.T_s,
        record={**order.to_dict(), "h_capture_flown_m": h_capture},
    )


# ── the drawn replay (the validation clock) ────────────────────────────────────
# The loop's checkpoint selection replays a batch every epoch; the rollout costs seconds
# per flight, a drawn route milliseconds. The replay draws the single-step flight — the
# route through the predicted next fix (or straight onto the join) at the predicted
# schedule — on the normalized grid: positions along the route at the schedule's clock,
# the height by the controller's own law (`reference_height`: to the instruction's height
# by the end of its turn, to the capture height at the join, then the glidepath capture),
# velocities from the tangent and the speed. The closure path's "drawn, not rolled out"
# replay is the precedent; the rolled prediction proper is `fly_rolling_orders`.

def draw_order(
    order: PlanOrder, anchor_e: float, anchor_n: float, anchor_u: float, heading_rad: float, speed_mps: float,
    skeleton: RunwaySkeleton, nodes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One order drawn on ``nodes`` equal fractions of its arrival time: ``[nodes, C]``
    channels, the ``[nodes]`` node durations, and the arrival time."""
    anchor = Anchor(anchor_e, anchor_n, anchor_u, heading_rad, speed_mps, order.remaining_m, 0.0, {})
    h_capture, _clamped = capture_height_for(order, skeleton)
    plan = PlanToFly(order.V_mid_mps, order.d_decel_m, order.V_final_mps, h_capture, ())
    route, leg_plan, route_join = leg_route(anchor, order.instruction, order.d_join_m, skeleton, plan)
    to_threshold = route.arc_m <= route.threshold_arc_m + 1e-9
    arc = route.arc_m[to_threshold]
    points = route.points[to_threshold]
    remaining = route.threshold_arc_m - arc
    speed = speed_schedule_mps(
        remaining, v_mid=leg_plan.v_mid_mps, d_decel_m=leg_plan.d_decel_m, v_final=leg_plan.v_final_mps,
        points=leg_plan.speed_points,
    )
    pace = np.maximum(0.5 * (speed[1:] + speed[:-1]), 1.0)
    clock = np.concatenate(([0.0], np.cumsum(np.diff(arc) / pace)))
    T = float(order.T_s)
    t = np.arange(1, nodes + 1, dtype=np.float64) / nodes * T
    # the route's own clock may end before or after T: past the threshold the aircraft is
    # held at it (the record is cut there anyway), short of it the draw runs the route out
    t_route = np.minimum(t, clock[-1])
    e = np.interp(t_route, clock, points[:, 0])
    n = np.interp(t_route, clock, points[:, 1])
    v = np.interp(t_route, clock, speed)
    s = np.interp(t_route, clock, remaining)
    s0 = float(remaining[0])
    s_join = float(remaining[min(route_join, len(remaining) - 1)])
    if order.instruction is None:
        u = np.array([
            reference_height(float(si), s0=s0, s_join=s_join, anchor_height_m=anchor_u, h_capture_m=h_capture,
                             glidepath_tan=skeleton.glidepath_tan)
            for si in s
        ])
    else:
        s_turn = float(remaining[min(route.join_index, len(remaining) - 1)])
        h_turn = leg_plan.h_capture_m
        u = np.array([
            anchor_u + (h_turn - anchor_u) * (s0 - si) / max(s0 - s_turn, 1e-6) if si > s_turn else
            reference_height(float(si), s0=s_turn, s_join=s_join, anchor_height_m=h_turn, h_capture_m=h_capture,
                             glidepath_tan=skeleton.glidepath_tan)
            for si in s
        ])
    de = np.gradient(e, t, edge_order=1) if nodes > 1 else np.zeros_like(e)
    dn = np.gradient(n, t, edge_order=1) if nodes > 1 else np.zeros_like(n)
    du = np.gradient(u, t, edge_order=1) if nodes > 1 else np.zeros_like(u)
    norm = np.maximum(np.hypot(de, dn), 1e-6)
    channels = np.zeros((nodes, len(IDX)), dtype=np.float64)
    channels[:, IDX["e"]], channels[:, IDX["n"]], channels[:, IDX["u"]] = e, n, u
    channels[:, IDX["edot"]], channels[:, IDX["ndot"]], channels[:, IDX["udot"]] = de / norm * v, dn / norm * v, du
    durations = np.full(nodes, T / nodes, dtype=np.float64)
    return channels, durations, T
