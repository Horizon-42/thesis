"""Fly plans: a batch of plans through the guidance on the shared rollout, as `Forecast`s.

The plan path's forecast entry point, the shape `outputs.control.forecast.forecast_control_batch`
has: the per-flight physical context, the dense query grid, one rollout under the command
hook, one `Forecast` per flight with the schedule FLOWN in newtons and the hook's own
per-flight counts. Here the hook is the whole controller and the network's schedule is
zeros (the guidance overrides every command); the schedule's total is the plan's ``T``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

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
from ts_transformer.inference.forecast import Forecast
from ts_transformer.outputs.dynamics import rollout as control_rollout
from ts_transformer.outputs.dynamics.context import dynamics_arrays
from ts_transformer.outputs.dynamics.hooks import per_flight_hook_diagnostics
from ts_transformer.outputs.dynamics.rollout import padded_dense_queries
from ts_transformer.outputs.envelope import physical_controls
from ts_transformer.outputs.plan.extractors import PlanLabels
from ts_transformer.outputs.plan.guidance.controller import PlanGuidance, PlanToFly
from ts_transformer.outputs.plan.guidance.route import Route, build_route, route_time_s, speed_schedule_mps
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
    dynamics = {
        name: torch.from_numpy(np.stack([row[name] for row in rows])).to(device)
        for name in rows[0]
    }
    anchor_heights = np.array([float(item.values[a, IDX["u"]]) for item, a in zip(series, anchors, strict=True)])
    routes = [
        route_for(item, a, lab, skeleton, row["initial_state"], route=route)
        for item, a, lab, skeleton, row in zip(series, anchors, labels, skeletons, rows, strict=True)
    ]
    flown = [plan_to_fly(lab, h, sk, route=route) for lab, h, sk in zip(labels, anchor_heights, skeletons, strict=True)]
    plans = [plan for plan, _clamped in flown]
    clamped = [clamped for _plan, clamped in flown]
    totals = np.array(
        [lab.T_s for lab in labels] if durations_s is None else list(durations_s), dtype=np.float64
    )
    if np.any(totals <= 0.0):
        raise ValueError("every plan needs a positive duration")
    n_segments = n_segments_for(float(totals.max()))
    durations = np.repeat(totals[:, None] / n_segments, n_segments, axis=1)
    offsets, padded, valid = padded_dense_queries(durations, config.control_rollout_integrator_dt_s)
    hook = PlanGuidance(config, dynamics, routes, plans, anchor_heights)
    zeros = torch.zeros((len(series), n_segments, 3), dtype=torch.float64, device=device)
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
    for row, was_clamped, total in zip(diagnostics, clamped, totals, strict=True):
        row["planCaptureHeightClamped"] = float(was_clamped)
        row["planHoldS"] = float(total / n_segments)
    forecasts: list[Forecast] = []
    for row, (item, a, row_offsets) in enumerate(zip(series, anchors, offsets, strict=True)):
        count = len(row_offsets)
        final_time_s = float(row_offsets[-1])
        forecasts.append(Forecast(
            times=float(item.times[a]) + row_offsets,
            values=query_channels[row, :count],
            normalized_progress=row_offsets / final_time_s,
            anchor=a,
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
    times = [
        route_time_s(route, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps,
                     points=plan.speed_points)
        for route, plan in zip(routes, plans, strict=True)
    ]
    assert all(math.isfinite(t) for t in times)
    return forecasts, routes, times
