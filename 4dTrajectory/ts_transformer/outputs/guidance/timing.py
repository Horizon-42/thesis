"""The time closure (design §4.6; v5.4, §9 step 4): an assigned arrival time flown by the
guidance — the speed lever first, then the path, and what neither absorbs reported as X.

Per ask, on the route in force: the route's time at the schedule (`route_time_s`, the
whole route to the threshold) against the assigned remaining time. ONE factor scales the
speed the plan may choose — the HELD speed: the instruction's at its fix on an instruction
leg; on a closing, where the schedule holds the anchor's own speed to the deceleration
point, a new held speed reached now at the deceleration rate ("reduce speed to …", the
instruction ATC gives) — never `V_final` (the approach speed, the aircraft's); it is
bisected between the FLOOR (the larger of `V_final` and the stall
margin × the 1 g stall speed at the flight's mass and altitude — the controller's
`floor_speed` criterion, read once at the state the plan starts from) and `SPEED_MAX_MPS`.
A delay the floor cannot absorb goes to the PATH: the route re-laid with the schedule
coordinate lengthened by what the residual needs at the floor speed (the route builder lays
the length as a hold or a dog-leg, `build_route(pre_final_m=)`), one secant correction.
What remains is X, signed: positive a delay the envelope and the builder's reach cannot
hold, negative an advance the speed cannot make. The speed lever comes first because it is
the procedure-conforming one and ATC's own for small amounts; the stretch is what a hook
could not decide and a plan can (§1, L3.e). The identity behind the path lever — the delay
the remaining path cannot absorb at the floor speed, as extra path flown at that speed — is
the trombone hook's (`outputs/constraints/trombone.py`, its `ΔL = V_e·T_r − D`), here decided
at planning time on the route's true length and laid by the route builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ts_transformer.geometry.flyability import isa_density
from ts_transformer.outputs.constraints.speed_floor import stall_speed_mps
from ts_transformer.outputs.guidance.controller import STALL_MARGIN, PlanToFly
from ts_transformer.outputs.guidance.route import DECEL_RATE_MPS2, OVERRUN_M, Route, route_time_s

#: The fastest ground speed a plan may ask for (was the plan head's label ceiling).
SPEED_MAX_MPS = 140.0

#: An assignment met within this is closed; the lockstep's own step is 30 s and the
#: guidance tracks the schedule to a few seconds.
TIME_TOLERANCE_S = 2.0
#: The speed factor's bisection: 14 halvings of a [0.5, 2] range resolve it to ~1e-4.
BISECTION_STEPS = 14
#: The path lever's lays per ask: the builder lays path in QUANTA (a dog-leg needs a turn
#: radius's worth, a hold the return leg — measured: nothing at all below ~2 km of request,
#: then 30–45 km at once), so the request is bracketed — doubled while nothing is laid,
#: halved when the lay overshoots the time — and a lay that would make the flight LATER
#: than assigned is never taken (the speed would then have to climb back over a longer
#: detour, the failure the floor sizing exists to avoid).
STRETCH_PROBES = 5
#: The most path the closure asks the builder to add at one ask: twice the builder's
#: own offset reach (`MAX_STRETCH_OFFSET_M`, 30 km of dog-leg offset lays ~60 km more path).
MAX_STRETCH_M = 60_000.0
#: A lay that adds less than this is no stretch: the builder could not lay the path asked
#: for (a closing already heading at the join inside the RNP box, an instruction whose
#: heading points at the join), and the lever is exhausted.
MIN_STRETCH_LAID_M = 100.0

#: A lay: the schedule coordinate lengthened by ``extra_m`` → the route and its leg plan
#: (`forecast.leg_route` at a lengthened anchor).
Lay = Callable[[float], tuple[Route, PlanToFly]]


@dataclass(frozen=True)
class TimeClosure:
    """One ask's closure: what was asked, what the plan as given needed, the levers, the
    time the closed plan needs and X."""

    assigned_remaining_s: float
    plan_time_s: float
    speed_factor: float
    stretch_m: float                 # the path ADDED as laid (the route's length grew by this)
    closed_time_s: float
    stretch_requested_m: float = 0.0  # what the lay was asked for (the builder lays what it can)

    @property
    def unabsorbed_s(self) -> float:
        """X: the assigned remaining time less what the closed plan can fly (signed)."""
        return self.assigned_remaining_s - self.closed_time_s

    def record(self) -> dict[str, float]:
        return {
            "assigned_remaining_s": self.assigned_remaining_s, "plan_time_s": self.plan_time_s,
            "speed_factor": self.speed_factor, "stretch_m": self.stretch_m,
            "stretch_requested_m": self.stretch_requested_m, "unabsorbed_s": self.unabsorbed_s,
        }


def stall_floor_mps(row: dict[str, np.ndarray]) -> float:
    """`STALL_MARGIN` × the 1 g stall speed at the flight's mass and altitude — the dynamics
    row the step rolls from (`dynamics_arrays`' layout: the geodetic state's altitude and
    mass, the airframe's area and `Cl_max`) — `floor_speed`'s criterion as one scalar."""
    altitude_m, mass_kg = float(row["initial_state"][2]), float(row["initial_state"][6])
    area, cl_max = float(row["aero_params"][0]), float(row["aero_params"][1])
    return STALL_MARGIN * float(stall_speed_mps(1.0, mass_kg, isa_density(altitude_m), area, cl_max))


def held_speed_mps(plan: PlanToFly) -> float:
    """The speed the lever scales: the last speed point's — the instruction's at its fix on an
    instruction leg; on a closing the anchor's own, which `speed_schedule_mps` holds to the
    deceleration point — or `V_mid` where the plan carries no points (the whole-path form)."""
    return plan.speed_points[-1][1] if plan.speed_points else plan.v_mid_mps


def decel_point_for(plan: PlanToFly, held_mps: float) -> float | None:
    """The deceleration point a held speed needs: at least the path that bleeds it off to
    `V_final` at `DECEL_RATE_MPS2` — the plan's own where that is already farther out, and
    None where the plan never decelerates (the track never slower than the target speed).
    Left where the head put it, a faster held speed crossed the threshold above the
    approach speed (measured: ~126 m/s against a 77 m/s `V_final` at the lever's ceiling)."""
    if plan.d_decel_m is None:
        return None
    needed = max(held_mps * held_mps - plan.v_final_mps * plan.v_final_mps, 0.0) / (2.0 * DECEL_RATE_MPS2)
    return max(plan.d_decel_m, needed)


def scaled(plan: PlanToFly, factor: float) -> PlanToFly:
    """The plan with its held speed scaled by ``factor``: `V_mid` and every speed point but
    the first (the anchor's own speed, the state); on a closing — one point, the anchor's —
    a second point is added, the new held speed reached at `DECEL_RATE_MPS2` from the
    anchor's and held to the deceleration point ("reduce speed to …" now). `V_final`, the
    approach speed, is never scaled; the deceleration point moves out to where the new held
    speed can still be bled off to it (`decel_point_for`)."""
    if not plan.speed_points:
        held = plan.v_mid_mps * factor
        return replace(plan, v_mid_mps=held, d_decel_m=decel_point_for(plan, held))
    (r0, v0), rest = plan.speed_points[0], plan.speed_points[1:]
    if rest:
        points = ((r0, v0),) + tuple((r, v * factor) for r, v in rest)
    else:
        v_new = v0 * factor
        d_change = abs(v0 * v0 - v_new * v_new) / (2.0 * DECEL_RATE_MPS2)
        points = ((r0, v0), (max(r0 - d_change, 0.0), v_new))
    return replace(plan, v_mid_mps=plan.v_mid_mps * factor, speed_points=points, d_decel_m=decel_point_for(plan, points[-1][1]))


def plan_time_s(route: Route, plan: PlanToFly, start: int = 0) -> float:
    """The time the plan needs to fly ``route`` from its point ``start`` to the threshold."""
    return route_time_s(
        route, v_mid=plan.v_mid_mps, d_decel_m=plan.d_decel_m, v_final=plan.v_final_mps, points=plan.speed_points,
        start=start,
    )


def close_speed(
    route: Route, plan: PlanToFly, remaining_s: float, *, v_floor_mps: float, v_max_mps: float = SPEED_MAX_MPS,
    start: int = 0,
) -> tuple[PlanToFly, float]:
    """The speed lever on a fixed route, flown from its point ``start``: the factor between
    the floor and the maximum whose route time meets ``remaining_s`` (the time falls as the
    factor rises, so a bisection); at a bound, the bound. A faster held speed moves the
    deceleration point out with it (`scaled`), so the threshold is crossed at `V_final`
    whatever the factor. Returns the scaled plan and its factor."""
    held = held_speed_mps(plan)
    low = min(max(v_floor_mps, plan.v_final_mps) / held, v_max_mps / held)
    high = v_max_mps / held
    if plan_time_s(route, scaled(plan, low), start) <= remaining_s:
        return scaled(plan, low), low          # even the floor is too fast: the path's turn
    if plan_time_s(route, scaled(plan, high), start) >= remaining_s:
        return scaled(plan, high), high        # even the maximum is too slow: X < 0
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if plan_time_s(route, scaled(plan, mid), start) > remaining_s:
            low = mid          # too slow at mid: faster
        else:
            high = mid
    factor = 0.5 * (low + high)
    return scaled(plan, factor), factor


def reanchored(plan: PlanToFly, route: Route) -> PlanToFly:
    """The plan as laid for a STRETCHED route: its first speed point moved to the route's own
    path to go, so the schedule reads the extra path as path to fly at the plan's speeds —
    left at the schedule coordinate the lay was asked from, the anchor's speed is held over
    the whole stretch and the reduction begins only where the stretch has been flown off."""
    if not plan.speed_points:
        return plan
    v0, rest = plan.speed_points[0][1], plan.speed_points[1:]
    return replace(plan, speed_points=((route.remaining_m(0), v0),) + tuple(rest))


def close_time(
    route: Route, plan: PlanToFly, remaining_s: float, *, v_floor_mps: float, lay: Lay, start: int = 0,
) -> tuple[TimeClosure, Route, PlanToFly, PlanToFly]:
    """The closure for one ask: the speed lever on the route in force; then, for a delay the
    floor cannot absorb, the path lever — ``lay(extra_m)`` re-lays the route with that much
    more path asked of the builder, the request bracketed over `STRETCH_PROBES` lays (doubled
    while the builder lays nothing, halved between the last lay that left the flight EARLY
    and one that made it LATE), sized at the floor speed; a lay that makes the flight later
    than assigned is never taken; the speed is re-closed on the stretched route. ``start``
    is the point the aircraft has reached on the route in force (a lay starts at the
    aircraft, from 0). Returns the closure, the route (the stretched one where the path
    lever was used), the closed plan and the plan AS LAID for that route (what the next ask
    scales from); at or past the threshold the plan is returned unchanged with X the
    assigned remaining time."""
    if route.remaining_m(start) <= 0.0:
        return TimeClosure(remaining_s, 0.0, 1.0, 0.0, 0.0), route, plan, plan
    base_time = plan_time_s(route, plan, start)
    closed, factor = close_speed(route, plan, remaining_s, v_floor_mps=v_floor_mps, start=start)
    time = plan_time_s(route, closed, start)
    residual = remaining_s - time
    requested = laid = 0.0
    base = plan
    if residual > TIME_TOLERANCE_S:
        to_fly = route.remaining_m(start)
        extra = min(residual * held_speed_mps(closed), MAX_STRETCH_M)
        short, long = 0.0, None            # the bracket: the largest request still EARLY, the smallest LATE
        best = None                        # (residual, added, extra, route, plan as laid, time) — never late
        for _ in range(STRETCH_PROBES):
            route_x, plan_x = lay(extra)
            added = route_x.length_m - OVERRUN_M - to_fly
            if added < MIN_STRETCH_LAID_M:
                # nothing laid at this request: a larger one may lay the quantum the
                # builder has, until the reach runs out
                if extra >= MAX_STRETCH_M:
                    break
                short = extra
                extra = min(extra * 2.0, MAX_STRETCH_M) if long is None else 0.5 * (extra + long)
                continue
            plan_x = reanchored(plan_x, route_x)
            floor_x, _f = close_speed(route_x, plan_x, math.inf, v_floor_mps=v_floor_mps)
            time_x = plan_time_s(route_x, floor_x)
            residual_x = remaining_s - time_x
            if residual_x < -TIME_TOLERANCE_S:
                long = extra                  # too much path even at the floor: never taken
            else:
                if best is None or residual_x < best[0]:
                    best = (residual_x, added, extra, route_x, plan_x, time_x)
                if residual_x <= TIME_TOLERANCE_S:
                    break
                short = extra
            if long is None:
                extra = min(extra * 2.0, MAX_STRETCH_M)
            else:
                extra = 0.5 * (short + long)
            if extra <= short or (long is not None and long - short < MIN_STRETCH_LAID_M):
                break
        if best is not None:
            _residual, laid, requested, route, base, _floor_time = best
            closed, factor = close_speed(route, base, remaining_s, v_floor_mps=v_floor_mps)
            time = plan_time_s(route, closed)
    return TimeClosure(remaining_s, base_time, factor, laid, time, requested), route, closed, base


__all__ = [
    "BISECTION_STEPS", "Lay", "MAX_STRETCH_M", "MIN_STRETCH_LAID_M", "STRETCH_PROBES", "TIME_TOLERANCE_S",
    "TimeClosure", "close_speed", "close_time", "decel_point_for", "held_speed_mps", "plan_time_s", "reanchored",
    "scaled", "stall_floor_mps",
]
