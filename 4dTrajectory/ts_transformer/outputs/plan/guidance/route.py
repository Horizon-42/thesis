"""The route builder: a plan's route parameters laid as a path in the chart (design §4.2).

The plan says WHERE the flight joins the final (``d_join``), from WHICH side (``side``) and
HOW MUCH path it flies before that (``L_pre``); the skeleton says where the final is. The
route is: the shortest turn-straight-turn path from the anchor pose to the join pose
(`outputs.closure.geometry.dubins_csc`, at the aircraft's own turn radius), lengthened to
``L_pre`` by a dog-leg on the plan's side when the shortest path is shorter than the plan,
then the final leg to the threshold. What the shortest path does not allow — a plan whose
``L_pre`` is shorter than the minimum, or longer than the dog-leg can lay — is reported on
the route (``shortfall_m``), never silently absorbed.

The time closure (§4.6) is a reading, not a search, at this step: the time the speed
schedule needs to fly the route (`route_time_s`) against the plan's ``T`` is the
unabsorbable residual ``X``; the assigned-time stretch (§9 step 4) builds on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ts_transformer.geometry.dubins import (
    PATH_STEP_M,
    arc_points,
    chart_from_axes_np,
    dubins_csc,
    segment_points,
    unit_vector,
)
from ts_transformer.geometry.final_approach_geometry import ALIGNMENT_MAX_DEG
from ts_transformer.geometry.flyability import G as GRAVITY_MPS2
from ts_transformer.outputs.plan.extractors import ON_COURSE_FIX_M
from ts_transformer.outputs.plan.skeleton import RNP_HALF_WIDTH_M, RunwaySkeleton

#: How far past the threshold the route continues, so the tracker has a leg to follow
#: while the rollout's last hold crosses the plane (the record is cut at the crossing).
OVERRUN_M = 500.0
#: The dog-leg's widest lateral offset: past this the plan's `L_pre` is reported as a
#: shortfall rather than flown as a loop.
MAX_STRETCH_OFFSET_M = 30_000.0
#: A `L_pre` within this of the shortest path is the shortest path (no dog-leg).
STRETCH_TOLERANCE_M = 50.0
#: A flight established at the anchor converges onto the centreline over this many times
#: its offset (an intercept under ~20°), at least this far.
CONVERGE_RATIO = 3.0
CONVERGE_MIN_M = 500.0
#: The route's turns are laid at the bank the tracker flies comfortably, below its 25° cap,
#: at the AIRCRAFT's own speed — a turn laid at a capped approach speed (the closure
#: decoder's rule) is one a faster aircraft cannot follow.
ROUTE_BANK_RAD = math.radians(20.0)


def route_turn_radius_m(speed_mps: float) -> float:
    return max(float(speed_mps), 40.0) ** 2 / (GRAVITY_MPS2 * math.tan(ROUTE_BANK_RAD))
#: The join may be flown at any intercept angle up to the on-final gate's alignment limit
#: (the standard 30°): the ALIGNED join where the plan's length affords it, else the
#: smallest intercept whose path fits — a join pose aligned with the course from a pose
#: beside it inside two turn radii is a loop, an intercept is a turn, and a downwind
#: flight's turns onto the final lay ~130 m of path per degree of intercept (2026-09-10:
#: five discrete steps left 5 of 48 flights 1.8–5.6 km short). Searched on the coarse
#: grid, refined on the fine one toward the course.
INTERCEPT_MAX_RAD = math.radians(ALIGNMENT_MAX_DEG)
INTERCEPT_STEP_RAD = math.radians(2.0)
INTERCEPT_FINE_STEP_RAD = math.radians(0.1)
#: A turn-straight-turn path whose two arcs together sweep more than this is a LOOP (or a
#: teardrop of two half-circles), not a route: from a pose beside the centreline heading
#: in, the aligned join at the anchor's turn radius is a full circle (2026-09-10: 30 of 73
#: routed straight-in flights, up to 18 km — the tracker cut through them, so the flown
#: path was right and the route's length and time were not). A downwind's base and final
#: turns are 180° together; a flight heading away from the join turns up to ~250°. Such a
#: pose takes the straight chord onto the join instead, where the chord's intercept and
#: the turn onto it are inside the alignment limit; a loop is laid only when nothing else
#: reaches the join at all.
LOOP_SWEEP_RAD = math.radians(300.0)
KIND_DIRECT = "dubins"
KIND_DOWNWIND = "downwind+dubins"
KIND_STRETCHED = "dubins+dog-leg"
KIND_FINAL_ONLY = "final"
#: The fixed-K waypoints representation (design §3b / §8, 2026-09-11): the polyline through
#: the plan's fly-by fixes with the corners rounded at the schedule's radius, then the
#: turns onto the join.
KIND_WAYPOINTS = "waypoints"
#: The pre-final length is laid first as the design's route (§4.2): the current heading
#: held to a turn point, then the turns onto the join — the downwind a controller extends.
#: A plan the hold cannot lay within this tolerance falls back to the dog-leg, and a
#: dog-leg that cannot either leaves the shortest path with the gap reported.
HOLD_TOLERANCE_M = 500.0
#: Both searches (the hold's length, the dog-leg's offset) walk a coarse grid and then a
#: fine one about its best: the laid length is piecewise smooth in either parameter — a
#: Dubins type switch jumps it by up to a turn's circumference — so a bisection lands on
#: the wrong branch (measured 2026-09-10: 10–23 km MORE than the plan on 3 of 48 flights)
#: and one coarse grid alone misses by up to twice its step.
SEARCH_STEP_M = 250.0
SEARCH_FINE_STEP_M = 10.0


@dataclass(frozen=True)
class Route:
    """A planned path in the chart, the anchor first and the threshold (plus the overrun)
    last, with the arc length it lays before the join and what the plan asked for."""

    points: np.ndarray          # [M, 2] chart (e, n)
    arc_m: np.ndarray           # [M] cumulative arc length from the anchor
    join_index: int             # the first point of the final leg
    pre_final_m: float          # arc length to the join, as laid
    requested_pre_final_m: float
    shortfall_m: float          # requested − laid (positive: the plan asked for more path than laid)
    stretch_offset_m: float     # the dog-leg's lateral offset (0 = the shortest path)
    intercept_rad: float        # the join's signed intercept angle (0 = aligned with the course)
    kind: str

    @property
    def length_m(self) -> float:
        return float(self.arc_m[-1])

    @property
    def threshold_arc_m(self) -> float:
        """The arc length at the threshold (the overrun is past it)."""
        return float(self.arc_m[-1]) - OVERRUN_M

    def remaining_m(self, index: int) -> float:
        """Path still to fly to the threshold from route point ``index``."""
        return self.threshold_arc_m - float(self.arc_m[index])

    def curvature_rad_per_m(self, index: int) -> float:
        """The route's signed curvature at point ``index`` (CCW positive, the dynamics' turn
        sign): the heading change over the two adjacent steps, per metre."""
        lo, hi = max(0, index - 1), min(len(self.points) - 1, index + 1)
        if hi - lo < 2:
            return 0.0
        a = math.atan2(*(self.points[index] - self.points[lo])[::-1])
        b = math.atan2(*(self.points[hi] - self.points[index])[::-1])
        turn = math.atan2(math.sin(b - a), math.cos(b - a))
        span = float(self.arc_m[hi] - self.arc_m[lo]) / 2.0
        return turn / span if span > 0.0 else 0.0


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _arc(points: np.ndarray) -> np.ndarray:
    steps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    return np.concatenate(([0.0], np.cumsum(steps)))


def _heading(points: np.ndarray, index: int) -> float:
    a = points[max(0, index - 1)]
    b = points[min(len(points) - 1, index + 1)]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _join_pose(skeleton: RunwaySkeleton, d_join: float) -> tuple[np.ndarray, float]:
    e, n = chart_from_axes_np(np.array([d_join]), np.array([0.0]), skeleton.course_rad)
    return np.array([float(e[0]) + skeleton.target_e, float(n[0]) + skeleton.target_n]), skeleton.course_rad


def _final_leg(skeleton: RunwaySkeleton, d_join: float) -> np.ndarray:
    d = np.arange(d_join - PATH_STEP_M, -OVERRUN_M, -PATH_STEP_M)
    d = np.concatenate((d, [-OVERRUN_M]))
    e, n = chart_from_axes_np(d, np.zeros_like(d), skeleton.course_rad)
    return np.stack([e + skeleton.target_e, n + skeleton.target_n], 1)


def _shortest_to_join(p0, h0, join, course, radius, limit_m: float, end_radius: float):
    """The turn-straight-turn path onto the join pose: the ALIGNED join (the aircraft arrives
    on the centreline heading down the ``course``, inside the design corridor by
    construction) when it is not longer than ``limit_m``; else the smallest intercept whose
    path fits; else the shortest of all (the plan's length is then a shortfall the route
    reports). ``course`` is always the runway course — never a heading a previous choice
    bent. ``(points, heading at the join)``, or None."""
    def candidates(angles, max_sweep=LOOP_SWEEP_RAD):
        found = []
        for intercept in angles:
            path = dubins_csc(
                p0, h0, join, course + float(intercept), radius, end_radius=end_radius, max_sweep_rad=max_sweep,
            )
            if path is not None:
                found.append((float(_arc(path)[-1]), abs(float(intercept)), path, course + float(intercept)))
        return found

    def chord():
        """The straight chord onto the join, when the aircraft is already heading in."""
        heading = math.atan2(join[1] - p0[1], join[0] - p0[0])
        intercept = _wrap(heading - course)
        if abs(intercept) > INTERCEPT_MAX_RAD or abs(_wrap(heading - h0)) > INTERCEPT_MAX_RAD:
            return []
        points = segment_points(p0, join, PATH_STEP_M)
        return [(float(_arc(points)[-1]), abs(intercept), points, heading)] if len(points) > 1 else []

    def pick(pool):
        within = [c for c in pool if c[0] <= limit_m + STRETCH_TOLERANCE_M]
        if within:
            return min(within, key=lambda c: (c[1], c[0])), True
        return min(pool, key=lambda c: (c[0], c[1])), False

    grid = np.arange(-INTERCEPT_MAX_RAD, INTERCEPT_MAX_RAD + 1e-9, INTERCEPT_STEP_RAD)
    coarse = candidates(grid) + chord()
    if not coarse:
        # nothing reaches the join without a loop: the loop is the last resort
        coarse = candidates(grid, max_sweep=None)
    if not coarse:
        return None
    best, fits = pick(coarse)
    if fits and best[1] > 1e-9:
        # toward the course: the smallest intercept that still fits, between the best grid
        # angle and its neighbour nearer the aligned join
        angle = best[3] - course
        nearer = angle - math.copysign(INTERCEPT_STEP_RAD, angle)
        steps = int(round(INTERCEPT_STEP_RAD / INTERCEPT_FINE_STEP_RAD)) + 1
        best, _fits = pick(candidates(np.linspace(nearer, angle, steps)) + [best])
    _length, _angle, path, heading = best
    return path, heading

def _held_heading_path(p0, h0, join, course, radius, hold_m: float, limit_m: float, end_radius: float):
    """The current heading held for ``hold_m``, then the turns onto the join under what
    ``limit_m`` leaves after the hold (the join's intercept lays the residual, as
    `_shortest_to_join` chooses it). ``(points, heading at the join)``, or None."""
    turn = p0 + hold_m * np.array([math.cos(h0), math.sin(h0)])
    rest = _shortest_to_join(turn, h0, join, course, radius, limit_m=max(0.0, limit_m - hold_m), end_radius=end_radius)
    if rest is None:
        return None
    path, heading = rest
    if hold_m <= 0.0:
        return path, heading
    return np.concatenate([segment_points(p0, turn, PATH_STEP_M), path[1:]]), heading

def _closest_length(lay, upper_m: float, target_m: float):
    """The parameter on ``[0, upper_m]`` whose laid path (``lay(x)`` → ``(points, heading)``,
    or None) is closest in length to ``target_m`` without exceeding it by more than
    `HOLD_TOLERANCE_M`: the coarse grid, then the fine grid about its best.
    ``(gap, parameter, (points, heading))``, or None when nothing lays."""
    def best_over(values, best):
        for value in values:
            laid = lay(float(value))
            if laid is None:
                continue
            length = float(_arc(laid[0])[-1])
            if length > target_m + HOLD_TOLERANCE_M:
                continue
            gap = abs(length - target_m)
            if best is None or gap < best[0]:
                best = (gap, float(value), laid)
        return best

    best = best_over(np.arange(0.0, upper_m + SEARCH_STEP_M, SEARCH_STEP_M), None)
    if best is None:
        return None
    low, high = max(0.0, best[1] - SEARCH_STEP_M), best[1] + SEARCH_STEP_M
    return best_over(np.arange(low, high + SEARCH_FINE_STEP_M, SEARCH_FINE_STEP_M), best)

def _via_path(p0, h0, join, course, radius, offset: float, side: float, skeleton: RunwaySkeleton,
              limit_m: float, via_radius: float, end_radius: float):
    """Anchor → via → join, the via displaced ``offset`` to ``side`` of the chord's midpoint,
    the turn at the via at ``via_radius``, the join's intercept chosen as `_shortest_to_join`
    chooses it under what ``limit_m`` leaves after the first leg. ``(points, heading at
    the join)``, or None."""
    mid = 0.5 * (p0 + join)
    # + right of the inbound course, the sign `side` and `xt` share
    right = np.array([math.sin(skeleton.course_rad), -math.cos(skeleton.course_rad)])
    via = mid + side * offset * right
    h_via = math.atan2(join[1] - via[1], join[0] - via[0])
    first = dubins_csc(p0, h0, via, h_via, radius, end_radius=via_radius)
    if first is None:
        return None
    second = _shortest_to_join(
        via, h_via, join, course, via_radius, limit_m=max(0.0, limit_m - float(_arc(first)[-1])),
        end_radius=end_radius,
    )
    if second is None:
        return None
    path, heading = second
    return np.concatenate([first, path[1:]]), heading

def build_route(
    anchor_e: float, anchor_n: float, anchor_heading_rad: float, anchor_speed_mps: float,
    *, d_join_m: float, side: int, pre_final_m: float, skeleton: RunwaySkeleton,
    join_at_anchor: bool = False, speed_at: Callable[[float], float] | None = None,
    waypoints: Sequence[tuple[float, float]] | None = None,
    waypoint_speeds: Sequence[tuple[float, float]] | None = None,
) -> Route:
    """Lay the plan's route from the anchor pose.

    ``d_join_m`` is placed on the centreline at that distance to go; ``side`` (+1 right,
    −1 left, 0 straight-in — a stretch then goes to the anchor's own side); ``pre_final_m``
    is what the plan wants flown before the join. A flight already established at the
    anchor (``join_at_anchor``) has no pre-final path: its route is the final from its own
    distance to go, not a turn onto a join pose it is already at. Every turn is sized at
    the speed flown where it is: the first at the anchor's, the later ones at what
    ``speed_at(remaining path in m)`` — the plan's speed schedule — says at the turn point,
    the via and the join (the anchor's speed throughout when not given). A downwind
    flight's base and final turns are flown well below its anchor speed, and at the
    anchor's radius the turn-straight-turn lengths jump by a whole circumference exactly
    where the real path lies (2026-09-10: 5 of 48 flights 2–10 km short). With
    ``waypoints`` (the fixed-K representation: the fly-by fix of each of the path's
    turns, in path order) the pre-final path is the polyline through them with the
    corners rounded at the schedule's radius — at each fix's own speed where
    ``waypoint_speeds`` (``(remaining path, ground speed)`` per fix) is given — and from
    the last the turns onto the join under what the plan's length leaves.
    """
    p0 = np.array([anchor_e, anchor_n], dtype=np.float64)
    d0, xt0 = skeleton.axes(np.array([anchor_e]), np.array([anchor_n]))
    converge_m = max(CONVERGE_RATIO * abs(float(xt0[0])), CONVERGE_MIN_M)
    inside_box = abs(float(xt0[0])) <= RNP_HALF_WIDTH_M
    ahead_m = float(d0[0]) - d_join_m
    if join_at_anchor or (inside_box and -CONVERGE_MIN_M <= ahead_m <= converge_m):
        # established already, or — inside the RNP box — the plan's join closer ahead than
        # the converging leg the tracker needs (a join a few hundred metres ahead from a
        # pose beside the centreline is a teardrop at any turn radius — 29 of 73 routed
        # straight-in flights, 12–24 km of route for 200–1600 m of plan, 2026-09-10):
        # converge onto the centreline over a gentle leg (an intercept of at most ~20°,
        # at least a hold's travel), then the final — never a lateral step
        meet = max(float(d0[0]) - converge_m, PATH_STEP_M)
        final = _final_leg(skeleton, meet)
        converge = segment_points(p0, final[0], PATH_STEP_M)
        points = np.concatenate([converge, final[1:]])
        arc = _arc(points)
        intercept = _wrap(math.atan2(final[0][1] - p0[1], final[0][0] - p0[0]) - skeleton.course_rad)
        return Route(
            points=points, arc_m=arc, join_index=len(converge), pre_final_m=0.0,
            requested_pre_final_m=float(pre_final_m), shortfall_m=float(pre_final_m),
            stretch_offset_m=0.0, intercept_rad=intercept, kind=KIND_FINAL_ONLY,
        )
    join, course = _join_pose(skeleton, max(d_join_m, PATH_STEP_M))
    radius = route_turn_radius_m(anchor_speed_mps)
    speed = (lambda _remaining: float(anchor_speed_mps)) if speed_at is None else speed_at
    end_radius = route_turn_radius_m(speed(d_join_m))

    def radius_after(laid_m: float) -> float:
        """The turn radius ``laid_m`` along a pre-final path of the plan's length."""
        return route_turn_radius_m(speed(max(pre_final_m - laid_m, 0.0) + d_join_m))

    if waypoints:
        return _waypoints_route(
            p0, anchor_heading_rad, radius, radius_after, end_radius, waypoints, join, course,
            pre_final_m, d_join_m, skeleton,
            speeds=None if waypoint_speeds is None else [v for _r, v in waypoint_speeds],
        )
    shortest = _shortest_to_join(p0, anchor_heading_rad, join, course, radius, limit_m=pre_final_m, end_radius=end_radius)
    if shortest is None:
        raise ValueError("no turn-straight-turn path from the anchor to the join")
    direct, heading_at_join = shortest
    laid = direct
    offset = 0.0
    kind = KIND_DIRECT
    minimum = float(_arc(direct)[-1])
    if pre_final_m > minimum + STRETCH_TOLERANCE_M:
        # the design's route first: the heading held to a turn point, then the turns onto
        # the join — the hold that lays the plan's length; the dog-leg only where the hold
        # leaves more than the tolerance. Either is taken when it lays the plan's length
        # CLOSER than the shortest path does, and never longer than the plan by more than
        # the tolerance (`_closest_length`): a shortfall is reported, an extra is never
        # flown.
        gap = pre_final_m - minimum
        best_hold = _closest_length(
            lambda hold: _held_heading_path(
                p0, anchor_heading_rad, join, course, radius_after(hold), hold,
                limit_m=pre_final_m, end_radius=end_radius,
            ),
            pre_final_m, pre_final_m,
        )
        if best_hold is not None and best_hold[0] < gap:
            gap, (laid, heading_at_join), kind = best_hold[0], best_hold[2], KIND_DOWNWIND
        if gap > HOLD_TOLERANCE_M:
            if side == 0:
                side = 1 if float(xt0[0]) >= 0.0 else -1
            best = _closest_length(
                lambda offset_m: _via_path(
                    p0, anchor_heading_rad, join, course, radius, offset_m, float(side), skeleton,
                    limit_m=pre_final_m, via_radius=radius_after(0.5 * pre_final_m), end_radius=end_radius,
                ),
                MAX_STRETCH_OFFSET_M, pre_final_m,
            )
            if best is not None and best[0] < gap:
                gap, offset, (laid, heading_at_join), kind = best[0], best[1], best[2], KIND_STRETCHED
    pre_final = np.asarray(laid, dtype=np.float64)
    final = _final_leg(skeleton, max(d_join_m, PATH_STEP_M))
    points = np.concatenate([pre_final, final])
    arc = _arc(points)
    join_index = len(pre_final)
    laid_m = float(arc[join_index - 1])
    return Route(
        points=points, arc_m=arc, join_index=join_index, pre_final_m=laid_m,
        requested_pre_final_m=float(pre_final_m), shortfall_m=float(pre_final_m) - laid_m,
        stretch_offset_m=float(offset), intercept_rad=_wrap(heading_at_join - course), kind=kind,
    )


def _waypoints_route(p0, h0, first_radius, radius_after, end_radius, waypoints, join, course,
                     pre_final_m: float, d_join_m: float, skeleton: RunwaySkeleton,
                     speeds: Sequence[float] | None = None) -> Route:
    """The pre-final path through the plan's fly-by waypoints: the polyline anchor → fixes
    → join, each fix's corner replaced by the arc tangent to both legs at the schedule's
    radius there (shrunk to fit a short leg — the tracker then cuts the corner and the
    cap counts it), the anchor's own heading joined onto the first leg by a Dubins path.
    A last fix ON the centreline before the plan's join is the turn onto the final: its
    corner is rounded onto the course like any other and the join is where that arc ends
    (a Dubins onto a join pose behind the corner took a 20° intercept where the aligned
    join needed an S, and the tracker overshot the centreline by 130–210 m, 2026-09-11).
    Otherwise, from the last arc, the turns onto the join ALIGNED wherever that needs no
    loop (`_shortest_to_join` with no length budget: the fixes already lay the plan's
    length, and an intercept here is a corridor entry the tracker overshoots). Then the
    final. A fix on top of its predecessor is passed over."""
    fixes = [np.asarray(fix, dtype=np.float64) for fix in waypoints]
    fix_speeds = [None] * len(fixes) if speeds is None else [float(v) for v in speeds]
    onto_final = False
    if fixes:
        d_last, xt_last = skeleton.axes(fixes[-1][:1], fixes[-1][1:])
        # a fix at the plan's join IS the join (a rolled closing lays its polyline onto it)
        if abs(float(xt_last[0])) <= ON_COURSE_FIX_M and PATH_STEP_M < float(d_last[0]) <= d_join_m:
            onto_final = True
            fixes[-1] = _join_pose(skeleton, float(d_last[0]))[0]   # the fix, on the centreline
    vertices = [np.array(p0, dtype=np.float64)]
    vertex_speeds: list[float | None] = [None]
    for fix, fix_speed in zip(fixes, fix_speeds, strict=True):
        if float(np.hypot(*(fix - vertices[-1]))) >= PATH_STEP_M:
            vertices.append(fix)
            vertex_speeds.append(fix_speed)
    # the polyline runs on to the threshold past a fix onto the final, so that corner's
    # outgoing leg IS the course; else to the join pose
    vertices.append(np.array([skeleton.target_e, skeleton.target_n]) if onto_final else np.asarray(join, dtype=np.float64))
    pieces = [vertices[0][None, :]]
    pos, heading, laid = vertices[0], float(h0), 0.0
    i = 1
    while i < len(vertices) - 1:
        legs = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(vertices[:-1], vertices[1:])]
        fix, h_in, h_out = vertices[i], legs[i - 1], legs[i]
        theta = _wrap(h_out - h_in)
        radius = radius_after(laid) if vertex_speeds[i] is None else route_turn_radius_m(vertex_speeds[i])
        room = 0.5 * min(float(np.hypot(*(fix - vertices[i - 1]))), float(np.hypot(*(vertices[i + 1] - fix))))
        tangent = radius * math.tan(0.5 * abs(theta))
        if tangent > room:
            tangent = room
            radius = room / math.tan(0.5 * abs(theta))
        arc_start = fix - tangent * unit_vector(h_in)
        if i == 1:
            # the anchor's own heading onto the first leg — a first fix that cannot be
            # turned onto without a loop is passed over (the route ran 27 km through such
            # a loop, 2026-09-11)
            leg = dubins_csc(pos, heading, arc_start, h_in, first_radius, end_radius=radius, max_sweep_rad=LOOP_SWEEP_RAD)
            if leg is None and len(vertices) > 3:
                del vertices[1], vertex_speeds[1]
                continue
            if leg is None:
                leg = dubins_csc(pos, heading, arc_start, h_in, first_radius, end_radius=radius)
            if leg is None:
                raise ValueError("no turn-straight-turn path from the anchor onto the first leg")
            pieces.append(leg[1:])
            laid += float(_arc(leg)[-1])
        else:
            pieces.append(segment_points(pos, arc_start, PATH_STEP_M)[1:])
            laid += float(np.hypot(*(arc_start - pos)))
        if abs(theta) > 1e-6:
            sign = 1.0 if theta > 0.0 else -1.0
            centre = arc_start + radius * np.array([-sign * math.sin(h_in), sign * math.cos(h_in)])
            arc = arc_points(centre, radius, h_in - sign * math.pi / 2, theta)
            pieces.append(arc[1:])
            pos, laid = arc[-1], laid + radius * abs(theta)
        else:
            pos = arc_start
        heading = h_out
        i += 1
    if onto_final and len(vertices) > 2:
        # the join is where the last corner's arc ends, on the centreline heading down it
        d_end, _xt_end = skeleton.axes(pos[:1], pos[1:])
        d_join_m = max(float(d_end[0]), PATH_STEP_M)
        pos = _join_pose(skeleton, d_join_m)[0]
        pieces[-1] = np.concatenate([pieces[-1][:-1], pos[None, :]])
        heading_at_join = course
    else:
        closing = _shortest_to_join(
            pos, heading, join, course, radius_after(laid), limit_m=math.inf, end_radius=end_radius,
        )
        if closing is None:
            raise ValueError("no turn-straight-turn path from the last waypoint to the join")
        path, heading_at_join = closing
        pieces.append(path[1:])
    pre_final = np.concatenate(pieces)
    points = np.concatenate([pre_final, _final_leg(skeleton, max(d_join_m, PATH_STEP_M))])
    arc_m = _arc(points)
    join_index = len(pre_final)
    laid_m = float(arc_m[join_index - 1])
    return Route(
        points=points, arc_m=arc_m, join_index=join_index, pre_final_m=laid_m,
        requested_pre_final_m=float(pre_final_m), shortfall_m=float(pre_final_m) - laid_m,
        stretch_offset_m=0.0, intercept_rad=_wrap(heading_at_join - course), kind=KIND_WAYPOINTS,
    )


#: The deceleration from `V_mid` to `V_final`, once it starts, is flown at this rate
#: (`V² = V_mid² − 2 a Δs`, then `V_final` held): a linear ramp over the whole remaining
#: path reached the threshold 13 s early on the straight-in stratum and 36 s early on the
#: vectored one (2026-09-10) — real flights slow first and then hold the final speed.
DECEL_RATE_MPS2 = 0.5


def speed_schedule_mps(remaining_m, *, v_mid: float, d_decel_m: float | None, v_final: float,
                       points: Sequence[tuple[float, float]] = ()):
    """The plan's speed as a function of the path still to fly: ``v_mid`` held to the
    deceleration point, then a `DECEL_RATE_MPS2` deceleration to ``v_final``, held to the
    threshold. A plan with no deceleration point (never slower than the target speed + the
    margin on the track) holds ``v_mid`` throughout. With ``points`` (``(remaining path,
    speed)`` — the anchor's and each fix's, the waypoints route) the speed runs through
    them, held before the first, and past the last decelerates at the same rate from the
    last point's speed (from the deceleration point where that is earlier) to ``v_final``
    — or holds the last speed where it is already below."""
    remaining = np.asarray(remaining_m, dtype=np.float64)
    if points:
        far_to_near = sorted(points, key=lambda point: -point[0])
        r = np.array([point[0] for point in far_to_near], dtype=np.float64)
        v = np.array([point[1] for point in far_to_near], dtype=np.float64)
        through = np.interp(-remaining, -r, v)
        r_last, v_last = float(r[-1]), float(v[-1])
        start = r_last if d_decel_m is None else min(r_last, float(d_decel_m))
        flown = np.clip(start - remaining, 0.0, None)
        decelerated = np.sqrt(np.clip(v_last * v_last - 2.0 * DECEL_RATE_MPS2 * flown, 0.0, None))
        after = np.maximum(decelerated, min(v_final, v_last))
        return np.where(remaining >= r_last, through, after)
    if d_decel_m is None:
        return np.full(remaining.shape, v_mid)
    flown = np.clip(float(d_decel_m) - remaining, 0.0, None)
    decelerated = np.sqrt(np.clip(v_mid * v_mid - 2.0 * DECEL_RATE_MPS2 * flown, 0.0, None))
    return np.where(remaining > float(d_decel_m), v_mid, np.maximum(decelerated, v_final))


def route_time_s(route: Route, *, v_mid: float, d_decel_m: float | None, v_final: float,
                 points: Sequence[tuple[float, float]] = (), start: int = 0) -> float:
    """The time the speed schedule needs to fly the route to the threshold (the time
    closure's reading; §4.6): ``∫ ds / V(s)`` over the laid path from route point ``start``
    — the point the aircraft has reached on a route kept in force (`FlightState.progress`);
    read from 0 on a route the aircraft is still flying, the closure saw a time that never
    fell as the aircraft flew and pushed the speed to its maximum (2026-09-12)."""
    to_threshold = route.arc_m <= route.threshold_arc_m + 1e-9
    to_threshold[:max(int(start), 0)] = False
    arc = route.arc_m[to_threshold]
    remaining = route.threshold_arc_m - arc
    speed = speed_schedule_mps(remaining, v_mid=v_mid, d_decel_m=d_decel_m, v_final=v_final, points=points)
    steps = np.diff(arc)
    pace = 0.5 * (speed[1:] + speed[:-1])
    return float(np.sum(steps / np.maximum(pace, 1.0)))
