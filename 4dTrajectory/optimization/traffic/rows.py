"""Judged losses as smooth NLP rows on the commanded aircraft's nodes (design §5.3).

The judge decides the kind of each loss; the rows against one recorded aircraft use ONE branch, chosen
by :func:`branch_for` and kept by the loop for the rest of the window (a branch changes only from
vertical to horizontal, when an in-trail loss with that aircraft appears):

  * ``horizontal`` — ``(n − n_j(t))² + (e − e_j(t))² ≥ (S·(1 + κ))²`` on the nodes within the row window
    of a loss time: in trail on one final, and the horizontal branch of "radar or vertical" / diagonal;
  * ``vertical`` — ``σ·(h − h_j(t)) ≥ V·(1 + κ)``, the vertical branch of those two, ``σ`` the side the
    commanded aircraft was on;
  * ``landing_after`` — the wake at the threshold (TBL 5-5-2), the commanded aircraft behind a recorded
    leader: its landing time ``≥`` the leader's on the approach clock plus the wake distance at its own
    target speed. Linear in the phase durations; it does not use a branch.

``n_j(t)``, ``e_j(t)``, ``h_j(t)`` are the recorded track at the node's SYMBOLIC time, piecewise linear
(``casadi.pw_lin``) over a slice reaching two row windows beyond the losses. Before the record's first
sample (its entry into the arrival slice) and after its last (its landing) the aircraft is not in the
air: there it holds its end position, and a presence weight ``a(t)`` ramps from 0 to 1 over
:data:`ABSENT_RAMP_S` and relaxes the row fully (``gap² + S'²·a ≥ S'²``,
``σ·(h − h_j) + (V' + 10 km)·a ≥ V'``), so
a re-solve can wait for a landing without the aircraft sweeping across anything. The frozen branch, the
slice and this ramp are the approximations of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
from geokit import FT_M, NM_M

from . import rules
from .check import Conflict, Window

#: The vertical minimum (7110.65BB 4-5-1 a: 1,000 ft), metres — the rules' own value.
VERTICAL_M = rules.FAA_VERTICAL_FT * FT_M

HORIZONTAL, VERTICAL, LANDING_AFTER = "horizontal", "vertical", "landing_after"
#: Seconds over which an aircraft that is not in the air (before its entry, after its landing) stops counting.
ABSENT_RAMP_S = 10.0
#: More than any height difference the NLP allows: its altitude box tops out at 10,000 m
#: (``collocation.components.make_state_bounds`` — MIRROR, a literal there).
_HEIGHT_SPAN_M = 10_000.0
#: The loss kinds that allow either branch (the rules: horizontal OR vertical separation).
_EITHER = (rules.RADAR_OR_VERTICAL, rules.DIAGONAL)


@dataclass(frozen=True)
class Branch:
    family: str           # HORIZONTAL or VERTICAL
    sign: float = 1.0     # VERTICAL: +1 the commanded aircraft above, −1 below


@dataclass(frozen=True)
class RowSpec:
    """One loss as a row family; ``bound`` is metres (horizontal / vertical) or seconds (landing_after)."""

    family: str
    t_s: float
    other: int
    bound: float
    sign: float = 1.0


def branch_for(conflicts: list[Conflict]) -> Branch:
    """The branch for the position losses with one aircraft: horizontal when any is in trail (only
    horizontal separation counts there), else the larger-margin branch at the tightest step."""
    for c in conflicts:
        if c.kind not in (rules.IN_TRAIL, *_EITHER):
            raise ValueError(f"no position branch for a {c.kind!r} loss")
    if any(c.kind == rules.IN_TRAIL for c in conflicts):
        return Branch(HORIZONTAL)
    worst = min(conflicts, key=lambda c: max(c.distance_m / c.required_m, c.vertical_m / VERTICAL_M))
    if worst.distance_m / worst.required_m >= worst.vertical_m / VERTICAL_M:
        return Branch(HORIZONTAL)
    return Branch(VERTICAL, 1.0 if worst.above else -1.0)


def row_spec(conflict: Conflict, window: Window, *, margin: float, branch: Branch | None) -> RowSpec:
    """The row for a loss the commanded aircraft answers for; ``branch`` is that aircraft's (None for
    the wake at the threshold, which has none)."""
    if conflict.kind == rules.AT_THRESHOLD:
        leader = window.recorded[conflict.other]
        a_own = window.rules.along_nm[window.runway] * NM_M
        a_leader = window.rules.along_nm[leader.runway] * NM_M
        landed_s = leader.end_utc_s - window.t0_utc_s
        return RowSpec(LANDING_AFTER, conflict.t_s, conflict.other,
                       landed_s + (a_own - a_leader + conflict.required_m * (1.0 + margin)) / window.rules.speed_mps)
    if branch.family == HORIZONTAL:
        return RowSpec(HORIZONTAL, conflict.t_s, conflict.other, conflict.required_m * (1.0 + margin))
    return RowSpec(VERTICAL, conflict.t_s, conflict.other, VERTICAL_M * (1.0 + margin), branch.sign)


def _slice(window: Window, other: int, start_s: float, end_s: float):
    """The recorded track over ``[start, end]`` (seconds from the window start) in the window frame, one
    sample beyond each end, held over the absent ramp where the slice reaches the record's entry or
    landing: ``(t, n, e, h, absent)``, ``absent`` 1 on a ramp knot, else 0."""
    flight = window.recorded[other]
    t = flight.t_utc_s - window.t0_utc_s
    keep = np.flatnonzero((t >= start_s) & (t <= end_s))
    keep = np.arange(max(keep[0] - 1, 0), min(keep[-1] + 2, len(t)))
    n, e = window.frame.to_ne(flight.lat_deg[keep], flight.lon_deg[keep])
    t, h, absent = t[keep], flight.alt_m[keep], np.zeros(len(keep))
    if keep[0] == 0:
        t, n, e, h, absent = (np.r_[t[0] - ABSENT_RAMP_S, t], np.r_[n[0], n], np.r_[e[0], e], np.r_[h[0], h],
                              np.r_[1.0, absent])
    if keep[-1] == len(flight.t_utc_s) - 1:
        t, n, e, h, absent = (np.r_[t, t[-1] + ABSENT_RAMP_S], np.r_[n, n[-1]], np.r_[e, e[-1]], np.r_[h, h[-1]],
                              np.r_[absent, 1.0])
    return tuple(ca.DM(v) for v in (t, n, e, h, absent))


def _absent(tk, t, absent):
    """The presence weight at ``tk``: 0 in the air, 1 a ramp past an end, held beyond (no extrapolation)."""
    return ca.fmin(ca.fmax(ca.pw_lin(tk, t, absent), 0.0), 1.0)


def extra_rows(specs: list[RowSpec], window: Window, node_times_s, *, row_window_s: float):
    """The ``extra_rows`` callback of ``CollocationOptimizer`` for ``specs``. ``node_times_s`` are the
    node times of the solve the specs were judged on: a spec gets rows on the nodes within
    ``row_window_s`` of its loss time. One node gets ONE row per family and aircraft: the largest bound."""
    node_times_s = np.asarray(node_times_s)
    landing = [s.bound for s in specs if s.family == LANDING_AFTER]
    bounds: dict[tuple, float] = {}
    span: dict[int, tuple[float, float]] = {}
    for spec in specs:
        if spec.family == LANDING_AFTER:
            continue
        for k in np.flatnonzero(np.abs(node_times_s - spec.t_s) <= row_window_s):
            key = (spec.family, spec.other, spec.sign, int(k))
            bounds[key] = max(bounds.get(key, spec.bound), spec.bound)
        lo, hi = span.get(spec.other, (spec.t_s, spec.t_s))
        span[spec.other] = (min(lo, spec.t_s), max(hi, spec.t_s))
    slices = {other: _slice(window, other, lo - 2.0 * row_window_s, hi + 2.0 * row_window_s)
              for other, (lo, hi) in span.items()}

    def rows(nodes, times):
        out = [(times[-1], max(landing), ca.inf)] if landing else []
        for (family, other, sign, k), bound in bounds.items():
            t, n, e, h, absent = slices[other]
            node, tk = nodes[k], times[k]
            away = _absent(tk, t, absent)
            if family == HORIZONTAL:
                gap2 = (node[0] - ca.pw_lin(tk, t, n)) ** 2 + (node[1] - ca.pw_lin(tk, t, e)) ** 2
                out.append((gap2 + bound ** 2 * away, bound ** 2, ca.inf))
            else:
                out.append((sign * (node[2] - ca.pw_lin(tk, t, h)) + (bound + _HEIGHT_SPAN_M) * away, bound, ca.inf))
        return out

    return rows
