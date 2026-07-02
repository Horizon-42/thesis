"""Reusable NLP components: bounds, the altitude floor, control-effort / smoothness costs, the
terminal-bank constraint, the solver factory, and geodetic-state ↔ decision-vector conversions.

Everything here is optimizer-agnostic (no phase/segment knowledge) so it is shared by the
optimizer and by the benchmark / comparison scripts.
"""

import math

import casadi as ca
import numpy as np

from aerodynamic_model.casadi_coordinates_converter import radians_expr
from aerodynamic_model.common import GeodeticState

from .schemes import STATE_DIM, _V, _PSI, _GAMMA


# Selectable NLP solver backend.  ``ipopt`` (interior point) is the robust
# default; ``sqpmethod`` (SQP + active-set QP) can be faster on benign /
# warm-started solves but is less robust on hard cold starts.
_SOLVER_BACKENDS = ("ipopt", "sqpmethod")
_DEFAULT_SOLVER_BACKEND = "ipopt"


def _make_nlp_solver(nlp, solver_backend, verbose=False):
    """Build the NLP solver for the chosen backend (see ``_SOLVER_BACKENDS``).

    The exact Hessian of the point-mass dynamics is nonconvex, so the SQP
    backend regularises it (``convexify_strategy``) to keep its QP subproblem
    solvable, and tells the QP not to throw on a failed subproblem.

    The solver is **silent by default** — no IPOPT banner, no per-iteration
    table, no CasADi timing line — so a batch of solves doesn't flood the
    console. Pass ``verbose=True`` to restore the full solver log for debugging.
    """
    if solver_backend == "sqpmethod":
        return ca.nlpsol('solver', 'sqpmethod', nlp, {
            'qpsol': 'qpoases',
            'qpsol_options': {'printLevel': 'none', 'error_on_fail': False},
            'convexify_strategy': 'regularize',
            'max_iter': 100,
            'print_iteration': verbose,
            'print_header': verbose,
            'print_time': verbose,
        })
    if verbose:
        return ca.nlpsol('solver', 'ipopt', nlp)
    return ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0,   # no per-iteration table
        'ipopt.sb': 'yes',        # no IPOPT startup banner
        'print_time': False,      # no CasADi wall-time line
    })


# --------------------------------------------------------------------------
# Bounds and frame-conversion helpers
# --------------------------------------------------------------------------

def make_control_bounds(max_thrust: float, min_load_factor: float, max_load_factor: float):
    # Identical envelope to ``casadi_optimizer.make_control_bounds`` so
    # the two CasADi optimisers compete on the same control space.
    T_min, T_max = 0.0, max_thrust
    mu_min, mu_max = -ca.pi / 4.0, ca.pi / 4.0
    n_cmd_min, n_cmd_max = min_load_factor, max_load_factor
    return [T_min, mu_min, n_cmd_min], [T_max, mu_max, n_cmd_max]


def make_state_bounds(min_altitude: float, min_velocity: float):
    # lat/lon are bounded loosely at the whole-globe scale (radians) —
    # far larger than any realistic terminal-area trajectory, but enough
    # to keep IPOPT inside a well-defined search box.  cos(lat) in the
    # RHS stays away from its pole singularity inside these bounds.
    lat_min, lat_max = -ca.pi / 2.0 + 0.01, ca.pi / 2.0 - 0.01
    lon_min, lon_max = -ca.pi, ca.pi
    alt_min, alt_max = min_altitude, 10_000.0
    V_min, V_max = min_velocity, 1_000.0
    # Heading is cyclic: bounding it to a single [-pi, pi] branch makes any
    # turn that crosses the +-180 deg cut infeasible (e.g. -135 deg -> +135 deg,
    # a short ~90 deg turn).  We bound it loosely over several turns so the
    # target heading can be unwrapped to the shortest turn (see
    # ``_unwrap_target_heading``).  psi has no singularity in the RHS.
    psi_min, psi_max = -3.0 * ca.pi, 3.0 * ca.pi
    gamma_min, gamma_max = -radians_expr(6.0), radians_expr(15.0)
    return (
        [lat_min, lon_min, alt_min, V_min, psi_min, gamma_min],
        [lat_max, lon_max, alt_max, V_max, psi_max, gamma_max],
    )


# The altitude state lower bound is a NUMERICAL search box, not a physical/operational limit:
# the glidepath window + the pinned terminal already enforce the real low-altitude protection.
# It only has to sit safely BELOW the destination threshold (the trajectory's lowest point) so
# the pinned terminal state never contradicts it. It MUST therefore be anchored to the target,
# not an absolute MSL constant: the target altitude is ``field_elevation + threshold_crossing``,
# so an absolute floor (the old ``threshold_crossing + 10`` ≈ 25 m) wrongly sat ABOVE the target
# for near-sea-level airports (KMSY threshold ≈ 16 m), making every solve genuinely infeasible.
# The margin is generous barrier breathing room: a floor at/just-below the target puts IPOPT's
# log-barrier ``-μ·ln(alt - lb)`` right on the pinned terminal and stiffens the solve, so the box
# is kept well clear of it (the floor never binds — it is not an operational minimum).
ALTITUDE_FLOOR_MARGIN_M = 300.0


def altitude_floor_m(target_altitude_m: float) -> float:
    """Altitude state lower bound: a generous margin below the destination threshold."""
    return target_altitude_m - ALTITUDE_FLOOR_MARGIN_M


# Boundary bookkeeping is now just a units conversion: ``GeodeticState``
# carries lat/lon in degrees, the RHS and decision variables in radians.
# There is no tangent-plane projection any more, so nothing here can
# introduce its own modelling drift.


def _geodetic_state_to_decision(state: GeodeticState) -> list[float]:
    """Convert a geodetic state to the 7-vector NLP parameter
    ``(lat_rad, lon_rad, alt, V, psi, gamma, m)``.

    Only lat/lon need converting (degrees → radians).  The third entry is
    the geodetic altitude directly — there is no tangent-plane
    u-coordinate any more, and the ISA density model reads it as the true
    geodetic altitude, so there is no curvature discrepancy to absorb.
    """
    return [
        math.radians(state.latitude),
        math.radians(state.longitude),
        state.altitude,
        state.V,
        state.psi,
        state.gamma,
        state.m,
    ]


def _unwrap_target_heading(initial_param: list[float], target_param: list[float]) -> list[float]:
    """Shift the target heading by multiples of 2π so it lies within π of the
    initial heading — i.e. the aircraft turns the SHORT way to the target
    course, without the path having to cross the ±180° branch cut that the
    (widened) heading box would otherwise trap it on.  Returns a copy of
    ``target_param`` with only the psi entry adjusted.
    """
    unwrapped = list(target_param)
    dpsi = initial_param[_PSI] - target_param[_PSI]
    unwrapped[_PSI] = target_param[_PSI] + 2.0 * math.pi * round(dpsi / (2.0 * math.pi))
    return unwrapped


_TARGET_STATE_STEP_S = 3.0
_MAX_STATE_SUBSTEPS = 16

# Terminal realised-bank cap (degrees), always applied. Set generously high
# (e.g. 89) at construction to effectively disable it.
_DEFAULT_MAX_TERMINAL_BANK_DEG = 5.0


def select_state_substeps(max_duration: float, segment_num: int) -> int:
    """Pick M so the state step ``T/(N*M)`` is about ``_TARGET_STATE_STEP_S``.

    ``max_duration`` is the horizon upper bound; if the solved time comes
    out shorter the state step is only finer, never coarser.
    """
    control_step = max_duration / segment_num
    substeps = round(control_step / _TARGET_STATE_STEP_S)
    return max(1, min(_MAX_STATE_SUBSTEPS, int(substeps)))


_G = 9.81  # gravity (m/s^2), matches the dynamics model


def terminal_bank_constraint_expr(state_nodes, start_state, state_h, max_bank_rad):
    """Bound the realised coordinated bank at the terminal, from STATE only.

    Bank angle is the control ``mu`` in this model, but the *realised* bank in
    a coordinated turn is fixed by how fast the heading turns:
    ``tan(mu) = V cos(gamma) * psi_dot / g``.  We reconstruct ``psi_dot`` at
    the terminal from the last two STATE sub-nodes (``Δpsi / state_h``) — so
    this is a pure state constraint, never touching the control ``mu``.

    Returns ``(expr, lb, ub)`` for the scalar inequality ``lb <= expr <= ub``
    with ``expr = V·cos(gamma)·psi_dot = g·tan(mu_eff)`` and
    ``|expr| <= g·tan(max_bank)`` — i.e. ``|mu_eff| <= max_bank``.  Keeping
    ``expr`` in the ``g·tan`` form avoids an ``atan`` in the NLP.
    """
    terminal = state_nodes[-1]
    prev = state_nodes[-2] if len(state_nodes) >= 2 else start_state[:STATE_DIM]
    psi_dot = (terminal[_PSI] - prev[_PSI]) / state_h
    expr = terminal[_V] * ca.cos(terminal[_GAMMA]) * psi_dot
    bound = _G * math.tan(max_bank_rad)
    return expr, -bound, bound


# Per-control smoothness weights on the scaled segment-to-segment change
# (thrust, bank μ, load n).  Bank and load factor are attitude-driven and
# physically smooth in real flight, so they are penalised most; thrust gets a
# small weight (engine spool already limits its rate).
_DEFAULT_SMOOTHNESS_WEIGHTS = (0.1, 1.0, 1.0)


def _control_scales(aircraft_meta):
    return (
        aircraft_meta['max_thrust'],
        ca.pi / 2.0,
        max(abs(aircraft_meta['min_load_factor']), abs(aircraft_meta['max_load_factor'])),
    )


def _scaled_control_cost(seg_controls, aircraft_meta):
    """Mean squared scaled control effort over the N control segments."""
    scale_thrust, scale_mu, scale_n = _control_scales(aircraft_meta)
    cost = ca.SX(0)
    for uk in seg_controls:
        scaled = ca.vertcat(uk[0] / scale_thrust, uk[1] / scale_mu, uk[2] / scale_n)
        cost += ca.dot(scaled, scaled)
    return cost / len(seg_controls)


def _control_smoothness_cost(seg_controls, aircraft_meta, weights):
    """Mean squared weighted, scaled control change between adjacent segments.

    ``weights = (w_thrust, w_bank, w_load)`` emphasise bank (μ) and load
    factor (n) — the smooth, attitude-driven quantities in real flight — over
    thrust.  Penalising ``u_{k+1} - u_k`` discourages segment-to-segment jumps.
    """
    if len(seg_controls) < 2:
        return ca.SX(0)
    scale_thrust, scale_mu, scale_n = _control_scales(aircraft_meta)
    w_thrust, w_bank, w_load = weights
    cost = ca.SX(0)
    for prev, cur in zip(seg_controls[:-1], seg_controls[1:]):
        d = ca.vertcat(
            w_thrust * (cur[0] - prev[0]) / scale_thrust,
            w_bank * (cur[1] - prev[1]) / scale_mu,
            w_load * (cur[2] - prev[2]) / scale_n,
        )
        cost += ca.dot(d, d)
    return cost / (len(seg_controls) - 1)


_DEFAULT_TIME_REGULARIZATION = 1e-3
_DEFAULT_MIN_DURATION_S = 1.0
