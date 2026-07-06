"""Reusable NLP components: bounds, the altitude floor, control-effort / smoothness costs, the
terminal-bank constraint, the solver factory, and geodetic-state ↔ decision-vector conversions.

Everything here is optimizer-agnostic (no phase/segment knowledge) so it is shared by the
optimizer and by the benchmark / comparison scripts.
"""

import math
import os

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


# IPOPT iteration ceiling. IPOPT's own default is 3000; we set it EXPLICITLY so every
# solve is guaranteed to terminate (with Maximum_Iterations_Exceeded) — before this the
# option was unset and a crawling high-density NLP (e.g. an over-meshed constrained
# solve) could grind for hours while the backend's solve lock queued every request.
DEFAULT_MAX_ITERATIONS = 3000

# Optional HSL linear solver, opt-in via environment so the repo default stays portable
# (the committed default is MUMPS — bundled with the casadi wheel, no license). On a machine
# with a compiled CoinHSL library, set:
#   AEROVIZ_IPOPT_LINSOL=ma57     (or ma27)   — the HSL solver to use for IPOPT's KKT step
#   AEROVIZ_IPOPT_HSLLIB=/abs/path/to/libcoinhsl.dylib   — the library IPOPT dlopens at runtime
# MA57/MA27 are typically 2–4x faster than MUMPS on these small-medium sparse KKT systems.
# The casadi wheel's IPOPT is built with the runtime HSL loader (the `hsllib` option), so this
# needs NO rebuild of IPOPT/casadi — only the CoinHSL library. See docs (build guide).
_IPOPT_LINEAR_SOLVER = os.environ.get("AEROVIZ_IPOPT_LINSOL", "mumps").strip() or "mumps"
_IPOPT_HSLLIB = (os.environ.get("AEROVIZ_IPOPT_HSLLIB") or "").strip()


def _ipopt_linear_solver_options() -> dict:
    """IPOPT linear-solver options from the environment (empty = the default MUMPS)."""
    if _IPOPT_LINEAR_SOLVER == "mumps":
        return {}
    opts = {"ipopt.linear_solver": _IPOPT_LINEAR_SOLVER}
    if _IPOPT_HSLLIB:
        opts["ipopt.hsllib"] = _IPOPT_HSLLIB   # the CoinHSL lib IPOPT loads at runtime
    return opts


def _make_nlp_solver(nlp, solver_backend, verbose=False, max_iterations=DEFAULT_MAX_ITERATIONS):
    """Build the NLP solver for the chosen backend (see ``_SOLVER_BACKENDS``).

    The exact Hessian of the point-mass dynamics is nonconvex, so the SQP
    backend regularises it (``convexify_strategy``) to keep its QP subproblem
    solvable, and tells the QP not to throw on a failed subproblem.

    The solver is **silent by default** — no IPOPT banner, no per-iteration
    table, no CasADi timing line — so a batch of solves doesn't flood the
    console. Pass ``verbose=True`` to restore the full solver log for debugging.
    ``max_iterations`` caps the IPOPT iteration count (the termination guarantee).
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
    linsol = _ipopt_linear_solver_options()
    if verbose:
        return ca.nlpsol('solver', 'ipopt', nlp, {'ipopt.max_iter': int(max_iterations), **linsol})
    return ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.max_iter': int(max_iterations),
        'ipopt.print_level': 0,   # no per-iteration table
        'ipopt.sb': 'yes',        # no IPOPT startup banner
        'print_time': False,      # no CasADi wall-time line
        **linsol,
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


# The altitude state lower bound is a REAL operational floor anchored to the LANDING
# TARGET: an approach trajectory never needs to fly below the altitude it lands at, and
# anything lower is the min-time "dive for speed" pathology. The old generous 300 m
# margin was documented as "a numerical box that never binds" — empirically false: real
# batches RODE the bound (60%+ of solved trajectories dipped below field elevation
# mid-flight, to exactly target − 300 m, before climbing back to the procedure).
# Anchoring to the TARGET, not an absolute MSL constant, remains essential: the target
# altitude is ``field_elevation + threshold_crossing``, so an absolute floor (the very
# old ``threshold_crossing + 10`` ≈ 25 m) wrongly sat ABOVE the target for near-sea-level
# airports (KMSY threshold ≈ 16 m), making every solve genuinely infeasible. The small
# margin keeps IPOPT's log-barrier ``-μ·ln(alt - lb)`` finite at the pinned terminal
# (which sits exactly ``margin`` above the bound). A start below the floor is bad input
# data and fails loudly as an infeasible boundary condition (landing trajectories start
# ABOVE their target).
ALTITUDE_FLOOR_MARGIN_M = 5.0


def altitude_floor_m(target_altitude_m: float) -> float:
    """Altitude state lower bound: a small slack below the landing target."""
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


def unwrap_angle(value_rad: float, toward_rad: float) -> float:
    """``value_rad`` shifted by the multiple of 2π that brings it within π of ``toward_rad``.

    THE one angle-unwrap of the optimizer: the target-heading unwrap, the route-chained ψ
    branch, the branch-aware join course and the per-phase heading guesses all go through it.
    The decision ψ is an ACCUMULATING real (a full turn changes it by 2π), so "which 2π branch"
    is always resolved by unwrapping toward a reference value.
    """
    return value_rad + 2.0 * math.pi * round((toward_rad - value_rad) / (2.0 * math.pi))


def _unwrap_target_heading(initial_param: list[float], target_param: list[float]) -> list[float]:
    """Shift the target heading by multiples of 2π so it lies within π of the
    initial heading — i.e. the aircraft turns the SHORT way to the target
    course, without the path having to cross the ±180° branch cut that the
    (widened) heading box would otherwise trap it on.  Returns a copy of
    ``target_param`` with only the psi entry adjusted.
    """
    unwrapped = list(target_param)
    unwrapped[_PSI] = unwrap_angle(target_param[_PSI], initial_param[_PSI])
    return unwrapped


_TARGET_STATE_STEP_S = 3.0
_MAX_STATE_SUBSTEPS = 16

# Terminal realised-bank cap (degrees), always applied. Set generously high
# (e.g. 89) at construction to effectively disable it.
_DEFAULT_MAX_TERMINAL_BANK_DEG = 5.0


def select_state_substeps(duration_s: float, segment_num: int) -> int:
    """Pick M so the state step ``duration/(N*M)`` is about ``_TARGET_STATE_STEP_S``.

    ``duration_s`` is the (per-phase) duration estimate — the horizon upper
    bound for a single free phase, or the leg's duration guess for a
    procedure phase; if the solved time comes out shorter the state step is
    only finer, never coarser.
    """
    control_step = duration_s / segment_num
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
    ``start_state`` is the node ONE STEP BEFORE ``state_nodes[0]`` (the
    phase's start state); it is the ``prev`` sample when the phase has a
    single node, so it must be a real state, not a placeholder.

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
