"""Direct-collocation trajectory optimizer in geodetic coordinates.

Why this file exists
--------------------
The existing ``CasadiOptimizer`` uses multiple shooting on top of the
re-anchored stepper ``make_geo_step_from_enu_integrator``.  That stepper
swaps the ENU anchor at every RK4 step so the integration always sees a
fresh local-tangent plane — accurate, but **discontinuous** as a function
of time.

Direct collocation enforces the dynamics through algebraic defects on a
discrete mesh.  Each defect compares two evaluations of one continuous
RHS ``xdot = f(x, u)``.  A frame swap mid-segment would break that, so
the dynamics have to be one continuous ODE over the whole horizon.

``make_geodetic_dynamics_model()`` in ``casadi_simulator.py`` exposes
exactly that: a continuous RHS written directly in geodetic coordinates
``(lat, lon, h)``, with the coordinate transformation folded in as the
WGS84 curvature factors and transport-rate terms — no tangent plane, no
re-anchoring.  This file is the optimiser that *uses* it.

Method
------
Compressed Hermite-Simpson collocation on N segments with
piecewise-constant controls:

    f_k     = f(x_k, u_k)
    f_{k+1} = f(x_{k+1}, u_k)
    x_mid   = (x_k + x_{k+1})/2 + (h/8)·(f_k - f_{k+1})
    f_mid   = f(x_mid, u_k)
    defect  = x_{k+1} - x_k - (h/6)·(f_k + 4·f_mid + f_{k+1})  == 0

``x_mid`` is **not** a decision variable — it is read off the Hermite
interpolating polynomial between (x_k, f_k) and (x_{k+1}, f_{k+1}).
That keeps the NLP roughly the same size as the multiple-shooting form
while still capturing 4th-order dynamics fidelity per segment (Simpson
quadrature — same order as RK4, differing only in construction).

Frame
-----
This optimiser no longer uses any tangent plane.  It collocates
**directly in geodetic coordinates** ``(lat, lon, h)`` using the
continuous RHS from ``make_geodetic_dynamics_model()``.  The coordinate
transformation that the old fixed-ENU transcription had to perform at
the solve boundary is now folded into the continuous RHS itself as the
WGS84 radius-of-curvature factors ``1/(R_M+h)`` and
``1/((R_N+h)cos lat)``, plus the transport-rate terms on ``psi``/``gamma``.
Because that RHS is one smooth function over the whole horizon, there is
no fixed-frame curvature error to budget, and a plain RK4 of the same
RHS (the playback path) shares the optimiser's continuous vector field.

The only boundary bookkeeping left is a units conversion: ``GeodeticState``
carries lat/lon in **degrees**, while the RHS and the NLP decision
variables use **radians**.
"""

import math
import time

import casadi as ca
import numpy as np

from aerodynamic_model.casadi_simulator import (
    make_geodetic_dynamics_model,
    make_dynamics_model,
    make_geo_step_from_enu_integrator,
    geodetic_state_to_enu_expr,
    rk4_step_expr,
)
from aircraft.aero_params import AeroParams, aero_params_for_aircraft
from aircraft.aircraft_sets import AircraftSpec
from aerodynamic_model.common import GeodeticState


# 6 geodetic optimisation states (lat, lon, h, V, psi, gamma) with
# lat/lon in RADIANS.  The shape matches the existing CasadiOptimizer
# decision shape so the backend response format does not have to
# special-case this optimiser.  Mass is held constant and lives only on
# the parameter side, the same way casadi_optimizer.py treats it.
STATE_DIM = 6
CONTROL_DIM = 3

# Indices into the 6-state decision vector.
_LAT, _LON, _ALT, _V, _PSI, _GAMMA = range(6)

# Earth radius for the metric position normalization (see _normalization_cb).
_EARTH_RADIUS_M = 6_378_137.0
# Loose metric box for the normalized schemes' position state (metres from the
# anchor), replacing the radian lat/lon bounds, which are meaningless once
# position is expressed in metres.
_NORMALIZED_POS_BOUND_M = 1.0e7


def radians_expr(degrees):
    return degrees * ca.pi / 180.0


def degrees_expr(radians):
    return radians * 180.0 / ca.pi


# --------------------------------------------------------------------------
# Hermite-Simpson defect expression
# --------------------------------------------------------------------------

def hermite_simpson_defect_expr(rhs_func, x_k, x_kp1, u_k, aero_params, h):
    """Compressed Hermite-Simpson defect for ONE segment.

    With piecewise-constant control ``u_k`` over [t_k, t_{k+1}] of length
    ``h``, the segment is feasible iff this expression equals zero in
    every state component.  Only ``rhs_func`` (the continuous geodetic
    RHS from ``make_geodetic_dynamics_model``) appears here — no
    integrator is invoked at solve time.
    """
    f_k = rhs_func(x_k, u_k, aero_params)
    f_kp1 = rhs_func(x_kp1, u_k, aero_params)

    # Hermite interpolant midpoint: a cubic polynomial whose value and
    # derivative match (x_k, f_k) and (x_{k+1}, f_{k+1}) at the segment
    # endpoints, evaluated at the centre.
    x_mid = 0.5 * (x_k + x_kp1) + (h / 8.0) * (f_k - f_kp1)
    f_mid = rhs_func(x_mid, u_k, aero_params)

    # Simpson quadrature of the interpolant, rearranged so the defect is
    # zero when the dynamics are satisfied to third order in h.
    return x_kp1 - x_k - (h / 6.0) * (f_k + 4.0 * f_mid + f_kp1)


def trapezoidal_defect_expr(rhs_func, x_k, x_kp1, u_k, aero_params, h):
    """Trapezoidal collocation defect (state piecewise-LINEAR, 2nd order).

    The crudest of the three: the state between knots is a straight line, and
    the integral of ``f`` is approximated by the trapezoid rule using only the
    two endpoint derivatives — no interior evaluation.  Cheapest per defect,
    but needs a finer mesh for the same accuracy.
    """
    f_k = rhs_func(x_k, u_k, aero_params)
    f_kp1 = rhs_func(x_kp1, u_k, aero_params)
    return x_kp1 - x_k - (h / 2.0) * (f_k + f_kp1)


def rk4_defect_expr(rhs_func, x_k, x_kp1, u_k, aero_params, h):
    """RK4 'shooting' defect (4th order): the next knot must equal the RK4
    integral of the segment.

    This is the *integrator-as-defect* form rather than a polynomial fit:
    its discrete operator F **is** the same RK4 *scheme* the playback runs,
    so a converged solution has no scheme mismatch with playback.  A small
    step-size gap remains, though: the defect takes ONE RK4 step per
    sub-interval, while playback integrates the same scheme with a much
    finer step — so it is playback-*consistent*, not byte-for-byte exact.
    The price is a denser constraint Jacobian (the RK4 chain rule) and a
    slower solve.
    """
    return x_kp1 - rk4_step_expr(rhs_func, x_k, u_k, aero_params, h)


def enu_step_defect_expr(step_func, x_k, x_kp1, u_k, aero_params, h):
    """Shooting defect built on an ENU one-step integrator ``Φ(x,u,h)``.

    Shared by both ENU schemes — ``step_func`` is either the *re-anchored*
    stepper (``reanchoredEnu``: ref = current point, the exact operator the
    playback runs) or the *fixed-frame* stepper anchored at the target
    (``localEnu``: ref = target, baked in by the builder).

    The "dynamics" here is a discrete step map, not a continuous RHS, so this
    is a *multiple-shooting* defect rather than a polynomial collocation (the
    trapezoidal/Hermite fits cannot be built from a stepper).  It still lives
    in the same dense-state direct transcription (state nodes are decision
    variables, one defect per sub-interval), at the cost of a denser NLP.

    The optimiser carries lat/lon in radians; the ENU stepper expects degrees,
    so convert on the way in and back out.
    """
    x_k_deg = ca.vertcat(degrees_expr(x_k[0]), degrees_expr(x_k[1]), x_k[2:])
    x_next_deg = step_func(x_k_deg, u_k, aero_params, h)
    x_next = ca.vertcat(radians_expr(x_next_deg[0]), radians_expr(x_next_deg[1]), x_next_deg[2:])
    return x_kp1 - x_next


# A collocation scheme = a DYNAMICS × a FITTING.  The fitting (trapezoidal /
# Hermite-Simpson / rk4) is the transcription; the dynamics decides what RHS the
# fitting is applied to and in which coordinates:
#
#   geodetic       – the continuous geodetic RHS, collocated in (lat, lon, h).
#   localEnu       – the flat point-mass RHS in a FIXED ENU tangent frame
#                    anchored at the target: the geodetic nodes are converted
#                    into that frame and the fitting is applied THERE.  It is a
#                    continuous RHS too, so it takes any fitting (just like
#                    geodetic) -- but it is a LOCAL approximation that drifts far
#                    from the anchor (see dynamics_comparison_30km).
#   reanchoredEnu  – the re-anchored ENU one-step map (ref = current point).  It
#                    re-anchors every step, so it is *discrete* (no continuous
#                    RHS) -> only a shooting defect exists; no polynomial fit.
#
# Each scheme is a pair ``(make_dynamics, make_defect)``:
#   make_dynamics()                -> the RHS / stepper.  Built in the builder
#                                     BEFORE the NLP decision symbols so the
#                                     symbolic graph (and hence the solve) is
#                                     stable -- IPOPT is sensitive to the order
#                                     symbols are created in.
#   make_defect(dynamics, target)  -> the per-interval defect callable
#                                     ``defect(x_k, x_kp1, u, aero, h)``.
# (the target is the fixed ENU anchor; geodetic / re-anchored schemes ignore it)

def _geodetic_scheme(fitting_fn, transport: str = "approx"):
    """A geodetic scheme = the continuous geodetic RHS (with the chosen
    ``transport`` model) collocated by ``fitting_fn``.  ``transport="approx"``
    is the historical default (exact gamma + psi meridian-convergence, psi cross
    term dropped); ``transport="full"`` uses the EXACT transport (adds the psi
    cross term) -- see ``make_geodetic_dynamics_model``."""
    def make_dynamics():
        return make_geodetic_dynamics_model(transport=transport)["rhs_func"]

    def make_defect(rhs, target_state):
        return lambda x_k, x_kp1, u, aero, h: fitting_fn(rhs, x_k, x_kp1, u, aero, h)

    return make_dynamics, make_defect


def _geodetic_node_to_enu(x_geo, ref_geo):
    """Optimiser geodetic node (lat/lon in RADIANS, 7-vec) -> ENU 7-state in the
    fixed frame anchored at ``ref_geo`` (degrees)."""
    x_deg = ca.vertcat(degrees_expr(x_geo[_LAT]), degrees_expr(x_geo[_LON]), x_geo[2:6])
    return ca.vertcat(*geodetic_state_to_enu_expr(x_deg, ref_geo), x_geo[6])


def _local_enu_scheme(fitting_fn):
    def make_dynamics():
        return make_dynamics_model()["rhs_func"]

    def make_defect(flat_rhs, target_state):
        ref_geo = ca.vertcat(degrees_expr(target_state[_LAT]), degrees_expr(target_state[_LON]), 0.0)
        def defect(x_k, x_kp1, u, aero, h):
            return fitting_fn(flat_rhs, _geodetic_node_to_enu(x_k, ref_geo),
                              _geodetic_node_to_enu(x_kp1, ref_geo), u, aero, h)
        return defect

    return make_dynamics, make_defect


def _reanchored_enu_scheme():
    def make_dynamics():
        return make_geo_step_from_enu_integrator()["step_func"]

    def make_defect(stepper, target_state):
        return lambda x_k, x_kp1, u, aero, h: enu_step_defect_expr(stepper, x_k, x_kp1, u, aero, h)

    return make_dynamics, make_defect


# --------------------------------------------------------------------------
# Metric position normalization (the geodetic-*Normalized* schemes)
# --------------------------------------------------------------------------
#
# The plain geodetic schemes carry lat/lon as RADIANS in the decision vector,
# next to metre-scale altitude and m/s-scale speed.  The position derivative is
# ``lat_dot = V cos(gamma) cos(psi) / (R_M + h)`` -- the ``1/(R+h)`` factor makes
# the position defect rows ~7 orders of magnitude smaller than the altitude
# rows, so the constraint Jacobian is badly conditioned and IPOPT can fail to
# converge on harder problems (e.g. KRDU HEAVE -> RW05L at the default mesh with
# a loose arrival window).
#
# The *Normalized* schemes express position as METRES from the target anchor:
#
#     x = b + c (.*) z        (elementwise)
#     n = (lat - lat_t) R                     <- north metres   (c_lat = 1/R)
#     e = (lon - lon_t) R cos(lat_t)          <- east  metres   (c_lon = 1/(R cos lat_t))
#
# with h/V/psi/gamma unchanged (c = 1, b = 0).  The dynamics is the SAME exact
# geodetic RHS -- the defect just evaluates it on the reconstructed physical
# state and scales the derivative back -- so there is zero modelling change (the
# (n,e) <-> (lat,lon) map is a pure affine change of variables, unlike localEnu
# which approximates the dynamics in a flat tangent frame).  Now every decision
# component AND every defect residual is metric, which conditions the NLP well.


def _normalization_cb(target_state):
    """Per-state scale ``c`` and offset ``b`` for the metric position
    normalization, anchored at ``target_state`` (a 6/7-vector, SX or DM), such
    that the physical geodetic state is ``x = b + c (.*) z``.

    Only position (lat/lon) is rescaled; h/V/psi/gamma keep ``c=1, b=0``."""
    lat_t = target_state[_LAT]
    lon_t = target_state[_LON]
    c = ca.vertcat(
        1.0 / _EARTH_RADIUS_M,
        1.0 / (_EARTH_RADIUS_M * ca.cos(lat_t)),
        1.0, 1.0, 1.0, 1.0,
    )
    b = ca.vertcat(lat_t, lon_t, 0.0, 0.0, 0.0, 0.0)
    return c, b


def _identity_cb():
    """Identity transform (``x = z``) used by every non-normalized scheme so
    their decision/boundary path stays byte-identical to the original."""
    return ca.DM.ones(STATE_DIM), ca.DM.zeros(STATE_DIM)


def _geodetic_normalized_scheme(fitting_fn, transport: str = "approx"):
    """A geodetic scheme whose decision STATE is metric position offsets from
    the target (see above).  Same continuous geodetic RHS (with the chosen
    ``transport`` model); the defect evaluates it on the reconstructed physical
    state and scales the derivative back into metric coords, so the residual is
    metric.  Anchored at the target, like ``localEnu`` (reuses the
    ``target_state`` the builder already passes).  ``transport="full"`` uses the
    EXACT transport (adds the psi cross term); the metric-position change of
    variables is orthogonal to the transport model, so it composes cleanly."""
    def make_dynamics():
        return make_geodetic_dynamics_model(transport=transport)["rhs_func"]

    def make_defect(rhs, target_state):
        c, b = _normalization_cb(target_state)

        def normalized_rhs(z7, u, aero):
            # z7 = [z6 ; mass]; reconstruct physical state, evaluate the geodetic
            # RHS, scale the position derivatives back into metric coordinates.
            phys6 = b + c * z7[:6]
            phys7 = ca.vertcat(phys6, z7[6])
            xdot = rhs(phys7, u, aero)
            return ca.vertcat(xdot[:6] / c, xdot[6])

        return lambda x_k, x_kp1, u, aero, h: fitting_fn(
            normalized_rhs, x_k, x_kp1, u, aero, h,
        )

    return make_dynamics, make_defect


_DEFECT_SCHEMES = {
    "trapezoidal":            _geodetic_scheme(trapezoidal_defect_expr),
    "hermiteSimpson":         _geodetic_scheme(hermite_simpson_defect_expr),
    "rk4":                    _geodetic_scheme(rk4_defect_expr),
    # FULL-transport geodetic: the EXACT transport (adds the psi cross term the
    # default "approx" geodetic schemes drop).  Same fittings; an explicit
    # opt-in so the default geodetic results are unchanged.
    "trapezoidalFullTransport":    _geodetic_scheme(trapezoidal_defect_expr, transport="full"),
    "hermiteSimpsonFullTransport": _geodetic_scheme(hermite_simpson_defect_expr, transport="full"),
    "rk4FullTransport":            _geodetic_scheme(rk4_defect_expr, transport="full"),
    "trapezoidalNormalized":  _geodetic_normalized_scheme(trapezoidal_defect_expr),
    "hermiteSimpsonNormalized": _geodetic_normalized_scheme(hermite_simpson_defect_expr),
    "rk4Normalized":          _geodetic_normalized_scheme(rk4_defect_expr),
    # Normalized + FULL transport: the well-conditioned metric-position decision
    # state AND the EXACT transport.  The two are orthogonal (change of variables
    # vs RHS model), so they compose; a new opt-in, the approx-transport
    # normalized schemes above are unchanged.
    "trapezoidalNormalizedFullTransport":    _geodetic_normalized_scheme(trapezoidal_defect_expr, transport="full"),
    "hermiteSimpsonNormalizedFullTransport": _geodetic_normalized_scheme(hermite_simpson_defect_expr, transport="full"),
    "rk4NormalizedFullTransport":            _geodetic_normalized_scheme(rk4_defect_expr, transport="full"),
    "localEnuTrapezoidal":    _local_enu_scheme(trapezoidal_defect_expr),
    "localEnuHermiteSimpson": _local_enu_scheme(hermite_simpson_defect_expr),
    "localEnu":               _local_enu_scheme(rk4_defect_expr),
    "reanchoredEnu":          _reanchored_enu_scheme(),
}
# Schemes whose decision STATE is the metric position normalization -- the
# boundary/bounds/guess/output code below applies the (c, b) transform for these
# and the identity for all others.
_NORMALIZED_SCHEMES = frozenset({
    "trapezoidalNormalized", "hermiteSimpsonNormalized", "rk4Normalized",
    "trapezoidalNormalizedFullTransport", "hermiteSimpsonNormalizedFullTransport",
    "rk4NormalizedFullTransport",
})
_DEFAULT_SCHEME = "hermiteSimpson"


# Selectable NLP solver backend.  ``ipopt`` (interior point) is the robust
# default; ``sqpmethod`` (SQP + active-set QP) can be faster on benign /
# warm-started solves but is less robust on hard cold starts.
_SOLVER_BACKENDS = ("ipopt", "sqpmethod")
_DEFAULT_SOLVER_BACKEND = "ipopt"


def _make_nlp_solver(nlp, solver_backend):
    """Build the NLP solver for the chosen backend (see ``_SOLVER_BACKENDS``).

    The exact Hessian of the point-mass dynamics is nonconvex, so the SQP
    backend regularises it (``convexify_strategy``) to keep its QP subproblem
    solvable, and tells the QP not to throw on a failed subproblem.
    """
    if solver_backend == "sqpmethod":
        return ca.nlpsol('solver', 'sqpmethod', nlp, {
            'qpsol': 'qpoases',
            'qpsol_options': {'printLevel': 'none', 'error_on_fail': False},
            'convexify_strategy': 'regularize',
            'max_iter': 100,
            'print_iteration': False,
            'print_header': False,
            'print_time': False,
        })
    return ca.nlpsol('solver', 'ipopt', nlp)


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


def _decision_to_geodetic_state(row: np.ndarray, mass: float) -> GeodeticState:
    """Inverse of :func:`_geodetic_state_to_decision` (radians → degrees)."""
    return GeodeticState(
        latitude=math.degrees(float(row[_LAT])),
        longitude=math.degrees(float(row[_LON])),
        altitude=float(row[_ALT]),
        V=float(row[_V]),
        psi=float(row[_PSI]),
        gamma=float(row[_GAMMA]),
        m=mass,
    )


# --------------------------------------------------------------------------
# Dense-state transcription core
# --------------------------------------------------------------------------
#
# Control is piecewise-constant over ``segment_num`` (= N) operational
# segments, but the state dynamics are collocated on a finer grid of
# ``N * sub_steps`` Hermite-Simpson sub-intervals -- the same control u_k is
# shared across the ``sub_steps`` (= M) sub-intervals of segment k.  This
# makes the optimiser's discrete operator converge to the playback
# integrator (fine RK4 of a piecewise-constant control), so the raw solution
# is playback-consistent WITHOUT any post-solve polish, while keeping the
# control DOF coarse (3N) to avoid the convergence / "wrinkle" pathology of
# refining the control mesh.

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


def _build_collocation_decision(
    defect, aero_params, start_state, target_state,
    segment_h, segment_num, sub_steps, normalize_cb,
):
    """Build controls, state nodes and defects for the dense-state NLP.

    ``defect`` is the scheme's resolved per-interval callable
    ``defect(x_k, x_kp1, u, aero, h)`` (already bound to its dynamics and, for
    localEnu / the normalized schemes, the target anchor; see ``_DEFECT_SCHEMES``).

    ``normalize_cb = (c, b)`` is the decision-state transform ``x = b + c (.*) z``
    (identity for non-normalized schemes).  The state *nodes* are the optimisation
    variables ``z``; the physical start/target are mapped into ``z`` here so the
    first interval and the terminal-equality defect are written in the same
    coordinates as the nodes.

    Returns ``(seg_controls, state_nodes, defects)``: ``seg_controls`` has
    ``segment_num`` 3-vectors, ``state_nodes`` has ``segment_num*sub_steps``
    6-vectors (z_1 .. z_{N*M}).  Each sub-interval gets its own defect using
    the shared control of its segment; the terminal-equality defect (last node
    == target) is last.
    """
    c, b = normalize_cb
    mass_param = start_state[6]
    state_h = segment_h / sub_steps
    seg_controls = []
    state_nodes = []
    defects = []
    x_prev = (start_state[:6] - b) / c          # start in z-coordinates
    for k in range(segment_num):
        uk = ca.SX.sym(f'u_{k}', CONTROL_DIM)
        seg_controls.append(uk)
        for j in range(sub_steps):
            xnode = ca.SX.sym(f'x_{k}_{j}', STATE_DIM)
            state_nodes.append(xnode)
            x_prev_full = ca.vertcat(x_prev, mass_param)
            x_next_full = ca.vertcat(xnode, mass_param)
            # Mass derivative is identically zero, so drop the 7th defect
            # component to keep the constraint Jacobian rectangular.
            defects.append(
                defect(
                    x_prev_full, x_next_full, uk, aero_params, state_h,
                )[:STATE_DIM]
            )
            x_prev = xnode
    defects.append(state_nodes[-1] - (target_state[:STATE_DIM] - b) / c)
    return seg_controls, state_nodes, defects


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


# --------------------------------------------------------------------------
# NLP builder
# --------------------------------------------------------------------------

def _scheme_normalization(collocation_scheme, target_state):
    """The decision-state transform ``(c, b)`` for a scheme: the metric position
    normalization for the ``*Normalized`` schemes, identity for all others."""
    if collocation_scheme in _NORMALIZED_SCHEMES:
        return _normalization_cb(target_state)
    return _identity_cb()


def _normalized_position_bounds(state_lb, state_ub, collocation_scheme):
    """Replace the radian lat/lon state bounds with a loose metric box for the
    normalized schemes (position is metres there); h/V/psi/gamma are unchanged."""
    if collocation_scheme not in _NORMALIZED_SCHEMES:
        return state_lb, state_ub
    lb, ub = list(state_lb), list(state_ub)
    lb[_LAT] = lb[_LON] = -_NORMALIZED_POS_BOUND_M
    ub[_LAT] = ub[_LON] = _NORMALIZED_POS_BOUND_M
    return lb, ub


def make_direct_collocation_solver(
    segment_num: int,
    aero_params_obj: AeroParams,
    aircraft_meta: dict,
    sub_steps: int = 1,
    max_terminal_bank_deg: float = _DEFAULT_MAX_TERMINAL_BANK_DEG,
    smoothness_weights: tuple = _DEFAULT_SMOOTHNESS_WEIGHTS,
    collocation_scheme: str = _DEFAULT_SCHEME,
    solver_backend: str = _DEFAULT_SOLVER_BACKEND,
):
    """Build the fixed-time NLP symbolically with the chosen defect scheme.

    Initial state, target state, and segment duration are passed as
    parameters so the same compiled solver can serve every optimisation
    request without rebuilding.  This matches the strategy used in
    ``casadi_optimizer.make_multiple_shooting_solver``.

    A terminal inequality keeps the *realised* bank at the final node below
    ``max_terminal_bank_deg`` — a pure state constraint (see
    ``terminal_bank_constraint_expr``).
    """
    # 1. Build the scheme's dynamics BEFORE the NLP symbols (solve stability).
    make_dynamics, make_defect = _DEFECT_SCHEMES[collocation_scheme]
    dynamics = make_dynamics()

    # 2. Symbolic parameters (lat/lon in radians).
    start_state = ca.SX.sym('start_state', 7)   # lat, lon, h, V, psi, gamma, m
    target_state = ca.SX.sym('target_state', 7)
    duration = ca.SX.sym('duration')

    # 3. Bind the per-interval defect (for localEnu / *Normalized*, anchored at
    # the target) and resolve the decision-state transform for the boundary.
    defect = make_defect(dynamics, target_state)
    normalize_cb = _scheme_normalization(collocation_scheme, target_state)

    aero_params = ca.vertcat(
        aero_params_obj.S,
        aero_params_obj.Cl_max,
        aero_params_obj.Cd0,
        aero_params_obj.k,
        aero_params_obj.stall_threshold,
        aero_params_obj.k_stall,
    )

    # 3. Control step h (per operational segment); the state step is
    # ``segment_h / sub_steps`` inside the collocation core.
    segment_h = duration / segment_num

    # 4. Decision variables + defects: N controls, N*sub_steps state nodes.
    seg_controls, state_nodes, defects = _build_collocation_decision(
        defect, aero_params, start_state, target_state,
        segment_h, segment_num, sub_steps, normalize_cb,
    )

    # 5. Bounds.  Identical envelope to ``casadi_optimizer.py`` so the
    # two CasADi optimisers stay comparable.
    state_lb, state_ub = make_state_bounds(
        min_altitude=aircraft_meta['min_altitude'],
        min_velocity=aircraft_meta['min_terminal_speed'],
    )
    state_lb, state_ub = _normalized_position_bounds(state_lb, state_ub, collocation_scheme)
    control_lb, control_ub = make_control_bounds(
        aircraft_meta['max_thrust'],
        aircraft_meta['min_load_factor'],
        aircraft_meta['max_load_factor'],
    )

    w = seg_controls + state_nodes
    lbw = control_lb * segment_num + state_lb * len(state_nodes)
    ubw = control_ub * segment_num + state_ub * len(state_nodes)

    # Equality defects (lbg = ubg = 0) plus the terminal bank inequality
    # (finite two-sided bounds), appended last.
    bank_expr, bank_lb, bank_ub = terminal_bank_constraint_expr(
        state_nodes, start_state, segment_h / sub_steps,
        math.radians(max_terminal_bank_deg),
    )
    g = ca.vertcat(*defects, bank_expr)
    n_defect_rows = STATE_DIM * len(defects)
    lbg = [0.0] * n_defect_rows + [bank_lb]
    ubg = [0.0] * n_defect_rows + [bank_ub]

    # 6. Objective: control effort + segment-to-segment smoothness (mostly
    # on bank μ and load n).  Keeps the NLP well-posed when arrival time is
    # fixed and discourages control jumps.
    cost = (
        _scaled_control_cost(seg_controls, aircraft_meta)
        + _control_smoothness_cost(seg_controls, aircraft_meta, smoothness_weights)
    )

    nlp = {
        'f': cost,
        'x': ca.vertcat(*w),
        'g': g,
        'p': ca.vertcat(start_state, target_state, duration),
    }
    solver = _make_nlp_solver(nlp, solver_backend)
    return solver, lbw, ubw, lbg, ubg


# --------------------------------------------------------------------------
# Free-final-time NLP builder
# --------------------------------------------------------------------------
#
# Why a second NLP?
#   The fixed-time solver above treats ``duration`` as a parameter, which
#   forces an outer bisection loop in ``optimize_time_to_target`` to find
#   the shortest feasible horizon.  Direct collocation can avoid that
#   loop by promoting ``T`` to a decision variable: the Hermite-Simpson
#   defect already depends on ``h = T/N`` symbolically, so IPOPT
#   differentiates through it and optimises trajectory + arrival time
#   jointly in a single solve.

_DEFAULT_TIME_REGULARIZATION = 1e-3
_DEFAULT_MIN_DURATION_S = 1.0


def make_direct_collocation_solver_free_time(
    segment_num: int,
    aero_params_obj: AeroParams,
    aircraft_meta: dict,
    time_regularization: float = _DEFAULT_TIME_REGULARIZATION,
    sub_steps: int = 1,
    max_terminal_bank_deg: float = _DEFAULT_MAX_TERMINAL_BANK_DEG,
    smoothness_weights: tuple = _DEFAULT_SMOOTHNESS_WEIGHTS,
    collocation_scheme: str = _DEFAULT_SCHEME,
    solver_backend: str = _DEFAULT_SOLVER_BACKEND,
):
    """Build the free-final-time NLP with ``T`` as a decision variable.

    Objective is ``T/T_max + λ · mean(||u_scaled||²)`` so the time term
    dominates (the regularisation is a tie-breaker that keeps the
    control profile from drifting along an arbitrary null-space when
    several controls reach the same arrival time).

    The terminal realised-bank inequality is the same as the fixed-time
    builder; here ``state_h`` depends on the decision variable ``T``, which
    IPOPT differentiates through.
    """
    # 1. Build the scheme's dynamics BEFORE the NLP symbols (solve stability).
    make_dynamics, make_defect = _DEFECT_SCHEMES[collocation_scheme]
    dynamics = make_dynamics()

    # 2. Symbolic parameters.  ``max_duration`` is the upper bound for
    # the decision-variable duration and is also the scale factor used
    # to non-dimensionalise the time term in the objective.
    start_state = ca.SX.sym('start_state', 7)
    target_state = ca.SX.sym('target_state', 7)
    max_duration = ca.SX.sym('max_duration')

    # 3. Bind the per-interval defect (same as the fixed-time NLP) and resolve
    # the decision-state transform for the boundary.
    defect = make_defect(dynamics, target_state)
    normalize_cb = _scheme_normalization(collocation_scheme, target_state)

    aero_params = ca.vertcat(
        aero_params_obj.S,
        aero_params_obj.Cl_max,
        aero_params_obj.Cd0,
        aero_params_obj.k,
        aero_params_obj.stall_threshold,
        aero_params_obj.k_stall,
    )

    # 3. ``T`` is a decision variable now, not a parameter.  Bounds are
    # supplied through ``lbw/ubw`` below.
    duration_var = ca.SX.sym('T')
    segment_h = duration_var / segment_num

    # 4. Decision variables + defects: N controls, N*sub_steps state nodes,
    # then T appended last so we can slice it out of ``sol['x']`` cleanly.
    seg_controls, state_nodes, defects = _build_collocation_decision(
        defect, aero_params, start_state, target_state,
        segment_h, segment_num, sub_steps, normalize_cb,
    )

    # 5. Bounds.
    state_lb, state_ub = make_state_bounds(
        min_altitude=aircraft_meta['min_altitude'],
        min_velocity=aircraft_meta['min_terminal_speed'],
    )
    state_lb, state_ub = _normalized_position_bounds(state_lb, state_ub, collocation_scheme)
    control_lb, control_ub = make_control_bounds(
        aircraft_meta['max_thrust'],
        aircraft_meta['min_load_factor'],
        aircraft_meta['max_load_factor'],
    )
    # Duration floor keeps ``segment_h > 0``; the upper bound is a finite
    # placeholder that ``optimize_free_time`` re-binds to the requested
    # max_duration at solve time via ``lbx``/``ubx``.
    duration_lb = aircraft_meta.get('min_duration', _DEFAULT_MIN_DURATION_S)
    duration_ub = 1e6

    w = seg_controls + state_nodes + [duration_var]
    lbw = control_lb * segment_num + state_lb * len(state_nodes) + [duration_lb]
    ubw = control_ub * segment_num + state_ub * len(state_nodes) + [duration_ub]

    # Equality defects (lbg = ubg = 0) plus the terminal bank inequality.
    # state_h = segment_h / sub_steps depends on the decision variable T here.
    bank_expr, bank_lb, bank_ub = terminal_bank_constraint_expr(
        state_nodes, start_state, segment_h / sub_steps,
        math.radians(max_terminal_bank_deg),
    )
    g = ca.vertcat(*defects, bank_expr)
    n_defect_rows = STATE_DIM * len(defects)
    lbg = [0.0] * n_defect_rows + [bank_lb]
    ubg = [0.0] * n_defect_rows + [bank_ub]

    # 6. Objective.  Time dominates; control effort is a tiny tie-breaker.
    # Smoothness (mostly bank μ and load n) is added directly (NOT scaled by
    # time_regularization) so it actually shapes the control profile on the
    # default free-time path -- trading a little arrival time for smoother
    # attitude commands.  All terms are non-dimensional.
    cost = (
        duration_var / max_duration
        + time_regularization * _scaled_control_cost(seg_controls, aircraft_meta)
        + _control_smoothness_cost(seg_controls, aircraft_meta, smoothness_weights)
    )

    nlp = {
        'f': cost,
        'x': ca.vertcat(*w),
        'g': g,
        'p': ca.vertcat(start_state, target_state, max_duration),
    }
    solver = _make_nlp_solver(nlp, solver_backend)
    return solver, lbw, ubw, lbg, ubg


# --------------------------------------------------------------------------
# Optimiser class
# --------------------------------------------------------------------------

class CasadiDirectCollocationOptimizer:
    """Direct-collocation trajectory optimiser in a fixed ENU frame.

    API-compatible with ``CasadiOptimizer`` so the backend dispatcher
    can substitute either optimiser behind the same call shape.
    """

    def __init__(
        self,
        n_segments: int,
        dt: float,
        max_duration: float,
        aircraft: AircraftSpec,
        max_terminal_bank_deg: float = _DEFAULT_MAX_TERMINAL_BANK_DEG,
        smoothness_weights: tuple = _DEFAULT_SMOOTHNESS_WEIGHTS,
        collocation_scheme: str = _DEFAULT_SCHEME,
        solver_backend: str = _DEFAULT_SOLVER_BACKEND,
    ):
        if n_segments < 2:
            raise ValueError("n_segments must be at least 2 for direct collocation")
        if collocation_scheme not in _DEFECT_SCHEMES:
            raise ValueError(
                f"unknown collocation_scheme {collocation_scheme!r}; "
                f"choose from {sorted(_DEFECT_SCHEMES)}"
            )
        if solver_backend not in _SOLVER_BACKENDS:
            raise ValueError(
                f"unknown solver_backend {solver_backend!r}; "
                f"choose from {list(_SOLVER_BACKENDS)}"
            )
        self.n_segments = n_segments
        # Defect scheme = dynamics x fitting (e.g. hermiteSimpson, localEnu,
        # reanchoredEnu, or a *Normalized* metric-position variant).  Both the
        # fixed-time and free-time solvers use it.
        self.collocation_scheme = collocation_scheme
        # Wall-clock breakdown of the most recent ``optimize_free_time`` call
        # (cold-start solve vs free-time solve); read by the backend for the
        # whole-flow timing log.  ``None`` until the first solve.
        self.last_solve_timings: dict[str, float] | None = None
        # NLP solver backend: ipopt (default) or sqpmethod.
        self.solver_backend = solver_backend
        # Keep the realised bank at the terminal node below this angle (a
        # pure state constraint).
        self.max_terminal_bank_deg = max_terminal_bank_deg
        # Per-control smoothness weights (thrust, bank μ, load n) on the
        # segment-to-segment control change.
        self.smoothness_weights = smoothness_weights
        # ``dt`` is accepted for API parity with the multiple-shooting
        # optimiser.  Control stays piecewise-constant over ``n_segments``;
        # dynamics fidelity comes from collocating the state on a finer
        # grid of ``n_segments * state_substeps`` Hermite-Simpson
        # sub-intervals.  ``state_substeps`` (= M) is auto-selected from the
        # horizon so the state step is a few seconds -- small enough that
        # the raw solution is playback-consistent (no polish needed).
        self.dt = dt
        self.max_duration = max_duration
        self.aircraft = aircraft
        self.state_substeps = select_state_substeps(max_duration, n_segments)

        # Mass-based stall model, SHARED with the playback (CasadiSimulator) via
        # ``aero_params_for_aircraft`` so an optimized trajectory replays
        # consistently -- the optimiser and the playback must agree on Cl_max.
        self.aero_params = aero_params_for_aircraft(aircraft)
        
        aircraft_meta = {
            "max_thrust": aircraft.max_thrust_n,
            "min_load_factor": 0.5,
            "max_load_factor": 2.0,
            "min_terminal_speed": aircraft.terminal_speed_kt * 0.51444,
            "min_altitude": aircraft.threshold_crossing_height_m + 10.0,
            "min_duration": _DEFAULT_MIN_DURATION_S,
        }

        # Fixed-time NLP -- duration is a parameter.  Used by
        # ``optimize_trajectory`` when the caller supplies an explicit
        # duration (e.g. a Controlled Time of Arrival), and as the cold-start
        # seed for the free-time solve.  Same scheme as the free-time solve, so
        # the seed's decision vector drops straight in.
        self.solver, self.lbw, self.ubw, self.lbg, self.ubg = (
            make_direct_collocation_solver(
                segment_num=n_segments,
                aero_params_obj=self.aero_params,
                aircraft_meta=aircraft_meta,
                sub_steps=self.state_substeps,
                max_terminal_bank_deg=max_terminal_bank_deg,
                smoothness_weights=smoothness_weights,
                collocation_scheme=collocation_scheme,
                solver_backend=solver_backend,
            )
        )

        # Free-final-time NLP -- duration is a decision variable.  Used
        # by ``optimize_time_to_target`` so a single solve handles both
        # the trajectory shape and the optimal arrival time without any
        # outer bisection loop.
        (
            self.free_time_solver,
            self.free_time_lbw,
            self.free_time_ubw,
            self.free_time_lbg,
            self.free_time_ubg,
        ) = make_direct_collocation_solver_free_time(
            segment_num=n_segments,
            aero_params_obj=self.aero_params,
            aircraft_meta=aircraft_meta,
            sub_steps=self.state_substeps,
            max_terminal_bank_deg=max_terminal_bank_deg,
            smoothness_weights=smoothness_weights,
            collocation_scheme=collocation_scheme,
            solver_backend=solver_backend,
        )

    # ------------------------------------------------------------------
    # Public API matching CasadiOptimizer
    # ------------------------------------------------------------------

    def optimize_trajectory(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
        duration: float | None = None,
        initial_guess=None,
    ):
        """Solve the fixed-time NLP.  Returns the same tuple shape as
        ``CasadiOptimizer.optimize_trajectory`` so the backend handler
        treats both optimisers identically."""
        solve_duration = self.max_duration if duration is None else duration

        # 1. Convert the geodetic endpoints to the radian decision units.
        # No tangent plane, no anchor -- the RHS is global.  Unwrap the target
        # heading so the optimiser turns the short way to it.
        initial_param = _geodetic_state_to_decision(initial_state)
        target_param = _unwrap_target_heading(
            initial_param, _geodetic_state_to_decision(target_state),
        )

        # 2. Build (or accept) the initial guess.  The default linear
        # interpolation is the same shape as casadi_optimizer; replace
        # it via the ``initial_guess`` argument for warm starts.
        x0 = (
            self._build_initial_guess(initial_param, target_param)
            if initial_guess is None
            else initial_guess
        )

        # 3. Solve and unpack.
        w_opt = self._solve_fixed_time_raw(
            initial_param, target_param, solve_duration, x0,
        )
        control_opt = w_opt[: self.n_segments * CONTROL_DIM].reshape(
            (self.n_segments, CONTROL_DIM),
        )
        # 4. Return one geodetic state per control segment (segment
        # endpoints), so the response stays aligned with the N controls.
        state_geo = self._extract_node_states_geo(
            w_opt[self.n_segments * CONTROL_DIM:], initial_state.m, target_param,
        )
        return solve_duration, control_opt, state_geo

    def optimize_free_time(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
        max_duration: float,
        initial_guess=None,
    ):
        """Solve the free-final-time NLP.

        Decision variables include trajectory states/controls AND the
        arrival time ``T``.  IPOPT minimises ``T/T_max`` (with a small
        control-effort regulariser) subject to the Hermite-Simpson
        defects.  Returns the same ``(final_time, controls, states)``
        tuple shape as ``optimize_trajectory`` so the backend dispatcher
        does not have to special-case it.

        Warm-start strategy: cold-starting the joint (state, control, T)
        search from a linearly-interpolated trajectory leaves IPOPT far
        from the dynamics manifold and convergence is unreliable on
        realistic-scale problems.  We instead solve the fixed-time NLP
        at ``T = max_duration`` first -- which already returns a
        dynamically feasible trajectory -- and use that as the seed for
        the free-time solve.  The free-time solver then only has to
        shrink T along the feasible manifold, which is a much
        better-conditioned search.

        Playback consistency: the dense-state transcription collocates the
        state on ``n_segments * state_substeps`` sub-intervals, so the raw
        solution already lands on the target when the controls are replayed
        through the playback integrator -- no post-solve polish is needed.
        """
        initial_param = _geodetic_state_to_decision(initial_state)
        target_param = _unwrap_target_heading(
            initial_param, _geodetic_state_to_decision(target_state),
        )

        # Cold start: build the warm-start seed (solves the fixed-time NLP).
        # Timed separately from the free-time solve so the backend can log the
        # whole-flow breakdown.
        cold_start_started = time.perf_counter()
        if initial_guess is None:
            x0 = self._build_free_time_initial_guess(
                initial_state, target_state, max_duration,
            )
        else:
            x0 = list(initial_guess)
        cold_start_s = time.perf_counter() - cold_start_started

        # Rebind the upper bound on T at solve time.  The NLP was built
        # with a placeholder upper bound (1e6); we tighten it here to
        # the caller's max_duration so IPOPT does not search outside
        # the operational window.
        lbw = list(self.free_time_lbw)
        ubw = list(self.free_time_ubw)
        ubw[-1] = max_duration
        # If the caller asked for an unrealistically short max_duration,
        # also lift the floor so the bounds are consistent.
        lbw[-1] = min(lbw[-1], max_duration * 0.5)

        p = ca.vertcat(initial_param, target_param, max_duration)
        free_time_started = time.perf_counter()
        sol = self.free_time_solver(
            x0=x0,
            lbx=lbw,
            ubx=ubw,
            lbg=self.free_time_lbg,
            ubg=self.free_time_ubg,
            p=p,
        )
        free_time_s = time.perf_counter() - free_time_started
        self.last_solve_timings = {
            "coldStartS": cold_start_s,
            "freeTimeSolveS": free_time_s,
            "solveTotalS": cold_start_s + free_time_s,
        }
        stats = self.free_time_solver.stats()
        if not stats.get("success", False):
            raise ValueError(
                "Direct collocation free-time optimization failed: "
                + stats.get("return_status", "unknown")
            )

        w_opt = sol['x'].full().flatten()
        control_opt = w_opt[: self.n_segments * CONTROL_DIM].reshape(
            (self.n_segments, CONTROL_DIM),
        )
        n_state_block = self.n_segments * self.state_substeps * STATE_DIM
        state_block = w_opt[
            self.n_segments * CONTROL_DIM : self.n_segments * CONTROL_DIM + n_state_block
        ]
        final_time = float(w_opt[-1])
        state_geo = self._extract_node_states_geo(state_block, initial_state.m, target_param)
        return final_time, control_opt, state_geo

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_fixed_time_raw(self, initial_param, target_param, duration, x0):
        """Run the fixed-time solver and return the raw decision vector
        ``[controls(3N), states(6*N*M)]`` (radians).  Raises on failure."""
        p = ca.vertcat(initial_param, target_param, duration)
        sol = self.solver(
            x0=x0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p,
        )
        stats = self.solver.stats()
        if not stats.get("success", False):
            raise ValueError(
                "Direct collocation optimization failed: "
                + stats.get("return_status", "unknown")
            )
        return sol['x'].full().flatten()

    def _normalize_cb_numeric(self, target_param):
        """Numeric decision-state transform ``(c, b)`` (``x = b + c (.*) z``)
        anchored at the numeric target -- metric position for the ``*Normalized``
        schemes, identity (c=1, b=0) otherwise."""
        c, b = _scheme_normalization(self.collocation_scheme, ca.DM(target_param))
        return np.array(c).flatten(), np.array(b).flatten()

    def _extract_node_states_geo(self, state_block, mass: float, target_param) -> np.ndarray:
        """Pick the segment-endpoint state of each of the N control
        segments from the dense ``N*M`` state block and convert to
        geodetic degrees -- returns an ``(N, 6)`` array.  The block is in the
        scheme's decision coordinates ``z``; map back to physical via
        ``x = b + c (.*) z`` before converting to geodetic."""
        c, b = self._normalize_cb_numeric(target_param)
        nodes = np.asarray(state_block).reshape(
            (self.n_segments * self.state_substeps, STATE_DIM),
        )
        endpoints = nodes[self.state_substeps - 1 :: self.state_substeps]
        phys_endpoints = endpoints * c + b
        return np.array([
            self._decision_row_to_geo_array(row, mass) for row in phys_endpoints
        ])

    def _build_initial_guess(self, initial_param, target_param):
        """Straight-line interpolation across the N*M state nodes (in the
        scheme's decision coordinates -- metric position for the ``*Normalized``
        schemes, radians otherwise), plus a neutral approach control profile."""
        c, b = self._normalize_cb_numeric(target_param)
        control_guess = [
            self.aircraft.approach_thrust_guess_n,
            0.0,
            1.0,
        ] * self.n_segments
        n_nodes = self.n_segments * self.state_substeps
        state_guess: list[float] = []
        for i in range(n_nodes):
            ratio = (i + 1) / n_nodes
            for d in range(STATE_DIM):
                phys = initial_param[d] + (target_param[d] - initial_param[d]) * ratio
                state_guess.append((phys - b[d]) / c[d])
        return control_guess + state_guess

    def _build_free_time_initial_guess(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
        max_duration: float,
    ) -> list[float]:
        """Solve the fixed-time NLP at the upper bound to get a
        dynamically feasible trajectory, then append ``max_duration`` as
        the initial value for ``T``.

        The fixed-time decision vector ``[controls(3N), states(6*N*M)]`` is
        exactly the free-time decision minus the trailing ``T``, so the raw
        solution is reused verbatim.  If the fixed-time warm-start itself
        fails (rare), fall back to the cold linear-interpolation guess.
        """
        initial_param = _geodetic_state_to_decision(initial_state)
        target_param = _unwrap_target_heading(
            initial_param, _geodetic_state_to_decision(target_state),
        )
        cold = self._build_initial_guess(initial_param, target_param)
        try:
            w_opt = self._solve_fixed_time_raw(
                initial_param, target_param, max_duration, cold,
            )
        except ValueError:
            return cold + [max_duration]
        return list(w_opt) + [max_duration]

    @staticmethod
    def _decision_row_to_geo_array(row: np.ndarray, mass: float) -> np.ndarray:
        geo = _decision_to_geodetic_state(row, mass)
        return np.array([
            geo.latitude,
            geo.longitude,
            geo.altitude,
            geo.V,
            geo.psi,
            geo.gamma,
        ])
