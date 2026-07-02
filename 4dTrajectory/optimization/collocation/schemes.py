"""Collocation schemes: a DYNAMICS × a FITTING, plus the metric-position normalization.

A scheme is a pair ``(make_dynamics, make_defect)`` in :data:`SCHEMES`. The FITTING
(trapezoidal / Hermite-Simpson / rk4) is the transcription; the DYNAMICS decides which RHS the
fitting is applied to and in which coordinates (geodetic / localEnu / reanchoredEnu, optionally
with the metric-position ``*Normalized`` change of variables). Pure math — no optimizer state.
"""

from functools import lru_cache

import casadi as ca

from geokit import WGS84_A as _EARTH_RADIUS_M
from aerodynamic_model.casadi_coordinates_converter import degrees_expr, radians_expr
from aerodynamic_model.casadi_simulator import (
    make_geodetic_dynamics_model,
    make_dynamics_model,
    make_geo_step_from_enu_integrator,
    geodetic_state_to_enu_expr,
    rk4_step_expr,
)

# 6 geodetic optimisation states (lat, lon, h, V, psi, gamma), lat/lon in RADIANS.
STATE_DIM = 6
CONTROL_DIM = 3
# Indices into the 6-state decision vector.
_LAT, _LON, _ALT, _V, _PSI, _GAMMA = range(6)
# Loose metric box for the normalized schemes' position state (metres from the anchor).
_NORMALIZED_POS_BOUND_M = 1.0e7


# The dynamics RHS / stepper is a pure function of (at most) the transport model — it does NOT
# depend on the aircraft (aero is a symbolic input), the initial/target, or the mesh. A compiled
# casadi Function is stateless and reusable, so build each ONCE and reuse across every _build
# (avoids reconstructing the full symbolic point-mass graph on every solve).
@lru_cache(maxsize=None)
def _geodetic_rhs(transport):
    return make_geodetic_dynamics_model(transport=transport)["rhs_func"]


@lru_cache(maxsize=None)
def _flat_enu_rhs():
    return make_dynamics_model()["rhs_func"]


@lru_cache(maxsize=None)
def _reanchored_enu_step():
    return make_geo_step_from_enu_integrator()["step_func"]


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
        return _geodetic_rhs(transport)

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
        return _flat_enu_rhs()

    def make_defect(flat_rhs, target_state):
        ref_geo = ca.vertcat(degrees_expr(target_state[_LAT]), degrees_expr(target_state[_LON]), 0.0)
        def defect(x_k, x_kp1, u, aero, h):
            return fitting_fn(flat_rhs, _geodetic_node_to_enu(x_k, ref_geo),
                              _geodetic_node_to_enu(x_kp1, ref_geo), u, aero, h)
        return defect

    return make_dynamics, make_defect


def _reanchored_enu_scheme():
    def make_dynamics():
        return _reanchored_enu_step()

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
        return _geodetic_rhs(transport)

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


# Approach-procedure path constraints (constraints package) are only wired into the
# *Normalized* full-transport schemes: the state nodes there are metric (n, e) offsets from
# the target, so the corridor/glidepath/floor inequalities are written directly on the decision
# variables and stay well-conditioned (the whole reason the Normalized scheme exists). See
# 4dTrajectory/optimization/approach_constraints/ and docs/optimization_constraint_design.md.
_NORMALIZED_FULL_TRANSPORT_SCHEMES = frozenset({
    "trapezoidalNormalizedFullTransport",
    "hermiteSimpsonNormalizedFullTransport",
    "rk4NormalizedFullTransport",
})


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

