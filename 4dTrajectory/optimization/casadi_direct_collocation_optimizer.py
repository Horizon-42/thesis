"""Direct-collocation trajectory optimizer in a fixed ENU frame.

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
the dynamics have to live in **one** ENU frame for the whole horizon.

``make_dynamics_model()`` in ``casadi_simulator.py`` already exposes
exactly that continuous RHS — what was missing is an optimiser that
*uses* it without the per-step re-anchoring wrapper.  This file is that
optimiser.

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
while still capturing 3rd-order dynamics fidelity per segment.

Frame
-----
The fixed ENU frame is anchored at the **initial geodetic position**
(reference altitude = 0).  Endpoint conversions are performed once at
the boundary of the solve and are NOT inside the NLP.  Earth-curvature
modeling error over a 5 km final-approach window is documented in
``fixed_enu_frame_error.py`` (Δh ≈ 1.96 m, gravity tilt ≈ 0.045°,
meridian convergence ≈ 0.056° at 51 °N) — small enough that direct
collocation in a single fixed frame is a reasonable transcription.
"""

import casadi as ca
import numpy as np

from aerodynamic_model.casadi_simulator import make_dynamics_model, AeroParams
from aerodynamic_model.casadi_coordinates_converter import (
    enu_to_geodetic_expr,
    geodetic_to_enu_expr,
)
from aerodynamic_model.aircraft_sets import AircraftSpec
from aerodynamic_model.common import GeodeticState


# 6 ENU optimisation states (e, n, h, V, psi, gamma) match the existing
# CasadiOptimizer decision shape so the backend response format does not
# have to special-case this optimiser.  Mass is held constant and lives
# only on the parameter side, the same way casadi_optimizer.py treats it.
STATE_DIM = 6
CONTROL_DIM = 3


def radians_expr(degrees):
    return degrees * ca.pi / 180.0


# --------------------------------------------------------------------------
# Hermite-Simpson defect expression
# --------------------------------------------------------------------------

def hermite_simpson_defect_expr(rhs_func, x_k, x_kp1, u_k, aero_params, h):
    """Compressed Hermite-Simpson defect for ONE segment.

    With piecewise-constant control ``u_k`` over [t_k, t_{k+1}] of length
    ``h``, the segment is feasible iff this expression equals zero in
    every state component.  Only ``rhs_func`` (the continuous ENU RHS
    from ``make_dynamics_model``) appears here — no integrator is
    invoked at solve time.
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
    # ENU horizontal range is loosely bounded at ±200 km — far larger
    # than any realistic terminal-area trajectory, but enough to keep
    # IPOPT inside a well-defined search box.  Lat/lon become a function
    # of (e, n) via the anchor, so we do not bound lat/lon directly.
    e_min, e_max = -200_000.0, 200_000.0
    n_min, n_max = -200_000.0, 200_000.0
    alt_min, alt_max = min_altitude, 10_000.0
    V_min, V_max = min_velocity, 1_000.0
    psi_min, psi_max = -ca.pi, ca.pi
    gamma_min, gamma_max = -radians_expr(10.0), radians_expr(30.0)
    return (
        [e_min, n_min, alt_min, V_min, psi_min, gamma_min],
        [e_max, n_max, alt_max, V_max, psi_max, gamma_max],
    )


# Cached CasADi Functions for geodetic <-> ENU at the solve boundary.
# They are pure WGS84 — we share the exact same conversion the
# continuous-time model would see, so the boundary projection cannot
# introduce its own modelling drift.
_GEO_TO_ENU_FN: ca.Function | None = None
_ENU_TO_GEO_FN: ca.Function | None = None


def _get_geo_to_enu_fn() -> ca.Function:
    global _GEO_TO_ENU_FN
    if _GEO_TO_ENU_FN is None:
        lat = ca.SX.sym('lat')
        lon = ca.SX.sym('lon')
        alt = ca.SX.sym('alt')
        ref_lat = ca.SX.sym('ref_lat')
        ref_lon = ca.SX.sym('ref_lon')
        e, n, u = geodetic_to_enu_expr(lat, lon, alt, ref_lat, ref_lon, 0.0)
        _GEO_TO_ENU_FN = ca.Function(
            'geo_to_enu_fn',
            [lat, lon, alt, ref_lat, ref_lon],
            [ca.vertcat(e, n, u)],
        )
    return _GEO_TO_ENU_FN


def _get_enu_to_geo_fn() -> ca.Function:
    global _ENU_TO_GEO_FN
    if _ENU_TO_GEO_FN is None:
        e = ca.SX.sym('e')
        n = ca.SX.sym('n')
        u = ca.SX.sym('u')
        ref_lat = ca.SX.sym('ref_lat')
        ref_lon = ca.SX.sym('ref_lon')
        lat, lon, alt = enu_to_geodetic_expr(e, n, u, ref_lat, ref_lon, 0.0)
        _ENU_TO_GEO_FN = ca.Function(
            'enu_to_geo_fn',
            [e, n, u, ref_lat, ref_lon],
            [ca.vertcat(lat, lon, alt)],
        )
    return _ENU_TO_GEO_FN


def _geodetic_state_to_enu_decision(state: GeodeticState, anchor_lat: float, anchor_lon: float) -> list[float]:
    """Convert a geodetic state to the 7-vector used as an NLP parameter.

    The decision-variable portion uses six entries (e, n, u, V, psi,
    gamma) and the seventh slot carries the constant mass for the RHS.

    The third state is the **ENU u-coordinate**, not the geodetic
    altitude — these differ by the curvature drop ``d²/(2R)`` at lateral
    distance ``d`` from the anchor.  We use the ENU u-coord because:

    * the continuous fixed-ENU dynamics interpret it as the ``z`` of
      the tangent-plane state, so ``dh/dt = V·sin γ`` is consistent;
    * the geo↔ENU round-trip is then exact, which is essential when
      ``_build_free_time_initial_guess`` re-encodes the fixed-time
      solution as a warm-start for the free-time NLP.

    The ISA density model still uses this value as if it were geodetic
    altitude; the discrepancy is bounded by ~2 m over a 5 km horizon
    (see ``fixed_enu_frame_error.py``), which is below the fidelity of
    the simplified ``isa_density_expr`` model.
    """
    converter = _get_geo_to_enu_fn()
    enu = converter(state.latitude, state.longitude, state.altitude, anchor_lat, anchor_lon)
    return [
        float(enu[0]),
        float(enu[1]),
        float(enu[2]),
        state.V,
        state.psi,
        state.gamma,
        state.m,
    ]


def _enu_decision_to_geodetic_state(
    enu_row: np.ndarray,
    mass: float,
    anchor_lat: float,
    anchor_lon: float,
) -> GeodeticState:
    converter = _get_enu_to_geo_fn()
    # ``enu_row[2]`` is the geodetic altitude already (see the comment
    # in _geodetic_state_to_enu_decision).  We still hand it to the
    # converter as the 'u' component so the resulting geodetic altitude
    # accounts for the curvature drop between the anchor and the
    # displaced point at distance d ≈ √(e² + n²).
    geo = converter(
        float(enu_row[0]),
        float(enu_row[1]),
        float(enu_row[2]),
        anchor_lat,
        anchor_lon,
    )
    return GeodeticState(
        latitude=float(geo[0]),
        longitude=float(geo[1]),
        altitude=float(geo[2]),
        V=float(enu_row[3]),
        psi=float(enu_row[4]),
        gamma=float(enu_row[5]),
        m=mass,
    )


# --------------------------------------------------------------------------
# NLP builder
# --------------------------------------------------------------------------

def make_direct_collocation_solver(
    segment_num: int,
    aero_params_obj: AeroParams,
    aircraft_meta: dict,
):
    """Build the Hermite-Simpson NLP symbolically.

    Initial state, target state, and segment duration are passed as
    parameters so the same compiled solver can serve every optimisation
    request without rebuilding.  This matches the strategy used in
    ``casadi_optimizer.make_multiple_shooting_solver``.
    """
    # 1. Continuous fixed-ENU RHS reused from the simulator.  There is
    # no second dynamics model to keep in sync.
    model = make_dynamics_model()
    rhs_func = model['rhs_func']

    # 2. Symbolic parameters.
    start_state = ca.SX.sym('start_state', 7)   # e, n, h, V, psi, gamma, m
    target_state = ca.SX.sym('target_state', 7)
    duration = ca.SX.sym('duration')

    aero_params = ca.vertcat(
        aero_params_obj.S,
        aero_params_obj.Cl_max,
        aero_params_obj.Cd0,
        aero_params_obj.k,
        aero_params_obj.stall_threshold,
        aero_params_obj.k_stall,
    )

    # 3. Per-segment step h.  All segments are equal length; this is
    # the natural choice when control is piecewise-constant.
    segment_h = duration / segment_num

    # 4. Decision variables.  Layout matches the multiple-shooting
    # optimiser: controls first, then segment-end states.  Initial
    # state is fixed (parameter) so it does not appear in w.
    seg_states = []
    seg_controls = []
    defects = []

    # Mass is a scalar parameter; we splice it back onto the 6-state
    # decision vector whenever we need to call rhs_func, which expects
    # the full 7-state.
    mass_param = start_state[6]
    x_prev_decision = start_state[:6]

    for k in range(segment_num):
        uk = ca.SX.sym(f'u_{k}', CONTROL_DIM)
        xk_next_decision = ca.SX.sym(f'x_{k+1}', STATE_DIM)
        seg_controls.append(uk)
        seg_states.append(xk_next_decision)

        x_prev_full = ca.vertcat(x_prev_decision, mass_param)
        x_next_full = ca.vertcat(xk_next_decision, mass_param)

        # Mass derivative is identically zero in this model, so the
        # 7th defect component is structurally zero.  Drop it to keep
        # the constraint Jacobian rectangular.
        segment_defect = hermite_simpson_defect_expr(
            rhs_func, x_prev_full, x_next_full, uk, aero_params, segment_h,
        )[:STATE_DIM]
        defects.append(segment_defect)

        x_prev_decision = xk_next_decision

    # 5. Terminal equality: last knot must reach the requested target.
    defects.append(seg_states[-1] - target_state[:STATE_DIM])

    # 6. Bounds.  Identical envelope to ``casadi_optimizer.py`` so the
    # two CasADi optimisers stay comparable.
    state_lb, state_ub = make_state_bounds(
        min_altitude=aircraft_meta['min_altitude'],
        min_velocity=aircraft_meta['min_terminal_speed'],
    )
    control_lb, control_ub = make_control_bounds(
        aircraft_meta['max_thrust'],
        aircraft_meta['min_load_factor'],
        aircraft_meta['max_load_factor'],
    )

    w = seg_controls + seg_states
    lbw = control_lb * segment_num + state_lb * segment_num
    ubw = control_ub * segment_num + state_ub * segment_num

    g = ca.vertcat(*defects)
    lbg = [0.0] * g.shape[0]
    ubg = [0.0] * g.shape[0]

    # 7. Objective.  Same fixed-time cost as the multiple-shooting
    # CasADi optimiser: minimise mean squared scaled control effort so
    # the NLP stays well-posed when arrival time is fixed.
    scale_thrust = aircraft_meta['max_thrust']
    scale_mu = ca.pi / 2.0
    scale_n = max(
        abs(aircraft_meta['min_load_factor']),
        abs(aircraft_meta['max_load_factor']),
    )
    cost = ca.SX(0)
    for uk in seg_controls:
        scaled = ca.vertcat(uk[0] / scale_thrust, uk[1] / scale_mu, uk[2] / scale_n)
        cost += ca.dot(scaled, scaled)
    cost = cost / segment_num

    nlp = {
        'f': cost,
        'x': ca.vertcat(*w),
        'g': g,
        'p': ca.vertcat(start_state, target_state, duration),
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp)
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
):
    """Build the Hermite-Simpson NLP with ``T`` as a decision variable.

    Objective is ``T/T_max + λ · mean(||u_scaled||²)`` so the time term
    dominates (the regularisation is a tie-breaker that keeps the
    control profile from drifting along an arbitrary null-space when
    several controls reach the same arrival time).
    """
    # 1. Same continuous RHS as the fixed-time NLP.
    model = make_dynamics_model()
    rhs_func = model['rhs_func']

    # 2. Symbolic parameters.  ``max_duration`` is the upper bound for
    # the decision-variable duration and is also the scale factor used
    # to non-dimensionalise the time term in the objective.
    start_state = ca.SX.sym('start_state', 7)
    target_state = ca.SX.sym('target_state', 7)
    max_duration = ca.SX.sym('max_duration')

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

    # 4. Decision variables: controls, segment-end states, then T at
    # the end so we can slice it out of ``sol['x']`` cleanly.
    seg_states = []
    seg_controls = []
    defects = []

    mass_param = start_state[6]
    x_prev_decision = start_state[:6]

    for k in range(segment_num):
        uk = ca.SX.sym(f'u_{k}', CONTROL_DIM)
        xk_next_decision = ca.SX.sym(f'x_{k+1}', STATE_DIM)
        seg_controls.append(uk)
        seg_states.append(xk_next_decision)

        x_prev_full = ca.vertcat(x_prev_decision, mass_param)
        x_next_full = ca.vertcat(xk_next_decision, mass_param)

        segment_defect = hermite_simpson_defect_expr(
            rhs_func, x_prev_full, x_next_full, uk, aero_params, segment_h,
        )[:STATE_DIM]
        defects.append(segment_defect)

        x_prev_decision = xk_next_decision

    # 5. Terminal equality, same as fixed-time NLP.
    defects.append(seg_states[-1] - target_state[:STATE_DIM])

    # 6. Bounds.
    state_lb, state_ub = make_state_bounds(
        min_altitude=aircraft_meta['min_altitude'],
        min_velocity=aircraft_meta['min_terminal_speed'],
    )
    control_lb, control_ub = make_control_bounds(
        aircraft_meta['max_thrust'],
        aircraft_meta['min_load_factor'],
        aircraft_meta['max_load_factor'],
    )
    # Duration is bounded at the bottom by ``T_min`` (a small but
    # positive floor that keeps ``segment_h > 0``) and at the top by
    # ``max_duration`` (passed in by the caller).  We use a placeholder
    # in the symbolic bounds and overwrite the actual numeric value at
    # solve time -- but because IPOPT requires numeric bounds, we
    # cannot put a parameter there.  Instead we bind T_max into the
    # bound at solve time via the caller (see ``optimize_free_time``).
    duration_lb = aircraft_meta.get('min_duration', _DEFAULT_MIN_DURATION_S)
    # Upper bound is initially set to a very large finite value here;
    # ``optimize_free_time`` re-binds it to the requested max_duration
    # by passing ``lbx``/``ubx`` to the solver call.
    duration_ub = 1e6

    w = seg_controls + seg_states + [duration_var]
    lbw = control_lb * segment_num + state_lb * segment_num + [duration_lb]
    ubw = control_ub * segment_num + state_ub * segment_num + [duration_ub]

    g = ca.vertcat(*defects)
    lbg = [0.0] * g.shape[0]
    ubg = [0.0] * g.shape[0]

    # 7. Objective.  Time dominates; control effort is a small
    # regulariser.  Both terms are non-dimensional so IPOPT sees a
    # well-scaled gradient.
    scale_thrust = aircraft_meta['max_thrust']
    scale_mu = ca.pi / 2.0
    scale_n = max(
        abs(aircraft_meta['min_load_factor']),
        abs(aircraft_meta['max_load_factor']),
    )
    control_cost = ca.SX(0)
    for uk in seg_controls:
        scaled = ca.vertcat(uk[0] / scale_thrust, uk[1] / scale_mu, uk[2] / scale_n)
        control_cost += ca.dot(scaled, scaled)
    control_cost = control_cost / segment_num

    cost = duration_var / max_duration + time_regularization * control_cost

    nlp = {
        'f': cost,
        'x': ca.vertcat(*w),
        'g': g,
        'p': ca.vertcat(start_state, target_state, max_duration),
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp)
    return solver, lbw, ubw, lbg, ubg


# --------------------------------------------------------------------------
# Optimiser class
# --------------------------------------------------------------------------

class CasadiDirectCollocationOptimizer:
    """Direct-collocation trajectory optimiser in a fixed ENU frame.

    API-compatible with ``CasadiOptimizer`` so the backend dispatcher
    can substitute either optimiser behind the same call shape.
    """

    def __init__(self, n_segments: int, dt: float, max_duration: float, aircraft: AircraftSpec):
        if n_segments < 2:
            raise ValueError("n_segments must be at least 2 for direct collocation")
        self.n_segments = n_segments
        # ``dt`` is accepted for API parity with the multiple-shooting
        # optimiser.  Direct collocation does NOT subdivide a segment
        # with explicit RK4 substeps; dynamics fidelity is controlled by
        # ``n_segments`` and the 3rd-order Hermite-Simpson formula.
        self.dt = dt
        self.max_duration = max_duration
        self.aircraft = aircraft
        self.aero_params = AeroParams(S=aircraft.wing_area_m2)
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
        # duration (e.g. a Controlled Time of Arrival).
        self.solver, self.lbw, self.ubw, self.lbg, self.ubg = (
            make_direct_collocation_solver(
                segment_num=n_segments,
                aero_params_obj=self.aero_params,
                aircraft_meta=aircraft_meta,
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

        # 1. Anchor the fixed ENU frame at the initial geodetic point.
        # This keeps the initial state at (e, n) = (0, 0), which gives
        # the smallest possible curvature error budget for the segment
        # furthest from the anchor (i.e. the terminal segment).
        anchor_lat = initial_state.latitude
        anchor_lon = initial_state.longitude
        initial_param = _geodetic_state_to_enu_decision(initial_state, anchor_lat, anchor_lon)
        target_param = _geodetic_state_to_enu_decision(target_state, anchor_lat, anchor_lon)

        # 2. Build (or accept) the initial guess.  The default linear
        # interpolation is the same shape as casadi_optimizer; replace
        # it via the ``initial_guess`` argument for warm starts.
        x0 = (
            self._build_initial_guess(initial_param, target_param)
            if initial_guess is None
            else initial_guess
        )

        # 3. Pack parameters and solve.
        p = ca.vertcat(initial_param, target_param, solve_duration)
        sol = self.solver(
            x0=x0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p,
        )
        stats = self.solver.stats()
        if not stats.get("success", False):
            raise ValueError(
                "Direct collocation optimization failed: "
                + stats.get("return_status", "unknown")
            )

        w_opt = sol['x'].full().flatten()
        control_opt = w_opt[: self.n_segments * CONTROL_DIM].reshape(
            (self.n_segments, CONTROL_DIM),
        )
        state_enu = w_opt[self.n_segments * CONTROL_DIM:].reshape(
            (self.n_segments, STATE_DIM),
        )

        # 4. Re-project each node back to geodetic.  This is the only
        # place the boundary conversion happens, so any curvature
        # bookkeeping is concentrated here.
        state_geo = np.array([
            self._enu_decision_row_to_geo_array(
                row,
                initial_state.m,
                anchor_lat,
                anchor_lon,
            )
            for row in state_enu
        ])
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
        """
        anchor_lat = initial_state.latitude
        anchor_lon = initial_state.longitude
        initial_param = _geodetic_state_to_enu_decision(initial_state, anchor_lat, anchor_lon)
        target_param = _geodetic_state_to_enu_decision(target_state, anchor_lat, anchor_lon)

        if initial_guess is None:
            x0 = self._build_free_time_initial_guess(
                initial_state, target_state, max_duration,
            )
        else:
            x0 = list(initial_guess)

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
        sol = self.free_time_solver(
            x0=x0,
            lbx=lbw,
            ubx=ubw,
            lbg=self.free_time_lbg,
            ubg=self.free_time_ubg,
            p=p,
        )
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
        state_enu_block = w_opt[
            self.n_segments * CONTROL_DIM : self.n_segments * (CONTROL_DIM + STATE_DIM)
        ]
        state_enu = state_enu_block.reshape((self.n_segments, STATE_DIM))
        final_time = float(w_opt[-1])

        state_geo = np.array([
            self._enu_decision_row_to_geo_array(row, initial_state.m, anchor_lat, anchor_lon)
            for row in state_enu
        ])
        return final_time, control_opt, state_geo

    @staticmethod
    def solution_to_initial_guess(controls, states):
        # Backwards-compatible packing so warm-starts can be passed
        # back into ``optimize_trajectory`` without reformatting.  Note:
        # ``states`` here is the geodetic Nx6 array we return from
        # ``optimize_trajectory``; we do NOT re-project it to ENU
        # because the warm-start is only used as a search-space anchor,
        # not as a feasibility certificate.  The structure mirrors
        # ``CasadiOptimizer.solution_to_initial_guess`` for the same
        # reason.
        return list(controls.flatten()) + list(states.flatten())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_initial_guess(self, initial_param, target_param):
        """Straight-line interpolation between the ENU endpoints, plus
        a neutral approach control profile."""
        control_guess = [
            self.aircraft.approach_thrust_guess_n,
            0.0,
            1.0,
        ] * self.n_segments
        state_guess: list[float] = []
        for k in range(self.n_segments):
            ratio = (k + 1) / self.n_segments
            state_guess.extend([
                initial_param[i] + (target_param[i] - initial_param[i]) * ratio
                for i in range(STATE_DIM)
            ])
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

        If the fixed-time warm-start itself fails (rare), fall back to
        the cold linear-interpolation guess and let IPOPT do its best.
        """
        try:
            _, controls, states_geo = self.optimize_trajectory(
                initial_state, target_state, duration=max_duration,
            )
        except ValueError:
            anchor_lat = initial_state.latitude
            anchor_lon = initial_state.longitude
            initial_param = _geodetic_state_to_enu_decision(
                initial_state, anchor_lat, anchor_lon,
            )
            target_param = _geodetic_state_to_enu_decision(
                target_state, anchor_lat, anchor_lon,
            )
            cold = self._build_initial_guess(initial_param, target_param)
            return cold + [max_duration]

        # The states returned by ``optimize_trajectory`` are geodetic;
        # the free-time NLP expects ENU on its decision side.  Convert
        # each row back through the same anchor used by both solvers.
        anchor_lat = initial_state.latitude
        anchor_lon = initial_state.longitude
        states_enu = []
        for state_row in states_geo:
            geo_state = GeodeticState(
                latitude=float(state_row[0]),
                longitude=float(state_row[1]),
                altitude=float(state_row[2]),
                V=float(state_row[3]),
                psi=float(state_row[4]),
                gamma=float(state_row[5]),
                m=initial_state.m,
            )
            enu_param = _geodetic_state_to_enu_decision(
                geo_state, anchor_lat, anchor_lon,
            )
            # _geodetic_state_to_enu_decision returns a 7-vector
            # (including mass).  The decision vector uses only the
            # first 6 entries.
            states_enu.extend(enu_param[:STATE_DIM])

        return list(controls.flatten()) + states_enu + [max_duration]

    @staticmethod
    def _enu_decision_row_to_geo_array(
        enu_row: np.ndarray,
        mass: float,
        anchor_lat: float,
        anchor_lon: float,
    ) -> np.ndarray:
        geo = _enu_decision_to_geodetic_state(enu_row, mass, anchor_lat, anchor_lon)
        return np.array([
            geo.latitude,
            geo.longitude,
            geo.altitude,
            geo.V,
            geo.psi,
            geo.gamma,
        ])
