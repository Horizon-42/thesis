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
while still capturing 3rd-order dynamics fidelity per segment.

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

import casadi as ca
import numpy as np

from aerodynamic_model.casadi_simulator import make_geodetic_dynamics_model, AeroParams
from aerodynamic_model.aircraft_sets import AircraftSpec
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


def radians_expr(degrees):
    return degrees * ca.pi / 180.0


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
    psi_min, psi_max = -ca.pi, ca.pi
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


def select_state_substeps(max_duration: float, segment_num: int) -> int:
    """Pick M so the state step ``T/(N*M)`` is about ``_TARGET_STATE_STEP_S``.

    ``max_duration`` is the horizon upper bound; if the solved time comes
    out shorter the state step is only finer, never coarser.
    """
    control_step = max_duration / segment_num
    substeps = round(control_step / _TARGET_STATE_STEP_S)
    return max(1, min(_MAX_STATE_SUBSTEPS, int(substeps)))


def _build_collocation_decision(
    rhs_func, aero_params, start_state, target_state,
    segment_h, segment_num, sub_steps,
):
    """Build controls, state nodes and defects for the dense-state HS NLP.

    Returns ``(seg_controls, state_nodes, defects)``: ``seg_controls`` has
    ``segment_num`` 3-vectors, ``state_nodes`` has ``segment_num*sub_steps``
    6-vectors (x_1 .. x_{N*M}).  Each sub-interval gets its own
    Hermite-Simpson defect using the shared control of its segment; the
    terminal-equality defect (last node == target) is appended last.
    """
    mass_param = start_state[6]
    state_h = segment_h / sub_steps
    seg_controls = []
    state_nodes = []
    defects = []
    x_prev = start_state[:6]
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
                hermite_simpson_defect_expr(
                    rhs_func, x_prev_full, x_next_full, uk, aero_params, state_h,
                )[:STATE_DIM]
            )
            x_prev = xnode
    defects.append(state_nodes[-1] - target_state[:STATE_DIM])
    return seg_controls, state_nodes, defects


def _scaled_control_cost(seg_controls, aircraft_meta):
    """Mean squared scaled control effort over the N control segments."""
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
    return cost / len(seg_controls)


# --------------------------------------------------------------------------
# NLP builder
# --------------------------------------------------------------------------

def make_direct_collocation_solver(
    segment_num: int,
    aero_params_obj: AeroParams,
    aircraft_meta: dict,
    sub_steps: int = 1,
):
    """Build the Hermite-Simpson NLP symbolically.

    Initial state, target state, and segment duration are passed as
    parameters so the same compiled solver can serve every optimisation
    request without rebuilding.  This matches the strategy used in
    ``casadi_optimizer.make_multiple_shooting_solver``.
    """
    # 1. Continuous geodetic RHS reused from the simulator.  There is
    # no second dynamics model to keep in sync, and no tangent plane.
    model = make_geodetic_dynamics_model()
    rhs_func = model['rhs_func']

    # 2. Symbolic parameters (lat/lon in radians).
    start_state = ca.SX.sym('start_state', 7)   # lat, lon, h, V, psi, gamma, m
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

    # 3. Control step h (per operational segment); the state step is
    # ``segment_h / sub_steps`` inside the collocation core.
    segment_h = duration / segment_num

    # 4. Decision variables + defects: N controls, N*sub_steps state nodes.
    seg_controls, state_nodes, defects = _build_collocation_decision(
        rhs_func, aero_params, start_state, target_state,
        segment_h, segment_num, sub_steps,
    )

    # 5. Bounds.  Identical envelope to ``casadi_optimizer.py`` so the
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

    w = seg_controls + state_nodes
    lbw = control_lb * segment_num + state_lb * len(state_nodes)
    ubw = control_ub * segment_num + state_ub * len(state_nodes)

    g = ca.vertcat(*defects)
    lbg = [0.0] * g.shape[0]
    ubg = [0.0] * g.shape[0]

    # 6. Objective: minimise mean squared scaled control effort so the NLP
    # stays well-posed when arrival time is fixed.
    cost = _scaled_control_cost(seg_controls, aircraft_meta)

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
    sub_steps: int = 1,
):
    """Build the Hermite-Simpson NLP with ``T`` as a decision variable.

    Objective is ``T/T_max + λ · mean(||u_scaled||²)`` so the time term
    dominates (the regularisation is a tie-breaker that keeps the
    control profile from drifting along an arbitrary null-space when
    several controls reach the same arrival time).
    """
    # 1. Same continuous geodetic RHS as the fixed-time NLP.
    model = make_geodetic_dynamics_model()
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

    # 4. Decision variables + defects: N controls, N*sub_steps state nodes,
    # then T appended last so we can slice it out of ``sol['x']`` cleanly.
    seg_controls, state_nodes, defects = _build_collocation_decision(
        rhs_func, aero_params, start_state, target_state,
        segment_h, segment_num, sub_steps,
    )

    # 5. Bounds.
    state_lb, state_ub = make_state_bounds(
        min_altitude=aircraft_meta['min_altitude'],
        min_velocity=aircraft_meta['min_terminal_speed'],
    )
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

    g = ca.vertcat(*defects)
    lbg = [0.0] * g.shape[0]
    ubg = [0.0] * g.shape[0]

    # 6. Objective.  Time dominates; control effort is a small regulariser.
    # Both terms are non-dimensional so IPOPT sees a well-scaled gradient.
    cost = (
        duration_var / max_duration
        + time_regularization * _scaled_control_cost(seg_controls, aircraft_meta)
    )

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
                sub_steps=self.state_substeps,
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
        # No tangent plane, no anchor -- the RHS is global.
        initial_param = _geodetic_state_to_decision(initial_state)
        target_param = _geodetic_state_to_decision(target_state)

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
            w_opt[self.n_segments * CONTROL_DIM:], initial_state.m,
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
        target_param = _geodetic_state_to_decision(target_state)

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
        n_state_block = self.n_segments * self.state_substeps * STATE_DIM
        state_block = w_opt[
            self.n_segments * CONTROL_DIM : self.n_segments * CONTROL_DIM + n_state_block
        ]
        final_time = float(w_opt[-1])
        state_geo = self._extract_node_states_geo(state_block, initial_state.m)
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

    def _extract_node_states_geo(self, state_block, mass: float) -> np.ndarray:
        """Pick the segment-endpoint state of each of the N control
        segments from the dense ``N*M`` state block and convert to
        geodetic degrees -- returns an ``(N, 6)`` array."""
        nodes = np.asarray(state_block).reshape(
            (self.n_segments * self.state_substeps, STATE_DIM),
        )
        endpoints = nodes[self.state_substeps - 1 :: self.state_substeps]
        return np.array([
            self._decision_row_to_geo_array(row, mass) for row in endpoints
        ])

    def _build_initial_guess(self, initial_param, target_param):
        """Straight-line interpolation across the N*M state nodes
        (radians), plus a neutral approach control profile (N controls)."""
        control_guess = [
            self.aircraft.approach_thrust_guess_n,
            0.0,
            1.0,
        ] * self.n_segments
        n_nodes = self.n_segments * self.state_substeps
        state_guess: list[float] = []
        for i in range(n_nodes):
            ratio = (i + 1) / n_nodes
            state_guess.extend([
                initial_param[d] + (target_param[d] - initial_param[d]) * ratio
                for d in range(STATE_DIM)
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

        The fixed-time decision vector ``[controls(3N), states(6*N*M)]`` is
        exactly the free-time decision minus the trailing ``T``, so the raw
        solution is reused verbatim.  If the fixed-time warm-start itself
        fails (rare), fall back to the cold linear-interpolation guess.
        """
        initial_param = _geodetic_state_to_decision(initial_state)
        target_param = _geodetic_state_to_decision(target_state)
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
