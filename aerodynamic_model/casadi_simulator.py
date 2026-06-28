import casadi as ca
import numpy as np
from .casadi_exprs import isa_density_expr, aerodynamic_coefficients_expr
from .casadi_coordinates_converter import (
    ecef_to_enu_rotation_matrix_expr,
    enu_to_geodetic_expr,
    geodetic_to_enu_expr,
    WGS84_A,
    WGS84_E2,
)

from aircraft.aircraft_sets import Aircraft
from aircraft.aero_params import AeroParams, aero_params_for_aircraft

from .common import GeodeticState, LoadFactorControl

# Dynamic model for load factor control, using CasADi for symbolic computation
def make_dynamics_model():
    # Define symbolic variables for state and control
    x = ca.SX.sym('x')  # horizontal position (m)
    y = ca.SX.sym('y')  # vertical position (m)
    h = ca.SX.sym('h')  # altitude (m)
    V = ca.SX.sym('V')  # velocity (m/s)
    psi = ca.SX.sym('psi')  # heading angle (rad)
    gamma = ca.SX.sym('gamma')  # flight path angle (rad)
    m = ca.SX.sym('m')  # mass (kg)

    T = ca.SX.sym('T')  # thrust (N)
    mu = ca.SX.sym('mu')  # bank angle (rad)
    n_cmd = ca.SX.sym('n_cmd')  # commanded load factor

    state_vec = ca.vertcat(x, y, h, V, psi, gamma, m)
    control_vec = ca.vertcat(T, mu, n_cmd)

    rho = isa_density_expr(h)

    # Define aircraft metadata and aerodynamic parameters (these could be made symbolic if needed)
    S = ca.SX.sym('S')  # reference area (m^2)
    # CL
    Cl_max = ca.SX.sym('Cl_max')  # maximum lift coefficient
    # Drag coefficients
    Cd0 = ca.SX.sym('Cd0')  # zero-lift drag coefficient
    k = ca.SX.sym('k')  # induced drag factor

    # Cd
    stall_threshold = ca.SX.sym('stall_threshold')  # stall threshold as a fraction of Cl_max
    k_stall = ca.SX.sym('k_stall')  # stall drag coefficient factor

    # Define and compute symbols for Cl and Cd
    aero_coeffs = aerodynamic_coefficients_expr(
        n_cmd,
        V,
        m,
        rho,
        [S, Cl_max, Cd0, k, stall_threshold, k_stall],
    )
    Cd, stalled =   aero_coeffs['Cd'], aero_coeffs['stalled']

    g = 9.81  # gravity (m/s^2)
    # Compute actually load factor
    n = ca.if_else(stalled, 0.5*rho*V**2*Cl_max*S/(m*g), n_cmd)

    # Compute aerodynamic forces
    D = 0.5 * rho * V**2 * Cd * S

    # Define dynamics equations
    dxdt = V * ca.cos(gamma) * ca.cos(psi)
    dydt = V * ca.cos(gamma) * ca.sin(psi)
    dhdt = V * ca.sin(gamma)
    dVdt = (T - D) / m - g * ca.sin(gamma)
    dpsidt = g*n * ca.sin(mu) / (V * ca.cos(gamma))
    dgamadt = g * (n * ca.cos(mu) - ca.cos(gamma)) / V
    dmdt = 0.0  # assume mass is constant

    # Return the dae system
    xdot = ca.vertcat(dxdt, dydt, dhdt, dVdt, dpsidt, dgamadt, dmdt)
    # pack meta parameters 
    aero_params = ca.vertcat(S, Cl_max, Cd0, k, stall_threshold, k_stall)

    p = ca.vertcat(control_vec, aero_params)
    dae = {'x': state_vec, 'p': p, 'ode': xdot}

    # RHS function
    rhs_func = ca.Function('rhs_func', 
                           [state_vec, control_vec, aero_params], 
                           [xdot],
                           ['x','u','aero_params'],
                           ['xdot'])

    return {
        'x': state_vec,
        'u': control_vec,
        'aero_params': aero_params,
        'xdot': xdot,
        'dae': dae,
        'rhs_func': rhs_func,
        "aux":{
            'rho': rho,
            'Cl': aero_coeffs['Cl'],
            'Cd': aero_coeffs['Cd'],
            'stalled': aero_coeffs['stalled'],
            'n': n,
            'D': D,
        }
    }

def make_integrator(model, dt):
    # Create an integrator for the dynamics using RK4
    opts = {'tf': dt}
    integrator = ca.integrator('step', 'rk', model['dae'], opts)
    return integrator

# Speed decomposition and conversion between ECEF and ENU frames
def get_enu_velocity_components_expr(V, psi, gamma):
    V_east = V * ca.cos(gamma) * ca.cos(psi) # psi=0 heading east
    V_north = V * ca.cos(gamma) * ca.sin(psi)
    V_up = V * ca.sin(gamma)
    return V_east, V_north, V_up

def enu_to_ecef_velocity_expr(V_east, V_north, V_up, ref_geo):
    """
    Convert local ENU velocity components to ECEF velocity components using the reference geodetic coordinate.
    @param V_east: velocity component in the east direction (m/s)
    @param V_north: velocity component in the north direction (m/s)
    @param V_up: velocity component in the up direction (m/s)
    @param ref_geo: reference geodetic coordinate (latitude, longitude, altitude)
    """
    # Get rotation matrix from ENU to ECEF
    R = ecef_to_enu_rotation_matrix_expr(ref_geo[0], ref_geo[1])

    # Convert ENU velocity to ECEF velocity
    V_ecef = R.T @ ca.vertcat(V_east, V_north, V_up)  # transpose of rotation matrix for ENU to ECEF conversion
    return V_ecef[0], V_ecef[1], V_ecef[2]

def ecef_to_enu_velocity_expr(vx, vy, vz, ref_geo):
    # Get rotation matrix from ECEF to ENU
    R = ecef_to_enu_rotation_matrix_expr(ref_geo[0], ref_geo[1])

    # Convert ECEF velocity to ENU velocity
    V_enu = R @ ca.vertcat(vx, vy, vz)
    return V_enu[0], V_enu[1], V_enu[2]

def enu_state_to_geodetic_expr(x_enu, ref_geo):
    # Convert ENU state to geodetic coordinates
    lat, lon, alt = enu_to_geodetic_expr(x_enu[0], x_enu[1], x_enu[2], ref_geo[0], ref_geo[1], ref_geo[2])
    
    V_east, V_north, V_up = get_enu_velocity_components_expr(x_enu[3], x_enu[4], x_enu[5]) # V, psi, 
    # Convert ENU velocity to ECEF velocity
    vx, vy, vz = enu_to_ecef_velocity_expr(V_east, V_north, V_up, ref_geo)
    # Convert ECEF velocity to geodetic velocity components (V, psi, gamma) with new reference point
    new_geo_frame = ca.vertcat(lat, lon, 0.0) # new reference frame for velocity conversion, using updated lat/lon and 0 altitude
    V_east_new, V_north_new, V_up_new = ecef_to_enu_velocity_expr(vx, vy, vz, new_geo_frame)
    V_new = ca.sqrt(V_east_new**2 + V_north_new**2 + V_up_new**2)
    horizontal_V = ca.sqrt(V_east_new**2 + V_north_new**2)
    psi_new = ca.atan2(V_north_new, V_east_new)
    gamma_new = ca.atan2(V_up_new, horizontal_V)
    return lat, lon, alt, V_new, psi_new, gamma_new


def geodetic_state_to_enu_expr(x_geo, ref_geo):
    """Forward of ``enu_state_to_geodetic_expr``: a geodetic state
    ``(lat, lon [deg], alt, V, psi, gamma)`` -> the ENU state
    ``(east, north, up, V, psi_enu, gamma_enu)`` in the fixed frame anchored at
    ``ref_geo = (lat, lon [deg], alt)``.  Position by the WGS84 conversion;
    velocity reprojected (local frame -> ECEF -> ref frame)."""
    lat, lon, alt = x_geo[0], x_geo[1], x_geo[2]
    east, north, up = geodetic_to_enu_expr(lat, lon, alt, ref_geo[0], ref_geo[1], ref_geo[2])
    V_east, V_north, V_up = get_enu_velocity_components_expr(x_geo[3], x_geo[4], x_geo[5])
    vx, vy, vz = enu_to_ecef_velocity_expr(V_east, V_north, V_up, ca.vertcat(lat, lon, 0.0))
    Ve_r, Vn_r, Vu_r = ecef_to_enu_velocity_expr(vx, vy, vz, ref_geo)
    V_r = ca.sqrt(Ve_r**2 + Vn_r**2 + Vu_r**2)
    horiz = ca.sqrt(Ve_r**2 + Vn_r**2)
    return east, north, up, V_r, ca.atan2(Vn_r, Ve_r), ca.atan2(Vu_r, horiz)

def rk4_step_expr(rhs_expr, x, u, aero_params, dt):
    k1 = rhs_expr(x, u, aero_params)
    k2 = rhs_expr(x + 0.5 * dt * k1, u, aero_params)
    k3 = rhs_expr(x + 0.5 * dt * k2, u, aero_params)
    k4 = rhs_expr(x + dt * k3, u, aero_params)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# create geodetic dynamics model
def make_geo_step_from_enu_integrator():
    # Define symbolic variables for state and control
    lat = ca.SX.sym('lat')  # latitude (degrees)
    lon = ca.SX.sym('lon')  # longitude (degrees)
    alt = ca.SX.sym('alt')  # altitude (m)
    V = ca.SX.sym('V')  # velocity (m/s)
    psi = ca.SX.sym('psi')  # heading angle (rad)
    gamma = ca.SX.sym('gamma')  # flight path angle (rad)
    m = ca.SX.sym('m')  # mass (kg)
    x_geo = ca.vertcat(lat, lon, alt, V, psi, gamma, m)

    dt = ca.SX.sym('dt')  # time step for integration

    T = ca.SX.sym('T')  # thrust (N)
    mu = ca.SX.sym('mu')  # bank angle (rad)
    n_cmd = ca.SX.sym('n_cmd')  # commanded load factor
    u = ca.vertcat(T, mu, n_cmd)

    # Define aircraft parameters (these could be made symbolic if needed)
    S = ca.SX.sym('S')  # reference area (m^2)
    Cl_max = ca.SX.sym('Cl_max')  # maximum lift coefficient
    Cd0 = ca.SX.sym('Cd0')  # zero-lift drag coefficient
    k = ca.SX.sym('k')  # induced drag factor
    stall_threshold = ca.SX.sym('stall_threshold')  # stall threshold as a fraction of Cl_max
    k_stall = ca.SX.sym('k_stall')  # stall drag coefficient factor
    aero_params = ca.vertcat(S, Cl_max, Cd0, k, stall_threshold, k_stall)


    enu_model = make_dynamics_model()
    rhs_expr = enu_model['rhs_func']

    x_enu0 = ca.vertcat(0.0, 0.0, alt, V, psi, gamma, m) 
    p = ca.vertcat(u, aero_params)
    x_enu_next = rk4_step_expr(rhs_expr, x_enu0, u, aero_params, dt)

    # Convert the next ENU state back to geodetic coordinates
    ref_geo = ca.vertcat(lat, lon, 0.0)  # surface reference point for ENU frame, using current lat/lon and 0 altitude
    x_geo_next = ca.vertcat(*enu_state_to_geodetic_expr(x_enu_next, ref_geo), m)

    step_func = ca.Function('geo_step_func',
                            [x_geo, u, aero_params, dt],
                            [x_geo_next],                            
                            ['x_geo', 'u', 'aero_params', 'dt'],
                            ['x_geo_next'])
    return {
        "x": x_geo,
        "u": u,
        "aero_params": aero_params,
        "step_func": step_func,
        "enu_model": enu_model,
    }


# Note: a FIXED local-ENU tangent frame integrates entirely in ENU and converts
# geodetic <-> ENU only at the boundary (``geodetic_state_to_enu_expr`` in, then
# step the flat ``make_dynamics_model`` RHS in ENU, then
# ``enu_state_to_geodetic_expr`` out).  A per-STEP geo->ENU->geo stepper for it
# would just round-trip redundantly, so it does not need its own integrator --
# see ``dynamics_comparison_30km`` (system A) for the boundary-only version, and
# the ``localEnu`` optimiser scheme, which converts each node for its defect.


# ---------------------------------------------------------------------------
# Geodetic-frame continuous dynamics (single continuous ODE, no re-anchoring)
# ---------------------------------------------------------------------------
#
# Why this exists
# ---------------
# ``make_dynamics_model`` integrates in a flat ENU tangent plane, and
# ``make_geo_step_from_enu_integrator`` re-anchors that plane every RK4 step
# so the integration always sees a fresh local-tangent frame.  Because that
# stepper is a *discrete* one-step map ``Phi(x, u, h)`` rather than a
# continuous RHS, it cannot be embedded in a *polynomial collocation* defect
# (trapezoidal / Hermite-Simpson evaluate ``f`` and fit a polynomial — they
# need the RHS, not a stepper).  It CAN be used as a *shooting* defect
# (``x_{k+1} - Phi(x_k, u_k, h) = 0``); the optimiser exposes exactly that as
# the ``reanchoredEnu`` scheme.  The continuous model below is what unlocks
# the polynomial schemes AND lets a plain RK4 of one smooth RHS reproduce the
# re-anchored stepper's continuous limit.
#
# This model removes the need for any tangent plane at all: it writes the
# point-mass kinematics *directly* in geodetic coordinates ``(lat, lon, h)``.
# The "coordinate transformation" that the re-anchored stepper performed
# discretely is now folded into the continuous RHS as the WGS84 radius-of-
# curvature factors ``1/(R_M+h)`` and ``1/((R_N+h)cos lat)``.  Because the
# RHS is one smooth function over the whole horizon, direct collocation
# applies natively AND a plain RK4 of this same RHS reproduces the
# re-anchored stepper's continuous limit (the two discrete systems share the
# same continuous vector field).
#
# Frame rotation / transport rate
# -------------------------------
# ``psi`` and ``gamma`` are defined relative to the *local* geographic
# (ENU) frame.  As the aircraft moves, that frame rotates over the curved
# earth, so ``psi`` and ``gamma`` change even with zero aerodynamic moment.
# That apparent rate is the **transport rate**.  The re-anchored ENU stepper
# captures it EXACTLY (it re-projects the velocity into the new local frame
# every step); to match it, the continuous geodetic RHS must carry the
# transport terms explicitly.
#
# Derivation.  A velocity direction fixed in the earth frame changes, inside
# the rotating ENU frame, at ``d(u_hat)/dt = -omega_en x u_hat`` with the
# transport angular velocity of the local frame relative to earth
#
#   omega_en (ENU) = ( -V_north/(R_M+h),  V_east/(R_N+h),  V_east*tan(lat)/(R_N+h) )
#
# Differentiating ``psi = atan2(u_N, u_E)`` and ``gamma = atan2(u_U, |u_h|)``
# and substituting gives the EXACT transport contribution:
#
#   gamma_dot += V cos(gamma) * ( sin^2(psi)/(R_M+h) + cos^2(psi)/(R_N+h) )           (exact, two terms)
#   psi_dot   += -V cos(gamma) cos(psi) tan(lat) / (R_N+h)                            (MAIN: meridian convergence = -omega_U)
#              +  V sin(gamma) sin(psi) cos(psi) * ( 1/(R_N+h) - 1/(R_M+h) )          (CROSS term)
#
# The ``gamma`` expression is exact as written.  The ``psi`` MAIN term
# dominates; the ``psi`` CROSS term is a product of two small factors --
# ``sin(gamma)`` (zero in level flight) and the curvature-radius difference
# ``1/(R_N+h) - 1/(R_M+h) = O(e^2)`` (zero on a sphere) -- so it is ~3-4
# orders of magnitude smaller in terminal-area flight.  It is NOT dropped
# silently: the ``transport`` argument selects how much is modelled.
#
#   "none"   -- no transport terms          (isolates how much they matter)
#   "approx" -- gamma (exact) + psi MAIN     (drops the cross term; historical default)
#   "full"   -- gamma + psi main + psi CROSS (the EXACT transport)
#
# See ``transport_term_comparison`` for the approx-vs-full quantification and
# ``geodetic_vs_reanchored_error`` for the transport-vs-no-transport one.

#: Valid ``transport`` modes for the geodetic dynamics (see comment above).
_TRANSPORT_MODES = ("none", "approx", "full")


def make_geodetic_dynamics_model(transport: str = "approx"):
    """Continuous point-mass dynamics expressed DIRECTLY in geodetic
    coordinates.

    State ``x = (lat, lon, h, V, psi, gamma, m)`` with **lat/lon in
    radians**, ``h`` geodetic altitude in metres.  Control and aero
    parameters match ``make_dynamics_model`` so the same callers and
    ``AeroParams`` work unchanged.

    ``transport`` selects the frame-rotation (transport-rate) model on
    ``psi``/``gamma`` -- one of ``"none"`` / ``"approx"`` / ``"full"`` (see
    the comment block above).  ``"approx"`` (the historical default) keeps
    the exact ``gamma`` transport and the dominant ``psi`` meridian-
    convergence term but DROPS the ``psi`` cross term; ``"full"`` adds it
    back for the exact transport; ``"none"`` drops all transport terms.
    """
    if transport not in _TRANSPORT_MODES:
        raise ValueError(
            f"unknown transport {transport!r}; choose from {list(_TRANSPORT_MODES)}"
        )
    lat = ca.SX.sym('lat')   # latitude (rad)
    lon = ca.SX.sym('lon')   # longitude (rad)
    h = ca.SX.sym('h')       # geodetic altitude (m)
    V = ca.SX.sym('V')       # velocity (m/s)
    psi = ca.SX.sym('psi')   # heading (rad), 0 = East, +ve toward North
    gamma = ca.SX.sym('gamma')  # flight path angle (rad)
    m = ca.SX.sym('m')       # mass (kg)
    state_vec = ca.vertcat(lat, lon, h, V, psi, gamma, m)

    T = ca.SX.sym('T')       # thrust (N)
    mu = ca.SX.sym('mu')     # bank angle (rad)
    n_cmd = ca.SX.sym('n_cmd')  # commanded load factor
    control_vec = ca.vertcat(T, mu, n_cmd)

    S = ca.SX.sym('S')
    Cl_max = ca.SX.sym('Cl_max')
    Cd0 = ca.SX.sym('Cd0')
    k = ca.SX.sym('k')
    stall_threshold = ca.SX.sym('stall_threshold')
    k_stall = ca.SX.sym('k_stall')
    aero_params = ca.vertcat(S, Cl_max, Cd0, k, stall_threshold, k_stall)

    # Atmosphere + aero use the same simplified models as the ENU RHS.  The
    # ISA density reads ``h`` as geodetic altitude (exact here, unlike the
    # fixed-ENU model which approximated it by the tangent-plane u-coord).
    rho = isa_density_expr(h)
    aero_coeffs = aerodynamic_coefficients_expr(
        n_cmd, V, m, rho,
        [S, Cl_max, Cd0, k, stall_threshold, k_stall],
    )
    Cd, stalled = aero_coeffs['Cd'], aero_coeffs['stalled']

    g = 9.81
    n = ca.if_else(stalled, 0.5 * rho * V**2 * Cl_max * S / (m * g), n_cmd)
    D = 0.5 * rho * V**2 * Cd * S

    # WGS84 radii of curvature at this latitude.
    sin_lat = ca.sin(lat)
    denom = 1.0 - WGS84_E2 * sin_lat**2
    R_N = WGS84_A / ca.sqrt(denom)                    # prime-vertical radius
    R_M = WGS84_A * (1.0 - WGS84_E2) / denom**1.5     # meridional radius

    cos_g = ca.cos(gamma)
    V_east = V * cos_g * ca.cos(psi)
    V_north = V * cos_g * ca.sin(psi)
    V_up = V * ca.sin(gamma)

    # Position kinematics: the continuous coordinate transformation.
    lat_dot = V_north / (R_M + h)
    lon_dot = V_east / ((R_N + h) * ca.cos(lat))
    h_dot = V_up

    # Force / orientation dynamics (identical to the ENU RHS).
    V_dot = (T - D) / m - g * ca.sin(gamma)
    psi_dot = g * n * ca.sin(mu) / (V * cos_g)
    gamma_dot = g * (n * ca.cos(mu) - cos_g) / V

    if transport in ("approx", "full"):
        # gamma transport is exact (both terms); psi gets the dominant
        # meridian-convergence (MAIN) term.
        gamma_dot = gamma_dot + V * cos_g * (
            ca.sin(psi)**2 / (R_M + h) + ca.cos(psi)**2 / (R_N + h)
        )
        psi_dot = psi_dot - V * cos_g * ca.cos(psi) * ca.tan(lat) / (R_N + h)
    if transport == "full":
        # Exact psi CROSS term that "approx" drops -- the product of
        # sin(gamma) and the curvature-radius difference (O(e^2 sin gamma)).
        psi_dot = psi_dot + V * ca.sin(gamma) * ca.sin(psi) * ca.cos(psi) * (
            1.0 / (R_N + h) - 1.0 / (R_M + h)
        )

    dmdt = 0.0
    xdot = ca.vertcat(lat_dot, lon_dot, h_dot, V_dot, psi_dot, gamma_dot, dmdt)

    p = ca.vertcat(control_vec, aero_params)
    dae = {'x': state_vec, 'p': p, 'ode': xdot}

    rhs_func = ca.Function(
        'geo_rhs_func',
        [state_vec, control_vec, aero_params],
        [xdot],
        ['x', 'u', 'aero_params'],
        ['xdot'],
    )

    return {
        'x': state_vec,
        'u': control_vec,
        'aero_params': aero_params,
        'xdot': xdot,
        'dae': dae,
        'rhs_func': rhs_func,
        'transport': transport,
        'aux': {
            'rho': rho,
            'Cl': aero_coeffs['Cl'],
            'Cd': aero_coeffs['Cd'],
            'stalled': stalled,
            'n': n,
            'D': D,
            'R_M': R_M,
            'R_N': R_N,
        },
    }


def make_geodetic_step_integrator(transport: str = "approx"):
    """RK4 stepper for ``make_geodetic_dynamics_model``.

    The external state uses lat/lon in **degrees** (so it is a drop-in
    counterpart to ``make_geo_step_from_enu_integrator`` and matches
    ``GeodeticState``), while the RK4 advance happens in radians on the
    continuous geodetic RHS.  Unlike the re-anchored stepper, there is no
    per-step frame swap: it integrates one continuous ODE.

    ``transport`` is forwarded to ``make_geodetic_dynamics_model`` (``"none"``
    / ``"approx"`` / ``"full"``).
    """
    model = make_geodetic_dynamics_model(transport=transport)
    rhs_expr = model['rhs_func']

    lat = ca.SX.sym('lat')   # degrees
    lon = ca.SX.sym('lon')   # degrees
    alt = ca.SX.sym('alt')
    V = ca.SX.sym('V')
    psi = ca.SX.sym('psi')
    gamma = ca.SX.sym('gamma')
    m = ca.SX.sym('m')
    x_geo = ca.vertcat(lat, lon, alt, V, psi, gamma, m)

    dt = ca.SX.sym('dt')

    T = ca.SX.sym('T')
    mu = ca.SX.sym('mu')
    n_cmd = ca.SX.sym('n_cmd')
    u = ca.vertcat(T, mu, n_cmd)

    S = ca.SX.sym('S')
    Cl_max = ca.SX.sym('Cl_max')
    Cd0 = ca.SX.sym('Cd0')
    k = ca.SX.sym('k')
    stall_threshold = ca.SX.sym('stall_threshold')
    k_stall = ca.SX.sym('k_stall')
    aero_params = ca.vertcat(S, Cl_max, Cd0, k, stall_threshold, k_stall)

    deg2rad = ca.pi / 180.0
    x_rad = ca.vertcat(lat * deg2rad, lon * deg2rad, alt, V, psi, gamma, m)
    x_rad_next = rk4_step_expr(rhs_expr, x_rad, u, aero_params, dt)
    x_geo_next = ca.vertcat(
        x_rad_next[0] / deg2rad,
        x_rad_next[1] / deg2rad,
        x_rad_next[2],
        x_rad_next[3],
        x_rad_next[4],
        x_rad_next[5],
        x_rad_next[6],
    )

    step_func = ca.Function(
        'geodetic_step_func',
        [x_geo, u, aero_params, dt],
        [x_geo_next],
        ['x_geo', 'u', 'aero_params', 'dt'],
        ['x_geo_next'],
    )
    return {
        "x": x_geo,
        "u": u,
        "aero_params": aero_params,
        "step_func": step_func,
        "model": model,
    }


class CasadiSimulator:
    g = 9.81  # gravity (m/s^2)

    def __init__(self, aircraft: Aircraft, dt: float):
        self.aircraft = aircraft
        self.aero_params = aero_params_for_aircraft(aircraft)
        self.geo_step = make_geo_step_from_enu_integrator()["step_func"]

    def step(self, state: GeodeticState, control: LoadFactorControl, dt: float) -> GeodeticState:
        x_geo = ca.vertcat(state.latitude, state.longitude, state.altitude, state.V, state.psi, state.gamma, state.m)
        u = ca.vertcat(control.thrust, control.bank_rad, control.load_factor)
        aero_params_vec = ca.vertcat(self.aero_params.S, self.aero_params.Cl_max, self.aero_params.Cd0, self.aero_params.k, self.aero_params.stall_threshold, self.aero_params.k_stall)

        result = self.geo_step(x_geo=x_geo, u=u, aero_params=aero_params_vec, dt=dt)
        x_geo_next = np.array(result["x_geo_next"], dtype=float).reshape(-1)
        return GeodeticState(*x_geo_next.tolist())

    def optimize_trajectory(self, initial_state: GeodeticState, target_state: GeodeticState, N: int, dt: float):
        # This function can be implemented to optimize a trajectory using CasADi's NLP solvers, given an initial state and a sequence of controls.
        pass
