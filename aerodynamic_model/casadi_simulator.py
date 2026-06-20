import casadi as ca
import numpy as np
from .casadi_exprs import isa_density_expr, aerodynamic_coefficients_expr
from .casadi_coordinates_converter import ecef_to_enu_rotation_matrix_expr, enu_to_geodetic_expr

from .aircraft_sets import AircraftSpec, A320
from dataclasses import dataclass

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

@dataclass
class GeoState:
    latitude: float
    longitude: float
    altitude: float
    V: float
    psi: float
    gamma: float
    m: float

@dataclass
class AeroParams:
    S: float 
    Cl_max: float = 1.5
    Cd0: float = 0.02
    k: float = 0.04
    stall_threshold: float = 0.9
    k_stall: float = 0.1

@dataclass
class GeoControl:
    thrust: float
    bank_rad: float
    load_factor: float

class CasadiSimulator:
    g = 9.81  # gravity (m/s^2)

    def __init__(self, aircraft: AircraftSpec, dt: float):
        self.aircraft = aircraft
        self.aero_params = AeroParams(S=aircraft.wing_area_m2)
        self.geo_step = make_geo_step_from_enu_integrator()["step_func"]

    def step(self, state: GeoState, control: GeoControl, dt: float) -> GeoState:
        x_geo = ca.vertcat(state.latitude, state.longitude, state.altitude, state.V, state.psi, state.gamma, state.m)
        u = ca.vertcat(control.thrust, control.bank_rad, control.load_factor)
        aero_params_vec = ca.vertcat(self.aero_params.S, self.aero_params.Cl_max, self.aero_params.Cd0, self.aero_params.k, self.aero_params.stall_threshold, self.aero_params.k_stall)

        result = self.geo_step(x_geo=x_geo, u=u, aero_params=aero_params_vec, dt=dt)
        x_geo_next = np.array(result["x_geo_next"], dtype=float).reshape(-1)
        return GeoState(*x_geo_next.tolist())

    def optimize_trajectory(self, initial_state: GeoState, target_state: GeoState, N: int, dt: float):
        # This function can be implemented to optimize a trajectory using CasADi's NLP solvers, given an initial state and a sequence of controls.
        pass
