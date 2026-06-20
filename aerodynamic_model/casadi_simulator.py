import casadi as ca
import numpy as np
from .casadi_exprs import isa_density_expr, aerodynamic_coefficients_expr
from .casadi_coordinates_converter import geodetic_deg_to_ecef_expr, ecef_to_geodetic_expr, ecef_to_enu_rotation_matrix_expr, geodetic_to_enu_expr, enu_to_geodetic_expr
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
    aero_coeffs = aerodynamic_coefficients_expr(n_cmd, V, h, m, [S, Cl_max, Cd0, k, stall_threshold, k_stall])
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

# create geodetic dynamics model
def make_geo_step_from_enu_integrator(dt):
    # Define symbolic variables for state and control
    lat = ca.SX.sym('lat')  # latitude (degrees)
    lon = ca.SX.sym('lon')  # longitude (degrees)
    alt = ca.SX.sym('alt')  # altitude (m)
    V = ca.SX.sym('V')  # velocity (m/s)
    psi = ca.SX.sym('psi')  # heading angle (rad)
    gamma = ca.SX.sym('gamma')  # flight path angle (rad)
    m = ca.SX.sym('m')  # mass (kg)
    x_geo = ca.vertcat(lat, lon, alt, V, psi, gamma, m)

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
    enu_integrator = make_integrator(enu_model, dt=dt) # create an integrator for the ENU dynamics

    x_enu0 = ca.vertcat(0.0, 0.0, alt, V, psi, gamma, m) 
    p = ca.vertcat(u, aero_params)
    x_enu_next = enu_integrator(x0=x_enu0, p=p)['xf']  # integrate ENU dynamics for one step, xf is the final state after integration

    # Convert the next ENU state back to geodetic coordinates
    ref_geo = ca.vertcat(lat, lon, 0.0)  # surface reference point for ENU frame, using current lat/lon and 0 altitude
    x_geo_next = ca.vertcat(*enu_state_to_geodetic_expr(x_enu_next, ref_geo), m)

    step_func = ca.Function('geo_step_func',
                            [x_geo, u, aero_params],
                            [x_geo_next],                            
                            ['x_geo', 'u', 'aero_params'],
                            ['x_geo_next'])
    return {
        "x": x_geo,
        "u": u,
        "aero_params": aero_params,
        "step_func": step_func,
        "enu_model": enu_model,
        "enu_integrator": enu_integrator,
    }




    