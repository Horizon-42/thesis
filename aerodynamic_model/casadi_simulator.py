import casadi as ca
import numpy as np
from casadi_atmosphere import make_atmosphere_model


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

    atmosphere_model = make_atmosphere_model()
    rho = atmosphere_model(h)

    # Define aircraft metadata and aerodynamic parameters (these could be made symbolic if needed)
    S = ca.SX.sym('S')  # reference area (m^2)
    # CL
    Cl_max = ca.SX.sym('Cl_max')  # maximum lift coefficient
    # Drag coefficients
    Cd0 = ca.SX.sym('Cd0')  # zero-lift drag coefficient
    k = ca.SX.sym('k')  # induced drag factor

    # Define and compute symbols for Cl and Cd
    g = 9.81  # gravity (m/s^2)
    Cl_req = n_cmd*m*g/(0.5*rho*S*V**2)  
    # Cl
    r = Cl_req / Cl_max
    stalled = r > 1.0
    Cl = ca.if_else(stalled, Cl_max, Cl_req)  # cap Cl at Cl_max to model stall
    # Cd
    stall_threshold = ca.SX.sym('stall_threshold')  # stall threshold as a fraction of Cl_max
    k_stall = ca.SX.sym('k_stall')  # stall drag coefficient factor
    # smooth factor
    stall_fraction = ca.fmin(r, 1.0)
    x_stall = ca.fmin(ca.fmax((stall_fraction - stall_threshold) / (1.0 - stall_threshold), 0.0), 1.0)
    smooth = x_stall * x_stall * (3 - 2 * x_stall)  # smooth transition for stall drag
    Cd_stall = ca.if_else(r > stall_threshold, smooth * k_stall, 0.0)  # simple stall drag model
    Cd = Cd0 + k * Cl**2 + Cd_stall

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
    }

def make_integrator(model, dt):
    # Create an integrator for the dynamics using RK4
    opts = {'tf': dt}
    integrator = ca.integrator('step', 'rk', model['dae'], opts)
    return integrator

