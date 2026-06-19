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

    
