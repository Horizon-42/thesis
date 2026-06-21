import math

import casadi as ca
from aerodynamic_model.casadi_simulator import make_geo_step_from_enu_integrator, AeroParams

def segement_integrate_expr(step_func, x_start, u, aero_params, dt:float, duration: float, n_steps: int):
    dt_step = duration / n_steps
    xk = x_start
    for k in range(n_steps):
        xk = step_func(x_geo=xk, u=u, aero_params=aero_params, dt=dt_step)['x_geo_next']
    return xk

def make_control_bounds(max_thrust: float, min_load_factor: float, max_load_factor: float):
    # Define control bounds
    T_min = 0.0
    T_max = max_thrust
    mu_min = -ca.pi / 4  # max bank angle of 45 degrees
    mu_max = ca.pi / 4
    n_cmd_min = min_load_factor  # minimum load factor (can be adjusted based on aircraft capabilities)
    n_cmd_max = max_load_factor  # maximum load factor (can be adjusted based on aircraft capabilities)
    return [T_min, mu_min, n_cmd_min], [T_max, mu_max, n_cmd_max]

def make_state_bounds():
    # Define state bounds (these can be adjusted based on the expected operating envelope of the aircraft)
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    alt_min, alt_max = 0.0, 10000.0  # altitude in meters
    V_min, V_max = 0.0, 1000.0  # velocity in m/s
    psi_min, psi_max = -ca.pi, ca.pi  # heading angle in radians
    gamma_min, gamma_max = -ca.pi/2, ca.pi/2  # flight path angle in radians
    return [lat_min, lon_min, alt_min, V_min, psi_min, gamma_min], [lat_max, lon_max, alt_max, V_max, psi_max, gamma_max]

def geo_state_to_decision_vector(state):
    return state[:6] # exclude mass from optimization state, it is constant in this model

def decision_vector_to_geo_state(vec, mass):
    return ca.vertcat(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], mass)

def make_multiple_shooting_solver(segment_num: int, dt: float, max_duration: float, aero_params_obj: AeroParams, aircraft_meta:dict):
    # State lat, lon, alt, V, psi, gamma
    # Control T, mu, n_cmd

    aero_params = ca.vertcat(aero_params_obj.S, aero_params_obj.Cl_max, aero_params_obj.Cd0, aero_params_obj.k, aero_params_obj.stall_threshold, aero_params_obj.k_stall)

    start_state = ca.SX.sym('start_state', 7) # excluding mass
    target_state = ca.SX.sym('target_state', 7)

    duration = ca.SX.sym('duration') # total duration of the trajectory, the optimization object

    step_func = make_geo_step_from_enu_integrator()['step_func']

    # start with an empty NLP
    w = [duration] # decision variable list, starting with duration
    lbw = [0.0] # lower bounds on decision variables
    ubw = [max_duration] # upper bounds on decision variables

    segment_duration = duration / segment_num
    segment_substeps = max(1, math.ceil((max_duration / segment_num) / dt))
    xk = geo_state_to_decision_vector(start_state) # exclude mass from optimization state, it is constant in this model
    seg_states = []
    
    seg_controls = []

    defects = []

    for k in range(segment_num):
        uk = ca.SX.sym(f'u_{k}', 3)
        seg_controls.append(uk)

        cur_geo_state = decision_vector_to_geo_state(xk, start_state[6]) # reconstruct the full geodetic state with mass for integration
        cur_geo_state = segement_integrate_expr(step_func, cur_geo_state, uk, aero_params, dt, segment_duration, segment_substeps)
        xk_next = geo_state_to_decision_vector(cur_geo_state) # convert back to optimization state format for the next segment
        xk_next_sym = ca.SX.sym(f'x_{k+1}', 6)

        # Add defect constraint
        defects.append(xk_next - xk_next_sym)

        xk = xk_next_sym
        seg_states.append(xk)
    
    # create states bounds
    state_lb, state_ub = make_state_bounds()
    state_lb = state_lb * segment_num # apply the same bounds to all states in all segments
    state_ub = state_ub * segment_num

    # create control bounds
    control_lb, control_ub = make_control_bounds(aircraft_meta['max_thrust'], aircraft_meta['min_load_factor'], aircraft_meta['max_load_factor'])
    control_lb = control_lb * segment_num # apply the same bounds to all controls in all segments
    control_ub = control_ub * segment_num

    # Add all decision variables and constraints to the NLP
    w.extend(seg_controls)
    w.extend(seg_states) # exclude the initial state from decision variables, it is a parameter
    lbw.extend(control_lb)
    ubw.extend(control_ub)
    lbw.extend(state_lb) # exclude initial state bounds
    ubw.extend(state_ub)
    
    # final state constraint
    defects.append(seg_states[-1] - target_state[:6])

    g = ca.vertcat(*defects)
    lbg = [0.0] * g.shape[0] # equality constraints
    ubg = [0.0] * g.shape[0]

    nlp = {'f': duration, 'x': ca.vertcat(*w), 'g': g, 'p': ca.vertcat(start_state, target_state)}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    return solver, lbw, ubw, lbg, ubg
