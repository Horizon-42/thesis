import math

import casadi as ca
from geokit import kt_to_ms
from aerodynamic_model.casadi_simulator import make_geo_step_from_enu_integrator
from aircraft.aero_params import AeroParams
from aircraft.aircraft_sets import AircraftSpec
from aerodynamic_model.common import GeodeticState

def radians_expr(degrees):
    return degrees * ca.pi / 180.0

def segement_integrate_expr(step_func, x_start, u, aero_params, dt:float, duration, n_steps: int):
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

def make_state_bounds(min_altitude:float, min_velocity:float):
    # Define state bounds (these can be adjusted based on the expected operating envelope of the aircraft)
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    alt_min, alt_max = min_altitude, 10000.0  # altitude in meters
    V_min, V_max = min_velocity, 1000.0  # velocity in m/s
    psi_min, psi_max = -ca.pi, ca.pi  # heading angle in radians
    gamma_min, gamma_max = -radians_expr(10), radians_expr(30)  # flight path angle in radians
    return [lat_min, lon_min, alt_min, V_min, psi_min, gamma_min], [lat_max, lon_max, alt_max, V_max, psi_max, gamma_max]

def state_vector_to_decision_vector(state):
    return state[:6] # exclude mass from optimization state, it is constant in this model

def decision_vector_to_state_vector(vec, mass):
    return ca.vertcat(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], mass)

def make_multiple_shooting_solver(segment_num: int, dt: float, max_duration: float, aero_params_obj: AeroParams, aircraft_meta:dict):
    # State lat, lon, alt, V, psi, gamma
    # Control T, mu, n_cmd

    aero_params = ca.vertcat(aero_params_obj.S, aero_params_obj.Cl_max, aero_params_obj.Cd0, aero_params_obj.k, aero_params_obj.stall_threshold, aero_params_obj.k_stall)

    start_state = ca.SX.sym('start_state', 7) # excluding mass
    target_state = ca.SX.sym('target_state', 7)
    duration = ca.SX.sym('duration')

    step_func = make_geo_step_from_enu_integrator()['step_func']

    # start with an empty NLP
    w = []
    lbw = []
    ubw = []

    segment_duration = duration / segment_num
    segment_substeps = max(1, math.ceil((max_duration / segment_num) / dt))
    xk = state_vector_to_decision_vector(start_state) # exclude mass from optimization state, it is constant in this model
    seg_states = []
    
    seg_controls = []

    defects = []

    for k in range(segment_num):
        uk = ca.SX.sym(f'u_{k}', 3)
        seg_controls.append(uk)

        cur_geo_state = decision_vector_to_state_vector(xk, start_state[6]) # reconstruct the full geodetic state with mass for integration
        cur_geo_state = segement_integrate_expr(step_func, cur_geo_state, uk, aero_params, dt, segment_duration, segment_substeps)
        xk_next = state_vector_to_decision_vector(cur_geo_state) # convert back to optimization state format for the next segment
        xk_next_sym = ca.SX.sym(f'x_{k+1}', 6)

        # Add defect constraint
        defects.append(xk_next - xk_next_sym)

        xk = xk_next_sym
        seg_states.append(xk)
    
    # create states bounds
    state_lb, state_ub = make_state_bounds(min_altitude=aircraft_meta['min_altitude'], min_velocity=aircraft_meta['min_terminal_speed'])
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

    nlp = {'f': ca.SX(0), 'x': ca.vertcat(*w), 'g': g, 'p': ca.vertcat(start_state, target_state, duration)}
    # opts = {
    #     "ipopt.max_cpu_time": 10,
    #     "ipopt.max_iter": 200,
    #     "ipopt.print_level": 5,
    # }
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    return solver, lbw, ubw, lbg, ubg

class CasadiOptimizer:
    def __init__(self, n_segments: int, dt: float, max_duration: float, aircraft: AircraftSpec):
        if n_segments < 2:
            raise ValueError("n_segments must be at least 2 for this multiple-shooting NLP")
        self.n_segments = n_segments
        self.dt = dt
        self.max_duration = max_duration
        self.aircraft = aircraft
        # build AeroParams from aircraft spec
        self.aero_params = AeroParams(S=aircraft.wing_area_m2)
        self.solver, self.lbw, self.ubw, self.lbg, self.ubg = make_multiple_shooting_solver(
            segment_num=n_segments,
            dt=dt,
            max_duration=max_duration,
            aero_params_obj=self.aero_params,
            aircraft_meta={
                "max_thrust": aircraft.max_thrust_n,
                "min_load_factor": 0.5,
                "max_load_factor": 2, # need to check the actual limits for the aircraft, these are just example values
                "min_terminal_speed": kt_to_ms(aircraft.terminal_speed_kt),
                "min_altitude": aircraft.threshold_crossing_height_m + 10.0, # set minimum altitude slightly above threshold crossing height to avoid infeasible solutions, can be tuned
            },)
    
    @staticmethod
    def geo_state_to_decision_vector(state: GeodeticState):
        return [state.latitude, state.longitude, state.altitude, state.V, state.psi, state.gamma, state.m]
    
    @staticmethod    
    def decision_vector_to_geo_state(vec):
        return GeodeticState(*vec.tolist())
    
    def build_initial_guess(self, initial, target):
        control_guess = [
            self.aircraft.approach_thrust_guess_n,
            0.0,
            1.0,
        ] * self.n_segments
        state_guess = []
        for k in range(self.n_segments):
            ratio = (k + 1) / self.n_segments
            state_guess.extend([
                initial[i] + (target[i] - initial[i]) * ratio
                for i in range(6)
            ])
        return control_guess + state_guess
    
    @staticmethod
    def solution_to_initial_guess(controls, states):
        return list(controls.flatten()) + list(states.flatten())
    
    def optimize_trajectory(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
        duration: float | None = None,
        initial_guess=None,
    ):
        # This function can be implemented to optimize a trajectory using CasADi's NLP solvers, given an initial state and a sequence of controls.
        initial = self.geo_state_to_decision_vector(initial_state)
        target = self.geo_state_to_decision_vector(target_state)
        solve_duration = self.max_duration if duration is None else duration
        
        x0 = self.build_initial_guess(initial, target) if initial_guess is None else initial_guess

        p = ca.vertcat(initial, target, solve_duration) # parameters for the NLP: initial state, target state, and fixed duration
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
            raise ValueError("CasADi optimization failed: " + stats.get("return_status", "unknown"))
        w_opt = sol['x'].full().flatten()
        control_opt = w_opt[:self.n_segments*3].reshape((self.n_segments, 3))
        state_opt = w_opt[self.n_segments*3:].reshape((self.n_segments, 6))
        return solve_duration, control_opt, state_opt
    
    def optimize_time_to_target(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
        max_duration: float,
        max_attempts: int = 10,
    ):
        low = 0.0
        high = max_duration
        initial_guess = None
        best_result = self.optimize_trajectory(initial_state, target_state, high)
        initial_guess = self.solution_to_initial_guess(best_result[1], best_result[2])
        for _ in range(max_attempts):
            mid = 0.5 * (low + high)
            try:
                best_result = self.optimize_trajectory(
                    initial_state,
                    target_state,
                    mid,
                    initial_guess=initial_guess,
                )
                initial_guess = self.solution_to_initial_guess(best_result[1], best_result[2])
                high = mid
            except ValueError:
                low = mid
        return best_result
