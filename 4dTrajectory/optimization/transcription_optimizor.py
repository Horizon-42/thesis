import math
from scipy.optimize import minimize
import numpy as np

try:
    from ...aerodynamic_model.simulator import Simulator, Atmosphere, Control
    from ...aerodynamic_model.simulation_server import GeodeticState, SimulationServer
except ImportError:
    from aerodynamic_model.simulator import Simulator, Atmosphere, Control
    from aerodynamic_model.simulation_server import GeodeticState, SimulationServer

class TranscriptionOptimizor:
    # multiple shooting args
    n_segments: int = 10

    # control dim
    control_dim: int = 3 # Match to control input, thrust, bank angle, attack angle
    
    # state dim
    state_dim : int = 7 # Match to GeodeticState, lat, lon, alt, V, psi, gamma, mass

    def __init__(self, sim_server: SimulationServer, n_segments: int = 10):
        self.sim_server = sim_server
        self.n_segments = n_segments

    def unpack_z(self, z):
        z = np.asarray(z, dtype=float)

        final_time = z[0]
        node_control = z[1:1+self.n_segments*self.control_dim].reshape((self.n_segments, self.control_dim))
        node_state = z[1+self.n_segments*self.control_dim:].reshape((self.n_segments, self.state_dim))
        return final_time, node_control, node_state
    
    @staticmethod
    def geodetic_state_to_array(state: GeodeticState) -> np.ndarray:
        return np.array([state.lat, state.lon, state.alt, state.V, state.psi, state.gamma, state.mass])
        
    def optimize_trajectory(self, initial_state: GeodeticState, target_state: GeodeticState) -> list:
        # initialize guess and bounds
        # final time guess
        final_time_guess = 100.0 # seconds, this is a placeholder and should be set based on the problem
        final_time_bounds = (1.0, 1000.0) # seconds

        # control guess
        node_control_guess = np.zeros((self.n_segments, self.control_dim))
        # control bounds, for example, thrust between 0 and max thrust 1000KN, bank angle between -90 and 90 degrees, attack angle between -10 and 10 degrees
        control_bounds = [(0.0, 1000000.0), (-math.pi/2, math.pi/2), (-math.radians(18), math.radians(18))] * self.n_segments
        control_bounds = control_bounds * self.n_segments

        # state guess
        node_state_guess = np.zeros((self.n_segments, self.state_dim))
        # bounds for state set to None for now, can be set based on problem requirements
        state_bounds = [(None, None)] * self.state_dim * self.n_segments

        initial_guess = np.hstack((final_time_guess, node_control_guess.flatten(), node_state_guess.flatten()))
        bounds = [final_time_bounds] + control_bounds + state_bounds

        # objective function to minimize final time
        def trajectory_objective(z):
            final_time, _, _ = self.unpack_z(z)
            return final_time
        
        # defect constraints to ensure dynamics are satisfied at each segment
        def defect_constraints(z):
            final_time, node_control, node_state = self.unpack_z(z)
            dt = final_time / self.n_segments

            defects = []

            start_state = self.geodetic_state_to_array(initial_state)
            for i in range(self.n_segments):
                # get current state and control
                state_i = node_state[i]
                control_i = node_control[i]

                # simulate dynamics for one segment
                # this is a placeholder, you would need to implement the actual dynamics simulation based on your model
                # next_state = simulate_dynamics(state_i, control_i, dt)
                geo_predicted_state = self.sim_server.step(start_state, Control(*control_i), dt)
                predicted_state = self.geodetic_state_to_array(geo_predicted_state)

                defects.append(predicted_state - state_i)

                start_state = state_i # for next segment

            return np.concatenate(defects)
        
        def final_state_constraint(z):
            _, _, node_state = self.unpack_z(z)
            final_state = node_state[-1]
            target_state_array = self.geodetic_state_to_array(target_state)
            return final_state - target_state_array

        constraints = [{'type': 'eq', 'fun': defect_constraints},
                       {'type': 'eq', 'fun': final_state_constraint}]
        
        # perform optimization
        result = minimize(trajectory_objective, 
                          initial_guess, 
                          bounds=bounds, 
                          constraints=constraints, 
                          method='SLSQP', 
                          options={'maxiter': 1000, 'ftol': 1e-6})

        return []