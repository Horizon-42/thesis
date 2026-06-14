import math
import numpy as np
from scipy.optimize import minimize

from pathlib import Path
import sys

from aerodynamic_model.simulator import Simulator, Atmosphere, Control
from aerodynamic_model.geodetic_simulator import GeodeticSimulator, GeodeticState

_MIN_SPEED_MPS = 1.0
_MAX_ABS_GAMMA_RAD = math.pi / 2.0 - 1e-3
_DEFAULT_THRUST_GUESS_N = 12000.0
_MIN_ATTACK_RAD = -math.radians(18.0)
_MAX_ATTACK_RAD = math.radians(18.0)
_DEFAULT_DT_S = 0.2
_INVALID_DEFECT_MAGNITUDE = 1e6

class SingleShootingOptimizor:
    
    # control args
    n_control_segments: int # number of control segments

    control_dim: int = 3 # number of control variables (thrust, bank angle, attack angle)

    # state dim
    state_dim: int = 6 # number of state variables (x, y, h, V, psi, gamma)

    # dt
    dt: float = _DEFAULT_DT_S # time step for simulation

    # simulator
    sim: GeodeticSimulator

    # optimization args
    max_iterations: int = 1000
    ftol: float = 1e-6

    def __init__(
        self,
        sim:GeodeticSimulator,
        n_control_segments: int,
        dt: float = _DEFAULT_DT_S,
        max_iterations: int = 1000,
        ftol: float = 1e-6,
    ):
        self.sim = sim
        self.n_control_segments = n_control_segments
        self.dt = dt
        self.max_iterations = max_iterations
        self.ftol = ftol
    
    def unpack_z(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        final_time = z[0]
        controls = z[1:].reshape((self.n_control_segments, self.control_dim))
        return final_time, controls
    
    @staticmethod
    def geodetic_state_to_array(state: GeodeticState) -> np.ndarray:
        return np.array([state.latitude, state.longitude, state.altitude, state.V, state.psi, state.gamma])
    
    @staticmethod
    def validate_endpoint_state(state: GeodeticState, label: str) -> None:
        state_array = SingleShootingOptimizor.geodetic_state_to_array(state)
        if not np.all(np.isfinite(state_array)):
            raise ValueError(f"{label} state contains non-finite values: {state_array}")
        if state.V < _MIN_SPEED_MPS:
            raise ValueError(f"{label} state has invalid speed {state.V:.2f} m/s, must be >= {_MIN_SPEED_MPS} m/s")
        if abs(state.gamma) > _MAX_ABS_GAMMA_RAD:
            raise ValueError(f"{label} state has invalid flight path angle {math.degrees(state.gamma):.2f} deg, must be between {-math.degrees(_MAX_ABS_GAMMA_RAD):.2f} and {math.degrees(_MAX_ABS_GAMMA_RAD):.2f} deg")
        if not -90.0 <= state.latitude <= 90.0:
            raise ValueError(f"{label}.latitude must be between -90.0 and 90.0")
        if not -180.0 <= state.longitude <= 180.0:
            raise ValueError(f"{label}.longitude must be between -180.0 and 180.0")
        if state.altitude < 0.0:
            raise ValueError(f"{label}.altitude must be non-negative")
    
    def segment_simulate(self, start_state: GeodeticState, control: Control, duration: float) -> GeodeticState:
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be positive and finite")
        
        n_substeps = max(1, math.ceil(duration / self.dt))
        substep_duration = duration / n_substeps
        state = start_state
        for _ in range(n_substeps):
            state = self.sim_server.step(state, control, substep_duration)
        return state


    def optimize_trajectory(self, initial_state: GeodeticState, target_state: GeodeticState) -> tuple[float, np.ndarray]:
        self.validate_endpoint_state(initial_state, "Initial")
        self.validate_endpoint_state(target_state, "Target")

        # initial guess and bounds
        final_time_guess = 100.0
        final_time_bound = (1.0, 1000.0) 

        # control guesses and bounds
        control_guesses = np.zeros((self.n_control_segments, self.control_dim))
        control_bounds = [(0.0, 1000000.0), (-math.pi/2, math.pi/2), (_MIN_ATTACK_RAD, _MAX_ATTACK_RAD)] * self.n_control_segments

        inital_guess = np.hstack(([final_time_guess], control_guesses.flatten()))
        bounds = [final_time_bound] + control_bounds

        # optimization
        def objective(z):
            final_time, controls = self.unpack_z(z)
            duration_per_segment = final_time / self.n_control_segments

            state = initial_state

            for i in range(self.n_control_segments):
                control = Control(*controls[i])
                state = self.segment_simulate(state, control, duration_per_segment)
            
            
            target_state_array = self.geodetic_state_to_array(target_state)
            final_state_array = self.geodetic_state_to_array(state)
            return np.linalg.norm(target_state_array - final_state_array)
        
        result = minimize(
            objective,
            inital_guess,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': self.ftol},
        )

        if result.success:
            final_time, controls = self.unpack_z(result.x)
            return final_time, controls, []
        else:
            raise RuntimeError(
                f"Optimization failed: {result.message}. Final time guess was {final_time_guess:.2f} s. This may be due to an infeasible problem setup or insufficient iterations. Consider adjusting the initial guess, increasing max_iterations, or relaxing ftol. If the problem is expected to be feasible, this may indicate a bug in the simulator or optimization code.",
            )
