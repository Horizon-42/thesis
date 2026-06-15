from pathlib import Path
import sys
import math

import numpy as np
from scipy.optimize import minimize

_AERODYNAMIC_MODEL_DIR = Path(__file__).resolve().parents[2] / "aerodynamic_model"
if str(_AERODYNAMIC_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_AERODYNAMIC_MODEL_DIR))

from simulator import Control
from geodetic_simulator import GeodeticSimulator, GeodeticState

_MIN_SPEED_MPS = 1.0
_MIN_MASS_KG = 1.0
_MAX_ABS_GAMMA_RAD = math.pi / 2.0 - 1e-3
_DEFAULT_THRUST_GUESS_N = 12000.0
_DEFAULT_ATTACK_GUESS_RAD = math.radians(4.0)
_MIN_ATTACK_RAD = -math.radians(18.0)
_MAX_ATTACK_RAD = math.radians(18.0)
_DEFAULT_DT_S = 0.2
_METRES_PER_DEGREE_LAT = 111_320.0
# SLSQP will probe controls that are physically impossible. Returning a large
# finite objective lets it reject those candidates without turning one bad probe
# into an HTTP 400 response.
_INFEASIBLE_OBJECTIVE_VALUE = 1e9

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
        if n_control_segments < 1:
            raise ValueError("n_control_segments must be >= 1")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        self.sim = sim
        self.n_control_segments = n_control_segments
        self.dt = dt
        self.max_iterations = max_iterations
        self.ftol = ftol
        aircraft = getattr(getattr(sim, "simulator", None), "aircraft", None)
        self.max_thrust_n = float(getattr(aircraft, "max_thrust_n", 1000000.0))
        self.default_thrust_guess_n = float(
            getattr(aircraft, "approach_thrust_guess_n", _DEFAULT_THRUST_GUESS_N)
        )
    
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
        if not math.isfinite(state.m):
            raise ValueError(f"{label}.m must be finite")
        if state.V < _MIN_SPEED_MPS:
            raise ValueError(f"{label} state has invalid speed {state.V:.2f} m/s, must be >= {_MIN_SPEED_MPS} m/s")
        if state.m < _MIN_MASS_KG:
            raise ValueError(f"{label}.m must be >= {_MIN_MASS_KG} kg")
        if abs(state.gamma) > _MAX_ABS_GAMMA_RAD:
            raise ValueError(f"{label} state has invalid flight path angle {math.degrees(state.gamma):.2f} deg, must be between {-math.degrees(_MAX_ABS_GAMMA_RAD):.2f} and {math.degrees(_MAX_ABS_GAMMA_RAD):.2f} deg")
        if not -90.0 <= state.latitude <= 90.0:
            raise ValueError(f"{label}.latitude must be between -90.0 and 90.0")
        if not -180.0 <= state.longitude <= 180.0:
            raise ValueError(f"{label}.longitude must be between -180.0 and 180.0")
        if state.altitude < 0.0:
            raise ValueError(f"{label}.altitude must be non-negative")

    @staticmethod
    def angular_difference_rad(left: float, right: float) -> float:
        # Heading angles wrap at +/-pi. A plain subtraction would say 179 deg and
        # -179 deg differ by 358 deg, even though physically they are only 2 deg
        # apart. Shift into [0, 2*pi), then back to [-pi, pi) so the optimizer
        # sees the shortest signed angular difference.
        return (left - right + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def position_constraint_error(
        state: GeodeticState,
        target_state: GeodeticState,
    ) -> np.ndarray:
        # Equality constraints only pin terminal position: lateral position in
        # kilometres, altitude in hundreds of metres. Scaling changes numerical
        # conditioning but not the required zero-error endpoint.
        mean_latitude = math.radians((state.latitude + target_state.latitude) * 0.5)
        metres_per_degree_lon = _METRES_PER_DEGREE_LAT * max(
            0.1,
            abs(math.cos(mean_latitude)),
        )
        return np.array([
            (state.latitude - target_state.latitude) * _METRES_PER_DEGREE_LAT / 1000.0,
            (state.longitude - target_state.longitude) * metres_per_degree_lon / 1000.0,
            (state.altitude - target_state.altitude) / 100.0,
        ])

    @staticmethod
    def terminal_soft_cost(
        state: GeodeticState,
        target_state: GeodeticState,
    ) -> float:
        speed_error = (state.V - target_state.V) / 10.0
        heading_error = (
            SingleShootingOptimizor.angular_difference_rad(
                state.psi,
                target_state.psi,
            )
            / math.radians(10.0)
        )
        flight_path_error = (state.gamma - target_state.gamma) / math.radians(2.0)
        return math.sqrt(
            speed_error**2 +
            heading_error**2 +
            flight_path_error**2
        )
    
    def segment_simulate(self, start_state: GeodeticState, control: Control, duration: float) -> GeodeticState:
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be positive and finite")
        
        n_substeps = max(1, math.ceil(duration / self.dt))
        substep_duration = duration / n_substeps
        state = start_state
        for _ in range(n_substeps):
            state = self.sim.step(state, control, substep_duration)
        return state

    def simulate_controls(
        self,
        initial_state: GeodeticState,
        controls: np.ndarray,
        final_time: float,
    ) -> GeodeticState:
        duration_per_segment = final_time / self.n_control_segments
        state = initial_state
        for control_values in controls:
            state = self.segment_simulate(
                state,
                Control(*control_values),
                duration_per_segment,
            )
        return state

    def terminal_objective(
        self,
        z: np.ndarray,
        initial_state: GeodeticState,
        target_state: GeodeticState,
    ) -> float:
        final_time, controls = self.unpack_z(z)
        try:
            final_state = self.simulate_controls(initial_state, controls, final_time)
        except (ValueError, ZeroDivisionError, OverflowError):
            return _INFEASIBLE_OBJECTIVE_VALUE

        return self.terminal_soft_cost(final_state, target_state) + 1e-4 * final_time

    def terminal_position_constraint(
        self,
        z: np.ndarray,
        initial_state: GeodeticState,
        target_state: GeodeticState,
    ) -> np.ndarray:
        final_time, controls = self.unpack_z(z)
        try:
            final_state = self.simulate_controls(initial_state, controls, final_time)
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.full(3, _INFEASIBLE_OBJECTIVE_VALUE)

        return self.position_constraint_error(final_state, target_state)

    def optimize_trajectory(self, initial_state: GeodeticState, target_state: GeodeticState) -> tuple[float, np.ndarray, None]:
        self.validate_endpoint_state(initial_state, "Initial")
        self.validate_endpoint_state(target_state, "Target")

        # initial guess and bounds
        final_time_bound = (1.0, 1000.0)
        mean_latitude_rad = math.radians(
            (initial_state.latitude + target_state.latitude) * 0.5,
        )
        metres_per_degree_lon = _METRES_PER_DEGREE_LAT * max(
            0.1,
            abs(math.cos(mean_latitude_rad)),
        )
        latitude_delta_m = (
            (initial_state.latitude - target_state.latitude)
            * _METRES_PER_DEGREE_LAT
        )
        longitude_delta_m = (
            (initial_state.longitude - target_state.longitude)
            * metres_per_degree_lon
        )
        altitude_delta_m = initial_state.altitude - target_state.altitude
        distance_m = math.sqrt(
            latitude_delta_m**2 +
            longitude_delta_m**2 +
            altitude_delta_m**2
        )
        average_speed_mps = max(initial_state.V, target_state.V, 50.0)
        final_time_guess = min(
            final_time_bound[1],
            max(
                final_time_bound[0],
                distance_m / average_speed_mps * 1.5,
            ),
        )

        # control guesses and bounds
        control_guesses = np.tile(
            np.array([self.default_thrust_guess_n, 0.0, _DEFAULT_ATTACK_GUESS_RAD]),
            (self.n_control_segments, 1),
        )
        control_bounds = [
            (0.0, self.max_thrust_n),
            (-math.pi / 2, math.pi / 2),
            (_MIN_ATTACK_RAD, _MAX_ATTACK_RAD),
        ] * self.n_control_segments

        initial_guess = np.hstack(([final_time_guess], control_guesses.flatten()))
        bounds = [final_time_bound] + control_bounds

        constraints = {
            'type': 'eq',
            'fun': lambda z: self.terminal_position_constraint(
                z,
                initial_state,
                target_state,
            ),
        }
        
        result = minimize(
            lambda z: self.terminal_objective(z, initial_state, target_state),
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.ftol},
        )

        if result.success:
            final_time, controls = self.unpack_z(result.x)
            return final_time, controls, None
        else:
            raise RuntimeError(
                f"Optimization failed: {result.message}. Final time guess was {final_time_guess:.2f} s. This may be due to an infeasible problem setup or insufficient iterations. Consider adjusting the initial guess, increasing max_iterations, or relaxing ftol. If the problem is expected to be feasible, this may indicate a bug in the simulator or optimization code.",
            )
