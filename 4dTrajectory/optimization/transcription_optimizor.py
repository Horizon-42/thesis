import math
from pathlib import Path
import sys

from scipy.optimize import minimize
import numpy as np

_AERODYNAMIC_MODEL_DIR = Path(__file__).resolve().parents[2] / "aerodynamic_model"
if str(_AERODYNAMIC_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_AERODYNAMIC_MODEL_DIR))

from simulator import Simulator, Atmosphere, Control
from geodetic_simulator import GeodeticSimulator, GeodeticState

_MIN_SPEED_MPS = 1.0
_MIN_MASS_KG = 1.0
_MAX_ABS_GAMMA_RAD = math.pi / 2.0 - 1e-3
_DEFAULT_THRUST_GUESS_N = 12000.0
_MIN_ATTACK_RAD = -math.radians(18.0)
_MAX_ATTACK_RAD = math.radians(18.0)
_INVALID_DEFECT_MAGNITUDE = 1e9


class TranscriptionOptimizor:
    # multiple shooting args
    n_segments: int = 10

    # control dim
    control_dim: int = 3 # Match to control input, thrust, bank angle, attack angle
    
    # state dim
    state_dim : int = 7 # Match to GeodeticState, lat, lon, alt, V, psi, gamma, mass

    def __init__(self, sim_server: GeodeticSimulator, n_segments: int = 10, max_iterations: int = 1000, ftol: float = 1e-6):
        self.sim_server = sim_server
        self.n_segments = n_segments
        self.max_iterations = max_iterations
        self.ftol = ftol

    def unpack_z(self, z):
        z = np.asarray(z, dtype=float)

        final_time = z[0]
        node_control = z[1:1+self.n_segments*self.control_dim].reshape((self.n_segments, self.control_dim))
        node_state = z[1+self.n_segments*self.control_dim:].reshape((self.n_segments, self.state_dim))
        return final_time, node_control, node_state
    
    @staticmethod
    def geodetic_state_to_array(state: GeodeticState) -> np.ndarray:
        return np.array([state.latitude, state.longitude, state.altitude, state.V, state.psi, state.gamma, state.m])

    @staticmethod
    def validate_endpoint_state(state: GeodeticState, label: str) -> None:
        state_array = TranscriptionOptimizor.geodetic_state_to_array(state)
        if not np.all(np.isfinite(state_array)):
            raise ValueError(f"{label} must contain only finite values")
        if not -90.0 <= state.latitude <= 90.0:
            raise ValueError(f"{label}.latitude must be between -90.0 and 90.0")
        if not -180.0 <= state.longitude <= 180.0:
            raise ValueError(f"{label}.longitude must be between -180.0 and 180.0")
        if state.altitude < 0.0:
            raise ValueError(f"{label}.altitude must be >= 0.0")
        if state.V < _MIN_SPEED_MPS:
            raise ValueError(f"{label}.V must be >= {_MIN_SPEED_MPS}")
        if abs(state.gamma) >= _MAX_ABS_GAMMA_RAD:
            raise ValueError(f"{label}.gamma must stay away from +/- pi/2")
        if state.m < _MIN_MASS_KG:
            raise ValueError(f"{label}.m must be >= {_MIN_MASS_KG}")

    def build_control_guess(
        self,
        initial_state: GeodeticState,
        node_state_guess: np.ndarray,
    ) -> np.ndarray:
        # Controls are optimized per shooting segment, so their initial guess
        # should describe the control used while travelling from the segment
        # start to the next node. node_state_guess stores segment end points;
        # prepend the fixed initial state and drop the last node to get the
        # matching list of segment start states.
        start_states = np.vstack((
            self.geodetic_state_to_array(initial_state),
            node_state_guess[:-1],
        ))

        # The first SLSQP probe must be simulatable. A zero-alpha guess can make
        # low-speed/heavy-aircraft segments descend so fast that solve_ivp fails,
        # which turns the whole defect vector into the constant invalid residual.
        # A constant residual has almost no useful Jacobian information, and SLSQP
        # can report "Singular matrix C in LSQ subproblem" before it ever gets a
        # meaningful search direction. Estimating trim alpha per segment keeps the
        # initial probe in the differentiable part of the simulator.
        return np.array([
            [
                # Thrust and bank are still simple neutral guesses; SLSQP can
                # adjust them after it gets a valid first Jacobian.
                _DEFAULT_THRUST_GUESS_N,
                0.0,
                self.estimate_trim_attack_rad(start_state),
            ]
            for start_state in start_states
        ])

    def estimate_trim_attack_rad(self, state: np.ndarray) -> float:
        # This is not an autopilot law and it is not meant to satisfy the target
        # trajectory. It is only an initialization heuristic: choose an attack
        # angle that approximately balances the vertical-plane force equation at
        # one segment start. That gives the optimizer a numerically healthy point
        # from which it can change thrust, bank, alpha, node states, and time.
        simulator = getattr(self.sim_server, "simulator", None)
        atmosphere = getattr(self.sim_server, "atmosphere", None)
        if simulator is None or atmosphere is None:
            # Unit tests often pass a small fake simulator that only implements
            # step(). In that case keep the legacy neutral alpha guess because
            # there is no aerodynamic model to trim against.
            return 0.0

        altitude = float(state[2])
        speed = float(state[3])
        gamma = float(state[5])
        mass = float(state[6])
        if speed < _MIN_SPEED_MPS or mass < _MIN_MASS_KG:
            # Endpoint validation should normally reject these states before this
            # method runs. Keep this guard because the optimizer may reuse helpers
            # in tests or future callers with partially constructed guesses.
            return 0.0

        # Lift in the alpha model is q S CL(alpha), with q = 0.5 rho V^2.
        # Keeping the "q S" product separate makes the balance equation below read
        # like the dynamics: lift plus thrust-normal should offset weight normal
        # to the current flight path.
        dynamic_pressure_area = (
            0.5
            * atmosphere.get_ISA_density(altitude)
            * speed**2
            * simulator.S
        )
        required_normal_force = mass * simulator.g * math.cos(gamma)

        def vertical_balance_error(attack_rad: float) -> float:
            # Positive error means this alpha produces more normal force than a
            # locally trimmed segment needs; negative means it needs more alpha.
            lift_coefficient = simulator.CL0 + simulator.CL_alpha * attack_rad
            lift = dynamic_pressure_area * lift_coefficient
            thrust_normal = _DEFAULT_THRUST_GUESS_N * math.sin(attack_rad)
            return lift + thrust_normal - required_normal_force

        low_error = vertical_balance_error(_MIN_ATTACK_RAD)
        high_error = vertical_balance_error(_MAX_ATTACK_RAD)
        if low_error * high_error > 0.0:
            # Some target guesses are outside the simplified aircraft envelope.
            # For example, a heavy A320 near 70 m/s may need more than 18 degrees
            # alpha to trim. In that case use the closer bound instead of raising:
            # the goal is only to avoid an obviously bad initial point.
            return (
                _MIN_ATTACK_RAD
                if abs(low_error) < abs(high_error)
                else _MAX_ATTACK_RAD
            )

        # The balance is monotonic for this linear-CL model over the allowed alpha
        # interval, so bisection is deterministic and avoids pulling another
        # nonlinear solver into the optimizer initialization path.
        low = _MIN_ATTACK_RAD
        high = _MAX_ATTACK_RAD
        for _ in range(60):
            attack_rad = 0.5 * (low + high)
            error = vertical_balance_error(attack_rad)
            if error == 0.0:
                return attack_rad
            if low_error * error < 0.0:
                high = attack_rad
                high_error = error
            else:
                low = attack_rad
                low_error = error
        return 0.5 * (low + high)

    def build_state_guess(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
    ) -> np.ndarray:
        initial = self.geodetic_state_to_array(initial_state)
        target = self.geodetic_state_to_array(target_state)

        # Multiple shooting nodes are segment end points. A straight interpolation
        # keeps the first solver calls close to the requested trajectory instead
        # of feeding the aircraft model impossible all-zero speed/mass states.
        return np.vstack([
            initial + (target - initial) * ((index + 1) / self.n_segments)
            for index in range(self.n_segments)
        ])

    def build_state_bounds(self) -> list[tuple[float | None, float | None]]:
        # The point-mass dynamics divide by m, V, and cos(gamma). These bounds
        # keep SLSQP inside the simulatable flight envelope while still allowing
        # broad terminal-area trajectories.
        state_bounds = [
            (-90.0, 90.0),
            (-180.0, 180.0),
            (0.0, None),
            (_MIN_SPEED_MPS, None),
            (None, None),
            (-_MAX_ABS_GAMMA_RAD, _MAX_ABS_GAMMA_RAD),
            (_MIN_MASS_KG, None),
        ]
        return state_bounds * self.n_segments

    @staticmethod
    def is_simulatable_state_array(state: np.ndarray) -> bool:
        return (
            np.all(np.isfinite(state))
            and -90.0 <= state[0] <= 90.0
            and -180.0 <= state[1] <= 180.0
            and state[2] >= 0.0
            and state[3] >= _MIN_SPEED_MPS
            and abs(state[5]) < _MAX_ABS_GAMMA_RAD
            and state[6] >= _MIN_MASS_KG
        )

    def invalid_defects(self) -> np.ndarray:
        return np.full(
            self.n_segments * self.state_dim,
            _INVALID_DEFECT_MAGNITUDE,
        )

    def optimize_trajectory(self, initial_state: GeodeticState, target_state: GeodeticState) -> list:
        self.validate_endpoint_state(initial_state, "initial_state")
        self.validate_endpoint_state(target_state, "target_state")

        # initialize guess and bounds
        # final time guess
        final_time_guess = 100.0 # seconds, this is a placeholder and should be set based on the problem
        final_time_bounds = (1.0, 1000.0) # seconds

        # state guess
        node_state_guess = self.build_state_guess(initial_state, target_state)
        state_bounds = self.build_state_bounds()

        # control guess
        node_control_guess = self.build_control_guess(initial_state, node_state_guess)
        # control bounds, for example, thrust between 0 and max thrust 1000KN, bank angle between -90 and 90 degrees, attack angle between -18 and 18 degrees
        control_bounds = [(0.0, 1000000.0), (-math.pi/2, math.pi/2), (_MIN_ATTACK_RAD, _MAX_ATTACK_RAD)] * self.n_segments

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

                if (
                    not self.is_simulatable_state_array(start_state)
                    or not self.is_simulatable_state_array(state_i)
                    or not np.all(np.isfinite(control_i))
                ):
                    return self.invalid_defects()

                # SLSQP will still probe infeasible candidates. Convert simulator
                # failures into a large equality residual so one bad probe does
                # not abort the whole HTTP optimization request.
                try:
                    geo_predicted_state = self.sim_server.step(GeodeticState(*start_state), Control(*control_i), dt)
                except (ValueError, ZeroDivisionError, OverflowError):
                    return self.invalid_defects()
                predicted_state = self.geodetic_state_to_array(geo_predicted_state)
                if not self.is_simulatable_state_array(predicted_state):
                    return self.invalid_defects()

                defects.append(predicted_state - state_i)

                start_state = state_i # for next segment

            return np.concatenate(defects)
        
        def final_state_constraint(z):
            _, _, node_state = self.unpack_z(z)
            final_state = node_state[-1]
            target_state_array = self.geodetic_state_to_array(target_state)
            # Mass has no dynamics in the current point-mass model (dmdt = 0).
            # The defect constraints already force every node mass to match the
            # initial mass, so constraining terminal mass again makes SLSQP's
            # equality Jacobian rank-deficient.
            return final_state[:-1] - target_state_array[:-1]

        constraints = [{'type': 'eq', 'fun': defect_constraints},
                       {'type': 'eq', 'fun': final_state_constraint}]
        
        # perform optimization
        result = minimize(trajectory_objective, 
                          initial_guess, 
                          bounds=bounds, 
                          constraints=constraints, 
                          method='SLSQP', 
                          options={'maxiter': self.max_iterations, 'ftol': self.ftol})
        
        # return sates and controls at each node
        if result.success:
            final_time, node_control, node_state = self.unpack_z(result.x)
            return final_time, node_control, node_state

        else:
            raise ValueError("Optimization failed: " + result.message)
