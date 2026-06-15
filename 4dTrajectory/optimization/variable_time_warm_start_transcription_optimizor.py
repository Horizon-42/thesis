import math

import numpy as np
from scipy.optimize import least_squares, minimize

from transcription_optimizor import (
    Control,
    GeodeticState,
    TranscriptionOptimizor,
    _INVALID_DEFECT_MAGNITUDE,
    _MAX_ATTACK_RAD,
    _MIN_ATTACK_RAD,
    _MIN_MASS_KG,
)

# These weights only shape the least_squares warm-start residual. They do not
# change the final SLSQP constraints, which still enforce dynamics and terminal
# state as equalities.
_DEFECT_RESIDUAL_WEIGHT = 5.0
_TERMINAL_RESIDUAL_WEIGHT = 50.0
_REPLAY_STATE_RESIDUAL_WEIGHT = 5.0
_CONTROL_RESIDUAL_WEIGHT = 1e-3
_TIME_RESIDUAL_WEIGHT = 1.0
# Keep the warm start cheap enough for interactive frontend use. It is expected
# to improve the guess, not to fully solve the trajectory.
_WARM_START_MAX_NFEV = 2
_MIN_FINAL_TIME_S = 1.0
_MAX_FINAL_TIME_S = 1000.0
# Same idea as LeastSquaresTranscriptionOptimizor.build_variable_scale(): give
# finite-difference steps sensible units for lat/lon, altitude, speed, and
# angles.
_STATE_VARIABLE_SCALE = np.array([0.01, 0.01, 1000.0, 100.0, 1.0, 0.1])


class VariableTimeWarmStartTranscriptionOptimizor(TranscriptionOptimizor):
    """Variable-time warm start followed by constrained multiple shooting.

    The frontend arrival time is only the initial guess and soft reference. The
    optimized final time is part of z, then the final SLSQP phase enforces the
    multiple-shooting defects and target state as hard equality constraints.

    Variable layout:

        z = [final_time_s, controls for all segments, node states for all segments]

    This file intentionally stays separate from the fixed-time warm-start
    optimizer. The two classes are easier to test and compare when their z
    layouts are explicit instead of hidden behind mode flags.
    """

    def unpack_z(self, z):
        """Split the flat optimizer vector into physical blocks.

        Keeping final_time_s at z[0] makes it obvious which part of the vector
        changes only in the variable-time formulation. The remaining blocks
        match TranscriptionOptimizor: segment controls followed by segment end
        states.
        """
        z = np.asarray(z, dtype=float)

        final_time = float(z[0])
        control_start = 1
        control_end = control_start + self.n_segments * self.control_dim
        node_control = z[control_start:control_end].reshape((
            self.n_segments,
            self.control_dim,
        ))
        node_state = z[control_end:].reshape((self.n_segments, self.state_dim))
        return final_time, node_control, node_state

    def build_initial_guess(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
    ) -> np.ndarray:
        # Use the same straight-line state mesh and trim-inspired controls as
        # the other transcription optimizers. The only extra value is the
        # frontend arrival time, used here as the initial final_time guess.
        node_state_guess = self.build_state_guess(initial_state, target_state)
        node_control_guess = self.build_control_guess(initial_state, node_state_guess)
        return np.hstack((
            self.arrival_time_s,
            node_control_guess.flatten(),
            node_state_guess.flatten(),
        ))

    def build_final_time_bound(self) -> tuple[float, float]:
        # Keep the time search local to the user's guess. A much wider bound
        # gives SLSQP another large-scale variable to explore and made real
        # frontend-style cases noticeably slower.
        lower = max(_MIN_FINAL_TIME_S, self.arrival_time_s * 0.5)
        upper = min(_MAX_FINAL_TIME_S, self.arrival_time_s * 1.5)
        if lower >= upper:
            return _MIN_FINAL_TIME_S, _MAX_FINAL_TIME_S
        return lower, upper

    def build_variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # scipy.least_squares wants lower/upper arrays; scipy.minimize wants
        # tuple bounds. Build the arrays once here and convert back for SLSQP
        # later, so both phases use exactly the same box constraints.
        control_bounds = [
            (0.0, self.max_thrust_n),
            (-math.pi / 2.0, math.pi / 2.0),
            (_MIN_ATTACK_RAD, _MAX_ATTACK_RAD),
        ] * self.n_segments
        bounds = (
            [self.build_final_time_bound()]
            + control_bounds
            + self.build_state_bounds()
        )
        lower_bounds = np.array([low for low, _ in bounds], dtype=float)
        upper_bounds = np.array([high for _, high in bounds], dtype=float)
        return lower_bounds, upper_bounds

    def build_variable_scale(self) -> np.ndarray:
        # x_scale is not a physical constraint. It tells least_squares how large
        # a meaningful step is for each variable type, so final_time, thrust,
        # altitude, and angles do not compete on raw numeric magnitude.
        control_variable_scale = np.array([self.max_thrust_n, 1.0, 0.1])
        return np.hstack((
            np.array([max(self.arrival_time_s, 1.0)]),
            np.tile(control_variable_scale, self.n_segments),
            np.tile(_STATE_VARIABLE_SCALE, self.n_segments),
        ))

    def invalid_residuals(self) -> np.ndarray:
        # least_squares requires a fixed-length finite residual vector. When a
        # trial point is outside the simulator domain, return a large residual
        # with the same shape instead of raising and aborting the optimization.
        return np.full(
            self.n_segments * self.state_dim
            + self.state_dim
            + self.n_segments * self.state_dim
            + self.n_segments * self.control_dim
            + 1,
            _INVALID_DEFECT_MAGNITUDE,
        )

    @staticmethod
    def is_invalid_residuals(residuals: np.ndarray) -> bool:
        return np.all(residuals == _INVALID_DEFECT_MAGNITUDE)

    def defects_and_replay_residuals(
        self,
        z: np.ndarray,
        initial_state: GeodeticState,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute both node-chain defects and real replay residuals.

        The node-chain defect follows the multiple-shooting graph: propagate
        from the previous optimized node and compare against the next optimized
        node. The replay residual follows only controls from the real initial
        state. Penalizing both in the warm start helps avoid a visually plausible
        node mesh whose replayed controls still miss the path badly.
        """
        final_time, node_control, node_state = self.unpack_z(z)
        if not math.isfinite(final_time) or final_time <= 0.0:
            return None

        duration = final_time / self.n_segments
        mass = initial_state.m
        defects = []
        replay_residuals = []
        start_state = self.geodetic_state_to_array(initial_state)
        replay_state = initial_state

        for i in range(self.n_segments):
            state_i = node_state[i]
            control_i = node_control[i]
            if (
                not self.is_simulatable_state_array(start_state)
                or not self.is_simulatable_state_array(state_i)
                or not np.all(np.isfinite(control_i))
            ):
                return None

            try:
                control = Control(*control_i)
                predicted_geo_state = self.step_simulator(
                    self.array_to_geodetic_state(start_state, mass),
                    control,
                    duration,
                )
                # Replay is independent of the optimized node chain; it answers
                # "what trajectory do these controls actually fly from the real
                # initial state?"
                replay_state = self.step_simulator(
                    replay_state,
                    control,
                    duration,
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                return None

            predicted_state = self.geodetic_state_to_array(predicted_geo_state)
            replay_state_array = self.geodetic_state_to_array(replay_state)
            if (
                not self.is_simulatable_state_array(predicted_state)
                or not self.is_simulatable_state_array(replay_state_array)
            ):
                return None

            defects.append(self.state_constraint_error(predicted_state, state_i))
            replay_residuals.append(
                self.state_constraint_error(replay_state_array, state_i)
            )
            start_state = state_i

        return np.concatenate(defects), np.concatenate(replay_residuals)

    def trajectory_residuals(
        self,
        z: np.ndarray,
        initial_state: GeodeticState,
        target_state: GeodeticState,
    ) -> np.ndarray:
        """Soft residual vector for the least_squares warm-start phase.

        This residual intentionally mixes four ideas:

        - defects: make the optimized node chain obey the simulator;
        - terminal residual: pull the final node toward the target;
        - replay residual: make controls explain the trajectory from the real
          initial state;
        - control/time residuals: small regularization terms.

        The final SLSQP phase later turns defects and terminal state into hard
        equality constraints.
        """
        values = self.defects_and_replay_residuals(z, initial_state)
        if values is None:
            return self.invalid_residuals()

        final_time, node_control, node_state = self.unpack_z(z)
        defects, replay_residuals = values
        target_state_array = self.geodetic_state_to_array(target_state)

        terminal_residual = (
            _TERMINAL_RESIDUAL_WEIGHT
            * self.state_constraint_error(node_state[-1], target_state_array)
        )
        replay_residual = _REPLAY_STATE_RESIDUAL_WEIGHT * replay_residuals
        control_scales = np.array([self.max_thrust_n, math.pi / 2.0, _MAX_ATTACK_RAD])
        control_residual = _CONTROL_RESIDUAL_WEIGHT * (
            node_control / control_scales
        ).reshape(-1)
        time_residual = np.array([
            _TIME_RESIDUAL_WEIGHT
            * (final_time - self.arrival_time_s)
            / max(self.arrival_time_s, 1.0)
        ])

        return np.concatenate((
            _DEFECT_RESIDUAL_WEIGHT * defects,
            terminal_residual,
            replay_residual,
            control_residual,
            time_residual,
        ))

    def defect_constraints(
        self,
        z: np.ndarray,
        initial_state: GeodeticState,
    ) -> np.ndarray:
        """Hard dynamic equality constraints used by SLSQP."""
        values = self.defects_and_replay_residuals(z, initial_state)
        if values is None:
            return self.invalid_defects()
        defects, _ = values
        return defects

    def final_state_constraint(
        self,
        z: np.ndarray,
        target_state: GeodeticState,
    ) -> np.ndarray:
        """Hard terminal equality constraint used by SLSQP."""
        _, _, node_state = self.unpack_z(z)
        return self.state_constraint_error(
            node_state[-1],
            self.geodetic_state_to_array(target_state),
        )

    def trajectory_objective(self, z: np.ndarray) -> float:
        """Small objective for choosing among feasible SLSQP solutions.

        Dynamics and terminal accuracy are constraints. The objective keeps the
        variable final time near the frontend guess and mildly discourages
        extreme controls when more than one feasible point exists.
        """
        final_time, node_control, _ = self.unpack_z(z)
        control_scales = np.array([self.max_thrust_n, math.pi / 2.0, _MAX_ATTACK_RAD])
        control_effort = float(np.mean((node_control / control_scales) ** 2))
        time_error = (final_time - self.arrival_time_s) / max(self.arrival_time_s, 1.0)
        return float(time_error**2 + 1e-3 * control_effort)

    def optimize_trajectory(
        self,
        initial_state: GeodeticState,
        target_state: GeodeticState,
    ) -> list:
        self.validate_endpoint_state(initial_state, "initial_state")
        self.validate_endpoint_state(target_state, "target_state")
        if initial_state.m < _MIN_MASS_KG:
            raise ValueError(
                f"initial_state.m must be >= {_MIN_MASS_KG} kg "
                "to avoid singularities in the dynamics"
            )

        initial_guess = self.build_initial_guess(initial_state, target_state)
        lower_bounds, upper_bounds = self.build_variable_bounds()
        initial_residuals = self.trajectory_residuals(
            initial_guess,
            initial_state,
            target_state,
        )
        if self.is_invalid_residuals(initial_residuals):
            raise ValueError(
                "Variable-time warm-start initial trajectory guess is not simulatable; "
                "check target speed, target altitude, arrival time, and initial state"
            )

        # A non-success least_squares result is acceptable here if its best point
        # is finite and simulatable. With max_nfev=2, "maximum evaluations
        # exceeded" is a normal warm-start outcome, not necessarily a failure.
        warm_start = least_squares(
            lambda z: self.trajectory_residuals(z, initial_state, target_state),
            initial_guess,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=min(self.max_iterations, _WARM_START_MAX_NFEV),
            ftol=self.ftol,
            xtol=self.ftol,
            gtol=self.ftol,
            x_scale=self.build_variable_scale(),
        )
        warm_start_residuals = self.trajectory_residuals(
            warm_start.x,
            initial_state,
            target_state,
        )
        if self.is_invalid_residuals(warm_start_residuals):
            raise ValueError(
                "Variable-time warm-start least-squares failed: "
                + warm_start.message
            )

        # SLSQP receives the least_squares best point and enforces the actual
        # multiple-shooting problem. For this variable-time class, the first
        # optimization variable remains final_time_s all the way through SLSQP.
        constraints = [
            {
                "type": "eq",
                "fun": lambda z: self.defect_constraints(z, initial_state),
            },
            {
                "type": "eq",
                "fun": lambda z: self.final_state_constraint(z, target_state),
            },
        ]
        result = minimize(
            self.trajectory_objective,
            warm_start.x,
            bounds=list(zip(lower_bounds, upper_bounds)),
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.max_iterations, "ftol": self.ftol},
        )
        if result.success:
            return self.unpack_z(result.x)

        raise ValueError(
            "Variable-time warm-start transcription failed: " + result.message
        )
