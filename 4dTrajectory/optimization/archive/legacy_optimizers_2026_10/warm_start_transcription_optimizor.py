import math

import numpy as np
from scipy.optimize import least_squares, minimize

from least_squares_transcription_optimizor import LeastSquaresTranscriptionOptimizor
from transcription_optimizor import (
    Control,
    GeodeticState,
    _MAX_ATTACK_RAD,
    _MIN_ATTACK_RAD,
    _MIN_MASS_KG,
)

# Warm start is intentionally shallow. The goal is to nudge the initial mesh
# away from the straight-line guess, not to let least_squares solve the whole
# trajectory before SLSQP starts.
_WARM_START_MAX_NFEV = 2


class WarmStartTranscriptionOptimizor(LeastSquaresTranscriptionOptimizor):
    """Least-squares warm start followed by constrained multiple shooting.

    This optimizer keeps the same variable layout as the fixed-time
    LeastSquaresTranscriptionOptimizor:

        z = [controls for all segments, node states for all segments]

    The difference is the solve sequence. First, a very small least_squares
    pass improves the initial controls and node states. Then SLSQP solves the
    real multiple-shooting problem with hard equality constraints:

    - every segment end node must match simulator propagation from the previous
      node;
    - the final node must match the target state.

    Arrival time stays fixed at the frontend value, so this is the cheaper
    comparison point for the variable-time warm-start optimizer.
    """

    def defect_constraints(
        self,
        z: np.ndarray,
        initial_state: GeodeticState,
    ) -> np.ndarray:
        """Return hard multiple-shooting defect constraints for SLSQP.

        least_squares uses soft residuals and can tolerate imperfect defects.
        SLSQP needs equality constraints, so this method rebuilds only the
        dynamic defects that must become exactly zero in the final solve.
        """
        final_time, node_control, node_state = self.unpack_z(z)
        duration = final_time / self.n_segments
        mass = initial_state.m
        defects = []

        # node_state contains segment end points. The first segment starts from
        # the fixed aircraft state supplied by the frontend; later segments
        # start from the previous optimized node.
        start_state = self.geodetic_state_to_array(initial_state)

        for i in range(self.n_segments):
            state_i = node_state[i]
            control_i = node_control[i]
            if (
                not self.is_simulatable_state_array(start_state)
                or not self.is_simulatable_state_array(state_i)
                or not np.all(np.isfinite(control_i))
            ):
                # SLSQP may probe outside the simulator domain while estimating
                # finite-difference Jacobians. Returning a full-size large
                # residual keeps the constraint shape stable.
                return self.invalid_defects()

            try:
                predicted_geo_state = self.step_simulator(
                    self.array_to_geodetic_state(start_state, mass),
                    Control(*control_i),
                    duration,
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                return self.invalid_defects()

            predicted_state = self.geodetic_state_to_array(predicted_geo_state)
            if not self.is_simulatable_state_array(predicted_state):
                return self.invalid_defects()

            # The equality row is scaled by state_constraint_error, so latitude,
            # longitude, altitude, speed, heading, and gamma live on comparable
            # numeric magnitudes for SLSQP.
            defects.append(self.state_constraint_error(predicted_state, state_i))
            start_state = state_i

        return np.concatenate(defects)

    def final_state_constraint(
        self,
        z: np.ndarray,
        target_state: GeodeticState,
    ) -> np.ndarray:
        _, _, node_state = self.unpack_z(z)
        return self.state_constraint_error(
            node_state[-1],
            self.geodetic_state_to_array(target_state),
        )

    def trajectory_objective(self, z: np.ndarray) -> float:
        """Small regularization objective used after constraints carry physics.

        The actual trajectory requirements are enforced by equality constraints.
        The objective only discourages unnecessarily large controls among
        feasible solutions.
        """
        _, node_control, _ = self.unpack_z(z)
        control_scales = np.array([self.max_thrust_n, math.pi / 2.0, _MAX_ATTACK_RAD])
        return float(np.mean((node_control / control_scales) ** 2))

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

        node_state_guess = self.build_state_guess(initial_state, target_state)
        node_control_guess = self.build_control_guess(initial_state, node_state_guess)
        initial_guess = np.hstack((
            node_control_guess.flatten(),
            node_state_guess.flatten(),
        ))

        control_bounds = [
            (0.0, self.max_thrust_n),
            (-math.pi / 2.0, math.pi / 2.0),
            (_MIN_ATTACK_RAD, _MAX_ATTACK_RAD),
        ] * self.n_segments
        bounds = control_bounds + self.build_state_bounds()
        lower_bounds = np.array([low for low, _ in bounds], dtype=float)
        upper_bounds = np.array([high for _, high in bounds], dtype=float)

        initial_residuals = self.trajectory_residuals(
            initial_guess,
            initial_state,
            target_state,
        )
        if self.is_invalid_residuals(initial_residuals):
            raise ValueError(
                "Warm-start initial trajectory guess is not simulatable; "
                "check target speed, target altitude, arrival time, and initial state"
            )

        # This pass is deliberately limited by _WARM_START_MAX_NFEV. A
        # non-converged result is still useful if it produces a finite,
        # simulatable point for SLSQP.
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
            raise ValueError("Warm-start least-squares failed: " + warm_start.message)

        # The second phase is the real optimizer: hard defects plus hard terminal
        # state, initialized from the best point least_squares produced.
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
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": self.max_iterations, "ftol": self.ftol},
        )
        if result.success:
            return self.unpack_z(result.x)

        raise ValueError("Warm-start transcription failed: " + result.message)
