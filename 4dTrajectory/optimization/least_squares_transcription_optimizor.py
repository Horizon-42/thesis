import math

import numpy as np
from scipy.optimize import least_squares

from transcription_optimizor import (
    Control,
    GeodeticState,
    TranscriptionOptimizor,
    _INVALID_DEFECT_MAGNITUDE,
    _MAX_ATTACK_RAD,
    _MAX_THRUST_N,
    _MIN_ATTACK_RAD,
)

_DEFECT_RESIDUAL_WEIGHT = 5.0
_TERMINAL_RESIDUAL_WEIGHT = 50.0
_CONTROL_RESIDUAL_WEIGHT = 1e-3
_CONTROL_VARIABLE_SCALE = np.array([100000.0, 1.0, 0.1])
_STATE_VARIABLE_SCALE = np.array([0.01, 0.01, 1000.0, 100.0, 1.0, 0.1])


class LeastSquaresTranscriptionOptimizor(TranscriptionOptimizor):
    """Multiple-shooting transcription solved as nonlinear least squares.

    The parent TranscriptionOptimizor formulates the same mesh as constrained
    SLSQP: defects and terminal errors are hard equality constraints. This
    variant keeps the variable layout, bounds, guesses, and simulator stepping,
    but turns those equalities into residuals minimized by scipy.least_squares.
    That makes it a smaller algorithm swap while preserving the frontend/backend
    response shape.
    """

    def invalid_residuals(self) -> np.ndarray:
        # least_squares expects a finite residual vector even when it probes an
        # impossible state/control candidate. A large constant residual marks the
        # probe as bad without letting simulator exceptions abort the solve.
        return np.full(
            self.n_segments * self.state_dim
            + self.state_dim
            + self.n_segments * self.control_dim,
            _INVALID_DEFECT_MAGNITUDE,
        )

    def build_variable_scale(self) -> np.ndarray:
        # least_squares uses this scale for trust-region steps and numerical
        # differentiation. Without it, thrust, altitude, lat/lon, and angles live
        # on wildly different magnitudes, and the solver can stop after moving
        # node states while barely changing the controls that replay uses.
        return np.hstack((
            np.tile(_CONTROL_VARIABLE_SCALE, self.n_segments),
            np.tile(_STATE_VARIABLE_SCALE, self.n_segments),
        ))

    def trajectory_residuals(
        self,
        z: np.ndarray,
        initial_state,
        target_state,
    ) -> np.ndarray:
        # z contains only controls and node states. Arrival time is fixed by the
        # request and returned by unpack_z for API compatibility with the SLSQP
        # optimizer and backend response.
        final_time, node_control, node_state = self.unpack_z(z)
        duration = final_time / self.n_segments
        mass = initial_state.m

        defects = []
        start_state = self.geodetic_state_to_array(initial_state)
        for i in range(self.n_segments):
            state_i = node_state[i]
            control_i = node_control[i]

            # Keep finite-difference probes inside the simulator domain. If one
            # row is invalid, return a full-size residual so SciPy never sees a
            # changing residual dimension.
            if (
                not self.is_simulatable_state_array(start_state)
                or not self.is_simulatable_state_array(state_i)
                or not np.all(np.isfinite(control_i))
            ):
                return self.invalid_residuals()

            try:
                # Multiple shooting defect: propagate from the previous node
                # with this segment's constant control, then compare the
                # propagated state with the optimized next node.
                predicted_geo_state = self.step_simulator(
                    self.array_to_geodetic_state(start_state, mass),
                    Control(*control_i),
                    duration,
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                return self.invalid_residuals()

            predicted_state = self.geodetic_state_to_array(predicted_geo_state)
            if not self.is_simulatable_state_array(predicted_state):
                return self.invalid_residuals()

            defects.append(self.state_constraint_error(predicted_state, state_i))
            start_state = state_i

        # Terminal residual keeps the last optimized node close to the target.
        # Unlike SLSQP equality constraints, this is still a soft objective term.
        # Weight it above per-segment defects so least_squares does not prefer a
        # dynamically smooth trajectory that misses the runway altitude badly.
        target_state_array = self.geodetic_state_to_array(target_state)
        terminal_residual = (
            _TERMINAL_RESIDUAL_WEIGHT
            * self.state_constraint_error(node_state[-1], target_state_array)
        )

        # Tiny regularization discourages needlessly extreme controls but stays
        # small enough that dynamics defects and terminal accuracy dominate.
        control_scales = np.array([_MAX_THRUST_N, math.pi / 2.0, _MAX_ATTACK_RAD])
        control_residual = _CONTROL_RESIDUAL_WEIGHT * (
            node_control / control_scales
        ).reshape(-1)

        return np.concatenate((
            _DEFECT_RESIDUAL_WEIGHT * np.concatenate(defects),
            terminal_residual,
            control_residual,
        ))

    def optimize_trajectory(self, initial_state, target_state) -> list:
        self.validate_endpoint_state(initial_state, "initial_state")
        self.validate_endpoint_state(target_state, "target_state")

        # Reuse the SLSQP transcription initial mesh: linearly interpolated node
        # states plus trim-inspired control guesses. This keeps the two solvers
        # comparable and avoids a separate initialization path.
        node_state_guess = self.build_state_guess(initial_state, target_state)
        node_control_guess = self.build_control_guess(initial_state, node_state_guess)

        # least_squares uses simple variable bounds instead of constraint
        # dictionaries. The variable order is [all controls, all node states],
        # matching TranscriptionOptimizor.unpack_z().
        control_bounds = [
            (0.0, _MAX_THRUST_N),
            (-math.pi / 2.0, math.pi / 2.0),
            (_MIN_ATTACK_RAD, _MAX_ATTACK_RAD),
        ] * self.n_segments
        bounds = control_bounds + self.build_state_bounds()
        lower_bounds = np.array([low for low, _ in bounds], dtype=float)
        upper_bounds = np.array([high for _, high in bounds], dtype=float)
        initial_guess = np.hstack((
            node_control_guess.flatten(),
            node_state_guess.flatten(),
        ))

        # The residual function already contains defects, terminal error, and
        # control regularization. There is no separate objective function or hard
        # equality constraint in this formulation.
        result = least_squares(
            lambda z: self.trajectory_residuals(z, initial_state, target_state),
            initial_guess,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=self.max_iterations,
            ftol=self.ftol,
            xtol=self.ftol,
            gtol=self.ftol,
            x_scale=self.build_variable_scale(),
        )

        if result.success:
            return self.unpack_z(result.x)

        raise ValueError("Optimization failed: " + result.message)
