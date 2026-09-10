"""The per-flight physical context a rollout needs, read off a `FlightSeries` at an anchor.

Shared by every path that rolls the point-mass dynamics (the control strategy's forecast
and batch context, the plan path's guidance): the anchor state, the aircraft's aero row and
installed thrust, the frame, the runway course and glidepath, and the controls the observed
lookback implies are in effect at the anchor (the lagged model's actuator initial
condition). Moved out of `outputs/control/supervision.py` on 2026-09-10 (the membership
rule: a module belongs under a path only if every consumer is that path's).
"""

from __future__ import annotations

import numpy as np

from ts_transformer.data.channels import states_from_channels
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.geometry.final_approach_geometry import final_approach_arrays
from ts_transformer.outputs.conditioning import condition_vector
from ts_transformer.outputs.dynamics.inverse import actual_controls
from ts_transformer.outputs.envelope import CONTROL_LOWER, CONTROL_UPPER

# How much observed lookback the anchor-state control inversion differentiates. It needs
# at least three samples for a second-order gradient; a few more absorb ADS-B jitter
# without reaching back into a different phase of flight (11 x 2 s = 20 s).
ANCHOR_CONTROL_SAMPLES = 11


def anchor_controls(series: FlightSeries, anchor: int, mass_kg: float) -> np.ndarray:
    """Return the controls the observed lookback implies are in effect at ``anchor``.

    This is the lagged model's actuator initial condition. It reads only samples at or
    before the anchor, so it is as deployable as the history window itself, and it is the
    ACTUAL control (never a command) for every flight model — the commands that produced
    it are a separate inversion in ``control_inverse_dynamics``.
    """
    start = max(0, anchor + 1 - ANCHOR_CONTROL_SAMPLES)
    window = slice(start, anchor + 1)
    times = series.times[window]
    samples = states_from_channels(
        times, series.values[window], series.frame, mass_kg=mass_kg
    )
    states = np.asarray(
        [
            [s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
            for _t, s in samples
        ],
        dtype=np.float64,
    )
    aero = series.scenario.aero
    return actual_controls(
        states,
        times,
        aero_params=np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        max_thrust_n=float(series.scenario.aircraft.engine.max_thrust_total_n),
    )[-1]


def dynamics_arrays(series: FlightSeries, anchor: int) -> dict[str, np.ndarray]:
    """Physical per-flight tensors required by a control model and its rollout."""
    scenario = series.scenario
    mass_kg = float(scenario.initial.m)
    initial = states_from_channels(
        np.array([0.0], dtype=np.float64),
        series.values[anchor : anchor + 1],
        series.frame,
        mass_kg=mass_kg,
    )[0][1]
    aero = scenario.aero
    max_thrust = float(scenario.aircraft.engine.max_thrust_total_n)
    condition = condition_vector(mass_kg, max_thrust, aero)
    heading = float(getattr(series.frame, "heading_rad", 0.0))
    return {
        "condition": condition,
        "initial_state": np.array(
            [
                initial.latitude,
                initial.longitude,
                initial.altitude,
                initial.V,
                initial.psi,
                initial.gamma,
                initial.m,
            ],
            dtype=np.float64,
        ),
        "aero_params": np.array(
            [aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall],
            dtype=np.float64,
        ),
        "max_thrust_n": np.array(max_thrust, dtype=np.float64),
        # The dimensionless envelope is the same box on every airframe (see
        # control_envelope); the flight's installed thrust enters through max_thrust_n.
        "control_lower": CONTROL_LOWER.astype(np.float32),
        "control_upper": CONTROL_UPPER.astype(np.float32),
        # The lagged model's actuator initial condition. Emitted unconditionally so the
        # batch contract does not depend on the flight model, and clipped to the same box
        # the head predicts in — an anchor whose implied thrust is outside the envelope is
        # a starting point the model could not have commanded.
        "initial_controls": np.clip(
            anchor_controls(series, anchor, mass_kg), CONTROL_LOWER, CONTROL_UPPER
        ).astype(np.float64),
        "frame_params": np.array(
            [series.frame.lat0, series.frame.lon0, series.frame.alt0, heading],
            dtype=np.float64,
        ),
        # The rollout frame rotation and the runway heading coincide only for the
        # runway-aligned coordinate frame.  Keep the terminal-loss reference separate
        # so ENU rollouts are decomposed along/across the actual runway, not east/north.
        # Runway course and glidepath: one definition with the state path's
        # final-approach context (the procedure penalty on the rollout reads both). The
        # FAF distance is not carried: no control recipe gates at the FAF.
        **{
            key: value
            for key, value in final_approach_arrays(series).items()
            if key in ("runway_heading_rad", "glidepath_tan")
        },
    }
