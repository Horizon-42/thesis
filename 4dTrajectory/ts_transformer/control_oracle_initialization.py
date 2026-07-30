"""Future-aware initializers for the direct-control oracle.

These functions are diagnostic-only: they use the complete reference future to estimate
the controls implied by the point-mass equations.  Nothing here is imported by deployable
training or forecasting code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerodynamic_model.torch_dynamics import (
    GRAVITY_MPS2,
    ISA_DENSITY_EXPONENT,
    ISA_LAPSE_K_PER_M,
    ISA_RHO0_KG_M3,
    ISA_T0_K,
)


@dataclass(frozen=True)
class InverseDynamicsInitialization:
    controls: np.ndarray
    raw_control_min: np.ndarray
    raw_control_max: np.ndarray
    clipped_fraction: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_control_min": self.raw_control_min.tolist(),
            "raw_control_max": self.raw_control_max.tolist(),
            "clipped_fraction": self.clipped_fraction.tolist(),
            "initialized_control_min": self.controls.min(axis=0).tolist(),
            "initialized_control_max": self.controls.max(axis=0).tolist(),
        }


def refine_piecewise_constant_schedule(
    controls: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    target_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every source segment equally while preserving its exact trajectory."""
    source_controls = np.asarray(controls, dtype=np.float64)
    source_durations = np.asarray(segment_durations_s, dtype=np.float64)
    if source_controls.ndim != 2 or source_controls.shape[1] != 3:
        raise ValueError("controls must be [N,3]")
    if source_durations.shape != (len(source_controls),):
        raise ValueError("segment durations must align with controls")
    if not np.all(np.isfinite(source_controls)):
        raise ValueError("controls must be finite")
    if not np.all(np.isfinite(source_durations)) or not np.all(
        source_durations > 0.0
    ):
        raise ValueError("segment durations must be positive and finite")
    if target_segments <= len(source_controls):
        raise ValueError("target segment count must exceed the source count")
    factor, remainder = divmod(target_segments, len(source_controls))
    if remainder:
        raise ValueError("target segment count must be an integer source multiple")
    refined_controls = np.repeat(source_controls, factor, axis=0)
    refined_durations = np.repeat(source_durations / factor, factor)
    return refined_controls, refined_durations


def _drag_force(
    altitude_m: np.ndarray,
    speed_mps: np.ndarray,
    mass_kg: np.ndarray,
    load_factor: np.ndarray,
    aero_params: np.ndarray,
) -> np.ndarray:
    area, cl_max, cd0, induced_k, stall_threshold, k_stall = aero_params
    temperature = ISA_T0_K - ISA_LAPSE_K_PER_M * altitude_m
    density = ISA_RHO0_KG_M3 * (temperature / ISA_T0_K) ** ISA_DENSITY_EXPONENT
    cl_required = (
        load_factor
        * mass_kg
        * GRAVITY_MPS2
        / (0.5 * density * area * np.square(speed_mps))
    )
    ratio = cl_required / cl_max
    cl = np.minimum(cl_required, cl_max)
    stall_fraction = np.minimum(ratio, 1.0)
    transition = np.clip(
        (stall_fraction - stall_threshold) / (1.0 - stall_threshold), 0.0, 1.0
    )
    smooth = np.square(transition) * (3.0 - 2.0 * transition)
    stall_drag = np.where(ratio > stall_threshold, smooth * k_stall, 0.0)
    cd = cd0 + induced_k * np.square(cl) + stall_drag
    return 0.5 * density * np.square(speed_mps) * cd * area


def inverse_dynamics_controls(
    reference_states: np.ndarray,
    reference_times_s: np.ndarray,
    *,
    aero_params: np.ndarray,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
    n_segments: int,
    total_duration_s: float,
) -> InverseDynamicsInitialization:
    """Estimate one bounded control at each uniform segment midpoint.

    ``reference_states`` uses ``[lat, lon, alt, V, psi, gamma, mass]``.  The inverse
    equations are the algebraic inverse of ``torch_dynamics.enu_rhs`` outside stall; the
    final controls are clipped to the same aircraft-specific bounds as the learned head.
    """
    states = np.asarray(reference_states, dtype=np.float64)
    times = np.asarray(reference_times_s, dtype=np.float64)
    aero = np.asarray(aero_params, dtype=np.float64)
    lower = np.asarray(control_lower, dtype=np.float64)
    upper = np.asarray(control_upper, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError("reference_states must be [M,7]")
    if times.shape != (len(states),) or len(times) < 3:
        raise ValueError("reference_times_s must align with at least three states")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("reference times must be strictly increasing")
    if aero.shape != (6,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("aero parameters and control bounds have invalid shapes")
    if n_segments < 1 or total_duration_s <= 0.0:
        raise ValueError("n_segments and total_duration_s must be positive")

    altitude = states[:, 2]
    speed = np.maximum(states[:, 3], 1e-3)
    heading = np.unwrap(states[:, 4])
    gamma = states[:, 5]
    mass = states[:, 6]
    edge_order = 2 if len(times) >= 3 else 1
    speed_rate = np.gradient(speed, times, edge_order=edge_order)
    heading_rate = np.gradient(heading, times, edge_order=edge_order)
    gamma_rate = np.gradient(gamma, times, edge_order=edge_order)

    lateral = heading_rate * speed * np.cos(gamma) / GRAVITY_MPS2
    vertical = gamma_rate * speed / GRAVITY_MPS2 + np.cos(gamma)
    load_factor = np.hypot(lateral, vertical)
    bank = np.arctan2(lateral, vertical)
    drag = _drag_force(altitude, speed, mass, load_factor, aero)
    thrust = mass * (speed_rate + GRAVITY_MPS2 * np.sin(gamma)) + drag
    raw = np.column_stack((thrust, bank, load_factor))

    segment_midpoints = (
        np.arange(n_segments, dtype=np.float64) + 0.5
    ) * (total_duration_s / n_segments)
    sampled = np.column_stack(
        [
            np.interp(segment_midpoints, times, raw[:, channel])
            for channel in range(3)
        ]
    )
    clipped = np.clip(sampled, lower, upper)
    return InverseDynamicsInitialization(
        controls=clipped,
        raw_control_min=sampled.min(axis=0),
        raw_control_max=sampled.max(axis=0),
        clipped_fraction=np.mean((sampled < lower) | (sampled > upper), axis=0),
    )
