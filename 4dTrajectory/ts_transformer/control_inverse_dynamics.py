"""Recover the controls a known trajectory was flown with, per dynamics model.

Two callers need this and they need it to mean the same thing:

1. the **teacher**, which is allowed to look at a whole outer-train future and solve for
   the control schedule that produced it, as the starting point for direct shooting;
2. the **lag model's initial condition**, which needs the controls already in effect at the
   anchor and may look only at the observed lookback.

Both are inversions of a forward model, and there is more than one forward model. The
failure this module exists to prevent is a teacher solved against equations the training
rollout does not use: the schedule then looks converged and reproduces nothing. So the
inverse is registered under the SAME key as its forward model
(:mod:`control_dynamics_backends`), and ``tests/test_control_inverse_dynamics.py`` closes
the loop numerically for every registered key — roll a known schedule forward, invert the
result, and require the original schedule back. A model added without an inverse fails at
lookup; a model whose inverse drifts fails that test.

Everything here is numpy and future-aware. Nothing in it is imported by the model forward
pass, and the anchor-only use (2) passes a lookback window, never a future.
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
from config import (
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    TSConfig,
)
from control_envelope import CONTROL_NAMES, fraction_controls
from geokit import WGS84_A, WGS84_E2

# The command inversion differentiates a control signal that is itself built from second
# derivatives of position, so it is smoothed first over this many samples (centred, and
# shrinking at the edges). Without it a 2 s ADS-B grid turns reception jitter into tens of
# degrees of commanded bank. Stated rather than silent: it is an explicit smoothing of the
# TEACHER's command estimate, not of any measured quantity.
COMMAND_SMOOTHING_SAMPLES = 5

STATE_COLUMNS = ("latitude", "longitude", "altitude", "V", "psi", "gamma", "mass")


@dataclass(frozen=True)
class InvertedControls:
    """Controls in the dimensionless envelope, plus what the bounds had to clip."""

    controls: np.ndarray             # [M, 3], unit-box contract, unclipped values kept
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


def _drag_force(
    altitude_m: np.ndarray,
    speed_mps: np.ndarray,
    mass_kg: np.ndarray,
    load_factor: np.ndarray,
    aero_params: np.ndarray,
) -> np.ndarray:
    """Numpy twin of ``torch_dynamics.aerodynamic_coefficients`` drag, stall included."""
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
    transition = np.clip(
        (np.minimum(ratio, 1.0) - stall_threshold) / (1.0 - stall_threshold), 0.0, 1.0
    )
    stall_drag = np.where(
        ratio > stall_threshold, np.square(transition) * (3.0 - 2.0 * transition) * k_stall, 0.0
    )
    cd = cd0 + induced_k * np.square(cl) + stall_drag
    return 0.5 * density * np.square(speed_mps) * cd * area


def _transport_rate(
    states: np.ndarray, frame_params: np.ndarray | None
) -> np.ndarray:
    """Return ``omega_transport x v`` in local ENU, or zero when the frame is absent.

    ``transport_chart_rhs`` integrates ``v_dot = a_force - omega x v``, so recovering the
    force from a measured ``v_dot`` has to add this back. It is small — of order ``v^2/R``,
    ~1.6e-3 m/s^2 at 100 m/s, i.e. 1.6e-4 g — but including it is what makes this the exact
    algebraic inverse of the chart RHS rather than a close one.
    """
    latitude, _lon, altitude, speed, psi, gamma, _mass = states.T
    if frame_params is None:
        return np.zeros((len(states), 3), dtype=np.float64)
    cos_gamma = np.cos(gamma)
    velocity = np.column_stack(
        (
            speed * cos_gamma * np.cos(psi),
            speed * cos_gamma * np.sin(psi),
            speed * np.sin(gamma),
        )
    )
    lat_rad = np.deg2rad(latitude)
    denominator = 1.0 - WGS84_E2 * np.square(np.sin(lat_rad))
    radius_n = WGS84_A / np.sqrt(denominator)
    radius_m = WGS84_A * (1.0 - WGS84_E2) / denominator**1.5
    omega = np.column_stack(
        (
            -velocity[:, 1] / (radius_m + altitude),
            velocity[:, 0] / (radius_n + altitude),
            velocity[:, 0] * np.tan(lat_rad) / (radius_n + altitude),
        )
    )
    return np.cross(omega, velocity)


def actual_controls(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    aero_params: np.ndarray,
    max_thrust_n: float,
    frame_params: np.ndarray | None = None,
) -> np.ndarray:
    """Return the ``[M,3]`` control the aircraft was flying at each reference sample.

    ``states`` is ``[M,7] = (lat, lon, alt, V, psi, gamma, mass)``. The result is in the
    dimensionless envelope contract and is NOT clipped — clipping is a bound decision the
    caller makes and reports, not something an inversion may do silently.

    These are the ACTUAL controls: for the lagged model they are the actuator states, and
    the commands that produced them are :func:`commanded_controls`.
    """
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(STATE_COLUMNS):
        raise ValueError(f"states must be [M,{len(STATE_COLUMNS)}]")
    if times.shape != (len(states),) or len(times) < 3:
        raise ValueError("times must align with at least three states")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("reference times must be strictly increasing")
    if np.asarray(aero_params).shape != (6,):
        raise ValueError("aero parameters must contain six values")

    altitude = states[:, 2]
    speed = np.maximum(states[:, 3], 1e-3)
    heading = np.unwrap(states[:, 4])
    gamma = states[:, 5]
    mass = states[:, 6]
    cos_gamma = np.cos(gamma)
    speed_rate = np.gradient(speed, times, edge_order=2)
    heading_rate = np.gradient(heading, times, edge_order=2)
    gamma_rate = np.gradient(gamma, times, edge_order=2)

    # Body-frame components of the specific force, exactly as the forward RHS composes it:
    # tangent -> speed rate, gamma-normal -> g(n cos mu - cos gamma), heading-normal ->
    # g n sin mu. The transport term is expressed in the same basis before inverting.
    transport = _transport_rate(states, frame_params)
    tangent = np.column_stack(
        (cos_gamma * np.cos(heading), cos_gamma * np.sin(heading), np.sin(gamma))
    )
    gamma_normal = np.column_stack(
        (
            -np.sin(gamma) * np.cos(heading),
            -np.sin(gamma) * np.sin(heading),
            cos_gamma,
        )
    )
    heading_normal = np.column_stack(
        (-np.sin(heading), np.cos(heading), np.zeros_like(heading))
    )
    tangential = speed_rate + np.einsum("ij,ij->i", transport, tangent)
    vertical = (
        (gamma_rate * speed + np.einsum("ij,ij->i", transport, gamma_normal))
        / GRAVITY_MPS2
        + cos_gamma
    )
    lateral = (
        heading_rate * speed * cos_gamma
        + np.einsum("ij,ij->i", transport, heading_normal)
    ) / GRAVITY_MPS2

    load_factor = np.hypot(lateral, vertical)
    bank = np.arctan2(lateral, vertical)
    drag = _drag_force(altitude, speed, mass, load_factor, np.asarray(aero_params))
    thrust_n = mass * (tangential + GRAVITY_MPS2 * np.sin(gamma)) + drag
    return fraction_controls(
        np.column_stack((thrust_n, bank, load_factor)),
        np.asarray(float(max_thrust_n)),
    )


def _smoothed(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that shrinks at the edges instead of padding them."""
    if window <= 1:
        return values
    half = window // 2
    out = np.empty_like(values)
    for index in range(len(values)):
        low = max(0, index - half)
        high = min(len(values), index + half + 1)
        out[index] = values[low:high].mean(axis=0)
    return out


def commanded_controls(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    aero_params: np.ndarray,
    max_thrust_n: float,
    time_constants_s: np.ndarray,
    frame_params: np.ndarray | None = None,
) -> np.ndarray:
    """Invert the first-order lag: ``u_cmd = u + tau * du/dt``.

    Exact for the continuous system, so a command schedule recovered from a trajectory the
    lagged model produced reproduces that trajectory. The derivative is taken on a smoothed
    copy (:data:`COMMAND_SMOOTHING_SAMPLES`) because ``u`` already carries two numerical
    differentiations of position.
    """
    actual = actual_controls(
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        frame_params=frame_params,
    )
    tau = np.asarray(time_constants_s, dtype=np.float64).reshape(-1)
    if tau.shape != (len(CONTROL_NAMES),) or not np.all(tau > 0.0):
        raise ValueError("time constants must be three positive seconds")
    times = np.asarray(times_s, dtype=np.float64)
    smoothed = _smoothed(actual, COMMAND_SMOOTHING_SAMPLES)
    rate = np.gradient(smoothed, times, axis=0, edge_order=2)
    return actual + tau * rate


def _point_mass_inverse(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    aero_params: np.ndarray,
    max_thrust_n: float,
    frame_params: np.ndarray | None,
    config: TSConfig,
) -> np.ndarray:
    del config
    return actual_controls(
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        frame_params=frame_params,
    )


def _first_order_lag_inverse(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    aero_params: np.ndarray,
    max_thrust_n: float,
    frame_params: np.ndarray | None,
    config: TSConfig,
) -> np.ndarray:
    return commanded_controls(
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        time_constants_s=config.control_time_constants_s,
        frame_params=frame_params,
    )


CONTROL_INVERSES = {
    CONTROL_DYNAMICS_POINT_MASS: _point_mass_inverse,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG: _first_order_lag_inverse,
}


def reference_controls(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    config: TSConfig,
    aero_params: np.ndarray,
    max_thrust_n: float,
    frame_params: np.ndarray | None = None,
) -> np.ndarray:
    """Return the ``[M,3]`` control schedule the CONFIGURED forward model would need."""
    return CONTROL_INVERSES[config.control_dynamics_model](
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        frame_params=frame_params,
        config=config,
    )


def segment_controls(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    config: TSConfig,
    aero_params: np.ndarray,
    max_thrust_n: float,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
    n_segments: int,
    total_duration_s: float,
    frame_params: np.ndarray | None = None,
) -> InvertedControls:
    """Sample the inverted schedule at uniform segment midpoints and clip to the box."""
    if n_segments < 1 or total_duration_s <= 0.0:
        raise ValueError("n_segments and total_duration_s must be positive")
    lower = np.asarray(control_lower, dtype=np.float64)
    upper = np.asarray(control_upper, dtype=np.float64)
    if lower.shape != (len(CONTROL_NAMES),) or upper.shape != lower.shape:
        raise ValueError("control bounds must each contain three values")

    raw = reference_controls(
        states,
        times_s,
        config=config,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        frame_params=frame_params,
    )
    midpoints = (np.arange(n_segments, dtype=np.float64) + 0.5) * (
        total_duration_s / n_segments
    )
    times = np.asarray(times_s, dtype=np.float64)
    sampled = np.column_stack(
        [np.interp(midpoints, times, raw[:, channel]) for channel in range(raw.shape[1])]
    )
    return InvertedControls(
        controls=np.clip(sampled, lower, upper),
        raw_control_min=sampled.min(axis=0),
        raw_control_max=sampled.max(axis=0),
        clipped_fraction=np.mean((sampled < lower) | (sampled > upper), axis=0),
    )


def refine_piecewise_constant_schedule(
    controls: np.ndarray,
    segment_durations_s: np.ndarray,
    *,
    target_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split every source segment equally while preserving its exact trajectory."""
    source_controls = np.asarray(controls, dtype=np.float64)
    source_durations = np.asarray(segment_durations_s, dtype=np.float64)
    if source_controls.ndim != 2 or source_controls.shape[1] != len(CONTROL_NAMES):
        raise ValueError("controls must be [N,3]")
    if source_durations.shape != (len(source_controls),):
        raise ValueError("segment durations must align with controls")
    if not np.all(np.isfinite(source_controls)):
        raise ValueError("controls must be finite")
    if not np.all(np.isfinite(source_durations)) or not np.all(source_durations > 0.0):
        raise ValueError("segment durations must be positive and finite")
    if target_segments <= len(source_controls):
        raise ValueError("target segment count must exceed the source count")
    factor, remainder = divmod(target_segments, len(source_controls))
    if remainder:
        raise ValueError("target segment count must be an integer source multiple")
    return (
        np.repeat(source_controls, factor, axis=0),
        np.repeat(source_durations / factor, factor),
    )
