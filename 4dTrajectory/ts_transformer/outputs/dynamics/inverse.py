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
from ts_transformer.config import (
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    TSConfig,
)
from ts_transformer.outputs.envelope import CONTROL_NAMES, InverseKinematics, control_contract
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

    controls: np.ndarray             # [M, 3], the contract's units, unclipped values kept
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


#: The fewest observed states an inversion differentiates (a second-order difference).
MINIMUM_INVERSE_STATES = 3


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


def _transport_rate(states: np.ndarray) -> np.ndarray:
    """Return ``omega_transport x v`` in local ENU. Never optional.

    The inverse recovers a force from a ``v_dot`` expressed in a LOCAL ENU frame, and a
    trajectory that follows a curved earth carries ``omega x v`` in that frame whatever the
    forward integrator does internally. So this is added back for every backend — including
    ``reanchored-rk4``, which re-anchors into geodetic state each substep instead of writing
    a transport term explicitly, and whose rolled trajectories nonetheless invert 50x more
    accurately with it than without (``test_the_transport_term_is_required_by_every_backend``).

    It is small — of order ``v^2/R``, ~1.6e-3 m/s^2 at 100 m/s, i.e. 1.6e-4 g, and 0.03 % of
    the inverted bank RMS on KRDU — but including it is what makes this the exact algebraic
    inverse rather than a close one.

    It depends only on the STATE (latitude, altitude, velocity), never on the frame origin.
    It used to be gated on ``frame_params is not None``, whose values were never read: the
    argument was a flag wearing a data parameter's clothes, and every caller that forgot it
    silently got a different quantity — which is what happened to the scoring scripts while
    the training target was correct. There is no correct "off" setting, so there is no
    longer a way to ask for one.
    """
    latitude, _lon, altitude, speed, psi, gamma, _mass = states.T
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
    parameterization: str,
) -> np.ndarray:
    """Return the ``[M,3]`` control the aircraft was flying at each reference sample.

    ``states`` is ``[M,7] = (lat, lon, alt, V, psi, gamma, mass)``. The inversion is the same
    for every contract; the contract's teacher (``outputs/envelope.py``) writes its columns
    from it. The result is in the contract ``parameterization`` names (the one exception, the
    speed command's ABSOLUTE target, is below) and is NOT
    clipped — clipping is a bound decision the caller makes and reports, not something an
    inversion may do silently. REQUIRED, never defaulted: the two longitudinal columns
    differ by a factor of ~3 and an offset of ~0.08, and a teacher inverted in the wrong
    one is a bounded, plausible, silently wrong target.

    Under ``specific-force`` the first column is ``(T - D)/W = tangential/g + sin(gamma)``,
    read off the kinematics alone — the drag is computed for every contract, but only the
    thrust-fraction teacher reads it (or ``max_thrust_n``), which is the parameterisation's point:
    the tracks identify ``(T - D)/m``, never T and m apart.

    Under ``specific-force+path-angle`` the first column is that same specific force and the
    THIRD is the path-angle target ``γ + τ_γ·γ̇`` in radians, an ABSOLUTE angle (design §14.1:
    an anchor-relative target would inherit the anchor's own ADS-B path-angle noise as a
    per-flight bias, the quantity §7.5.3 shows integrating into the height error).

    Under ``speed-command`` the first column is the speed loop's ABSOLUTE target,
    ``V + τ_V·tangential`` (m/s) — the airspeed that ``V' = (target − V)/τ_V`` needs, again
    from the kinematics alone. The contract's column is that target RELATIVE to the anchor's
    airspeed, which this function does not know: ``ControlContract.relative_to_anchor`` makes
    it so, and the two callers that know the anchor (the teacher and the anchor actuator) call
    it.

    These are the ACTUAL controls: for the lagged model they are the actuator states, and
    the commands that produced them are :func:`commanded_controls`.
    """
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(STATE_COLUMNS):
        raise ValueError(f"states must be [M,{len(STATE_COLUMNS)}]")
    if times.shape != (len(states),) or len(times) < MINIMUM_INVERSE_STATES:
        raise ValueError(f"times must align with at least {MINIMUM_INVERSE_STATES} states")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("reference times must be strictly increasing")
    if np.asarray(aero_params).shape != (6,):
        raise ValueError("aero parameters must contain six values")
    contract = control_contract(parameterization)

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
    transport = _transport_rate(states)
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
    return contract.teacher(InverseKinematics(
        speed_mps=speed,
        gamma_rad=gamma,
        mass_kg=mass,
        tangential_mps2=tangential,
        bank_rad=np.arctan2(lateral, vertical),
        load_factor=load_factor,
        drag_n=_drag_force(altitude, speed, mass, load_factor, np.asarray(aero_params)),
        max_thrust_n=float(max_thrust_n),
    ))


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
    parameterization: str,
) -> np.ndarray:
    """Invert the first-order lag: ``u_cmd = u + tau * du/dt``.

    Exact for the continuous system, so a command schedule recovered from a trajectory the
    lagged model produced reproduces that trajectory. The derivative is taken on a smoothed
    copy (:data:`COMMAND_SMOOTHING_SAMPLES`) because ``u`` already carries two numerical
    differentiations of position. The lag is the same first-order ODE on whichever
    quantity ``parameterization`` makes an actuator — the first column under three of the
    contracts and the THIRD under ``specific-force+path-angle`` — so one inversion serves all
    four. Under ``speed-command`` it acts on the ABSOLUTE target :func:`actual_controls`
    returns, which commutes with ``ControlContract.relative_to_anchor``'s constant shift; under
    the path-angle contract the third column is already absolute.
    """
    actual = actual_controls(
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        parameterization=parameterization,
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
    config: TSConfig,
) -> np.ndarray:
    return actual_controls(
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        parameterization=config.control_thrust_parameterization,
    )


def _first_order_lag_inverse(
    states: np.ndarray,
    times_s: np.ndarray,
    *,
    aero_params: np.ndarray,
    max_thrust_n: float,
    config: TSConfig,
) -> np.ndarray:
    return commanded_controls(
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        time_constants_s=config.control_time_constants_s,
        parameterization=config.control_thrust_parameterization,
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
) -> np.ndarray:
    """Return the ``[M,3]`` control schedule the CONFIGURED forward model would need, in its
    contract's units. ``states[0]`` is the anchor the rollout starts from, so under
    ``speed-command`` the command is relative to ITS airspeed."""
    schedule = CONTROL_INVERSES[config.control_dynamics_model](
        states,
        times_s,
        aero_params=aero_params,
        max_thrust_n=max_thrust_n,
        config=config,
    )
    return control_contract(config.control_thrust_parameterization).relative_to_anchor(
        schedule, float(states[0][3])
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
