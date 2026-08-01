"""Differentiable PyTorch twin of :mod:`aerodynamic_model.casadi_simulator`.

This module deliberately mirrors ``CasadiSimulator`` rather than introducing a second
flight model.  One step starts in a local ENU frame anchored at the current geodetic
position, advances the same load-factor point-mass equations with explicit RK4, then
converts position and velocity back to the new local geodetic frame.  All operations stay
in Torch so gradients can flow from rolled-out states to controls and segment durations.

External geodetic states use ``(lat_deg, lon_deg, alt_m, V, psi, gamma, mass_kg)`` and
controls use ``(thrust_N, bank_rad, load_factor)``, exactly like ``CasadiSimulator``.
"""

from __future__ import annotations

import math

import torch

from geokit import WGS84_A, WGS84_B, WGS84_E2, WGS84_E_PRIME2
from .torch_piecewise_rollout import rollout_piecewise_constant_with_step


GRAVITY_MPS2 = 9.81
ISA_T0_K = 288.15
ISA_LAPSE_K_PER_M = 0.0065
ISA_RHO0_KG_M3 = 1.225
ISA_DENSITY_EXPONENT = 4.25588
AERO_PARAMETER_NAMES = ("S", "Cl_max", "Cd0", "k", "stall_threshold", "k_stall")
CONTROL_NAMES = ("thrust_N", "bank_rad", "load_factor")
STATE_NAMES = ("lat_deg", "lon_deg", "alt_m", "V", "psi", "gamma", "mass_kg")


def _require_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(f"{name} must end in {expected} values, got shape {tuple(tensor.shape)}")


def isa_density(altitude_m: torch.Tensor) -> torch.Tensor:
    """The same simplified ISA density expression used by the CasADi model."""
    temperature = ISA_T0_K - ISA_LAPSE_K_PER_M * altitude_m
    return ISA_RHO0_KG_M3 * (temperature / ISA_T0_K).pow(ISA_DENSITY_EXPONENT)


def aerodynamic_coefficients(
    load_factor: torch.Tensor,
    speed_mps: torch.Tensor,
    mass_kg: torch.Tensor,
    density: torch.Tensor,
    aero_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(Cl, Cd, stalled)`` with CasADi-identical stall handling."""
    _require_last_dim(aero_params, len(AERO_PARAMETER_NAMES), "aero_params")
    area, cl_max, cd0, induced_k, stall_threshold, k_stall = aero_params.unbind(-1)
    cl_required = (
        load_factor * mass_kg * GRAVITY_MPS2
        / (0.5 * density * area * speed_mps.square())
    )
    ratio = cl_required / cl_max
    stalled = ratio > 1.0
    cl = torch.where(stalled, cl_max, cl_required)
    stall_fraction = torch.minimum(ratio, torch.ones_like(ratio))
    transition = ((stall_fraction - stall_threshold) / (1.0 - stall_threshold)).clamp(
        min=0.0, max=1.0
    )
    smooth = transition.square() * (3.0 - 2.0 * transition)
    stall_drag = torch.where(
        ratio > stall_threshold, smooth * k_stall, torch.zeros_like(ratio)
    )
    cd = cd0 + induced_k * cl.square() + stall_drag
    return cl, cd, stalled


def enu_rhs(
    state_enu: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
) -> torch.Tensor:
    """Flat-ENU point-mass RHS mirrored from ``make_dynamics_model``."""
    _require_last_dim(state_enu, len(STATE_NAMES), "state_enu")
    _require_last_dim(controls, len(CONTROL_NAMES), "controls")
    east, north, altitude, speed, psi, gamma, mass = state_enu.unbind(-1)
    thrust, bank, load_command = controls.unbind(-1)
    density = isa_density(altitude)
    _cl, cd, stalled = aerodynamic_coefficients(
        load_command, speed, mass, density, aero_params
    )
    area = aero_params[..., 0]
    cl_max = aero_params[..., 1]
    realized_load = torch.where(
        stalled,
        0.5 * density * speed.square() * cl_max * area / (mass * GRAVITY_MPS2),
        load_command,
    )
    drag = 0.5 * density * speed.square() * cd * area
    cos_gamma = torch.cos(gamma)
    return torch.stack(
        (
            speed * cos_gamma * torch.cos(psi),
            speed * cos_gamma * torch.sin(psi),
            speed * torch.sin(gamma),
            (thrust - drag) / mass - GRAVITY_MPS2 * torch.sin(gamma),
            GRAVITY_MPS2 * realized_load * torch.sin(bank) / (speed * cos_gamma),
            GRAVITY_MPS2
            * (realized_load * torch.cos(bank) - cos_gamma)
            / speed,
            torch.zeros_like(east),
        ),
        dim=-1,
    )


def rk4_enu_step(
    state_enu: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
) -> torch.Tensor:
    """One explicit RK4 step of the shared ENU RHS."""
    dt = torch.as_tensor(dt_s, dtype=state_enu.dtype, device=state_enu.device)
    while dt.ndim < state_enu.ndim:
        dt = dt.unsqueeze(-1)
    k1 = enu_rhs(state_enu, controls, aero_params)
    k2 = enu_rhs(state_enu + 0.5 * dt * k1, controls, aero_params)
    k3 = enu_rhs(state_enu + 0.5 * dt * k2, controls, aero_params)
    k4 = enu_rhs(state_enu + dt * k3, controls, aero_params)
    return state_enu + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def geodetic_deg_to_ecef(geodetic: torch.Tensor) -> torch.Tensor:
    """Torch twin of ``geodetic_deg_to_ecef_expr`` for ``[...,3]`` values."""
    _require_last_dim(geodetic, 3, "geodetic")
    lat_deg, lon_deg, altitude = geodetic.unbind(-1)
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    prime_vertical = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * torch.sin(lat).square())
    return torch.stack(
        (
            (prime_vertical + altitude) * torch.cos(lat) * torch.cos(lon),
            (prime_vertical + altitude) * torch.cos(lat) * torch.sin(lon),
            (prime_vertical * (1.0 - WGS84_E2) + altitude) * torch.sin(lat),
        ),
        dim=-1,
    )


def ecef_to_geodetic(ecef: torch.Tensor) -> torch.Tensor:
    """Torch twin of the Bowring-form ``ecef_to_geodetic_expr``."""
    _require_last_dim(ecef, 3, "ecef")
    x, y, z = ecef.unbind(-1)
    horizontal = torch.sqrt(x.square() + y.square())
    lon = torch.atan2(y, x)
    q = torch.atan2(z * WGS84_A, horizontal * WGS84_B)
    lat = torch.atan2(
        z + WGS84_E_PRIME2 * WGS84_B * torch.sin(q).pow(3),
        horizontal - WGS84_E2 * WGS84_A * torch.cos(q).pow(3),
    )
    prime_vertical = WGS84_A / torch.sqrt(1.0 - WGS84_E2 * torch.sin(lat).square())
    altitude = horizontal / torch.cos(lat) - prime_vertical
    return torch.stack((torch.rad2deg(lat), torch.rad2deg(lon), altitude), dim=-1)


def ecef_to_enu_rotation(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Return the CasADi-identical ECEF→ENU rotation matrix ``[...,3,3]``."""
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    zeros = torch.zeros_like(lat)
    east = torch.stack((-torch.sin(lon), torch.cos(lon), zeros), dim=-1)
    north = torch.stack(
        (-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)),
        dim=-1,
    )
    up = torch.stack(
        (torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)),
        dim=-1,
    )
    return torch.stack((east, north, up), dim=-2)


def enu_state_to_geodetic(
    state_enu: torch.Tensor, reference_lat_lon_deg: torch.Tensor
) -> torch.Tensor:
    """Map a stepped ENU state back to the moving geodetic state contract."""
    _require_last_dim(state_enu, len(STATE_NAMES), "state_enu")
    _require_last_dim(reference_lat_lon_deg, 2, "reference_lat_lon_deg")
    east, north, up, speed, psi, gamma, mass = state_enu.unbind(-1)
    ref_lat, ref_lon = reference_lat_lon_deg.unbind(-1)
    ref_geo = torch.stack((ref_lat, ref_lon, torch.zeros_like(ref_lat)), dim=-1)
    ref_ecef = geodetic_deg_to_ecef(ref_geo)
    ref_rotation = ecef_to_enu_rotation(ref_lat, ref_lon)
    offset_enu = torch.stack((east, north, up), dim=-1)
    offset_ecef = torch.matmul(ref_rotation.transpose(-1, -2), offset_enu.unsqueeze(-1)).squeeze(-1)
    new_geo = ecef_to_geodetic(ref_ecef + offset_ecef)

    cos_gamma = torch.cos(gamma)
    velocity_enu = torch.stack(
        (
            speed * cos_gamma * torch.cos(psi),
            speed * cos_gamma * torch.sin(psi),
            speed * torch.sin(gamma),
        ),
        dim=-1,
    )
    velocity_ecef = torch.matmul(
        ref_rotation.transpose(-1, -2), velocity_enu.unsqueeze(-1)
    ).squeeze(-1)
    new_rotation = ecef_to_enu_rotation(new_geo[..., 0], new_geo[..., 1])
    velocity_new = torch.matmul(new_rotation, velocity_ecef.unsqueeze(-1)).squeeze(-1)
    ve, vn, vu = velocity_new.unbind(-1)
    new_speed = torch.linalg.vector_norm(velocity_new, dim=-1)
    horizontal_speed = torch.sqrt(ve.square() + vn.square())
    new_psi = torch.atan2(vn, ve)
    new_gamma = torch.atan2(vu, horizontal_speed)
    return torch.stack(
        (
            new_geo[..., 0],
            new_geo[..., 1],
            new_geo[..., 2],
            new_speed,
            new_psi,
            new_gamma,
            mass,
        ),
        dim=-1,
    )


def geodetic_step(
    state_geo: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
) -> torch.Tensor:
    """One differentiable step with the exact discrete structure of ``CasadiSimulator``."""
    _require_last_dim(state_geo, len(STATE_NAMES), "state_geo")
    reference = state_geo[..., :2]
    zeros = torch.zeros_like(state_geo[..., 0])
    state_enu = torch.stack(
        (
            zeros,
            zeros,
            state_geo[..., 2],
            state_geo[..., 3],
            state_geo[..., 4],
            state_geo[..., 5],
            state_geo[..., 6],
        ),
        dim=-1,
    )
    stepped = rk4_enu_step(state_enu, controls, aero_params, dt_s)
    return enu_state_to_geodetic(stepped, reference)


_COMPILED_CUDA_INFERENCE_STEP = None
_COMPILED_CUDA_AUTOGRAD_STEP = None


def _cuda_inference_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
) -> torch.Tensor:
    """Distinct code object so no-grad shape caches do not consume VJP entries."""
    return geodetic_step(state, controls, aero_params, dt_s)


def _cuda_autograd_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
) -> torch.Tensor:
    """Distinct code object for the grad-enabled local discrete-adjoint step."""
    return geodetic_step(state, controls, aero_params, dt_s)


def _rollout_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
) -> torch.Tensor:
    """Use the eager reference on CPU and a fused, safe Inductor graph on CUDA."""
    if not state.is_cuda:
        return geodetic_step(state, controls, aero_params, dt_s)
    global _COMPILED_CUDA_INFERENCE_STEP, _COMPILED_CUDA_AUTOGRAD_STEP
    grad_enabled = torch.is_grad_enabled()
    compiled = (
        _COMPILED_CUDA_AUTOGRAD_STEP
        if grad_enabled
        else _COMPILED_CUDA_INFERENCE_STEP
    )
    if compiled is None:
        # ``dynamic=True, mode="reduce-overhead"`` reproducibly segfaults in backward
        # with the project's Torch/CUDA stack, so the autograd kernel remains static.
        # Inference has no backward graph and must accept full and partial batches across
        # train/validation replay without consuming Dynamo's finite recompile cache.
        compiled = torch.compile(
            _cuda_autograd_step if grad_enabled else _cuda_inference_step,
            fullgraph=True,
            dynamic=not grad_enabled,
            mode="reduce-overhead",
        )
        if grad_enabled:
            _COMPILED_CUDA_AUTOGRAD_STEP = compiled
        else:
            _COMPILED_CUDA_INFERENCE_STEP = compiled
    return compiled(state, controls, aero_params, dt_s)


def rollout_piecewise_constant(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    *,
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> torch.Tensor:
    """Roll batched non-uniform controls and return segment-end states ``[B,N,7]``.

    Each segment is subdivided exactly like the numeric rollout: steps are capped at
    ``integrator_dt_s`` and the final step is clamped to the segment boundary. Flights
    advance through their own segment schedules independently inside one global batch loop;
    a short segment in one row therefore does not wait for another row's long segment. The
    integer schedule is chosen from detached durations, while every actual step duration
    remains a tensor, so gradients flow through the active branch to duration logits.
    """
    _require_last_dim(initial_states, len(STATE_NAMES), "initial_states")
    _require_last_dim(controls, len(CONTROL_NAMES), "controls")
    if controls.ndim != 3 or segment_durations_s.shape != controls.shape[:2]:
        raise ValueError("controls must be [B,N,3] and segment_durations_s must be [B,N]")
    if controls.shape[1] == 0:
        raise ValueError("control rollout requires at least one segment")
    if initial_states.shape[0] != controls.shape[0] or aero_params.shape[0] != controls.shape[0]:
        raise ValueError("initial states, controls, and aero parameters must share batch size")
    if integrator_dt_s <= 0.0:
        raise ValueError("integrator_dt_s must be positive")
    if torch.any(segment_durations_s <= 0.0):
        raise ValueError("every segment duration must be positive")

    step_context = initial_states.new_empty((len(initial_states), 0))
    return rollout_piecewise_constant_with_step(
        initial_states,
        controls,
        segment_durations_s,
        aero_params,
        step_context,
        _baseline_rollout_step,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )


def _baseline_rollout_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """Adapt the re-anchored baseline step to the backend-neutral engine."""
    del step_context
    return _rollout_step(state, controls, aero_params, dt_s)


def geodetic_states_to_channels(
    states_geo: torch.Tensor,
    frame_params: torch.Tensor,
    *,
    runway_aligned: bool,
) -> torch.Tensor:
    """Differentiable twin of ``channels.channels_from_states``.

    ``frame_params`` is ``[B,4] = (lat0_deg, lon0_deg, alt0_m, heading_rad)``.
    """
    _require_last_dim(states_geo, len(STATE_NAMES), "states_geo")
    _require_last_dim(frame_params, 4, "frame_params")
    lat, lon, altitude, speed, psi, gamma, _mass = states_geo.unbind(-1)
    lat0, lon0, alt0, heading = frame_params.unbind(-1)
    while lat0.ndim < lat.ndim:
        lat0 = lat0.unsqueeze(-1)
        lon0 = lon0.unsqueeze(-1)
        alt0 = alt0.unsqueeze(-1)
        heading = heading.unsqueeze(-1)

    metres_per_degree = WGS84_A * (math.pi / 180.0)
    east = (lon - lon0) * metres_per_degree * torch.cos(torch.deg2rad(lat0))
    north = (lat - lat0) * metres_per_degree

    lat_rad = torch.deg2rad(lat)
    denom = 1.0 - WGS84_E2 * torch.sin(lat_rad).square()
    r_n = WGS84_A / torch.sqrt(denom)
    r_m = WGS84_A * (1.0 - WGS84_E2) / denom.pow(1.5)
    ground_speed = speed * torch.cos(gamma)
    factor_e = (
        WGS84_A
        * torch.cos(torch.deg2rad(lat0))
        / ((r_n + altitude) * torch.cos(lat_rad))
    )
    factor_n = WGS84_A / (r_m + altitude)
    east_dot = ground_speed * torch.cos(psi) * factor_e
    north_dot = ground_speed * torch.sin(psi) * factor_n

    if runway_aligned:
        cosine, sine = torch.cos(heading), torch.sin(heading)
        first = east * cosine + north * sine
        second = -east * sine + north * cosine
        first_dot = east_dot * cosine + north_dot * sine
        second_dot = -east_dot * sine + north_dot * cosine
    else:
        first, second = east, north
        first_dot, second_dot = east_dot, north_dot
    return torch.stack(
        (
            first,
            second,
            altitude - alt0,
            first_dot,
            second_dot,
            speed * torch.sin(gamma),
        ),
        dim=-1,
    )
