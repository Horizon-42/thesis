"""Continuous WGS84 chart dynamics with physical local-ENU velocity state.

The persistent state is ``(e, n, u, V_E, V_N, V_U, mass)``.  Position uses the
threshold-centred chart shared by the trajectory model, while velocity components are
expressed in the moving local geographic ENU basis.  The full transport-rate cross product
is therefore part of the velocity ODE.  Earth rotation is deliberately absent because the
re-anchored CasADi/PyTorch baseline omits it as well.
"""

from __future__ import annotations

import math

import torch

from geokit import WGS84_A, WGS84_E2
from aerodynamic_model.torch_dense_rollout import (
    DenseControlRollout,
    rollout_piecewise_constant_at_times_with_step,
)
from aerodynamic_model.torch_dynamics import (
    AERO_PARAMETER_NAMES,
    CONTROL_NAMES,
    GRAVITY_MPS2,
    STATE_NAMES,
    aerodynamic_coefficients,
    isa_density,
)
from aerodynamic_model.torch_piecewise_rollout import (
    rollout_piecewise_constant_with_step,
)


TRANSPORT_CHART_STATE_NAMES = (
    "e_m",
    "n_m",
    "u_m",
    "ve_mps",
    "vn_mps",
    "vu_mps",
    "mass_kg",
)


def _require_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} must end in {expected} values, got shape {tuple(tensor.shape)}"
        )


def _broadcast_frame(
    frame_params: torch.Tensor, target_ndim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_last_dim(frame_params, 4, "frame_params")
    lat0, lon0, alt0, heading = frame_params.unbind(-1)
    while lat0.ndim < target_ndim - 1:
        lat0 = lat0.unsqueeze(-1)
        lon0 = lon0.unsqueeze(-1)
        alt0 = alt0.unsqueeze(-1)
        heading = heading.unsqueeze(-1)
    return lat0, lon0, alt0, heading


def _wgs84_geometry(
    north_m: torch.Tensor,
    up_m: torch.Tensor,
    frame_params: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    lat0_deg, _lon0_deg, alt0_m, _heading = _broadcast_frame(
        frame_params, north_m.ndim + 1
    )
    lat0_rad = torch.deg2rad(lat0_deg)
    lat_rad = lat0_rad + north_m / WGS84_A
    altitude_m = alt0_m + up_m
    sin_lat = torch.sin(lat_rad)
    denominator = 1.0 - WGS84_E2 * sin_lat.square()
    radius_n = WGS84_A / torch.sqrt(denominator)
    radius_m = WGS84_A * (1.0 - WGS84_E2) / denominator.pow(1.5)
    return lat0_rad, lat_rad, altitude_m, radius_m, radius_n, alt0_m


def geodetic_to_transport_chart_state(
    states_geo: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    """Convert geodetic state values to the persistent transport-chart state."""
    _require_last_dim(states_geo, len(STATE_NAMES), "states_geo")
    lat, lon, altitude, speed, psi, gamma, mass = states_geo.unbind(-1)
    lat0, lon0, alt0, _heading = _broadcast_frame(frame_params, states_geo.ndim)
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    lat0_rad = torch.deg2rad(lat0)
    lon0_rad = torch.deg2rad(lon0)
    cos_gamma = torch.cos(gamma)
    return torch.stack(
        (
            (lon_rad - lon0_rad) * WGS84_A * torch.cos(lat0_rad),
            (lat_rad - lat0_rad) * WGS84_A,
            altitude - alt0,
            speed * cos_gamma * torch.cos(psi),
            speed * cos_gamma * torch.sin(psi),
            speed * torch.sin(gamma),
            mass,
        ),
        dim=-1,
    )


def transport_chart_state_to_geodetic(
    states_chart: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    """Convert persistent chart/velocity state values to the public geodetic contract."""
    _require_last_dim(
        states_chart, len(TRANSPORT_CHART_STATE_NAMES), "states_chart"
    )
    east, north, up, ve, vn, vu, mass = states_chart.unbind(-1)
    lat0, lon0, alt0, _heading = _broadcast_frame(frame_params, states_chart.ndim)
    lat0_rad = torch.deg2rad(lat0)
    latitude = lat0 + torch.rad2deg(north / WGS84_A)
    longitude = lon0 + torch.rad2deg(
        east / (WGS84_A * torch.cos(lat0_rad))
    )
    horizontal_speed = torch.sqrt(ve.square() + vn.square())
    speed = torch.sqrt(horizontal_speed.square() + vu.square())
    return torch.stack(
        (
            latitude,
            longitude,
            alt0 + up,
            speed,
            torch.atan2(vn, ve),
            torch.atan2(vu, horizontal_speed),
            mass,
        ),
        dim=-1,
    )


def transport_chart_state_to_channels(
    states_chart: torch.Tensor,
    frame_params: torch.Tensor,
    *,
    runway_aligned: bool,
) -> torch.Tensor:
    """Map backend state to the existing six-channel supervision contract."""
    _require_last_dim(
        states_chart, len(TRANSPORT_CHART_STATE_NAMES), "states_chart"
    )
    east, north, up, ve, vn, vu, _mass = states_chart.unbind(-1)
    lat0_rad, lat_rad, altitude, radius_m, radius_n, _alt0 = _wgs84_geometry(
        north, up, frame_params
    )
    factor_e = (
        WGS84_A
        * torch.cos(lat0_rad)
        / ((radius_n + altitude) * torch.cos(lat_rad))
    )
    factor_n = WGS84_A / (radius_m + altitude)
    east_dot = ve * factor_e
    north_dot = vn * factor_n

    if runway_aligned:
        _lat0, _lon0, _alt0, heading = _broadcast_frame(
            frame_params, states_chart.ndim
        )
        cosine, sine = torch.cos(heading), torch.sin(heading)
        first = east * cosine + north * sine
        second = -east * sine + north * cosine
        first_dot = east_dot * cosine + north_dot * sine
        second_dot = -east_dot * sine + north_dot * cosine
    else:
        first, second = east, north
        first_dot, second_dot = east_dot, north_dot
    return torch.stack(
        (first, second, up, first_dot, second_dot, vu), dim=-1
    )


def transport_chart_rhs(
    state_chart: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    """Full-transport continuous RHS in chart position/local-ENU velocity form."""
    _require_last_dim(
        state_chart, len(TRANSPORT_CHART_STATE_NAMES), "state_chart"
    )
    _require_last_dim(controls, len(CONTROL_NAMES), "controls")
    _require_last_dim(aero_params, len(AERO_PARAMETER_NAMES), "aero_params")
    east, north, up, ve, vn, vu, mass = state_chart.unbind(-1)
    del east
    thrust, bank, load_command = controls.unbind(-1)
    lat0_rad, lat_rad, altitude, radius_m, radius_n, _alt0 = _wgs84_geometry(
        north, up, frame_params
    )
    horizontal_speed = torch.sqrt(ve.square() + vn.square())
    speed = torch.sqrt(horizontal_speed.square() + vu.square())
    cos_gamma = horizontal_speed / speed
    sin_gamma = vu / speed
    cos_psi = ve / horizontal_speed
    sin_psi = vn / horizontal_speed

    density = isa_density(altitude)
    _cl, cd, stalled = aerodynamic_coefficients(
        load_command, speed, mass, density, aero_params
    )
    area = aero_params[..., 0]
    cl_max = aero_params[..., 1]
    realized_load = torch.where(
        stalled,
        0.5
        * density
        * speed.square()
        * cl_max
        * area
        / (mass * GRAVITY_MPS2),
        load_command,
    )
    drag = 0.5 * density * speed.square() * cd * area
    speed_rate = (thrust - drag) / mass - GRAVITY_MPS2 * sin_gamma

    tangent = torch.stack(
        (cos_gamma * cos_psi, cos_gamma * sin_psi, sin_gamma), dim=-1
    )
    gamma_normal = torch.stack(
        (-sin_gamma * cos_psi, -sin_gamma * sin_psi, cos_gamma), dim=-1
    )
    heading_normal = torch.stack(
        (-sin_psi, cos_psi, torch.zeros_like(sin_psi)), dim=-1
    )
    force_acceleration = (
        speed_rate.unsqueeze(-1) * tangent
        + (
            GRAVITY_MPS2
            * (realized_load * torch.cos(bank) - cos_gamma)
        ).unsqueeze(-1)
        * gamma_normal
        + (GRAVITY_MPS2 * realized_load * torch.sin(bank)).unsqueeze(-1)
        * heading_normal
    )

    # Moving-local-frame transport.  This is the ENU form documented by the full
    # transport CasADi model; no earth-rotation/Coriolis term is part of either backend.
    omega_transport = torch.stack(
        (
            -vn / (radius_m + altitude),
            ve / (radius_n + altitude),
            ve * torch.tan(lat_rad) / (radius_n + altitude),
        ),
        dim=-1,
    )
    velocity = torch.stack((ve, vn, vu), dim=-1)
    velocity_rate = force_acceleration - torch.linalg.cross(
        omega_transport, velocity, dim=-1
    )

    east_rate = ve * (
        WGS84_A
        * torch.cos(lat0_rad)
        / ((radius_n + altitude) * torch.cos(lat_rad))
    )
    north_rate = vn * WGS84_A / (radius_m + altitude)
    return torch.cat(
        (
            torch.stack((east_rate, north_rate, vu), dim=-1),
            velocity_rate,
            torch.zeros_like(mass).unsqueeze(-1),
        ),
        dim=-1,
    )


def rk4_transport_chart_step(
    state_chart: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    """One explicit RK4 step of the continuous full-transport chart ODE."""
    dt = torch.as_tensor(
        dt_s, dtype=state_chart.dtype, device=state_chart.device
    )
    while dt.ndim < state_chart.ndim:
        dt = dt.unsqueeze(-1)
    k1 = transport_chart_rhs(state_chart, controls, aero_params, frame_params)
    k2 = transport_chart_rhs(
        state_chart + 0.5 * dt * k1, controls, aero_params, frame_params
    )
    k3 = transport_chart_rhs(
        state_chart + 0.5 * dt * k2, controls, aero_params, frame_params
    )
    k4 = transport_chart_rhs(
        state_chart + dt * k3, controls, aero_params, frame_params
    )
    return state_chart + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_COMPILED_CUDA_INFERENCE_STEP = None
_COMPILED_CUDA_AUTOGRAD_STEP = None


def _cuda_inference_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    return rk4_transport_chart_step(
        state, controls, aero_params, dt_s, frame_params
    )


def _cuda_autograd_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    return rk4_transport_chart_step(
        state, controls, aero_params, dt_s, frame_params
    )


def _rollout_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    if not state.is_cuda:
        return rk4_transport_chart_step(
            state, controls, aero_params, dt_s, frame_params
        )
    global _COMPILED_CUDA_INFERENCE_STEP, _COMPILED_CUDA_AUTOGRAD_STEP
    grad_enabled = torch.is_grad_enabled()
    compiled = (
        _COMPILED_CUDA_AUTOGRAD_STEP
        if grad_enabled
        else _COMPILED_CUDA_INFERENCE_STEP
    )
    if compiled is None:
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
    return compiled(state, controls, aero_params, dt_s, frame_params)


def rollout_piecewise_constant(
    initial_geodetic_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    *,
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> torch.Tensor:
    """Return transport-chart segment endpoints ``[B,N,7]``."""
    initial_chart = geodetic_to_transport_chart_state(
        initial_geodetic_states, frame_params
    )
    return rollout_piecewise_constant_with_step(
        initial_chart,
        controls,
        segment_durations_s,
        aero_params,
        frame_params,
        _rollout_step,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )


def rollout_piecewise_constant_at_times(
    initial_geodetic_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    segment_valid: torch.Tensor | None = None,
    integrator_dt_s: float = 0.5,
    max_total_steps: int = 65536,
) -> DenseControlRollout:
    """Return exact event-aligned query and endpoint transport-chart states."""
    initial_chart = geodetic_to_transport_chart_state(
        initial_geodetic_states, frame_params
    )
    return rollout_piecewise_constant_at_times_with_step(
        initial_chart,
        controls,
        segment_durations_s,
        aero_params,
        frame_params,
        _rollout_step,
        query_offsets_s,
        query_valid,
        segment_valid=segment_valid,
        integrator_dt_s=integrator_dt_s,
        max_total_steps=max_total_steps,
    )
