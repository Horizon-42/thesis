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


def _piecewise_schedule(
    segment_durations_s: torch.Tensor,
    integrator_dt_s: float,
    max_steps_per_segment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Detached integer routing plus differentiable per-step durations."""
    batch, segments = segment_durations_s.shape
    device, dtype = segment_durations_s.device, segment_durations_s.dtype
    dt_cap = torch.as_tensor(integrator_dt_s, dtype=dtype, device=device)
    step_counts = torch.ceil(
        segment_durations_s.detach() / integrator_dt_s
    ).to(torch.long)
    endpoint_steps = torch.cumsum(step_counts, dim=1)
    total_steps = endpoint_steps[:, -1]
    max_segment_steps, max_total_steps = torch.stack(
        (step_counts.max(), total_steps.max())
    ).cpu().tolist()
    if max_segment_steps > max_steps_per_segment:
        row, segment = torch.nonzero(
            step_counts == max_segment_steps, as_tuple=False
        )[0].cpu().tolist()
        raise ValueError(
            f"segment {segment} in batch row {row} needs {max_segment_steps} "
            f"integration steps; limit is {max_steps_per_segment}"
        )

    global_steps = torch.arange(max_total_steps, device=device)
    expanded_steps = global_steps.unsqueeze(0).expand(batch, -1).contiguous()
    segment_schedule = torch.searchsorted(
        endpoint_steps.contiguous(), expanded_steps, right=True
    ).clamp(max=segments - 1)
    segment_starts = torch.cat(
        (
            torch.zeros((batch, 1), dtype=torch.long, device=device),
            endpoint_steps[:, :-1],
        ),
        dim=1,
    )
    local_steps = expanded_steps - torch.gather(segment_starts, 1, segment_schedule)
    active_schedule = expanded_steps < total_steps.unsqueeze(1)
    return (
        endpoint_steps,
        segment_schedule,
        local_steps,
        active_schedule,
        max_total_steps,
    )


def _scheduled_rollout_forward(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    *,
    integrator_dt_s: float,
    max_steps_per_segment: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Execute the global schedule and retain only seven state values per step."""
    (
        endpoint_steps,
        segment_schedule,
        local_steps,
        active_schedule,
        max_total_steps,
    ) = _piecewise_schedule(
        segment_durations_s, integrator_dt_s, max_steps_per_segment
    )
    dt_cap = torch.as_tensor(
        integrator_dt_s,
        dtype=initial_states.dtype,
        device=initial_states.device,
    )
    scheduled_controls = torch.gather(
        controls,
        1,
        segment_schedule.unsqueeze(-1).expand(-1, -1, controls.shape[-1]),
    )
    scheduled_segment_durations = torch.gather(
        segment_durations_s, 1, segment_schedule
    )
    remaining_schedule = (
        scheduled_segment_durations
        - local_steps.to(initial_states.dtype) * dt_cap
    ).clamp(min=0.0)
    step_duration_schedule = torch.minimum(remaining_schedule, dt_cap)

    # Step-major storage gives every compiled call the same dense input strides, independent
    # of this batch's total rollout length.  ``transpose(...).contiguous()`` is NOT sufficient
    # for single-flight inference: [1,S,*] -> [S,1,*] is considered contiguous because the
    # batch dimension has size one, so PyTorch may retain an S-dependent singleton stride.
    # Allocate canonical C-order buffers explicitly; ``copy_`` remains differentiable for the
    # controls/durations while removing S from every per-step view's stride guards.
    def canonical_step_major(tensor: torch.Tensor) -> torch.Tensor:
        transposed = tensor.transpose(0, 1)
        return torch.empty(
            transposed.shape,
            dtype=transposed.dtype,
            device=transposed.device,
        ).copy_(transposed)

    controls_by_step = canonical_step_major(scheduled_controls)
    durations_by_step = canonical_step_major(step_duration_schedule)
    active_by_step = canonical_step_major(active_schedule)

    state = initial_states
    states_by_step: list[torch.Tensor] = []
    for step in range(max_total_steps):
        stepped = _rollout_step(
            state,
            controls_by_step[step],
            aero_params,
            durations_by_step[step],
        )
        state = torch.where(active_by_step[step].unsqueeze(-1), stepped, state)
        states_by_step.append(state)

    timeline = torch.stack(states_by_step, dim=1)
    gather_index = (endpoint_steps - 1).unsqueeze(-1).expand(
        -1, -1, state.shape[-1]
    )
    endpoints = torch.gather(timeline, 1, gather_index)
    schedule = (endpoint_steps, segment_schedule, local_steps, active_schedule)
    return endpoints, timeline, schedule


class _PiecewiseRolloutAdjoint(torch.autograd.Function):
    """Memory-bounded discrete adjoint of the exact RK4/geodetic step sequence."""

    @staticmethod
    def forward(
        ctx,
        initial_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        integrator_dt_s: float,
        max_steps_per_segment: int,
    ) -> torch.Tensor:
        # Custom Function.forward runs with grad recording disabled, but Inductor still
        # selects an AOT-training wrapper when input flags carry requires_grad=True. Use
        # detached value views for the numerical pass; the originals are saved below and
        # receive the explicitly reconstructed discrete-adjoint gradients.
        endpoints, timeline, schedule = _scheduled_rollout_forward(
            initial_states.detach(),
            controls.detach(),
            segment_durations_s.detach(),
            aero_params.detach(),
            integrator_dt_s=integrator_dt_s,
            max_steps_per_segment=max_steps_per_segment,
        )
        ctx.integrator_dt_s = integrator_dt_s
        ctx.save_for_backward(
            initial_states,
            controls,
            segment_durations_s,
            aero_params,
            timeline,
            *schedule,
        )
        return endpoints

    @staticmethod
    def backward(ctx, endpoint_gradient: torch.Tensor):
        (
            initial_states,
            controls,
            segment_durations_s,
            aero_params,
            timeline,
            endpoint_steps,
            segment_schedule,
            local_steps,
            active_schedule,
        ) = ctx.saved_tensors
        batch, total_steps, state_size = timeline.shape
        rows = torch.arange(batch, device=timeline.device)
        gradient_by_step = torch.zeros_like(timeline)
        gradient_by_step.scatter_add_(
            1,
            (endpoint_steps - 1).unsqueeze(-1).expand(-1, -1, state_size),
            endpoint_gradient,
        )

        needs_initial, needs_controls, needs_durations, needs_aero = (
            ctx.needs_input_grad[:4]
        )
        control_gradient = torch.zeros_like(controls) if needs_controls else None
        duration_gradient = (
            torch.zeros_like(segment_durations_s) if needs_durations else None
        )
        aero_gradient = torch.zeros_like(aero_params) if needs_aero else None
        state_adjoint = torch.zeros_like(initial_states)
        dt_cap = torch.as_tensor(
            ctx.integrator_dt_s,
            dtype=initial_states.dtype,
            device=initial_states.device,
        )

        for step in range(total_steps - 1, -1, -1):
            output_adjoint = state_adjoint + gradient_by_step[:, step]
            previous_value = initial_states if step == 0 else timeline[:, step - 1]
            segment = segment_schedule[:, step]

            with torch.enable_grad():
                # ``timeline[:, step]`` is a view whose batch stride depends on the
                # rollout length. Normalize it before the compiled local VJP so every
                # reverse step reuses the same static CUDA graph.
                previous = previous_value.detach().contiguous().requires_grad_(True)
                step_control = controls[rows, segment].detach().requires_grad_(
                    needs_controls
                )
                step_duration = segment_durations_s[rows, segment].detach()
                step_duration.requires_grad_(needs_durations)
                step_aero = aero_params.detach().requires_grad_(needs_aero)
                remaining = (
                    step_duration
                    - local_steps[:, step].to(initial_states.dtype) * dt_cap
                ).clamp(min=0.0)
                actual_dt = torch.minimum(remaining, dt_cap)
                stepped = _rollout_step(
                    previous, step_control, step_aero, actual_dt
                )
                next_state = torch.where(
                    active_schedule[:, step].unsqueeze(-1), stepped, previous
                )
                differentiable_inputs = [previous]
                if needs_controls:
                    differentiable_inputs.append(step_control)
                if needs_durations:
                    differentiable_inputs.append(step_duration)
                if needs_aero:
                    differentiable_inputs.append(step_aero)
                local_gradients = torch.autograd.grad(
                    next_state,
                    differentiable_inputs,
                    grad_outputs=output_adjoint,
                    allow_unused=False,
                )

            state_adjoint = local_gradients[0]
            gradient_index = 1
            if needs_controls:
                control_gradient.index_put_(
                    (rows, segment),
                    local_gradients[gradient_index],
                    accumulate=True,
                )
                gradient_index += 1
            if needs_durations:
                duration_gradient.index_put_(
                    (rows, segment),
                    local_gradients[gradient_index],
                    accumulate=True,
                )
                gradient_index += 1
            if needs_aero:
                aero_gradient.add_(local_gradients[gradient_index])

        return (
            state_adjoint if needs_initial else None,
            control_gradient,
            duration_gradient,
            aero_gradient,
            None,
            None,
        )


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

    requires_gradient = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (initial_states, controls, segment_durations_s, aero_params)
    )
    if requires_gradient:
        return _PiecewiseRolloutAdjoint.apply(
            initial_states,
            controls,
            segment_durations_s,
            aero_params,
            integrator_dt_s,
            max_steps_per_segment,
        )
    endpoints, _timeline, _schedule = _scheduled_rollout_forward(
        initial_states,
        controls,
        segment_durations_s,
        aero_params,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )
    return endpoints


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
