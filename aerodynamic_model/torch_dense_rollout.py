"""Differentiable piecewise-control rollout sampled on a fixed physical-time grid.

The existing endpoint rollout deliberately exposes only learned segment boundaries.  Dense
state supervision has a different contract: control switches remain non-uniform, while
states are queried at regular physical timestamps.  This module builds the sorted union of
the global integration grid and learned switch times, so a query is an actual RK4 state —
never a linear interpolation between segment endpoints and never a second rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aerodynamic_model.torch_dynamics import (
    AERO_PARAMETER_NAMES,
    CONTROL_NAMES,
    STATE_NAMES,
    _rollout_step,
)
from aerodynamic_model.torch_piecewise_rollout import RolloutStep


@dataclass(frozen=True)
class DenseControlRollout:
    query_states: torch.Tensor       # [B,M,7]
    segment_end_states: torch.Tensor  # [B,N,7]


def _require_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _build_event_schedule(
    segment_durations_s: torch.Tensor,
    segment_valid: torch.Tensor,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    integrator_dt_s: float,
    max_total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable step durations plus detached routing/index tensors."""
    batch, segments = segment_durations_s.shape
    if query_offsets_s.ndim != 2 or query_offsets_s.shape[0] != batch:
        raise ValueError("dense query offsets must be [B,M]")
    _require_shape(query_valid, tuple(query_offsets_s.shape), "dense query mask")
    if query_valid.dtype != torch.bool:
        raise ValueError("dense query mask must be boolean")
    _require_shape(segment_valid, tuple(segment_durations_s.shape), "segment mask")
    if segment_valid.dtype != torch.bool:
        raise ValueError("segment mask must be boolean")
    if not torch.all(segment_valid.any(dim=1)):
        raise ValueError("every trajectory must retain at least one control segment")
    if torch.any(segment_valid[:, 1:] & ~segment_valid[:, :-1]):
        raise ValueError("valid control segments must form a prefix")
    if torch.any(segment_durations_s[segment_valid] <= 0.0):
        raise ValueError("every valid control segment duration must be positive")
    segment_durations_s = torch.where(
        segment_valid, segment_durations_s, torch.zeros_like(segment_durations_s)
    )

    device, dtype = segment_durations_s.device, segment_durations_s.dtype
    dt_cap = torch.as_tensor(integrator_dt_s, dtype=dtype, device=device)
    boundaries = segment_durations_s.cumsum(dim=1)
    total_duration = boundaries[:, -1]
    fixed_count = int(torch.ceil(total_duration.detach().max() / dt_cap).cpu())
    query_count = query_offsets_s.shape[1]
    scheduled_count = fixed_count + segments + query_count
    if scheduled_count > max_total_steps:
        raise ValueError(
            f"dense control rollout needs {scheduled_count} scheduled events; "
            f"limit is {max_total_steps}"
        )

    fixed_times = (
        torch.arange(1, fixed_count + 1, dtype=dtype, device=device) * dt_cap
    ).unsqueeze(0).expand(batch, -1)
    tolerance = torch.finfo(dtype).eps * total_duration.detach().abs().clamp(min=1.0) * 16
    fixed_valid = fixed_times <= total_duration.detach().unsqueeze(1) + tolerance.unsqueeze(1)
    query_times = query_offsets_s.to(dtype=dtype, device=device)
    query_in_range = (
        torch.isfinite(query_times)
        & (query_times > 0.0)
        & (query_times <= total_duration.detach().unsqueeze(1) + tolerance.unsqueeze(1))
    )
    if torch.any(query_valid & ~query_in_range):
        invalid = query_valid & ~query_in_range
        invalid_row, invalid_column = torch.nonzero(invalid, as_tuple=True)
        row = int(invalid_row[0].detach().cpu())
        column = int(invalid_column[0].detach().cpu())
        raise ValueError(
            "every valid dense query timestamp must be finite, positive, and no later "
            "than its trajectory duration; "
            f"first invalid query[{row},{column}]="
            f"{float(query_times[row, column].detach().cpu()):.17g}s, "
            f"duration={float(total_duration[row].detach().cpu()):.17g}s, "
            f"tolerance={float(tolerance[row].detach().cpu()):.17g}s"
        )

    # Queries are first-class events. They need not divide the integration-step cap: adding
    # them to the sorted union creates shorter RK4 steps on either side while preserving
    # ``integrator_dt_s`` as an upper bound.
    event_times = torch.cat((fixed_times, boundaries, query_times), dim=1)
    event_valid = torch.cat(
        (
            fixed_valid,
            segment_valid,
            query_valid,
        ),
        dim=1,
    )
    sort_key = torch.where(
        event_valid,
        event_times.detach(),
        torch.full_like(event_times, torch.inf),
    )
    order = torch.argsort(sort_key, dim=1, stable=True)
    sorted_times = torch.gather(event_times, 1, order)
    sorted_valid = torch.gather(event_valid, 1, order)
    previous_times = torch.cat(
        (torch.zeros((batch, 1), dtype=dtype, device=device), sorted_times[:, :-1]),
        dim=1,
    )
    step_durations = torch.where(
        sorted_valid,
        sorted_times - previous_times,
        torch.zeros_like(sorted_times),
    )
    if torch.any(step_durations.detach() < -tolerance.unsqueeze(1)):
        raise RuntimeError("dense control event schedule is not monotonic")
    step_durations = step_durations.clamp(min=0.0)
    if torch.any(step_durations.detach() > dt_cap + tolerance.unsqueeze(1)):
        raise RuntimeError("dense control schedule exceeded the integration-step cap")

    midpoints = (0.5 * (previous_times + sorted_times)).detach().contiguous()
    control_schedule = torch.searchsorted(
        boundaries.detach().contiguous(), midpoints, right=False
    ).clamp(max=segments - 1)

    positions = torch.arange(
        event_times.shape[1], dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch, -1)
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(1, order, positions)

    if query_count:
        query_original = fixed_count + segments + torch.arange(
            query_count, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch, -1)
        query_steps = torch.gather(
            inverse_order,
            1,
            query_original,
        )
    else:
        query_steps = torch.empty(
            (batch, 0), dtype=torch.long, device=device
        )
    endpoint_original = fixed_count + torch.arange(
        segments, dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch, -1)
    endpoint_steps = torch.gather(inverse_order, 1, endpoint_original)
    return (
        step_durations,
        control_schedule,
        sorted_valid,
        query_steps,
        endpoint_steps,
    )


def _scheduled_forward(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    step_durations: torch.Tensor,
    aero_params: torch.Tensor,
    step_context: torch.Tensor,
    control_schedule: torch.Tensor,
    active_schedule: torch.Tensor,
    step_function: RolloutStep,
) -> torch.Tensor:
    """Execute a prebuilt event schedule and retain its complete discrete timeline."""
    scheduled_controls = torch.gather(
        controls,
        1,
        control_schedule.unsqueeze(-1).expand(-1, -1, controls.shape[-1]),
    )

    def step_major(tensor: torch.Tensor) -> torch.Tensor:
        transposed = tensor.transpose(0, 1)
        return torch.empty(
            transposed.shape, dtype=transposed.dtype, device=transposed.device
        ).copy_(transposed)

    controls_by_step = step_major(scheduled_controls)
    durations_by_step = step_major(step_durations)
    active_by_step = step_major(active_schedule)
    state = initial_states
    timeline = []
    for step in range(step_durations.shape[1]):
        stepped = step_function(
            state,
            controls_by_step[step],
            aero_params,
            durations_by_step[step],
            step_context,
        )
        state = torch.where(active_by_step[step].unsqueeze(-1), stepped, state)
        timeline.append(state)
    return torch.stack(timeline, dim=1)


class _DenseScheduledRolloutAdjoint(torch.autograd.Function):
    """Memory-bounded adjoint with losses allowed at query and segment-end events."""

    @staticmethod
    def forward(
        ctx,
        initial_states,
        controls,
        step_durations,
        aero_params,
        step_context,
        control_schedule,
        active_schedule,
        query_steps,
        query_valid,
        endpoint_steps,
        step_function,
    ):
        timeline = _scheduled_forward(
            initial_states.detach(),
            controls.detach(),
            step_durations.detach(),
            aero_params.detach(),
            step_context.detach(),
            control_schedule,
            active_schedule,
            step_function,
        )
        state_size = timeline.shape[-1]
        query_states = torch.gather(
            timeline,
            1,
            query_steps.unsqueeze(-1).expand(-1, -1, state_size),
        )
        endpoint_states = torch.gather(
            timeline,
            1,
            endpoint_steps.unsqueeze(-1).expand(-1, -1, state_size),
        )
        ctx.step_function = step_function
        ctx.save_for_backward(
            initial_states,
            controls,
            step_durations,
            aero_params,
            step_context,
            timeline,
            control_schedule,
            active_schedule,
            query_steps,
            query_valid,
            endpoint_steps,
        )
        return query_states, endpoint_states

    @staticmethod
    def backward(ctx, query_gradient, endpoint_gradient):
        (
            initial_states,
            controls,
            step_durations,
            aero_params,
            step_context,
            timeline,
            control_schedule,
            active_schedule,
            query_steps,
            query_valid,
            endpoint_steps,
        ) = ctx.saved_tensors
        batch, total_steps, state_size = timeline.shape
        rows = torch.arange(batch, device=timeline.device)
        gradient_by_step = torch.zeros_like(timeline)
        if query_gradient is not None and query_steps.shape[1]:
            gradient_by_step.scatter_add_(
                1,
                query_steps.unsqueeze(-1).expand(-1, -1, state_size),
                query_gradient * query_valid.unsqueeze(-1),
            )
        if endpoint_gradient is not None:
            gradient_by_step.scatter_add_(
                1,
                endpoint_steps.unsqueeze(-1).expand(-1, -1, state_size),
                endpoint_gradient,
            )

        needs_initial, needs_controls, needs_durations, needs_aero = (
            ctx.needs_input_grad[:4]
        )
        control_gradient = torch.zeros_like(controls) if needs_controls else None
        duration_gradient = (
            torch.zeros_like(step_durations) if needs_durations else None
        )
        aero_gradient = torch.zeros_like(aero_params) if needs_aero else None
        state_adjoint = torch.zeros_like(initial_states)

        for step in range(total_steps - 1, -1, -1):
            output_adjoint = state_adjoint + gradient_by_step[:, step]
            previous_value = initial_states if step == 0 else timeline[:, step - 1]
            segment = control_schedule[:, step]
            with torch.enable_grad():
                # For batch size one, ``contiguous()`` may retain the timeline's
                # horizon-dependent singleton stride.  A fresh canonical allocation keeps
                # the static CUDA VJP kernel reusable across curriculum horizons.
                previous = previous_value.detach().clone(
                    memory_format=torch.contiguous_format
                ).requires_grad_(True)
                variables = [previous]
                labels = ["state"]
                step_control = controls[rows, segment].detach()
                if needs_controls:
                    step_control.requires_grad_(True)
                    variables.append(step_control)
                    labels.append("control")
                step_duration = step_durations[:, step].detach().clone(
                    memory_format=torch.contiguous_format
                )
                if needs_durations:
                    step_duration.requires_grad_(True)
                    variables.append(step_duration)
                    labels.append("duration")
                step_aero = aero_params.detach()
                if needs_aero:
                    step_aero.requires_grad_(True)
                    variables.append(step_aero)
                    labels.append("aero")
                stepped = ctx.step_function(
                    previous,
                    step_control,
                    step_aero,
                    step_duration,
                    step_context,
                )
                next_state = torch.where(
                    active_schedule[:, step].unsqueeze(-1), stepped, previous
                )
                gradients = torch.autograd.grad(
                    next_state,
                    variables,
                    grad_outputs=output_adjoint,
                    allow_unused=True,
                )
            by_name = dict(zip(labels, gradients))
            state_adjoint = by_name["state"]
            if needs_controls:
                control_gradient.scatter_add_(
                    1,
                    segment.view(batch, 1, 1).expand(-1, 1, controls.shape[-1]),
                    by_name["control"].unsqueeze(1),
                )
            if needs_durations:
                duration_gradient[:, step] = by_name["duration"]
            if needs_aero:
                aero_gradient += by_name["aero"]

        return (
            state_adjoint if needs_initial else None,
            control_gradient,
            duration_gradient,
            aero_gradient,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rollout_piecewise_constant_at_times_with_step(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    step_context: torch.Tensor,
    step_function: RolloutStep,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    segment_valid: torch.Tensor | None = None,
    integrator_dt_s: float = 0.5,
    max_total_steps: int = 65536,
) -> DenseControlRollout:
    """Roll once and return exact RK4 states at fixed queries and control boundaries."""
    if initial_states.ndim != 2 or initial_states.shape[-1] != len(STATE_NAMES):
        raise ValueError("initial states must be [B,7]")
    if controls.ndim != 3 or controls.shape[-1] != len(CONTROL_NAMES):
        raise ValueError("controls must be [B,N,3]")
    if segment_durations_s.shape != controls.shape[:2]:
        raise ValueError("segment durations must align with controls [B,N]")
    if aero_params.shape != (len(initial_states), len(AERO_PARAMETER_NAMES)):
        raise ValueError("aero parameters must be [B,6]")
    if len(initial_states) != len(controls):
        raise ValueError("dense rollout inputs must share batch size")
    if len(step_context) != len(initial_states):
        raise ValueError("dense rollout step context must share batch size")
    if step_context.requires_grad:
        raise ValueError("dense rollout step context must be constant")
    if segment_valid is None:
        segment_valid = torch.ones_like(segment_durations_s, dtype=torch.bool)
    else:
        segment_valid = segment_valid.to(device=segment_durations_s.device)
    if integrator_dt_s <= 0.0:
        raise ValueError("integrator_dt_s must be positive")
    if max_total_steps <= 0:
        raise ValueError("max_total_steps must be positive")

    query_offsets_s = query_offsets_s.to(
        dtype=segment_durations_s.dtype, device=segment_durations_s.device
    )
    query_valid = query_valid.to(device=segment_durations_s.device)
    schedule = _build_event_schedule(
        segment_durations_s,
        segment_valid,
        query_offsets_s,
        query_valid,
        integrator_dt_s=integrator_dt_s,
        max_total_steps=max_total_steps,
    )
    step_durations, control_schedule, active, query_steps, endpoint_steps = schedule
    requires_gradient = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (initial_states, controls, step_durations, aero_params)
    )
    if requires_gradient:
        query_states, endpoint_states = _DenseScheduledRolloutAdjoint.apply(
            initial_states,
            controls,
            step_durations,
            aero_params,
            step_context,
            control_schedule,
            active,
            query_steps,
            query_valid,
            endpoint_steps,
            step_function,
        )
    else:
        timeline = _scheduled_forward(
            initial_states,
            controls,
            step_durations,
            aero_params,
            step_context,
            control_schedule,
            active,
            step_function,
        )
        query_states = torch.gather(
            timeline,
            1,
            query_steps.unsqueeze(-1).expand(-1, -1, timeline.shape[-1]),
        )
        endpoint_states = torch.gather(
            timeline,
            1,
            endpoint_steps.unsqueeze(-1).expand(-1, -1, timeline.shape[-1]),
        )
    return DenseControlRollout(query_states, endpoint_states)


def _baseline_dense_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    del step_context
    return _rollout_step(state, controls, aero_params, dt_s)


def rollout_piecewise_constant_at_times(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    segment_valid: torch.Tensor | None = None,
    integrator_dt_s: float = 0.5,
    max_total_steps: int = 65536,
) -> DenseControlRollout:
    """Backward-compatible re-anchored baseline wrapper around the generic engine."""
    step_context = initial_states.new_empty((len(initial_states), 0))
    return rollout_piecewise_constant_at_times_with_step(
        initial_states,
        controls,
        segment_durations_s,
        aero_params,
        step_context,
        _baseline_dense_step,
        query_offsets_s,
        query_valid,
        segment_valid=segment_valid,
        integrator_dt_s=integrator_dt_s,
        max_total_steps=max_total_steps,
    )
