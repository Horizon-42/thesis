"""Backend-neutral piecewise-constant rollout with a memory-bounded adjoint.

Dynamics modules provide one differentiable step and keep their state representation
private.  This module owns only control scheduling and reverse accumulation, so adding a
state representation does not duplicate the long-horizon rollout machinery.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


RolloutStep = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


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
    local_steps = expanded_steps - torch.gather(
        segment_starts, 1, segment_schedule
    )
    active_schedule = expanded_steps < total_steps.unsqueeze(1)
    return (
        endpoint_steps,
        segment_schedule,
        local_steps,
        active_schedule,
        max_total_steps,
    )


def _canonical_step_major(tensor: torch.Tensor) -> torch.Tensor:
    transposed = tensor.transpose(0, 1)
    return torch.empty(
        transposed.shape,
        dtype=transposed.dtype,
        device=transposed.device,
    ).copy_(transposed)


def _scheduled_rollout_forward(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    step_context: torch.Tensor,
    *,
    integrator_dt_s: float,
    max_steps_per_segment: int,
    step_function: RolloutStep,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Execute the global schedule and retain only backend state values per step."""
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

    controls_by_step = _canonical_step_major(scheduled_controls)
    durations_by_step = _canonical_step_major(step_duration_schedule)
    active_by_step = _canonical_step_major(active_schedule)

    state = initial_states
    states_by_step: list[torch.Tensor] = []
    for step in range(max_total_steps):
        stepped = step_function(
            state,
            controls_by_step[step],
            aero_params,
            durations_by_step[step],
            step_context,
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
    """Memory-bounded discrete adjoint shared by every dynamics backend."""

    @staticmethod
    def forward(
        ctx,
        initial_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        step_context: torch.Tensor,
        integrator_dt_s: float,
        max_steps_per_segment: int,
        step_function: RolloutStep,
    ) -> torch.Tensor:
        endpoints, timeline, schedule = _scheduled_rollout_forward(
            initial_states.detach(),
            controls.detach(),
            segment_durations_s.detach(),
            aero_params.detach(),
            step_context.detach(),
            integrator_dt_s=integrator_dt_s,
            max_steps_per_segment=max_steps_per_segment,
            step_function=step_function,
        )
        ctx.integrator_dt_s = integrator_dt_s
        ctx.step_function = step_function
        ctx.save_for_backward(
            initial_states,
            controls,
            segment_durations_s,
            aero_params,
            step_context,
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
            step_context,
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
                # A batch-size-one timeline view can report contiguous while retaining a
                # horizon-dependent singleton stride. A fresh canonical clone keeps the
                # compiled local VJP reusable across endpoint rollouts of different lengths.
                previous = previous_value.detach().clone(
                    memory_format=torch.contiguous_format
                ).requires_grad_(True)
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
                stepped = ctx.step_function(
                    previous,
                    step_control,
                    step_aero,
                    actual_dt,
                    step_context,
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
            None,
            None,
        )


def rollout_piecewise_constant_with_step(
    initial_states: torch.Tensor,
    controls: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    step_context: torch.Tensor,
    step_function: RolloutStep,
    *,
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> torch.Tensor:
    """Return segment endpoints while delegating one-step physics to a backend."""
    if initial_states.ndim != 2:
        raise ValueError("initial states must be [B,S]")
    if controls.ndim != 3 or segment_durations_s.shape != controls.shape[:2]:
        raise ValueError("controls must be [B,N,C] and durations must be [B,N]")
    if controls.shape[1] == 0:
        raise ValueError("control rollout requires at least one segment")
    batch = len(initial_states)
    if len(controls) != batch or len(aero_params) != batch or len(step_context) != batch:
        raise ValueError("rollout tensors and step context must share batch size")
    if step_context.requires_grad:
        raise ValueError("rollout step context must be constant")
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
            step_context,
            integrator_dt_s,
            max_steps_per_segment,
            step_function,
        )
    endpoints, _timeline, _schedule = _scheduled_rollout_forward(
        initial_states,
        controls,
        segment_durations_s,
        aero_params,
        step_context,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
        step_function=step_function,
    )
    return endpoints
