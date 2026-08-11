"""Nondimensional state coordinates for the transport-chart dynamics.

This backend is an affine-free diagonal change of variables around the existing physical
transport chart. Time, controls, aerodynamic parameters, WGS84 transport terms and RK4
substeps remain physical and unchanged. Only the persistent state carried through the
long rollout is scaled to order-one coordinates.
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_dense_rollout import (
    DenseControlRollout,
    rollout_piecewise_constant_at_times_with_step,
)
from aerodynamic_model.torch_piecewise_rollout import (
    rollout_piecewise_constant_with_step,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    geodetic_to_transport_chart_state,
    transport_chart_rhs,
)


# Fixed physical reference units, deliberately independent of train/validation statistics.
# Horizontal terminal-area extent, approach height, horizontal/vertical speed and transport
# mass then all occupy roughly order-one ranges in the persistent state.
SCALED_TRANSPORT_CHART_REFERENCE_UNITS = (
    10_000.0,  # east m
    10_000.0,  # north m
    1_000.0,   # up m
    100.0,     # east velocity m/s
    100.0,     # north velocity m/s
    10.0,      # up velocity m/s
    100_000.0, # mass kg
)


def _state_scale(state: torch.Tensor) -> torch.Tensor:
    return state.new_tensor(SCALED_TRANSPORT_CHART_REFERENCE_UNITS)


def physical_to_scaled_transport_chart_state(
    physical_state: torch.Tensor,
) -> torch.Tensor:
    """Convert the seven physical chart coordinates to nondimensional values."""
    if physical_state.shape[-1] != len(SCALED_TRANSPORT_CHART_REFERENCE_UNITS):
        raise ValueError("physical transport-chart state must end in seven values")
    return physical_state / _state_scale(physical_state)


def scaled_to_physical_transport_chart_state(
    scaled_state: torch.Tensor,
) -> torch.Tensor:
    """Restore the existing physical transport-chart state contract."""
    if scaled_state.shape[-1] != len(SCALED_TRANSPORT_CHART_REFERENCE_UNITS):
        raise ValueError("scaled transport-chart state must end in seven values")
    return scaled_state * _state_scale(scaled_state)


def scaled_transport_chart_rhs(
    scaled_state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the unchanged physical RHS and apply the exact chain rule."""
    physical_state = scaled_to_physical_transport_chart_state(scaled_state)
    physical_rate = transport_chart_rhs(
        physical_state, controls, aero_params, frame_params
    )
    return physical_rate / _state_scale(scaled_state)


def rk4_scaled_transport_chart_step(
    scaled_state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    """One physical-time RK4 step in nondimensional state coordinates."""
    dt = torch.as_tensor(
        dt_s, dtype=scaled_state.dtype, device=scaled_state.device
    )
    while dt.ndim < scaled_state.ndim:
        dt = dt.unsqueeze(-1)
    k1 = scaled_transport_chart_rhs(
        scaled_state, controls, aero_params, frame_params
    )
    k2 = scaled_transport_chart_rhs(
        scaled_state + 0.5 * dt * k1, controls, aero_params, frame_params
    )
    k3 = scaled_transport_chart_rhs(
        scaled_state + 0.5 * dt * k2, controls, aero_params, frame_params
    )
    k4 = scaled_transport_chart_rhs(
        scaled_state + dt * k3, controls, aero_params, frame_params
    )
    return scaled_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_COMPILED_CUDA_INFERENCE_STEP = None
_COMPILED_CUDA_AUTOGRAD_STEP = None


def _cuda_inference_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    return rk4_scaled_transport_chart_step(
        state, controls, aero_params, dt_s, frame_params
    )


def _cuda_autograd_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    frame_params: torch.Tensor,
) -> torch.Tensor:
    return rk4_scaled_transport_chart_step(
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
        return rk4_scaled_transport_chart_step(
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
    """Return nondimensional transport-chart endpoints ``[B,N,7]``."""
    initial_physical = geodetic_to_transport_chart_state(
        initial_geodetic_states, frame_params
    )
    initial_scaled = physical_to_scaled_transport_chart_state(initial_physical)
    return rollout_piecewise_constant_with_step(
        initial_scaled,
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
    """Return event-aligned nondimensional states on the existing physical clock."""
    initial_physical = geodetic_to_transport_chart_state(
        initial_geodetic_states, frame_params
    )
    initial_scaled = physical_to_scaled_transport_chart_state(initial_physical)
    return rollout_piecewise_constant_at_times_with_step(
        initial_scaled,
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
