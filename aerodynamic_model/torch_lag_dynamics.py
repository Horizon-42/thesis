"""Transport-chart dynamics with first-order actuator lag on the three controls.

The point-mass model applies a piecewise-constant control instantly, so a learned schedule
produces a bank angle that steps at every segment boundary — a trajectory whose curvature
is discontinuous N times and whose roll rate is unbounded. Real aircraft roll into a turn,
spool an engine and load a wing over a finite time.

This model makes the three controls **states** driven towards the commanded value::

    d(delta_T)/dt = (delta_T_cmd - delta_T)/tau_thrust
    d(mu)/dt      = (mu_cmd - mu)          /tau_bank
    d(n)/dt       = (n_cmd - n)            /tau_load

and feeds the ACTUAL values into the unchanged :func:`transport_chart_rhs`. The force
equations, the stall handling, the WGS84 transport term and the chart projection are
literally the same code as the instantaneous model — this module adds three scalar ODEs
and nothing else, and reduces to the point-mass model as every tau goes to zero. That
equivalence is what makes the two models comparable rather than two flight models with two
sets of results.

State is the seven transport-chart values followed by the three actuator values::

    (e, n, u, ve, vn, vu, mass, thrust_fraction, bank_rad, load_factor)

Thrust is carried as a fraction of the flight's installed thrust rather than in newtons,
so all three actuator states are already order one and the long rollout does not need a
separate scaling for them (see ``ts_transformer/control_envelope.py``). ``state_scale``
nondimensionalises the seven point-mass coordinates and is all-ones for the physical
variant.

Controls are the COMMANDS, in the same units as the actuator states.
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_dense_rollout import (
    DenseControlRollout,
    rollout_piecewise_constant_at_times_with_step,
)
from aerodynamic_model.torch_dynamics import CONTROL_NAMES, STATE_NAMES
from aerodynamic_model.torch_piecewise_rollout import (
    rollout_piecewise_constant_with_step,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    TRANSPORT_CHART_STATE_NAMES,
    geodetic_to_transport_chart_state,
    transport_chart_rhs,
)


LAG_ACTUATOR_NAMES = ("thrust_fraction", "bank_rad", "load_factor")
LAG_STATE_NAMES = (*TRANSPORT_CHART_STATE_NAMES, *LAG_ACTUATOR_NAMES)
CHART_WIDTH = len(TRANSPORT_CHART_STATE_NAMES)
# Actuator states are dimensionless or radians already; only the chart half is rescaled.
UNIT_ACTUATOR_SCALE = (1.0, 1.0, 1.0)


def _require_last_dim(tensor: torch.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] != expected:
        raise ValueError(
            f"{name} must end in {expected} values, got shape {tuple(tensor.shape)}"
        )


def lag_state_scale(
    chart_scale: tuple[float, ...] | None, reference: torch.Tensor
) -> torch.Tensor:
    """Return the ``[10]`` divisor for the augmented state, ones when unscaled."""
    chart = (1.0,) * CHART_WIDTH if chart_scale is None else tuple(chart_scale)
    if len(chart) != CHART_WIDTH:
        raise ValueError(f"chart scale must contain {CHART_WIDTH} values")
    return reference.new_tensor((*chart, *UNIT_ACTUATOR_SCALE))


def lag_state_from_geodetic(
    states_geo: torch.Tensor,
    initial_controls: torch.Tensor,
    frame_params: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """Build the augmented state from a geodetic state and the controls in effect.

    ``initial_controls`` is what the aircraft is ALREADY doing at the anchor, not the
    first command: starting the actuators at the first command would place the aircraft in
    a bank it has not rolled into yet, and the lag would then be paid twice.
    """
    _require_last_dim(states_geo, len(STATE_NAMES), "states_geo")
    _require_last_dim(initial_controls, len(CONTROL_NAMES), "initial_controls")
    chart = geodetic_to_transport_chart_state(states_geo, frame_params)
    state = torch.cat((chart, initial_controls.to(chart.dtype)), dim=-1)
    return state / state_scale


def lag_state_to_transport_chart(
    state_lag: torch.Tensor, state_scale: torch.Tensor
) -> torch.Tensor:
    """Return the physical point-mass part every public conversion already understands."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    return (state_lag * state_scale)[..., :CHART_WIDTH]


def lag_actuator_states(
    state_lag: torch.Tensor, state_scale: torch.Tensor
) -> torch.Tensor:
    """Return the realised ``(thrust_fraction, bank, load)`` along the rollout."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    return (state_lag * state_scale)[..., CHART_WIDTH:]


def lag_rhs(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """Continuous RHS of the chart state plus three first-order actuators."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    _require_last_dim(commands, len(CONTROL_NAMES), "commands")
    _require_last_dim(time_constants_s, len(CONTROL_NAMES), "time_constants_s")
    physical = state_lag * state_scale
    chart = physical[..., :CHART_WIDTH]
    actual = physical[..., CHART_WIDTH:]
    thrust_n = actual[..., :1] * max_thrust_n.unsqueeze(-1)
    physical_controls = torch.cat((thrust_n, actual[..., 1:]), dim=-1)
    rate = torch.cat(
        (
            transport_chart_rhs(chart, physical_controls, aero_params, frame_params),
            (commands - actual) / time_constants_s,
        ),
        dim=-1,
    )
    return rate / state_scale


def rk4_lag_step(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """One explicit RK4 step over the coupled point-mass/actuator system."""
    dt = torch.as_tensor(dt_s, dtype=state_lag.dtype, device=state_lag.device)
    while dt.ndim < state_lag.ndim:
        dt = dt.unsqueeze(-1)

    def rhs(state: torch.Tensor) -> torch.Tensor:
        return lag_rhs(
            state,
            commands,
            aero_params,
            frame_params,
            time_constants_s,
            max_thrust_n,
            state_scale,
        )

    k1 = rhs(state_lag)
    k2 = rhs(state_lag + 0.5 * dt * k1)
    k3 = rhs(state_lag + 0.5 * dt * k2)
    k4 = rhs(state_lag + dt * k3)
    return state_lag + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# The rollout engine hands one ``step_context`` tensor to the step function. The lag model
# needs three per-flight/per-run constants there, packed in a fixed order.
_FRAME_WIDTH = 4


def _pack_context(
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
) -> torch.Tensor:
    constants = time_constants_s.reshape(1, -1).expand(len(frame_params), -1)
    return torch.cat(
        (
            frame_params,
            constants.to(frame_params.dtype),
            max_thrust_n.reshape(-1, 1).to(frame_params.dtype),
        ),
        dim=-1,
    )


def _make_step(state_scale: torch.Tensor):
    def step(
        state: torch.Tensor,
        commands: torch.Tensor,
        aero_params: torch.Tensor,
        dt_s: torch.Tensor,
        step_context: torch.Tensor,
    ) -> torch.Tensor:
        return rk4_lag_step(
            state,
            commands,
            aero_params,
            dt_s,
            step_context[..., :_FRAME_WIDTH],
            step_context[..., _FRAME_WIDTH : _FRAME_WIDTH + len(CONTROL_NAMES)],
            step_context[..., -1],
            state_scale,
        )

    return step


def rollout_piecewise_constant(
    initial_geodetic_states: torch.Tensor,
    initial_controls: torch.Tensor,
    commands: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    *,
    chart_scale: tuple[float, ...] | None = None,
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> torch.Tensor:
    """Return augmented segment-end states ``[B,N,10]``."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    return rollout_piecewise_constant_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        _pack_context(frame_params, time_constants_s, max_thrust_n),
        _make_step(state_scale),
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )


def rollout_piecewise_constant_at_times(
    initial_geodetic_states: torch.Tensor,
    initial_controls: torch.Tensor,
    commands: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    chart_scale: tuple[float, ...] | None = None,
    segment_valid: torch.Tensor | None = None,
    integrator_dt_s: float = 0.5,
    max_total_steps: int = 65536,
) -> DenseControlRollout:
    """Return event-aligned augmented states at queries and control boundaries."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    return rollout_piecewise_constant_at_times_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        _pack_context(frame_params, time_constants_s, max_thrust_n),
        _make_step(state_scale),
        query_offsets_s,
        query_valid,
        segment_valid=segment_valid,
        integrator_dt_s=integrator_dt_s,
        max_total_steps=max_total_steps,
    )
