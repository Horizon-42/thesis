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

The first actuator is the longitudinal command, and WHICH quantity it is is the control
law (``control_law=`` on every rollout below):

* :data:`THRUST_FRACTION_LAW` — thrust as a fraction of the flight's installed thrust,
  ``T = a_x · T_max`` (``ts_transformer/outputs/envelope.py``);
* :class:`SpecificForceLaw` — the specific force along the path, ``a_x = (T - D)/W``, with
  the thrust recomputed at EVERY RHS evaluation as ``clamp(W·a_x + D, T_lo, T_max)``. The
  RHS subtracts the same drag, so wherever the clamp does not bind ``V' = g·(a_x - sin γ)``
  and the airframe leaves the speed equation (``ts_transformer/docs/
  2026-09-14_specific_force_control_design.md``);
* :class:`SpeedCommandLaw` — a speed command relative to the anchor airspeed, ``a_Δ`` m/s,
  flown by a first-order speed loop through the specific-force law's thrust: wherever the
  clamp does not bind ``V' = (V₀ + a_Δ − V)/τ_V`` (the same design, §12).

The actuator states are order one under the first two laws and metres per second under the
third; a fixed-step RK4 is invariant to a linear rescaling of a state, so none of them needs
a scale of its own. ``state_scale`` nondimensionalises the seven point-mass coordinates and is
all-ones for the physical variant.

Controls are the COMMANDS, in the same units as the actuator states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from aerodynamic_model.torch_dense_rollout import (
    DenseControlRollout,
    rollout_piecewise_constant_at_times_with_step,
)
from aerodynamic_model.torch_dynamics import CONTROL_NAMES, STATE_NAMES
from aerodynamic_model.torch_piecewise_rollout import (
    RawCommandHook,
    rollout_piecewise_constant_hooked_with_step,
    rollout_piecewise_constant_with_step,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    TRANSPORT_CHART_STATE_NAMES,
    geodetic_to_transport_chart_state,
    transport_chart_rhs,
    transport_chart_specific_force_thrust_n,
    transport_chart_speed_command_thrust_n,
)


# The first actuator is the control law's longitudinal command (thrust fraction under
# THRUST_FRACTION_LAW, specific force under SpecificForceLaw, the speed command relative to
# the anchor under SpeedCommandLaw); the name is the default law's.
LAG_ACTUATOR_NAMES = ("thrust_fraction", "bank_rad", "load_factor")
LAG_STATE_NAMES = (*TRANSPORT_CHART_STATE_NAMES, *LAG_ACTUATOR_NAMES)
CHART_WIDTH = len(TRANSPORT_CHART_STATE_NAMES)
# Actuator states are dimensionless, radians or (speed-command) m/s; a fixed-step RK4 is
# invariant to a linear rescaling of a state, so only the chart half is rescaled.
UNIT_ACTUATOR_SCALE = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ThrustFractionLaw:
    """The first actuator is thrust as a fraction of installed thrust, ``T = a_x · T_max``."""


@dataclass(frozen=True)
class SpecificForceLaw:
    """The first actuator is the specific force along the path, ``a_x = (T - D)/W``.

    The thrust that flies it is recomputed at every RHS evaluation — every RK4 stage — as
    ``clamp(W·a_x + D, min_thrust_fraction·T_max, T_max)``: a thrust held across a stage
    would not hold the specific force, because the drag moves with the state.
    ``min_thrust_fraction`` is the engine floor in the thrust-fraction law's own units, so a
    caller that passes that law's box floor makes the two laws admit the same thrusts.
    """

    min_thrust_fraction: float

    def __post_init__(self) -> None:
        # 1.0 is the ceiling by definition: the fraction is of the INSTALLED thrust.
        if not math.isfinite(self.min_thrust_fraction) or self.min_thrust_fraction >= 1.0:
            raise ValueError(
                "min_thrust_fraction must be finite and below the installed thrust (1.0), "
                f"got {self.min_thrust_fraction!r}"
            )


@dataclass(frozen=True)
class SpeedCommandLaw:
    """The first actuator is a speed command RELATIVE to the anchor's airspeed, ``a_Δ`` m/s,
    flown by a first-order speed loop (``ts_transformer/docs/
    2026-09-14_specific_force_control_design.md`` §12).

    At every RHS evaluation the loop asks for the specific force
    ``sin γ + (V₀ + a_Δ − V)/(g·τ_V)`` and the thrust that flies it is re-solved exactly as
    under :class:`SpecificForceLaw` — the same clamp, the same drag — so wherever the clamp
    does not bind ``V' = (V₀ + a_Δ − V)/τ_V`` on every airframe: the specific force's
    invariance, plus the restoring force ``∂V'/∂V = −1/τ_V`` that law lacks. ``V₀`` is each
    flight's anchor airspeed, read off the rollout's initial state.
    """

    min_thrust_fraction: float
    speed_time_constant_s: float

    def __post_init__(self) -> None:
        SpecificForceLaw(self.min_thrust_fraction)   # the same engine-floor contract
        if not math.isfinite(self.speed_time_constant_s) or self.speed_time_constant_s <= 0.0:
            raise ValueError(
                "speed_time_constant_s must be finite and positive, "
                f"got {self.speed_time_constant_s!r}"
            )


THRUST_FRACTION_LAW = ThrustFractionLaw()
LagControlLaw = ThrustFractionLaw | SpecificForceLaw | SpeedCommandLaw


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
    """Return the realised ``(a_x, bank, load)`` along the rollout — ``a_x`` in the
    control law's own unit (thrust fraction, specific force, or the speed command relative
    to the anchor's airspeed)."""
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    return (state_lag * state_scale)[..., CHART_WIDTH:]


def _require_lag_inputs(
    state_lag: torch.Tensor, commands: torch.Tensor, time_constants_s: torch.Tensor
) -> None:
    _require_last_dim(state_lag, len(LAG_STATE_NAMES), "state_lag")
    _require_last_dim(commands, len(CONTROL_NAMES), "commands")
    _require_last_dim(time_constants_s, len(CONTROL_NAMES), "time_constants_s")


def _physical_split(
    state_lag: torch.Tensor, state_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(chart, actuators)`` of the augmented state, in physical units."""
    physical = state_lag * state_scale
    return physical[..., :CHART_WIDTH], physical[..., CHART_WIDTH:]


def _lag_rate(
    chart: torch.Tensor,
    actual: torch.Tensor,
    thrust_n: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """The chart RHS flown at ``(thrust_n, bank, load)``, plus the actuators' ODE."""
    physical_controls = torch.cat((thrust_n.unsqueeze(-1), actual[..., 1:]), dim=-1)
    rate = torch.cat(
        (
            transport_chart_rhs(chart, physical_controls, aero_params, frame_params),
            (commands - actual) / time_constants_s,
        ),
        dim=-1,
    )
    return rate / state_scale


def lag_rhs(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """Continuous RHS of the chart state plus three first-order actuators — the
    thrust-fraction law, ``T = a_x · T_max``."""
    _require_lag_inputs(state_lag, commands, time_constants_s)
    chart, actual = _physical_split(state_lag, state_scale)
    return _lag_rate(
        chart, actual, actual[..., 0] * max_thrust_n, commands, aero_params,
        frame_params, time_constants_s, state_scale,
    )


def lag_rhs_specific_force(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    min_thrust_n: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """The same RHS under :class:`SpecificForceLaw`: the thrust that flies the actuator's
    specific force at THIS state, clamped to ``[min_thrust_n, max_thrust_n]``."""
    _require_lag_inputs(state_lag, commands, time_constants_s)
    chart, actual = _physical_split(state_lag, state_scale)
    thrust_n = transport_chart_specific_force_thrust_n(
        chart, actual[..., 0], actual[..., 2], aero_params, frame_params,
        min_thrust_n, max_thrust_n,
    )
    return _lag_rate(
        chart, actual, thrust_n, commands, aero_params, frame_params,
        time_constants_s, state_scale,
    )


def lag_rhs_speed_command(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    reference_speed_mps: torch.Tensor,
    speed_time_constant_s: torch.Tensor,
    min_thrust_n: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """The same RHS under :class:`SpeedCommandLaw`: the thrust the speed loop asks for at
    THIS state toward ``reference_speed_mps + a_Δ``, clamped to ``[min_thrust_n,
    max_thrust_n]``."""
    _require_lag_inputs(state_lag, commands, time_constants_s)
    chart, actual = _physical_split(state_lag, state_scale)
    thrust_n = transport_chart_speed_command_thrust_n(
        chart, reference_speed_mps + actual[..., 0], speed_time_constant_s, actual[..., 2],
        aero_params, frame_params, min_thrust_n, max_thrust_n,
    )
    return _lag_rate(
        chart, actual, thrust_n, commands, aero_params, frame_params,
        time_constants_s, state_scale,
    )


def _rk4(rhs, state: torch.Tensor, dt_s: torch.Tensor | float) -> torch.Tensor:
    """One explicit RK4 step of ``rhs`` from ``state``."""
    dt = torch.as_tensor(dt_s, dtype=state.dtype, device=state.device)
    while dt.ndim < state.ndim:
        dt = dt.unsqueeze(-1)
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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
    return _rk4(
        lambda state: lag_rhs(
            state, commands, aero_params, frame_params, time_constants_s,
            max_thrust_n, state_scale,
        ),
        state_lag,
        dt_s,
    )


def rk4_lag_step_specific_force(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    min_thrust_n: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """:func:`rk4_lag_step` under :class:`SpecificForceLaw` — the thrust is re-solved at
    each of the four stages, which is what holds the specific force across the step."""
    return _rk4(
        lambda state: lag_rhs_specific_force(
            state, commands, aero_params, frame_params, time_constants_s,
            min_thrust_n, max_thrust_n, state_scale,
        ),
        state_lag,
        dt_s,
    )


def rk4_lag_step_speed_command(
    state_lag: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor | float,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    reference_speed_mps: torch.Tensor,
    speed_time_constant_s: torch.Tensor,
    min_thrust_n: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """:func:`rk4_lag_step` under :class:`SpeedCommandLaw` — the loop's thrust is re-solved
    at each of the four stages, as under the specific-force law."""
    return _rk4(
        lambda state: lag_rhs_speed_command(
            state, commands, aero_params, frame_params, time_constants_s,
            reference_speed_mps, speed_time_constant_s, min_thrust_n, max_thrust_n,
            state_scale,
        ),
        state_lag,
        dt_s,
    )


# The rollout engine hands one ``step_context`` tensor to the step function, so every
# per-run constant travels in it, in this fixed layout per control law:
#
#     thrust-fraction: [ frame (4) | time_constants_s (3) | max_thrust_n (1) | state_scale (10) ]
#     specific-force:  [ frame (4) | time_constants_s (3) | max_thrust_n (1) | min_thrust_n (1)
#                        | state_scale (10) ]
#     speed-command:   [ frame (4) | time_constants_s (3) | max_thrust_n (1) | min_thrust_n (1)
#                        | reference_speed_mps (1) | speed_time_constant_s (1) | state_scale (10) ]
#
# state_scale rides along rather than being captured in a closure specifically so the step
# stays ONE module-level function per law. A closure would be a new code object per rollout,
# and torch.compile caches per code object — the compiled step would be rebuilt every batch
# and would exhaust Dynamo's recompile budget instead of paying off. The same reason gives
# each law its own step functions and its own compiled cache below.
_FRAME_WIDTH = 4
_TAU_WIDTH = len(CONTROL_NAMES)
_THRUST_AT = _FRAME_WIDTH + _TAU_WIDTH
_SCALE_AT = _THRUST_AT + 1
_MIN_THRUST_AT = _THRUST_AT + 1
_SPECIFIC_FORCE_SCALE_AT = _MIN_THRUST_AT + 1
_REFERENCE_SPEED_AT = _MIN_THRUST_AT + 1
_SPEED_TAU_AT = _REFERENCE_SPEED_AT + 1
_SPEED_COMMAND_SCALE_AT = _SPEED_TAU_AT + 1


def _pack_context(
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    row_values: tuple[torch.Tensor, ...],
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """The layout above; ``row_values`` is ``(max,)``, ``(max, min)`` or ``(max, min,
    reference speed, τ_V)`` by law — one column each, per flight (a scalar broadcasts)."""
    rows = len(frame_params)
    return torch.cat(
        (
            frame_params,
            time_constants_s.reshape(1, -1).expand(rows, -1).to(frame_params.dtype),
            *(
                torch.as_tensor(value).to(frame_params).reshape(-1, 1).expand(rows, 1)
                for value in row_values
            ),
            state_scale.reshape(1, -1).expand(rows, -1).to(frame_params.dtype),
        ),
        dim=-1,
    )


def _unpacked_step(
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
        step_context[..., _FRAME_WIDTH:_THRUST_AT],
        step_context[..., _THRUST_AT],
        step_context[0, _SCALE_AT:],
    )


def _unpacked_step_specific_force(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    return rk4_lag_step_specific_force(
        state,
        commands,
        aero_params,
        dt_s,
        step_context[..., :_FRAME_WIDTH],
        step_context[..., _FRAME_WIDTH:_THRUST_AT],
        step_context[..., _MIN_THRUST_AT],
        step_context[..., _THRUST_AT],
        step_context[0, _SPECIFIC_FORCE_SCALE_AT:],
    )


def _unpacked_step_speed_command(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    return rk4_lag_step_speed_command(
        state,
        commands,
        aero_params,
        dt_s,
        step_context[..., :_FRAME_WIDTH],
        step_context[..., _FRAME_WIDTH:_THRUST_AT],
        step_context[..., _REFERENCE_SPEED_AT],
        step_context[..., _SPEED_TAU_AT],
        step_context[..., _MIN_THRUST_AT],
        step_context[..., _THRUST_AT],
        step_context[0, _SPEED_COMMAND_SCALE_AT:],
    )


def _cuda_inference_step(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """Distinct code object so no-grad shape caches do not consume VJP entries."""
    return _unpacked_step(state, commands, aero_params, dt_s, step_context)


def _cuda_autograd_step(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """Distinct code object for the grad-enabled local discrete-adjoint step."""
    return _unpacked_step(state, commands, aero_params, dt_s, step_context)


def _cuda_inference_step_specific_force(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The specific-force law's no-grad code object (its own compile cache)."""
    return _unpacked_step_specific_force(state, commands, aero_params, dt_s, step_context)


def _cuda_autograd_step_specific_force(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The specific-force law's grad-enabled code object (its own compile cache)."""
    return _unpacked_step_specific_force(state, commands, aero_params, dt_s, step_context)


def _cuda_inference_step_speed_command(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The speed-command law's no-grad code object (its own compile cache)."""
    return _unpacked_step_speed_command(state, commands, aero_params, dt_s, step_context)


def _cuda_autograd_step_speed_command(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The speed-command law's grad-enabled code object (its own compile cache)."""
    return _unpacked_step_speed_command(state, commands, aero_params, dt_s, step_context)


#: One compiled kernel per (step function) — i.e. per law and per grad mode — built on
#: first use on CUDA and reused for the life of the process.
_COMPILED_CUDA_STEPS: dict = {}


def _dispatch_step(
    eager, cuda_inference, cuda_autograd, state, commands, aero_params, dt_s, step_context
) -> torch.Tensor:
    """Eager on CPU, a fused Inductor graph on CUDA — as the point-mass backends do.

    Measured before this: a lagged epoch cost ~95 s against the point-mass backend's ~15 s
    on the same KSJC fold, entirely because this step ran eager while the others cached a
    compiled one.
    """
    if not state.is_cuda:
        return eager(state, commands, aero_params, dt_s, step_context)
    grad_enabled = torch.is_grad_enabled()
    source = cuda_autograd if grad_enabled else cuda_inference
    compiled = _COMPILED_CUDA_STEPS.get(source)
    if compiled is None:
        # ``dynamic=True`` in backward reproducibly segfaults on this Torch/CUDA stack
        # (see torch_dynamics), so the autograd kernel stays static.
        compiled = torch.compile(
            source,
            fullgraph=True,
            dynamic=not grad_enabled,
            mode="reduce-overhead",
        )
        _COMPILED_CUDA_STEPS[source] = compiled
    return compiled(state, commands, aero_params, dt_s, step_context)


def _rollout_step(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The thrust-fraction law's step, as the engines call it."""
    return _dispatch_step(
        _unpacked_step, _cuda_inference_step, _cuda_autograd_step,
        state, commands, aero_params, dt_s, step_context,
    )


def _rollout_step_specific_force(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The specific-force law's step, as the engines call it."""
    return _dispatch_step(
        _unpacked_step_specific_force,
        _cuda_inference_step_specific_force,
        _cuda_autograd_step_specific_force,
        state, commands, aero_params, dt_s, step_context,
    )


def _rollout_step_speed_command(
    state: torch.Tensor,
    commands: torch.Tensor,
    aero_params: torch.Tensor,
    dt_s: torch.Tensor,
    step_context: torch.Tensor,
) -> torch.Tensor:
    """The speed-command law's step, as the engines call it."""
    return _dispatch_step(
        _unpacked_step_speed_command,
        _cuda_inference_step_speed_command,
        _cuda_autograd_step_speed_command,
        state, commands, aero_params, dt_s, step_context,
    )


def _law_step(
    control_law: LagControlLaw,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    state_scale: torch.Tensor,
    initial_geodetic_states: torch.Tensor,
):
    """``(step function, step context)`` for a control law — the one dispatch on it. The
    speed-command law's reference speed is each flight's anchor airspeed, the initial
    geodetic state's ``V``."""
    if isinstance(control_law, SpeedCommandLaw):
        values = (
            max_thrust_n,
            control_law.min_thrust_fraction * max_thrust_n,
            initial_geodetic_states[..., STATE_NAMES.index("V")],
            control_law.speed_time_constant_s,
        )
        return _rollout_step_speed_command, _pack_context(
            frame_params, time_constants_s, values, state_scale
        )
    if isinstance(control_law, SpecificForceLaw):
        limits = (max_thrust_n, control_law.min_thrust_fraction * max_thrust_n)
        return _rollout_step_specific_force, _pack_context(
            frame_params, time_constants_s, limits, state_scale
        )
    if isinstance(control_law, ThrustFractionLaw):
        return _rollout_step, _pack_context(
            frame_params, time_constants_s, (max_thrust_n,), state_scale
        )
    raise TypeError(f"unknown lag control law {control_law!r}")


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
    control_law: LagControlLaw = THRUST_FRACTION_LAW,
    chart_scale: tuple[float, ...] | None = None,
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> torch.Tensor:
    """Return augmented segment-end states ``[B,N,10]``."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    step, context = _law_step(
        control_law, frame_params, time_constants_s, max_thrust_n, state_scale,
        initial_geodetic_states,
    )
    return rollout_piecewise_constant_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        context,
        step,
        integrator_dt_s=integrator_dt_s,
        max_steps_per_segment=max_steps_per_segment,
    )


def rollout_piecewise_constant_hooked(
    initial_geodetic_states: torch.Tensor,
    initial_controls: torch.Tensor,
    commands: torch.Tensor,
    segment_durations_s: torch.Tensor,
    aero_params: torch.Tensor,
    frame_params: torch.Tensor,
    time_constants_s: torch.Tensor,
    max_thrust_n: torch.Tensor,
    command_hook: RawCommandHook,
    *,
    control_law: LagControlLaw = THRUST_FRACTION_LAW,
    track_reference: bool = False,
    chart_scale: tuple[float, ...] | None = None,
    integrator_dt_s: float = 0.5,
    max_steps_per_segment: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmented segment-end states ``[B,N,10]`` and the effective commands ``[B,N,3]``
    when a hook rewrites each segment's command from the (scaled, augmented) state."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    step, context = _law_step(
        control_law, frame_params, time_constants_s, max_thrust_n, state_scale,
        initial_geodetic_states,
    )
    return rollout_piecewise_constant_hooked_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        context,
        step,
        command_hook,
        track_reference=track_reference,
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
    control_law: LagControlLaw = THRUST_FRACTION_LAW,
    chart_scale: tuple[float, ...] | None = None,
    segment_valid: torch.Tensor | None = None,
    integrator_dt_s: float = 0.5,
    max_total_steps: int = 65536,
) -> DenseControlRollout:
    """Return event-aligned augmented states at queries and control boundaries."""
    state_scale = lag_state_scale(chart_scale, frame_params)
    step, context = _law_step(
        control_law, frame_params, time_constants_s, max_thrust_n, state_scale,
        initial_geodetic_states,
    )
    return rollout_piecewise_constant_at_times_with_step(
        lag_state_from_geodetic(
            initial_geodetic_states, initial_controls, frame_params, state_scale
        ),
        commands,
        segment_durations_s,
        aero_params,
        context,
        step,
        query_offsets_s,
        query_valid,
        segment_valid=segment_valid,
        integrator_dt_s=integrator_dt_s,
        max_total_steps=max_total_steps,
    )
