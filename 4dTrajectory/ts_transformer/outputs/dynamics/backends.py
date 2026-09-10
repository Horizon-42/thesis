"""Registry of the flight models a learned control schedule can be rolled through.

Two independent choices meet here and are kept independent:

* ``control_dynamics_model`` — the physics. ``point-mass`` applies each piecewise-constant
  control instantly; ``first-order-lag`` makes the three controls states that chase their
  command, reusing the same force equations.
* ``control_dynamics_backend`` — the state representation the long rollout carries.
  Re-anchored local ENU, or the continuous WGS84 transport chart in nondimensional
  coordinates. (The chart in PHYSICAL coordinates was a third representation until
  2026-09-07; it was a measured regression and is retired — see ``config.py``.)

A backend is a ROW, not a class: ``(endpoint_fn, dense_fn, post_fn)``. The first two roll
the schedule and hand back whatever state their integrator carries, ALREADY converted to
the representation ``post_fn`` reads — geodetic ``[B,N,7]`` for the re-anchored row, a
PHYSICAL transport-chart state for the other two (the scaled row rescales, the lagged row
drops its actuator block). ``post_fn`` turns that into the one public pair every consumer
reads, ``(channels, geodetic)``. Two rows share `_chart_post` and two share their hook
policy, which is the whole reason for the table: the three implementations differed in six
lines each and agreed on the rest.

Training, validation and forecasting consume one channel/geodetic result contract, so a
representation change never reaches the model, the loss or the data pipeline. Controls
arrive in the dimensionless envelope (``control_envelope``); conversion to the newton
contract ``aerodynamic_model`` expects happens once, here, at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

from ts_transformer.outputs.dynamics.hooks import CommandHook, RolloutStateView

from aerodynamic_model.torch_dense_rollout import (
    rollout_piecewise_constant_at_times as reanchored_dense_rollout,
)
from aerodynamic_model.torch_dynamics import (
    geodetic_states_to_channels,
    rollout_piecewise_constant as reanchored_endpoint_rollout,
)
from aerodynamic_model.torch_lag_dynamics import (
    lag_actuator_states,
    lag_state_scale,
    lag_state_to_transport_chart,
    rollout_piecewise_constant as lag_endpoint_rollout,
    rollout_piecewise_constant_at_times as lag_dense_rollout,
    rollout_piecewise_constant_hooked as lag_hooked_rollout,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    transport_chart_state_to_channels,
    transport_chart_state_to_geodetic,
)
from aerodynamic_model.torch_scaled_transport_chart_dynamics import (
    SCALED_TRANSPORT_CHART_REFERENCE_UNITS,
    rollout_piecewise_constant as scaled_transport_endpoint_rollout,
    rollout_piecewise_constant_at_times as scaled_transport_dense_rollout,
    scaled_to_physical_transport_chart_state,
)
from ts_transformer.config import (
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    TSConfig,
)
from ts_transformer.outputs.envelope import physical_controls


@dataclass(frozen=True)
class RolloutInputs:
    """Everything a rollout needs, already on one dtype and device."""

    initial_state: torch.Tensor       # [B,7] geodetic
    initial_controls: torch.Tensor    # [B,3] envelope units, in effect at the anchor
    controls: torch.Tensor            # [B,N,3] envelope units (commands, if lagged)
    segment_durations_s: torch.Tensor  # [B,N]
    aero_params: torch.Tensor         # [B,6]
    frame_params: torch.Tensor        # [B,4]
    max_thrust_n: torch.Tensor        # [B]

    @property
    def newton_controls(self) -> torch.Tensor:
        """The schedule in the newton contract ``aerodynamic_model`` integrates."""
        return physical_controls(self.controls, self.max_thrust_n)


@dataclass(frozen=True)
class EndpointControlRollout:
    channels: torch.Tensor
    geodetic_states: torch.Tensor
    # The schedule actually flown, envelope units: the network's commands, or what a
    # command hook made of them. Records and downstream losses read THIS.
    controls: torch.Tensor
    # ``[B,N,3]``: what the aircraft was actually doing AT each segment end, envelope
    # units. Under the point-mass model a piecewise-constant command is in effect
    # instantly, so this IS ``controls``; under the first-order lag the three controls are
    # states chasing their command and this is where they got to. A term that prices what
    # the rollout flew (the heading-rate loss) must read this, never the command.
    actual_controls: torch.Tensor


@dataclass(frozen=True)
class DenseControlRolloutChannels:
    query_channels: torch.Tensor
    segment_end_channels: torch.Tensor
    query_geodetic_states: torch.Tensor
    segment_end_geodetic_states: torch.Tensor
    controls: torch.Tensor


#: ``(states, controls_flown, actual_controls)`` in the backend's own representation.
EndpointFn = Callable[[RolloutInputs, TSConfig, CommandHook | None], tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]]
#: ``(query_states, segment_end_states, controls_flown)``.
DenseFn = Callable[
    [RolloutInputs, torch.Tensor, torch.Tensor, TSConfig, CommandHook | None],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]
#: One backend state ``->`` the public ``(channels, geodetic)`` pair.
PostFn = Callable[[torch.Tensor, RolloutInputs, TSConfig], tuple[torch.Tensor, torch.Tensor]]


def _reads_command_hook(function):
    """Mark a rollout function that actually looks at ``command_hook``.

    ``runs_hooks=True`` on a row whose functions ``del command_hook`` would advertise a
    hook and silently fly the unhooked schedule — a wrong trajectory with no error. The
    assertion under ``_BACKENDS`` requires the flag and the mark to agree, in both
    directions, so the claim cannot drift from the code.
    """
    function.reads_command_hook = True
    return function


def _runway_aligned(config: TSConfig) -> bool:
    return config.coordinate_frame == "runway-aligned"


def _geodetic_post(
    states: torch.Tensor, inputs: RolloutInputs, config: TSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """The re-anchored backend already carries geodetic state; only channels are derived."""
    return (
        geodetic_states_to_channels(
            states, inputs.frame_params, runway_aligned=_runway_aligned(config)
        ),
        states,
    )


def _chart_post(
    states: torch.Tensor, inputs: RolloutInputs, config: TSConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """A PHYSICAL transport-chart state to the public contract."""
    return (
        transport_chart_state_to_channels(
            states, inputs.frame_params, runway_aligned=_runway_aligned(config)
        ),
        transport_chart_state_to_geodetic(states, inputs.frame_params),
    )


def _reanchored_endpoint(
    inputs: RolloutInputs, config: TSConfig, command_hook: CommandHook | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del command_hook
    geodetic = reanchored_endpoint_rollout(
        inputs.initial_state,
        inputs.newton_controls,
        inputs.segment_durations_s,
        inputs.aero_params,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    return geodetic, inputs.controls, inputs.controls


def _reanchored_dense(
    inputs: RolloutInputs,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    config: TSConfig,
    command_hook: CommandHook | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del command_hook
    rollout = reanchored_dense_rollout(
        inputs.initial_state,
        inputs.newton_controls,
        inputs.segment_durations_s,
        inputs.aero_params,
        query_offsets_s,
        query_valid,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    return rollout.query_states, rollout.segment_end_states, inputs.controls


def _scaled_chart_endpoint(
    inputs: RolloutInputs, config: TSConfig, command_hook: CommandHook | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del command_hook
    scaled = scaled_transport_endpoint_rollout(
        inputs.initial_state,
        inputs.newton_controls,
        inputs.segment_durations_s,
        inputs.aero_params,
        inputs.frame_params,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    return (
        scaled_to_physical_transport_chart_state(scaled),
        inputs.controls,
        inputs.controls,
    )


def _scaled_chart_dense(
    inputs: RolloutInputs,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    config: TSConfig,
    command_hook: CommandHook | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del command_hook
    rollout = scaled_transport_dense_rollout(
        inputs.initial_state,
        inputs.newton_controls,
        inputs.segment_durations_s,
        inputs.aero_params,
        inputs.frame_params,
        query_offsets_s,
        query_valid,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    return (
        scaled_to_physical_transport_chart_state(rollout.query_states),
        scaled_to_physical_transport_chart_state(rollout.segment_end_states),
        inputs.controls,
    )


def _lag_hooked_schedule(
    inputs: RolloutInputs,
    config: TSConfig,
    command_hook: CommandHook,
    chart_scale: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment-end augmented states and the effective commands under the hook."""
    state_scale = lag_state_scale(chart_scale, inputs.frame_params)
    # How much of the schedule is left at the START of each segment, this hold included:
    # the reversed cumulative sum of the durations. The whole schedule is known before the
    # first segment is integrated, so a hook that reasons about the time still ahead reads
    # it from the schedule rather than accumulating it and hoping it was called in order.
    remaining_per_segment = torch.flip(
        torch.cumsum(torch.flip(inputs.segment_durations_s, dims=(1,)), dim=1), dims=(1,)
    )

    # The hook-free reference is the SAME ``[B,N+1,S]`` tensor at every segment (the engine
    # integrates the unhooked schedule once, before the first hooked one), so its chart is
    # built once here rather than rebuilt on every call.
    reference_chart: torch.Tensor | None = None

    def raw_hook(
        state: torch.Tensor, command: torch.Tensor, duration_s: torch.Tensor, segment: int,
        reference: torch.Tensor | None,
    ) -> torch.Tensor:
        nonlocal reference_chart
        if reference is not None and reference_chart is None:
            reference_chart = lag_state_to_transport_chart(reference, state_scale)
        view = RolloutStateView(
            chart=lag_state_to_transport_chart(state, state_scale),
            actuators=lag_actuator_states(state, state_scale),
            duration_s=duration_s,
            remaining_s=remaining_per_segment[:, segment],
            reference=reference_chart,
        )
        return command_hook(view, command, segment)

    return lag_hooked_rollout(
        inputs.initial_state,
        inputs.initial_controls,
        inputs.controls,
        inputs.segment_durations_s,
        inputs.aero_params,
        inputs.frame_params,
        inputs.frame_params.new_tensor(config.control_time_constants_s),
        inputs.max_thrust_n,
        raw_hook,
        track_reference=command_hook.needs_reference,
        chart_scale=chart_scale,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )


@_reads_command_hook
def _lag_endpoint(
    inputs: RolloutInputs,
    config: TSConfig,
    command_hook: CommandHook | None,
    *,
    chart_scale: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The predicted schedule is a COMMAND schedule; the state carries what was flown.

    It starts from ``inputs.initial_controls`` — the controls the observed lookback implies
    at the anchor. ``chart_scale`` is the point-mass half's nondimensionalisation, so the
    lag stays orthogonal to the state representation rather than being one of its own.
    """
    if command_hook is not None:
        states, flown = _lag_hooked_schedule(inputs, config, command_hook, chart_scale)
    else:
        flown = inputs.controls
        states = lag_endpoint_rollout(
            inputs.initial_state,
            inputs.initial_controls,
            inputs.controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            inputs.frame_params,
            inputs.frame_params.new_tensor(config.control_time_constants_s),
            inputs.max_thrust_n,
            chart_scale=chart_scale,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
    state_scale = lag_state_scale(chart_scale, inputs.frame_params)
    return (
        lag_state_to_transport_chart(states, state_scale),
        flown,
        # The three controls the actuators had REACHED at each segment end.
        lag_actuator_states(states, state_scale),
    )


@_reads_command_hook
def _lag_dense(
    inputs: RolloutInputs,
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    config: TSConfig,
    command_hook: CommandHook | None,
    *,
    chart_scale: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A hook decides each segment's command from the state at its start; the dense
    # engine's adjoint takes the schedule as an input, so the hooked schedule is settled
    # first (one segmented rollout) and then integrated densely as-is.
    commands = inputs.controls
    if command_hook is not None:
        _states, commands = _lag_hooked_schedule(inputs, config, command_hook, chart_scale)
    rollout = lag_dense_rollout(
        inputs.initial_state,
        inputs.initial_controls,
        commands,
        inputs.segment_durations_s,
        inputs.aero_params,
        inputs.frame_params,
        inputs.frame_params.new_tensor(config.control_time_constants_s),
        inputs.max_thrust_n,
        query_offsets_s,
        query_valid,
        chart_scale=chart_scale,
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    state_scale = lag_state_scale(chart_scale, inputs.frame_params)
    return (
        lag_state_to_transport_chart(rollout.query_states, state_scale),
        lag_state_to_transport_chart(rollout.segment_end_states, state_scale),
        commands,
    )


@dataclass(frozen=True)
class ControlDynamicsBackend:
    """One ``(control_dynamics_model, control_dynamics_backend)`` pair, as three functions."""

    #: The ``(control_dynamics_model, control_dynamics_backend)`` pair this row is keyed
    #: by, quoted verbatim in the refusal below so the message names what a config says.
    key: tuple[str, str]
    endpoint_fn: EndpointFn
    dense_fn: DenseFn
    post_fn: PostFn
    #: Only a state that carries the actuators can be shown to a hook, which is the
    #: first-order-lag rollout. Anywhere else a hook would be silently ignored.
    runs_hooks: bool = False

    def _admit(self, command_hook: CommandHook | None) -> None:
        if command_hook is not None and not self.runs_hooks:
            model, backend = self.key
            raise NotImplementedError(
                f"control_dynamics_model={model!r} with control_dynamics_backend="
                f"{backend!r} does not run command hooks; the first-order-lag backends do "
                "(their state carries the chart and the actuators a hook reads)"
            )

    def endpoint_rollout(
        self,
        inputs: RolloutInputs,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> EndpointControlRollout:
        """Roll learned segments and expose the shared public representations."""
        self._admit(command_hook)
        states, controls, actual_controls = self.endpoint_fn(inputs, config, command_hook)
        channels, geodetic = self.post_fn(states, inputs, config)
        return EndpointControlRollout(channels, geodetic, controls, actual_controls)

    def dense_rollout(
        self,
        inputs: RolloutInputs,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> DenseControlRolloutChannels:
        """Roll once and expose channel states at queries and segment boundaries."""
        self._admit(command_hook)
        query_states, end_states, controls = self.dense_fn(
            inputs, query_offsets_s, query_valid, config, command_hook
        )
        query_channels, query_geodetic = self.post_fn(query_states, inputs, config)
        end_channels, end_geodetic = self.post_fn(end_states, inputs, config)
        return DenseControlRolloutChannels(
            query_channels, end_channels, query_geodetic, end_geodetic, controls
        )


_BACKENDS: dict[tuple[str, str], ControlDynamicsBackend] = {
    (CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_REANCHORED_RK4): ControlDynamicsBackend(
        # Local ENU RK4 re-anchored into geodetic state every substep; casadi's twin.
        key=(CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_REANCHORED_RK4),
        endpoint_fn=_reanchored_endpoint,
        dense_fn=_reanchored_dense,
        post_fn=_geodetic_post,
    ),
    (
        CONTROL_DYNAMICS_POINT_MASS,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ): ControlDynamicsBackend(
        # Order-one internal state with the existing physical public contract.
        key=(CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY),
        endpoint_fn=_scaled_chart_endpoint,
        dense_fn=_scaled_chart_dense,
        post_fn=_chart_post,
    ),
    (
        CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ): ControlDynamicsBackend(
        key=(
            CONTROL_DYNAMICS_FIRST_ORDER_LAG,
            CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        ),
        endpoint_fn=partial(
            _lag_endpoint, chart_scale=SCALED_TRANSPORT_CHART_REFERENCE_UNITS
        ),
        dense_fn=partial(_lag_dense, chart_scale=SCALED_TRANSPORT_CHART_REFERENCE_UNITS),
        post_fn=_chart_post,
        runs_hooks=True,
    ),
}


def _unwrap(function):
    return getattr(function, "func", function)


for _key, _row in _BACKENDS.items():
    assert _row.key == _key, f"{_row.key} is registered under {_key}"
    for _fn in (_row.endpoint_fn, _row.dense_fn):
        _reads = getattr(_unwrap(_fn), "reads_command_hook", False)
        assert _reads == _row.runs_hooks, (
            f"{_key} declares runs_hooks={_row.runs_hooks} but "
            f"{_unwrap(_fn).__name__} {'reads' if _reads else 'ignores'} command_hook"
        )


def control_dynamics_backend(config: TSConfig) -> ControlDynamicsBackend:
    """Resolve a validated serialized model/representation pair without branching."""
    return _BACKENDS[(config.control_dynamics_model, config.control_dynamics_backend)]
