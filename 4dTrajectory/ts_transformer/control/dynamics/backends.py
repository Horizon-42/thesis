"""Registry of the flight models a learned control schedule can be rolled through.

Two independent choices meet here and are kept independent:

* ``control_dynamics_model`` — the physics. ``point-mass`` applies each piecewise-constant
  control instantly; ``first-order-lag`` makes the three controls states that chase their
  command, reusing the same force equations.
* ``control_dynamics_backend`` — the state representation the long rollout carries.
  Re-anchored local ENU, the continuous WGS84 transport chart, or that chart in
  nondimensional coordinates.

Training, validation and forecasting consume one channel/geodetic result contract, so a
representation change never reaches the model, the loss or the data pipeline. Controls
arrive in the dimensionless envelope (``control_envelope``); conversion to the newton
contract ``aerodynamic_model`` expects happens once, here, at the boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from control.dynamics.hooks import CommandHook, RolloutStateView

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
    rollout_piecewise_constant as transport_endpoint_rollout,
    rollout_piecewise_constant_at_times as transport_dense_rollout,
    transport_chart_state_to_channels,
    transport_chart_state_to_geodetic,
)
from aerodynamic_model.torch_scaled_transport_chart_dynamics import (
    SCALED_TRANSPORT_CHART_REFERENCE_UNITS,
    rollout_piecewise_constant as scaled_transport_endpoint_rollout,
    rollout_piecewise_constant_at_times as scaled_transport_dense_rollout,
    scaled_to_physical_transport_chart_state,
)
from config import (
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    TSConfig,
)
from control.envelope import physical_controls


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


@dataclass(frozen=True)
class DenseControlRolloutChannels:
    query_channels: torch.Tensor
    segment_end_channels: torch.Tensor
    query_geodetic_states: torch.Tensor
    segment_end_geodetic_states: torch.Tensor
    controls: torch.Tensor


class ControlDynamicsBackend(ABC):
    @abstractmethod
    def endpoint_rollout(
        self,
        inputs: RolloutInputs,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> EndpointControlRollout:
        """Roll learned segments and expose the shared public representations."""

    @abstractmethod
    def dense_rollout(
        self,
        inputs: RolloutInputs,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
        command_hook: CommandHook | None = None,
    ) -> DenseControlRolloutChannels:
        """Roll once and expose channel states at queries and segment boundaries."""


def _refuse_hook(backend: ControlDynamicsBackend, command_hook: CommandHook | None) -> None:
    if command_hook is not None:
        raise NotImplementedError(
            f"{type(backend).__name__} does not run command hooks; the first-order-lag "
            "backends do (their state carries the chart and the actuators a hook reads)"
        )


class ReanchoredRK4Backend(ControlDynamicsBackend):
    """Local ENU RK4 re-anchored into geodetic state every substep; casadi's twin."""

    def endpoint_rollout(
        self,
        inputs: RolloutInputs,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> EndpointControlRollout:
        _refuse_hook(self, command_hook)
        geodetic = reanchored_endpoint_rollout(
            inputs.initial_state,
            inputs.newton_controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return EndpointControlRollout(
            geodetic_states_to_channels(
                geodetic,
                inputs.frame_params,
                runway_aligned=config.coordinate_frame == "runway-aligned",
            ),
            geodetic,
            inputs.controls,
        )

    def dense_rollout(
        self,
        inputs: RolloutInputs,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
        command_hook: CommandHook | None = None,
    ) -> DenseControlRolloutChannels:
        _refuse_hook(self, command_hook)
        rollout = reanchored_dense_rollout(
            inputs.initial_state,
            inputs.newton_controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            query_offsets_s,
            query_valid,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        runway_aligned = config.coordinate_frame == "runway-aligned"
        return DenseControlRolloutChannels(
            geodetic_states_to_channels(
                rollout.query_states, inputs.frame_params, runway_aligned=runway_aligned
            ),
            geodetic_states_to_channels(
                rollout.segment_end_states,
                inputs.frame_params,
                runway_aligned=runway_aligned,
            ),
            rollout.query_states,
            rollout.segment_end_states,
            inputs.controls,
        )


class _TransportChartResults:
    """Shared conversion from a physical transport-chart state to the public contract."""

    @staticmethod
    def endpoint(
        chart_states: torch.Tensor,
        inputs: RolloutInputs,
        config: TSConfig,
        controls: torch.Tensor,
    ) -> EndpointControlRollout:
        return EndpointControlRollout(
            transport_chart_state_to_channels(
                chart_states,
                inputs.frame_params,
                runway_aligned=config.coordinate_frame == "runway-aligned",
            ),
            transport_chart_state_to_geodetic(chart_states, inputs.frame_params),
            controls,
        )

    @staticmethod
    def dense(
        query_states: torch.Tensor,
        endpoint_states: torch.Tensor,
        inputs: RolloutInputs,
        config: TSConfig,
        controls: torch.Tensor,
    ) -> DenseControlRolloutChannels:
        runway_aligned = config.coordinate_frame == "runway-aligned"
        return DenseControlRolloutChannels(
            transport_chart_state_to_channels(
                query_states, inputs.frame_params, runway_aligned=runway_aligned
            ),
            transport_chart_state_to_channels(
                endpoint_states, inputs.frame_params, runway_aligned=runway_aligned
            ),
            transport_chart_state_to_geodetic(query_states, inputs.frame_params),
            transport_chart_state_to_geodetic(endpoint_states, inputs.frame_params),
            controls,
        )


class TransportChartVelocityBackend(ControlDynamicsBackend):
    """Continuous WGS84 chart position with physical local-ENU velocity state."""

    def endpoint_rollout(
        self,
        inputs: RolloutInputs,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> EndpointControlRollout:
        _refuse_hook(self, command_hook)
        return _TransportChartResults.endpoint(
            transport_endpoint_rollout(
                inputs.initial_state,
                inputs.newton_controls,
                inputs.segment_durations_s,
                inputs.aero_params,
                inputs.frame_params,
                integrator_dt_s=config.control_rollout_integrator_dt_s,
            ),
            inputs,
            config,
            inputs.controls,
        )

    def dense_rollout(
        self,
        inputs: RolloutInputs,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
        command_hook: CommandHook | None = None,
    ) -> DenseControlRolloutChannels:
        _refuse_hook(self, command_hook)
        rollout = transport_dense_rollout(
            inputs.initial_state,
            inputs.newton_controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            inputs.frame_params,
            query_offsets_s,
            query_valid,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return _TransportChartResults.dense(
            rollout.query_states, rollout.segment_end_states, inputs, config, inputs.controls
        )


class ScaledTransportChartVelocityBackend(ControlDynamicsBackend):
    """Order-one internal state with the existing physical public contract."""

    def endpoint_rollout(
        self,
        inputs: RolloutInputs,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> EndpointControlRollout:
        _refuse_hook(self, command_hook)
        scaled = scaled_transport_endpoint_rollout(
            inputs.initial_state,
            inputs.newton_controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            inputs.frame_params,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return _TransportChartResults.endpoint(
            scaled_to_physical_transport_chart_state(scaled), inputs, config, inputs.controls
        )

    def dense_rollout(
        self,
        inputs: RolloutInputs,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
        command_hook: CommandHook | None = None,
    ) -> DenseControlRolloutChannels:
        _refuse_hook(self, command_hook)
        rollout = scaled_transport_dense_rollout(
            inputs.initial_state,
            inputs.newton_controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            inputs.frame_params,
            query_offsets_s,
            query_valid,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return _TransportChartResults.dense(
            scaled_to_physical_transport_chart_state(rollout.query_states),
            scaled_to_physical_transport_chart_state(rollout.segment_end_states),
            inputs,
            config,
            inputs.controls,
        )


class FirstOrderLagBackend(ControlDynamicsBackend):
    """Transport-chart dynamics whose three controls are lagged states.

    The predicted schedule becomes a COMMAND schedule; the state carries what the aircraft
    is actually doing, starting from ``inputs.initial_controls`` — the controls the
    observed lookback implies at the anchor. ``chart_scale`` selects whether the
    point-mass half of the state is carried physically or nondimensionally, so the lag is
    orthogonal to that choice rather than a fourth backend.
    """

    def __init__(self, chart_scale: tuple[float, ...] | None):
        self.chart_scale = chart_scale

    def _to_chart(self, states: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return lag_state_to_transport_chart(
            states, lag_state_scale(self.chart_scale, reference)
        )

    def _hooked_schedule(
        self, inputs: RolloutInputs, config: TSConfig, command_hook: CommandHook
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment-end augmented states and the effective commands under the hook."""
        state_scale = lag_state_scale(self.chart_scale, inputs.frame_params)

        def raw_hook(
            state: torch.Tensor, command: torch.Tensor, duration_s: torch.Tensor, segment: int
        ) -> torch.Tensor:
            view = RolloutStateView(
                chart=lag_state_to_transport_chart(state, state_scale),
                actuators=lag_actuator_states(state, state_scale),
                duration_s=duration_s,
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
            chart_scale=self.chart_scale,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )

    def endpoint_rollout(
        self,
        inputs: RolloutInputs,
        config: TSConfig,
        *,
        command_hook: CommandHook | None = None,
    ) -> EndpointControlRollout:
        if command_hook is not None:
            states, effective = self._hooked_schedule(inputs, config, command_hook)
            return _TransportChartResults.endpoint(
                self._to_chart(states, inputs.frame_params), inputs, config, effective
            )
        states = lag_endpoint_rollout(
            inputs.initial_state,
            inputs.initial_controls,
            inputs.controls,
            inputs.segment_durations_s,
            inputs.aero_params,
            inputs.frame_params,
            inputs.frame_params.new_tensor(config.control_time_constants_s),
            inputs.max_thrust_n,
            chart_scale=self.chart_scale,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return _TransportChartResults.endpoint(
            self._to_chart(states, inputs.frame_params), inputs, config, inputs.controls
        )

    def dense_rollout(
        self,
        inputs: RolloutInputs,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
        command_hook: CommandHook | None = None,
    ) -> DenseControlRolloutChannels:
        # A hook decides each segment's command from the state at its start; the dense
        # engine's adjoint takes the schedule as an input, so the hooked schedule is
        # settled first (one segmented rollout) and then integrated densely as-is.
        commands = inputs.controls
        if command_hook is not None:
            _states, commands = self._hooked_schedule(inputs, config, command_hook)
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
            chart_scale=self.chart_scale,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return _TransportChartResults.dense(
            self._to_chart(rollout.query_states, inputs.frame_params),
            self._to_chart(rollout.segment_end_states, inputs.frame_params),
            inputs,
            config,
            commands,
        )


_BACKENDS: dict[tuple[str, str], ControlDynamicsBackend] = {
    (CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_REANCHORED_RK4): (
        ReanchoredRK4Backend()
    ),
    (CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY): (
        TransportChartVelocityBackend()
    ),
    (CONTROL_DYNAMICS_POINT_MASS, CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY): (
        ScaledTransportChartVelocityBackend()
    ),
    (CONTROL_DYNAMICS_FIRST_ORDER_LAG, CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY): (
        FirstOrderLagBackend(None)
    ),
    (
        CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    ): FirstOrderLagBackend(SCALED_TRANSPORT_CHART_REFERENCE_UNITS),
}


def control_dynamics_backend(config: TSConfig) -> ControlDynamicsBackend:
    """Resolve a validated serialized model/representation pair without branching."""
    return _BACKENDS[(config.control_dynamics_model, config.control_dynamics_backend)]
