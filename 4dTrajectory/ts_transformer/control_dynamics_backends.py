"""Registry of deterministic-control rollout state representations.

Training, validation and forecasting consume one channel/geodetic result contract.  Each
backend owns its internal state and boundary conversion, keeping representation choices out
of the model, loss and data pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from aerodynamic_model.torch_dense_rollout import (
    rollout_piecewise_constant_at_times as reanchored_dense_rollout,
)
from aerodynamic_model.torch_dynamics import (
    geodetic_states_to_channels,
    rollout_piecewise_constant as reanchored_endpoint_rollout,
)
from aerodynamic_model.torch_transport_chart_dynamics import (
    rollout_piecewise_constant as transport_endpoint_rollout,
    rollout_piecewise_constant_at_times as transport_dense_rollout,
    transport_chart_state_to_channels,
    transport_chart_state_to_geodetic,
)
from aerodynamic_model.torch_scaled_transport_chart_dynamics import (
    rollout_piecewise_constant as scaled_transport_endpoint_rollout,
    rollout_piecewise_constant_at_times as scaled_transport_dense_rollout,
    scaled_to_physical_transport_chart_state,
)
from config import (
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    TSConfig,
)


@dataclass(frozen=True)
class EndpointControlRollout:
    channels: torch.Tensor
    geodetic_states: torch.Tensor


@dataclass(frozen=True)
class DenseControlRolloutChannels:
    query_channels: torch.Tensor
    segment_end_channels: torch.Tensor
    query_geodetic_states: torch.Tensor
    segment_end_geodetic_states: torch.Tensor


class ControlDynamicsBackend(ABC):
    @abstractmethod
    def endpoint_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        config: TSConfig,
    ) -> EndpointControlRollout:
        """Roll learned segments and expose the shared public representations."""

    @abstractmethod
    def dense_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
    ) -> DenseControlRolloutChannels:
        """Roll once and expose channel states at queries and segment boundaries."""


class ReanchoredRK4Backend(ControlDynamicsBackend):
    def endpoint_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        config: TSConfig,
    ) -> EndpointControlRollout:
        geodetic = reanchored_endpoint_rollout(
            initial_geodetic_states,
            controls,
            segment_durations_s,
            aero_params,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        channels = geodetic_states_to_channels(
            geodetic,
            frame_params,
            runway_aligned=config.coordinate_frame == "runway-aligned",
        )
        return EndpointControlRollout(channels, geodetic)

    def dense_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
    ) -> DenseControlRolloutChannels:
        rollout = reanchored_dense_rollout(
            initial_geodetic_states,
            controls,
            segment_durations_s,
            aero_params,
            query_offsets_s,
            query_valid,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        runway_aligned = config.coordinate_frame == "runway-aligned"
        return DenseControlRolloutChannels(
            geodetic_states_to_channels(
                rollout.query_states,
                frame_params,
                runway_aligned=runway_aligned,
            ),
            geodetic_states_to_channels(
                rollout.segment_end_states,
                frame_params,
                runway_aligned=runway_aligned,
            ),
            rollout.query_states,
            rollout.segment_end_states,
        )


class TransportChartVelocityBackend(ControlDynamicsBackend):
    def endpoint_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        config: TSConfig,
    ) -> EndpointControlRollout:
        chart_states = transport_endpoint_rollout(
            initial_geodetic_states,
            controls,
            segment_durations_s,
            aero_params,
            frame_params,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        return EndpointControlRollout(
            transport_chart_state_to_channels(
                chart_states,
                frame_params,
                runway_aligned=config.coordinate_frame == "runway-aligned",
            ),
            transport_chart_state_to_geodetic(chart_states, frame_params),
        )

    def dense_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
    ) -> DenseControlRolloutChannels:
        rollout = transport_dense_rollout(
            initial_geodetic_states,
            controls,
            segment_durations_s,
            aero_params,
            frame_params,
            query_offsets_s,
            query_valid,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        runway_aligned = config.coordinate_frame == "runway-aligned"
        query_geodetic = transport_chart_state_to_geodetic(
            rollout.query_states, frame_params
        )
        endpoint_geodetic = transport_chart_state_to_geodetic(
            rollout.segment_end_states, frame_params
        )
        return DenseControlRolloutChannels(
            transport_chart_state_to_channels(
                rollout.query_states,
                frame_params,
                runway_aligned=runway_aligned,
            ),
            transport_chart_state_to_channels(
                rollout.segment_end_states,
                frame_params,
                runway_aligned=runway_aligned,
            ),
            query_geodetic,
            endpoint_geodetic,
        )


class ScaledTransportChartVelocityBackend(ControlDynamicsBackend):
    """Order-one internal state with the existing physical public contract."""

    def endpoint_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        config: TSConfig,
    ) -> EndpointControlRollout:
        scaled_states = scaled_transport_endpoint_rollout(
            initial_geodetic_states,
            controls,
            segment_durations_s,
            aero_params,
            frame_params,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        physical_states = scaled_to_physical_transport_chart_state(scaled_states)
        return EndpointControlRollout(
            transport_chart_state_to_channels(
                physical_states,
                frame_params,
                runway_aligned=config.coordinate_frame == "runway-aligned",
            ),
            transport_chart_state_to_geodetic(physical_states, frame_params),
        )

    def dense_rollout(
        self,
        initial_geodetic_states: torch.Tensor,
        controls: torch.Tensor,
        segment_durations_s: torch.Tensor,
        aero_params: torch.Tensor,
        frame_params: torch.Tensor,
        query_offsets_s: torch.Tensor,
        query_valid: torch.Tensor,
        config: TSConfig,
        *,
        segment_valid: torch.Tensor | None,
    ) -> DenseControlRolloutChannels:
        rollout = scaled_transport_dense_rollout(
            initial_geodetic_states,
            controls,
            segment_durations_s,
            aero_params,
            frame_params,
            query_offsets_s,
            query_valid,
            segment_valid=segment_valid,
            integrator_dt_s=config.control_rollout_integrator_dt_s,
        )
        query_states = scaled_to_physical_transport_chart_state(
            rollout.query_states
        )
        endpoint_states = scaled_to_physical_transport_chart_state(
            rollout.segment_end_states
        )
        runway_aligned = config.coordinate_frame == "runway-aligned"
        query_geodetic = transport_chart_state_to_geodetic(
            query_states, frame_params
        )
        endpoint_geodetic = transport_chart_state_to_geodetic(
            endpoint_states, frame_params
        )
        return DenseControlRolloutChannels(
            transport_chart_state_to_channels(
                query_states, frame_params, runway_aligned=runway_aligned
            ),
            transport_chart_state_to_channels(
                endpoint_states, frame_params, runway_aligned=runway_aligned
            ),
            query_geodetic,
            endpoint_geodetic,
        )


_BACKENDS: dict[str, ControlDynamicsBackend] = {
    CONTROL_DYNAMICS_REANCHORED_RK4: ReanchoredRK4Backend(),
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY: TransportChartVelocityBackend(),
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY: (
        ScaledTransportChartVelocityBackend()
    ),
}


def control_dynamics_backend(config: TSConfig) -> ControlDynamicsBackend:
    """Resolve a validated serialized backend choice without caller branching."""
    return _BACKENDS[config.control_dynamics_backend]
