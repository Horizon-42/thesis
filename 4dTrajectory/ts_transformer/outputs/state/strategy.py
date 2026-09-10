"""The state path: channels in, channels out — the purely kinematic baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.data.batch_contract import LossComponents, anchor_state
from ts_transformer.config import PREDICTION_STATE, StateOutput, TSConfig
from ts_transformer.geometry.final_approach_geometry import final_approach_arrays, probe_final_approach
from ts_transformer.geometry.metrics import states_with_derived_velocity
from ts_transformer.backbone.adapters import build_state_forecaster
from ts_transformer.outputs.base import ForecastOptions, OutputStrategy, Replay, WindowContext
from ts_transformer.outputs.state.forecast import forecast_state
from ts_transformer.outputs.state.loss import (
    STATE_LOSS_COMPONENT_NAMES,
    STATE_TARGET_CONTRACTS,
    state_prediction_loss_components,
)
from ts_transformer.outputs.state.model import StateOutputLayer, StatePrediction
from ts_transformer.data.time_grids import numpy_inference_time_grid

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
    from ts_transformer.inference.forecast import Forecast
    from ts_transformer.training.objective import ProcedureMultipliers
    from ts_transformer.outputs.control.basis_fit import FittedTeacherTable


class FinalApproachContext(WindowContext):
    """The per-flight runway course and glidepath (never a model input) a recipe that
    bounds or penalises the corridor reads from every batch. Resolved once."""

    def __init__(self, windows: TrajectoryWindows):
        self._rows = [final_approach_arrays(s) for s in windows.series]
        self._index = windows.index

    def row(self, i: int) -> dict[str, np.ndarray]:
        return self._rows[self._index[i][0]]


class StateStrategy(OutputStrategy):
    name = PREDICTION_STATE
    view = StateOutput

    def bind_windows(
        self, windows: TrajectoryWindows, *, fitted_teacher: FittedTeacherTable | None = None
    ) -> WindowContext:
        if windows.config.uses_final_approach_context:
            return FinalApproachContext(windows)
        return WindowContext()

    def build_model(self, config: TSConfig, normalizer: Normalizer | None) -> nn.Module:
        return StateOutputLayer(build_state_forecaster(config), config, normalizer)

    def target_contract(self, config: TSConfig) -> str:
        return STATE_TARGET_CONTRACTS[config.horizon_mode]

    def loss_component_names(self, config: TSConfig) -> tuple[str, ...]:
        return STATE_LOSS_COMPONENT_NAMES

    def loss(
        self,
        prediction: Any,
        normalized_anchor_state: torch.Tensor,
        target_states: torch.Tensor,
        state_weights: torch.Tensor,
        target_final_time_s: torch.Tensor,
        flight_weights: torch.Tensor,
        config: TSConfig,
        normalizer: Normalizer,
        context: dict[str, torch.Tensor] | None,
        dense_supervision: FixedDTControlSupervision | None,
        *,
        multipliers: ProcedureMultipliers | None = None,
    ) -> LossComponents:
        return state_prediction_loss_components(
            prediction, normalized_anchor_state, target_states, state_weights,
            target_final_time_s, flight_weights, config, normalizer, context,
            multipliers=multipliers,
        )

    def probe_context(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> dict[str, torch.Tensor] | None:
        if config.uses_final_approach_context:
            return probe_final_approach(batch_size, device)
        return None

    def forecast(
        self,
        model: nn.Module,
        series: Sequence[FlightSeries],
        config: TSConfig,
        normalizer: Normalizer,
        anchor: int,
        device: torch.device,
        options: ForecastOptions,
    ) -> list[Forecast]:
        return [
            forecast_state(
                model, item, config, normalizer, anchor, device,
                options.truncate, options.project_final,
            )
            for item in series
        ]

    def replay(
        self,
        output: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        final_time_s: torch.Tensor,
        context: dict[str, torch.Tensor] | None,
        dataset: TrajectoryWindows,
    ) -> Replay:
        if not isinstance(output, StatePrediction):
            raise TypeError("state replay requires StatePrediction")
        channels = len(dataset.config.channels)
        predicted = dataset.normalizer.decode(
            output.states.detach().cpu().numpy().astype(np.float64)
        ).astype(np.float32)
        predicted_time_s = output.final_time_s.detach().cpu().numpy()
        segment_durations_s = numpy_inference_time_grid(predicted_time_s, dataset.config)[0]
        anchors = dataset.normalizer.decode(
            anchor_state(x, channels).detach().cpu().numpy().astype(np.float64)
        ).astype(np.float32)
        # The state output predicts positions + duration only; the velocities are derived
        # (the control rollout and the closure reconstruction carry exact ones).
        predicted = states_with_derived_velocity(
            anchors, predicted, segment_durations_s
        ).astype(np.float32)
        return Replay(
            predicted_physical=predicted,
            segment_durations_s=segment_durations_s,
            predicted_time_s=predicted_time_s,
            metric_targets=y,
            metric_weights=mask,
        )


STRATEGY = StateStrategy()
