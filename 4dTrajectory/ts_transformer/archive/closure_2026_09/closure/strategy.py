"""The closure path: a closed-form decision vector regressed on per-flight labels.

Frozen (review §5): stored checkpoints load, predict and publish; no new run selects it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.data.batch_contract import LossComponents, anchor_state
from ts_transformer.config import PREDICTION_CLOSURE, ClosureOutput, TSConfig
from ts_transformer.backbone.adapters import build_state_forecaster
from ts_transformer.outputs.base import ForecastOptions, OutputStrategy, Replay, WindowContext
from ts_transformer.outputs.closure.forecast import forecast_closure_batch
from ts_transformer.outputs.closure.model import (
    CLOSURE_LOSS_COMPONENT_NAMES,
    CONTEXT_VALID,
    ClosureOutputModel,
    closure_loss_components,
    label_context,
    load_labels,
    probe_closure_context,
    replay_batch,
)

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries, Normalizer, TrajectoryWindows
    from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
    from ts_transformer.inference.forecast import Forecast
    from ts_transformer.training.objective import ProcedureMultipliers


class ClosureContext(WindowContext):
    """The per-flight labels (never a model input): the decision vector, its validity, the
    label's path length and the runway course, from the config's labels file."""

    def __init__(self, windows: TrajectoryWindows):
        config = windows.config
        labels = load_labels(config.closure_labels_path)
        self._rows = [label_context(s, labels, config) for s in windows.series]
        self._index = windows.index
        present = sum(s.flight_id in labels.flights for s in windows.series)
        valid = sum(int(row[CONTEXT_VALID]) for row in self._rows)
        total = len(windows.series)
        # Stated by the caller under its own verbosity; refused here when nothing could
        # train (the two zeros mean different things).
        self.coverage = (present, valid, total)
        if windows.series and present == 0:
            raise ValueError(
                f"{config.closure_labels_path} carries none of these {total} flights — "
                "another cohort's labels?"
            )
        if windows.series and valid == 0:
            raise ValueError(
                f"{config.closure_labels_path} carries these {total} flights but marks "
                "every label non-canonical or above the residual cap: nothing to regress"
            )
        self.summary = (
            f"closure labels: {present} of {total} flights in the file, {valid} valid "
            f"({valid / max(total, 1):.1%} regress; the rest are in the batch, out of the loss)"
        )

    def row(self, i: int) -> dict[str, np.ndarray]:
        return self._rows[self._index[i][0]]


class ClosureStrategy(OutputStrategy):
    name = PREDICTION_CLOSURE
    view = ClosureOutput
    # Labels are fitted at the fixed anchor; TSConfig refuses random anchors for closure,
    # so the default policy only ever names the fixed-anchor population.

    def bind_windows(
        self, windows: TrajectoryWindows, *, training_input: Any | None = None
    ) -> WindowContext:
        return ClosureContext(windows)

    def build_model(self, config: TSConfig, normalizer: Normalizer | None) -> nn.Module:
        del normalizer  # the decision vector is regressed in physical units
        return ClosureOutputModel(config, build_state_forecaster(config))

    def target_contract(self, config: TSConfig) -> str:
        # The decision vector's shape IS the contract: a different knot count is a
        # different head, and a checkpoint of one must not load into the other.
        return (
            f"closure-v1-slowness{config.closure_slowness_knots}"
            f"-height{config.closure_height_knots}"
        )

    def loss_component_names(self, config: TSConfig) -> tuple[str, ...]:
        return CLOSURE_LOSS_COMPONENT_NAMES

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
        return closure_loss_components(
            prediction, normalized_anchor_state, target_states, state_weights,
            target_final_time_s, flight_weights, config, normalizer, context,
            dense_supervision, multipliers=multipliers,
        )

    def probe_context(
        self, batch_size: int, device: torch.device, config: TSConfig
    ) -> dict[str, torch.Tensor] | None:
        return probe_closure_context(batch_size, device, config)

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
        if options.project_final is not None:
            raise ValueError("the final-approach projection applies to state forecasts only")
        return forecast_closure_batch(model, series, config, normalizer, anchor, device)

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
        # Drawn, not rolled out: every decision reconstructed in numpy and sampled on the
        # target grid's fractions of its own duration (the context carries the course).
        if context is None:
            raise ValueError("closure replay requires the per-flight label context")
        anchors_physical = dataset.normalizer.decode(
            anchor_state(x, len(dataset.config.channels))
            .detach().cpu().numpy().astype(np.float64)
        )
        predicted, segment_durations_s, predicted_time_s = replay_batch(
            output, anchors_physical, context, dataset.config, dataset.config.pred_len
        )
        return Replay(
            predicted_physical=predicted,
            segment_durations_s=segment_durations_s,
            predicted_time_s=predicted_time_s,
            metric_targets=y,
            metric_weights=mask,
        )

    def record_fields(self, forecast: Forecast) -> dict[str, Any]:
        # The construction that drew the path (via-Dubins or a fallback) and whether it
        # was drawn from the flight's label (the oracle arm).
        return {
            "closureConstruction": forecast.closure_construction,
            "closureFromLabels": forecast.closure_from_labels,
        }


STRATEGY = ClosureStrategy()
