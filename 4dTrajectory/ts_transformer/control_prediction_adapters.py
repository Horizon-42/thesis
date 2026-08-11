"""Adapters shared by inference/evaluation code for control-output strategies."""

from __future__ import annotations

from collections.abc import Callable

import torch

from control_mixture import ControlMixturePrediction
from prediction_outputs import ControlPrediction


ControlStrategyPrediction = ControlPrediction | ControlMixturePrediction


def deployable_control_prediction(
    prediction: ControlStrategyPrediction,
) -> ControlPrediction:
    """Return the history-only deployable choice for any control strategy."""
    adapters: dict[type, Callable[[ControlStrategyPrediction], ControlPrediction]] = {
        ControlPrediction: lambda value: value,
        ControlMixturePrediction: lambda value: value.selected(),
    }
    try:
        adapter = adapters[type(prediction)]
    except KeyError as error:
        raise TypeError(
            f"unsupported control prediction type: {type(prediction).__name__}"
        ) from error
    return adapter(prediction)


def map_control_candidates(
    prediction: ControlStrategyPrediction,
    transform: Callable[[ControlPrediction], ControlPrediction],
) -> ControlStrategyPrediction:
    """Apply a diagnostic transform without collapsing a mixture's candidates."""
    if isinstance(prediction, ControlPrediction):
        return transform(prediction)
    if not isinstance(prediction, ControlMixturePrediction):
        raise TypeError(
            f"unsupported control prediction type: {type(prediction).__name__}"
        )
    candidates = [transform(prediction.candidate(index)) for index in range(prediction.expert_count)]
    return ControlMixturePrediction(
        controls=torch.stack([item.controls for item in candidates], dim=1),
        segment_durations=torch.stack(
            [item.segment_durations for item in candidates], dim=1
        ),
        final_time_s=torch.stack([item.final_time_s for item in candidates], dim=1),
        selection_logits=prediction.selection_logits,
    )
