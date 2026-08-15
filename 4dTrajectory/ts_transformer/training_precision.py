"""Shared network-compute precision boundary for every TS output strategy."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.nn as nn

from config import TRAINING_PRECISION_BFLOAT16, TRAINING_PRECISION_FLOAT32, TSConfig
from control_mixture import ControlMixturePrediction
from prediction_outputs import ControlPrediction, StatePrediction


Prediction = StatePrediction | ControlPrediction | ControlMixturePrediction
_MODEL_PRECISION_ATTRIBUTE = "_aeroviz_training_precision"


def configure_model_runtime(model: nn.Module, config: TSConfig) -> nn.Module:
    """Attach the serialized compute recipe and configure process-wide FP32 matmuls."""
    setattr(model, _MODEL_PRECISION_ATTRIBUTE, config.training_precision)
    torch.set_float32_matmul_precision(config.float32_matmul_precision)
    return model


def model_autocast(model: nn.Module, history: torch.Tensor):
    """Autocast only network work; callers restore typed outputs before the objective."""
    precision = getattr(
        model,
        _MODEL_PRECISION_ATTRIBUTE,
        TRAINING_PRECISION_FLOAT32,
    )
    if precision == TRAINING_PRECISION_FLOAT32 or history.device.type != "cuda":
        return nullcontext()
    if precision != TRAINING_PRECISION_BFLOAT16:
        raise ValueError(f"unsupported model training precision {precision!r}")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def prediction_to_float32(prediction: Prediction) -> Prediction:
    """Leave the network precision boundary in FP32 without detaching gradients."""
    if isinstance(prediction, StatePrediction):
        return replace(
            prediction,
            states=prediction.states.float(),
            final_time_s=prediction.final_time_s.float(),
        )
    if isinstance(prediction, ControlPrediction):
        return replace(
            prediction,
            controls=prediction.controls.float(),
            segment_durations=prediction.segment_durations.float(),
            final_time_s=prediction.final_time_s.float(),
        )
    if isinstance(prediction, ControlMixturePrediction):
        return replace(
            prediction,
            controls=prediction.controls.float(),
            segment_durations=prediction.segment_durations.float(),
            final_time_s=prediction.final_time_s.float(),
            selection_logits=prediction.selection_logits.float(),
        )
    raise TypeError(f"unsupported prediction type: {type(prediction).__name__}")
