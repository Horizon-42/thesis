"""Contracts for the shared network-compute precision backend."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


TS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(TS_ROOT), str(REPO_ROOT)]

from config import (  # noqa: E402
    FLOAT32_MATMUL_PRECISION_HIGH,
    TRAINING_PRECISION_BFLOAT16,
    TSConfig,
)
from control_mixture import ControlMixturePrediction  # noqa: E402
from prediction_outputs import ControlPrediction, StatePrediction  # noqa: E402
from training_precision import prediction_to_float32  # noqa: E402


def test_training_precision_is_a_strict_serialized_recipe_field() -> None:
    config = TSConfig(
        training_precision=TRAINING_PRECISION_BFLOAT16,
        float32_matmul_precision=FLOAT32_MATMUL_PRECISION_HIGH,
    )

    assert TSConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="training_precision"):
        TSConfig(training_precision="float16")
    with pytest.raises(ValueError, match="float32_matmul_precision"):
        TSConfig(float32_matmul_precision="medium")


@pytest.mark.parametrize(
    "prediction",
    [
        StatePrediction(
            states=torch.ones(2, 3, 6, dtype=torch.bfloat16),
            final_time_s=torch.ones(2, dtype=torch.bfloat16),
        ),
        ControlPrediction(
            controls=torch.ones(2, 3, 3, dtype=torch.bfloat16),
            segment_durations=torch.ones(2, 3, dtype=torch.bfloat16),
            final_time_s=torch.ones(2, dtype=torch.bfloat16),
        ),
        ControlMixturePrediction(
            controls=torch.ones(2, 2, 3, 3, dtype=torch.bfloat16),
            segment_durations=torch.ones(2, 2, 3, dtype=torch.bfloat16),
            final_time_s=torch.ones(2, 2, dtype=torch.bfloat16),
            selection_logits=torch.ones(2, 2, dtype=torch.bfloat16),
        ),
    ],
)
def test_prediction_leaves_autocast_boundary_in_float32(prediction) -> None:
    converted = prediction_to_float32(prediction)

    for value in vars(converted).values():
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float32
