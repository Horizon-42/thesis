"""Control-parameterization-specific effort and smoothness signals."""

from __future__ import annotations

from collections.abc import Callable

import torch

from config import CONTROL_VALUE_ABSOLUTE, CONTROL_VALUE_TRIM_RESIDUAL, TSConfig
from prediction_outputs import ControlPrediction
from trim_residual_control import interior_trim_controls, trim_control_baseline


def _absolute_signals(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = dynamics["control_lower"]
    upper = dynamics["control_upper"]
    effort_scale = torch.stack(
        (
            upper[:, 0],
            torch.full_like(upper[:, 1], torch.pi / 2.0),
            upper[:, 2],
        ),
        dim=-1,
    )
    effort = prediction.controls / effort_scale.unsqueeze(1)
    smoothness = prediction.controls / (upper - lower).unsqueeze(1)
    return effort, smoothness


def _trim_residual_signals(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = prediction.controls.dtype
    lower = dynamics["control_lower"].to(dtype=dtype)
    upper = dynamics["control_upper"].to(dtype=dtype)
    baseline = trim_control_baseline(
        dynamics["initial_state"].to(dtype=dtype),
        dynamics["aero_params"].to(dtype=dtype),
        lower,
        upper,
    )
    trim, _unit = interior_trim_controls(baseline.controls, lower, upper)
    residual = (prediction.controls - trim.unsqueeze(1)) / (
        upper - lower
    ).unsqueeze(1)
    return residual, residual


_REGULARIZATION_SIGNALS: dict[
    str,
    Callable[
        [ControlPrediction, dict[str, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor],
    ],
] = {
    CONTROL_VALUE_ABSOLUTE: _absolute_signals,
    CONTROL_VALUE_TRIM_RESIDUAL: _trim_residual_signals,
}


def control_regularization_signals(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless effort values and the signal whose differences are smooth."""
    return _REGULARIZATION_SIGNALS[config.control_value_parameterization](
        prediction, dynamics
    )
