"""Dimensionless control effort and smoothness signals.

Controls are divided by each flight's own envelope before the regularizers see them, so a
mixed-aircraft batch shares one dimensionless loss instead of being dominated by whichever
airframe has the largest installed thrust.
"""

from __future__ import annotations

import torch

from prediction_outputs import ControlPrediction


def control_regularization_signals(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless effort values and the signal whose differences are smooth.

    Effort measures each control against the magnitude that "using all of it" means —
    full thrust, a quarter turn of bank, the load-factor ceiling — so zero effort is the
    unforced trajectory. Smoothness measures step-to-step change against the full span of
    each bound, which is the only scale a difference has.
    """
    lower = dynamics["control_lower"]
    upper = dynamics["control_upper"]
    effort_scale = torch.stack(
        (upper[:, 0], torch.full_like(upper[:, 1], torch.pi / 2.0), upper[:, 2]),
        dim=-1,
    )
    effort = prediction.controls / effort_scale.unsqueeze(1)
    smoothness = prediction.controls / (upper - lower).unsqueeze(1)
    return effort, smoothness
