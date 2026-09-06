"""Control-schedule imitation objective isolated from rollout fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prediction_outputs import ControlPrediction


@dataclass(frozen=True)
class ImitationLoss:
    total: torch.Tensor
    control: torch.Tensor
    duration_fraction: torch.Tensor
    final_time: torch.Tensor


def control_imitation_loss(
    prediction: ControlPrediction,
    target_controls: torch.Tensor,
    target_durations_s: torch.Tensor,
    target_final_time_s: torch.Tensor,
    control_lower: torch.Tensor,
    control_upper: torch.Tensor,
    *,
    final_time_scale_s: float,
) -> ImitationLoss:
    """Balance heterogeneous physical controls in their per-aircraft unit box."""
    scale = (control_upper - control_lower).unsqueeze(1)
    control = ((prediction.controls - target_controls) / scale).square().mean()
    segments = prediction.segment_durations.shape[1]
    predicted_fraction = prediction.segment_durations / prediction.segment_durations.sum(
        dim=1, keepdim=True
    )
    target_fraction = target_durations_s / target_durations_s.sum(dim=1, keepdim=True)
    duration_fraction = (
        (predicted_fraction - target_fraction) * segments
    ).square().mean()
    final_time = (
        (prediction.final_time_s - target_final_time_s) / final_time_scale_s
    ).square().mean()
    return ImitationLoss(
        total=control + duration_fraction + final_time,
        control=control,
        duration_fraction=duration_fraction,
        final_time=final_time,
    )
