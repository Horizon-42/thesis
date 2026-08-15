"""Direct positive per-segment durations for deterministic control prediction.

This module changes only the duration parameterization. It reuses the same backbone
features, aircraft conditioning, bounded controls, RK4 rollout and ControlPrediction
contract as the historical factorized control strategy.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import TSConfig
from control_models import (
    ControlFeatureModel,
    _initialize_control_head,
    _initialize_final_time_head,
)
from prediction_outputs import (
    ControlOutputHead,
    ControlPrediction,
    FinalTimeHead,
)


class DirectDurationControlHead(ControlOutputHead):
    """Predict independent positive durations and derive total time by summation."""

    def __init__(
        self,
        input_dim: int,
        n_segments: int,
        final_time_scale_s: float,
        *,
        duration_uniform_floor: float = 0.0,
    ):
        super().__init__(
            input_dim,
            n_segments,
            duration_uniform_floor=duration_uniform_floor,
        )
        self.segment_scale_s = final_time_scale_s / n_segments

    def forward(
        self,
        features: torch.Tensor,
        global_duration_logit: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> ControlPrediction:
        controls = self.bounded_controls(features, lower=lower, upper=upper)
        local_logits = self.duration_projection(features)
        raw_durations = F.softplus(
            global_duration_logit.unsqueeze(-1) + local_logits
        ) * self.segment_scale_s
        total_duration = raw_durations.sum(dim=-1, keepdim=True)
        learned_fractions = raw_durations / total_duration
        fractions = (
            learned_fractions * (1.0 - self.duration_uniform_floor)
            + self.duration_uniform_floor / self.n_segments
        )
        segment_durations = fractions * total_duration
        return ControlPrediction(
            controls=controls,
            segment_durations=segment_durations,
            final_time_s=segment_durations.sum(dim=-1),
        )


class DirectDurationControlOutputModel(ControlFeatureModel):
    """Single control model whose duration head emits absolute segment seconds."""

    def __init__(self, config: TSConfig, feature_encoder):
        super().__init__(config, feature_encoder)
        # The network and initialization match the factorized mode's FinalTimeHead. Here
        # its raw scalar is a shared logit for all segment durations, not a predicted total.
        self.global_duration_head = FinalTimeHead(config)
        self.control_head = DirectDurationControlHead(
            config.d_model,
            int(config.n_segments),
            config.final_time_scale_s,
            duration_uniform_floor=config.control_duration_uniform_floor,
        )
        _initialize_control_head(self.control_head)
        _initialize_final_time_head(self.global_duration_head)

    def forward(self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]):
        features = self.fused_features(history, dynamics)
        return self.control_head(
            features,
            self.global_duration_head.raw(history),
            lower=dynamics["control_lower"],
            upper=dynamics["control_upper"],
        )
