"""Deterministic bounded controls on one explicit uniform fraction of total time."""

from __future__ import annotations

import torch

from config import TSConfig
from control.heads import (
    control_head_for,
    ControlFeatureModel,
    _initialize_control_head,
    _initialize_final_time_head,
)
from prediction_outputs import ControlPrediction, FinalTimeHead, UniformDurationControlHead


class UniformDurationControlOutputModel(ControlFeatureModel):
    """History-conditioned controls plus one total-time head, with no duration head."""

    def __init__(self, config: TSConfig, feature_encoder):
        super().__init__(config, feature_encoder)
        self.final_time_head = FinalTimeHead(config)
        self.control_head = control_head_for(config)
        _initialize_control_head(self.control_head)
        _initialize_final_time_head(self.final_time_head)

    def forward(self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]):
        features = self.fused_features(history, dynamics)
        return self.control_head(
            features,
            self.final_time(self.final_time_head(history), dynamics),
            lower=dynamics["control_lower"],
            upper=dynamics["control_upper"],
        )
