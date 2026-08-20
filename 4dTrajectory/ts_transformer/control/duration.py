"""Deterministic bounded controls on one explicit uniform fraction of total time."""

from __future__ import annotations

import torch

from config import TSConfig
from control.heads import (
    ControlFeatureModel,
    _initialize_control_head,
    _initialize_final_time_head,
)
from prediction_outputs import ControlOutputHead, ControlPrediction, FinalTimeHead


class UniformDurationControlHead(ControlOutputHead):
    """Decode controls while fixing every segment duration to ``final_time / N``."""

    def __init__(self, input_dim: int, n_segments: int):
        super().__init__(input_dim, n_segments)
        # The base class owns the established bounded-control projection. Removing this
        # module makes the simplified contract structural: no unused duration logits are
        # serialized, optimized, or accidentally revived by another loss.
        self.duration_projection = None

    def forward(
        self,
        features: torch.Tensor,
        final_time_s: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> ControlPrediction:
        controls = self.bounded_controls(features, lower=lower, upper=upper)
        segment_durations = final_time_s.unsqueeze(-1).expand(
            -1, self.n_segments
        ) / self.n_segments
        return ControlPrediction(
            controls=controls,
            segment_durations=segment_durations,
            final_time_s=final_time_s,
        )


class UniformDurationControlOutputModel(ControlFeatureModel):
    """History-conditioned controls plus one total-time head, with no duration head."""

    def __init__(self, config: TSConfig, feature_encoder):
        super().__init__(config, feature_encoder)
        self.final_time_head = FinalTimeHead(config)
        self.control_head = UniformDurationControlHead(
            config.d_model, int(config.n_segments)
        )
        _initialize_control_head(self.control_head)
        _initialize_final_time_head(self.final_time_head)

    def forward(self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]):
        features = self.fused_features(history, dynamics)
        return self.control_head(
            features,
            self.final_time_head(history),
            lower=dynamics["control_lower"],
            upper=dynamics["control_upper"],
        )
