"""Single and mixture control models built on one shared trajectory feature encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import TSConfig
from control_mixture import ControlMixtureHead
from dataset import DYNAMICS_CONDITION_NAMES
from prediction_outputs import ControlOutputHead, FinalTimeHead


class ControlFeatureModel(nn.Module):
    """Shared history/aircraft feature path; output strategies own only their heads."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.feature_encoder.discard_state_head()
        self.condition_encoder = nn.Sequential(
            nn.Linear(len(DYNAMICS_CONDITION_NAMES), config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear((config.enc_in + 1) * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

    def fused_features(
        self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        encoded = self.feature_encoder.encode_features(history)
        condition = self.condition_encoder(dynamics["condition"])
        return self.feature_fusion(torch.cat((encoded, condition), dim=-1))


def _neutral_control_bias(head: ControlOutputHead, bank_rad: float = 0.0) -> torch.Tensor:
    thrust_fraction = 0.2
    bank_fraction = (bank_rad + math.pi / 4.0) / (math.pi / 2.0)
    load_fraction = (1.0 - 0.5) / (2.0 - 0.5)
    return torch.tensor(
        (
            math.log(thrust_fraction / (1.0 - thrust_fraction)),
            math.log(bank_fraction / (1.0 - bank_fraction)),
            math.log(load_fraction / (1.0 - load_fraction)),
        ),
        dtype=head.control_projection.bias.dtype,
        device=head.control_projection.bias.device,
    ).repeat(head.n_segments)


def _initialize_control_head(
    head: ControlOutputHead, *, bank_rad: float = 0.0, feature_std: float = 0.0
) -> None:
    with torch.no_grad():
        if feature_std:
            nn.init.normal_(head.control_projection.weight, std=feature_std)
            nn.init.normal_(head.duration_projection.weight, std=feature_std)
        else:
            head.control_projection.weight.zero_()
            head.duration_projection.weight.zero_()
        head.control_projection.bias.copy_(_neutral_control_bias(head, bank_rad))
        head.duration_projection.bias.zero_()


def _initialize_final_time_head(head: FinalTimeHead, raw_bias: float = 0.0) -> None:
    with torch.no_grad():
        final_layer = head.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.fill_(raw_bias)


class ControlOutputModel(ControlFeatureModel):
    """Original single deterministic control strategy, state-dict compatible."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
        super().__init__(config, feature_encoder)
        self.final_time_head = FinalTimeHead(config)
        self.control_head = ControlOutputHead(
            config.d_model,
            int(config.n_segments),
            duration_uniform_floor=config.control_duration_uniform_floor,
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


class ControlMixtureOutputModel(ControlFeatureModel):
    """K independent control/duration heads with one history-only selector."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
        super().__init__(config, feature_encoder)
        self.final_time_heads = nn.ModuleList(
            FinalTimeHead(config) for _ in range(config.control_expert_count)
        )
        self.mixture_head = ControlMixtureHead(
            config.d_model,
            int(config.n_segments),
            config.control_expert_count,
            duration_uniform_floor=config.control_duration_uniform_floor,
        )
        offsets = torch.linspace(-1.0, 1.0, config.control_expert_count).tolist()
        for offset, time_head, control_head in zip(
            offsets, self.final_time_heads, self.mixture_head.experts
        ):
            _initialize_final_time_head(time_head, raw_bias=0.1 * offset)
            _initialize_control_head(
                control_head,
                bank_rad=math.radians(2.0) * offset,
                feature_std=1e-4,
            )
        with torch.no_grad():
            self.mixture_head.selector.weight.zero_()
            self.mixture_head.selector.bias.zero_()

    def forward(self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]):
        features = self.fused_features(history, dynamics)
        final_times = torch.stack(
            [head(history) for head in self.final_time_heads], dim=1
        )
        return self.mixture_head(
            features,
            final_times,
            lower=dynamics["control_lower"],
            upper=dynamics["control_upper"],
        )
