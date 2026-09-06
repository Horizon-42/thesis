"""Single and mixture control models built on one shared trajectory feature encoder."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import TSConfig
from control.envelope import CONTROL_LOWER, CONTROL_UPPER
from control.conditioning import DYNAMICS_CONDITION_NAMES
from config import CONTROL_DURATION_FACTORIZED, CONTROL_DURATION_UNIFORM, CTA_CONDITIONING_GIVEN
from prediction_outputs import ControlOutputHead, FinalTimeHead, UniformDurationControlHead


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
        # The controlled time of arrival as one more fused token (L3): the decoder must
        # know when it has to arrive to decide the path that does.
        self.cta_given = config.cta_conditioning == CTA_CONDITIONING_GIVEN
        self.cta_scale_s = config.final_time_scale_s
        self.cta_encoder = (
            nn.Sequential(nn.Linear(1, config.d_model), nn.GELU()) if self.cta_given else None
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear((config.enc_in + 1 + int(self.cta_given)) * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

    def fused_features(
        self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        encoded = self.feature_encoder.encode_features(history)
        condition = self.condition_encoder(dynamics["condition"])
        parts = [encoded, condition]
        if self.cta_given:
            cta = (dynamics["cta_s"] / self.cta_scale_s).to(encoded.dtype).unsqueeze(-1)
            parts.append(self.cta_encoder(cta))
        return self.feature_fusion(torch.cat(parts, dim=-1))

    def final_time(self, head_value: torch.Tensor, dynamics: dict[str, torch.Tensor]) -> torch.Tensor:
        """The duration the schedule is rolled over: the given CTA, or the head's prediction."""
        if self.cta_given:
            return dynamics["cta_s"].to(head_value.dtype)
        return head_value


# What "doing nothing" means before any gradient arrives: 20% of installed thrust, wings
# level, load factor one. Expressed as physical values and mapped through the envelope, so
# a bound change moves the initialization with it instead of silently relocating it.
NEUTRAL_CONTROLS = (0.2, 0.0, 1.0)


def _neutral_control_bias(head: ControlOutputHead, bank_rad: float = 0.0) -> torch.Tensor:
    """Sigmoid logits whose bounded output is :data:`NEUTRAL_CONTROLS`."""
    neutral = np.array(NEUTRAL_CONTROLS, dtype=np.float64)
    neutral[1] = bank_rad
    unit = np.clip(
        (neutral - CONTROL_LOWER) / (CONTROL_UPPER - CONTROL_LOWER), 1e-6, 1.0 - 1e-6
    )
    return torch.tensor(
        np.log(unit / (1.0 - unit)),
        dtype=head.control_projection.bias.dtype,
        device=head.control_projection.bias.device,
    ).repeat(head.n_segments)


def _initialize_control_head(
    head: ControlOutputHead, *, bank_rad: float = 0.0, feature_std: float = 0.0
) -> None:
    with torch.no_grad():
        duration_projection = getattr(head, "duration_projection", None)
        if feature_std:
            nn.init.normal_(head.control_projection.weight, std=feature_std)
            if duration_projection is not None:
                nn.init.normal_(duration_projection.weight, std=feature_std)
        else:
            head.control_projection.weight.zero_()
            if duration_projection is not None:
                duration_projection.weight.zero_()
        head.control_projection.bias.copy_(_neutral_control_bias(head, bank_rad))
        if duration_projection is not None:
            duration_projection.bias.zero_()


def _initialize_final_time_head(head: FinalTimeHead, raw_bias: float = 0.0) -> None:
    with torch.no_grad():
        final_layer = head.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.fill_(raw_bias)


# The control head per duration parameterization — the ONE construction site, so the
# deterministic models and the latent model that composes a head cannot build different
# heads for the same config, and an unknown parameterization fails loudly everywhere.
def control_head_for(config: TSConfig) -> ControlOutputHead:
    builders = {
        CONTROL_DURATION_FACTORIZED: lambda: ControlOutputHead(
            config.d_model, int(config.n_segments),
            duration_uniform_floor=config.control_duration_uniform_floor,
        ),
        CONTROL_DURATION_UNIFORM: lambda: UniformDurationControlHead(
            config.d_model, int(config.n_segments)
        ),
    }
    return builders[config.control_duration_parameterization]()


class ControlOutputModel(ControlFeatureModel):
    """Original single deterministic control strategy, state-dict compatible."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
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
