"""Single and mixture control models built on one shared trajectory feature encoder."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from config import TSConfig
from control_envelope import CONTROL_LOWER, CONTROL_UPPER
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


# How far, in logit units, the input-driven term is allowed to move the output away from
# the neutral control at step 0. Small enough that training still starts from "do nothing";
# NOT zero, because a zero output weight sends exactly zero gradient back through it.
NEUTRAL_LOGIT_PERTURBATION = 0.02


def _seed_projection(layer: nn.Linear, perturbation: float) -> None:
    """Give a layer a near-zero but GRADIENT-CARRYING weight.

    Zeroing an output layer's weight is a trap: the layer's own gradient is fine, but the
    gradient it passes back is ``Wᵀ·δ``, which is exactly zero while ``W`` is zero. Every
    parameter upstream — here the whole trajectory backbone — therefore receives no signal
    at all on the first step and can only start learning once ``W`` has bootstrapped away
    from zero on its own. Measured on the shipped initialisation: 20 of 20 backbone
    tensors had an exactly-zero gradient, and after 180 epochs at lr 3e-5 the control
    projection had grown to a weight norm of 0.55 against a bias norm of 5.84 — a head
    that had learned the population-average schedule and barely read its input.

    Scaling by ``1/sqrt(fan_in)`` keeps the initial output perturbation at
    ``perturbation`` logits regardless of ``d_model``.
    """
    nn.init.normal_(layer.weight, std=perturbation / math.sqrt(layer.in_features))


def _initialize_control_head(
    head: ControlOutputHead,
    *,
    bank_rad: float = 0.0,
    perturbation: float = NEUTRAL_LOGIT_PERTURBATION,
) -> None:
    """Start the head at the neutral control — through its BIAS, not by muting its input."""
    with torch.no_grad():
        _seed_projection(head.control_projection, perturbation)
        head.control_projection.bias.copy_(_neutral_control_bias(head, bank_rad))
        duration_projection = getattr(head, "duration_projection", None)
        if duration_projection is not None:
            _seed_projection(duration_projection, perturbation)
            duration_projection.bias.zero_()


def _initialize_final_time_head(
    head: FinalTimeHead,
    raw_bias: float = 0.0,
    *,
    perturbation: float = NEUTRAL_LOGIT_PERTURBATION,
) -> None:
    """Same contract for the duration head, whose hidden layer was equally starved."""
    with torch.no_grad():
        final_layer = head.network[-1]
        _seed_projection(final_layer, perturbation)
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
