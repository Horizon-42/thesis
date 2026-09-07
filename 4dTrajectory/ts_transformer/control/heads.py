"""Single and mixture control models built on one shared trajectory feature encoder."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from config import TSConfig
from control.envelope import CONTROL_LOWER, CONTROL_UPPER
from control.conditioning import DYNAMICS_CONDITION_NAMES
from config import (
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
    CTA_CONDITIONING_OFF,
    DURATION_HEAD_POINT,
    DURATION_HEAD_QUANTILE,
    DURATION_MEDIAN_INDEX,
)
from prediction_outputs import (
    ControlOutputHead,
    FinalTimeHead,
    QuantileFinalTimeHead,
    UniformDurationControlHead,
)


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
        # know when it has to arrive to decide the path that does. Both CTA modes build the
        # token — `given` hands it the truth, `self-q` (B3, predict only) hands it one of the
        # model's own duration quantiles — because they differ in WHERE the number comes
        # from, not in what the network does with it.
        self.cta_given = config.cta_conditioning != CTA_CONDITIONING_OFF
        self.quantile_duration = config.duration_head == DURATION_HEAD_QUANTILE
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

    def duration_quantiles(self, history: torch.Tensor) -> torch.Tensor | None:
        """The five ``DURATION_QUANTILES`` in seconds, ``[B, Q]``; None under the point head.

        Reads the HISTORY only — never the CTA, which enters through ``fused_features`` and
        reaches the control head, not this one. That is what lets B3 ask a ``given``
        checkpoint for its own quantiles without handing it an arrival time first.
        """
        return self.final_time_head(history) if self.quantile_duration else None

    def duration(
        self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(the duration the schedule is rolled over, the head's quantiles or None)``.

        The ONE rule for both, so a duration head cannot be read twice per forward pass —
        with dropout live in training, two calls are two different answers.

        Under a CTA the duration IS the CTA. The POINT head is then INERT for the life of
        the run — never called, never trained, kept at its initialization — so a given
        checkpoint resumed as ``off`` would start from an untrained duration head. The
        QUANTILE head is still read and still trained there, deliberately: B3 decodes a
        ``given`` checkpoint at its own quantiles, which only exist if the head learned them.
        """
        quantiles = self.duration_quantiles(history)
        if self.cta_given:
            return dynamics["cta_s"].to(history.dtype), quantiles
        if quantiles is None:
            return self.final_time_head(history), None
        return quantiles[:, DURATION_MEDIAN_INDEX], quantiles


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


def _quantile_increment_bias(raw_bias: float) -> float:
    """The increment logit that puts the MEDIAN quantile at the point head's own start.

    The point head starts at ``softplus(raw_bias)·scale``. The quantile head reaches its
    median by summing ``DURATION_MEDIAN_INDEX + 1`` positive increments, so each of them
    carries an equal share of that value and this inverts the softplus for one share. The
    two heads therefore predict the same duration on the first batch, and an arm that
    differs only in ``duration_head`` differs only in what it LEARNS.
    """
    share = math.log1p(math.exp(raw_bias)) / (DURATION_MEDIAN_INDEX + 1)
    return math.log(math.expm1(share))


#: The last-layer bias each duration head starts from, per `duration_head`. One table, so
#: "where the duration starts" is written once for both heads.
_DURATION_HEAD_INITIAL_BIAS = {
    DURATION_HEAD_POINT: lambda raw_bias: raw_bias,
    DURATION_HEAD_QUANTILE: _quantile_increment_bias,
}


def _initialize_duration_head(
    head: FinalTimeHead | QuantileFinalTimeHead, config: TSConfig, raw_bias: float = 0.0
) -> None:
    with torch.no_grad():
        final_layer = head.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.fill_(_DURATION_HEAD_INITIAL_BIAS[config.duration_head](raw_bias))


#: The duration head per `duration_head` — the ONE construction site, like `control_head_for`
#: below, so the deterministic model and the latent model cannot build different heads.
_DURATION_HEADS = {
    DURATION_HEAD_POINT: FinalTimeHead,
    DURATION_HEAD_QUANTILE: QuantileFinalTimeHead,
}


def duration_head_for(config: TSConfig) -> FinalTimeHead | QuantileFinalTimeHead:
    return _DURATION_HEADS[config.duration_head](config)


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
        self.final_time_head = duration_head_for(config)
        self.control_head = control_head_for(config)
        _initialize_control_head(self.control_head)
        _initialize_duration_head(self.final_time_head, config)

    def forward(self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]):
        features = self.fused_features(history, dynamics)
        final_time_s, quantiles = self.duration(history, dynamics)
        prediction = self.control_head(
            features,
            final_time_s,
            lower=dynamics["control_lower"],
            upper=dynamics["control_upper"],
        )
        # The control head knows nothing about quantiles — it partitions ONE duration — so
        # the interval is attached here rather than threaded through its signature.
        return replace(prediction, duration_quantiles_s=quantiles)
