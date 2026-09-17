"""The segment-plan head (two-tier v2 §4): the plan layer's model, prediction and loss.

The window's segment features (`features.segment_features`) go through the vendored
iTransformer ENCODER (`backbone.adapters.ITransformerEncoderStack`, built by the strategy)
in one of two token layouts — `segment_plan_attention`:

* ``channels``: each FEATURE's K-long series is one token (the inversion the requirement
  names: "the path itself, the start and end points, the overall direction as channels
  computing attention"); the vendored inverted embedding maps a length-K series to d_model.
* ``segments``: each SEGMENT's feature vector is one token (PatchTST's form with a joint
  segment embedding); the same inverted embedding applied to the transposed matrix, plus a
  learned position per segment (the inverted embedding has none, and segment order matters).

Either way the encoder's tokens are read out by ONE learned query (`TokenPool`) into a
d_model vector, so the two arms differ in their tokens and in nothing else — a flattened
read-out would hand the channels arm a 27·d_model head against the segments arm's K·d_model,
and gate L2's A-against-B reading would be a capacity reading.

The head then emits, for M coarse segments after the anchor, the position at each segment's
end in runway axes about the anchor (along, across, up), the logit of "arrived by this
segment's end" and the logit of the fraction of that segment flown before arriving. ONE
direct decode of all M segments (no autoregression, feasibility §4.5).

The loss (`segment_plan_loss_components`): the positions' L1 in their scales over the
segments the truth reaches (`state`), the arrival bits' cross-entropy over every segment
(`terminal`), the arrival fraction's L1 in the segment the truth arrives in (`final_time`);
`kinematic` is a structural zero (the four names are the package's fixed contract).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ts_transformer.config import (
    PLAN_WAYPOINT_SEGMENT_S,
    SEGMENT_PLAN_ATTENTION_CHANNELS,
    SEGMENT_PLAN_ATTENTION_SEGMENTS,
    TSConfig,
    segment_plan_input_segments,
    segment_plan_segment_samples,
)
from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.outputs.segment_plan.features import PATH_SUBSAMPLES, SEGMENT_FEATURES, segment_features
from ts_transformer.outputs.segment_plan.labels import (
    CONTEXT_ARRIVAL_FRACTION,
    CONTEXT_ARRIVAL_VALID,
    CONTEXT_ARRIVED,
    CONTEXT_KEYS,
    CONTEXT_RUNWAY_HEADING,
    CONTEXT_TARGET_CHART,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    POSITION_SCALE,
)

SEGMENT_PLAN_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")
#: Where every flight starts (the head's last bias): a straight-in at approach speed and a
#: 3° descent — plausible waypoints, so the first replays draw approaches.
INITIAL_ALONG_MPS = 80.0
INITIAL_DESCENT_RATE_MPS = 4.0


@dataclass(frozen=True)
class SegmentPlanPrediction:
    """``positions`` ``[B, M, 3]`` in metres — (along, across, up) about the anchor in runway
    axes — the arrival logits ``[B, M]`` and the arrival-fraction logits ``[B, M]``."""

    positions: torch.Tensor
    arrived_logits: torch.Tensor
    fraction_logits: torch.Tensor


def encoder_token_length(config: TSConfig) -> int:
    """How long the series each encoder token embeds is under the config's token axis: the
    K input segments under ``channels`` (a feature's series), the F features under
    ``segments`` (a segment's vector)."""
    if config.segment_plan_attention == SEGMENT_PLAN_ATTENTION_CHANNELS:
        return segment_plan_input_segments(config)
    return len(SEGMENT_FEATURES)


class SegmentChannelEncoder(nn.Module):
    """``channels`` attention: ``[B, K, F]`` features → one token per FEATURE (its K-series
    embedded), attention between the F features → ``[B, F, d_model]``."""

    def __init__(self, stack: nn.Module, segments: int) -> None:
        super().__init__()
        del segments
        self.stack = stack

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.stack(features)


class SegmentTokenEncoder(nn.Module):
    """``segments`` attention: ``[B, K, F]`` features → one token per SEGMENT (its feature
    vector embedded, plus a learned position), attention between the K segments →
    ``[B, K, d_model]``."""

    def __init__(self, stack: nn.Module, segments: int) -> None:
        super().__init__()
        self.stack = stack
        self.position = nn.Parameter(torch.zeros(segments, stack.d_model))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.stack.inner.enc_embedding(features.transpose(1, 2), None) + self.position
        encoded, _attentions = self.stack.inner.encoder(encoded, attn_mask=None)
        return encoded


_ENCODERS = {
    SEGMENT_PLAN_ATTENTION_CHANNELS: SegmentChannelEncoder,
    SEGMENT_PLAN_ATTENTION_SEGMENTS: SegmentTokenEncoder,
}


class TokenPool(nn.Module):
    """One learned query attending over the encoder's tokens: ``[B, T, d_model]`` →
    ``[B, d_model]`` under either token axis — the read-out that keeps the two arms'
    heads identical."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)


class SegmentPlanModel(nn.Module):
    """The plan layer: the physical window's segment features → the encoder → the token
    read-out → an MLP → M segments' positions, arrival logits and arrival-fraction logits.
    Carries the normalizer's statistics as buffers (bound at build, restored with the
    weights) because the features are geometry in metres; the labels in the context are
    targets, never inputs — the forward reads the runway course and the threshold only."""

    def __init__(self, config: TSConfig, stack: nn.Module, normalizer=None) -> None:
        super().__init__()
        self.config = config
        self.channel_count = len(config.channels)
        self.segments = int(config.segment_plan_segments)
        self.input_segments = segment_plan_input_segments(config)
        self.segment_samples = segment_plan_segment_samples(config)
        if self.segment_samples < PATH_SUBSAMPLES:
            raise ValueError(
                f"a coarse segment of {self.segment_samples} samples cannot carry {PATH_SUBSAMPLES} "
                f"distinct path sub-samples (dt_s={config.dt_s:g})"
            )
        self.register_buffer("channel_mean", torch.zeros(self.channel_count))
        self.register_buffer("channel_std", torch.ones(self.channel_count))
        if normalizer is not None:
            self.bind_normalizer(normalizer)
        self.encoder = _ENCODERS[config.segment_plan_attention](stack, self.input_segments)
        self.pool = TokenPool(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, 5 * self.segments),
        )
        self.register_buffer("position_scale", torch.as_tensor(POSITION_SCALE, dtype=torch.float32), persistent=False)
        with torch.no_grad():
            last = self.head[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            ends = PLAN_WAYPOINT_SEGMENT_S * torch.arange(1, self.segments + 1, dtype=torch.float32)
            initial = torch.stack((INITIAL_ALONG_MPS * ends, torch.zeros_like(ends), -INITIAL_DESCENT_RATE_MPS * ends), dim=1)
            last.bias[: 3 * self.segments] = (initial / self.position_scale).reshape(-1)

    def bind_normalizer(self, normalizer) -> None:
        self.channel_mean.copy_(torch.as_tensor(normalizer.mean, dtype=torch.float32))
        self.channel_std.copy_(torch.as_tensor(normalizer.std, dtype=torch.float32))

    def features(self, history: torch.Tensor, context: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[B, K, F]`` from the normalized history (its input-only conditioning columns, if
        any, are not geometry and are dropped), the flight's runway course and threshold."""
        window = history[..., : self.channel_count] * self.channel_std + self.channel_mean
        return segment_features(
            window, context[CONTEXT_RUNWAY_HEADING].to(window.dtype), context[CONTEXT_TARGET_CHART].to(window.dtype),
            segment_samples=self.segment_samples,
        )

    def forward(self, history: torch.Tensor, context: dict[str, torch.Tensor] | None = None) -> SegmentPlanPrediction:
        if context is None or CONTEXT_RUNWAY_HEADING not in context or CONTEXT_TARGET_CHART not in context:
            raise ValueError("the segment-plan model needs the flight's runway course and threshold in the batch context")
        raw = self.head(self.pool(self.encoder(self.features(history, context))))
        m = self.segments
        positions = raw[:, : 3 * m].reshape(-1, m, 3) * self.position_scale
        return SegmentPlanPrediction(
            positions=positions, arrived_logits=raw[:, 3 * m : 4 * m], fraction_logits=raw[:, 4 * m : 5 * m],
        )


def probe_segment_plan_context(batch_size: int, device: torch.device, config: TSConfig) -> dict[str, torch.Tensor]:
    """One representative context for the batch-size probe: every key a real batch carries."""
    m = int(config.segment_plan_segments)
    return {
        CONTEXT_TARGETS: torch.zeros((batch_size, m, 3), dtype=torch.float32, device=device),
        CONTEXT_VALID: torch.ones((batch_size, m), dtype=torch.float32, device=device),
        CONTEXT_ARRIVED: torch.zeros((batch_size, m), dtype=torch.float32, device=device),
        CONTEXT_ARRIVAL_FRACTION: torch.zeros((batch_size, m), dtype=torch.float32, device=device),
        CONTEXT_ARRIVAL_VALID: torch.zeros((batch_size, m), dtype=torch.float32, device=device),
        CONTEXT_RUNWAY_HEADING: torch.zeros(batch_size, dtype=torch.float32, device=device),
        CONTEXT_TARGET_CHART: torch.zeros((batch_size, 3), dtype=torch.float32, device=device),
    }


def segment_plan_loss_components(
    prediction: SegmentPlanPrediction, flight_weights: torch.Tensor, config: TSConfig,
    context: dict[str, torch.Tensor] | None,
) -> LossComponents:
    """Per flight: the positions' mean L1 (in the position scales) over the segments the truth
    reaches, the arrival bits' mean cross-entropy over every segment, the arrival fraction's L1
    in the arrival segment; each a flight-weighted mean over the flights that carry it (a
    flight with no reached segment contributes nothing to the position term, never zero)."""
    if context is None or any(key not in context for key in CONTEXT_KEYS):
        raise ValueError(f"the segment-plan loss needs the label context {CONTEXT_KEYS}")
    positions = prediction.positions
    dtype = positions.dtype
    scale = torch.as_tensor(POSITION_SCALE, dtype=dtype, device=positions.device)
    weight = flight_weights.to(dtype)

    def carried_mean(per_flight: torch.Tensor, carries: torch.Tensor) -> torch.Tensor:
        carriers = carries.to(dtype) * weight
        return (per_flight * carriers).sum() / carriers.sum().clamp(min=1e-6)

    valid = context[CONTEXT_VALID].to(dtype)                                             # [B, M]
    residual = ((positions - context[CONTEXT_TARGETS].to(dtype)).abs() / scale).mean(dim=2)  # [B, M]
    position = carried_mean((residual * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0), valid.sum(dim=1) > 0)
    arrived = F.binary_cross_entropy_with_logits(
        prediction.arrived_logits.to(dtype), context[CONTEXT_ARRIVED].to(dtype), reduction="none",
    ).mean(dim=1)
    arrival = (arrived * weight).sum() / weight.sum().clamp(min=1e-6)
    arrival_valid = context[CONTEXT_ARRIVAL_VALID].to(dtype)
    fraction_residual = (torch.sigmoid(prediction.fraction_logits.to(dtype)) - context[CONTEXT_ARRIVAL_FRACTION].to(dtype)).abs()
    fraction = carried_mean((fraction_residual * arrival_valid).sum(dim=1), arrival_valid.sum(dim=1) > 0)
    return LossComponents(
        state=position * config.segment_plan_position_loss_weight,
        final_time=fraction * config.segment_plan_arrival_loss_weight,
        kinematic=positions.new_zeros(()),
        terminal=arrival * config.segment_plan_arrival_loss_weight,
    )


def prediction_rows(prediction: SegmentPlanPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The prediction as numpy: positions ``[B, M, 3]`` (metres, runway axes about the anchor),
    the arrival probabilities ``[B, M]`` and the arrival fractions ``[B, M]`` in (0, 1)."""
    return (
        prediction.positions.detach().cpu().numpy().astype(np.float64),
        torch.sigmoid(prediction.arrived_logits).detach().cpu().numpy().astype(np.float64),
        torch.sigmoid(prediction.fraction_logits).detach().cpu().numpy().astype(np.float64),
    )


__all__ = [
    "SEGMENT_PLAN_LOSS_COMPONENT_NAMES", "SegmentChannelEncoder", "SegmentPlanModel", "SegmentPlanPrediction",
    "SegmentTokenEncoder", "TokenPool", "encoder_token_length", "prediction_rows", "probe_segment_plan_context",
    "segment_plan_loss_components",
]
