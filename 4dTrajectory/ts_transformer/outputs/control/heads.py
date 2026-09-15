"""Single and mixture control models built on one shared trajectory feature encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

from ts_transformer.config import TSConfig
from ts_transformer.outputs.envelope import ControlContract, control_contract
from ts_transformer.outputs.conditioning import condition_names
from ts_transformer.config import (
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
    CTA_CONDITIONING_OFF,
    DURATION_HEAD_POINT,
    DURATION_HEAD_QUANTILE,
    DURATION_HEAD_TWO_HEAD,
    DURATION_HEADS_WITH_POINT,
    DURATION_HEADS_WITH_QUANTILES,
    DURATION_MEDIAN_INDEX,
)
from ts_transformer.outputs.duration_heads import FinalTimeHead, QuantileFinalTimeHead


# Three columns in the order every contract shares; the labels are the thrust-fraction
# saturation diagnostics' historical keys (`training.diagnostics.saturation_labels`).
CONTROL_NAMES = ("thrust_N", "bank_rad", "load_factor")


@dataclass(frozen=True)
class ControlPrediction:
    controls: torch.Tensor           # [B, N, 3], in the run's control contract (envelope.py)
    segment_durations: torch.Tensor  # [B, N], physical seconds
    final_time_s: torch.Tensor       # [B], segment_durations.sum(dim=-1)
    # B1: the five `DURATION_QUANTILES` in seconds, `[B, Q]`, or None under the point head.
    # `final_time_s` above stays the ONE duration the schedule is rolled over (the median,
    # or the given CTA), so nothing downstream changes; these are the interval the ETA
    # calibration is built on. `kw_only` because `LatentControlPrediction` extends this
    # dataclass with required fields, and a defaulted one before them would not compile.
    duration_quantiles_s: torch.Tensor | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ControlBounds:
    """Per-flight bounds in the run's control-contract order (``outputs/envelope.py``)."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        expected = len(CONTROL_NAMES)
        if len(self.lower) != expected or len(self.upper) != expected:
            raise ValueError(f"control bounds must contain exactly {expected} values")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("every control lower bound must be smaller than its upper bound")


class ControlOutputHead(nn.Module):
    """Decode generic features into bounded controls and a non-uniform time partition.

    ``final_time_s`` is predicted by the model's duration head.  Duration logits only
    decide how that time is distributed over the N piecewise-constant control segments.
    """

    def __init__(
        self,
        input_dim: int,
        n_segments: int,
        bounds: ControlBounds | None = None,
        duration_uniform_floor: float = 0.0,
    ):
        super().__init__()
        if not 0.0 <= duration_uniform_floor < 1.0:
            raise ValueError("duration_uniform_floor must be in [0, 1)")
        self.n_segments = n_segments
        self.duration_uniform_floor = float(duration_uniform_floor)
        self.control_projection = nn.Linear(input_dim, n_segments * len(CONTROL_NAMES))
        self.duration_projection = nn.Linear(input_dim, n_segments)
        if bounds is None:
            self.register_buffer("lower", None)
            self.register_buffer("upper", None)
        else:
            self.register_buffer("lower", torch.tensor(bounds.lower, dtype=torch.float32))
            self.register_buffer("upper", torch.tensor(bounds.upper, dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor,
        final_time_s: torch.Tensor,
        *,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> ControlPrediction:
        controls = self.bounded_controls(features, lower=lower, upper=upper)
        fractions = stabilized_duration_fractions(
            self.duration_projection(features), self.duration_uniform_floor
        )
        segment_durations = fractions * final_time_s.unsqueeze(-1)
        return ControlPrediction(
            controls=controls,
            segment_durations=segment_durations,
            final_time_s=final_time_s,
        )

    def bounded_controls(
        self,
        features: torch.Tensor,
        *,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map logits to per-flight physical bounds for reusable control heads."""
        batch = features.shape[0]
        lower = self.lower if lower is None else lower
        upper = self.upper if upper is None else upper
        if lower is None or upper is None:
            raise ValueError("per-sample lower and upper bounds are required")
        if lower.shape[-1] != len(CONTROL_NAMES) or upper.shape != lower.shape:
            raise ValueError("control bounds must end in 3 aligned values")
        if lower.ndim == 1:
            lower = lower.unsqueeze(0).expand(batch, -1)
            upper = upper.unsqueeze(0).expand(batch, -1)
        if lower.shape != (batch, len(CONTROL_NAMES)):
            raise ValueError(
                f"per-sample bounds must be [B,3], got {tuple(lower.shape)} for B={batch}"
            )
        unit_controls = torch.sigmoid(self.control_projection(features)).view(
            batch, self.n_segments, len(CONTROL_NAMES)
        )
        return lower.unsqueeze(1) + unit_controls * (
            upper - lower
        ).unsqueeze(1)



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


def stabilized_duration_fractions(
    logits: torch.Tensor, uniform_floor: float
) -> torch.Tensor:
    """Keep a learnable partition while reserving duration mass uniformly.

    A raw softmax permits one segment to approach 100% of the trajectory. Reserving a
    fixed share of total time uniformly gives every segment a hard positive floor and
    bounds the largest possible segment without clipping gradients.
    """
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("duration logits must end in at least one segment")
    if not 0.0 <= uniform_floor < 1.0:
        raise ValueError("duration uniform floor must be in [0, 1)")
    learned = torch.softmax(logits, dim=-1)
    return learned * (1.0 - uniform_floor) + uniform_floor / logits.shape[-1]




class ControlFeatureModel(nn.Module):
    """Shared history/aircraft feature path; output strategies own only their heads."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.feature_encoder.discard_state_head()
        self.condition_encoder = nn.Sequential(
            nn.Linear(len(condition_names(config.control_condition_features)), config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        # The controlled time of arrival as one more fused token (L3): the decoder must
        # know when it has to arrive to decide the path that does. Both CTA modes build the
        # token — `given` hands it the truth, `self-q` (B3, predict only) hands it one of the
        # model's own duration quantiles — because they differ in WHERE the number comes
        # from, not in what the network does with it.
        self.cta_given = config.cta_conditioning != CTA_CONDITIONING_OFF
        # WHICH duration heads this run carries. Both are True only under `two-head`, where
        # the point head drives the rollout and the quantile head is published beside it.
        self.quantile_duration = config.duration_head in DURATION_HEADS_WITH_QUANTILES
        self.point_duration = config.duration_head in DURATION_HEADS_WITH_POINT
        self.cta_scale_s = config.final_time_scale_s
        self.cta_encoder = (
            nn.Sequential(nn.Linear(1, config.d_model), nn.GELU()) if self.cta_given else None
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear((config.enc_in + 1 + int(self.cta_given)) * config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        # The SECOND duration head (B1.b). Declared here because `duration_quantiles` below
        # reads it and both control models inherit that rule, but BUILT LAST by the subclass
        # that can have one (`ControlOutputModel`), for the reason `aux_duration` is built
        # last in `control/latent.py`: every module before it must draw the initialization it
        # would draw without it. Built here, the extra `nn.Linear` would consume RNG ahead of
        # `final_time_head`, whose hidden layer is a live random draw
        # (`_initialize_duration_head` zeroes only the last layer) — and a `two-head` arm
        # would then start from a different point head than the `point` arm gate 1 compares
        # it against, single-seed, inside a 30 m band.
        self.duration_quantile_head = None

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

    def quantile_head(self) -> QuantileFinalTimeHead | None:
        """WHICH module holds the quantiles, or None under ``point``.

        Under ``quantile`` the duration head IS the quantile head (``final_time_head``, so
        the state-dict keys are the ones B1 trained); under ``two-head`` it is the second
        head beside it. A method rather than an attribute because binding the same module
        under two names would publish it twice in ``state_dict``.
        """
        if not self.quantile_duration:
            return None
        return (
            self.duration_quantile_head
            if self.duration_quantile_head is not None
            else self.final_time_head
        )

    def duration_quantiles(self, history: torch.Tensor) -> torch.Tensor | None:
        """The five ``DURATION_QUANTILES`` in seconds, ``[B, Q]``; None under the point head.

        Reads the HISTORY only — never the CTA, which enters through ``fused_features`` and
        reaches the control head, not this one. That is what lets B3 ask a ``given``
        checkpoint for its own quantiles without handing it an arrival time first.
        """
        head = self.quantile_head()
        return None if head is None else head(history)

    def duration(
        self, history: torch.Tensor, dynamics: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(the duration the schedule is rolled over, the head's quantiles or None)``.

        The ONE rule for both, so a duration head cannot be read twice per forward pass —
        with dropout live in training, two calls are two different answers.

        WHICH number the rollout gets is the whole of `duration_head`: the point head's
        under ``point`` and ``two-head`` (B1.b: the quantile head then trains beside it and
        touches the path only through the shared trunk), the MEDIAN quantile under
        ``quantile`` (B1: it walks the `final_time_s` contract).

        Under a CTA the duration IS the CTA. The POINT head is then INERT for the life of
        the run — never called, never trained, kept at its initialization — so a given
        checkpoint resumed as ``off`` would start from an untrained duration head, and its
        ``durationHeadFinalTimeS`` would be that untrained head's number. ``cta_s`` is the
        truth duration, i.e. the objective's own target, so ``final_time_loss_weight`` then
        weighs an identically-zero term: on a ``given`` arm it is a weight with nothing to
        weigh, for BOTH ``point`` and ``two-head``. It is documented rather than refused
        because the refusal would be new law over the stored L3 ``cta=given`` arms. The
        QUANTILE head is still read and still trained there, deliberately: B3 decodes a
        ``given`` checkpoint at its own quantiles, which only exist if the head learned them.
        """
        quantiles = self.duration_quantiles(history)
        if self.cta_given:
            return dynamics["cta_s"].to(history.dtype), quantiles
        if self.point_duration:
            return self.final_time_head(history), quantiles
        return quantiles[:, DURATION_MEDIAN_INDEX], quantiles


def _neutral_control_bias(head: ControlOutputHead, contract: ControlContract) -> torch.Tensor:
    """Sigmoid logits whose bounded output is the contract's neutral command — what "doing
    nothing" means before any gradient arrives (``ControlContract.neutral``). Expressed as
    physical values and mapped through the contract's box, so a bound change moves the
    initialization with it instead of silently relocating it."""
    neutral = np.array(contract.neutral, dtype=np.float64)
    lower, upper = contract.lower_array, contract.upper_array
    unit = np.clip((neutral - lower) / (upper - lower), 1e-6, 1.0 - 1e-6)
    return torch.tensor(
        np.log(unit / (1.0 - unit)),
        dtype=head.control_projection.bias.dtype,
        device=head.control_projection.bias.device,
    ).repeat(head.n_segments)


def _initialize_control_head(head: ControlOutputHead, contract: ControlContract) -> None:
    """Zero the last layer so every flight starts at the contract's neutral controls.

    Every caller wants exactly this; the ``bank_rad`` / ``feature_std`` variants the
    function used to offer had no caller (review §4.7, deleted 2026-09-09).
    """
    with torch.no_grad():
        duration_projection = getattr(head, "duration_projection", None)
        head.control_projection.weight.zero_()
        if duration_projection is not None:
            duration_projection.weight.zero_()
        head.control_projection.bias.copy_(_neutral_control_bias(head, contract))
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


#: The last-layer bias each duration head starts from, keyed by the head's TYPE rather than
#: by `duration_head` — under `two-head` one config builds one of each, and "where the
#: duration starts" is a property of the head, written once.
_DURATION_HEAD_INITIAL_BIAS = {
    FinalTimeHead: lambda raw_bias: raw_bias,
    QuantileFinalTimeHead: _quantile_increment_bias,
}


def _initialize_duration_head(
    head: FinalTimeHead | QuantileFinalTimeHead, raw_bias: float = 0.0
) -> None:
    with torch.no_grad():
        final_layer = head.network[-1]
        final_layer.weight.zero_()
        final_layer.bias.fill_(_DURATION_HEAD_INITIAL_BIAS[type(head)](raw_bias))


#: The duration head per `duration_head` — the ONE construction site, like `control_head_for`
#: below, so the deterministic model and the latent model cannot build different heads.
#: Under `two-head` this is the POINT head: same class, same `final_time_head.*` state-dict
#: keys and the same rollout duration as `point`, with the quantile head added beside it by
#: `quantile_duration_head_for`.
_DURATION_HEADS = {
    DURATION_HEAD_POINT: FinalTimeHead,
    DURATION_HEAD_QUANTILE: QuantileFinalTimeHead,
    DURATION_HEAD_TWO_HEAD: FinalTimeHead,
}


def duration_head_for(config: TSConfig) -> FinalTimeHead | QuantileFinalTimeHead:
    return _DURATION_HEADS[config.duration_head](config)


def quantile_duration_head_for(config: TSConfig) -> QuantileFinalTimeHead | None:
    """The SECOND duration head — built under `two-head` alone, initialized with it.

    None everywhere else, and that None is load-bearing: it is what
    ``ControlFeatureModel.quantile_head`` reads to decide whether the quantiles come from a
    head of their own or from ``final_time_head`` itself, and it is why a `point` or
    `quantile` run registers not one parameter more than it did before B1.b.
    """
    if config.duration_head != DURATION_HEAD_TWO_HEAD:
        return None
    head = QuantileFinalTimeHead(config)
    _initialize_duration_head(head)
    return head


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
        _initialize_control_head(
            self.control_head, control_contract(config.control_thrust_parameterization)
        )
        _initialize_duration_head(self.final_time_head)
        # LAST (see the base class): with it built here, every parameter a `point` run has
        # draws exactly what it would have drawn, so two arms that differ only in
        # `duration_head` start from the same point head and the same backbone. What cannot
        # be matched is the TRAINING dropout stream — the second head draws inside
        # `duration()` — so the two are the same INITIALIZATION, not the same trajectory.
        self.duration_quantile_head = quantile_duration_head_for(config)

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
