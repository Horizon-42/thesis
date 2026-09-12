"""The plan head (design v5 §3, §6): the backbone's features → the operating parameters
and the next instruction, regressed on the labels the extractors read at each anchor.

A `PlanPrediction` is one target vector per flight in physical units (`labels.TARGETS`
order) and the logit of "no fix ahead" (the join is next, or the flight is already on the
final). The loss is L1 in each target's scale over the entries the truth defines, plus
the flag's cross-entropy — nothing goes through the guidance layer (§6: a network
trained through a corrective layer learns to lean on it).

**The fan (v5.3, §9 step 3(g); `plan_fan_components` = K ≥ 2).** The instruction group is
then a K-component diagonal-Gaussian MIXTURE — K means (decoded as the point head decodes
the group), K log σ in the targets' scaled units, K logits — trained by its negative
log-likelihood in place of the group's L1; the operating group and the no-fix flag are
unchanged. `values` carries the top-weight component, so everything downstream of the
point prediction (the order, the lockstep, the drawn replay) is unchanged in form; the
fan is `fan_rows`: every component as a full target vector with its weight and σ, each
flown as its own lockstep member (`strategy.lockstep_model_policy(first_component=)`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ts_transformer.config import TSConfig
from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.outputs.plan.labels import (
    CONTEXT_NEXT_IS_JOIN,
    CONTEXT_ROLLED,
    CONTEXT_SERIES,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    DISTANCE_SCALE_M,
    HEIGHT_SCALE_M,
    INSTRUCTION,
    OPERATING,
    SCALE_VECTOR,
    SCALES,
    SPEED_SCALE_MPS,
    TARGETS,
    TIME_SCALE_S,
)

#: The objective's components: the operating group less the arrival time (`state`), the
#: arrival time alone (`final_time`, so the ETA term stays readable), the next instruction
#: (`kinematic`) and the no-fix flag (`terminal`).
PLAN_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")
#: The regressed groups behind the first three components (the fourth, `terminal`, is the
#: flag's cross-entropy on every sample): ONE definition, read by the loss and by the
#: rolled-window readout that averages the loss over samples (`loss_group_carriers`).
PLAN_LOSS_GROUPS: dict[str, tuple[str, ...]] = {
    "state": tuple(name for name in OPERATING if name != "T_s"),
    "final_time": ("T_s",),
    "kinematic": INSTRUCTION,
}

#: Which decode each target gets: positive at its scale (a softplus), free at its scale, or
#: one of the heading's unit-vector pair.
_POSITIVE = {"T_s", "V_mid_mps", "V_final_mps", "next_speed_mps", "d_decel_m", "d_join_m", "remaining_m", "next_remaining_m"}
_FREE = {"h_capture_m", "next_height_m", "next_ahead_m", "next_across_m"}
_HEADING = ("next_heading_cos", "next_heading_sin")
#: Where every flight starts (the last layer's bias): a plausible plan, so the first
#: validation replays draw approaches, not loops.
_INITIAL = {
    "T_s": 300.0, "V_mid_mps": 90.0, "d_decel_m": 8_000.0, "V_final_mps": 70.0, "h_capture_m": 600.0,
    "d_join_m": 10_000.0, "remaining_m": 25_000.0,
    "next_ahead_m": 5_000.0, "next_across_m": 0.0, "next_heading_cos": 1.0, "next_heading_sin": 0.0,
    "next_speed_mps": 90.0, "next_remaining_m": 20_000.0, "next_height_m": 900.0,
}


#: The mixture's log σ (scaled units) is `FAN_LOG_SIGMA_MIN + softplus(raw)`: bounded
#: below, so a component cannot collapse onto one sample (σ → 0, the likelihood
#: unbounded), with a gradient that never dies — a hard clamp's is exactly zero outside
#: its range, and a component pushed onto the floor would have stayed there silently
#: (review 2026-09-12). e⁻⁴ is 180 m of distance, 0.9 m/s of speed, 9 m of height. No
#: upper bound: the `log σ` term of the likelihood already prices width.
FAN_LOG_SIGMA_MIN = -4.0
#: Where the components start: the point head's initial fix, spread ACROSS the course
#: this far per component (the lateral hypotheses — a base from either side — are the
#: ambiguity the fan exists for; identical starts would never separate), σ = one scale
#: unit (10 km, 50 m/s, 500 m — a wide start keeps the first gradients, z/σ per entry,
#: at the L1's order), equal weights.
FAN_INITIAL_ACROSS_SPREAD_M = 4_000.0
FAN_INITIAL_LOG_SIGMA = 0.0


@dataclass(frozen=True)
class PlanPrediction:
    """One target vector per flight, ``[B, P]`` in physical units (`TARGETS` order), and the
    no-fix flag's logit ``[B]``. Under the fan (`plan_fan_components` = K ≥ 2) also every
    component's instruction group in physical units ``[B, K, len(INSTRUCTION)]``, its log σ
    in the targets' SCALED units and the mixture logits ``[B, K]``; `values` then carries
    the top-weight component's instruction."""

    values: torch.Tensor
    join_logit: torch.Tensor
    fan_values: torch.Tensor | None = None
    fan_log_sigma: torch.Tensor | None = None
    fan_logits: torch.Tensor | None = None


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _decode_columns(raw: torch.Tensor, names: tuple[str, ...]) -> torch.Tensor:
    """Free outputs ``[N, len(names)]`` → physical values: a duration, a speed or a distance
    positive at its scale (softplus), a height or an offset free at its scale, the heading's
    pair a near-unit vector (exact away from the origin, its gradient bounded at it)."""
    columns = []
    for index, name in enumerate(names):
        column = raw[:, index]
        if name in _POSITIVE:
            columns.append(F.softplus(column) * float(SCALES[name]))
        elif name in _FREE:
            columns.append(column * float(SCALES[name]))
        else:
            columns.append(column)
    values = torch.stack(columns, dim=1)
    if _HEADING[0] not in names:        # the operating group carries no heading
        return values
    i_cos, i_sin = names.index(_HEADING[0]), names.index(_HEADING[1])
    pair = values[:, [i_cos, i_sin]]
    unit = pair / torch.sqrt(pair.pow(2).sum(dim=1, keepdim=True) + 1e-6)
    values = values.clone()
    values[:, i_cos] = unit[:, 0]
    values[:, i_sin] = unit[:, 1]
    return values


def fan_log_sigma(raw: torch.Tensor) -> torch.Tensor:
    """The mixture's log σ from its free output: bounded below by `FAN_LOG_SIGMA_MIN`,
    its gradient never zero (the ONE definition; the init inverts it)."""
    return FAN_LOG_SIGMA_MIN + F.softplus(raw)


def head_width(fan_components: int) -> int:
    """The last layer's width: the target vector and the join logit on the point head; the
    operating group, K instruction means, K log σ, K logits and the join logit under the fan."""
    if fan_components == 0:
        return len(TARGETS) + 1
    return len(OPERATING) + 2 * fan_components * len(INSTRUCTION) + fan_components + 1


def decode_raw(raw: torch.Tensor, fan_components: int = 0) -> PlanPrediction:
    """Free network outputs ``[B, head_width(K)]`` → the prediction. The point head's
    layout — the target vector then the join logit — is the one every plan checkpoint
    before 2026-09-12 stored, and is kept bit for bit."""
    if fan_components == 0:
        return PlanPrediction(values=_decode_columns(raw[:, :len(TARGETS)], TARGETS), join_logit=raw[:, len(TARGETS)])
    k, p_o, p_i = fan_components, len(OPERATING), len(INSTRUCTION)
    batch = raw.shape[0]
    operating = _decode_columns(raw[:, :p_o], OPERATING)
    offset = p_o
    means = raw[:, offset:offset + k * p_i].reshape(batch * k, p_i)
    offset += k * p_i
    log_sigma = fan_log_sigma(raw[:, offset:offset + k * p_i].reshape(batch, k, p_i))
    offset += k * p_i
    logits = raw[:, offset:offset + k]
    offset += k
    fan_values = _decode_columns(means, INSTRUCTION).reshape(batch, k, p_i)
    chosen = fan_values[torch.arange(batch, device=raw.device), logits.argmax(dim=1)]
    return PlanPrediction(
        values=torch.cat([operating, chosen], dim=1), join_logit=raw[:, offset],
        fan_values=fan_values, fan_log_sigma=log_sigma, fan_logits=logits,
    )


def _initial_bias(name: str) -> float:
    """The last layer's bias that decodes to `_INITIAL[name]`."""
    value = _INITIAL[name] / float(SCALES[name])
    if name in _POSITIVE:
        return _inverse_softplus(value)
    if name in _FREE:
        return value
    return _INITIAL[name]


class PlanOutputModel(nn.Module):
    """The backbone's flattened tokens → an MLP → the target vector and the join logit. The
    backbone's state head is discarded; the context (labels) is accepted and ignored — a
    label is never an input."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.feature_encoder = feature_encoder
        self.feature_encoder.discard_state_head()
        width = config.enc_in * config.d_model
        k = config.plan_fan_components
        self.head = nn.Sequential(
            nn.Linear(width, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, head_width(k)),
        )
        with torch.no_grad():
            last = self.head[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            if k == 0:
                for index, name in enumerate(TARGETS):
                    last.bias[index] = _initial_bias(name)
                # the join logit stays 0: "no fix ahead" is about half the anchors (§12.4)
            else:
                p_o, p_i = len(OPERATING), len(INSTRUCTION)
                for index, name in enumerate(OPERATING):
                    last.bias[index] = _initial_bias(name)
                across = INSTRUCTION.index("next_across_m")
                for component in range(k):
                    base = p_o + component * p_i
                    for index, name in enumerate(INSTRUCTION):
                        last.bias[base + index] = _initial_bias(name)
                    last.bias[base + across] += (
                        (component - (k - 1) / 2.0) * FAN_INITIAL_ACROSS_SPREAD_M / float(SCALES["next_across_m"])
                    )
                sigma_base = p_o + k * p_i
                last.bias[sigma_base:sigma_base + k * p_i] = _inverse_softplus(FAN_INITIAL_LOG_SIGMA - FAN_LOG_SIGMA_MIN)
                # the mixture logits (equal weights) and the join logit stay 0

    def forward(self, history: torch.Tensor, context: dict[str, torch.Tensor] | None = None) -> PlanPrediction:
        del context
        return decode_raw(self.head(self.feature_encoder.encode_features(history)), self.config.plan_fan_components)


def probe_plan_context(batch_size: int, device: torch.device, config: TSConfig) -> dict[str, torch.Tensor]:
    """One representative plan context for shape/throughput probes: every key a real batch
    carries (`PlanTargets.context`)."""
    del config
    targets = torch.tensor([_INITIAL[name] for name in TARGETS], dtype=torch.float32, device=device)
    return {
        CONTEXT_TARGETS: targets.expand(batch_size, -1).clone(),
        CONTEXT_VALID: torch.ones((batch_size, len(TARGETS)), dtype=torch.float32, device=device),
        CONTEXT_NEXT_IS_JOIN: torch.zeros(batch_size, dtype=torch.float32, device=device),
        CONTEXT_SERIES: torch.zeros(batch_size, dtype=torch.int64, device=device),
        CONTEXT_ROLLED: torch.zeros(batch_size, dtype=torch.float32, device=device),
    }


def plan_loss_components(
    prediction: PlanPrediction, flight_weights: torch.Tensor, config: TSConfig,
    context: dict[str, torch.Tensor] | None,
) -> LossComponents:
    """L1 regression of the target vector against the anchor's labels, in each target's
    scale, over the entries the truth defines; the no-fix flag's binary cross-entropy on
    every flight. Every group is a flight-weighted mean over the flights that carry it —
    a censored entry contributes nothing, never zero. **Under the fan
    (`plan_fan_components` ≥ 2) the `kinematic` component is the instruction group's
    mixture NEGATIVE LOG-LIKELIHOOD (`mixture_nll`), not its L1** — same name, same
    carriers, a different quantity (often negative): a fan run's objective and
    `kinematic` are comparable within the run, never with a point head's."""
    if context is None or CONTEXT_TARGETS not in context:
        raise ValueError("the plan loss needs the per-anchor label context")
    values = prediction.values
    target = context[CONTEXT_TARGETS].to(values.dtype)
    valid = context[CONTEXT_VALID].to(values.dtype)
    weight = flight_weights.to(values.dtype)
    scale = torch.as_tensor(SCALE_VECTOR, dtype=values.dtype, device=values.device)
    residual = (values - target).abs() / scale          # [B, P]

    def group(names: tuple[str, ...]) -> torch.Tensor:
        index = [TARGETS.index(name) for name in names]
        r, v = residual[:, index], valid[:, index]
        per_flight = (r * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)
        carries = (v.sum(dim=1) > 0).to(values.dtype) * weight
        return (per_flight * carries).sum() / carries.sum().clamp(min=1e-6)

    operating = group(PLAN_LOSS_GROUPS["state"])
    arrival = group(PLAN_LOSS_GROUPS["final_time"])
    instruction = (
        group(PLAN_LOSS_GROUPS["kinematic"]) if prediction.fan_logits is None
        else mixture_nll(prediction, target, valid, weight, scale)
    )
    join_target = context[CONTEXT_NEXT_IS_JOIN].to(values.dtype)
    bce = F.binary_cross_entropy_with_logits(prediction.join_logit.to(values.dtype), join_target, reduction="none")
    join = (bce * weight).sum() / weight.sum().clamp(min=1e-6)
    return LossComponents(
        state=operating * config.plan_operating_loss_weight,
        final_time=arrival * config.plan_operating_loss_weight,
        kinematic=instruction * config.plan_instruction_loss_weight,
        terminal=join * config.plan_instruction_loss_weight,
    )


def mixture_nll(
    prediction: PlanPrediction, target: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    """The instruction group's negative log-likelihood under the fan's K-component diagonal
    Gaussian mixture, in the targets' scaled units, divided by the number of defined entries
    (the labels write all seven or none, so this is the L1 group's per-entry magnitude, kept
    for comparability of the term's size), a flight-weighted mean over the samples that
    carry the group — a sample with no fix ahead carries nothing. The `kinematic` component
    under `plan_fan_components` ≥ 2, where the point head's L1 was."""
    index = [TARGETS.index(name) for name in INSTRUCTION]
    t = (target[:, index] / scale[index]).unsqueeze(1)                       # [B, 1, P_i]
    v = valid[:, index].unsqueeze(1)                                           # [B, 1, P_i]
    mu = prediction.fan_values.to(target.dtype) / scale[index]                # [B, K, P_i]
    log_sigma = prediction.fan_log_sigma.to(target.dtype)
    z = (t - mu) / log_sigma.exp()
    log_density = -(0.5 * z.square() + log_sigma + 0.5 * math.log(2.0 * math.pi))
    component_log_lik = (log_density * v).sum(dim=2) + F.log_softmax(prediction.fan_logits.to(target.dtype), dim=1)
    n_valid = valid[:, index].sum(dim=1)                                        # [B]
    nll = -torch.logsumexp(component_log_lik, dim=1) / n_valid.clamp(min=1.0)
    carries = (n_valid > 0).to(target.dtype) * weight
    return (nll * carries).sum() / carries.sum().clamp(min=1e-6)


def loss_group_carriers(valid: np.ndarray) -> dict[str, int]:
    """How many of a batch's ``[B, P]`` validity rows carry each component: a regressed
    group is carried where any of its entries is defined, the flag by every row. What
    `plan_loss_components` averages each component over, so a reader averaging batches
    weights each by its carriers rather than its size (a batch of "no fix ahead" samples
    carries no `kinematic` at all)."""
    valid = np.asarray(valid)
    carriers = {
        name: int(np.count_nonzero(valid[:, [TARGETS.index(target) for target in names]].sum(axis=1) > 0))
        for name, names in PLAN_LOSS_GROUPS.items()
    }
    carriers["terminal"] = int(len(valid))
    return carriers


def prediction_rows(prediction: PlanPrediction) -> tuple[np.ndarray, np.ndarray]:
    """The prediction as numpy: the target vectors ``[B, P]`` and the join probabilities ``[B]``."""
    values = prediction.values.detach().cpu().numpy().astype(np.float64)
    probability = torch.sigmoid(prediction.join_logit).detach().cpu().numpy().astype(np.float64)
    return values, probability


def fan_rows(prediction: PlanPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fan as numpy: every component as a FULL target vector ``[B, K, P]`` (the operating
    group shared, the instruction group the component's own), the mixture weights
    ``[B, K]`` and the instruction group's σ in physical units ``[B, K, len(INSTRUCTION)]``.
    A component's vector goes through `order_from_prediction` like the point prediction."""
    if prediction.fan_logits is None:
        raise ValueError("the point head has no fan (plan_fan_components = 0)")
    values, _probability = prediction_rows(prediction)
    fan = prediction.fan_values.detach().cpu().numpy().astype(np.float64)
    index = [TARGETS.index(name) for name in INSTRUCTION]
    full = np.repeat(values[:, None, :], fan.shape[1], axis=1)
    full[:, :, index] = fan
    weights = torch.softmax(prediction.fan_logits, dim=1).detach().cpu().numpy().astype(np.float64)
    sigma = np.exp(prediction.fan_log_sigma.detach().cpu().numpy().astype(np.float64)) * np.array(
        [SCALES[name] for name in INSTRUCTION], dtype=np.float64,
    )
    return full, weights, sigma


__all__ = [
    "PLAN_LOSS_COMPONENT_NAMES", "PLAN_LOSS_GROUPS", "PlanOutputModel", "PlanPrediction", "decode_raw",
    "fan_log_sigma", "fan_rows", "head_width", "loss_group_carriers", "mixture_nll", "plan_loss_components",
    "prediction_rows",
    "probe_plan_context",
    "DISTANCE_SCALE_M", "HEIGHT_SCALE_M", "SPEED_SCALE_MPS", "TIME_SCALE_S",
]
