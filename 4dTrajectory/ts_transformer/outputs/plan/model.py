"""The plan head (design v5 §3, §6): the backbone's features → the operating parameters
and the next instruction, regressed on the labels the extractors read at each anchor.

A `PlanPrediction` is one target vector per flight in physical units (`labels.TARGETS`
order) and the logit of "no fix ahead" (the join is next, or the flight is already on the
final). The loss is L1 in each target's scale over the entries the truth defines, plus
the flag's cross-entropy — nothing goes through the guidance layer (§6: a network
trained through a corrective layer learns to lean on it).
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
    CONTEXT_SERIES,
    CONTEXT_TARGETS,
    CONTEXT_VALID,
    DISTANCE_SCALE_M,
    HEIGHT_SCALE_M,
    INSTRUCTION,
    OPERATING,
    SCALE_VECTOR,
    SPEED_SCALE_MPS,
    TARGETS,
    TIME_SCALE_S,
)

#: The objective's components: the operating group less the arrival time (`state`), the
#: arrival time alone (`final_time`, so the ETA term stays readable), the next instruction
#: (`kinematic`) and the no-fix flag (`terminal`).
PLAN_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")

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


@dataclass(frozen=True)
class PlanPrediction:
    """One target vector per flight, ``[B, P]`` in physical units (`TARGETS` order), and the
    no-fix flag's logit ``[B]``."""

    values: torch.Tensor
    join_logit: torch.Tensor


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def decode_raw(raw: torch.Tensor) -> PlanPrediction:
    """Free network outputs ``[B, P + 1]`` → the physical target vector and the join logit:
    a duration, a speed or a distance is positive at its scale (softplus), a height or an
    offset free at its scale, the heading's pair a near-unit vector (exact away from the
    origin, its gradient bounded at it)."""
    columns = []
    scale = torch.as_tensor(SCALE_VECTOR, dtype=raw.dtype, device=raw.device)
    for index, name in enumerate(TARGETS):
        column = raw[:, index]
        if name in _POSITIVE:
            columns.append(F.softplus(column) * scale[index])
        elif name in _FREE:
            columns.append(column * scale[index])
        else:
            columns.append(column)
    values = torch.stack(columns, dim=1)
    i_cos, i_sin = TARGETS.index(_HEADING[0]), TARGETS.index(_HEADING[1])
    pair = values[:, [i_cos, i_sin]]
    unit = pair / torch.sqrt(pair.pow(2).sum(dim=1, keepdim=True) + 1e-6)
    values = values.clone()
    values[:, i_cos] = unit[:, 0]
    values[:, i_sin] = unit[:, 1]
    return PlanPrediction(values=values, join_logit=raw[:, len(TARGETS)])


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
        self.head = nn.Sequential(
            nn.Linear(width, config.d_model), nn.GELU(), nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model), nn.GELU(),
            nn.Linear(config.d_model, len(TARGETS) + 1),
        )
        with torch.no_grad():
            last = self.head[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            for index, name in enumerate(TARGETS):
                value = _INITIAL[name] / float(SCALE_VECTOR[index])
                if name in _POSITIVE:
                    last.bias[index] = _inverse_softplus(value)
                elif name in _FREE:
                    last.bias[index] = value
                else:
                    last.bias[index] = _INITIAL[name]
            last.bias[len(TARGETS)] = 0.0      # "no fix ahead" is about half the anchors (§12.4)

    def forward(self, history: torch.Tensor, context: dict[str, torch.Tensor] | None = None) -> PlanPrediction:
        del context
        return decode_raw(self.head(self.feature_encoder.encode_features(history)))


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
    }


def plan_loss_components(
    prediction: PlanPrediction, flight_weights: torch.Tensor, config: TSConfig,
    context: dict[str, torch.Tensor] | None,
) -> LossComponents:
    """L1 regression of the target vector against the anchor's labels, in each target's
    scale, over the entries the truth defines; the no-fix flag's binary cross-entropy on
    every flight. Every group is a flight-weighted mean over the flights that carry it —
    a censored entry contributes nothing, never zero."""
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

    operating = group(tuple(name for name in OPERATING if name != "T_s"))
    arrival = group(("T_s",))
    instruction = group(INSTRUCTION)
    join_target = context[CONTEXT_NEXT_IS_JOIN].to(values.dtype)
    bce = F.binary_cross_entropy_with_logits(prediction.join_logit.to(values.dtype), join_target, reduction="none")
    join = (bce * weight).sum() / weight.sum().clamp(min=1e-6)
    return LossComponents(
        state=operating * config.plan_operating_loss_weight,
        final_time=arrival * config.plan_operating_loss_weight,
        kinematic=instruction * config.plan_instruction_loss_weight,
        terminal=join * config.plan_instruction_loss_weight,
    )


def prediction_rows(prediction: PlanPrediction) -> tuple[np.ndarray, np.ndarray]:
    """The prediction as numpy: the target vectors ``[B, P]`` and the join probabilities ``[B]``."""
    values = prediction.values.detach().cpu().numpy().astype(np.float64)
    probability = torch.sigmoid(prediction.join_logit).detach().cpu().numpy().astype(np.float64)
    return values, probability


__all__ = [
    "PLAN_LOSS_COMPONENT_NAMES", "PlanOutputModel", "PlanPrediction", "decode_raw",
    "plan_loss_components", "prediction_rows", "probe_plan_context",
    "DISTANCE_SCALE_M", "HEIGHT_SCALE_M", "SPEED_SCALE_MPS", "TIME_SCALE_S",
]
