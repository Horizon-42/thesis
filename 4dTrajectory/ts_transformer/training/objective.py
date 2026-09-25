"""The training objective: what a prediction is scored against, and how.

What every path shares is built here — the procedure penalty and its dual multipliers,
the batch movers — and the dispatch that hands a batch to the configured
path's own objective (`outputs/<path>/loss`). It is deliberately NOT part of the training
loop: `train.fit_model` is
one consumer, and the batch-size probe (`batching`), the capacity-ceiling runner and the
benchmark are others that want the objective without the loop around it. That is the
same reason `batch_contract` exists, one layer down.

Adding a term: register its name in the path's ``OutputStrategy.loss_component_names`` as
well as putting it in the objective's ``extras``, or the first batch raises ``KeyError`` —
after the slow dataset build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.data.channels import IDX
from ts_transformer.config import PROCEDURE_LATERAL_SCALE_M, PROCEDURE_VERTICAL_SCALE_M, TSConfig
from ts_transformer.data.dataset import Normalizer
from ts_transformer.geometry.final_approach_geometry import corridor_violations, runway_axes, truth_final_gate
from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.state.model import StatePrediction
from ts_transformer.outputs import strategy


def target_contract(config: TSConfig) -> str:
    """The output path's supervision contract, stamped into every checkpoint."""
    return strategy(config).target_contract(config)


def loss_component_names(config: TSConfig) -> tuple[str, ...]:
    """Every component the objective reports under this config, in a fixed order."""
    return strategy(config).loss_component_names(config)


def move_dynamics(
    dynamics: dict[str, torch.Tensor] | None, device: torch.device
) -> dict[str, torch.Tensor] | None:
    if dynamics is None:
        return None
    return {name: value.to(device) for name, value in dynamics.items()}


def move_fixed_dt_supervision(
    supervision: FixedDTControlSupervision | None,
    device: torch.device,
) -> FixedDTControlSupervision | None:
    return None if supervision is None else supervision.to(device)


@dataclass
class ProcedureMultipliers:
    """The procedure penalty's weights λ, one per constraint family.

    Fixed at the configured weights when ``procedure_loss_dual_step`` is zero; otherwise
    the dual variables of ``min L_pred  s.t.  violation rate ≤ ε``, raised once per epoch
    by the measured excess (dual ascent on the RATE, with the hinge² as the primal
    surrogate), so the weight is found rather than swept.
    """

    lateral: float
    vertical: float

    @classmethod
    def from_config(cls, config: TSConfig) -> "ProcedureMultipliers | None":
        if not config.procedure_loss_active:
            return None
        return cls(
            lateral=config.procedure_loss_lateral_weight,
            vertical=config.procedure_loss_vertical_weight,
        )

    def update(self, lateral_rate: float, vertical_rate: float, config: TSConfig) -> None:
        step = config.procedure_loss_dual_step
        if step <= 0.0:
            return
        self.lateral = max(0.0, self.lateral + step * (lateral_rate - config.procedure_loss_epsilon))
        self.vertical = max(0.0, self.vertical + step * (vertical_rate - config.procedure_loss_epsilon))

    def to_dict(self) -> dict[str, float]:
        return {"lateral": self.lateral, "vertical": self.vertical}


PROCEDURE_DIAGNOSTICS = (
    "procedure_gated_rows", "procedure_lateral_violations", "procedure_vertical_violations",
)



def procedure_loss(
    predicted_states: torch.Tensor,
    target_states: torch.Tensor,
    point_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None,
    multipliers: ProcedureMultipliers | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The final-approach penalty, PER FLIGHT ``[B]``: metres outside the corridor /
    glidepath window, squared at the runway scale, on rows where the OBSERVED track is
    established (the truth gate), weighted by λ — and the counts the dual update needs.
    ``predicted_states`` are normalized rows aligned index-for-index with ``target_states``
    (the state output, or a control rollout's segment endpoints).

    Rows are paired by index: the same physical time under the ``full``/``window`` grids,
    the same progress fraction under ``normalized``. The gate is decided by the truth row
    (and a predicted row already past the threshold, ``d ≤ 0``, is not charged — it is
    truncated at inference), the violation measured on the predicted row, so a model that
    is early or late onto the final is charged where the flight actually was on it.
    """
    if not config.procedure_loss_active:
        return point_weights.new_zeros(point_weights.shape[0]), {}
    if dynamics is None:
        raise ValueError(
            "the procedure loss needs the per-flight final-approach context in the batch"
        )
    dtype, device = predicted_states.dtype, predicted_states.device
    mean = torch.as_tensor(normalizer.mean, dtype=dtype, device=device)
    std = torch.as_tensor(normalizer.std, dtype=dtype, device=device)
    predicted = predicted_states * std + mean
    truth = target_states.to(dtype) * std + mean
    psi = dynamics["runway_heading_rad"].to(dtype)
    tan_gpa = dynamics["glidepath_tan"].to(dtype)
    valid = point_weights > 0.0
    d_truth, xt_truth = runway_axes(truth[..., IDX["e"]], truth[..., IDX["n"]], psi)
    d_pred, xt_pred = runway_axes(predicted[..., IDX["e"]], predicted[..., IDX["n"]], psi)
    gate = truth_final_gate(d_truth, xt_truth, valid) & (d_pred > 0.0)
    lateral_m, vertical_m = corridor_violations(d_pred, xt_pred, predicted[..., IDX["u"]], tan_gpa)
    gate_weight = gate.to(dtype)
    gated_rows = gate_weight.sum(dim=1)
    lateral_sq = ((lateral_m / PROCEDURE_LATERAL_SCALE_M) ** 2 * gate_weight).sum(dim=1)
    vertical_sq = ((vertical_m / PROCEDURE_VERTICAL_SCALE_M) ** 2 * gate_weight).sum(dim=1)
    per_flight_lateral = lateral_sq / gated_rows.clamp(min=1.0)
    per_flight_vertical = vertical_sq / gated_rows.clamp(min=1.0)
    weights = multipliers or ProcedureMultipliers.from_config(config)
    term = weights.lateral * per_flight_lateral + weights.vertical * per_flight_vertical
    diagnostics = {
        "procedure_gated_rows": gate.sum().detach(),
        "procedure_lateral_violations": ((lateral_m > 0.0) & gate).sum().detach(),
        "procedure_vertical_violations": ((vertical_m > 0.0) & gate).sum().detach(),
    }
    return term, diagnostics


def prediction_loss_components(
    prediction: Any,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
    dense_supervision: FixedDTControlSupervision | None = None,
    *,
    multipliers: ProcedureMultipliers | None = None,
) -> LossComponents:
    """Dispatch the configured output path to its own objective."""
    return strategy(config).loss(
        prediction,
        normalized_anchor_state,
        target_states,
        state_weights,
        target_final_time_s,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense_supervision,
        multipliers=multipliers,
    )


def prediction_loss(
    prediction: StatePrediction | ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
    dense_supervision: FixedDTControlSupervision | None = None,
) -> torch.Tensor:
    """Airport-macro state/time/physics loss, one sample per flight and epoch."""
    # Weights are normalized to mean one across the complete epoch. Keeping the minibatch
    # denominator independent of its airport composition gives an unbiased stochastic
    # estimate of that fixed airport-macro objective.
    return prediction_loss_components(
        prediction,
        normalized_anchor_state,
        target_states,
        state_weights,
        target_final_time_s,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense_supervision,
    ).total
