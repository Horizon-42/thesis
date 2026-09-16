"""Gradient clipping and audit metrics for deterministic control training.

This module is deliberately independent of the training loop. A positive clip norm applies
one exact global L2 cap and records the pre-clip scale by model subsystem plus bounded-
control saturation. The model, objective, and optimizer remain owned by their existing
modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn

from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.envelope import control_contract


GRADIENT_GROUPS = ("backbone", "control_head", "final_time_head")
SATURATION_THRESHOLD_FRACTION = 0.01


def saturation_labels(parameterization: str) -> tuple[str, ...]:
    """The per-column keys of an epoch's ``control_saturation.by_control``
    (``ControlContract.saturation_labels``)."""
    return control_contract(parameterization).saturation_labels


def _gradient_group(parameter_name: str) -> str:
    if parameter_name.startswith("control_head."):
        return "control_head"
    # B1.b's second head is a duration head too, and it is grouped WITH the point head:
    # without this its gradients would be counted as the BACKBONE's, which is worse. The
    # cost is accepted and stated — on a `two-head` run this norm is the sum of both heads'
    # and cannot be read apart. Splitting them would need a fourth `GRADIENT_GROUPS` entry,
    # i.e. a new key in EVERY run's history, and that is not free.
    # (`global_duration_head.` was here and is gone: it matched no parameter anywhere in the
    # tree, `archive/` included.)
    if parameter_name.startswith(("final_time_head.", "duration_quantile_head.")):
        return "final_time_head"
    return "backbone"


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """Return finite global/group L2 norms using float64 accumulation."""
    device = next(model.parameters()).device
    squared = {
        name: torch.zeros((), dtype=torch.float64, device=device)
        for name in GRADIENT_GROUPS
    }
    found = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach().to(dtype=torch.float64)
        squared[_gradient_group(name)] += gradient.square().sum()
    if not found:
        raise RuntimeError("gradient diagnostics found no model gradients")
    result = {name: math.sqrt(float(value)) for name, value in squared.items()}
    result["total"] = math.sqrt(sum(value * value for value in result.values()))
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite control gradient detected before optimizer step")
    return result


def clip_gradients_by_global_norm(
    model: nn.Module, max_norm: float
) -> tuple[dict[str, float], float]:
    """Apply one exact global L2 cap and return the pre-clip audit snapshot.

    The second return value is the scaling coefficient actually applied — 1.0 when the
    pre-clip norm was already under the cap, so ``coefficient < 1.0`` IS "this batch was
    clipped".
    """
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("gradient clip max_norm must be positive and finite")
    norms = gradient_norms(model)
    triggered = norms["total"] > max_norm
    coefficient = max_norm / (norms["total"] + 1e-12) if triggered else 1.0
    if triggered:
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(coefficient)
    return norms, coefficient


@dataclass
class ControlTrainingDiagnosticsAccumulator:
    """Aggregate batch diagnostics into one JSON-serializable epoch record."""

    max_norm: float
    #: The column labels (:func:`saturation_labels`), one per control.
    control_names: tuple[str, ...]
    batch_count: int = 0
    clipped_batches: int = 0
    gradient_sum: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in (*GRADIENT_GROUPS, "total")}
    )
    gradient_max: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in (*GRADIENT_GROUPS, "total")}
    )
    coefficient_sum: float = 0.0
    coefficient_min: float = 1.0
    saturated_by_control: list[int] = field(init=False)
    controls_by_control: int = 0

    def __post_init__(self) -> None:
        self.saturated_by_control = [0 for _name in self.control_names]

    def record_prediction(
        self,
        prediction: ControlPrediction,
        dynamics: dict[str, torch.Tensor],
    ) -> None:
        lower = dynamics["control_lower"].unsqueeze(1)
        upper = dynamics["control_upper"].unsqueeze(1)
        unit = (prediction.controls.detach() - lower) / (upper - lower)
        saturated = (unit <= SATURATION_THRESHOLD_FRACTION) | (
            unit >= 1.0 - SATURATION_THRESHOLD_FRACTION
        )
        counts = saturated.sum(dim=(0, 1)).to(device="cpu").tolist()
        self.saturated_by_control = [
            old + int(new)
            for old, new in zip(self.saturated_by_control, counts)
        ]
        self.controls_by_control += prediction.controls.shape[0] * prediction.controls.shape[1]

    def record_gradients_and_clip(self, model: nn.Module) -> None:
        norms, coefficient = clip_gradients_by_global_norm(model, self.max_norm)
        self.batch_count += 1
        self.clipped_batches += int(coefficient < 1.0)
        for name, value in norms.items():
            self.gradient_sum[name] += value
            self.gradient_max[name] = max(self.gradient_max[name], value)
        self.coefficient_sum += coefficient
        self.coefficient_min = min(self.coefficient_min, coefficient)

    def summary(self) -> dict[str, object]:
        if self.batch_count <= 0 or self.controls_by_control <= 0:
            raise RuntimeError("control training diagnostics contain no batches")
        total_controls = self.controls_by_control * len(self.control_names)
        return {
            "gradient_norm_pre_clip": {
                "mean": {
                    name: self.gradient_sum[name] / self.batch_count
                    for name in (*GRADIENT_GROUPS, "total")
                },
                "max": dict(self.gradient_max),
            },
            "clip": {
                "max_norm": self.max_norm,
                "batches": self.batch_count,
                "triggered_batches": self.clipped_batches,
                "trigger_rate": self.clipped_batches / self.batch_count,
                "coefficient_mean": self.coefficient_sum / self.batch_count,
                "coefficient_min": self.coefficient_min,
            },
            "control_saturation": {
                "threshold_fraction": SATURATION_THRESHOLD_FRACTION,
                "overall_rate": sum(self.saturated_by_control) / total_controls,
                "by_control": {
                    name: count / self.controls_by_control
                    for name, count in zip(
                        self.control_names, self.saturated_by_control, strict=True
                    )
                },
            },
        }
