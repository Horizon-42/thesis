"""Gradient clipping and audit metrics for deterministic control training.

This module is deliberately independent of the training loop. A positive clip norm enables
an explicit clipping policy and records the pre-clip scale by model subsystem plus bounded-
control saturation. The model, objective, and optimizer remain owned by their existing
modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn

from config import (
    CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
    CONTROL_GRADIENT_CLIP_GLOBAL,
    CONTROL_GRADIENT_CLIP_POLICIES,
)
from prediction_outputs import CONTROL_NAMES, ControlPrediction


GRADIENT_GROUPS = ("backbone", "control_head", "final_time_head")
SATURATION_THRESHOLD_FRACTION = 0.01


@dataclass(frozen=True)
class GradientClipScope:
    """One independently scaled gradient vector in a clipping policy."""

    groups: tuple[str, ...]
    capped: bool


_CLIP_SCOPES = {
    CONTROL_GRADIENT_CLIP_GLOBAL: {
        "global": GradientClipScope(groups=GRADIENT_GROUPS, capped=True),
    },
    CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED: {
        "control_backbone": GradientClipScope(
            groups=("backbone", "control_head"), capped=True
        ),
        "final_time_head": GradientClipScope(
            groups=("final_time_head",), capped=False
        ),
    },
}


def _gradient_group(parameter_name: str) -> str:
    if parameter_name.startswith("control_head."):
        return "control_head"
    if parameter_name.startswith(("final_time_head.", "global_duration_head.")):
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
) -> tuple[dict[str, float], bool]:
    """Apply one exact global L2 cap and return the pre-clip audit snapshot."""
    norms, scopes = clip_gradients_by_policy(
        model, max_norm=max_norm, policy=CONTROL_GRADIENT_CLIP_GLOBAL
    )
    return norms, bool(scopes["global"]["triggered"])


def clip_gradients_by_policy(
    model: nn.Module,
    *,
    max_norm: float,
    policy: str,
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Apply a named group policy and return the pre-clip audit snapshot."""
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("gradient clip max_norm must be positive and finite")
    if policy not in CONTROL_GRADIENT_CLIP_POLICIES:
        raise ValueError(
            f"unknown gradient clip policy {policy!r}; expected one of "
            f"{CONTROL_GRADIENT_CLIP_POLICIES}"
        )
    norms = gradient_norms(model)
    scope_results: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for scope_name, scope in _CLIP_SCOPES[policy].items():
            pre_clip_norm = math.sqrt(
                sum(norms[group] * norms[group] for group in scope.groups)
            )
            triggered = scope.capped and pre_clip_norm > max_norm
            coefficient = (
                max_norm / (pre_clip_norm + 1e-12) if triggered else 1.0
            )
            if triggered:
                for parameter_name, parameter in model.named_parameters():
                    if (
                        parameter.grad is not None
                        and _gradient_group(parameter_name) in scope.groups
                    ):
                        parameter.grad.mul_(coefficient)
            scope_results[scope_name] = {
                "groups": list(scope.groups),
                "capped": scope.capped,
                "pre_clip_norm": pre_clip_norm,
                "triggered": triggered,
                "coefficient": coefficient,
            }
    return norms, scope_results


@dataclass
class ControlTrainingDiagnosticsAccumulator:
    """Aggregate batch diagnostics into one JSON-serializable epoch record."""

    max_norm: float
    policy: str = CONTROL_GRADIENT_CLIP_GLOBAL
    batch_count: int = 0
    clipped_batches: int = 0
    gradient_sum: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in (*GRADIENT_GROUPS, "total")}
    )
    gradient_max: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in (*GRADIENT_GROUPS, "total")}
    )
    clip_scope_sum: dict[str, float] = field(default_factory=dict)
    clip_scope_max: dict[str, float] = field(default_factory=dict)
    clip_scope_coefficient_sum: dict[str, float] = field(default_factory=dict)
    clip_scope_coefficient_min: dict[str, float] = field(default_factory=dict)
    clip_scope_triggered: dict[str, int] = field(default_factory=dict)
    saturated_by_control: list[int] = field(
        default_factory=lambda: [0 for _name in CONTROL_NAMES]
    )
    controls_by_control: int = 0

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
        norms, scopes = clip_gradients_by_policy(
            model, max_norm=self.max_norm, policy=self.policy
        )
        self.batch_count += 1
        self.clipped_batches += int(
            any(bool(scope["triggered"]) for scope in scopes.values())
        )
        for name, value in norms.items():
            self.gradient_sum[name] += value
            self.gradient_max[name] = max(self.gradient_max[name], value)
        for name, scope in scopes.items():
            pre_clip_norm = float(scope["pre_clip_norm"])
            coefficient = float(scope["coefficient"])
            self.clip_scope_sum[name] = self.clip_scope_sum.get(name, 0.0) + pre_clip_norm
            self.clip_scope_max[name] = max(
                self.clip_scope_max.get(name, 0.0), pre_clip_norm
            )
            self.clip_scope_coefficient_sum[name] = (
                self.clip_scope_coefficient_sum.get(name, 0.0) + coefficient
            )
            self.clip_scope_coefficient_min[name] = min(
                self.clip_scope_coefficient_min.get(name, 1.0), coefficient
            )
            self.clip_scope_triggered[name] = (
                self.clip_scope_triggered.get(name, 0) + int(bool(scope["triggered"]))
            )

    def summary(self) -> dict[str, object]:
        if self.batch_count <= 0 or self.controls_by_control <= 0:
            raise RuntimeError("control training diagnostics contain no batches")
        total_controls = self.controls_by_control * len(CONTROL_NAMES)
        return {
            "gradient_norm_pre_clip": {
                "mean": {
                    name: self.gradient_sum[name] / self.batch_count
                    for name in (*GRADIENT_GROUPS, "total")
                },
                "max": dict(self.gradient_max),
            },
            "clip": {
                "policy": self.policy,
                "max_norm": self.max_norm,
                "batches": self.batch_count,
                "triggered_batches": self.clipped_batches,
                "trigger_rate": self.clipped_batches / self.batch_count,
                "scopes": {
                    name: {
                        "groups": list(scope.groups),
                        "capped": scope.capped,
                        "pre_clip_norm_mean": (
                            self.clip_scope_sum[name] / self.batch_count
                        ),
                        "pre_clip_norm_max": self.clip_scope_max[name],
                        "triggered_batches": self.clip_scope_triggered[name],
                        "trigger_rate": (
                            self.clip_scope_triggered[name] / self.batch_count
                        ),
                        "coefficient_mean": (
                            self.clip_scope_coefficient_sum[name] / self.batch_count
                        ),
                        "coefficient_min": self.clip_scope_coefficient_min[name],
                    }
                    for name, scope in _CLIP_SCOPES[self.policy].items()
                },
            },
            "control_saturation": {
                "threshold_fraction": SATURATION_THRESHOLD_FRACTION,
                "overall_rate": sum(self.saturated_by_control) / total_controls,
                "by_control": {
                    name: count / self.controls_by_control
                    for name, count in zip(CONTROL_NAMES, self.saturated_by_control)
                },
            },
        }
