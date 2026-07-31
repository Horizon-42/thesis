"""Deployable trim-baseline plus residual deterministic control strategy.

The baseline uses only the observed anchor state and per-flight aerodynamic parameters.
It approximately holds speed, heading, and flight-path angle under the shared point-mass
dynamics. The learned head predicts zero-centred logits around that baseline; no future
trajectory value is consumed by the model forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from aerodynamic_model.torch_dynamics import (
    GRAVITY_MPS2,
    aerodynamic_coefficients,
    isa_density,
)
from config import TSConfig
from control_models import ControlFeatureModel
from prediction_outputs import CONTROL_NAMES, ControlPrediction, FinalTimeHead


TRIM_INTERIOR_FRACTION = 1e-6


@dataclass(frozen=True)
class TrimControlBaseline:
    """Constrained trim controls plus audit flags for an anchor batch."""

    controls: torch.Tensor
    raw_controls: torch.Tensor
    clamped: torch.Tensor
    stalled: torch.Tensor


def trim_control_baseline(
    initial_state: torch.Tensor,
    aero_params: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> TrimControlBaseline:
    """Compute ``(thrust, bank, load)`` that locally holds ``V/psi/gamma``.

    For bank zero, ``n=cos(gamma)`` makes the nominal gamma derivative zero. Drag is
    evaluated with that load command using the exact shared aerodynamic equations, and
    ``T=D+m*g*sin(gamma)`` makes the nominal speed derivative zero. Infeasible values are
    explicitly clamped to each aircraft's physical control envelope.
    """
    if initial_state.shape[-1] != 7:
        raise ValueError("trim initial_state must end in 7 geodetic state values")
    if aero_params.shape[-1] != 6:
        raise ValueError("trim aero_params must end in 6 aerodynamic values")
    if lower.shape != upper.shape or lower.shape[-1] != len(CONTROL_NAMES):
        raise ValueError("trim control bounds must be aligned [...,3] tensors")

    altitude = initial_state[..., 2]
    speed = initial_state[..., 3]
    gamma = initial_state[..., 5]
    mass = initial_state[..., 6]
    load = torch.cos(gamma)
    density = isa_density(altitude)
    _cl, cd, stalled = aerodynamic_coefficients(
        load, speed, mass, density, aero_params
    )
    drag = 0.5 * density * speed.square() * cd * aero_params[..., 0]
    thrust = drag + mass * GRAVITY_MPS2 * torch.sin(gamma)
    raw = torch.stack((thrust, torch.zeros_like(thrust), load), dim=-1)
    constrained = torch.maximum(torch.minimum(raw, upper), lower)
    return TrimControlBaseline(
        controls=constrained,
        raw_controls=raw,
        clamped=raw.ne(constrained),
        stalled=stalled,
    )


def interior_trim_controls(
    trim_controls: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a finite-logit trim point and its normalized position in the bounds."""
    span = upper - lower
    unit = ((trim_controls - lower) / span).clamp(
        min=TRIM_INTERIOR_FRACTION,
        max=1.0 - TRIM_INTERIOR_FRACTION,
    )
    return lower + unit * span, unit


def bounded_trim_residual_controls(
    residual_logits: torch.Tensor,
    trim_controls: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Shift bounded-control logits so zero residual reproduces the trim baseline."""
    _interior_trim, trim_unit = interior_trim_controls(trim_controls, lower, upper)
    unit = torch.sigmoid(torch.logit(trim_unit).unsqueeze(1) + residual_logits)
    return lower.unsqueeze(1) + unit * (upper - lower).unsqueeze(1)


class TrimResidualControlHead(nn.Module):
    """Factorized-duration head whose zero control logits mean anchor-state trim."""

    def __init__(self, input_dim: int, n_segments: int):
        super().__init__()
        self.n_segments = n_segments
        self.control_projection = nn.Linear(
            input_dim, n_segments * len(CONTROL_NAMES)
        )
        self.duration_projection = nn.Linear(input_dim, n_segments)
        with torch.no_grad():
            self.control_projection.weight.zero_()
            self.control_projection.bias.zero_()
            self.duration_projection.weight.zero_()
            self.duration_projection.bias.zero_()

    def forward(
        self,
        features: torch.Tensor,
        final_time_s: torch.Tensor,
        *,
        initial_state: torch.Tensor,
        aero_params: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> ControlPrediction:
        dtype = features.dtype
        baseline = trim_control_baseline(
            initial_state.to(dtype=dtype),
            aero_params.to(dtype=dtype),
            lower.to(dtype=dtype),
            upper.to(dtype=dtype),
        )
        batch = features.shape[0]
        residual_logits = self.control_projection(features).view(
            batch, self.n_segments, len(CONTROL_NAMES)
        )
        controls = bounded_trim_residual_controls(
            residual_logits,
            baseline.controls,
            lower.to(dtype=dtype),
            upper.to(dtype=dtype),
        )
        fractions = torch.softmax(self.duration_projection(features), dim=-1)
        durations = fractions * final_time_s.unsqueeze(-1)
        return ControlPrediction(
            controls=controls,
            segment_durations=durations,
            final_time_s=final_time_s,
        )


class TrimResidualControlOutputModel(ControlFeatureModel):
    """Backbone-agnostic trim-residual output model."""

    def __init__(self, config: TSConfig, feature_encoder: nn.Module):
        super().__init__(config, feature_encoder)
        self.final_time_head = FinalTimeHead(config)
        self.control_head = TrimResidualControlHead(
            config.d_model, int(config.n_segments)
        )
        with torch.no_grad():
            final_layer = self.final_time_head.network[-1]
            final_layer.weight.zero_()
            final_layer.bias.zero_()

    def forward(
        self,
        history: torch.Tensor,
        dynamics: dict[str, torch.Tensor],
    ) -> ControlPrediction:
        features = self.fused_features(history, dynamics)
        return self.control_head(
            features,
            self.final_time_head(history),
            initial_state=dynamics["initial_state"],
            aero_params=dynamics["aero_params"],
            lower=dynamics["control_lower"],
            upper=dynamics["control_upper"],
        )
