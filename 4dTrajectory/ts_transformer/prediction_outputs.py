"""Typed state/control prediction contracts and replaceable output layers.

The selected output strategy is explicit in :class:`config.TSConfig`. The control head is
backbone-agnostic and receives per-flight physical bounds, while the caller owns the
differentiable dynamics rollout that turns its controls into supervised states.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TSConfig

CONTROL_NAMES = ("thrust_N", "bank_rad", "load_factor")


@dataclass(frozen=True)
class StatePrediction:
    states: torch.Tensor             # [B, N, C], normalized channel space
    final_time_s: torch.Tensor       # [B], physical seconds from anchor to endpoint


@dataclass(frozen=True)
class ControlPrediction:
    controls: torch.Tensor           # [B, N, 3], physical control units
    segment_durations: torch.Tensor  # [B, N], physical seconds
    final_time_s: torch.Tensor       # [B], segment_durations.sum(dim=-1)


@dataclass(frozen=True)
class ControlBounds:
    """Aircraft-specific bounds in ``(thrust_N, bank_rad, load_factor)`` order."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        expected = len(CONTROL_NAMES)
        if len(self.lower) != expected or len(self.upper) != expected:
            raise ValueError(f"control bounds must contain exactly {expected} values")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("every control lower bound must be smaller than its upper bound")


class FinalTimeHead(nn.Module):
    """Predict a positive physical duration from the normalized observed history."""

    def __init__(self, config: TSConfig):
        super().__init__()
        self.scale_s = config.final_time_scale_s
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.seq_len * config.enc_in, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def raw(self, history: torch.Tensor) -> torch.Tensor:
        """Return the shared unconstrained global duration logit."""
        return self.network(history).squeeze(-1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.raw(history)) * self.scale_s


class StateOutputLayer(nn.Module):
    """Attach the state forecast and duration prediction to one structured contract."""

    def __init__(self, state_forecaster: nn.Module, config: TSConfig):
        super().__init__()
        self.state_forecaster = state_forecaster
        self.final_time_head = FinalTimeHead(config)

    def forward(self, history: torch.Tensor) -> StatePrediction:
        return StatePrediction(
            states=self.state_forecaster(history),
            final_time_s=self.final_time_head(history),
        )


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
    ):
        super().__init__()
        self.n_segments = n_segments
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
        fractions = torch.softmax(self.duration_projection(features), dim=-1)
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
