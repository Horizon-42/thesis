"""Typed state/control prediction contracts and replaceable output layers.

The selected output strategy is explicit in :class:`config.TSConfig`. The control head is
backbone-agnostic and receives per-flight physical bounds, while the caller owns the
differentiable dynamics rollout that turns its controls into supervised states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from batch_contract import anchor_state
from channels import IDX, POSITION_IDX
from config import (
    STATE_POSITION_ANCHOR_RELATIVE,
    STATE_POSITION_CORRIDOR_BOUNDED,
    TSConfig,
)
from final_approach_geometry import (
    FINAL_APPROACH_KEYS,
    alignment_cosine,
    bound_to_final,
    chart_from_axes,
    membership,
    position_direction,
    runway_axes,
)

if TYPE_CHECKING:  # the data-plane value type; importing it at runtime would be a cycle
    from dataset import Normalizer

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
    """Attach the state forecast and duration prediction to one structured contract.

    Under ``state_position_reference="anchor-relative"`` the forecaster's position
    channels are read as displacements from the anchor (the history's last observed row)
    and the anchor's normalized position is added back here, so ``states`` keeps the same
    contract downstream — absolute normalized chart coordinates — while the network's
    zero output means "the aircraft stays where it is" instead of "the chart origin".

    Under ``"corridor-bounded"`` the absolute output is kept and, on the rows the output
    itself places on the runway's final (``config.corridor_gate``), its cross-track is
    saturated inside the LPV corridor and its height inside the glidepath window
    (``final_approach_geometry``): the constraint holds by construction, continuously, with
    no weight to calibrate. The layer then needs physical units, so it carries the
    normalizer statistics as buffers (bound at construction by ``models.build_model``,
    restored from the state dict on load) and the per-flight ``context`` in ``forward``.
    """

    def __init__(
        self,
        state_forecaster: nn.Module,
        config: TSConfig,
        normalizer: Normalizer | None = None,
    ):
        super().__init__()
        self.state_forecaster = state_forecaster
        self.final_time_head = FinalTimeHead(config)
        self.anchor_relative = (
            config.state_position_reference == STATE_POSITION_ANCHOR_RELATIVE
        )
        self.corridor_bounded = (
            config.state_position_reference == STATE_POSITION_CORRIDOR_BOUNDED
        )
        self.corridor_gate = config.corridor_gate
        self.channel_count = len(config.channels)
        offset_mask = torch.zeros(self.channel_count)
        offset_mask[list(POSITION_IDX)] = 1.0
        # A pure function of the channel contract, so NOT persisted: checkpoints written
        # before it existed (the 2026-09-03 arm-A checkpoints) must keep loading.
        self.register_buffer("offset_mask", offset_mask, persistent=False)
        if self.corridor_bounded:
            # Identity statistics until bound: an unbound layer (the batch-size probe)
            # runs finite; a trained checkpoint restores the real ones with its weights.
            self.register_buffer("channel_mean", torch.zeros(self.channel_count))
            self.register_buffer("channel_std", torch.ones(self.channel_count))
            if normalizer is not None:
                self.bind_normalizer(normalizer)

    def bind_normalizer(self, normalizer: Normalizer) -> None:
        """Give the bounded output the chart's physical scale (metres, metres/second)."""
        if not self.corridor_bounded:
            raise ValueError("only the corridor-bounded state output decodes to physical units")
        self.channel_mean.copy_(torch.as_tensor(normalizer.mean, dtype=torch.float32))
        self.channel_std.copy_(torch.as_tensor(normalizer.std, dtype=torch.float32))

    def forward(
        self, history: torch.Tensor, context: dict[str, torch.Tensor] | None = None
    ) -> StatePrediction:
        states = self.state_forecaster(history)
        if self.anchor_relative:
            anchor = anchor_state(history, self.channel_count)
            states = states + (anchor * self.offset_mask).unsqueeze(1)
        if self.corridor_bounded:
            if context is None or any(key not in context for key in FINAL_APPROACH_KEYS):
                raise ValueError(
                    "the corridor-bounded state output needs the per-flight final-approach "
                    f"context {FINAL_APPROACH_KEYS} in the batch's context slot"
                )
            states = self._bound_to_final(states, history, context)
        return StatePrediction(
            states=states,
            final_time_s=self.final_time_head(history),
        )

    def _bound_to_final(
        self, states: torch.Tensor, history: torch.Tensor, context: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mean = self.channel_mean.to(states.dtype)
        std = self.channel_std.to(states.dtype)
        physical = states * std + mean
        anchor = anchor_state(history, self.channel_count) * std + mean
        psi = context["runway_heading_rad"].to(states.dtype)
        d, xt = runway_axes(physical[..., IDX["e"]], physical[..., IDX["n"]], psi)
        # Direction from the predicted POSITIONS (the velocity channels are unsupervised).
        step_e, step_n = position_direction(
            physical[..., IDX["e"]], physical[..., IDX["n"]],
            anchor[:, IDX["e"]], anchor[:, IDX["n"]],
        )
        cos_align = alignment_cosine(step_e, step_n, psi)
        weight = membership(
            self.corridor_gate,
            d=d,
            xt=xt,
            cos_align=cos_align,
            d_faf=context["final_approach_fix_m"].to(states.dtype),
            hard=False,
        )
        xt_bounded, u_bounded = bound_to_final(
            d=d,
            xt=xt,
            u=physical[..., IDX["u"]],
            weight=weight,
            tan_gpa=context["glidepath_tan"].to(states.dtype),
            hard=False,
        )
        e_bounded, n_bounded = chart_from_axes(d, xt_bounded, psi)
        columns = list(physical.unbind(dim=-1))
        columns[IDX["e"]], columns[IDX["n"]], columns[IDX["u"]] = e_bounded, n_bounded, u_bounded
        return (torch.stack(columns, dim=-1) - mean) / std


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
