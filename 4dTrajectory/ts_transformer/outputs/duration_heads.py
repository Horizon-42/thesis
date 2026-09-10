"""The duration heads every path that predicts a clock shares: the point head, the
quantile head (B1) and the pinball loss the quantile head is trained under."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_transformer.config import DURATION_QUANTILES, TSConfig


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


class QuantileFinalTimeHead(nn.Module):
    """The same duration, as the five ``DURATION_QUANTILES`` (B1, design §三 3.1).

    Monotone BY CONSTRUCTION rather than by penalty: the network emits one raw number per
    level and the head reads them as positive increments —
    ``q_0 = softplus(r_0)·scale``, ``q_{k+1} = q_k + softplus(r_{k+1})·scale`` — so no
    input can produce a crossing pair and the loss never has to price one. It is the same
    body as :class:`FinalTimeHead` with a wider last layer, and its median column is what
    the rollout flies, so a quantile run and a point run differ in exactly this head.
    """

    def __init__(self, config: TSConfig):
        super().__init__()
        self.scale_s = config.final_time_scale_s
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.seq_len * config.enc_in, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, len(DURATION_QUANTILES)),
        )

    def raw(self, history: torch.Tensor) -> torch.Tensor:
        """The unconstrained increment logits, ``[B, Q]``."""
        return self.network(history)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """``[B, Q]`` positive, strictly increasing durations in seconds."""
        return torch.cumsum(F.softplus(self.raw(history)), dim=-1) * self.scale_s


def pinball_duration_loss(
    quantiles_s: torch.Tensor, target_final_time_s: torch.Tensor, scale_s: float
) -> torch.Tensor:
    """Per-flight sum of the five pinball losses, in the point head's own units.

    The point head's ``final_time`` component is the SQUARED residual in units of
    ``final_time_scale_s``; this is the same residual, in the same units, priced by the
    check function ``max(τ·r, (τ−1)·r)`` and summed over ``DURATION_QUANTILES`` — the one
    place the pinball loss is written, so the training objective, a unit test and any
    later readout cannot disagree about what "the quantile loss" means. Returns ``[B]``.
    """
    if quantiles_s.shape[-1] != len(DURATION_QUANTILES):
        raise ValueError(
            f"the quantile head must emit {len(DURATION_QUANTILES)} levels, got "
            f"{quantiles_s.shape[-1]}"
        )
    residual = (target_final_time_s.unsqueeze(-1) - quantiles_s) / scale_s
    tau = torch.as_tensor(
        DURATION_QUANTILES, dtype=residual.dtype, device=residual.device
    )
    return torch.maximum(tau * residual, (tau - 1.0) * residual).sum(dim=-1)
