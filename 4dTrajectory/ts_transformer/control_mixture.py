"""Typed multi-expert control output contract and its isolated decoder head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from prediction_outputs import ControlOutputHead, ControlPrediction


@dataclass(frozen=True)
class ControlMixturePrediction:
    """K complete control futures plus deployable history-only selection logits."""

    controls: torch.Tensor             # [B,K,N,3]
    segment_durations: torch.Tensor    # [B,K,N]
    final_time_s: torch.Tensor         # [B,K]
    selection_logits: torch.Tensor     # [B,K]

    @property
    def expert_count(self) -> int:
        return self.controls.shape[1]

    def candidate(self, index: int) -> ControlPrediction:
        if not 0 <= index < self.expert_count:
            raise IndexError(f"expert {index} outside [0,{self.expert_count})")
        return ControlPrediction(
            controls=self.controls[:, index],
            segment_durations=self.segment_durations[:, index],
            final_time_s=self.final_time_s[:, index],
        )

    def selected(self, indices: torch.Tensor | None = None) -> ControlPrediction:
        """Gather one deployable expert per flight; defaults to selector argmax."""
        if indices is None:
            indices = self.selection_logits.argmax(dim=1)
        if indices.shape != (self.controls.shape[0],):
            raise ValueError("expert indices must have shape [B]")
        indices = indices.to(device=self.controls.device, dtype=torch.long)
        rows = torch.arange(self.controls.shape[0], device=self.controls.device)
        return ControlPrediction(
            controls=self.controls[rows, indices],
            segment_durations=self.segment_durations[rows, indices],
            final_time_s=self.final_time_s[rows, indices],
        )


class ControlMixtureHead(nn.Module):
    """Independent control/duration experts and a separate deployable selector."""

    def __init__(self, input_dim: int, n_segments: int, expert_count: int):
        super().__init__()
        self.experts = nn.ModuleList(
            ControlOutputHead(input_dim, n_segments) for _ in range(expert_count)
        )
        self.selector = nn.Linear(input_dim, expert_count)

    def forward(
        self,
        features: torch.Tensor,
        final_time_s: torch.Tensor,
        *,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> ControlMixturePrediction:
        if final_time_s.shape != (features.shape[0], len(self.experts)):
            raise ValueError("expert final times must have shape [B,K]")
        candidates = [
            expert(
                features,
                final_time_s[:, index],
                lower=lower,
                upper=upper,
            )
            for index, expert in enumerate(self.experts)
        ]
        return ControlMixturePrediction(
            controls=torch.stack([item.controls for item in candidates], dim=1),
            segment_durations=torch.stack(
                [item.segment_durations for item in candidates], dim=1
            ),
            final_time_s=final_time_s,
            selection_logits=self.selector(features),
        )
