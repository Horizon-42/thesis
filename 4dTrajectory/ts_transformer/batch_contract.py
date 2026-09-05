"""The shape of a training batch, and how a model is called with one.

`dataset` produces these tuples, `train` consumes them, and the train-only oracle consumes
them too. Keeping the two helpers here rather than in `train` is what lets `control.oracle`
read a batch without importing the training loop it runs inside — an import that would make
the package unusable anywhere else and put a cycle in the layering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


def unpack_batch(batch: tuple) -> tuple:
    """Widen a ragged trajectory batch to the full 7-field contract.

    The optional tail is genuinely optional: `dynamics` is present only for control output
    and `dense_supervision` only under the fixed-dt loss grid, so a fixed-width unpack at
    the call site breaks the moment a recipe turns one of them off.
    """
    if len(batch) == 5:
        return (*batch, None, None)
    if len(batch) == 6:
        return (*batch, None)
    if len(batch) == 7:
        return batch
    raise ValueError(f"unexpected trajectory batch with {len(batch)} fields")


def anchor_state(history: torch.Tensor, channel_count: int) -> torch.Tensor:
    """The normalized observed state at the anchor: the history's last row, state part.

    A history is ``[B, L, C + K]`` — the state contract followed by ``K`` input-only
    conditioning columns (``target_conditioning`` and ``intent_conditioning``; ``K = 0``
    when both are off). Everything that
    scores or rolls out from the anchor wants the ``C`` state columns, never the
    conditioning, so the slice lives here rather than as ``x[:, -1]`` at every call site.
    """
    return history[:, -1, :channel_count]


def model_forward(
    model: nn.Module,
    history: torch.Tensor,
    dynamics: dict[str, torch.Tensor] | None,
):
    return model(history) if dynamics is None else model(history, dynamics)


@dataclass(frozen=True)
class LossComponents:
    """Weighted scalar contributions whose sum is the optimization objective."""

    state: torch.Tensor
    final_time: torch.Tensor
    kinematic: torch.Tensor
    terminal: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    # Batch-level COUNTS that are not part of the objective (the procedure penalty's gated
    # rows and violations, which the dual update turns into a rate over the epoch).
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def total(self) -> torch.Tensor:
        return (
            self.state + self.final_time + self.kinematic + self.terminal
            + sum(self.extras.values(), self.state.new_zeros(()))
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "state": self.state,
            "final_time": self.final_time,
            "kinematic": self.kinematic,
            "terminal": self.terminal,
            **self.extras,
        }
