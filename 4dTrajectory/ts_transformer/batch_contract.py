"""The shape of a training batch, and how a model is called with one.

`dataset` produces these tuples, `train` consumes them, and the train-only oracle consumes
them too. Keeping the two helpers here rather than in `train` is what lets `control.oracle`
read a batch without importing the training loop it runs inside — an import that would make
the package unusable anywhere else and put a cycle in the layering.
"""
from __future__ import annotations

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


def model_forward(
    model: nn.Module,
    history: torch.Tensor,
    dynamics: dict[str, torch.Tensor] | None,
):
    return model(history) if dynamics is None else model(history, dynamics)
