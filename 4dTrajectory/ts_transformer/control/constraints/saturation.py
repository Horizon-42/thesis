"""The two soft saturations every constraint module shares.

``hook_saturation`` is one axis across all of them: ``soft`` is the C^1 training form
(gradients survive a bound that binds), ``hard`` the deployed clamp. The soft form is a
scaled softplus, so the pair below is what "soft" MEANS in this package — one definition,
because a second one that merely agreed today would let two hooks saturate differently
under the same word.
"""

from __future__ import annotations

import torch


def soft_max(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    """Smooth ``max(x, bound)``: equals ``bound`` well below it, ``x`` well above."""
    return bound + softness * torch.nn.functional.softplus((x - bound) / softness)


def soft_min(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    return bound - softness * torch.nn.functional.softplus((bound - x) / softness)
