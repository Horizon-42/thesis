"""The two soft saturations every constraint module shares.

``hook_saturation`` is one axis across all of them: ``soft`` is the C^1 training form
(gradients survive a bound that binds), ``hard`` the deployed clamp. The soft form is a
scaled softplus, so the pair below is what "soft" MEANS in this package — one definition,
because a second one that merely agreed today would let two hooks saturate differently
under the same word.
"""

from __future__ import annotations

import math

import torch

#: Width of the soft max/min around a BANK bound, in radians. Two modules now saturate a
#: bank command — the barrier's admissible interval and the trombone's turn cap — and a
#: bank is a bank: one number, so ``soft`` cannot mean 2 deg in one module and 5 deg in the
#: next. (The speed floor's bound is a thrust FRACTION and carries its own width, in its
#: own units, in its own module.)
SATURATION_SOFTNESS_RAD = math.radians(2.0)
#: A commanded bank moved by more than this counts, in the diagnostics, as the hook having
#: acted on that step. Shared for the same reason: ``clamped``/``bound`` shares reported by
#: two modules must count the same thing before a readout may put them side by side.
ACTIVE_BANK_CHANGE_RAD = math.radians(0.5)


def soft_max(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    """Smooth ``max(x, bound)``: equals ``bound`` well below it, ``x`` well above."""
    return bound + softness * torch.nn.functional.softplus((x - bound) / softness)


def soft_min(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    return bound - softness * torch.nn.functional.softplus((bound - x) / softness)
