"""The per-step barrier filter: a safety layer on the bank command.

The corridor is two barriers, ``h_R = k·hw(d) − xt ≥ 0`` (right edge) and
``h_L = k·hw(d) + xt ≥ 0`` (left edge). With ``ẋt = −V_h sin ψ_err`` and the half-width
closing as the aircraft approaches (``hw' = cw / d_GARP`` before the threshold, zero at and
past it where the half-width is the flat LTP width — ``corridor_halfwidth_slope``), the
barrier conditions ``ḣ + α h ≥ 0`` bound the sine of the heading error from below (right
edge) and above (left edge):

    sin ψ_err ≥  k c cos ψ_err − α h_R / V_h
    sin ψ_err ≤ −k c cos ψ_err + α h_L / V_h        c = hw'(d)

Outside the corridor a margin is negative and the bound demands motion back inside at
rate ≥ α·|h| — the filter does act on a state already outside, as long as the gate says
the aircraft is on the final. The heading interval is a barrier pair in its own right:
with the heading gain β the admissible turn rate is ``−β(ψ_err − lo) ≤ ψ̇ ≤ β(hi − ψ_err)``
(a turn toward an edge at most as fast as the distance to it, a turn back demanded
beyond it), and the level-turn relation ``tan μ = V_h ψ̇ / g`` makes that a bank interval
``[μ_min, μ_max]``; the command's bank is saturated into it — softly (a scaled softplus,
C¹, keeps gradients) in training, hard at inference — blended by the gate weight, and
clamped to the envelope. ``hard`` selects the hard saturation AND the hard gate: a
deployed filter has no partially-gated rows, and the soft pair is the C¹ training form
of the same rule.

The command is HELD for the segment, so both rates are discrete-time barriers: a rate
above ``1/Δt`` would carry the state past the edge within the hold it is meant to protect
(``h_{k+1} ≥ (1 − αΔt) h_k`` needs ``αΔt ≤ 1``), so each gain is used as
``min(gain, 1/Δt)`` with Δt the segment's hold — the configured value binds at short
holds, the hold at long ones.

Closed-form single-constraint action projection (Dalal et al. 2018); α is the class-K
rate of a control barrier function (Ames et al. 2019), in its discrete-time form
(Agrawal & Sreenath 2017). Lateral only: the glidepath window is left to the penalty or
the nominal-law hook.
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from config import TSConfig
from control.constraints.gates import on_final_weight, runway_axes_view
from control.dynamics.hooks import RolloutStateView
from control.envelope import MAX_BANK_RAD
from final_approach_geometry import K_MARGIN, corridor_halfwidth_slope

SATURATION_SOFTNESS_RAD = math.radians(2.0)   # width of the soft max/min around a bound
_ACTIVE_BANK_CHANGE_RAD = math.radians(0.5)   # a step counts as "clamped" past this
_DIAGNOSTIC_KEYS = ("hook_steps", "hook_gated_steps", "hook_clamped_steps", "hook_bank_change_rad")


def soft_max(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    """Smooth ``max(x, bound)``: equals ``bound`` well below it, ``x`` well above."""
    return bound + softness * torch.nn.functional.softplus((x - bound) / softness)


def soft_min(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    return bound - softness * torch.nn.functional.softplus((bound - x) / softness)


class BarrierFilter:
    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        self.alpha = config.control_barrier_alpha
        self.heading_gain = config.control_barrier_heading_gain
        self.hard = hard
        self._counts: torch.Tensor | None = None   # one entry per _DIAGNOSTIC_KEYS, on the device

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        margin_right = K_MARGIN * view.halfwidth - view.xt
        margin_left = K_MARGIN * view.halfwidth + view.xt
        closing = K_MARGIN * corridor_halfwidth_slope(view.d) * view.cos_align
        alpha = view.hold_rate.clamp(max=self.alpha)
        heading_gain = view.hold_rate.clamp(max=self.heading_gain)
        sin_lower = (closing - alpha * margin_right / view.ground_speed).clamp(-1.0, 1.0)
        sin_upper = (-closing + alpha * margin_left / view.ground_speed).clamp(-1.0, 1.0)
        lower = torch.asin(sin_lower)
        upper = torch.asin(sin_upper)
        # An empty interval (both edges violated at once, or a very narrow margin) collapses
        # to its midpoint: the heading that splits the difference.
        midpoint = 0.5 * (lower + upper)
        lower = torch.minimum(lower, midpoint)
        upper = torch.maximum(upper, midpoint)
        # Second layer: the heading interval is itself a barrier pair, so the admissible
        # turn rate is ψ̇ ≥ −β·(ψ_err − lower) and ψ̇ ≤ β·(upper − ψ_err): inside the interval
        # a turn toward an edge is allowed at a rate proportional to the distance left,
        # at the edge it is zero, beyond it a turn back is demanded — one continuous rule.
        turn_rate_min = -heading_gain * (view.heading_error - lower)
        turn_rate_max = heading_gain * (upper - view.heading_error)
        # Both bounds into the envelope (min ≤ max survives, so the interval cannot invert).
        bank_min = torch.atan(view.ground_speed * turn_rate_min / GRAVITY_MPS2).to(dtype).clamp(-MAX_BANK_RAD, MAX_BANK_RAD)
        bank_max = torch.atan(view.ground_speed * turn_rate_max / GRAVITY_MPS2).to(dtype).clamp(-MAX_BANK_RAD, MAX_BANK_RAD)
        bank = command[:, 1]
        if self.hard:
            bounded = torch.minimum(torch.maximum(bank, bank_min), bank_max)
        else:
            bounded = soft_min(soft_max(bank, bank_min, SATURATION_SOFTNESS_RAD), bank_max, SATURATION_SOFTNESS_RAD)
        weight = on_final_weight(view, hard=self.hard).to(dtype)
        filtered = (bank + weight * (bounded - bank)).clamp(min=-MAX_BANK_RAD, max=MAX_BANK_RAD)
        change = (filtered - bank).abs().detach()
        counts = torch.stack((
            bank.new_full((), float(bank.numel()), dtype=torch.float64),
            (weight > 0.5).sum().to(torch.float64),
            (change > _ACTIVE_BANK_CHANGE_RAD).sum().to(torch.float64),
            change.sum().to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        return torch.stack((command[:, 0], filtered, command[:, 2]), dim=-1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Step counts (and the summed bank change) over every call; read on the host once."""
        counts = torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64) if self._counts is None else self._counts.cpu()
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))
