"""Nominal tracking law + bounded residual: the planner/tracker split inside the rollout.

Where the gate says the aircraft is on the final, the command is read as a bounded
correction to a fixed tracking law (L1 lateral guidance to the centreline, a
flight-path-angle law to the glidepath — ``control/guidance_laws.py``):

    u' = w · (u_nominal(x) + sat(u − u_nominal(x); ±r_max)) + (1 − w) · u

so a command that agrees with the nominal passes unchanged, one that disagrees by more
than the residual bound is pulled to ``nominal ± r_max``, and the path converges to the
centreline and glidepath at the nominal law's rate from ANYWHERE the gate is on — inside
or outside the corridor. The vertical gain is capped at ``1/Δt`` for a segment held Δt
(the L1 law's own time scale, ``L1 / V``, is longer than any hold and needs no cap).
Thrust passes through. Soft saturation (tanh) in training keeps
the gradient of the residual path alive; hard saturation is for inference-only arms.
Residual policy learning (Silver et al. 2018; Johannink et al. 2019) and the learned
planner + tracking controller split of self-driving stacks, applied per segment.
"""

from __future__ import annotations

import torch

from config import TSConfig
from control.constraints.gates import RunwayAxesView, on_final_weight, runway_axes_view
from control.dynamics.hooks import RolloutStateView
from control.envelope import MAX_BANK_RAD, MAX_LOAD_FACTOR, MIN_LOAD_FACTOR
from control.guidance_laws import glidepath_load_factor, l1_bank
from final_approach_geometry import glidepath_height

_DIAGNOSTIC_KEYS = (
    "hook_steps", "hook_gated_steps", "hook_bank_residual_saturated_steps",
    "hook_load_residual_saturated_steps", "hook_bank_change_rad",
)


def bounded_residual(delta: torch.Tensor, bound: float, *, hard: bool) -> torch.Tensor:
    if hard:
        return delta.clamp(min=-bound, max=bound)
    return bound * torch.tanh(delta / bound)


class NominalResidual:
    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        self.glidepath_tan = dynamics["glidepath_tan"]
        self.l1_distance = config.control_nominal_l1_distance_m
        self.vertical_lookahead = config.control_nominal_vertical_lookahead_m
        self.vertical_gain = config.control_nominal_vertical_gain
        self.bank_residual_max = config.control_nominal_residual_bank_max_rad
        self.load_residual_max = config.control_nominal_residual_load_max
        self.hard = hard
        self._counts: torch.Tensor | None = None   # one entry per _DIAGNOSTIC_KEYS, on the device

    def nominal(self, state: RolloutStateView) -> tuple[torch.Tensor, torch.Tensor]:
        """The nominal ``(bank, load factor)`` at this state."""
        return self._nominal(runway_axes_view(state, self.runway_heading))

    def _nominal(self, view: RunwayAxesView) -> tuple[torch.Tensor, torch.Tensor]:
        bank = l1_bank(
            view.xt, view.heading_error, view.ground_speed,
            l1_distance_m=self.l1_distance, bank_limit_rad=MAX_BANK_RAD,
        )
        height_error = view.height - glidepath_height(
            view.d.unsqueeze(1), self.glidepath_tan.to(view.d.dtype)
        )[:, 0]
        load = glidepath_load_factor(
            height_error, view.path_angle, view.speed, bank,
            glidepath_tan=self.glidepath_tan.to(view.d.dtype),
            lookahead_m=self.vertical_lookahead,
            gain_per_s=view.hold_rate.clamp(max=self.vertical_gain),
            load_limits=(MIN_LOAD_FACTOR, MAX_LOAD_FACTOR),
        )
        return bank, load

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        bank_nominal, load_nominal = self._nominal(view)
        bank_nominal, load_nominal = bank_nominal.to(dtype), load_nominal.to(dtype)
        bank, load = command[:, 1], command[:, 2]
        bank_delta, load_delta = bank - bank_nominal, load - load_nominal
        bank_tracked = bank_nominal + bounded_residual(bank_delta, self.bank_residual_max, hard=self.hard)
        load_tracked = load_nominal + bounded_residual(load_delta, self.load_residual_max, hard=self.hard)
        weight = on_final_weight(view, hard=self.hard).to(dtype)
        bank_out = (bank + weight * (bank_tracked - bank)).clamp(min=-MAX_BANK_RAD, max=MAX_BANK_RAD)
        load_out = (load + weight * (load_tracked - load)).clamp(min=MIN_LOAD_FACTOR, max=MAX_LOAD_FACTOR)
        gated = (weight > 0.5).detach()
        counts = torch.stack((
            bank.new_full((), float(bank.numel()), dtype=torch.float64),
            gated.sum().to(torch.float64),
            (gated & (bank_delta.abs().detach() > self.bank_residual_max)).sum().to(torch.float64),
            (gated & (load_delta.abs().detach() > self.load_residual_max)).sum().to(torch.float64),
            (bank_out - bank).abs().detach().sum().to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        return torch.stack((command[:, 0], bank_out, load_out), dim=-1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Step counts (and the summed bank change) over every call; read on the host once."""
        counts = torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64) if self._counts is None else self._counts.cpu()
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))
