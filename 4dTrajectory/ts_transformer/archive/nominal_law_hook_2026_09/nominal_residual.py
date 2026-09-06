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
Thrust is not a residual but a coordination (``speed_hold_thrust``): steering the path
angle changes the gravity component along the path, and without the matching thrust the
first campaign's arm arrived 30 m/s slow and 600 m short of the threshold. The network's
intent is read from its own unhooked rollout, which the engine integrates alongside for
this hook (``needs_reference``): the thrust pulls the speed toward the speed that rollout
has now, so the schedule's own deceleration is kept and only the hook's energy cost is
paid back — the way the barrier keeps its lift factor. Soft saturation (tanh) in training
keeps the gradient of the residual path alive; hard saturation is for inference-only arms.
Residual policy learning (Silver et al. 2018; Johannink et al. 2019) and the learned
planner + tracking controller split of self-driving stacks, applied per segment.
"""

from __future__ import annotations

import torch

from config import TSConfig
from control.constraints.gates import RunwayAxesView, on_final_weight, runway_axes_view
from control.dynamics.hooks import RolloutStateView
from control.envelope import (
    MAX_BANK_RAD, MAX_LOAD_FACTOR, MAX_THRUST_FRACTION, MIN_LOAD_FACTOR, MIN_THRUST_FRACTION,
)
from control.guidance_laws import glidepath_load_factor, l1_bank, speed_hold_thrust
from final_approach_geometry import glidepath_height

_DIAGNOSTIC_KEYS = (
    "hook_steps", "hook_gated_steps", "hook_bank_residual_saturated_steps",
    "hook_load_residual_saturated_steps", "hook_bank_change_rad", "hook_thrust_change",
)


def bounded_residual(delta: torch.Tensor, bound: float, *, hard: bool) -> torch.Tensor:
    if hard:
        return delta.clamp(min=-bound, max=bound)
    return bound * torch.tanh(delta / bound)


class NominalResidual:
    needs_reference = True

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        self.glidepath_tan = dynamics["glidepath_tan"]
        self.max_thrust_n = dynamics["max_thrust_n"]
        self.l1_distance = config.control_nominal_l1_distance_m
        self.vertical_lookahead = config.control_nominal_vertical_lookahead_m
        self.vertical_gain = config.control_nominal_vertical_gain
        self.speed_gain = config.control_nominal_speed_gain
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
        glidepath_tan = self.glidepath_tan.to(view.d.dtype)
        height_error = view.height - glidepath_height(view.d.unsqueeze(1), glidepath_tan)[:, 0]
        load = glidepath_load_factor(
            height_error, view.path_angle, view.speed, bank,
            glidepath_tan=glidepath_tan, lookahead_m=self.vertical_lookahead,
            gain_per_s=view.hold_rate.clamp(max=self.vertical_gain),
            load_limits=(MIN_LOAD_FACTOR, MAX_LOAD_FACTOR),
        )
        return bank, load

    def coordinated_thrust(self, view: RunwayAxesView, thrust: torch.Tensor, reference: RolloutStateView) -> torch.Tensor:
        """The command's thrust plus what pulls the speed back to the unhooked rollout's."""
        dtype = view.d.dtype
        reference_speed = runway_axes_view(reference, self.runway_heading).speed
        return speed_hold_thrust(
            thrust.to(dtype), view.speed, reference_speed,
            gain_per_s=view.hold_rate.clamp(max=self.speed_gain),
            mass_kg=view.mass, max_thrust_n=self.max_thrust_n.to(dtype),
            thrust_limits=(MIN_THRUST_FRACTION, MAX_THRUST_FRACTION),
        )

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        if state.reference is None:
            raise ValueError("the nominal-law hook reads the unhooked rollout (needs_reference); the backend passed none")
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        bank_nominal, load_nominal = self._nominal(view)
        bank_nominal, load_nominal = bank_nominal.to(dtype), load_nominal.to(dtype)
        thrust, bank, load = command[:, 0], command[:, 1], command[:, 2]
        bank_delta, load_delta = bank - bank_nominal, load - load_nominal
        bank_tracked = bank_nominal + bounded_residual(bank_delta, self.bank_residual_max, hard=self.hard)
        load_tracked = load_nominal + bounded_residual(load_delta, self.load_residual_max, hard=self.hard)
        thrust_tracked = self.coordinated_thrust(view, thrust, state.reference).to(dtype)
        weight = on_final_weight(view, hard=self.hard).to(dtype)
        bank_out = (bank + weight * (bank_tracked - bank)).clamp(min=-MAX_BANK_RAD, max=MAX_BANK_RAD)
        load_out = (load + weight * (load_tracked - load)).clamp(min=MIN_LOAD_FACTOR, max=MAX_LOAD_FACTOR)
        thrust_out = (thrust + weight * (thrust_tracked - thrust)).clamp(min=MIN_THRUST_FRACTION, max=MAX_THRUST_FRACTION)
        gated = (weight > 0.5).detach()
        counts = torch.stack((
            bank.new_full((), float(bank.numel()), dtype=torch.float64),
            gated.sum().to(torch.float64),
            (gated & (bank_delta.abs().detach() > self.bank_residual_max)).sum().to(torch.float64),
            (gated & (load_delta.abs().detach() > self.load_residual_max)).sum().to(torch.float64),
            (bank_out - bank).abs().detach().sum().to(torch.float64),
            (thrust_out - thrust).abs().detach().sum().to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        return torch.stack((thrust_out, bank_out, load_out), dim=-1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Step counts (and the summed bank change) over every call; read on the host once."""
        counts = torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64) if self._counts is None else self._counts.cpu()
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))
