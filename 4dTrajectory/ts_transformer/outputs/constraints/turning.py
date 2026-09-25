"""The bank that turns the aircraft through a heading change over a hold, through the lagging bank actuator, and the
load factor that keeps its lift — the one inversion the barrier filter (`barrier_filter`) and the trombone
(`trombone`) both command with, so its three physical assumptions live once: the lag credit (``lag_effective_s``),
the floor on the flown vertical lift factor (``LIFT_FLOOR``) and the coordination law (``coordinated_load``).

Over a hold ``dt`` a first-order bank actuator of time constant ``τ`` flies the bank it is in NOW for
``τ_eff = τ(1 − e^{−dt/τ})`` seconds' worth, the command for the rest, so the command that turns the heading by ``Δψ``
solves ``tan μ_c = [ (V_h / (g·L)) · Δψ − tan μ_0 · τ_eff ] / (dt − τ_eff)`` with ``L = n cos μ`` the vertical lift
factor being flown.
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from ts_transformer.outputs.constraints.vertical import VerticalChannel
from ts_transformer.outputs.dynamics.hooks import RolloutStateView

#: The flown vertical lift factor ``n cos μ`` is floored here, well below any flown value: a guard on the division, not
#: a model of flight.
LIFT_FLOOR = 0.5


def lag_effective_s(bank_lag_s: float, hold: torch.Tensor) -> torch.Tensor:
    """``τ_eff``: the seconds' worth of the hold flown at the bank the actuator is in now."""
    return bank_lag_s * (1.0 - torch.exp(-hold / bank_lag_s))


def committed_turn(actuators: torch.Tensor, tau_eff: torch.Tensor) -> torch.Tensor:
    """``tan μ_0 · τ_eff``: the part of the turn the hold has already committed to."""
    return torch.tan(actuators[:, 1]) * tau_eff


def turn_scale(ground_speed: torch.Tensor, vertical: VerticalChannel, state: RolloutStateView,
               actuators: torch.Tensor) -> torch.Tensor:
    """``V_h / (g·L)``, with ``L`` the vertical lift factor being flown (the contract's law resolves the load from the
    actuator state, so the scale is a function of the state alone)."""
    lift = (vertical.flown_load(state, actuators) * torch.cos(actuators[:, 1])).clamp(min=LIFT_FLOOR)
    return ground_speed / (GRAVITY_MPS2 * lift)


def bank_for_heading_change(change: torch.Tensor, scale: torch.Tensor, committed: torch.Tensor, hold: torch.Tensor,
                            tau_eff: torch.Tensor) -> torch.Tensor:
    """The bank command ``μ_c`` that turns the heading by ``change`` over the hold (unclamped)."""
    return torch.atan((scale * change - committed) / (hold - tau_eff))


def coordinated_load(vertical: VerticalChannel, load: torch.Tensor, bank: torch.Tensor,
                     filtered: torch.Tensor) -> torch.Tensor:
    """The third command column once a hook moved the bank from ``bank`` to ``filtered``: the lift the network paired
    with its vertical command kept (`VerticalChannel.keep_lift`) — and the column untouched on a row whose bank did not
    move, because ``(n cos μ) / cos μ`` is NOT the identity in IEEE (an ULP apart for ~40 % of operand pairs), and a
    hook that is inert on a row must not perturb it."""
    return torch.where(filtered == bank, load, vertical.keep_lift(load, bank, filtered))
