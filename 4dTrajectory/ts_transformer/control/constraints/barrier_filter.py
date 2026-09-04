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
the aircraft is on the final. The heading interval ``[lo, hi]`` is a barrier pair in its
own right: over the hold Δt the heading error must move at least the fraction ``β·Δt``
of its excess back toward the interval, and may move at most that fraction of its
distance toward an edge — ``Δψ ∈ [−βΔt(ψ_err − lo), βΔt(hi − ψ_err)]`` (a continuous
rule: inside the interval a turn toward an edge is allowed in proportion to the room
left, at the edge it is zero, beyond it a turn back is demanded).

The command is HELD for Δt and the bank is a first-order actuator with time constant τ,
so the heading change a command produces over the hold is the turn rate
``ψ̇ = g (n cos μ) tan μ / V_h`` (the point-mass ``g n sin μ / (V cos γ)``) integrated over
a bank that decays from the bank being flown NOW, ``μ_0`` (the actuator state the backend
exposes), to the command ``μ_c``:

    Δψ ≈ (g L / V_h) · [tan μ_c · (Δt − τ_eff) + tan μ_0 · τ_eff],   τ_eff = τ (1 − e^{−Δt/τ})

with ``L = n cos μ`` the vertical lift factor being flown (read from the actuator state,
so the bounds remain a function of the state alone; the load coordination below keeps
the commanded factor, so the two agree once the actuators settle — a head flying
``L = 1.18`` turns 18 % faster than a rule assuming ``L = 1`` would model).
Inverting that for ``tan μ_c`` turns the heading-change interval into the bank interval
``[μ_min, μ_max]``. For the same reason the corridor barriers are evaluated at the
position the aircraft reaches by carrying its current velocity for ``τ_eff`` — where the
command first bites — rather than where it is now. (The heading-change budget is still
taken against the current heading error while ``[lo, hi]`` is the interval at the lead
point; the committed turn moves the error by a few degrees in between — a second-order
simplification that errs on the conservative side when the aircraft is already turning
toward an edge.) The command's bank is saturated into the interval — softly (a scaled
softplus, C¹, keeps gradients) in training, hard at inference — blended by the gate
weight, and clamped to the envelope. Crediting the bank already flown is what stops the
limit cycle a rate-only rule falls into (two holds at +28° then −29° on a 7 s hold with
τ = 2 s, measured on the first campaign): the rule asks only for the part of the turn the
hold has not already committed to. Both rates are discrete-time barriers as well: a gain
above ``1/Δt`` would carry the state past the edge within the hold it protects
(``h_{k+1} ≥ (1 − αΔt) h_k`` needs ``αΔt ≤ 1``), so each is used as ``min(gain, 1/Δt)``.

Degenerate holds: as ``Δt`` shrinks toward ``τ`` the command has almost no authority
inside the hold (``Δt − τ_eff → 0``), the inverted bounds blow up and both land on the
envelope, i.e. the interval collapses to a corner and the learned bank is fully
overridden (measured: 24 % of gated steps at 1 s holds, 7 % at 2 s, none from 5 s).
``hook_saturated_interval_steps`` counts those steps so a readout can tell a hold-length
artefact from a network that leans on the filter (the "lazy" veto reads
``hook_clamped_steps``).

A bank change alters the vertical lift component ``n cos μ`` the network paired with its
load factor, so the load factor is re-coordinated to keep it: ``n' = n cos μ / cos μ'``,
clamped to the envelope (the clamp binds only above ``n ≈ 1.41`` with a full-scale bank
change; on the measured band of the heads it never does — 0 of 80 000 sampled states).
Without it a filtered turn steepens the path — the first campaign's worst flight went
from γ = −1° to −10°, 93 to 200 m/s, and 16 km past the threshold. ``hard`` selects the
hard saturation AND the hard gate: a deployed filter has no partially-gated rows, and the
soft pair is the C¹ training form of the same rule.

Closed-form single-constraint action projection (Dalal et al. 2018); α is the class-K
rate of a control barrier function (Ames et al. 2019), in its discrete-time form
(Agrawal & Sreenath 2017). Lateral: the glidepath window itself is left to the penalty or
the nominal-law hook; the load coordination only preserves what the network asked for.
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from config import TSConfig
from control.constraints.gates import on_final_weight, runway_axes_view
from control.dynamics.hooks import RolloutStateView
from control.envelope import MAX_BANK_RAD, MAX_LOAD_FACTOR, MIN_LOAD_FACTOR
from final_approach_geometry import K_MARGIN, corridor_halfwidth, corridor_halfwidth_slope

SATURATION_SOFTNESS_RAD = math.radians(2.0)   # width of the soft max/min around a bound
_ACTIVE_BANK_CHANGE_RAD = math.radians(0.5)   # a step counts as "clamped" past this
_SATURATED_INTERVAL_RAD = math.radians(0.1)   # a bank interval this narrow is a corner, not a bound
_DIAGNOSTIC_KEYS = (
    "hook_steps", "hook_gated_steps", "hook_clamped_steps", "hook_saturated_interval_steps",
    "hook_bank_change_rad", "hook_load_change",
)


def soft_max(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    """Smooth ``max(x, bound)``: equals ``bound`` well below it, ``x`` well above."""
    return bound + softness * torch.nn.functional.softplus((x - bound) / softness)


def soft_min(x: torch.Tensor, bound: torch.Tensor, softness: float) -> torch.Tensor:
    return bound - softness * torch.nn.functional.softplus((bound - x) / softness)


class BarrierFilter:
    needs_reference = False   # reads the hooked state only

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        self.alpha = config.control_barrier_alpha
        self.heading_gain = config.control_barrier_heading_gain
        self.bank_lag_s = config.control_bank_time_constant_s
        self.hard = hard
        self._counts: torch.Tensor | None = None   # one entry per _DIAGNOSTIC_KEYS, on the device

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        hold = view.hold_s
        # The bank actuator lags: over the hold the aircraft flies the bank it is in NOW for
        # τ_eff seconds' worth and the command for the rest, so the command only has to
        # produce the part of the turn the hold has not already committed to — and the
        # corridor barriers are evaluated where the aircraft will be when the command
        # first bites (the current velocity carried for τ_eff), not where it is now.
        tau_eff = self.bank_lag_s * (1.0 - torch.exp(-hold / self.bank_lag_s))
        d_lead = view.d - view.ground_speed * view.cos_align * tau_eff
        xt_lead = view.xt - view.ground_speed * torch.sin(view.heading_error) * tau_eff
        halfwidth_lead = corridor_halfwidth(d_lead)
        margin_right = K_MARGIN * halfwidth_lead - xt_lead
        margin_left = K_MARGIN * halfwidth_lead + xt_lead
        closing = K_MARGIN * corridor_halfwidth_slope(d_lead) * view.cos_align
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
        # Second layer: the heading interval is itself a barrier pair — over the hold the
        # heading error must move at least the fraction βΔt of its excess back toward the
        # interval and may move at most that fraction of its room toward an edge (one
        # continuous rule: proportional inside, zero at the edge, a turn back beyond it).
        heading_change_min = -heading_gain * hold * (view.heading_error - lower)
        heading_change_max = heading_gain * hold * (upper - view.heading_error)
        actuators = state.actuators.to(view.d.dtype)
        committed = torch.tan(actuators[:, 1]) * tau_eff
        # The vertical lift factor n·cos μ being flown sets the turn rate a bank produces
        # (the coordination below keeps the commanded one, so the two agree once the
        # actuators settle); read from the actuator state so the bounds stay a function of
        # the state alone. Floored well below any flown value.
        lift = (actuators[:, 2] * torch.cos(actuators[:, 1])).clamp(min=0.5)
        scale = view.ground_speed / (GRAVITY_MPS2 * lift)
        bank, load = command[:, 1], command[:, 2]
        tan_min = (scale * heading_change_min - committed) / (hold - tau_eff)
        tan_max = (scale * heading_change_max - committed) / (hold - tau_eff)
        # Both bounds into the envelope (min ≤ max survives, so the interval cannot invert).
        bank_min = torch.atan(tan_min).to(dtype).clamp(-MAX_BANK_RAD, MAX_BANK_RAD)
        bank_max = torch.atan(tan_max).to(dtype).clamp(-MAX_BANK_RAD, MAX_BANK_RAD)
        if self.hard:
            bounded = torch.minimum(torch.maximum(bank, bank_min), bank_max)
        else:
            bounded = soft_min(soft_max(bank, bank_min, SATURATION_SOFTNESS_RAD), bank_max, SATURATION_SOFTNESS_RAD)
        weight = on_final_weight(view, hard=self.hard).to(dtype)
        filtered = (bank + weight * (bounded - bank)).clamp(min=-MAX_BANK_RAD, max=MAX_BANK_RAD)
        # Keep the vertical lift component the network paired with its load factor.
        coordinated = (load * torch.cos(bank) / torch.cos(filtered)).clamp(min=MIN_LOAD_FACTOR, max=MAX_LOAD_FACTOR)
        change = (filtered - bank).abs().detach()
        gated = (weight > 0.5).detach()
        counts = torch.stack((
            bank.new_full((), float(bank.numel()), dtype=torch.float64),
            gated.sum().to(torch.float64),
            (change > _ACTIVE_BANK_CHANGE_RAD).sum().to(torch.float64),
            (gated & ((bank_max - bank_min).detach() < _SATURATED_INTERVAL_RAD)).sum().to(torch.float64),
            change.sum().to(torch.float64),
            (coordinated - load).abs().detach().sum().to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        return torch.stack((command[:, 0], filtered, coordinated), dim=-1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Step counts (and the summed bank change) over every call; read on the host once."""
        counts = torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64) if self._counts is None else self._counts.cpu()
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))
