"""The per-step speed floor: a stall margin held THROUGH the thrust command.

The measured problem (L3.c, ``docs/2026-09-07_latent_intent_design.zh.md`` §六 L3.c): on the
CTA base the control output is unflyable almost everywhere, 99.9 % of the hard violations
are the STALL term, and **77–94 % of the stall samples sit at ≥ 20 km remaining** — the
model commands infeasibly low speeds on the OUTER segment, and a late CTA pushes more of
the delay into exactly that segment. Clipping ``V`` after the rollout would not fix it: the
record would stop being a trajectory of the dynamics. So the floor is a command hook, the
way the corridor is, and it acts on the one control that changes speed.

**The floor.** ``flyability``'s only hard term that ever fires is
``Cl_required = 2 n m g / (rho V^2 S) > Cl_max``. Read as a speed that is exactly
``V < V_stall(n) = sqrt(2 n m g / (rho S Cl_max))`` — ``aircraft.aero_params.stall_speed_ms``'s
formula, evaluated at the flight's own ``(S, Cl_max)`` row of the batch ``aero_params`` (the
same numbers the RHS integrates and ``flyability`` grades against) and at the ISA density of
the height it will be at, not at sea level. The hook holds

    V >= V_floor = k_margin * V_stall(n_commanded, mass, rho, Cl_max)

with ``k_margin`` = ``config.control_speed_floor_margin`` (default 1.10, the package's
existing stall margin — see ``config.CONTROL_SPEED_FLOOR_MARGIN_DEFAULT``). ``n`` is the
COMMANDED load factor, which under ``barrier+speed-floor`` is the barrier's re-coordinated
one: the floor prices the manoeuvre the aircraft is actually being asked to fly.

**Through the controls, with the lag compensated.** Speed obeys
``V' = (T - D)/m - g sin(gamma)``, the command is HELD for the segment, and the thrust
actuator is first order with ``tau_T``, so over a hold ``dt`` the aircraft flies the thrust
it is at NOW, ``T_0`` (the actuator state the backend exposes), for ``tau_eff`` seconds'
worth and the command for the rest — the same credit the barrier gives a bank already
rolled into::

    T_mean = [T_c (dt - tau_eff) + T_0 tau_eff] / dt,   tau_eff = tau_T (1 - e^{-dt/tau_T})
    V(dt)  = V + [(T_mean - D)/m - g sin(gamma)] dt

Inverting ``V(dt) = V_floor`` for ``T_c`` gives the thrust the segment must be commanded at
for the speed to still be on the floor when the hold ends (``tau_eff < dt`` strictly, so
the inversion never divides by zero; as ``dt`` approaches ``tau_T`` the command's authority
inside the hold vanishes and the demand runs into the envelope, which is the conservative
direction and is counted). The command's thrust is then raised to that demand — softly (a
scaled softplus, C^1) or hard — and clamped to the envelope. **Bank and load factor are
returned untouched**: the floor owns one channel, the barrier owns the other two, which is
what makes the two composable.

The constraint is imposed at the END of the hold because that is where the speed is lowest
on a decelerating segment, and the density is read at the height reached there
(``u + vu*dt``): over one hold that moves ``V_floor`` by well under half a percent, but it
is the height at which the command's effect is measured, so it is the honest one. Drag and
the path angle are frozen at the segment start, and the drag freeze errs the safe way where
it matters: a hold in which the floor BINDS starts at ``V < V_floor``, i.e. on the back side
of the drag curve (induced drag goes as ``1/V^2``, plus the smoothstep stall drag), so
``D(V_start) > D(V_end)`` and the frozen value over-estimates the drag the command has to
overcome.

**Ungated, deliberately.** The barrier acts only where its gate says the aircraft is on the
final, because a corridor is a statement about the final approach. A stall is not: it is a
statement about the airframe, it fires wherever the model commands it, and the measurement
says it fires mostly far from the runway. A gate here would leave the entire problem
untouched.
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_dynamics import (
    GRAVITY_MPS2,
    aerodynamic_coefficients,
    isa_density,
)
from config import TSConfig
from control.constraints.gates import runway_axes_view
from control.constraints.saturation import soft_max
from control.dynamics.hooks import HOOK_STEPS_KEY, RolloutStateView
from control.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

#: Width of the soft max around the thrust demand, in fractions of installed thrust. The
#: barrier's soft bank saturation is 2 deg of a +/-45 deg box, i.e. ~4 % of that box's half
#: width; 0.02 is the same order on a thrust box whose half width is 0.6.
SATURATION_SOFTNESS_FRACTION = 0.02
#: How far below the envelope a demand that binds nothing is parked. It must be far enough
#: that ``soft_max`` is EXACTLY the command there — the scaled softplus overshoots its bound
#: by ``softness x ln 2``, so a demand clamped to ``MIN_THRUST_FRACTION`` would add 2.8 kN to
#: every idle command the floor never touched (measured on the fixture: 0.0139 of installed
#: thrust, 0.042 m/s^2, ~12 m/s over a 300 s remainder — and the whole ``[-0.2, -0.1]`` band
#: is the COMMON case on an approach, see ``control/envelope.py``). ``softplus`` is linear
#: past ~20, so twenty softnesses below the box is inert to 1e-8 in float32. It must still be
#: a FLOOR: an unclamped demand of -1e6 cancels in ``bound + softness * softplus(...)`` and
#: reintroduces a larger error than the one this avoids.
_INERT_DEMAND = MIN_THRUST_FRACTION - 20.0 * SATURATION_SOFTNESS_FRACTION
_DIAGNOSTIC_KEYS = (
    HOOK_STEPS_KEY, "hook_floor_bound_steps", "hook_floor_saturated_steps",
    "hook_thrust_change",
)


class SpeedFloor:
    needs_reference = False   # reads the hooked state only

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], *, hard: bool):
        self.runway_heading = dynamics["runway_heading_rad"]
        # (S, Cl_max, ...) and the frame's origin altitude: the floor is a density at a
        # geodetic height, and the chart carries height ABOVE that origin.
        self.aero_params = dynamics["aero_params"]
        self.origin_altitude_m = dynamics["frame_params"][:, 2]
        self.max_thrust_n = dynamics["max_thrust_n"]
        self.margin = config.control_speed_floor_margin
        self.thrust_lag_s = config.control_thrust_time_constant_s
        self.hard = hard
        # ``[len(_DIAGNOSTIC_KEYS), B]`` — per row, so a prediction record carries its own.
        self._counts: torch.Tensor | None = None

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        hold = view.hold_s
        thrust, bank, load = command[:, 0], command[:, 1], command[:, 2]
        aero = self.aero_params.to(view.d.dtype)
        area, cl_max = aero[:, 0], aero[:, 1]
        # The height the command's effect is measured at, and the density there.
        altitude = self.origin_altitude_m.to(view.d.dtype) + view.height + view.vertical_speed * hold
        density = isa_density(altitude)
        commanded_load = load.to(view.d.dtype)
        stall_speed = torch.sqrt(
            2.0 * commanded_load * view.mass * GRAVITY_MPS2 / (density * area * cl_max)
        )
        floor = self.margin * stall_speed
        # Drag at the segment start, under the commanded load factor: the same coefficient
        # function the RHS integrates, so the hook and the dynamics share one polar.
        _cl, cd, _stalled = aerodynamic_coefficients(
            commanded_load, view.speed, view.mass, density, aero
        )
        drag = 0.5 * density * view.speed.square() * cd * area
        # The mean thrust over the hold that leaves the speed on the floor when it ends.
        mean_required = view.mass * (
            (floor - view.speed) / hold + GRAVITY_MPS2 * torch.sin(view.path_angle)
        ) + drag
        max_thrust = self.max_thrust_n.to(view.d.dtype)
        # The actuator is first order: credit the thrust already being flown for the part
        # of the hold before the command bites (tau_eff < hold strictly, so this is safe).
        tau_eff = self.thrust_lag_s * (1.0 - torch.exp(-hold / self.thrust_lag_s))
        flying = state.actuators.to(view.d.dtype)[:, 0]
        demand = (mean_required / max_thrust * hold - flying * tau_eff) / (hold - tau_eff)
        bounded = demand.clamp(_INERT_DEMAND, MAX_THRUST_FRACTION).to(dtype)
        if self.hard:
            raised = torch.maximum(thrust, bounded)
        else:
            raised = soft_max(thrust, bounded, SATURATION_SOFTNESS_FRACTION)
        # The soft form overshoots its bound by up to `softness * ln 2`, so a demand that
        # already sits at full thrust would leave the box; the hard form cannot.
        raised = raised.clamp(min=MIN_THRUST_FRACTION, max=MAX_THRUST_FRACTION)
        change = (raised - thrust).abs().detach()
        # The floor BOUND when its demand exceeded what the network asked for — the physical
        # statement, read off the demand rather than the change, so the soft and hard forms
        # count the same steps. It SATURATED when the demand ran into the envelope, whether
        # or not it also exceeded the command: that is "full thrust is not enough", and
        # gating it on the change would hide it exactly where the network is already there.
        bound = (bounded - thrust).detach() > 0.0
        saturated = demand.detach().to(dtype) >= MAX_THRUST_FRACTION
        counts = torch.stack((
            torch.ones_like(thrust, dtype=torch.float64),
            bound.to(torch.float64),
            saturated.to(torch.float64),
            change.to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        # One hook per batch (built in `objective`, `validation` and `forecast` per call):
        # the rows ARE the flights, so a second batch through the same hook would broadcast
        # into the first one's rows instead of failing.
        assert self._counts.shape[1] == counts.shape[1], "a hook is built per batch"
        return torch.stack((raised, bank, load), dim=-1)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Step counts (and the summed thrust change) over every call, summed over the batch."""
        counts = (
            torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64)
            if self._counts is None else self._counts.sum(dim=1).cpu()
        )
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))

    def per_flight_diagnostics(self) -> dict[str, torch.Tensor]:
        """The same counts, one row per flight (``[B]`` each)."""
        if self._counts is None:
            return {name: torch.zeros(0, dtype=torch.float64) for name in _DIAGNOSTIC_KEYS}
        return dict(zip(_DIAGNOSTIC_KEYS, self._counts.cpu().unbind()))
