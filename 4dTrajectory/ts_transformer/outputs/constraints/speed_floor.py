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

**Under the specific-force law** (``control_thrust_parameterization="specific-force"``,
``docs/2026-09-14_specific_force_control_design.md``) the first column is ``n_x = (T - D)/W``
and the RHS makes ``V' = g·(a_x - sin γ)`` wherever the thrust clamp does not bind, so the
same inversion is drag-free and mass-free::

    a_mean = [a_c (dt - tau_eff) + a_0 tau_eff] / dt
    a_c    = (a_req·dt - a_0·tau_eff) / (dt - tau_eff),   a_req = (V_floor - V)/(g·dt) + sin γ

with the lag credit on the specific-force actuator ``a_0`` — credited no higher than the
engine gives, because the RHS's thrust clamp caps what a spooled actuator above it flies.
What the airframe still decides is the CEILING: the most the engine gives here,
``(T_max - D)/W``, with D read at the segment-start speed, the end-of-hold density and the
commanded load factor (the floor's own state). "Saturated" means exactly what it means under
thrust-fraction: the demand reached full thrust. **The hook is NOT held inside the head's
box**: that box (up to 0.23 g) is the network's search space, sized for the imitation
scale, and on this fleet it sits BELOW the engine's ceiling (0.24-0.33 g at 1.1·V_s, 1 g,
M2 review) — capping a physical safety layer there would give it less authority than the
thrust-fraction floor, whose box IS the engine. The soft width is the same share of the
longitudinal box under both contracts (0.02 of the thrust-fraction box's 0.6 half width).
``hook_thrust_change`` is then a change of specific force, in g; a record says which by
``source.controlThrustParameterization``.
"""

from __future__ import annotations

import torch

from aerodynamic_model.torch_dynamics import (
    GRAVITY_MPS2,
    aerodynamic_coefficients,
    drag_force_n,
    isa_density,
)
from ts_transformer.config import CONTROL_SPECIFIC_FORCE, TSConfig
from ts_transformer.outputs.constraints.gates import RunwayAxesView, runway_axes_view
from ts_transformer.outputs.constraints.saturation import SOFTPLUS_LINEAR_THRESHOLD, soft_max
from ts_transformer.outputs.dynamics.hooks import HOOK_STEPS_KEY, RolloutStateView
from ts_transformer.outputs.envelope import (
    MAX_THRUST_FRACTION,
    MIN_THRUST_FRACTION,
    THRUST_FRACTION_CONTRACT,
    control_contract,
)

#: Width of the soft max around the thrust demand, in fractions of installed thrust. The
#: barrier's soft bank saturation is 2 deg of a +/-45 deg box, i.e. ~4 % of that box's half
#: width; 0.02 is the same order on a thrust box whose half width is 0.6.
SATURATION_SOFTNESS_FRACTION = 0.02
#: How far below the envelope a demand that binds nothing is parked. It must be far enough
#: that ``soft_max`` is EXACTLY the command there — the scaled softplus overshoots its bound
#: by ``softness x ln 2``, so a demand clamped to ``MIN_THRUST_FRACTION`` would add 2.8 kN to
#: every idle command the floor never touched (measured on the fixture: 0.0139 of installed
#: thrust, 0.042 m/s^2, ~12 m/s over a 300 s remainder — and the whole ``[-0.2, -0.1]`` band
#: is the COMMON case on an approach, see ``outputs/envelope.py``). ``softplus`` is linear
#: past ~20, so twenty softnesses below the box is inert to 1e-8 in float32. It must still be
#: a FLOOR: an unclamped demand of -1e6 cancels in ``bound + softness * softplus(...)`` and
#: reintroduces a larger error than the one this avoids.
_INERT_DEMAND = MIN_THRUST_FRACTION - SOFTPLUS_LINEAR_THRESHOLD * SATURATION_SOFTNESS_FRACTION
_DIAGNOSTIC_KEYS = (
    HOOK_STEPS_KEY, "hook_floor_bound_steps", "hook_floor_saturated_steps",
    "hook_thrust_change",
)


def floor_speed(
    view: RunwayAxesView,
    *,
    aero: torch.Tensor,
    origin_altitude_m: torch.Tensor,
    margin: float,
    commanded_load: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(V_floor, rho)`` for this segment: the floor, and the density it was read at.

    One definition, because a second module now asks the same question. The trombone turns
    "the delay the remaining path cannot absorb AT THE FLOOR SPEED" into an extra path
    length, and the speed it divides by has to be the speed THIS module holds on the same
    segment — a detour sized against a slightly different floor would be one the floor then
    does not need, or one it still does.

    The height is the one the command's effect is measured at (``u + vu*dt``), the airframe
    row is the flight's own ``(S, Cl_max)`` and the load factor is the COMMANDED one: this
    is ``flyability``'s stall criterion read as a speed, at the state it will be graded at.
    """
    altitude = origin_altitude_m + view.height + view.vertical_speed * view.hold_s
    density = isa_density(altitude)
    return margin * stall_speed_mps(commanded_load, view.mass, density, aero[:, 0], aero[:, 1]), density


def stall_speed_mps(load, mass_kg, density, area_m2, cl_max):
    """The stall speed at load factor ``load``, ``√(2 n m g / (ρ S Cl_max))`` — the ONE
    expression behind the tensor floor above and the plan's scalar closure
    (`outputs.plan.guidance.timing.stall_floor_mps`); ``** 0.5`` so it reads a tensor or a
    float alike."""
    return (2.0 * load * mass_kg * GRAVITY_MPS2 / (density * area_m2 * cl_max)) ** 0.5


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
        # WHICH quantity the first column is (the module docstring's last section).
        self.specific_force = config.control_thrust_parameterization == CONTROL_SPECIFIC_FORCE
        contract = control_contract(config.control_thrust_parameterization)
        # The head's floor: an in-box command is always at or above it (the hook's parking).
        self.command_floor = contract.lower[0]
        self.softness = (
            SATURATION_SOFTNESS_FRACTION
            * contract.half_width[0] / THRUST_FRACTION_CONTRACT.half_width[0]
        )
        # ``[len(_DIAGNOSTIC_KEYS), B]`` — per row, so a prediction record carries its own.
        self._counts: torch.Tensor | None = None

    def __call__(
        self, state: RolloutStateView, command: torch.Tensor, segment_index: int
    ) -> torch.Tensor:
        if self.specific_force:
            return self._specific_force_floor(state, command)
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        hold = view.hold_s
        thrust, bank, load = command[:, 0], command[:, 1], command[:, 2]
        aero = self.aero_params.to(view.d.dtype)
        area = aero[:, 0]
        commanded_load = load.to(view.d.dtype)
        # The floor at the height the command's effect is measured at, and the density there.
        floor, density = floor_speed(
            view,
            aero=aero,
            origin_altitude_m=self.origin_altitude_m.to(view.d.dtype),
            margin=self.margin,
            commanded_load=commanded_load,
        )
        # Drag at the segment start, under the commanded load factor: the same coefficient
        # function the RHS integrates, so the hook and the dynamics share one polar.
        _cl, cd, _stalled = aerodynamic_coefficients(
            commanded_load, view.speed, view.mass, density, aero
        )
        drag = drag_force_n(density, view.speed, cd, area)
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
        self._count(bound, saturated, change)
        return torch.stack((raised, bank, load), dim=-1)

    def _specific_force_floor(
        self, state: RolloutStateView, command: torch.Tensor
    ) -> torch.Tensor:
        """The same floor under the specific-force law: the demand is drag- and mass-free,
        the airframe enters only through the engine's ceiling at this state."""
        view = runway_axes_view(state, self.runway_heading)
        dtype = command.dtype
        hold = view.hold_s
        specific_force, bank, load = command[:, 0], command[:, 1], command[:, 2]
        aero = self.aero_params.to(view.d.dtype)
        commanded_load = load.to(view.d.dtype)
        floor, density = floor_speed(
            view,
            aero=aero,
            origin_altitude_m=self.origin_altitude_m.to(view.d.dtype),
            margin=self.margin,
            commanded_load=commanded_load,
        )
        # The engine's ceiling here, in g: full thrust less the drag (the RHS's own polar).
        _cl, cd, _stalled = aerodynamic_coefficients(
            commanded_load, view.speed, view.mass, density, aero
        )
        drag = drag_force_n(density, view.speed, cd, aero[:, 0])
        engine = (self.max_thrust_n.to(view.d.dtype) - drag) / (view.mass * GRAVITY_MPS2)
        # The mean specific force over the hold that leaves the speed on the floor, with the
        # spooled actuator credited no higher than the engine lets it fly.
        mean_required = (floor - view.speed) / (GRAVITY_MPS2 * hold) + torch.sin(view.path_angle)
        tau_eff = self.thrust_lag_s * (1.0 - torch.exp(-hold / self.thrust_lag_s))
        flying = torch.minimum(state.actuators.to(view.d.dtype)[:, 0], engine)
        demand = (mean_required * hold - flying * tau_eff) / (hold - tau_eff)
        # A demand that binds nothing is parked ONE softness past softplus's linear
        # threshold below the box floor, so every command in the box (the head's floor) is
        # STRICTLY past the switch below and the soft form returns it exactly — by
        # construction, not by how `x - b` happens to round (M2 review).
        switch = SOFTPLUS_LINEAR_THRESHOLD * self.softness
        parked = self.command_floor - switch - self.softness
        bounded = torch.minimum(demand.clamp(min=parked), engine).to(dtype)
        if self.hard:
            raised = torch.maximum(specific_force, bounded)
        else:
            raised = torch.where(
                specific_force - bounded >= switch,
                specific_force,
                soft_max(specific_force, bounded, self.softness),
            )
            # The soft form overshoots its bound by up to `softness * ln 2`: never past the
            # engine on a demand the command did not already exceed.
            raised = torch.minimum(raised, torch.maximum(engine.to(dtype), specific_force))
        change = (raised - specific_force).abs().detach()
        bound = (bounded - specific_force).detach() > 0.0
        # "Full thrust is not enough", as under thrust-fraction — the engine, not the box.
        saturated = (demand >= engine).detach()
        self._count(bound, saturated, change)
        return torch.stack((raised, bank, load), dim=-1)

    def _count(self, bound: torch.Tensor, saturated: torch.Tensor, change: torch.Tensor) -> None:
        counts = torch.stack((
            torch.ones_like(change, dtype=torch.float64),
            bound.to(torch.float64),
            saturated.to(torch.float64),
            change.to(torch.float64),
        ))
        self._counts = counts if self._counts is None else self._counts + counts
        # One hook per batch (built in `objective`, `validation` and `forecast` per call):
        # the rows ARE the flights, so a second batch through the same hook would broadcast
        # into the first one's rows instead of failing.
        assert self._counts.shape[1] == counts.shape[1], "a hook is built per batch"

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

    def diagnostic_labels(self) -> dict[str, str]:
        """No named variants: this module computes one thing one way."""
        return {}
