"""The speed law (executor design §6): the speed word as an airspeed rate.

- A ground-speed word ``V_g*`` is flown as the airspeed ``V_g* / cos γ`` (no wind).
- "Unspecified" is the pilot's own speed: the aircraft TYPE's published approach speed
  (`aircraft.reference_speeds`, every number traced to its document), quoted as an indicated speed at the
  maximum landing weight, scaled by ``√(m / MALW)`` and flown as the true airspeed ``· √(ρ0 / ρ(h))``. The
  type is the flight's IDENTIFIED one (`scenario.source["resolved_typecode"]`), never the dynamics'
  stand-in; a flight whose type publishes no approach speed cannot fly an "unspecified" stretch. A flight
  on a stand-in's dynamics carries the stand-in's mass, so its type's speed is taken as published, unscaled.
- The rate: ``V̇* = sat((V_ref − V) / τ_V, [−a, +a_acc])`` with ``τ_V = δv / a`` — a constant
  deceleration (``a_dec``, or ``a_unspec`` for the pilot's own) until one band half-width δv from the
  target, then an exponential approach, so the transition is monotone and does not pass the target.
- The stall floor: ``V_ref ≥ margin · V_stall(n)`` at the load factor the inverse commands this cycle
  (`outputs.constraints.speed_floor.stall_speed_mps`, the control path's margin).
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import ISA_RHO0_KG_M3, isa_density
from aircraft.reference_speeds import reference_speed
from geokit import KT_MS
from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.outputs.constraints.speed_floor import stall_speed_mps


def approach_speed_ias_mps(typecode: str | None, mass_kg: float | None) -> float:
    """The type's published approach speed, indicated, m/s: at this mass (``√(m / MALW)``), or as published
    — at its maximum landing weight — when ``mass_kg`` is None (a flight flown on a stand-in's dynamics carries
    the stand-in's mass, not its own type's). NaN when the flight has no identified type or its type
    publishes none (it then cannot fly "unspecified")."""
    reference = None if typecode is None else reference_speed(typecode)
    if reference is None:
        return math.nan
    scale = 1.0 if mass_kg is None else math.sqrt(mass_kg / reference.malw_kg)
    return reference.approach_speed_kt * KT_MS * scale


class Speed:
    def __init__(self, approach_ias_mps: torch.Tensor, params: ExecutorParams, spec: VocabularySpec) -> None:
        self.approach_ias_mps = approach_ias_mps
        self.params, self.band_mps = params, spec.speed_tolerance_mps
        self.margin = EXECUTOR_DYNAMICS.control_speed_floor_margin

    def rate(self, state: Kinematics, speed_mps: torch.Tensor, unspecified: torch.Tensor, go_around: torch.Tensor,
             load_factor: torch.Tensor,
             aero_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The airspeed rate for this cycle, the rate the law wanted before the stall floor, and whether the
        floor set it; a go-around holds the airspeed it has (§4.6)."""
        if (unspecified & self.approach_ias_mps.isnan()).any():
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        params = self.params
        density = isa_density(state.height_m)
        own = self.approach_ias_mps * torch.sqrt(ISA_RHO0_KG_M3 / density)
        wanted = torch.where(go_around, state.speed_mps,
                             torch.where(unspecified, own, speed_mps / torch.cos(state.gamma_rad)))
        floor = self.margin * stall_speed_mps(load_factor, state.mass_kg, density, aero_params[:, 0], aero_params[:, 1])
        slowing = torch.where(unspecified, params.unspecified_decel_mps2, params.decel_mps2)

        def toward(reference: torch.Tensor) -> torch.Tensor:
            faster = reference > state.speed_mps
            tau = self.band_mps / torch.where(faster, torch.full_like(slowing, params.accel_mps2), slowing)
            return torch.maximum(((reference - state.speed_mps) / tau).clamp(max=params.accel_mps2), -slowing)

        return toward(torch.maximum(wanted, floor)), toward(wanted), {"stall_floor": floor > wanted}
