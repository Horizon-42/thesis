"""The speed law (executor design §8): the speed word as an airspeed rate.

- A ground-speed word ``V_g*`` is flown as the airspeed ``V_g* / cos γ`` (no wind).
- "Unspecified" is the pilot's own speed: the aircraft TYPE's published approach speed
  (`aircraft.reference_speeds`, every number traced to its document), quoted as an indicated speed at the
  maximum landing weight, scaled by ``√(m / MALW)`` to the mass, raised by ``√n`` at a load factor above 1 (the stall
  speed grows with ``√n``, and the published speed is a 1 g speed: physics, §8.1) and flown as the true airspeed
  ``· √(ρ0 / ρ(h))``. The type is the flight's IDENTIFIED one (`scenario.source["resolved_typecode"]`), never the
  dynamics' stand-in; a flight whose type publishes no approach speed cannot fly an "unspecified" stretch. A flight on a
  stand-in's dynamics carries the stand-in's mass, so its type's speed is taken as published, unscaled.
- While G is true, "unspecified" is the pilot's own speed in a missed approach: the airspeed the aircraft had when it
  heard the go-around, held (D27; Claude's reading, not checked in the regulation text). A speed word replaces it at
  any row.
- No speed is asked for below the stall speed at the load factor the inverse sets this cycle (D3: the stall itself, no
  margin; `outputs.constraints.speed_floor.stall_speed_mps`).

THE RATE (§8.2, R5.5, D8). A speed word ("unspecified" too) says the speed a lead ``L_v`` later: the law arrives on it
then, the change left spread over the time left, ``V̇* = (V* − V) / max(t_heard + L_v − t, Δt)`` — at least one cycle,
the shortest time the executor acts in; the same rule as the heading and angle words. Nothing limits it but the thrust
box (`inverse.thrust`): no acceleration value, no distance to the threshold (§8.2.3).
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import ISA_RHO0_KG_M3, isa_density
from aircraft.reference_speeds import reference_speed
from geokit import KT_MS
from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
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
    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("held_mps", "issued", "heard_s")

    def __init__(self, approach_ias_mps: torch.Tensor, params: ExecutorParams, spec: VocabularySpec) -> None:
        self.approach_ias_mps = approach_ias_mps
        self.cycle_s, self.lead_s = params.cycle_s, spec.speed_lead_s
        #: the airspeed each flight had when it last heard a go-around (NaN: none heard)
        self.held_mps = torch.full_like(approach_ias_mps, math.nan)
        #: the step the speed word in force was written at, and when the executor heard it
        self.issued = torch.full(approach_ias_mps.shape, -1, dtype=torch.long, device=approach_ias_mps.device)
        self.heard_s = torch.zeros_like(approach_ias_mps)

    def hear_go_around(self, heard: torch.Tensor, state: Kinematics) -> None:
        """The flights ``heard`` (``[B]`` bool) hear a go-around this cycle: their airspeed now is the one "unspecified"
        holds while it is in force (module docstring)."""
        self.held_mps = torch.where(heard, state.speed_mps, self.held_mps)

    def rate(self, state: Kinematics, speed_mps: torch.Tensor, unspecified: torch.Tensor, go_around: torch.Tensor,
             issued: torch.Tensor, time_s: float | torch.Tensor, load_factor: torch.Tensor, aero_params: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The airspeed rate for this cycle, the rate the law wanted before the stall floor, and whether the floor set
        it; ``issued`` the step the speed word in force was written at (a new one starts the clock), ``time_s`` the
        time flown, ``load_factor`` the one the inverse sets this cycle."""
        if (unspecified & ~go_around & self.approach_ias_mps.isnan()).any():
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        new_word = issued != self.issued
        self.heard_s = torch.where(new_word, time_s, self.heard_s)
        self.issued = issued.clone()
        density = isa_density(state.height_m)
        floor = stall_speed_mps(load_factor, state.mass_kg, density, aero_params[:, 0], aero_params[:, 1])
        own = (self.approach_ias_mps * torch.sqrt(load_factor.clamp(min=1.0)) * torch.sqrt(ISA_RHO0_KG_M3 / density))
        wanted = torch.where(unspecified, torch.where(go_around, self.held_mps, own),
                             speed_mps / torch.cos(state.gamma_rad))
        to_go = (self.heard_s + self.lead_s - time_s).clamp(min=self.cycle_s)
        return ((torch.maximum(wanted, floor) - state.speed_mps) / to_go, (wanted - state.speed_mps) / to_go,
                {"stall_floor": floor > wanted})
