"""The speed law (executor design §8): the speed word as an airspeed rate.

- A ground-speed word ``V_g*`` is flown as the airspeed ``V_g* / cos γ`` (no wind).
- "Unspecified" is the pilot's own speed: the MIDDLE of the evaluation's speed window at the threshold
  (`evaluation.speed_gate.speed_gate_bounds`, the user 2026-10-13), from the published approach speeds of the type the
  flight is flown on (`aircraft.reference_speeds`, every number traced to its document; for a flight on a stand-in's
  dynamics, the stand-in's: the evaluation reads that type, and the flight carries its mass). The window's lower edge
  is the lower flap configuration's speed, its upper edge the upper one's plus 20 kt; both are indicated speeds at the
  maximum landing weight, scaled by ``√(m / MALW)`` to the mass now, and the lower edge is raised by ``√n`` at a load
  factor above 1 (the stall speed grows with ``√n``, and the published speed is a 1 g speed: physics, §8.1). The
  middle is flown as the true airspeed ``· √(ρ0 / ρ(h))``. A flight whose type publishes no approach speed cannot fly
  an "unspecified" stretch.
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
from evaluation.speed_gate import MIN_BOUND_LOAD_FACTOR, SPEED_GATE_UPPER_ADDITIVE_MS
from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.outputs.constraints.speed_floor import stall_speed_mps


def approach_speeds(typecode: str) -> tuple[float, float, float]:
    """The type's published approach speeds (module docstring): the lower and the upper flap configuration's,
    indicated, m/s at its maximum landing weight, and that weight, kg. NaN when the type publishes none."""
    reference = reference_speed(typecode)
    if reference is None:
        return (math.nan, math.nan, math.nan)
    return reference.approach_speed_min_kt * KT_MS, reference.approach_speed_max_kt * KT_MS, reference.malw_kg


def unspecified_ias_mps(speeds: torch.Tensor, mass_kg: torch.Tensor, load_factor: torch.Tensor) -> torch.Tensor:
    """The middle of the evaluation's speed window (module docstring), indicated, m/s: ``speeds`` ``[B, 3]`` the
    flights' `approach_speeds`, at their mass and load factor now. The window's edges are `speed_gate_bounds`'s."""
    scale = torch.sqrt(mass_kg / speeds[:, 2])
    lower = speeds[:, 0] * scale * torch.sqrt(load_factor.clamp(min=MIN_BOUND_LOAD_FACTOR))
    upper = speeds[:, 1] * scale + SPEED_GATE_UPPER_ADDITIVE_MS
    return (lower + upper) / 2.0


class Speed:
    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("held_mps", "issued", "heard_s")

    def __init__(self, approach_speeds: torch.Tensor, params: ExecutorParams, spec: VocabularySpec) -> None:
        """``approach_speeds`` ``[B, 3]``: each flight's `approach_speeds`."""
        self.approach_speeds = approach_speeds
        self.cycle_s, self.lead_s = params.cycle_s, spec.speed_lead_s
        #: the airspeed each flight had when it last heard a go-around (NaN: none heard)
        self.held_mps = torch.full_like(approach_speeds[:, 0], math.nan)
        #: the step the speed word in force was written at, and when the executor heard it
        self.issued = torch.full(self.held_mps.shape, -1, dtype=torch.long, device=approach_speeds.device)
        self.heard_s = torch.zeros_like(self.held_mps)

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
        if (unspecified & ~go_around & self.approach_speeds.isnan().any(dim=1)).any():
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        new_word = issued != self.issued
        self.heard_s = torch.where(new_word, time_s, self.heard_s)
        self.issued = issued.clone()
        density = isa_density(state.height_m)
        floor = stall_speed_mps(load_factor, state.mass_kg, density, aero_params[:, 0], aero_params[:, 1])
        own = (unspecified_ias_mps(self.approach_speeds, state.mass_kg, load_factor)
               * torch.sqrt(ISA_RHO0_KG_M3 / density))
        wanted = torch.where(unspecified, torch.where(go_around, self.held_mps, own),
                             speed_mps / torch.cos(state.gamma_rad))
        to_go = (self.heard_s + self.lead_s - time_s).clamp(min=self.cycle_s)
        return ((torch.maximum(wanted, floor) - state.speed_mps) / to_go, (wanted - state.speed_mps) / to_go,
                {"stall_floor": floor > wanted})
