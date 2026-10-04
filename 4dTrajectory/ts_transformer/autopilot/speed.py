"""The speed law (executor design §6): the speed word as an airspeed rate.

- A ground-speed word ``V_g*`` is flown as the airspeed ``V_g* / cos γ`` (no wind).
- "Unspecified" is the pilot's own speed: the aircraft TYPE's published approach speed
  (`aircraft.reference_speeds`, every number traced to its document), quoted as an indicated speed at the
  maximum landing weight, scaled by ``√(m / MALW)`` and flown as the true airspeed ``· √(ρ0 / ρ(h))``. The
  type is the flight's IDENTIFIED one (`scenario.source["resolved_typecode"]`), never the dynamics'
  stand-in; a flight whose type publishes no approach speed cannot fly an "unspecified" stretch. A flight
  on a stand-in's dynamics carries the stand-in's mass, so its type's speed is taken as published, unscaled.
- The rate: ``V̇* = sat((V_ref − V) / τ_V, ±a)`` with ``τ_V = δv / a`` — a constant acceleration or
  deceleration ``a`` until one band half-width δv from the target, then an exponential approach, so the
  transition is monotone and does not pass the target. For a speed word ``a`` is a_max, the speed envelope's largest
  acceleration (`speed_accel_max_mps2` of the spec; design §5.6, D43): a speed word is a step of one grid value, so the
  executor makes each step in a few seconds and the rate of a longer change comes from the words, as the turn rate comes
  from the heading words. Where the thrust cannot give a_max the thrust limit binds (`inverse.thrust`), counted.
- "Unspecified" has its own rate: the pace a_U (`speed_change_mps2`: one speed step over the shortest hold a speed word
  keeps, 0.25 m/s²; the user's choice, 2026-09-24). The pilot's own speed is the one it lands at: slowing to it takes
  a_U, or the deceleration that reaches it over the straight-line distance left to the threshold when that is harder
  (the shortest path there, so it errs early) — at most a_max. Said late on a long final at a high speed, the pace alone
  crossed the threshold 20 m/s over the type's window.
- While G is true, "unspecified" is the pilot's own speed in a missed approach: the airspeed the aircraft had when it
  heard the go-around, held (D27; Claude's reading, not checked in the regulation text). A speed word replaces it at
  any row.
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


def speed_change_mps2(spec: VocabularySpec) -> float:
    """a_U, the pace of "unspecified", m/s²: one of the vocabulary's speed steps over the shortest hold a speed word keeps
    (5 m/s over 20 s = 0.25 m/s²; the user's choice, 2026-09-24; design §5.6)."""
    return spec.speed_step_mps / spec.speed_min_hold_s


class Speed:
    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("held_mps",)

    def __init__(self, approach_ias_mps: torch.Tensor, spec: VocabularySpec) -> None:
        self.approach_ias_mps = approach_ias_mps
        self.band_mps, self.accel_max_mps2 = spec.speed_tolerance_mps, spec.speed_accel_max_mps2
        self.pace_mps2 = speed_change_mps2(spec)
        self.margin = EXECUTOR_DYNAMICS.control_speed_floor_margin
        #: the airspeed each flight had when it last heard a go-around (NaN: none heard)
        self.held_mps = torch.full_like(approach_ias_mps, math.nan)

    def hear_go_around(self, heard: torch.Tensor, state: Kinematics) -> None:
        """The flights ``heard`` (``[B]`` bool) hear a go-around this cycle: their airspeed now is the one "unspecified"
        holds while it is in force (module docstring)."""
        self.held_mps = torch.where(heard, state.speed_mps, self.held_mps)

    def rate(self, state: Kinematics, speed_mps: torch.Tensor, unspecified: torch.Tensor, go_around: torch.Tensor,
             load_factor: torch.Tensor, aero_params: torch.Tensor,
             straight_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The airspeed rate for this cycle, the rate the law wanted before the stall floor, and whether the
        floor set it; ``go_around`` the go-around state G; ``straight_m`` is the straight-line distance to the pointed
        threshold."""
        if (unspecified & ~go_around & self.approach_ias_mps.isnan()).any():
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        density = isa_density(state.height_m)
        own = self.approach_ias_mps * torch.sqrt(ISA_RHO0_KG_M3 / density)
        wanted = torch.where(unspecified, torch.where(go_around, self.held_mps, own),
                             speed_mps / torch.cos(state.gamma_rad))
        floor = self.margin * stall_speed_mps(load_factor, state.mass_kg, density, aero_params[:, 0], aero_params[:, 1])
        landing = ((state.speed_mps.square() - own.square()) / (2.0 * straight_m.clamp(min=1.0))).clamp(
            self.pace_mps2, self.accel_max_mps2)
        # a speed word at a_max both ways (D43); "unspecified" at its own pace, slowing to land at the landing rate
        rising = torch.where(unspecified, torch.full_like(landing, self.pace_mps2),
                             torch.full_like(landing, self.accel_max_mps2))
        slowing = torch.where(unspecified & ~go_around, landing, rising)

        def toward(reference: torch.Tensor) -> torch.Tensor:
            faster = reference > state.speed_mps
            tau = self.band_mps / torch.where(faster, rising, slowing)
            return torch.minimum(torch.maximum((reference - state.speed_mps) / tau, -slowing), rising)

        return toward(torch.maximum(wanted, floor)), toward(wanted), {"stall_floor": floor > wanted}
