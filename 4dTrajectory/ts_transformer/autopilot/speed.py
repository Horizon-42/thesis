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
  transition is monotone and does not pass the target. ``a`` is the vocabulary's own pace of a speed change
  (`speed_change_mps2`: one speed step over the shortest hold a speed word keeps; the executor takes nothing
  beyond the vocabulary, 2026-09-24). The pilot's own speed is the one it lands at: slowing to it takes ``a``, or
  the deceleration that reaches it over the straight-line distance left to the threshold when that is harder (the
  shortest path there, so it errs early) — at most the vocabulary's largest acceleration. Said late on a long final at a
  high speed, the pace alone crossed the threshold 20 m/s over the type's window.
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
    """The executor's pace of a speed change, m/s²: one of the vocabulary's speed steps over the shortest hold a speed
    word keeps (5 m/s over 20 s = 0.25 m/s²; the user's choice, 2026-09-24). On train it flies the speed words as the
    data's measured paces did (0.28 / 0.19 / 0.30 m/s²): words inside 97.2 against 97.5 %, every speed word in its band."""
    return spec.speed_step_mps / spec.speed_min_hold_s


class Speed:
    def __init__(self, approach_ias_mps: torch.Tensor, spec: VocabularySpec) -> None:
        self.approach_ias_mps = approach_ias_mps
        self.band_mps, self.accel_max_mps2 = spec.speed_tolerance_mps, spec.speed_accel_max_mps2
        self.pace_mps2 = speed_change_mps2(spec)
        self.margin = EXECUTOR_DYNAMICS.control_speed_floor_margin

    def rate(self, state: Kinematics, speed_mps: torch.Tensor, unspecified: torch.Tensor, go_around: torch.Tensor,
             load_factor: torch.Tensor, aero_params: torch.Tensor,
             straight_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The airspeed rate for this cycle, the rate the law wanted before the stall floor, and whether the
        floor set it; a go-around holds the airspeed it has (§4.6); ``straight_m`` is the straight-line distance
        to the pointed threshold."""
        if (unspecified & self.approach_ias_mps.isnan()).any():
            raise ValueError("\"unspecified\" in force for a flight whose type publishes no approach speed")
        density = isa_density(state.height_m)
        own = self.approach_ias_mps * torch.sqrt(ISA_RHO0_KG_M3 / density)
        wanted = torch.where(go_around, state.speed_mps,
                             torch.where(unspecified, own, speed_mps / torch.cos(state.gamma_rad)))
        floor = self.margin * stall_speed_mps(load_factor, state.mass_kg, density, aero_params[:, 0], aero_params[:, 1])
        landing = ((state.speed_mps.square() - own.square()) / (2.0 * straight_m.clamp(min=1.0))).clamp(
            self.pace_mps2, self.accel_max_mps2)
        slowing = torch.where(unspecified, landing, torch.full_like(landing, self.pace_mps2))

        def toward(reference: torch.Tensor) -> torch.Tensor:
            faster = reference > state.speed_mps
            tau = self.band_mps / torch.where(faster, torch.full_like(slowing, self.pace_mps2), slowing)
            return torch.maximum(((reference - state.speed_mps) / tau).clamp(max=self.pace_mps2), -slowing)

        return toward(torch.maximum(wanted, floor)), toward(wanted), {"stall_floor": floor > wanted}
