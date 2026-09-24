"""Wanted rates → the three controls, exactly (executor design §3), with the limits in their order (§7).

The outer laws ask for a compass track rate χ̇*, a path-angle rate γ̇* and an airspeed rate V̇*. The
point-mass equations (`aerodynamic_model.torch_dynamics.enu_rhs`) are solved for the controls that give
them — the inversion `geometry.flyability.required_controls` makes, run forwards:

    A = −χ̇* V cos γ / g          (= n sin φ; the minus because compass and math turn opposite ways)
    B =  γ̇* V / g + cos γ         (= n cos φ)
    φ = atan2(A, B),   n = B / cos φ
    T = m (V̇* + g sin γ) + D(V, n, h)

The drag is the dynamics' own (`flight_aerodynamics` at the commanded load factor, ISA density at the
geometric height), so the inverse and the integration share one polar and nothing here is tuned.

Each limit acts on the quantity it bounds, in this order, and every one that binds is recorded (§8.1):

1. bank — at most ``bank_cap_rad``, and moving at most ``bank_rate_rad_s`` per second (the roll rate
   wins: a bank outside the cap comes back at that rate); the load factor is then ``B / cos φ``, which
   keeps γ̇, so a limited bank costs turn rate and never the path (§7.2). ``B ≤ 0`` would ask for lift
   pointing down; it reads as ``B = 0`` (no bank is invented from the sign of a zero turn) and the load
   band binds;
2. load factor — inside `geometry.flyability.Envelope`'s band, the grader's (review C-12);
3. thrust — inside the thrust-fraction contract's box (`outputs/envelope.py`); what a limited thrust
   costs is speed rate (§7.4).

The stall floor comes first in the design's order but belongs to the speed law (§6): it caps the speed
the law asks for, reading the load factor :func:`attitude` sets, before :func:`thrust` is solved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, FlightCondition, flight_aerodynamics, isa_density
from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.geometry.flyability import Envelope
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION, MIN_THRUST_FRACTION

_GRADER = {field.name: field.default for field in fields(Envelope)}
LOAD_FACTOR_MIN = float(_GRADER["min_load_factor"])
LOAD_FACTOR_MAX = float(_GRADER["max_load_factor"])
BANK_MAX_RAD = float(_GRADER["max_bank_rad"])


@dataclass(frozen=True)
class Attitude:
    bank_rad: torch.Tensor          # [B], the dynamics' sign: positive turns left
    load_factor: torch.Tensor       # [B]
    binds: dict[str, torch.Tensor]  # "bank_cap", "bank_rate", "load_factor": [B] bool


def attitude(state: Kinematics, track_rate_deg_s: torch.Tensor, gamma_rate_rad_s: torch.Tensor,
             previous_bank_rad: torch.Tensor, *, bank_cap_rad: float | torch.Tensor, bank_rate_rad_s: float,
             cycle_s: float) -> Attitude:
    """Bank and load factor for the wanted track and path-angle rates (limits 1 and 2); ``bank_cap_rad`` one cap, or
    one per flight."""
    cap = torch.as_tensor(bank_cap_rad, dtype=track_rate_deg_s.dtype, device=track_rate_deg_s.device)
    if not bool(((cap > 0.0) & (cap <= BANK_MAX_RAD)).all()):
        raise ValueError(f"bank cap {torch.rad2deg(cap).tolist()}° outside (0, {math.degrees(BANK_MAX_RAD):.0f}°]")
    if not (bank_rate_rad_s > 0.0 and cycle_s > 0.0):
        raise ValueError(f"the bank rate ({bank_rate_rad_s}) and the cycle ({cycle_s}) must be positive")
    speed, gamma = state.speed_mps, state.gamma_rad
    a = -torch.deg2rad(track_rate_deg_s) * speed * torch.cos(gamma) / GRAVITY_MPS2
    b = gamma_rate_rad_s * speed / GRAVITY_MPS2 + torch.cos(gamma)
    wanted = torch.atan2(a, b.clamp(min=0.0))
    capped = torch.minimum(torch.maximum(wanted, -cap), cap)
    step = bank_rate_rad_s * cycle_s
    bank = torch.minimum(torch.maximum(capped, previous_bank_rad - step), previous_bank_rad + step)
    load = b / torch.cos(bank)
    flown = load.clamp(LOAD_FACTOR_MIN, LOAD_FACTOR_MAX)
    return Attitude(bank_rad=bank, load_factor=flown,
                    binds={"bank_cap": capped != wanted, "bank_rate": bank != capped, "load_factor": flown != load})


@dataclass(frozen=True)
class Thrust:
    fraction: torch.Tensor          # [B], of the installed thrust
    binds: dict[str, torch.Tensor]  # "thrust_max", "thrust_min", "stall": [B] bool


def thrust(state: Kinematics, accel_mps2: torch.Tensor, load_factor: torch.Tensor, aero_params: torch.Tensor,
           max_thrust_n: torch.Tensor) -> Thrust:
    """The thrust fraction for the wanted airspeed rate at the load factor :func:`attitude` set
    (limit 3). ``stall`` records a load factor the wing cannot give at this speed — the speed floor
    exists so that it never does."""
    condition = FlightCondition(speed_mps=state.speed_mps, sin_gamma=torch.sin(state.gamma_rad),
                                mass_kg=state.mass_kg, density=isa_density(state.height_m))
    polar = flight_aerodynamics(condition, load_factor, aero_params)
    wanted = (state.mass_kg * (accel_mps2 + GRAVITY_MPS2 * condition.sin_gamma) + polar.drag_n) / max_thrust_n
    flown = wanted.clamp(MIN_THRUST_FRACTION, MAX_THRUST_FRACTION)
    return Thrust(fraction=flown, binds={"thrust_max": wanted > MAX_THRUST_FRACTION,
                                         "thrust_min": wanted < MIN_THRUST_FRACTION, "stall": polar.stalled})
