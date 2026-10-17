"""The vertical law (executor design §7; vocabulary §3.4, §3.5): the altitude word and the angle word, as a path-angle
rate. No landing aim, no glidepath floor (D3, D9): "no level-off" flies its class's angle and the words decide the
profile. Time comes only from the words, limits only from physics (§7.1).

The words give a CLASS ANGLE ``γ_k`` (its magnitude: the angle word's nominal, the climb's nominal, ``γ_GA`` for a climb
while G is true, 0 for "level") and, unless "no level-off" is in force, a level T (flown at T + E MSL, D58). The
direction comes from T and the present height, the steepness from the angle word (R5.4). Climbing positive:

| words in force                  | the goal angle       |
|---------------------------------|----------------------|
| the angle word "level"          | 0 — T is not chased (D5: "level" is said only at the first step, its T the present height rounded to the grid) |
| "no level-off" + descent k      | ``−γ_k``             |
| level T + a class               | ``−sign(h − T) γ_k`` |

THE RAMP (§7.2, R5.3). An angle word says the path angle a lead ``L_γ`` later; the altitude word said with it moves
with it (§17 C2), so a new altitude or angle word starts the clock: the goal is reached at ``t_heard + L_γ``, the change
left spread over the time left, ``γ_w = γ + (goal − γ) Δt / max(t_heard + L_γ − t, Δt)`` — at least one cycle, the
shortest time the executor acts in.

THE STOP (§7.3, D6). Toward a level the path angle is cut to the largest one the aircraft can still stop from before T,
at the physical limit: stopping a descent pulls to ``n_up = min(n_max, n_stall)``, stopping a climb pushes to ``n_min``.
The pull's path-angle rate is ``a = g (n_c cos φ̄ − cos γ) / V`` (a push's ``g (cos γ − n_min cos φ̄) / V``), reached
through the load factor's lag, which delays it by ``τ_n (n_c − n̂) / (n_c − cos γ / cos φ̄)`` (the asymptotic delay of a
first-order actuator stepped from n̂ to n_c), and by the cycle: the law decides once a cycle, so a stop it finds now
begins up to a cycle late — ``τ_d`` is the two together. Without the cycle the stop's gain near the level is that of a
loop whose time constant is under the cycle, and the hold rings (measured: ±2 m, ±2° without end; a sampled loop with
``τ_d < Δt`` has a root past −1). The height a stop from γ takes is then
``S(γ) = V τ_d sin γ + V (1 − cos γ) / a``, and ``S(γ_stop) = e'``, with e' the height left to T at the cycle's end,
solves in closed form: ``γ_stop = δ + asin((e' − B) / R)``, ``A = V τ_d``, ``B = V / a``, ``R = hypot(A, B)``,
``δ = atan2(B, A)``. The path angle flown toward T is ``min(|γ_w|, γ_stop)``.

A CLOSED-FORM APPROXIMATION of the stop (design §7.3), CONSERVATIVE every way it departs from the dynamics: the lag as a
pure delay (the actual stop ends sooner), constant V and φ̄ over the stop, and ``a`` and ``τ_d`` taken where they are
smallest over the stop — a pull's at γ = 0 (cos γ = 1: the rate shrinks as the descent flattens), a push's at the present
γ (it grows as the climb flattens). Its departure from a forward integration of the lag model is tested (T7).

CAPTURED (§7.6): a level whose stop has begun, or the angle word "level", is CAPTURED until a new altitude or angle word
arrives; then the goal is held — ``−sign(h − T) min(γ_k, γ_stop)`` — and the closed-loop reading starts no angle
correction (vocabulary §4.9 item 1, `closed_loop` reads ``captured``).

The inner loop (§7.5) flies ``γ̇* = (γ_target − γ) / Δt``; the inverse solves the load factor for it, within its
physical limits (`inverse.load`). No path-rate limit of its own.

THE GO-AROUND ANGLE (D28, vocabulary §5.5). While G is true a climb word climbs at ``γ_GA = min(3°, max(1.885°, γ_T))``:
γ_T the steady climb angle at the thrust limit and the present airspeed, ``sin γ_T = (T_max − D) / (m g)``, D the drag
of the present state at load factor 1 (`go_around_angle_rad`, the dynamics' own polar). 1.885° is the minimum gradient
of a missed approach, 200 ft per NM (AIM 5-4-21 b); 3° the upper limit. "Go-around" itself changes no target (D27):
with rules 5 and 6 "no level-off" is never in force while G is true, so a go-around climbs only toward a level an
altitude word said; the law refuses "no level-off" with G.
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, FlightCondition, flight_aerodynamics, isa_density
from geokit import FT_M, NM_M

from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.inverse import LOAD_FACTOR_MIN
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.words import ANGLE_LEVEL, Words
from ts_transformer.outputs.envelope import MAX_THRUST_FRACTION

#: The published minimum missed-approach climb gradient, 200 ft per NM (AIM 5-4-21 b, repo docs/literature/go_around) —
#: a regulation value, not a measurement: the go-around angle's floor (module docstring).
GO_AROUND_MIN_RAD = math.atan(200.0 * FT_M / NM_M)
#: The go-around angle's upper limit (D28, the user 2026-10-03).
GO_AROUND_MAX_RAD = math.radians(3.0)


def go_around_angle_rad(state: Kinematics, aero_params: torch.Tensor, max_thrust_n: torch.Tensor) -> torch.Tensor:
    """γ_GA, rad (module docstring): the steady climb angle at the thrust limit and the present airspeed, load factor 1,
    within [1.885°, 3°]."""
    condition = FlightCondition(speed_mps=state.speed_mps, sin_gamma=torch.sin(state.gamma_rad),
                                mass_kg=state.mass_kg, density=isa_density(state.height_m))
    drag = flight_aerodynamics(condition, torch.ones_like(state.speed_mps), aero_params).drag_n
    sine = ((MAX_THRUST_FRACTION * max_thrust_n - drag) / (state.mass_kg * GRAVITY_MPS2)).clamp(-1.0, 1.0)
    return torch.asin(sine).clamp(GO_AROUND_MIN_RAD, GO_AROUND_MAX_RAD)


def stop_angle_rad(left_m: torch.Tensor, speed_mps: torch.Tensor, gamma_rad: torch.Tensor, climbing: torch.Tensor,
                   bank_mean_rad: torch.Tensor, observed_load: torch.Tensor, ceiling: torch.Tensor,
                   load_time_constant_s: float, cycle_s: float) -> torch.Tensor:
    """``γ_stop`` (module docstring, THE STOP): the largest path angle (a magnitude) from which the aircraft stops within
    ``left_m`` (≥ 0) at the physical limit — pulling to ``ceiling`` out of a descent, pushing to ``n_min`` out of a climb
    (``climbing``); 0 where the limit cannot change the path angle that way at all."""
    cos_bank = torch.cos(bank_mean_rad)
    # the load factor that holds the path angle, where the stop's rate is least (module docstring): a pull's at γ = 0
    level = torch.where(climbing, torch.cos(gamma_rad), torch.ones_like(gamma_rad)) / cos_bank
    limit = torch.where(climbing, torch.full_like(ceiling, LOAD_FACTOR_MIN), ceiling)
    reach = torch.where(climbing, level - limit, limit - level)    # > 0: the limit stops the move
    rate = GRAVITY_MPS2 * reach * cos_bank / speed_mps             # a, rad/s
    from_now = torch.where(climbing, observed_load - limit, limit - observed_load)
    delay = (load_time_constant_s * from_now / reach).clamp(min=0.0) + cycle_s
    a_term, b_term = speed_mps * delay, speed_mps / rate
    radius = torch.hypot(a_term, b_term)
    # left ≥ 0 keeps the asin's argument at or above −B / R, its angle at or above 0
    angle = torch.atan2(b_term, a_term) + torch.asin(((left_m - b_term) / radius).clamp(max=1.0))
    return torch.where(reach > 0.0, angle, torch.zeros_like(angle))


class Vertical:
    """The batch's vertical state (which target it has captured, the words it flies and when it heard them) and law."""

    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("captured", "issued", "heard_s")

    def __init__(self, batch: int, params: ExecutorParams, words: Words, device: torch.device) -> None:
        self.params, self.words = params, words
        self.lead_s = words.spec.angle_lead_s
        self.climb_rad = math.radians(words.spec.climb_angle_centre_deg)
        self.captured = torch.zeros(batch, dtype=torch.bool, device=device)
        self.issued = torch.full((batch, 2), -1, dtype=torch.long, device=device)
        self.heard_s = torch.zeros(batch, dtype=torch.float64, device=device)

    def rate(self, state: Kinematics, level_m: torch.Tensor, elevation_m: torch.Tensor, no_level_off: torch.Tensor,
             angle_class: torch.Tensor, angle_deg: torch.Tensor, issued: torch.Tensor, go_around: torch.Tensor,
             aero_params: torch.Tensor, max_thrust_n: torch.Tensor, time_s: float | torch.Tensor,
             bank_mean_rad: torch.Tensor, observed_load: torch.Tensor, ceiling: torch.Tensor
             ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The path-angle rate for this cycle and the law's mode. ``level_m`` the level T in force above the airport
        elevation ``elevation_m`` (E, D58); ``issued`` ``[B, 2]`` the steps the altitude and angle words in force were
        written at (a new one starts the ramp's clock and releases a captured target); ``go_around`` the go-around state
        G; ``aero_params`` / ``max_thrust_n`` the airframes (the go-around angle); ``time_s`` the time flown;
        ``bank_mean_rad`` the cycle's mean bank (`inverse.bank`), ``observed_load`` the observer's load factor and
        ``ceiling`` ``n_up`` (the stop)."""
        words, params, cycle = self.words, self.params, self.params.cycle_s
        descent = (angle_class >= ANGLE_LEVEL + 1) & (angle_class <= words.n_descent)
        if (no_level_off & ~descent).any():
            raise ValueError("\"no level-off\" in force without a descent class (vocabulary §3.7, rule 4)")
        if (no_level_off & go_around).any():
            raise ValueError("\"no level-off\" in force while a go-around is (vocabulary §3.7, rules 5 and 6)")
        new_word = (issued != self.issued).any(dim=1)
        self.captured = self.captured & ~new_word
        self.heard_s = torch.where(new_word, time_s, self.heard_s)
        self.issued = issued.clone()

        gamma, speed = state.gamma_rad, state.speed_mps
        climb = angle_class == words.angle_climb
        magnitude = torch.where(climb, torch.where(go_around, go_around_angle_rad(state, aero_params, max_thrust_n),
                                                   torch.full_like(gamma, self.climb_rad)),
                                torch.deg2rad(angle_deg).abs())
        level_word = angle_class == ANGLE_LEVEL
        above = state.height_m - (level_m + elevation_m)                # NaN under "no level-off": never read then
        toward = torch.where(no_level_off, -torch.ones_like(gamma), -torch.sign(above))
        goal = torch.where(level_word, torch.zeros_like(gamma), toward * magnitude)
        to_go = (self.heard_s + self.lead_s - time_s).clamp(min=cycle)
        ramp = gamma + (goal - gamma) * cycle / to_go

        # the stop toward T: the height left at the cycle's end, the cycle flown at the present path angle (as the
        # stopping bank reads the present turn, §6.3)
        left = (above.abs() + torch.sign(above) * speed * cycle * torch.sin(gamma)).clamp(min=0.0)
        stop = stop_angle_rad(left, speed, gamma, toward > 0.0, bank_mean_rad, observed_load, ceiling,
                              params.load_time_constant_s, cycle)
        aimed = toward * ramp > 0.0                                     # the ramp moves toward T
        target_level = ~no_level_off & ~level_word
        reached = target_level & ~self.captured & aimed & (stop < ramp.abs())
        self.captured = self.captured | reached | (~no_level_off & level_word)
        held = toward * torch.minimum(magnitude, stop)
        moving = torch.where(aimed, toward * torch.minimum(ramp.abs(), stop), ramp)
        target = torch.where(no_level_off, ramp,
                             torch.where(level_word, torch.zeros_like(gamma),
                                         torch.where(self.captured, held, moving)))
        return (target - gamma) / cycle, {"level_captured": self.captured.clone()}
