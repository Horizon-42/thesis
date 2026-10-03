"""The vertical law (design §5.5; executor design §5.1, §5.4): the altitude word and the angle word, as a path-angle
rate. No landing aim, no glidepath floor (D3, D9): "no level-off" flies its class's angle and the words decide the
profile.

The inner loop (executor design §5.1) turns a reference path angle into the rate the inverse flies,
``γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max)``; the modes choose ``γ_ref`` (climbing positive):

| words in force                         | γ_ref                                                                  |
|----------------------------------------|------------------------------------------------------------------------|
| level T + level                        | the hold law ``−(h − T) / (V τ_h)``, ``τ_h = 4 τ_γ``                   |
| level T + descent k                    | ``−γ_k`` until ``h − T ≤ V γ_k² / (2 γ̇_max)``, then hold                |
| level T + climb, G false               | ``+γ_climb`` until ``T − h ≤ V γ_climb² / (2 γ̇_max)``, then hold         |
| level T + climb, G true                | ``+γ_GA`` (below), level-off as above                                  |
| "no level-off" + descent k             | ``−γ_k``, no level-off                                                 |

γ_k is the class's nominal angle (the spec's class centre, `Words.angle_deg`); γ_climb the climb class's nominal (the
spec's climb centre, D15, D28). Once a target is captured the flight holds it until a new altitude or angle word
arrives, so the mode does not chatter at the capture height. The hold law's reference is kept inside the angles the
words know to change height with — the steepest descent class's and the climb's (γ_GA while G is true). ``γ̇_max`` is
``path_rate_factor`` times the least rate that keeps an entry into the steepest class inside its tube
(``V γ_lo² / (2 ε)``, γ_lo the steepest class's lower edge, ε the narrowest level band of the grid,
`Words.altitude_tolerances`).

THE GO-AROUND ANGLE (D28, design §5.5). While G is true a climb word climbs at ``γ_GA = min(3°, max(1.885°, γ_T))``:
γ_T the steady climb angle at the thrust limit and the present airspeed, ``sin γ_T = (T_max − D) / (m g)``, D the drag
of the present state at load factor 1 (`go_around_angle_rad`, the dynamics' own polar). 1.885° is the minimum gradient
of a missed approach, 200 ft per NM (AIM 5-4-21 b); 3° the upper limit. Where γ_T is under 1.885° the reference is
1.885° and the inner loop gets what the thrust gives (the thrust limit is recorded). "Go-around" itself changes no
target (D27): with rules 5 and 6 "no level-off" is never in force while G is true, so a go-around climbs only toward a
level an altitude word said; the law refuses "no level-off" with G.
"""

from __future__ import annotations

import math

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2, FlightCondition, flight_aerodynamics, isa_density
from geokit import FT_M, NM_M

from ts_transformer.autopilot.frame import Kinematics
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


class Vertical:
    """The batch's vertical state (which target it has captured) and law."""

    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("captured", "issued")

    def __init__(self, batch: int, params: ExecutorParams, words: Words, device: torch.device) -> None:
        spec = words.spec
        self.params, self.words = params, words
        self.tolerance_m = float(words.altitude_tolerances.min())
        self.steepest_low_rad = math.radians(spec.descent_angle_edges_deg[-2])
        self.descent_max_rad = math.radians(max(spec.descent_angle_centres_deg))
        self.climb_rad = math.radians(spec.climb_angle_centre_deg)
        self.captured = torch.zeros(batch, dtype=torch.bool, device=device)
        self.issued = torch.full((batch, 2), -1, dtype=torch.long, device=device)

    def rate_limit(self, state: Kinematics) -> torch.Tensor:
        """γ̇_max, rad/s, at each flight's airspeed."""
        return self.rate_max(state.speed_mps)

    def rate_max(self, speed_mps: torch.Tensor | float) -> torch.Tensor | float:
        """γ̇_max, rad/s, at airspeed ``speed_mps`` (module docstring)."""
        return self.params.path_rate_factor * speed_mps * self.steepest_low_rad ** 2 / (2.0 * self.tolerance_m)

    def rate(self, state: Kinematics, altitude_m: torch.Tensor, no_level_off: torch.Tensor, angle_class: torch.Tensor,
             angle_deg: torch.Tensor, issued: torch.Tensor, go_around: torch.Tensor, aero_params: torch.Tensor,
             max_thrust_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The path-angle rate for this cycle, the rate the law wanted before its own limit ``γ̇_max``, and its modes;
        ``issued`` is ``[B, 2]``, the steps the altitude and angle words in force were written at (a new word releases
        a captured target); ``go_around`` the go-around state G; ``aero_params`` / ``max_thrust_n`` the airframes (the
        go-around angle, `go_around_angle_rad`)."""
        descent = (angle_class >= ANGLE_LEVEL + 1) & (angle_class <= self.words.n_descent)
        if (no_level_off & ~descent).any():
            raise ValueError("\"no level-off\" in force without a descent class (design §3.7, rule 4)")
        if (no_level_off & go_around).any():
            raise ValueError("\"no level-off\" in force while a go-around is (design §3.7, rules 5 and 6)")
        new_word = (issued != self.issued).any(dim=1)
        self.captured = self.captured & ~new_word
        self.issued = issued.clone()

        params = self.params
        rate_max = self.rate_limit(state)
        # the climb's angle: the class's nominal, or the go-around angle while G is true (module docstring)
        go_around_rad = go_around_angle_rad(state, aero_params, max_thrust_n)
        climb_rad = torch.where(go_around, go_around_rad, torch.full_like(go_around_rad, self.climb_rad))
        nominal = torch.where(go_around & (angle_class == self.words.angle_climb), -go_around_rad,
                              torch.deg2rad(angle_deg))             # descending positive; a climb negative
        height_to_go = state.height_m - altitude_m                  # NaN under "no level-off": never compared then
        level_off = state.speed_mps * nominal.square() / (2.0 * rate_max)
        moving = ~no_level_off & (angle_class != ANGLE_LEVEL) & ~self.captured
        reached = moving & (torch.where(nominal > 0.0, height_to_go, -height_to_go) <= level_off)
        self.captured = self.captured | reached | (~no_level_off & (angle_class == ANGLE_LEVEL))

        hold_tau = 4.0 * params.path_time_constant_s
        hold = torch.minimum((-height_to_go / (state.speed_mps * hold_tau)).clamp(min=-self.descent_max_rad), climb_rad)
        reference = torch.where(self.captured & ~no_level_off, hold, -nominal)
        wanted = (reference - state.gamma_rad) / params.path_time_constant_s
        gamma_rate = wanted.clamp(-rate_max, rate_max)
        return gamma_rate, wanted, {"level_captured": self.captured.clone(), "path_rate_limited": wanted.abs() > rate_max}
