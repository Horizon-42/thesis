"""The vertical law (design §5.5; executor design §5.1, §5.4): the altitude word and the angle word, as a path-angle
rate. No landing aim, no glidepath floor (D3, D9): "no level-off" flies its class's angle and the words decide the
profile.

The inner loop (executor design §5.1) turns a reference path angle into the rate the inverse flies,
``γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max)``; the modes choose ``γ_ref`` (climbing positive):

| words in force                         | γ_ref                                                                  |
|----------------------------------------|------------------------------------------------------------------------|
| level T + level                        | the hold law ``−(h − T) / (V τ_h)``, ``τ_h = 4 τ_γ``                   |
| level T + descent k                    | ``−γ_k`` until ``h − T ≤ V γ_k² / (2 γ̇_max)``, then hold                |
| level T + climb                        | ``+γ_climb`` until ``T − h ≤ V γ_climb² / (2 γ̇_max)``, then hold         |
| "no level-off" + descent k             | ``−γ_k``, no level-off                                                 |
| go-around, "no level-off" or a higher T | ``+γ_go-around`` (200 ft per NM, `GO_AROUND_CLIMB_GRADIENT`): with "no level-off" until a new altitude word; with a higher T to T, then hold |

γ_k is the class's nominal angle (the spec's class centre, `Words.angle_deg`). Once a target is captured the flight
holds it until a new altitude or angle word arrives, so the mode does not chatter at the capture height. The hold
law's reference is kept inside the vocabulary's own nominal angles — the steepest descent class's and the climb's (the
go-around's, while one is in force) — the ways the words know to change height. ``γ̇_max`` is ``path_rate_factor``
times the least rate that keeps an entry into the steepest class inside its tube (``V γ_lo² / (2 ε)``, γ_lo the steepest
class's lower edge, ε the narrowest level band of the grid, `Words.altitude_tolerances`).

While a go-around is in force (design §3.2, D10) every climb is flown at the published minimum missed-approach climb
gradient, not the climb class's centre: no other word gives it, and the climb class's centre (1.32° measured on
arrivals that seldom climb) is under the regulation's floor. "No level-off" in force when the go-around is said is
replaced by that climb (rule 5 forbids saying it while the go-around is in force).
"""

from __future__ import annotations

import math

import torch

from geokit import FT_M, NM_M

from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.words import ANGLE_LEVEL, Words

#: The published minimum missed-approach climb gradient, 200 ft per NM (AIM 5-4-21 b, repo docs/literature/go_around) —
#: a regulation value, not a measurement; climbs during a go-around are flown at its path angle (module docstring).
GO_AROUND_CLIMB_GRADIENT = 200.0 * FT_M / NM_M
GO_AROUND_CLIMB_RAD = math.atan(GO_AROUND_CLIMB_GRADIENT)


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

    def go_around_climb_s(self, height_m: float, gamma_rad: float, speed_mps: float, target_m: float) -> float:
        """How long this law's own go-around climb — "no level-off" in force under a go-around and nothing else said:
        the reference `GO_AROUND_CLIMB_RAD`, followed at ``τ_γ`` within ``±γ̇_max`` — takes to bring a flight at
        ``height_m`` with path angle ``gamma_rad`` (climbing positive) and airspeed ``speed_mps`` (held: a go-around
        holds the speed) up to ``target_m``; 0 at or above it. The vertical law alone, cycle by cycle, the height over a
        cycle at its mean path angle: a go-around reward's climb reference (multi-aircraft design §6.6 step 8 item 9)."""
        if not (speed_mps > 0.0 and all(math.isfinite(x) for x in (height_m, gamma_rad, speed_mps, target_m))):
            raise ValueError(f"a go-around climb from height {height_m}, path angle {gamma_rad}, airspeed {speed_mps} to "
                             f"{target_m}: a finite state flying forward")
        rate_max, dt, tau = float(self.rate_max(speed_mps)), self.params.cycle_s, self.params.path_time_constant_s
        seconds = 0.0
        while height_m < target_m:
            gamma_rate = min(rate_max, max(-rate_max, (GO_AROUND_CLIMB_RAD - gamma_rad) / tau))
            height_m += speed_mps * math.sin(gamma_rad + 0.5 * gamma_rate * dt) * dt
            gamma_rad += gamma_rate * dt
            seconds += dt
        return seconds

    def rate(self, state: Kinematics, altitude_m: torch.Tensor, no_level_off: torch.Tensor, angle_class: torch.Tensor,
             angle_deg: torch.Tensor, issued: torch.Tensor, go_around: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The path-angle rate for this cycle, the rate the law wanted before its own limit ``γ̇_max``, and its modes;
        ``issued`` is ``[B, 2]``, the steps the altitude and angle words in force were written at (a new word releases
        a captured target); ``go_around`` whether a go-around is in force."""
        descent = (angle_class >= ANGLE_LEVEL + 1) & (angle_class <= self.words.n_descent)
        if (no_level_off & ~descent).any():
            raise ValueError("\"no level-off\" in force without a descent class (design §3.7, rule 4)")
        new_word = (issued != self.issued).any(dim=1)
        self.captured = self.captured & ~new_word
        self.issued = issued.clone()

        params = self.params
        rate_max = self.rate_limit(state)
        nominal = torch.deg2rad(angle_deg)                          # descending positive; a climb negative
        # a go-around climbs at the published gradient (module docstring)
        climb_class = angle_class == self.words.angle_climb
        nominal = torch.where(go_around & climb_class, torch.full_like(nominal, -GO_AROUND_CLIMB_RAD), nominal)
        climb_rad = torch.where(go_around, torch.full_like(nominal, GO_AROUND_CLIMB_RAD),
                                torch.full_like(nominal, self.climb_rad))
        height_to_go = state.height_m - altitude_m                  # NaN under "no level-off": never compared then
        level_off = state.speed_mps * nominal.square() / (2.0 * rate_max)
        moving = ~no_level_off & (angle_class != ANGLE_LEVEL) & ~self.captured
        reached = moving & (torch.where(nominal > 0.0, height_to_go, -height_to_go) <= level_off)
        self.captured = self.captured | reached | (~no_level_off & (angle_class == ANGLE_LEVEL))

        hold_tau = 4.0 * params.path_time_constant_s
        hold = torch.minimum((-height_to_go / (state.speed_mps * hold_tau)).clamp(min=-self.descent_max_rad), climb_rad)
        reference = torch.where(no_level_off, torch.where(go_around, climb_rad, -nominal),
                                torch.where(self.captured, hold, -nominal))
        wanted = (reference - state.gamma_rad) / params.path_time_constant_s
        gamma_rate = wanted.clamp(-rate_max, rate_max)
        return gamma_rate, wanted, {"level_captured": self.captured.clone(), "path_rate_limited": wanted.abs() > rate_max}
