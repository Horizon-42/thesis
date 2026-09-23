"""The vertical law (executor design §5): the altitude word and the angle word, as a path-angle rate.

The inner loop (§5.1) turns a reference path angle into the rate the inverse flies,
``γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max)``; the outer modes (§5.2) choose ``γ_ref`` (climbing positive):

| words in force            | γ_ref                                                     |
|---------------------------|-----------------------------------------------------------|
| target T + level          | the hold law ``−(h − T) / (V τ_h)``, ``τ_h = 4 τ_γ`` (§5.4)  |
| target T + descent k      | ``−γ_k`` until ``h − T ≤ V γ_k² / (2 γ̇_max)``, then hold (§5.3) |
| target T + climb          | ``+γ_climb`` until ``T − h ≤ V γ_climb² / (2 γ̇_max)``, then hold |
| descend to land + descent k | the angle to the aim point, kept inside class k's range   |

γ_k is the class's nominal angle (the spec's class centre, `Words.angle_deg`). "Descend to land" means land on
the pointed runway, so once the lateral law has captured the line the descent aims, every cycle, at the point
``land_aim_height_m`` above its threshold — the angle from here to there along the centreline, kept inside the
class's own range (`Words.angle_bounds`, never a climb): any angle in that range stays inside the word's tube,
so the aim spends only what the word allows. Before the capture the path still to fly is unknown (a downwind
may run past the threshold before it turns back) and the descent flies the class's nominal angle. On 100 train
flights (20 per airport, own dynamics; 2026-09-24): the nominal angle all the way landed 58 (18 on the ground
before the threshold, 21 across it too high — a class is a degree wide, and the time-indexed words find the
executor elsewhere than the observed aircraft); aiming before the capture too, along the centreline plus the
offset, 80; the class's shallowest angle before the capture, 73; the nominal before and the aim after, 83. The
aim height is data: the median height at which the train flights cross the threshold, extrapolated along their
last rows (20.8 m; IQR 16–26 m). ``γ̇_max`` is ``path_rate_factor``
times the least rate that keeps an entry into the steepest class inside its tube (§5.1:
``V γ_lo² / (2 ε)``, γ_lo the steepest class's lower edge, ε the altitude tolerance). Once a target is
captured the flight holds it until a new altitude or angle word arrives, so the mode does not chatter
at the capture height. The hold law's reference is kept inside the vocabulary's own nominal angles —
the steepest descent class's and the climb's — the ways the words know to change height.
"""

from __future__ import annotations

import math

import torch

from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.words import ANGLE_LEVEL, Words


class Vertical:
    """The batch's vertical state (which target it has captured) and law."""

    def __init__(self, batch: int, params: ExecutorParams, words: Words, device: torch.device) -> None:
        spec = words.spec
        self.params, self.words = params, words
        self.tolerance_m = spec.altitude_tolerance_m
        self.steepest_low_rad = math.radians(spec.descent_angle_edges_deg[-2])
        self.descent_max_rad = math.radians(max(spec.descent_angle_centres_deg))
        self.climb_rad = math.radians(spec.climb_angle_centre_deg)
        self.captured = torch.zeros(batch, dtype=torch.bool, device=device)
        self.issued = torch.full((batch, 2), -1, dtype=torch.long, device=device)
        bounds = [words.angle_bounds(index) for index in range(words.n_descent + 2)]
        self.low_rad = torch.tensor([math.radians(low) for low, _ in bounds], dtype=torch.float64, device=device)
        self.steep_rad = torch.tensor([math.radians(steep) for _, steep in bounds], dtype=torch.float64, device=device)

    def rate_limit(self, state: Kinematics) -> torch.Tensor:
        """γ̇_max, rad/s, at each flight's airspeed."""
        return self.params.path_rate_factor * state.speed_mps * self.steepest_low_rad ** 2 / (2.0 * self.tolerance_m)

    def rate(self, state: Kinematics, altitude_m: torch.Tensor, land: torch.Tensor, angle_class: torch.Tensor,
             angle_deg: torch.Tensor, issued: torch.Tensor, to_go_m: torch.Tensor, threshold_elevation_m: torch.Tensor,
             line_captured: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The path-angle rate for this cycle; ``issued`` is ``[B, 2]``, the steps the altitude and angle
        words in force were written at (a new word releases a captured target); ``to_go_m`` is the distance
        along the centreline to the pointed threshold, whose elevation is ``threshold_elevation_m``, and
        ``line_captured`` whether the lateral law has captured that centreline."""
        if (land & (angle_class == ANGLE_LEVEL)).any():
            raise ValueError("\"descend to land\" in force with the level class (vocabulary §2.5, rule 3)")
        new_word = (issued != self.issued).any(dim=1)
        self.captured = self.captured & ~new_word
        self.issued = issued.clone()

        params = self.params
        rate_max = self.rate_limit(state)
        nominal = torch.deg2rad(angle_deg)                          # descending positive; a climb negative
        height_to_go = state.height_m - altitude_m                  # NaN where "land": never compared then
        level_off = state.speed_mps * nominal.square() / (2.0 * rate_max)
        moving = ~land & (angle_class != ANGLE_LEVEL) & ~self.captured
        reached = moving & (torch.where(nominal > 0.0, height_to_go, -height_to_go) <= level_off)
        self.captured = self.captured | reached | (~land & (angle_class == ANGLE_LEVEL))

        hold_tau = 4.0 * params.path_time_constant_s
        hold = (-height_to_go / (state.speed_mps * hold_tau)).clamp(-self.descent_max_rad, self.climb_rad)
        above_aim = state.height_m - threshold_elevation_m - params.land_aim_height_m
        low = self.low_rad[angle_class].clamp(min=0.0)
        aim = torch.atan2(above_aim, to_go_m.clamp(min=1.0))
        aim = torch.where(line_captured, torch.minimum(torch.maximum(aim, low), self.steep_rad[angle_class]), nominal)
        reference = torch.where(land, -aim, torch.where(self.captured, hold, -nominal))
        gamma_rate = ((reference - state.gamma_rad) / params.path_time_constant_s).clamp(-rate_max, rate_max)
        return gamma_rate, {"level_captured": self.captured.clone(), "path_rate_limited": gamma_rate.abs() >= rate_max}
