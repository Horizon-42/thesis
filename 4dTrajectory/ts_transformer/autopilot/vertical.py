"""The vertical law (executor design §5): the altitude word and the angle word, as a path-angle rate.

The inner loop (§5.1) turns a reference path angle into the rate the inverse flies,
``γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max)``; the outer modes (§5.2) choose ``γ_ref`` (climbing positive):

| words in force            | γ_ref                                                     |
|---------------------------|-----------------------------------------------------------|
| target T + level          | the hold law ``−(h − T) / (V τ_h)``, ``τ_h = 4 τ_γ`` (§5.4)  |
| target T + descent k      | ``−γ_k`` until ``h − T ≤ V γ_k² / (2 γ̇_max)``, then hold (§5.3) |
| target T + climb          | ``+γ_climb`` until ``T − h ≤ V γ_climb² / (2 γ̇_max)``, then hold |
| descend to land + descent k | before the capture: class k's nominal angle, never steeper than the straight line to the aim point; after it: the angle to a crossing point inside the word's tube and the landing window, from level to the steepest class |
| go-around, descend to land  | ``+γ_climb`` (§4.6: a new altitude word gives the target) |

γ_k is the class's nominal angle (the spec's class centre, `Words.angle_deg`). "Descend to land" means land on
the pointed runway, so once the lateral law has captured the line the descent aims, every cycle, at the point
``land_aim_height_m`` above its threshold — the angle from here to there along the centreline, from level to the
steepest class's steep edge: an aircraft below the path levels off and meets it, one above it descends as steep
as the vocabulary goes. A class is a degree wide, and an executor that flew its words at its own pace reaches the
line higher or lower than the observed aircraft did; held to class k's own range it crossed the threshold up to
280 m high or touched down short (executor design §5.2 has the numbers). The point it aims at stays inside the
word's tube when it can: the tube (`labeller.vertical.tube_bounds`: the class's two angles from where the altitude
or angle word in force was said, ± the altitude tolerance, along the path flown) projected to the threshold, its
inner half, meets the window of heights the observed flights cross at (``land_window_*``); the aim height is
moved into that overlap; when there is none the aim leaves the tube, to the window's nearer edge (mode
``aim_left_tube``). The landing that counts is the window's: kept on the tube's nearest edge up to the landing
condition's 100 m instead, 2,000 train flights crossed high enough to fail evaluation's glidepath gate (paired pass
96.5 → 94.1 %) and flew no more words inside their envelopes (2026-09-24). Before the capture the path still to
fly is unknown (a downwind may run past the threshold before it turns back) and the descent flies the class's
nominal angle — but never steeper than the straight line to the aim point, the shortest path there is, so it
never descends into the ground on the way. The aim height is data: the median height at which the train flights
cross the threshold, extrapolated along their last rows (20.8 m; IQR 16–26 m). ``γ̇_max`` is ``path_rate_factor``
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

#: The landing aim keeps this share of the word's tube tolerance in hand (the tube is judged on the smoothed
#: altitude, and the path-angle loop tracks its reference within metres).
TUBE_MARGIN_SHARE = 0.5


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
        self.steepest_rad = math.radians(words.angle_bounds(words.n_descent)[1])
        self.left_tube = torch.zeros(batch, dtype=torch.bool, device=device)
        bounds = [words.angle_bounds(index) for index in range(words.n_descent + 2)]
        self.shallow_tan = torch.tensor([math.tan(math.radians(low)) for low, _ in bounds], dtype=torch.float64,
                                        device=device)
        self.steep_tan = torch.tensor([math.tan(math.radians(steep)) for _, steep in bounds], dtype=torch.float64,
                                      device=device)
        # the path flown, and where the tube in force was anchored (the altitude or angle word said last)
        self.flown_m = torch.zeros(batch, dtype=torch.float64, device=device)
        self.anchor_m = torch.zeros(batch, dtype=torch.float64, device=device)
        self.anchor_height_m = torch.full((batch,), math.nan, dtype=torch.float64, device=device)

    def rate_limit(self, state: Kinematics) -> torch.Tensor:
        """γ̇_max, rad/s, at each flight's airspeed."""
        return self.params.path_rate_factor * state.speed_mps * self.steepest_low_rad ** 2 / (2.0 * self.tolerance_m)

    def rate(self, state: Kinematics, altitude_m: torch.Tensor, land: torch.Tensor, angle_class: torch.Tensor,
             angle_deg: torch.Tensor, issued: torch.Tensor, to_go_m: torch.Tensor, threshold_elevation_m: torch.Tensor,
             straight_m: torch.Tensor, line_captured: torch.Tensor,
             go_around: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The path-angle rate for this cycle, the rate the law wanted before its own limit ``γ̇_max``, and
        its modes; ``issued`` is ``[B, 2]``, the steps the altitude and angle words in force were written at
        (a new word releases a captured target); ``to_go_m`` is the distance along the centreline to the
        pointed threshold, whose elevation is ``threshold_elevation_m``, ``straight_m`` the straight-line distance
        to it, ``line_captured`` whether the
        lateral law has captured that centreline, and ``go_around`` whether a go-around is in force (§4.6:
        with a target it is climbed to — the hold law climbs at most at the climb class — and with "descend
        to land" in force the flight climbs at the climb class until an altitude word gives it one)."""
        descent = (angle_class >= ANGLE_LEVEL + 1) & (angle_class <= self.words.n_descent)
        if (land & ~descent).any():
            raise ValueError("\"descend to land\" in force without a descent class (vocabulary §2.5, rule 3)")
        new_word = (issued != self.issued).any(dim=1)
        self.captured = self.captured & ~new_word
        self.issued = issued.clone()
        anchored = new_word | self.anchor_height_m.isnan()
        self.anchor_m = torch.where(anchored, self.flown_m, self.anchor_m)
        self.anchor_height_m = torch.where(anchored, state.height_m, self.anchor_height_m)

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
        # the word's tube at the threshold, its inner half, above the threshold
        along = self.flown_m - self.anchor_m + to_go_m
        margin = TUBE_MARGIN_SHARE * self.tolerance_m
        tube_low = (self.anchor_height_m - along * self.steep_tan[angle_class] - self.tolerance_m + margin
                    - threshold_elevation_m)
        tube_high = (self.anchor_height_m - along * self.shallow_tan[angle_class] + self.tolerance_m - margin
                     - threshold_elevation_m)
        low, high = tube_low.clamp(min=params.land_window_low_m), tube_high.clamp(max=params.land_window_high_m)
        in_both = torch.minimum(torch.maximum(torch.full_like(low, params.land_aim_height_m), low), high)
        above = tube_low > params.land_window_high_m                   # the tube passes above the window
        window_edge = torch.where(above, torch.full_like(low, params.land_window_high_m),
                                  torch.full_like(low, params.land_window_low_m))
        crossing = torch.where(low <= high, in_both, window_edge)
        self.left_tube = land & line_captured & (low > high)
        on_line = torch.atan2(state.height_m - threshold_elevation_m - crossing, to_go_m.clamp(min=1.0)).clamp(
            0.0, self.steepest_rad)
        shortest = torch.atan2(above_aim, straight_m.clamp(min=1.0)).clamp(min=0.0)
        aim = torch.where(line_captured, on_line, torch.minimum(nominal, shortest))
        reference = torch.where(land, torch.where(go_around, torch.full_like(aim, self.climb_rad), -aim),
                                torch.where(self.captured, hold, -nominal))
        wanted = (reference - state.gamma_rad) / params.path_time_constant_s
        gamma_rate = wanted.clamp(-rate_max, rate_max)
        self.flown_m = self.flown_m + state.ground_speed_mps * params.cycle_s
        return gamma_rate, wanted, {"level_captured": self.captured.clone(), "aim_left_tube": self.left_tube.clone(),
                                    "path_rate_limited": wanted.abs() > rate_max}
