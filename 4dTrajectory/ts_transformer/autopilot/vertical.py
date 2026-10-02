"""The vertical law (executor design §5): the altitude word and the angle word, as a path-angle rate.

The inner loop (§5.1) turns a reference path angle into the rate the inverse flies,
``γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max)``; the outer modes (§5.2) choose ``γ_ref`` (climbing positive):

| words in force            | γ_ref                                                     |
|---------------------------|-----------------------------------------------------------|
| target T + level          | the hold law ``−(h − T) / (V τ_h)``, ``τ_h = 4 τ_γ`` (§5.4)  |
| target T + descent k      | ``−γ_k`` until ``h − T ≤ V γ_k² / (2 γ̇_max)``, then hold (§5.3) |
| target T + climb          | ``+γ_climb`` until ``T − h ≤ V γ_climb² / (2 γ̇_max)``, then hold |
| descend to land + descent k | toward a crossing point (below), inside the tube in force while the landing can still be reached from it (between level and the steepest class's lower edge); before the capture never steeper than the straight line to that point, after it never steeper than the hold law toward its height or the line to it; never under the glidepath's lower edge before the threshold |
| go-around, descend to land  | ``+γ_climb`` (§4.6: a new altitude word gives the target) |

γ_k is the class's nominal angle (the spec's class centre, `Words.angle_deg`). "Descend to land" means land on
the pointed runway, crossing its threshold at its published threshold crossing height (TCH, `lateral.Runways`). The
crossing point is the TCH moved into the word's tube extended to the threshold (`labeller.vertical.tube_bounds`: the
class's two angles from where the altitude or angle word in force was said, ± the altitude tolerance, along the path
flown; its inner half), within the heights admitted there — the TCH ± the altitude tolerance's inner half, within the
landing condition; where the two do not meet, the admitted edge nearer the tube. The descent aims at that point every
cycle — after the lateral capture the angle from here to there along the centreline (below the pointed runway's
published glidepath, level instead: the approach joined from below, as the observed aircraft fly it — the shallow classes'
readout of 2026-09-27, where aiming at the crossing point from below rode the class's steep edge), before it the class's
nominal angle, never steeper than the straight line there (the shortest path there is, so it never descends into the
ground on a downwind) — kept inside the tube in force (the class's two edges, the hold law's correction back from each) while
the admitted heights can still be reached from the tube, flying between level and the steepest class's lower edge;
past that, straight toward the point (mode ``aim_left_tube``, after the capture: the landing comes first). A tube that will not reach the threshold at the admitted
heights is flown while a later angle word could still bring it there — the judge re-anchors the tube at every angle word
(the review of 2026-09-24: leaving it as soon as the class in force missed the landing failed the land word on flights
whose classes steepened or shallowed toward the runway). After the capture the descent is never steeper than the hold
law toward the crossing height or the line to the point, whichever is steeper, so it cannot pass under that height
before the threshold. And before the threshold, captured or not, it never descends under the pointed runway's published
glidepath less `GLIDEPATH_BELOW_M` — the glidepath's lower edge, the one the post-training check draws
(`prior.procedure`), which binds only inside the FAF and the LPV cone; here it binds at any lateral position, at the
edge's height at the distance to go — by a hold law toward that line as the aircraft closes on it, at most level: under
the edge it descends less steeply than the edge falls, level when more than V τ_h times the edge's slope under it, and
rejoins the edge as the glidepath comes down (the approach joined from below, as flown); a cycle whose aim the edge
set is mode ``glidepath_floor`` (the judge counts it per word, so its cost is readable). A class's tube leaves the height
inside it open ("descend to land" with the shallowest class holds level flight as well as 1.5°), and aiming at the
crossing point from below the glidepath rides the tube's steep edge, far under the glidepath (prior readouts §12; the
floor's effect on the replays, readouts §13). Nothing here is measured from data: the TCH and the glidepath are the
runway's, the edge the procedure's, the rest the vocabulary's. (Stage 2 of the vocabulary-only plan, 500 train flights:
the altitude column's words inside their envelopes 86.9 % → 95.1 % against the law that left the tube as soon as the
class in force missed the landing, every flight landed, no crossing outside evaluation's ±22 m; planning the reach at the
steepest class's open upper edge, 10°, left the tube too late — 19 crossings above TCH + 22 m; aiming at the TCH itself,
94.5 %.) ``γ̇_max`` is ``path_rate_factor`` times the least rate that keeps an entry into the steepest class inside its tube (§5.1:
``V γ_lo² / (2 ε)``, γ_lo the steepest class's lower edge, ε the altitude tolerance). Once a target is
captured the flight holds it until a new altitude or angle word arrives, so the mode does not chatter
at the capture height. The hold law's reference is kept inside the vocabulary's own nominal angles —
the steepest descent class's and the climb's — the ways the words know to change height.

While a go-around is in force every climb — the go-around's own (§4.6, "descend to land" in force) and one toward an
altitude word above — is flown at the published minimum missed-approach climb gradient, `GO_AROUND_CLIMB_GRADIENT`,
not the climb class's centre: the climb class has no angle of its own to say, and its centre (1.32° on the current
spec, measured on arrivals that seldom climb) is under the regulation's floor (multi-aircraft design §6.6 step 8 item 7,
the user's 2026-10-02 decision). Elsewhere the climb class flies its centre as before.
"""

from __future__ import annotations

import math

import torch

from geokit import FT_M, NM_M

from ts_transformer.autopilot.frame import Kinematics
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.words import ANGLE_LEVEL, Words

#: How far under the pointed runway's published glidepath "descend to land" may go before the threshold, m: the
#: glidepath lower edge — MIRROR of `prior.procedure.GLIDEPATH_BELOW_M` (itself the optimizer's; `autopilot` imports no
#: model package), held equal by `tests/test_autopilot.py`.
GLIDEPATH_BELOW_M = 60.0
#: The landing aim keeps this share of the altitude tolerance in hand, in the word's tube and in the heights admitted
#: at the threshold (the tube is judged on the smoothed altitude, and the path-angle loop tracks its reference within
#: metres; at the full tolerance about the TCH, 17 of 500 train crossings fell outside evaluation's ±22 m, none at half).
TUBE_MARGIN_SHARE = 0.5
#: The published minimum missed-approach climb gradient, 200 ft per NM (AIM 5-4-21 b, repo docs/literature/go_around) —
#: a regulation value, not a measurement; climbs during a go-around are flown at its path angle (module docstring).
GO_AROUND_CLIMB_GRADIENT = 200.0 * FT_M / NM_M
GO_AROUND_CLIMB_RAD = math.atan(GO_AROUND_CLIMB_GRADIENT)


class Vertical:
    """The batch's vertical state (which target it has captured) and law."""

    def __init__(self, batch: int, params: ExecutorParams, words: Words, device: torch.device) -> None:
        spec = words.spec
        self.params, self.words = params, words
        self.tolerance_m = spec.altitude_tolerance_m
        self.landing_max_height_m = spec.landing_max_height_m
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

    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("captured", "issued", "left_tube", "flown_m", "anchor_m", "anchor_height_m")

    def rate_limit(self, state: Kinematics) -> torch.Tensor:
        """γ̇_max, rad/s, at each flight's airspeed."""
        return self.params.path_rate_factor * state.speed_mps * self.steepest_low_rad ** 2 / (2.0 * self.tolerance_m)

    def rate(self, state: Kinematics, altitude_m: torch.Tensor, land: torch.Tensor, angle_class: torch.Tensor,
             angle_deg: torch.Tensor, issued: torch.Tensor, to_go_m: torch.Tensor, threshold_elevation_m: torch.Tensor,
             crossing_height_m: torch.Tensor, glidepath_tan: torch.Tensor, off_course_deg: torch.Tensor,
             straight_m: torch.Tensor, line_captured: torch.Tensor, go_around: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """The path-angle rate for this cycle, the rate the law wanted before its own limit ``γ̇_max``, and
        its modes; ``issued`` is ``[B, 2]``, the steps the altitude and angle words in force were written at
        (a new word releases a captured target); ``to_go_m`` is the distance along the centreline to the
        pointed threshold, whose elevation is ``threshold_elevation_m``, published crossing height
        ``crossing_height_m`` and published glidepath's tangent ``glidepath_tan``, ``off_course_deg`` the track
        less its course, ``straight_m`` the straight-line distance to it, ``line_captured`` whether the
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
        # a go-around climbs at the published gradient (module docstring)
        climb_class = angle_class == self.words.angle_climb
        nominal = torch.where(go_around & climb_class, torch.full_like(nominal, -GO_AROUND_CLIMB_RAD), nominal)
        climb_rad = torch.where(go_around, torch.full_like(nominal, GO_AROUND_CLIMB_RAD),
                                torch.full_like(nominal, self.climb_rad))
        height_to_go = state.height_m - altitude_m                  # NaN where "land": never compared then
        level_off = state.speed_mps * nominal.square() / (2.0 * rate_max)
        moving = ~land & (angle_class != ANGLE_LEVEL) & ~self.captured
        reached = moving & (torch.where(nominal > 0.0, height_to_go, -height_to_go) <= level_off)
        self.captured = self.captured | reached | (~land & (angle_class == ANGLE_LEVEL))

        hold_tau = 4.0 * params.path_time_constant_s
        hold = torch.minimum((-height_to_go / (state.speed_mps * hold_tau)).clamp(min=-self.descent_max_rad), climb_rad)
        # "descend to land" (module docstring): the crossing point, the published TCH moved into the tube in force
        # extended to the threshold, within the heights admitted there (the TCH ± the altitude tolerance's inner half,
        # within the landing condition), or the admitted edge nearer that tube where the two do not meet
        margin = TUBE_MARGIN_SHARE * self.tolerance_m
        steep_tan, shallow_tan = self.steep_tan[angle_class], self.shallow_tan[angle_class]
        height = state.height_m - threshold_elevation_m
        anchor = self.anchor_height_m - threshold_elevation_m
        here, along = self.flown_m - self.anchor_m, self.flown_m - self.anchor_m + to_go_m
        admitted_low = (crossing_height_m - self.tolerance_m + margin).clamp(min=0.0)
        admitted_high = (crossing_height_m + self.tolerance_m - margin).clamp(max=self.landing_max_height_m)
        tube_low = anchor - along * steep_tan - self.tolerance_m + margin
        tube_high = anchor - along * shallow_tan + self.tolerance_m - margin
        low, high = torch.maximum(tube_low, admitted_low), torch.minimum(tube_high, admitted_high)
        nearer_edge = torch.where(tube_low > admitted_high, admitted_high, admitted_low)
        crossing = torch.where(low <= high, torch.minimum(torch.maximum(crossing_height_m, low), high), nearer_edge)
        above_crossing = height - crossing
        on_line = torch.atan2(above_crossing, to_go_m.clamp(min=1.0)).clamp(0.0, self.steepest_rad)
        shortest = torch.atan2(above_crossing, straight_m.clamp(min=1.0)).clamp(min=0.0)
        toward = torch.where(line_captured, on_line, torch.minimum(nominal, shortest))
        # inside the tube in force (its inner half; the hold law's correction back from each edge) while the admitted
        # heights can still be reached from it between level and the steepest class's lower edge; past that, straight
        # toward them
        upper = anchor - here * shallow_tan + self.tolerance_m - margin
        lower = anchor - here * steep_tan - self.tolerance_m + margin
        speed_tau = state.speed_mps * hold_tau
        # after the capture, below the pointed runway's published glidepath: the least descent the tube allows (level where
        # the class admits it) — the approach joined from below, as flown (the shallow classes' readout, 2026-09-27);
        # on or above it, toward the crossing point
        below_glidepath = line_captured & (height < crossing_height_m + to_go_m * glidepath_tan)
        in_tube = torch.minimum(torch.maximum(torch.where(below_glidepath, torch.zeros_like(toward), toward),
                                              torch.atan(shallow_tan) + (height - upper) / speed_tau),
                                torch.atan(steep_tan) + (height - lower) / speed_tau)
        in_reach = (lower - to_go_m * math.tan(self.steepest_low_rad) <= admitted_high) & (upper >= admitted_low)
        self.left_tube = land & ~go_around & line_captured & ~in_reach
        # never under the crossing height before the threshold: after the capture no steeper than the hold law toward
        # it or the line to it, whichever is steeper; before, than the straight line to it (never into the ground)
        crossing_floor = torch.where(line_captured, torch.maximum(above_crossing / speed_tau, on_line), shortest)
        # never under the published glidepath's lower edge before the threshold, captured or not, at the edge's height
        # at the distance to go: a hold law toward that line as the aircraft closes on it (the line falls at the
        # glidepath's slope times the share of the speed along the course; flying away from the runway it rises) — the
        # aim's own clamp below keeps the aircraft level under it until the glidepath comes down
        glidepath = crossing_height_m + to_go_m * glidepath_tan - GLIDEPATH_BELOW_M
        closing_tan = glidepath_tan * torch.cos(torch.deg2rad(off_course_deg))
        on_glidepath = torch.atan(closing_tan) + (height - glidepath) / speed_tau
        floor = torch.where(to_go_m > 0.0, torch.minimum(crossing_floor, on_glidepath), crossing_floor)
        wanted_aim = torch.where(in_reach, in_tube, toward)
        aim = torch.minimum(wanted_aim, floor).clamp(0.0, self.steepest_rad)
        # the glidepath's lower edge set the aim this cycle: it held a "descend to land" above where the word would fly
        glidepath_floor = land & ~go_around & (aim < torch.minimum(wanted_aim, crossing_floor).clamp(0.0, self.steepest_rad))
        reference = torch.where(land, torch.where(go_around, climb_rad, -aim),
                                torch.where(self.captured, hold, -nominal))
        wanted = (reference - state.gamma_rad) / params.path_time_constant_s
        gamma_rate = wanted.clamp(-rate_max, rate_max)
        self.flown_m = self.flown_m + state.ground_speed_mps * params.cycle_s
        return gamma_rate, wanted, {"level_captured": self.captured.clone(), "aim_left_tube": self.left_tube.clone(),
                                    "glidepath_floor": glidepath_floor, "path_rate_limited": wanted.abs() > rate_max}
