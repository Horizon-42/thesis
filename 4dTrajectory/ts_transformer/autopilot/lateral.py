"""The lateral law (executor design §4): a heading word, the clearance, the capture and the line.

Every law here returns a compass TRACK RATE for the inverse (`autopilot.inverse.attitude`) to fly:

- a heading word θ (§4.1): ``χ̇* = sat(wrap180(θ − χ) / τ_ψ, ±r_turn)`` — the shorter way at the steady
  rate, easing out over τ_ψ; the bank it takes is the inverse's, capped and rate-limited there;
- cleared, not yet captured (§4.3): when θ itself cannot reach the pointed runway's line but a track
  within the heading tolerance can (the labeller's own test, `instructions.envelope.heading_converges`),
  θ is flown bent by the tolerance toward the line — still inside the word's envelope;
- the capture (§4.4) starts when the turn onto the course would end on the line, flying toward it — or
  at once inside the corridor's width; captured is the executor's state, not a word. The turn is the one
  the laws fly: the bank first rolls to the turn's (``t_roll = |φ_turn − φ| / p``, the aircraft closing on
  the line at ``V sin |Δχ|`` meanwhile, as if half that time straight), then an arc of ``R = V / r_turn``,
  then the roll-out's exponential tail over the last ``r_turn τ_ψ`` degrees (``V r_turn τ_ψ² / 2`` beyond
  the arc's own). So it starts at ``|y| ≤ R (1 − cos |Δχ|) + V sin |Δχ| t_roll / 2 + V r_turn τ_ψ² / 2``,
  every term the executor's own, nothing tuned;
- captured, the aircraft flies the arc that ends tangent to the line from where it is NOW: the rate
  ``V (1 − cos |Δχ|) / |y|`` toward the course, solved again every cycle — so a deceleration inside the
  turn (a shorter radius) or a late roll-in does not leave it short of the line or past it — at most the
  rate the bank cap gives at this speed (``g tan φ_cap / V``: a base close in needs a tighter turn than the
  steady rate, and the bank cap is the word envelope's own bound), and at most ``|Δχ| / τ_ψ``: the heading
  law's own roll-out, which the bank rate can follow
  (§4.1 constraint 3), so the turn does not run past the course while the bank comes back; once its
  track is within the heading tolerance of the course the line's own target takes over
  (§4.5): ``course − sat(k_y · y, intercept angle)``,
  ``k_y = 1 / (4 V τ_ψ)`` (critical damping of ``τ_ψ ÿ + ẏ + V k_y y = 0``: the aircraft does not cross the
  centreline);
- a go-around (§4.6) cancels the clearance and the capture and flies the pointed runway's course.

Positions are the airport frame's (`autopilot.frame`), the runway geometry the artefact's candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from ts_transformer.autopilot.frame import Kinematics, wrap180
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import APPROACH_CLEARED, APPROACH_GO_AROUND


def heading_rate(track_deg: torch.Tensor, target_deg: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """§4.1: the compass track rate, deg/s, that turns the shorter way onto ``target_deg``."""
    error = wrap180(target_deg - track_deg)
    return (error / params.heading_time_constant_s).clamp(-params.turn_rate_deg_s, params.turn_rate_deg_s)


@dataclass(frozen=True)
class Runways:
    """Each flight's candidate runways, padded to the batch's widest airport: ``[B, C]`` each."""

    threshold_e_m: torch.Tensor
    threshold_n_m: torch.Tensor
    course_deg: torch.Tensor
    elevation_m: torch.Tensor

    @classmethod
    def of(cls, geometries: Sequence[AirportGeometry], *, dtype: torch.dtype, device: torch.device) -> Runways:
        width = max(len(g.candidates) for g in geometries)

        def table(field: str) -> torch.Tensor:
            rows = [[getattr(c, field) for c in g.candidates] + [math.nan] * (width - len(g.candidates))
                    for g in geometries]
            return torch.tensor(rows, dtype=dtype, device=device)

        return cls(threshold_e_m=table("threshold_e_m"), threshold_n_m=table("threshold_n_m"),
                   course_deg=table("course_deg"), elevation_m=table("elevation_m"))

    def pointed(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(threshold e, threshold n, course, elevation)`` of each flight's pointed runway."""
        rows = index[:, None]
        return tuple(t.gather(1, rows)[:, 0] for t in (self.threshold_e_m, self.threshold_n_m, self.course_deg,
                                                        self.elevation_m))


def relative(state: Kinematics, threshold_e_m: torch.Tensor, threshold_n_m: torch.Tensor,
             course_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(before the threshold, right of the centreline, track − course)``: MIRROR of
    `instructions.airport.relative_to_runway` in torch (checked equal in `tests/test_autopilot.py`)."""
    course = torch.deg2rad(course_deg)
    de, dn = state.e_m - threshold_e_m, state.n_m - threshold_n_m
    before = -(de * torch.sin(course) + dn * torch.cos(course))
    right = de * torch.cos(course) - dn * torch.sin(course)
    return before, right, wrap180(state.track_deg - course_deg)


def corridor_half_width(before_m: torch.Tensor, spec: VocabularySpec) -> torch.Tensor:
    """MIRROR of `instructions.envelope.corridor_half_width_m` in torch."""
    return spec.corridor_half_width_m + before_m.clamp(min=0.0) * math.tan(math.radians(spec.corridor_widening_deg))


def converges(heading_deg: torch.Tensor, course_deg: torch.Tensor, right_m: torch.Tensor, before_m: torch.Tensor,
              tolerance_deg: float, spec: VocabularySpec) -> torch.Tensor:
    """MIRROR of `instructions.envelope.heading_converges` in torch, batched (checked equal on random
    cases in `tests/test_autopilot.py`): some track within ``tolerance_deg`` of the heading, at no more
    than 90° to the course, is already within the corridor's width or crosses the extended centreline
    ahead of the threshold."""
    angle = wrap180(heading_deg - course_deg)
    within_angle = angle.abs() <= 90.0 + tolerance_deg
    inside = right_m.abs() <= corridor_half_width(before_m, spec)
    toward = torch.where(right_m >= 0.0, -1.0, 1.0).to(right_m.dtype)
    best = (angle + toward * tolerance_deg).clamp(-90.0, 90.0)
    pointing = best * toward > 0.0
    slope = torch.tan(torch.deg2rad(torch.where(pointing, best.abs(), torch.ones_like(best))))
    reaches = before_m - right_m.abs() / slope > 0.0
    return within_angle & (inside | (pointing & reaches))


class Lateral:
    """The batch's lateral state (captured or not) and law."""

    def __init__(self, batch: int, params: ExecutorParams, spec: VocabularySpec, device: torch.device) -> None:
        self.params, self.spec = params, spec
        self.captured = torch.zeros(batch, dtype=torch.bool, device=device)
        self.tracking = torch.zeros(batch, dtype=torch.bool, device=device)

    def capture_lead(self, state: Kinematics, off_course_deg: torch.Tensor, bank_rad: torch.Tensor) -> torch.Tensor:
        """How far from the line the capture turn must begin (§4.4), metres."""
        params = self.params
        rate = math.radians(params.turn_rate_deg_s)
        speed = state.ground_speed_mps
        off = torch.deg2rad(off_course_deg)
        # the turn toward the line is to the left when the track is right of the course (compass)
        turn_bank = torch.atan(speed * rate / GRAVITY_MPS2) * torch.where(off > 0.0, 1.0, -1.0).to(off.dtype)
        roll_s = (turn_bank - bank_rad).abs() / math.radians(params.bank_rate_deg_s)
        arc = speed / rate * (1.0 - torch.cos(off.abs()))
        return arc + speed * torch.sin(off.abs()) * roll_s / 2.0 + speed * rate * params.heading_time_constant_s ** 2 / 2.0

    def rate(self, state: Kinematics, heading_deg: torch.Tensor, approach: torch.Tensor, runway: torch.Tensor,
             runways: Runways, bank_rad: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The track rate for this cycle, and what the law did (``captured``, ``tracking`` the line, ``bent``,
        ``go_around``);
        ``bank_rad`` is the bank in force (the dynamics' sign), which the capture turn rolls from."""
        params, spec = self.params, self.spec
        e0, n0, course, _elevation = runways.pointed(runway)
        before, right, off_course = relative(state, e0, n0, course)
        go_around = approach == APPROACH_GO_AROUND
        cleared = approach == APPROACH_CLEARED
        toward_line = right * torch.sin(torch.deg2rad(off_course)) < 0.0
        turn_lands_on_line = right.abs() <= self.capture_lead(state, off_course, bank_rad)
        inside = right.abs() <= corridor_half_width(before, spec)
        start = cleared & ~self.captured & (before > 0.0) & ((toward_line & turn_lands_on_line) | inside)
        self.captured = (self.captured | start) & ~go_around
        self.tracking = self.captured & (self.tracking | (off_course.abs() <= spec.heading_tolerance_deg))

        # before the capture: the word, bent toward the line when only its tolerance reaches it
        misses = ~converges(heading_deg, course, right, before, 0.0, spec)
        bend = cleared & ~self.captured & misses & converges(heading_deg, course, right, before,
                                                              spec.heading_tolerance_deg, spec)
        side = torch.where(right >= 0.0, -1.0, 1.0).to(right.dtype)
        target = torch.where(bend, heading_deg + side * spec.heading_tolerance_deg, heading_deg)
        # captured: the line's own target, critically damped
        gain = 1.0 / (4.0 * state.ground_speed_mps * params.heading_time_constant_s)
        line = course - torch.rad2deg(gain * right).clamp(-spec.intercept_angle_deg, spec.intercept_angle_deg)
        target = torch.where(self.tracking, line, target)
        target = torch.where(go_around, course, target)
        # the capture turn: the arc to the line from here, toward the course
        off = torch.deg2rad(off_course)
        arc = state.ground_speed_mps * (1.0 - torch.cos(off)) / right.abs().clamp(min=1e-9)
        tightest = GRAVITY_MPS2 * math.tan(math.radians(params.bank_cap_deg)) / state.ground_speed_mps
        capture = -torch.sign(off_course) * torch.minimum(
            torch.rad2deg(torch.minimum(arc, tightest)), off_course.abs() / params.heading_time_constant_s)
        rate = torch.where(self.captured & ~self.tracking, capture, heading_rate(state.track_deg, target, params))
        return rate, {"captured": self.captured.clone(),
                                                               "tracking": self.tracking.clone(), "bent": bend,
                                                               "go_around": go_around}
