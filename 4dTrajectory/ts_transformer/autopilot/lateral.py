"""The lateral law (executor design §4): a heading word, the clearance, the capture and the line.

Every law here returns a compass TRACK RATE for the inverse (`autopilot.inverse.attitude`) to fly:

- a heading word θ (§4.1): ``χ̇* = sat(e / τ_ψ, ±r_turn)`` — at the steady rate, easing out over τ_ψ; the bank
  it takes is the inverse's, capped and rate-limited there. The error ``e`` is measured from the word in
  force, as the vocabulary measures a word (§2.3): a new word turns ``wrap180(θ_new − θ_old)`` further than
  the old one, so a split turn's next part continues the way the first part set even while the aircraft
  still lags it (the shorter way from the TRACK would turn back once the lag exceeds 180° less the part);
- cleared, not yet captured (§4.3): when θ itself cannot reach the pointed runway's line but a track
  within the heading tolerance can (the labeller's own test, `instructions.envelope.heading_converges`),
  θ is flown bent by the tolerance toward the line — still inside the word's envelope;
- the capture (§4.4) starts when the turn onto the course would end on the line, flying toward it — or
  at once inside the corridor's width; captured is the executor's state, not a word. The turn is the one
  the laws fly, at ``r = min(r_turn, g tan φ_cap / V)`` (the steady rate, where the bank cap allows it):
  the bank first rolls to the turn's (``t_roll = |φ_turn − φ| / p``, the aircraft closing on the line at
  ``V sin |Δχ|`` meanwhile, as if half that time straight), then an arc of ``R = V / r``, then the
  roll-out's tail — the ``|Δχ| / τ_ψ`` ease-out (``V r τ_ψ² / 2`` beyond the arc) and the bank's own
  return at p (``V r³ / (6 k²)``, ``k = 2 g p / V``, below). So it starts at ``|y| ≤`` the sum, every term
  the executor's own, nothing tuned;
- captured, the aircraft flies the arc that ends tangent to the line from where it is NOW: the rate
  ``V (1 − cos |Δχ|) / |y|`` toward the course, solved again every cycle — so a deceleration inside the
  turn (a shorter radius) or a late roll-in does not leave it short of the line or past it — at most the
  rate the bank cap gives at this speed (``g tan φ_cap / V``: a base close in needs a tighter turn than the
  steady rate), the vocabulary's largest turn rate, ``|Δχ| / τ_ψ`` (the heading law's own roll-out), and
  ``sqrt(k |Δχ|)``: at a turn rate r the bank is ``≈ V r / g``, and returning it at p turns the track
  ``r² V / (2 g p)`` further, so this is the rate the bank can still take out before the course (a
  small-bank reading). The capture hands over to the line (§4.5) once the track is within the corridor's
  course tolerance of the course, or once it no longer closes on the line (it crossed it: a late clearance,
  or a turn the bank cap slowed — the arc toward a line behind the aircraft has no rate left);
- tracking the line (§4.5): ``course − sat(k_y · y, a)``, ``k_y = 1 / (4 V τ_ψ)`` (critical damping of the
  first-order loop ``τ_ψ ÿ + ẏ + V k_y y = 0``; the flown track lags the bank, so it may cross the
  centreline by metres). ``a`` is the vocabulary's intercept angle outside the corridor and HALF the
  corridor's course tolerance inside it: the corridor holds the track within that tolerance of the
  course, and the heading law's own lag keeps the other half. The heading law flies it at no more than
  ``sqrt(k |e|)``, as the capture does: a recovery from a capture that ran past the line otherwise swings
  past the course by more than the tolerance while the bank comes back;
- a go-around (§4.6) cancels the clearance and the capture and flies the pointed runway's course. A change
  of the runway pointer once cleared is refused (§4.6), as the labeller refuses it.

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


def rate_for_error(error_deg: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """§4.1: the compass track rate, deg/s, that takes out a heading error of ``error_deg`` (target − track)."""
    return (error_deg / params.heading_time_constant_s).clamp(-params.turn_rate_deg_s, params.turn_rate_deg_s)


def heading_rate(track_deg: torch.Tensor, target_deg: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """§4.1: the compass track rate, deg/s, that turns the shorter way onto ``target_deg``."""
    return rate_for_error(wrap180(target_deg - track_deg), params)


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
    """The batch's lateral state — the heading word in force measured from its own predecessors, cleared,
    captured, tracking — and law."""

    def __init__(self, batch: int, params: ExecutorParams, spec: VocabularySpec, device: torch.device) -> None:
        self.params, self.spec = params, spec
        self.captured = torch.zeros(batch, dtype=torch.bool, device=device)
        self.tracking = torch.zeros(batch, dtype=torch.bool, device=device)
        self.cleared = torch.zeros(batch, dtype=torch.bool, device=device)
        self.runway: torch.Tensor | None = None
        # the word in force, and track and target unwrapped along the flight (§4.1)
        self.word_deg: torch.Tensor | None = None
        self.word_step = torch.zeros(batch, dtype=torch.long, device=device)
        self.track_unwrapped = torch.zeros(batch, dtype=torch.float64, device=device)
        self.target_unwrapped = torch.zeros(batch, dtype=torch.float64, device=device)
        self.last_track = torch.zeros(batch, dtype=torch.float64, device=device)

    def tightest_rad_s(self, state: Kinematics) -> torch.Tensor:
        """The turn rate the bank cap gives at this speed."""
        return GRAVITY_MPS2 * math.tan(math.radians(self.params.bank_cap_deg)) / state.ground_speed_mps

    def turn_rate_rad_s(self, state: Kinematics) -> torch.Tensor:
        """The capture turn's steady rate: r_turn where the bank cap allows it at this speed."""
        return torch.clamp(self.tightest_rad_s(state), max=math.radians(self.params.turn_rate_deg_s))

    def capture_lead(self, state: Kinematics, off_course_deg: torch.Tensor, bank_rad: torch.Tensor,
                     bank_rate_rad_s: float) -> torch.Tensor:
        """How far from the line the capture turn must begin (§4.4), metres."""
        params = self.params
        rate = self.turn_rate_rad_s(state)
        speed = state.ground_speed_mps
        off = torch.deg2rad(off_course_deg)
        # the turn toward the line is to the left when the track is right of the course (compass)
        turn_bank = torch.atan(speed * rate / GRAVITY_MPS2) * torch.where(off > 0.0, 1.0, -1.0).to(off.dtype)
        roll_s = (turn_bank - bank_rad).abs() / bank_rate_rad_s
        k = 2.0 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s) / speed
        return (speed / rate * (1.0 - torch.cos(off.abs())) + speed * torch.sin(off.abs()) * roll_s / 2.0
                + speed * rate * params.heading_time_constant_s ** 2 / 2.0 + speed * rate ** 3 / (6.0 * k ** 2))

    def word_error(self, state: Kinematics, heading_deg: torch.Tensor, issued: torch.Tensor) -> torch.Tensor:
        """The heading word's error, degrees: its target unwrapped from the words before it (§4.1)."""
        if self.word_deg is None:
            self.track_unwrapped = state.track_deg.clone()
            self.target_unwrapped = state.track_deg + wrap180(heading_deg - state.track_deg)
        else:
            self.track_unwrapped = self.track_unwrapped + wrap180(state.track_deg - self.last_track)
            new = issued != self.word_step
            self.target_unwrapped = torch.where(new, self.target_unwrapped + wrap180(heading_deg - self.word_deg),
                                                self.target_unwrapped)
        self.word_deg, self.word_step, self.last_track = heading_deg.clone(), issued.clone(), state.track_deg.clone()
        return self.target_unwrapped - self.track_unwrapped

    def rate(self, state: Kinematics, heading_deg: torch.Tensor, issued: torch.Tensor, approach: torch.Tensor,
             runway: torch.Tensor, runways: Runways, bank_rad: torch.Tensor,
             bank_rate_rad_s: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The track rate for this cycle, and what the law did (``captured``, ``tracking`` the line, ``bent``,
        ``go_around``); ``issued`` is the step the heading word in force was written at, ``bank_rad`` the bank
        in force (the dynamics' sign), which the capture turn rolls from at ``bank_rate_rad_s``."""
        params, spec = self.params, self.spec
        if self.runway is not None and bool(((runway != self.runway) & (self.cleared | self.captured)).any()):
            raise ValueError("the runway pointer changed after the clearance (executor design §4.6)")
        self.runway = runway.clone()
        e0, n0, course, _elevation = runways.pointed(runway)
        before, right, off_course = relative(state, e0, n0, course)
        go_around = approach == APPROACH_GO_AROUND
        cleared = approach == APPROACH_CLEARED
        self.cleared = (self.cleared | cleared) & ~go_around
        toward_line = right * torch.sin(torch.deg2rad(off_course)) < 0.0
        turn_lands_on_line = right.abs() <= self.capture_lead(state, off_course, bank_rad, bank_rate_rad_s)
        inside = right.abs() <= corridor_half_width(before, spec)
        start = cleared & ~self.captured & (before > 0.0) & ((toward_line & turn_lands_on_line) | inside)
        self.captured = (self.captured | start) & ~go_around
        on_course = off_course.abs() <= spec.corridor_course_tolerance_deg
        self.tracking = self.captured & (self.tracking | on_course | ~toward_line)

        # before the capture: the word, bent toward the line when only its tolerance reaches it
        misses = ~converges(heading_deg, course, right, before, 0.0, spec)
        bend = cleared & ~self.captured & misses & converges(heading_deg, course, right, before,
                                                              spec.heading_tolerance_deg, spec)
        side = torch.where(right >= 0.0, -1.0, 1.0).to(right.dtype)
        error = self.word_error(state, heading_deg, issued) + torch.where(bend, side * spec.heading_tolerance_deg, 0.0)
        # tracking: the line's own target, critically damped, steering inside the corridor's course tolerance
        gain = 1.0 / (4.0 * state.ground_speed_mps * params.heading_time_constant_s)
        steer = torch.where(inside, torch.full_like(right, spec.corridor_course_tolerance_deg / 2.0),
                            torch.full_like(right, spec.intercept_angle_deg))
        line = course - torch.maximum(torch.minimum(torch.rad2deg(gain * right), steer), -steer)
        error = torch.where(self.tracking, wrap180(line - state.track_deg), error)
        error = torch.where(go_around, wrap180(course - state.track_deg), error)
        # the capture turn: the arc to the line from here, toward the course, as fast as the bank allows
        off = torch.deg2rad(off_course).abs()
        arc = state.ground_speed_mps * (1.0 - torch.cos(off)) / right.abs().clamp(min=1e-9)
        k = 2.0 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s) / state.ground_speed_mps
        capture_rad_s = torch.minimum(torch.minimum(arc, self.tightest_rad_s(state).clamp(
            max=math.radians(spec.turn_rate_max_deg_s))), torch.sqrt(k * off))
        capture = -torch.sign(off_course) * torch.minimum(torch.rad2deg(capture_rad_s),
                                                          off_course.abs() / params.heading_time_constant_s)
        # on the line the heading law turns no faster than the bank can take out before the target: the
        # corridor's course tolerance is tighter than a heading word's
        line_rate = torch.sign(error) * torch.minimum(rate_for_error(error, params).abs(),
                                                      torch.rad2deg(torch.sqrt(k * torch.deg2rad(error.abs()))))
        rate = torch.where(self.captured & ~self.tracking, capture,
                           torch.where(self.tracking, line_rate, rate_for_error(error, params)))
        return rate, {"captured": self.captured.clone(), "tracking": self.tracking.clone(), "bent": bend,
                      "go_around": go_around}
