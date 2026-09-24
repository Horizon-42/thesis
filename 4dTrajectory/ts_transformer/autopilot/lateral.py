"""The lateral law (executor design §4): a heading word, the clearance, the capture and the line.

Every law here returns a compass TRACK RATE for the inverse (`autopilot.inverse.attitude`) to fly, within the
vocabulary's turn rates and under its bank limit (the executor takes nothing beyond the vocabulary, the user's rule
of 2026-09-24):

- a heading word θ (§4.1, instruction-v3): each says the track a lead ``L`` later, so the law arrives on it then —
  ``|χ̇*| = min(|e| / max(t_heard + L − t, 2Δt), sqrt(2 g p |e| / V), r_max)``: the error over the time left until the
  lead runs out, measured on the executor's own clock from the cycle it heard the word (heard on a cycle that starts
  a row, `sentence`; the judge reads each word from that row), floored at the law's stability floor 2Δt (after that,
  a hold); no faster than the bank can still be taken out before the word (`stopping_rate_deg_s`, as the line law:
  the executor cannot know a word is a turn's last); up to the vocabulary's largest turn rate ``r_max`` (the words set
  the pace; the bank limit binds first in the inverse at most speeds). A first-order law ``e / τ_ψ`` at τ_ψ = L arrives
  on a turn's last word only asymptotically — 37 % of its error left a lead after it — and leaves it outside its
  envelope; the time left alone, without the stopping limit, passes a turn's last word by up to 10° (§4.1, the
  reviews of 2026-09-24). The error ``e`` is measured from the word in force: a new word turns ``wrap180(θ_new −
  θ_old)`` further than the old one, so a turn said word by word keeps its way even while the aircraft lags it; a
  go-around starts its clock again;
- cleared, not yet captured (§4.3): when θ itself cannot reach the pointed runway's line but a track
  within the heading tolerance can (the labeller's own test, `instructions.envelope.heading_converges`),
  θ is flown bent by the tolerance toward the line — still inside the word's envelope; when not even that
  reaches it (the executor is not where the observed aircraft was: a wider turn rolled out parallel to the
  final), it intercepts at the vocabulary's intercept angle, the rule the labeller reads an inserted intercept
  by (§2.2) — outside the word's envelope, which the judge reports;
- the capture (§4.4) starts when the turn onto the course would end on the line, flying toward it — or
  at once inside the corridor's width; captured is the executor's state, not a word. The turn is planned at
  ``r = sqrt(r_min r_max)`` of the vocabulary's turn rates (`capture_planning_rate_deg_s`): the bank first rolls to
  the turn's (``t_roll = |φ_turn − φ| / p``, the aircraft closing on the line at ``V sin |Δχ|`` meanwhile, as if
  half that time straight), then an arc of ``R = V / r``, then the roll-out's ``|Δχ| / τ_ψ`` ease-out
  (``V r τ_ψ² / 2`` beyond the arc; the bank's own return at p adds centimetres). So it starts at ``|y| ≤`` the
  sum. A clearance the labeller says at the observed capture turn's onset can come before that lead: the executor
  flies its word, bent if need be, until the lead is reached;
- captured, the aircraft flies the arc that ends tangent to the line from where it is NOW: the rate
  ``V (1 − cos |Δχ|) / |y|`` toward the course, solved again every cycle — so a deceleration inside the
  turn (a shorter radius) or a late roll-in does not leave it short of the line or past it — at least the
  vocabulary's slowest turn rate (a capture begun inside the corridor, far from the line, would otherwise
  flatten into a turn slower than any word allows; it then reaches the course before the line, and the line
  law closes the rest), at most the
  rate the vocabulary's bank limit gives at this speed (``g tan φ_max / V``: a base close in needs a tighter turn
  than the planned rate), the vocabulary's largest turn rate and ``|Δχ| / τ_ψ`` (the heading law's own roll-out; the
  stopping limit never binds here, `rate_for_error`). The capture hands over to the line (§4.5) once the track is within the corridor's
  course tolerance of the course, or once it no longer closes on the line (it crossed it: a late clearance,
  or a turn the bank limit slowed — the arc toward a line behind the aircraft has no rate left);
- tracking the line (§4.5): ``course − sat(k_y · y, a)``, ``k_y = 1 / (4 V τ_ψ)`` (critical damping of the
  first-order loop ``τ_ψ ÿ + ẏ + V k_y y = 0``; the flown track lags the bank, so it may cross the
  centreline by metres). ``a`` is the vocabulary's intercept angle outside the corridor and HALF the
  corridor's course tolerance inside it: the corridor holds the track within that tolerance of the
  course, and the heading law's own lag keeps the other half. The executor's own heading law flies it
  (`rate_for_error`);
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


def bank_return_gain(speed_mps: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """``k = 2 g p / V``, 1/s²: at a turn rate r the bank is ``≈ V r / g``, and returning it at p turns the track
    ``r² / k`` further (radians; a small-bank reading)."""
    return 2.0 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s) / speed_mps


def stopping_rate_deg_s(error_deg: torch.Tensor, speed_mps: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """The fastest track rate, deg/s, whose bank the executor can still take out before ``error_deg`` is gone:
    ``r = sqrt(k |e|)`` (`bank_return_gain`)."""
    return torch.rad2deg(torch.sqrt(bank_return_gain(speed_mps, params) * torch.deg2rad(error_deg.abs())))


def capture_planning_rate_deg_s(spec: VocabularySpec) -> float:
    """The rate the capture turn is planned at (`Lateral.capture_lead`), deg/s: the geometric mean of the vocabulary's
    slowest and largest turn rates — the middle of a range that spans an order of magnitude (the user, 2026-09-24).
    Planned at the slowest rate itself the turn is flown on the envelope's edge and read just under it (17 of 500
    train captures outside); the turn flown is the arc to the line from where the aircraft is, inside the
    vocabulary's rates (`Lateral.rate`)."""
    return math.sqrt(spec.turn_rate_min_deg_s * spec.turn_rate_max_deg_s)


def word_rate(error_deg: torch.Tensor, to_go_s: torch.Tensor, speed_mps: torch.Tensor, params: ExecutorParams,
              spec: VocabularySpec) -> torch.Tensor:
    """§4.1 (instruction-v3): the compass track rate, deg/s, that arrives on the heading word in force when its lead
    runs out (``to_go_s`` from now, floored at the stability floor 2Δt), no faster than the bank can still be taken
    out before the word (`stopping_rate_deg_s`: the executor cannot know a word is a turn's last) — the words set the
    pace, so the law turns up to the vocabulary's largest turn rate (the bank cap binds in the inverse)."""
    rate = torch.minimum((error_deg / to_go_s.clamp(min=2.0 * params.cycle_s)).abs(),
                         stopping_rate_deg_s(error_deg, speed_mps, params))
    return torch.sign(error_deg) * rate.clamp(max=spec.turn_rate_max_deg_s)


def rate_for_error(error_deg: torch.Tensor, params: ExecutorParams, spec: VocabularySpec) -> torch.Tensor:
    """§4.1: the compass track rate, deg/s, of the executor's OWN turn that takes out a heading error of ``error_deg``
    (target − track) — its own intercept, a go-around, the line's heading: ``e / τ_ψ`` within the vocabulary's largest
    turn rate. It needs no stopping limit (`stopping_rate_deg_s`): with p τ_ψ equal to the vocabulary's bank limit
    (`derive`), ``e / τ_ψ`` exceeds the stopping rate only where the bank limit binds first (``2 φ > tan φ``)."""
    return (error_deg / params.heading_time_constant_s).clamp(-spec.turn_rate_max_deg_s, spec.turn_rate_max_deg_s)


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
        # the word in force, when the executor heard it, and track and target unwrapped along the flight (§4.1)
        self.word_deg: torch.Tensor | None = None
        self.word_step = torch.zeros(batch, dtype=torch.long, device=device)
        self.heard_s = torch.zeros(batch, dtype=torch.float64, device=device)
        self.track_unwrapped = torch.zeros(batch, dtype=torch.float64, device=device)
        self.target_unwrapped = torch.zeros(batch, dtype=torch.float64, device=device)
        self.last_track = torch.zeros(batch, dtype=torch.float64, device=device)

    def tightest_rad_s(self, state: Kinematics) -> torch.Tensor:
        """The turn rate the vocabulary's bank limit gives at this speed."""
        return GRAVITY_MPS2 * math.tan(math.radians(self.spec.turn_bank_max_deg)) / state.ground_speed_mps

    def capture_lead(self, state: Kinematics, off_course_deg: torch.Tensor, bank_rad: torch.Tensor,
                     bank_rate_rad_s: float) -> torch.Tensor:
        """How far from the line the capture turn must begin (§4.4), metres."""
        params = self.params
        rate = math.radians(capture_planning_rate_deg_s(self.spec))
        speed = state.ground_speed_mps
        off = torch.deg2rad(off_course_deg)
        # the turn toward the line is to the left when the track is right of the course (compass)
        turn_bank = torch.atan(speed * rate / GRAVITY_MPS2) * torch.where(off > 0.0, 1.0, -1.0).to(off.dtype)
        roll_s = (turn_bank - bank_rad).abs() / bank_rate_rad_s
        return (speed / rate * (1.0 - torch.cos(off.abs())) + speed * torch.sin(off.abs()) * roll_s / 2.0
                + speed * rate * params.heading_time_constant_s ** 2 / 2.0)

    def word_error(self, state: Kinematics, heading_deg: torch.Tensor, issued: torch.Tensor,
                   go_around: torch.Tensor, time_s: float) -> torch.Tensor:
        """The heading word's error, degrees: its target unwrapped from the words before it (§4.1); a new word's
        hearing time is ``time_s``. A go-around flies the course, not the word, so the word's target is anchored
        again at the track meanwhile: a word after it is measured from where the aircraft is."""
        if self.word_deg is None:
            self.track_unwrapped = state.track_deg.clone()
            self.target_unwrapped = state.track_deg + wrap180(heading_deg - state.track_deg)
        else:
            self.track_unwrapped = self.track_unwrapped + wrap180(state.track_deg - self.last_track)
            new = issued != self.word_step
            self.heard_s = torch.where(new | go_around, time_s, self.heard_s)
            self.target_unwrapped = torch.where(new, self.target_unwrapped + wrap180(heading_deg - self.word_deg),
                                                self.target_unwrapped)
            self.target_unwrapped = torch.where(go_around, self.track_unwrapped + wrap180(heading_deg - state.track_deg),
                                                self.target_unwrapped)
        self.word_deg, self.word_step, self.last_track = heading_deg.clone(), issued.clone(), state.track_deg.clone()
        return self.target_unwrapped - self.track_unwrapped

    def rate(self, state: Kinematics, heading_deg: torch.Tensor, issued: torch.Tensor, approach: torch.Tensor,
             runway: torch.Tensor, runways: Runways, bank_rad: torch.Tensor, bank_rate_rad_s: float,
             time_s: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The track rate for this cycle, and what the law did (``captured``, ``tracking`` the line, ``bent``,
        ``go_around``); ``issued`` is the step the heading word in force was written at, ``bank_rad`` the bank
        in force (the dynamics' sign), which the capture turn rolls from at ``bank_rate_rad_s``; ``time_s`` the
        time flown."""
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

        # before the capture: the word, bent toward the line when only its tolerance reaches it, and when not even
        # that reaches it, an intercept at the vocabulary's intercept angle (as the labeller reads one, §2.2)
        misses = ~converges(heading_deg, course, right, before, 0.0, spec)
        bent_reaches = converges(heading_deg, course, right, before, spec.heading_tolerance_deg, spec)
        waiting = cleared & ~self.captured & (before > 0.0)
        bend = waiting & misses & bent_reaches
        intercept = waiting & ~bent_reaches
        side = torch.where(right >= 0.0, -1.0, 1.0).to(right.dtype)
        error = self.word_error(state, heading_deg, issued, go_around, time_s) + torch.where(
            bend, side * spec.heading_tolerance_deg, 0.0)
        error = torch.where(intercept, wrap180(course + side * spec.intercept_angle_deg - state.track_deg), error)
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
        steady = torch.clamp(arc, min=math.radians(spec.turn_rate_min_deg_s))
        capture_rad_s = torch.minimum(steady, self.tightest_rad_s(state).clamp(max=math.radians(spec.turn_rate_max_deg_s)))
        capture = -torch.sign(off_course) * torch.minimum(torch.rad2deg(capture_rad_s),
                                                          off_course.abs() / params.heading_time_constant_s)
        line_rate = rate_for_error(error, params, spec)
        # the words set the pace of a turn they describe; the executor's own intercept and a go-around are its own
        to_go = self.heard_s + spec.heading_lead_s - time_s
        free = torch.where(intercept | go_around, rate_for_error(error, params, spec),
                           word_rate(error, to_go, state.ground_speed_mps, params, spec))
        rate = torch.where(self.captured & ~self.tracking, capture, torch.where(self.tracking, line_rate, free))
        intercept_target = course + side * spec.intercept_angle_deg
        off_word = intercept & (wrap180(intercept_target - heading_deg).abs() > spec.heading_tolerance_deg)
        return rate, {"captured": self.captured.clone(), "tracking": self.tracking.clone(), "bent": bend,
                      "intercepting": intercept, "intercepting_off_word": off_word, "go_around": go_around}
