"""The lateral law (vocabulary §5.4; executor design §4.1): a heading word. Nothing else — no capture, no turn onto the
final, no centreline tracking (D2, D3, D9): the words fly the aircraft onto the final, and "go-around" changes no
heading target (D27): the word in force stays, and the model turns the aircraft with heading words as at every row.

Every law here returns a compass TRACK RATE for the inverse (`autopilot.inverse.attitude`) to fly, within the
vocabulary's turn rates and under its bank limit (the executor takes nothing beyond the vocabulary, the user's rule
of 2026-09-24):

- a heading word (§3.3, D8) says a track RELATIVE to the course of the runway in force R. When the executor hears a new
  word it converts it with the course of R at that moment, θ = course(R) + 5k°, and keeps that absolute target until
  the next heading word: a change of R alone does not turn the aircraft. Each word says the track a lead ``L`` later,
  so the law arrives on it then —
  ``|χ̇*| = min(|e| / max(t_heard + L − t, 2Δt), sqrt(2 g p |e| / V), r_max)``: the error over the time left until the
  lead runs out, measured on the executor's own clock from the cycle it heard the word (heard on a cycle that starts
  a row, `sentence`; the judge reads each word from that row), floored at the law's stability floor 2Δt (after that,
  a hold); no faster than the bank can still be taken out before the word (`stopping_rate_deg_s`: the executor cannot
  know a word is a turn's last); up to the vocabulary's largest turn rate ``r_max`` (the bank limit binds first in the
  inverse at most speeds). The error ``e`` is measured from the word in force: a new word turns ``wrap180(θ_new −
  θ_old)`` further than the old one, so a turn said word by word keeps its way even while the aircraft lags it; the
  first word turns the shorter way.

Positions are the airport frame's (`autopilot.frame`), the runway geometry the artefact's candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from ts_transformer.autopilot.ends import runway_lateral_limit_m
from ts_transformer.autopilot.frame import Kinematics, wrap180
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.instructions.airport import AirportGeometry, landing_cross_limit_m
from ts_transformer.instructions.spec import VocabularySpec


def bank_return_gain(speed_mps: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """``k = 2 g p / V``, 1/s²: at a turn rate r the bank is ``≈ V r / g``, and returning it at p turns the track
    ``r² / k`` further (radians; a small-bank reading)."""
    return 2.0 * GRAVITY_MPS2 * math.radians(params.bank_rate_deg_s) / speed_mps


def stopping_rate_deg_s(error_deg: torch.Tensor, speed_mps: torch.Tensor, params: ExecutorParams) -> torch.Tensor:
    """The fastest track rate, deg/s, whose bank the executor can still take out before ``error_deg`` is gone:
    ``r = sqrt(k |e|)`` (`bank_return_gain`)."""
    return torch.rad2deg(torch.sqrt(bank_return_gain(speed_mps, params) * torch.deg2rad(error_deg.abs())))


def word_rate(error_deg: torch.Tensor, to_go_s: torch.Tensor, speed_mps: torch.Tensor, params: ExecutorParams,
              spec: VocabularySpec) -> torch.Tensor:
    """The compass track rate, deg/s, that arrives on the heading word in force when its lead runs out (``to_go_s``
    from now, floored at the stability floor 2Δt), no faster than the bank can still be taken out before the word
    (`stopping_rate_deg_s`), up to the vocabulary's largest turn rate (the bank cap binds in the inverse)."""
    rate = torch.minimum((error_deg / to_go_s.clamp(min=2.0 * params.cycle_s)).abs(),
                         stopping_rate_deg_s(error_deg, speed_mps, params))
    return torch.sign(error_deg) * rate.clamp(max=spec.turn_rate_max_deg_s)


@dataclass(frozen=True)
class Runways:
    """Each flight's candidate runways, padded to the batch's widest airport (NaN): ``[B, C]`` each — the threshold, the
    course, the elevation, how far off the centreline a crossing may lie and be an approach crossing (the landing
    screen's lateral limit, `instructions.airport.landing_cross_limit_m`) and how far a crossing of another candidate
    may lie and end the flight (its runway limit, `ends.runway_lateral_limit_m`, D79)."""

    threshold_e_m: torch.Tensor
    threshold_n_m: torch.Tensor
    course_deg: torch.Tensor
    elevation_m: torch.Tensor
    landing_limit_m: torch.Tensor
    on_runway_m: torch.Tensor

    @classmethod
    def of(cls, geometries: Sequence[AirportGeometry], spec: VocabularySpec, *, dtype: torch.dtype,
           device: torch.device) -> Runways:
        width = max(len(g.candidates) for g in geometries)

        def padded(rows: list[list[float]]) -> torch.Tensor:
            return torch.tensor([row + [math.nan] * (width - len(row)) for row in rows], dtype=dtype, device=device)

        def table(field: str) -> torch.Tensor:
            return padded([[getattr(c, field) for c in g.candidates] for g in geometries])

        return cls(threshold_e_m=table("threshold_e_m"), threshold_n_m=table("threshold_n_m"),
                   course_deg=table("course_deg"), elevation_m=table("elevation_m"),
                   landing_limit_m=padded([[landing_cross_limit_m(g, index, spec.landing_cross_limit_m,
                                                                  spec.parallel_course_delta_deg)
                                            for index in range(len(g.candidates))] for g in geometries]),
                   on_runway_m=padded([[runway_lateral_limit_m(g, index, spec) for index in range(len(g.candidates))]
                                       for g in geometries]))

    def pointed(self, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(threshold e, threshold n, course, elevation)`` of each flight's runway ``index``."""
        rows = index[:, None]
        return tuple(t.gather(1, rows)[:, 0] for t in (self.threshold_e_m, self.threshold_n_m, self.course_deg,
                                                        self.elevation_m))

    def relative(self, state: Kinematics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`relative` to every candidate, ``[B, C]`` each (NaN past an airport's last candidate)."""
        course = torch.deg2rad(self.course_deg)
        de, dn = state.e_m[:, None] - self.threshold_e_m, state.n_m[:, None] - self.threshold_n_m
        before = -(de * torch.sin(course) + dn * torch.cos(course))
        right = de * torch.cos(course) - dn * torch.sin(course)
        return before, right, wrap180(state.track_deg[:, None] - self.course_deg)


def relative(state: Kinematics, threshold_e_m: torch.Tensor, threshold_n_m: torch.Tensor,
             course_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(before the threshold, right of the centreline, track − course)``: MIRROR of
    `instructions.airport.relative_to_runway` in torch (checked equal in `tests/test_autopilot.py`)."""
    course = torch.deg2rad(course_deg)
    de, dn = state.e_m - threshold_e_m, state.n_m - threshold_n_m
    before = -(de * torch.sin(course) + dn * torch.cos(course))
    right = de * torch.cos(course) - dn * torch.sin(course)
    return before, right, wrap180(state.track_deg - course_deg)


class Lateral:
    """The batch's lateral state — the heading word in force (its absolute target, measured from its own
    predecessors), when the executor heard it — and law."""

    #: what a cycle changes, per flight (a multi-aircraft batch holds it for a flight that has not started)
    PER_FLIGHT = ("word_deg", "word_step", "heard_s", "track_unwrapped", "target_unwrapped", "last_track")

    def __init__(self, batch: int, params: ExecutorParams, spec: VocabularySpec, device: torch.device) -> None:
        self.params, self.spec = params, spec
        # the absolute target of the word in force, the step it was said at, when the executor heard it, and track and
        # target unwrapped along the flight
        self.word_deg: torch.Tensor | None = None
        self.word_step = torch.zeros(batch, dtype=torch.long, device=device)
        self.heard_s = torch.zeros(batch, dtype=torch.float64, device=device)
        self.track_unwrapped = torch.zeros(batch, dtype=torch.float64, device=device)
        self.target_unwrapped = torch.zeros(batch, dtype=torch.float64, device=device)
        self.last_track = torch.zeros(batch, dtype=torch.float64, device=device)

    def word_error(self, state: Kinematics, relative_deg: torch.Tensor, issued: torch.Tensor, course_deg: torch.Tensor,
                   time_s: float | torch.Tensor, fresh: torch.Tensor | None = None) -> torch.Tensor:
        """The heading word's error, degrees: a NEW word (``issued`` changed) is converted with ``course_deg`` (the course
        of R now) and its target unwrapped from the word before it; its hearing time is ``time_s``. ``fresh`` (a
        multi-aircraft batch): the flights at their own first cycle, anchored as the batch's first cycle anchors."""
        heard = torch.remainder(course_deg + relative_deg, 360.0)
        if self.word_deg is None:
            self.track_unwrapped = state.track_deg.clone()
            self.target_unwrapped = state.track_deg + wrap180(heard - state.track_deg)
            word = heard
        else:
            heard_before = self.heard_s
            self.track_unwrapped = self.track_unwrapped + wrap180(state.track_deg - self.last_track)
            new = issued != self.word_step
            word = torch.where(new, heard, self.word_deg)
            self.heard_s = torch.where(new, time_s, self.heard_s)
            self.target_unwrapped = torch.where(new, self.target_unwrapped + wrap180(word - self.word_deg),
                                                self.target_unwrapped)
            if fresh is not None and bool(fresh.any()):
                word = torch.where(fresh, heard, word)
                self.track_unwrapped = torch.where(fresh, state.track_deg, self.track_unwrapped)
                self.target_unwrapped = torch.where(fresh, state.track_deg + wrap180(heard - state.track_deg),
                                                    self.target_unwrapped)
                self.heard_s = torch.where(fresh, heard_before, self.heard_s)
        self.word_deg, self.word_step, self.last_track = word, issued.clone(), state.track_deg.clone()
        return self.target_unwrapped - self.track_unwrapped

    def rate(self, state: Kinematics, relative_deg: torch.Tensor, issued: torch.Tensor, runway: torch.Tensor,
             runways: Runways, time_s: float | torch.Tensor, *, fresh: torch.Tensor | None = None) -> torch.Tensor:
        """The track rate for this cycle; ``issued`` is the step the heading word in force was written at, ``runway``
        the runway in force, ``time_s`` the time flown."""
        course = runways.pointed(runway)[2]
        error = self.word_error(state, relative_deg, issued, course, time_s, fresh)
        to_go = self.heard_s + self.spec.heading_lead_s - time_s
        return word_rate(error, to_go, state.ground_speed_mps, self.params, self.spec)
