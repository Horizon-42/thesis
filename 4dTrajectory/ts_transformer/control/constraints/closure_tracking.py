"""Track a closed-form closure reference with the point-mass rollout (scene design P1.d).

The closure output says WHAT to fly — a path, a speed profile and a height profile drawn
in closed form from fourteen decision numbers — but its trajectory is a geometric
family member, not a solution of the flight equations: curvature jumps at the CSC
junctions and speed jumps at the profile knots, so only 22 % of its flights pass the
clean-polar flyability check (the observed tracks: 98 %). This hook lets the dynamics
fly it: a command hook that ignores the (empty) schedule and, once per segment, steers
the rollout's own state toward the reference —

* lateral: the L1 law (``control.guidance_laws.l1_bank``) on the cross-track and heading
  error relative to the reference's local course at the nearest node, plus the
  reference curvature as a feed-forward bank ``atan(V²κ / g)``;
* vertical: the glidepath law on the height error to the reference's height at that
  node, the reference's own local slope standing in for the glidepath;
* thrust: a PI speed hold toward the reference's ground speed there — the integrator
  (the nominal law's ``control_nominal_speed_gain``, from the anchor's implied thrust)
  finds the trim the law cannot know, the proportional part (``SPEED_PROPORTIONAL_GAIN``)
  answers within the segment — plus the reference's own acceleration as a feed-forward,
  and the ALONG-TRACK error (where the reference should be by now against where the
  aircraft is, at ``ALONG_TRACK_GAIN_PER_S``, capped at ``ALONG_TRACK_SPEED_MAX_MPS``)
  folded into the speed target. Without the along-track term the clean polar, which
  cannot decelerate as fast as a dirty-configuration reference, ran the aircraft a
  kilometre ahead by the end on synthetic flights; without the proportional part the
  integrator alone took four segments to undo the anchor's implied thrust.

Gains are the nominal law's config fields, capped at ``1/Δt`` of the segment hold like
every hook. Positions are always the dynamics' own; the reference is only ever a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from aerodynamic_model.torch_dynamics import GRAVITY_MPS2
from config import TSConfig
from control.dynamics.hooks import RolloutStateView
from control.envelope import (
    MAX_BANK_RAD, MAX_LOAD_FACTOR, MAX_THRUST_FRACTION, MIN_LOAD_FACTOR, MIN_THRUST_FRACTION,
)
from control.guidance_laws import glidepath_load_factor, l1_bank, wrap_angle

HOOK_NAME = "closure-tracker"
_SPEED_FLOOR_MPS = 1.0
_HOLD_FLOOR_S = 0.5
# The along-track loop (a P1.d measurement setting, not yet a config field): a lag of
# 500 m asks for 10 m/s more than the reference's speed there, capped at ±10 m/s.
ALONG_TRACK_GAIN_PER_S = 0.02
ALONG_TRACK_SPEED_MAX_MPS = 10.0
SPEED_PROPORTIONAL_GAIN = 0.3        # thrust (in weight units per m/s of speed error)·s
ACCELERATION_FEEDFORWARD_MAX = 0.5   # m/s²
_DIAGNOSTIC_KEYS = (
    "hook_steps", "tracker_bank_saturated_steps", "tracker_load_saturated_steps",
    "tracker_thrust_saturated_steps", "tracker_cross_track_abs_m", "tracker_height_error_abs_m",
    "tracker_speed_error_abs_mps", "tracker_along_track_abs_m",
)


@dataclass(frozen=True)
class ReferencePath:
    """One flight's reference at its nodes: chart position, height, ground speed, the
    time the reference passes each node (s from the anchor), its arc length there, the
    local course (math-ENU), signed curvature (CCW positive, 1/m) and height slope
    (du/ds)."""
    xy: np.ndarray
    u: np.ndarray
    speed: np.ndarray
    times: np.ndarray
    arc: np.ndarray
    course: np.ndarray
    curvature: np.ndarray
    slope: np.ndarray
    acceleration: np.ndarray        # dv/dt along the reference (m/s²)


def reference_path(xy: np.ndarray, u: np.ndarray, speed: np.ndarray, times: np.ndarray) -> ReferencePath:
    """Course, curvature and slope from the node sequence (central differences)."""
    xy, u, speed, times = (np.asarray(a, dtype=np.float64) for a in (xy, u, speed, times))
    step = np.gradient(xy, axis=0)
    course = np.arctan2(step[:, 1], step[:, 0])
    ds = np.maximum(np.hypot(step[:, 0], step[:, 1]), 1e-6)
    turn = np.concatenate([[0.0], (np.diff(course) + np.pi) % (2 * np.pi) - np.pi])
    curvature = np.gradient(np.cumsum(turn)) / ds
    slope = np.gradient(u) / ds
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    # The reference's own deceleration as a feed-forward, capped at what an approach
    # does (±ACCELERATION_FEEDFORWARD_MAX); the first node's one-sided gradient is not
    # used (it carries the label's knot-vs-anchor speed mismatch).
    acceleration = np.clip(np.gradient(speed) / np.maximum(np.gradient(times), 1e-6),
                           -ACCELERATION_FEEDFORWARD_MAX, ACCELERATION_FEEDFORWARD_MAX)
    acceleration[0] = 0.0
    return ReferencePath(xy, u, speed, times, arc, course, curvature, slope, acceleration)


class ClosureTracker:
    """The command hook: one reference per flight of the batch, padded to the longest."""

    needs_reference = False

    def __init__(self, config: TSConfig, dynamics: dict[str, torch.Tensor], references: Sequence[ReferencePath]) -> None:
        device = dynamics["max_thrust_n"].device
        width = max(len(r.xy) for r in references)

        def padded(values: Sequence[np.ndarray], fill: float) -> torch.Tensor:
            out = np.full((len(values), width) + values[0].shape[1:], fill, dtype=np.float64)
            for row, value in enumerate(values):
                out[row, :len(value)] = value
            return torch.from_numpy(out).to(device)

        self.xy = padded([r.xy for r in references], 0.0)
        self.u = padded([r.u for r in references], 0.0)
        self.speed = padded([r.speed for r in references], 0.0)
        self.course = padded([r.course for r in references], 0.0)
        self.curvature = padded([r.curvature for r in references], 0.0)
        self.slope = padded([r.slope for r in references], 0.0)
        self.acceleration = padded([r.acceleration for r in references], 0.0)
        # Padded with each reference's last time / arc, so a time past the end reads as
        # the end (np.interp semantics) and padding never wins a nearest-node search.
        self.times = padded([r.times for r in references], 0.0)
        self.arc = padded([r.arc for r in references], 0.0)
        for row, r in enumerate(references):
            self.times[row, len(r.times):] = float(r.times[-1])
            self.arc[row, len(r.arc):] = float(r.arc[-1])
        self.valid = padded([np.ones(len(r.xy)) for r in references], 0.0) > 0.5
        self._elapsed = torch.zeros(len(references), dtype=torch.float64, device=device)
        self.l1_distance = config.control_nominal_l1_distance_m
        self.vertical_lookahead = config.control_nominal_vertical_lookahead_m
        self.vertical_gain = config.control_nominal_vertical_gain
        self.speed_gain = config.control_nominal_speed_gain
        self.max_thrust_n = dynamics["max_thrust_n"].to(torch.float64)
        self._thrust = dynamics["initial_controls"][:, 0].to(torch.float64).clone()
        self._counts: torch.Tensor | None = None

    def __call__(self, state: RolloutStateView, command: torch.Tensor, segment_index: int) -> torch.Tensor:
        chart = state.chart.to(torch.float64)
        position, velocity, mass = chart[:, :2], chart[:, 3:6], chart[:, 6]
        distance2 = ((self.xy - position[:, None, :]) ** 2).sum(dim=-1)
        distance2 = torch.where(self.valid, distance2, torch.full_like(distance2, float("inf")))
        nearest = distance2.argmin(dim=1, keepdim=True)
        course = self.course.gather(1, nearest)[:, 0]
        curvature = self.curvature.gather(1, nearest)[:, 0]
        u_ref = self.u.gather(1, nearest)[:, 0]
        speed_ref = self.speed.gather(1, nearest)[:, 0]
        slope = self.slope.gather(1, nearest)[:, 0]
        accel_ref = self.acceleration.gather(1, nearest)[:, 0]
        xy_ref = self.xy.gather(1, nearest.unsqueeze(-1).expand(-1, 1, 2))[:, 0]
        arc_here = self.arc.gather(1, nearest)[:, 0]
        # Where the reference should be by now: its arc length at the elapsed time
        # (a monotone search over the node times, one row per flight).
        due = (self.times <= self._elapsed[:, None]).sum(dim=1, keepdim=True).clamp(min=1) - 1
        arc_due = self.arc.gather(1, due)[:, 0]
        along_track = arc_due - arc_here                    # + = behind the schedule

        ground_speed = torch.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2).clamp(min=_SPEED_FLOOR_MPS)
        speed = velocity.norm(dim=1).clamp(min=_SPEED_FLOOR_MPS)
        heading = torch.atan2(velocity[:, 1], velocity[:, 0])
        path_angle = torch.atan2(velocity[:, 2], ground_speed)
        hold_rate = 1.0 / state.duration_s.to(torch.float64).clamp(min=_HOLD_FLOOR_S)

        # Lateral: cross-track to the right of the reference course, heading error CCW.
        offset = position - xy_ref
        cross_track = offset[:, 0] * torch.sin(course) - offset[:, 1] * torch.cos(course)
        heading_error = wrap_angle(heading - course)
        bank = l1_bank(cross_track, heading_error, ground_speed,
                       l1_distance_m=self.l1_distance, bank_limit_rad=MAX_BANK_RAD)
        bank = (bank + torch.atan(ground_speed.square() * curvature / GRAVITY_MPS2)).clamp(min=-MAX_BANK_RAD, max=MAX_BANK_RAD)
        # Vertical: the reference height at the nearest node, its slope as the "glidepath".
        height_error = chart[:, 2] - u_ref
        load = glidepath_load_factor(
            height_error, path_angle, speed, bank, glidepath_tan=-slope,
            lookahead_m=self.vertical_lookahead, gain_per_s=hold_rate.clamp(max=self.vertical_gain),
            load_limits=(MIN_LOAD_FACTOR, MAX_LOAD_FACTOR),
        )
        # Thrust: a PI speed hold toward the reference speed (the along-track error folded
        # into the target) plus the reference's acceleration as a feed-forward. The
        # integrator is the trim (from the anchor's implied thrust), clamped so it cannot
        # wind up beyond the envelope.
        speed_target = speed_ref + (ALONG_TRACK_GAIN_PER_S * along_track).clamp(
            min=-ALONG_TRACK_SPEED_MAX_MPS, max=ALONG_TRACK_SPEED_MAX_MPS)
        speed_error = speed_target - speed
        weight_per_newton = mass / self.max_thrust_n
        self._thrust = (self._thrust + hold_rate.clamp(max=self.speed_gain) * weight_per_newton * speed_error).clamp(
            min=MIN_THRUST_FRACTION, max=MAX_THRUST_FRACTION).detach()
        thrust = (self._thrust + SPEED_PROPORTIONAL_GAIN * weight_per_newton * speed_error
                  + weight_per_newton * accel_ref).clamp(min=MIN_THRUST_FRACTION, max=MAX_THRUST_FRACTION)
        self._elapsed = self._elapsed + state.duration_s.to(torch.float64)

        counts = torch.stack((
            torch.tensor(float(len(bank)), dtype=torch.float64, device=bank.device),
            (bank.abs() >= MAX_BANK_RAD - 1e-9).sum().to(torch.float64),
            ((load <= MIN_LOAD_FACTOR + 1e-9) | (load >= MAX_LOAD_FACTOR - 1e-9)).sum().to(torch.float64),
            ((thrust <= MIN_THRUST_FRACTION + 1e-9) | (thrust >= MAX_THRUST_FRACTION - 1e-9)).sum().to(torch.float64),
            cross_track.abs().sum(), height_error.abs().sum(), (speed - speed_ref).abs().sum(),
            along_track.abs().sum(),
        )).detach()
        self._counts = counts if self._counts is None else self._counts + counts
        return torch.stack((thrust, bank, load), dim=-1).to(command.dtype)

    def diagnostics(self) -> dict[str, torch.Tensor]:
        counts = torch.zeros(len(_DIAGNOSTIC_KEYS), dtype=torch.float64) if self._counts is None else self._counts.cpu()
        return dict(zip(_DIAGNOSTIC_KEYS, counts.unbind()))
