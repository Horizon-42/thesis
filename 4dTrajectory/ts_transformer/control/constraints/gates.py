"""On-final membership of a rollout state, and the shared geometry every hook reads."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from control.dynamics.hooks import RolloutStateView
from final_approach_geometry import (
    alignment_cosine,
    corridor_halfwidth,
    hard_on_final,
    runway_axes,
    soft_on_final,
)

_SPEED_FLOOR_MPS = 1.0
_HOLD_FLOOR_S = 0.5      # the rollout's own integrator step; a shorter hold is not a hold


@dataclass(frozen=True)
class RunwayAxesView:
    """One segment start in runway axes: what the corridor rows are written in."""

    d: torch.Tensor                 # [B] along-course distance back from the threshold
    xt: torch.Tensor                # [B] cross-track, + right of the inbound course
    halfwidth: torch.Tensor         # [B] full-scale corridor half-width at d
    heading_error: torch.Tensor     # [B] ψ_aircraft − ψ_runway, wrapped
    cos_align: torch.Tensor         # [B] cos(heading_error)
    ground_speed: torch.Tensor      # [B] horizontal speed, floored
    speed: torch.Tensor             # [B] airspeed magnitude, floored
    path_angle: torch.Tensor        # [B] γ from the chart velocity
    height: torch.Tensor            # [B] chart height u
    hold_rate: torch.Tensor         # [B] 1 / segment hold (1/s): the fastest rate a hold can realise


def runway_axes_view(state: RolloutStateView, runway_heading_rad: torch.Tensor) -> RunwayAxesView:
    chart = state.chart
    e, n, u = chart[:, 0], chart[:, 1], chart[:, 2]
    ve, vn, vu = chart[:, 3], chart[:, 4], chart[:, 5]
    psi = runway_heading_rad.to(chart.dtype)
    d, xt = runway_axes(e.unsqueeze(1), n.unsqueeze(1), psi)
    d, xt = d[:, 0], xt[:, 0]
    ground_speed = torch.sqrt(ve * ve + vn * vn).clamp(min=_SPEED_FLOOR_MPS)
    speed = torch.sqrt(ve * ve + vn * vn + vu * vu).clamp(min=_SPEED_FLOOR_MPS)
    heading = torch.atan2(vn, ve)
    heading_error = torch.atan2(torch.sin(heading - psi), torch.cos(heading - psi))
    # alignment_cosine floors the vector norm at its position-step floor (metres); on a
    # velocity that is 1 m/s, far below any flying speed, so the floor never binds here.
    cos_align = alignment_cosine(ve.unsqueeze(1), vn.unsqueeze(1), psi)[:, 0]
    return RunwayAxesView(
        d=d, xt=xt, halfwidth=corridor_halfwidth(d), heading_error=heading_error,
        cos_align=cos_align, ground_speed=ground_speed, speed=speed,
        path_angle=torch.atan2(vu, ground_speed), height=u,
        hold_rate=1.0 / state.duration_s.to(chart.dtype).clamp(min=_HOLD_FLOOR_S),
    )


def on_final_weight(view: RunwayAxesView, *, hard: bool) -> torch.Tensor:
    """The ``on-final`` gate on a rollout state: ``[B]`` in ``[0, 1]`` (bool when hard)."""
    d, xt, cos_align = view.d.unsqueeze(1), view.xt.unsqueeze(1), view.cos_align.unsqueeze(1)
    if hard:
        return hard_on_final(d, xt, cos_align)[:, 0].to(view.d.dtype)
    return soft_on_final(d, xt, cos_align)[:, 0]
