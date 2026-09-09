"""The final-approach corridor in the threshold-anchored chart, for training and inference.

The optimizer constrains its final leg with two row families from
``4dTrajectory/optimization/approach_constraints``: the LPV angular corridor
``|cross-track| ≤ k · halfwidth(d)`` and the glidepath window
``h ∈ [h_GP(d) − below, h_GP(d) + above]``, with ``d`` the along-course distance back from
the landing threshold point.  Under the ``enu`` chart the same rows reduce to three
closed forms, because ``FlightSeries.target_chart`` is the origin and the target altitude
is the LTP + TCH aim point (``flight_scenarios.runway_target``):

    d          = −(e·cos ψ + n·sin ψ)          ψ = the runway course, math-ENU (scenario.target.psi)
    xt         =   e·sin ψ − n·cos ψ            + = RIGHT of the inbound course (the readouts' convention,
                                                and approach_constraints.lateral.fac_cross_track's)
    u_GP(d)    =   d · tan GPA                  chart height of the glidepath
    hw(d)      =   cw · (d + d_GARP) / d_GARP   flight_scenarios.fas_geometry, one source with the optimizer

Everything here is torch so one implementation serves the bounded output layer
(``outputs.state.model.StateOutputLayer``), the training-time penalty
(``objective.procedure_loss``) and the inference-time projection (``forecast``); NumPy callers wrap their arrays.

**Which rows are "on the final".**  The measured data (docs/2026-09-04_procedure_constraints_design.zh.md)
say 15–62 % of flights join the final INSIDE the FAF, so the FAF distance is not the
gate.  Two gates exist:

* ``on-final`` — a row is on the final when it sits inside the full-scale cone
  (``MEMBERSHIP_K``, floored at ``MEMBERSHIP_FLOOR_M`` near the threshold) AND the
  predicted PATH runs within ``ALIGNMENT_MAX_DEG`` of the course there.  The direction is
  read from the predicted positions (:func:`position_direction`), never from the velocity
  channels: the state objective supervises positions only, so the velocity channels are
  free outputs the network could steer to switch the gate off, and at inference they are
  overwritten by the position-derived velocity anyway.  Deployable (it reads only the
  prediction); soft (sigmoids) inside the network so the bounded output stays
  continuous, hard for the projection and the readouts.
* ``faf`` — every row with ``d ≤ d_FAF`` (the optimizer's vertical-gate convention);
  the ablation that shows why the self-gate is needed.

The TRUTH gate used by the penalty and the readouts is the measurement's definition:
the rows from which the observed track stays inside the ``K_MARGIN`` cone to the
threshold, beyond ``NEAR_THRESHOLD_M``.

**Where a trajectory LANDS** is one rule as well (:func:`threshold_crossing_index`): the
first row that is at or past the threshold plane AND on the final there.  Both consumers
that need an arrival point read it — ``forecast.cut_at_threshold_crossing`` (which row a
record ends on) and ``outputs/control/constraints/trombone.py`` (how much path the reference rollout
still intends to fly) — because the plane alone is crossed on a downwind, abeam.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from flight_scenarios.fas_geometry import course_halfwidth_m, fas_course_geometry

if TYPE_CHECKING:
    from ts_transformer.dataset import FlightSeries

# Mirrors of the optimizer's constraint defaults. ``4dTrajectory/optimization`` is not on
# this package's import path, so they cannot be imported here;
# tests/test_final_approach_geometry.py imports both sides and asserts they are equal.
K_MARGIN = 0.5                # approach_constraints.lateral.DEFAULT_K_MARGIN
GLIDEPATH_BELOW_M = 60.0      # approach_constraints.segments.DEFAULT_GLIDEPATH_BELOW_M
GLIDEPATH_ABOVE_M = 120.0     # approach_constraints.segments.DEFAULT_GLIDEPATH_ABOVE_M
ALIGNMENT_MAX_DEG = 30.0      # approach_constraints.segments.STANDARD_INTERCEPT_MAX_DEG

MEMBERSHIP_K = 1.0            # "on the final" = inside the FULL-SCALE cone …
MEMBERSHIP_FLOOR_M = 500.0    # … or within 500 m of the centreline: the cone's HALF-width is 107 m
                              # at the threshold, and a forecast biased by the measured 250–350 m
                              # NW translation must still count as on the final to be bound at all
NEAR_THRESHOLD_M = 300.0      # the truth gate ignores the last 300 m (ground-effect ADS-B rows)
LATERAL_SOFTNESS = 0.05       # soft-gate width across the cone edge, as a fraction of the half-width:
                              # a row inside the k=0.5 design corridor is fully bound (σ(10) ≈ 1)
ALIGNMENT_SOFTNESS = 0.02     # soft-gate width, in cos(heading error) (≈ ±2° about 30°)
_STEP_FLOOR_M = 1.0           # a shorter position step has no direction to align

FAS = fas_course_geometry()   # runway length unknown → the 9023 ft floor, as the optimizer

# The per-flight context every consumer reads: the runway course (math-ENU, the direction
# of travel on final) and tan of the coded glidepath. Built by
# ``dataset.final_approach_arrays``; never a model input. (A third key, the FAF distance,
# served the deleted `faf` gate — review §5, 2026-09-09.)
FINAL_APPROACH_KEYS = ("runway_heading_rad", "glidepath_tan")
_COS_ALIGNMENT = math.cos(math.radians(ALIGNMENT_MAX_DEG))


def runway_axes(
    e: torch.Tensor, n: torch.Tensor, psi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(d, xt)`` of chart positions ``[B, N]`` for runway courses ``psi`` ``[B]``."""
    ue, un = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    return -(e * ue + n * un), e * un - n * ue


def chart_from_axes(
    d: torch.Tensor, xt: torch.Tensor, psi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`runway_axes`: ``(e, n)`` from ``(d, xt)``."""
    ue, un = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    return -d * ue + xt * un, -d * un - xt * ue


def corridor_halfwidth(d: torch.Tensor) -> torch.Tensor:
    """Full-scale LPV half-width at ``d`` back from the threshold (the LTP width past it)."""
    return course_halfwidth_m(d.clamp(min=0.0), FAS)


def corridor_halfwidth_slope(d: torch.Tensor) -> torch.Tensor:
    """``d corridor_halfwidth / d d``: the cone's closing rate per metre flown toward the
    threshold, zero at and past it where the half-width is the LTP width."""
    return (FAS.course_width_m / FAS.d_garp_m) * (d > 0.0).to(d.dtype)


def glidepath_height(d: torch.Tensor, tan_gpa: torch.Tensor) -> torch.Tensor:
    """Chart height of the glidepath at ``d`` (zero at and past the threshold)."""
    return d.clamp(min=0.0) * tan_gpa.unsqueeze(-1)


def bounded_cross_track(xt: torch.Tensor, halfwidth: torch.Tensor, k: float = K_MARGIN) -> torch.Tensor:
    """Smoothly saturate ``xt`` inside ``±k·halfwidth``; identity-like near the centreline."""
    bound = k * halfwidth
    return bound * torch.tanh(xt / bound)


def bounded_height_residual(
    residual: torch.Tensor,
    below: float = GLIDEPATH_BELOW_M,
    above: float = GLIDEPATH_ABOVE_M,
) -> torch.Tensor:
    """Saturate a height residual from the glidepath inside ``[−below, +above]``.

    Asymmetric on purpose (below the glidepath is the dangerous side, so the optimizer's
    window is tighter there); slope one at zero on both sides, so an on-path prediction
    is left alone.
    """
    return torch.where(
        residual < 0.0,
        -below * torch.tanh(-residual / below),
        above * torch.tanh(residual / above),
    )


def position_direction(
    e: torch.Tensor, n: torch.Tensor, anchor_e: torch.Tensor, anchor_n: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The predicted path's direction at every row, from its positions ``[B, N]``.

    Central difference (the next row minus the previous, the anchor ``[B]`` standing in
    before the first row, a backward difference on the last), so one noisy step does not
    flip the gate.  Returned as chart step vectors; only the direction is meaningful.
    """
    prev_e = torch.cat((anchor_e.unsqueeze(-1), e[:, :-1]), dim=1)
    prev_n = torch.cat((anchor_n.unsqueeze(-1), n[:, :-1]), dim=1)
    next_e = torch.cat((e[:, 1:], e[:, -1:]), dim=1)
    next_n = torch.cat((n[:, 1:], n[:, -1:]), dim=1)
    return next_e - prev_e, next_n - prev_n


def alignment_cosine(v_e: torch.Tensor, v_n: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """cos of the angle between a chart direction ``[B, N]`` and the runway course."""
    ue, un = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    # Clamp BEFORE the square root: sqrt has an infinite gradient at zero.
    length = torch.sqrt((v_e * v_e + v_n * v_n).clamp(min=_STEP_FLOOR_M**2))
    return (v_e * ue + v_n * un) / length


def membership_halfwidth(d: torch.Tensor) -> torch.Tensor:
    """How far off the centreline a row may sit and still be "on the final"."""
    return (MEMBERSHIP_K * corridor_halfwidth(d)).clamp(min=MEMBERSHIP_FLOOR_M)


def soft_aligned(cos_align: torch.Tensor) -> torch.Tensor:
    """Alignment membership in ``[0, 1]``: is this direction down the final approach course?

    Half of ``on-final`` (the cone is the other half), named because a module can need the
    alignment WITHOUT the cone: ``outputs/control/constraints/trombone.py`` hands the command back
    as soon as the path is lined up, wide of the cone or not, and has to read the same 30°
    the gate reads rather than a second copy of it.
    """
    return torch.sigmoid((cos_align - _COS_ALIGNMENT) / ALIGNMENT_SOFTNESS)


def hard_aligned(cos_align: torch.Tensor) -> torch.Tensor:
    return cos_align >= _COS_ALIGNMENT


def soft_on_final(d: torch.Tensor, xt: torch.Tensor, cos_align: torch.Tensor) -> torch.Tensor:
    """Membership in ``[0, 1]``: inside the membership cone and aligned with the course."""
    halfwidth = membership_halfwidth(d)
    lateral = torch.sigmoid((halfwidth - xt.abs()) / (LATERAL_SOFTNESS * halfwidth))
    return lateral * soft_aligned(cos_align)


def hard_on_final(d: torch.Tensor, xt: torch.Tensor, cos_align: torch.Tensor) -> torch.Tensor:
    return (xt.abs() <= membership_halfwidth(d)) & hard_aligned(cos_align)


def threshold_crossing_mask(
    d: torch.Tensor, xt: torch.Tensor, cos_align: torch.Tensor
) -> torch.Tensor:
    """Rows ``[B, N]`` that cross the landing threshold ON THE FINAL.

    At or past the threshold plane (``d ≤ 0``) AND on the final there — the ``on-final``
    gate itself, so the cone, its floor and the 30° alignment are read from the gate rather
    than restated beside it.

    ``d ≤ 0`` ALONE is not the crossing, and the difference is not academic (2026-09-09). A
    vectored flight's downwind runs parallel to the runway course and opposite it, several
    kilometres abeam, and passes ``d = 0`` out there — so the plane-only rule fired ABEAM.
    Measured on ``l3e_path_stretch_20260908/L3e_stack_p60s_pred_val``, 96.5 % of the vectored
    cuts it made lay more than 1 km from the threshold (median ``|xt|`` 8.7 km, a median
    1.75 km above it), while the straight-in cuts were clean (``|xt|`` p95 50 m). The cone is
    what rejects the abeam pass; the alignment is what rejects a plane crossed on a heading
    that is not the final's.

    ``cos_align`` is the caller's: ``forecast`` takes the central difference of the predicted
    POSITIONS (:func:`position_direction`, the gate's own rule — the velocity channels are
    free outputs), the trombone the chord flown INTO each boundary. Two costs, both visible
    rather than silent (the ``truncatedAtThreshold`` flag, and ``predict``'s cut/whole
    count). The central difference folds the flying just AFTER a row into its direction, so a
    rollout that turns hard at the threshold can fail the 30° on the very row that crossed
    it; and ``MEMBERSHIP_FLOOR_M`` is now what decides whether a record is cut at all, so a
    straight-in carrying a lateral bias wider than 500 m reads as never having landed.
    """
    return (d <= 0.0) & hard_on_final(d, xt, cos_align)


def threshold_crossing_index(
    d: torch.Tensor, xt: torch.Tensor, cos_align: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The FIRST crossing run: ``(first row, one past its last row, crossed at all)``, ``[B]``.

    "First" so a trajectory that overshoots, wanders and comes back is read on its real
    arrival rather than on a later pass; the run is what a caller refines INSIDE
    (``forecast.cut_at_threshold_crossing`` takes the closest horizontal approach within it,
    ``outputs/control/constraints/trombone.py`` interpolates ``d = 0`` inside the crossing segment).

    Where ``crossed`` is false the trajectory never reached the final at all, and ``first``
    and ``end`` mean nothing: the caller must say so rather than cut somewhere, which would
    invent an arrival that never happened.
    """
    crossing = threshold_crossing_mask(d, xt, cos_align)
    crossed = crossing.any(dim=-1)
    first = crossing.to(torch.int64).argmax(dim=-1)
    rows = torch.arange(crossing.shape[-1], device=crossing.device)
    after = ~crossing & (rows >= first.unsqueeze(-1))
    end = torch.where(
        after.any(dim=-1),
        after.to(torch.int64).argmax(dim=-1),
        torch.full_like(first, crossing.shape[-1]),
    )
    return first, end, crossed


def membership(
    *,
    d: torch.Tensor,
    xt: torch.Tensor,
    cos_align: torch.Tensor,
    hard: bool,
) -> torch.Tensor:
    """The `on-final` gate, soft (``[0, 1]`` floats) or hard (bools) — the one gate the
    corridor-bounded output layer and the post-hoc projection apply."""
    return hard_on_final(d, xt, cos_align) if hard else soft_on_final(d, xt, cos_align)


def stays_mask(inside: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Rows from which every LATER valid row is inside: the "established from here" tail."""
    bad = (valid & ~inside).to(torch.int64)
    later_bad = torch.flip(torch.cumsum(torch.flip(bad, dims=(-1,)), dim=-1), dims=(-1,))
    return (later_bad == 0) & valid


def truth_final_gate(
    d: torch.Tensor, xt: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """The measurement's gate on OBSERVED rows: inside the k-cone from here to the end,
    beyond the last ``NEAR_THRESHOLD_M`` (those rows neither open nor close the gate)."""
    near = d <= NEAR_THRESHOLD_M
    inside = (xt.abs() <= K_MARGIN * corridor_halfwidth(d)) | near
    return stays_mask(inside, valid) & ~near


def corridor_violations(
    d: torch.Tensor, xt: torch.Tensor, u: torch.Tensor, tan_gpa: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Metres outside the ``K_MARGIN`` cone and outside the glidepath window, per row."""
    lateral = torch.relu(xt.abs() - K_MARGIN * corridor_halfwidth(d))
    glidepath = glidepath_height(d, tan_gpa)
    vertical = torch.relu((glidepath - GLIDEPATH_BELOW_M) - u) + torch.relu(u - (glidepath + GLIDEPATH_ABOVE_M))
    return lateral, vertical


def bound_to_final(
    *,
    d: torch.Tensor,
    xt: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    tan_gpa: torch.Tensor,
    hard: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend ``(xt, u)`` toward their bounded versions by the membership ``weight``.

    Soft: tanh saturation (the trained output layer).  Hard: clamp (the projection).
    """
    halfwidth = corridor_halfwidth(d)
    glidepath = glidepath_height(d, tan_gpa)
    residual = u - glidepath
    if hard:
        bound = K_MARGIN * halfwidth
        xt_bounded = torch.maximum(torch.minimum(xt, bound), -bound)
        residual_bounded = residual.clamp(min=-GLIDEPATH_BELOW_M, max=GLIDEPATH_ABOVE_M)
    else:
        xt_bounded = bounded_cross_track(xt, halfwidth)
        residual_bounded = bounded_height_residual(residual)
    return (
        xt + weight * (xt_bounded - xt),
        u + weight * (glidepath + residual_bounded - u),
    )


def final_approach_arrays(series: FlightSeries) -> dict[str, np.ndarray]:
    """``FINAL_APPROACH_KEYS`` for one flight: the runway course (math-ENU, the direction
    of travel on final) and tan of the coded glidepath. Needs no procedure document — the
    one gate (`on-final`) is geometry about the threshold."""
    target = series.scenario.target
    rows = {
        # The rollout frame rotation and the runway heading coincide only for the
        # runway-aligned coordinate frame.  Keep the terminal-loss reference separate
        # so ENU rollouts are decomposed along/across the actual runway, not east/north.
        "runway_heading_rad": np.array(float(target.psi), dtype=np.float64),
        # The target's gamma is the coded glidepath DESCENT (negative); the chart height of
        # the glidepath at distance d back from the threshold is d · tan(GPA).
        "glidepath_tan": np.array(math.tan(-float(target.gamma)), dtype=np.float64),
    }
    assert tuple(rows) == FINAL_APPROACH_KEYS
    return rows


def probe_final_approach(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """One representative final-approach context for shape/throughput probes."""
    rows = {
        "runway_heading_rad": 0.0,
        "glidepath_tan": math.tan(math.radians(3.0)),
    }
    return {
        name: torch.full((batch_size,), value, dtype=torch.float32, device=device)
        for name, value in rows.items()
    }

