"""FAS (final approach segment) lateral geometry — one definition for every consumer.

The LPV final's lateral containment is a cone that opens from the GARP toward the FAF
(FAA Order 8260.58D Formula 3-1-1): full-scale half-width ``course_width`` at the
landing threshold point (LTP), growing linearly with distance from the GARP.  The FAS
data block is not in the procedure documents this project reads, so the geometry is
derived from the threshold alone:

    d_FPAP        = max(runway length, 9023 ft)        past the LTP, along the course
    d_GARP        = d_FPAP + 1000 ft
    course_width  = max(350 ft, tan(1.5°) · d_GARP)
    halfwidth(d)  = course_width · (d + d_GARP) / d_GARP     d = distance back from the LTP

Two consumers share it and must never disagree: the optimizer's constraint bridge
(``aeroviz_backend.procedure_segments._lpv_spec`` → ``approach_constraints.LpvFinalSpec``,
where ``lateral.lpv_course_halfwidth`` evaluates the same cone from the GARP/LTP points)
and the learned model's final-approach corridor
(``4dTrajectory/ts_transformer/final_approach_geometry.py``).  This module lives in the
data→modeling seam because both of them import it and neither may import the other.
Runway length is not carried by the runway target, so every caller uses the 9023 ft floor
today; the parameter exists so that a caller that knows the length asks for it explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from geokit import FT_M

FPAP_FLOOR_M = 9023.0 * FT_M          # LTP → FPAP = max(runway length, 9023 ft)
GARP_BEYOND_FPAP_M = 1000.0 * FT_M    # FPAP → GARP
COURSE_WIDTH_FLOOR_M = 350.0 * FT_M   # course half-width at the LTP, never narrower
COURSE_SPLAY_DEG = 1.5


@dataclass(frozen=True)
class FasCourseGeometry:
    """Distances along the inbound course past the LTP, and the LTP course half-width."""

    d_fpap_m: float
    d_garp_m: float
    course_width_m: float


def fas_course_geometry(runway_length_m: float | None = None) -> FasCourseGeometry:
    """The FAS lateral geometry for a runway of ``runway_length_m`` (``None`` = unknown)."""
    d_fpap = max(float(runway_length_m or 0.0), FPAP_FLOOR_M)
    d_garp = d_fpap + GARP_BEYOND_FPAP_M
    course_width = max(
        COURSE_WIDTH_FLOOR_M, math.tan(math.radians(COURSE_SPLAY_DEG)) * d_garp
    )
    return FasCourseGeometry(d_fpap_m=d_fpap, d_garp_m=d_garp, course_width_m=course_width)


def course_halfwidth_m(distance_to_ltp_m, geometry: FasCourseGeometry):
    """Full-scale lateral half-width at ``distance_to_ltp_m`` back from the LTP.

    Plain arithmetic on purpose: ``distance_to_ltp_m`` may be a float, a NumPy array or a
    torch tensor, and the geometry folds in as Python floats.  Distances past the LTP
    (negative) narrow the cone further; callers that want the LTP width as a floor clamp
    the distance at zero before calling.
    """
    return geometry.course_width_m * (distance_to_ltp_m + geometry.d_garp_m) / geometry.d_garp_m
