"""Translate a canonical :class:`ProcedureConstraint` into constraints-package ``SegmentSpec``s.

This bridge lives in the backend on purpose: it depends on *both* the request-side
:mod:`aeroviz_backend.procedure_constraint` and the optimizer-side ``approach_constraints`` package
(``4dTrajectory/optimization/approach_constraints``). The optimizer itself depends only on ``approach_constraints``,
so the dependency direction stays clean (backend → {constraints, optimizer}; optimizer →
constraints; constraints → nothing).

Geometry, all in the target-anchored ``(n, e)`` metric frame (origin = the LTP = the optimizer
target), matching the optimizer's Normalized decision state:

- **Final (LPV)** segment(s): from the FAF to the threshold. The FAS data block is not in the
  ProcedureConstraint, so it is derived: ``LTP = (0,0)``; ``FPAP`` is ``max(runway length, 9023 ft)``
  past the LTP along the inbound course (runway length unknown ⇒ the 9023 ft floor); ``GARP`` is
  1000 ft beyond ``FPAP``; ``course_width = max(350 ft, tan(1.5°)·d_GARP)`` (8260.58D Formula
  3-1-1). The glidepath uses the procedure's coded GPA/TCH (defaults 3.0°/50 ft).
- **Non-final** legs: an RNP box corridor (default 1.0 NM), a step-down floor from the END
  waypoint's at-or-above altitude, and a descent-gradient cap.

(At-or-below / block-altitude ceilings are a documented follow-up — only floors are enforced.)
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from aeroviz_backend import paths  # noqa: F401  (puts the optimization dir on sys.path)
from geokit import FT_M, NM_M

import approach_constraints as ac
from approach_constraints import geometry as _geo
from aeroviz_backend.procedure_constraint import ProcedureConstraint
# The FAS cone (FPAP/GARP distances, course width) is defined ONCE in the data→modeling
# seam, because the learned model's final-approach corridor evaluates the same cone.
from flight_scenarios.fas_geometry import fas_course_geometry

_DEFAULT_RNP_NM = 1.0          # RNP APCH initial/intermediate lateral accuracy
_DEFAULT_GPA_DEG = 3.0
_DEFAULT_TCH_M = 50.0 * FT_M
_INTERMEDIATE_CAP_DEG = 3.0    # ≈ 318 ft/NM
_INITIAL_CAP_DEG = 4.7         # ≈ 500 ft/NM
# Defaults sourced from the approach_constraints package (the single source of truth):
# glidepath vertical-window half-tolerances, the lateral design margin, and the FAA standard
# PFAF intercept limit (data-validation check).
_DEFAULT_BELOW_M = ac.DEFAULT_GLIDEPATH_BELOW_M
_DEFAULT_ABOVE_M = ac.DEFAULT_GLIDEPATH_ABOVE_M
_DEFAULT_K_MARGIN = ac.DEFAULT_K_MARGIN
_MAX_INTERCEPT_DEG = ac.STANDARD_INTERCEPT_MAX_DEG


def _floor_alt_m(waypoint) -> float | None:
    """The at-or-above floor (m MSL) coded at a waypoint, or None."""
    alt = waypoint.altitude
    if alt is None or alt.min_ft_msl is None:
        return None
    return alt.min_ft_msl * FT_M


def _faf_index(waypoints) -> int:
    """Index of the FAF waypoint; falls back to the second-to-last (last leg = final)."""
    for i, wp in enumerate(waypoints):
        if "FAF" in wp.role.upper():
            return i
    return len(waypoints) - 2


def _lpv_spec(faf_ne: np.ndarray, target_alt_m: float, pc: ProcedureConstraint,
             below_m: float, above_m: float, prefaf_floor_m: float | None):
    ltp = np.array([0.0, 0.0])
    norm = float(np.hypot(faf_ne[0], faf_ne[1]))
    # inbound unit vector: from the FAF toward the LTP (= -faf_ne / |faf_ne|)
    inbound = (-faf_ne / norm) if norm > 0.0 else np.array([1.0, 0.0])
    # Runway length is not carried by the runway target, so the 9023 ft FPAP floor applies.
    fas = fas_course_geometry()
    fpap = fas.d_fpap_m * inbound                      # FPAP is past the LTP, inbound direction
    garp = fas.d_garp_m * inbound                      # |LTP → GARP|
    course_width = fas.course_width_m
    gpa = pc.glidepath.angle_deg if pc.glidepath else _DEFAULT_GPA_DEG
    tch_m = (
        pc.glidepath.tch_ft * FT_M
        if (pc.glidepath and pc.glidepath.tch_ft is not None)
        else _DEFAULT_TCH_M
    )
    # The optimizer's terminal node is pinned to the target, so the glidepath must pass through
    # the target altitude AT the threshold (d = 0): glidepath(0) = tdze + tch = target_alt.
    # Hence tdze = target_alt - tch (the target altitude is taken as the LTP+TCH aim point).
    # d_faf_m / prefaf_floor_m drive the flexible-join VERTICAL GATE (approach_constraints
    # README §4b): the glidepath window binds from the FAF toward the runway; upstream of the
    # FAF the published FAF crossing minimum applies. The GARP/LTP/FAF are collinear by
    # construction here, so the FAF's along-course distance back from the LTP is |faf_ne|.
    return ac.LpvFinalSpec(
        ltp_ne=ltp,
        fpap_ne=fpap,
        garp_ne=garp,
        course_width_m=course_width,
        tdze_m=target_alt_m - tch_m,
        tch_m=tch_m,
        gpa_deg=gpa,
        below_m=below_m,
        above_m=above_m,
        d_faf_m=norm if norm > 0.0 else None,
        prefaf_floor_m=prefaf_floor_m,
    )


def build_constraint_segments(
    pc: ProcedureConstraint,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float,
    *,
    rnp_nm: float = _DEFAULT_RNP_NM,
    k_margin: float = _DEFAULT_K_MARGIN,
    glidepath_below_m: float = _DEFAULT_BELOW_M,
    glidepath_above_m: float = _DEFAULT_ABOVE_M,
) -> list:
    """Build the ``SegmentSpec`` list from a ProcedureConstraint anchored at the target (LTP).

    Fixes are in the target ``(n,e)`` frame — the frame the optimizer solves in (it validates the
    anchoring at construction). The optimizer maps one PHASE per segment and prepends an
    unconstrained start→first-fix transition phase when the start is away from the first fix, so
    the start does NOT need to coincide with the procedure. Returns ``[]`` if the procedure has
    fewer than two waypoints.
    """
    wps = pc.waypoints
    if len(wps) < 2:
        return []
    frame = ac.TargetFrame(target_lat_deg, target_lon_deg)
    ne = [frame.to_ne(wp.lat_deg, wp.lon_deg) for wp in wps]

    faf = _faf_index(wps)
    dists = [wp.distance_from_start_m for wp in wps]
    rnp_half = rnp_nm * NM_M

    # (#6) Derive the FAS geometry ONCE, from the FAF, and share it across every final leg — the
    # FAS (FPAP/GARP/course width/glidepath) is one record per approach, not per leg. The FAF's
    # own at-or-above altitude is the published pre-FAF floor for the flexible-join gate.
    final_lpv = _lpv_spec(
        np.asarray(ne[faf], dtype=float), target_alt_m, pc, glidepath_below_m, glidepath_above_m,
        prefaf_floor_m=_floor_alt_m(wps[faf]),
    )

    # (#5) Data-validation: warn if the intermediate→final intercept at the PFAF exceeds 30°.
    if faf >= 1:
        intercept = _geo.intercept_angle_deg(
            _geo.course_bearing(ne[faf - 1], ne[faf]),   # intermediate (IF→FAF) track
            _geo.course_bearing(ne[faf], ne[-1]),        # final approach course (FAF→threshold)
        )
        if float(intercept) > _MAX_INTERCEPT_DEG:
            warnings.warn(
                f"PFAF intercept angle {float(intercept):.0f}° exceeds {_MAX_INTERCEPT_DEG:.0f}° "
                f"at '{wps[faf].ident}' — the coded procedure may be irregular.",
                stacklevel=2,
            )

    segments: list = []
    for j in range(len(wps) - 1):
        a_ne, b_ne = ne[j], ne[j + 1]
        if j >= faf:  # FAF onward -> the LPV final segment(s); share the one FAS spec
            seg = ac.SegmentSpec(
                ac.SegmentKind.FINAL_LPV,
                start_ne=a_ne,
                end_ne=b_ne,
                start_ident=wps[j].ident,
                end_ident=wps[j + 1].ident,
                k_margin=k_margin,
                lpv=final_lpv,
            )
        else:
            kind = (
                ac.SegmentKind.INTERMEDIATE
                if j == faf - 1
                else ac.SegmentKind.INITIAL
            )
            floor = _floor_alt_m(wps[j + 1])
            # "at or above the END waypoint's floor over the whole leg": the step-down covers
            # the leg's coded along-track length.
            step_downs = (
                [ac.StepDown(s_from_start_m=dists[j + 1] - dists[j], min_alt_m=floor)]
                if floor is not None
                else []
            )
            cap = _INTERMEDIATE_CAP_DEG if kind is ac.SegmentKind.INTERMEDIATE else _INITIAL_CAP_DEG
            seg = ac.SegmentSpec(
                kind,
                start_ne=a_ne,
                end_ne=b_ne,
                start_ident=wps[j].ident,
                end_ident=wps[j + 1].ident,
                k_margin=k_margin,
                halfwidth_m=rnp_half,
                step_downs=step_downs,
                max_descent_deg=cap,
            )
        segments.append(seg)
    return segments
