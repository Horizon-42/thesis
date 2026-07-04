"""Lateral containment — the corridor each state node must stay inside.

Two flavours (design-doc §3/§4; LPV details in ``lpv_final_segment.en.html``):

* **Box corridor** (feeder / initial / intermediate) — a constant half-width ``±RNP`` band
  around the leg centerline.
* **LPV angular corridor** (final) — a *converging cone* anchored at the GARP: widest at the
  PFAF, narrowing to the ``course_width`` (±350 ft) at the threshold (LTP).

The LPV corridor is a cone — apex at the GARP, narrowing to ``course_width`` at the LTP, wide
at the PFAF (read distances from the GARP apex on the right: ``dL`` = GARP→LTP, ``dP`` =
GARP→node; the farther down the cone, the wider it is)::

    PFAF (wide end)                                 LTP=(0,0)        GARP (apex)
         ╲       half-width = course_width·dP/dL         │             •
          ╲  ──────────────────────────────────────     │  ─────────╱      ← corridor edge
           ╲              • node (n,e)                    │          ╱
    FAC ────╲─────────────┼──────────────────────────────┼─────────•─────   centerline (GARP→LTP)
            ╱            e_xt = cross-track (sideways)     │          ╲
           ╱  ──────────────────────────────────────     │  ─────────╲      ← corridor edge
          ╱                                               │            ╲
          |◄──────────────── dP ────────────────────────►|◄─── dL ────►|

Glossary
--------
Aviation fixes / terms:
    LPV   Localizer Performance with Vertical Guidance (带垂直引导的航向性能) — the precise
          GPS/WAAS approach; its final lateral corridor is an angular cone, not a fixed band.
    RNP   Required Navigation Performance (所需导航性能) — lateral accuracy spec for the
          non-final legs. "RNP 1.0" = stay within ±1.0 NM of the centerline (a constant band).
    LTP   Landing Threshold Point (着陆入口点) — the runway threshold; the ORIGIN (0,0) of the
          (n,e) frame. The cone is narrowest here.
    GARP  GNSS Azimuth Reference Point (GNSS方位基准点) — ~1000 ft past the far runway end; the
          APEX the angular corridor fans out from (think "the localizer antenna").
    PFAF  Precise Final Approach Fix (精密最终进近定位点) — where the final segment begins (the
          wide end of the usable corridor).
    FAC   Final Approach Course (最终进近航道) — the final centerline = the straight line
          through the GARP and the LTP. "cross-track to FAC" = perpendicular distance from it.
    course_width  the corridor half-width AT the threshold (≈350 ft ≈ 106.7 m; 8260.58D Formula
          3-1-1) — the narrow end of the cone. Stored as ``lpv.course_width_m``.

Geometry:
    node (n,e)   one trajectory point; ``n`` = metres north of the LTP, ``e`` = metres east.
    centerline   the desired path line: ``A→B`` (box) or ``GARP→LTP`` (final).
    along-track  distance measured ALONG the centerline.
    cross-track (e_xt)  SIGNED perpendicular (sideways) distance from the centerline; the sign
          says which side. The corridor is enforced as two SMOOTH rows (``e_xt − margin`` and
          ``−e_xt − margin``), not ``|e_xt| − margin`` — the abs kink at the centerline breaks the
          gradient-based NLP.
    half-width   half the corridor width = how far sideways the node may stray on each side.
    dP = d_GARP(node)  along-track distance from the GARP to the node (how far down the cone).
    dL = d_GARP(LTP)   distance GARP→LTP = ‖LTP − GARP‖ (the reference length; ‖·‖ = vector
          length / Euclidean norm).

Software:
    *_violation   returns ``g`` with ``g ≤ 0 ⇔ satisfied`` (``g`` = metres over when > 0). This
          IS the IPOPT inequality form, so the optimizer bound is simply ``g ≤ 0``.
    k             design-margin fraction (0.5 ⇒ use only half the legal corridor — stay off the
          containment edge).
    halfwidth_m   box-corridor half-width in metres (e.g. 1×RNP = 1852 m or 2×RNP = 3704 m).
    scalar components / backend-agnostic   ``n``/``e`` arrive as separate values (a NumPy array
          OR a CasADi symbol); combine with ``+ − ×`` and ``mathx.*`` only, so the SAME function
          serves the NumPy tests and the CasADi optimizer.
    mathx.*   NumPy/CasADi op dispatch (``mathx.fabs`` = an ``abs`` that works on both backends).
    geo.along_track / geo.cross_track / geo.leg_unit   geometry primitives (geometry.py);
          ``leg_unit`` returns ``(u_n, u_e, length)``.
    lpv.garp_ne / lpv.ltp_ne / lpv.course_width_m   fields of the LpvFinalSpec (the FAS data
          block): the GARP's (n,e), the LTP's (n,e), and the course width in metres.
"""

from __future__ import annotations

from . import geometry as geo

# Lateral design-margin fraction of the corridor half-width (0.5 ⇒ use only half the legal
# corridor — stay off the containment edge). SINGLE source: SegmentSpec.k_margin and the
# backend's ``build_constraint_segments`` default to it too.
DEFAULT_K_MARGIN = 0.5


def box_corridor_violation(n, e, A, B, halfwidth_m: float, k: float = DEFAULT_K_MARGIN):
    """RNP box corridor: keep ``|cross-track| ≤ k · halfwidth``, as TWO smooth rows.

    Returns the pair ``(e_xt − margin, −e_xt − margin)`` (``margin = k · halfwidth``); both ``≤ 0``
    iff inside. We do **not** return ``|e_xt| − margin``: the absolute value has a
    non-differentiable kink at ``e_xt = 0`` — exactly where an on-centerline optimum sits — which
    makes the gradient-based NLP (IPOPT) fail to converge. Two linear rows are smooth everywhere
    and equivalent (their max is ``|e_xt| − margin``).

    ``halfwidth_m`` is the segment's containment half-width (``2·RNP`` or ``1·RNP`` — design-doc
    §3.2/§8); ``k`` is the design-margin fraction.

    Worked check: A=(0,0), B=(10,0), node=(3,7), halfwidth=20, k=0.5 → e_xt=−7, margin=10 →
    ``(e_xt−margin, −e_xt−margin) = (−17, −3)`` → max −3 (inside).
    """
    e_xt = geo.cross_track(n, e, A, B)
    margin = k * halfwidth_m
    return e_xt - margin, -e_xt - margin


def lpv_course_halfwidth(n, e, lpv):
    """⑥ — LPV full-scale lateral half-width at node ``(n, e)`` (metres) — the converging cone.

    The guidance sector originates at the GARP; its full-scale half-width grows linearly with
    distance from the GARP (8260.58D Formula 3-1-1, design-doc §3.1a)::

        d_GARP(node) = along-course distance from the GARP to the node
                     = along_track(n, e, GARP, LTP)            # projection onto the GARP→LTP axis
        d_GARP(LTP)  = ‖LTP − GARP‖
        halfwidth    = course_width · d_GARP(node) / d_GARP(LTP)

    So ``halfwidth = course_width`` at the LTP, larger toward the PFAF. ``lpv`` carries
    ``garp_ne``, ``ltp_ne``, ``course_width_m``.

    Hint: ``dP = geo.along_track(n, e, lpv.garp_ne, lpv.ltp_ne)``;
    ``_, _, dL = geo.leg_unit(lpv.garp_ne, lpv.ltp_ne)``; ``return lpv.course_width_m * dP / dL``.
    """
    dP = geo.along_track(n, e, lpv.garp_ne, lpv.ltp_ne)
    _, _, dL = geo.leg_unit(lpv.garp_ne, lpv.ltp_ne)
    return lpv.course_width_m * dP / dL


def fix_passage_violation(n, e, fix_ne, tolerance_m: float):
    """⑪ — pass a fix within ``tolerance_m``: a SMOOTH disc constraint, metre-scaled.

    The node must lie inside the disc of radius ``tolerance_m`` around ``fix_ne`` (the pre-FAF
    fix in the optimizer's wiring; the tolerance comes from the procedure — the leg's RNP
    containment half-width scaled by the design margin, never a hardcoded number). Enforced in
    the squared
    form — the Euclidean distance has a non-differentiable kink AT the fix, the squared form is
    smooth everywhere — and rescaled by ``1/(2·tol)`` so the violation reads in METRES near the
    boundary (first order: ``(d² − tol²)/(2·tol) ≈ d − tol``), keeping the report's metre family
    honest::

        g = ((n − fix_n)² + (e − fix_e)² − tol²) / (2·tol)     # ≤ 0  ⇔  inside the disc
    """
    dn = n - fix_ne[0]
    de = e - fix_ne[1]
    return (dn * dn + de * de - tolerance_m * tolerance_m) / (2.0 * tolerance_m)


def fac_cross_track(n, e, lpv):
    """Signed cross-track distance (metres) of node ``(n, e)`` from the final approach course
    (the GARP→LTP line). Zero ⇔ the node is ON the course — used as a LINEAR equality row for
    the flexible FAC join (§4b of the README). One source: the same axis the corridor and the
    glidepath distance use."""
    return geo.cross_track(n, e, lpv.garp_ne, lpv.ltp_ne)


def fac_distance_to_ltp(n, e, lpv):
    """Along-course distance (metres) of node ``(n, e)`` BACK from the LTP, measured on the
    GARP→LTP axis (0 at the threshold, + toward the PFAF/FAF). The single distance measure for
    the glidepath window, the vertical gate and the FAC join window."""
    _, _, d_garp_ltp = geo.leg_unit(lpv.garp_ne, lpv.ltp_ne)
    return geo.along_track(n, e, lpv.garp_ne, lpv.ltp_ne) - d_garp_ltp


def fac_join_window_violation(n, e, lpv, d_faf_m: float, max_offset_m: float,
                              min_offset_m: float = 0.0):
    """⑫ — hold the FAC join node inside the along-track window
    ``[d_FAF + min_offset, d_FAF + max_offset]`` — strictly UPSTREAM of the FAF.

    The intersection with the final approach course must happen EARLY: at least ``min_offset_m``
    before the FAF (the optimizer sets that to 1/5 of the final leg's length — established on
    the course with margin before the FAF), and no earlier than ``max_offset_m`` before it.
    Never between the FAF and the runway. Returns the pair (both ≤ 0 ⇔ inside)::

        ((d_FAF + min_offset) − d,  d − (d_FAF + max_offset))   # d = fac_distance_to_ltp(...)

    With both offsets 0 the window collapses to ``d = d_FAF``, which together with the
    on-course equality (:func:`fac_cross_track` = 0) reproduces the exact-FAF pin.
    """
    d = fac_distance_to_ltp(n, e, lpv)
    return (d_faf_m + min_offset_m) - d, d - (d_faf_m + max_offset_m)


def lpv_corridor_violation(n, e, lpv, k: float = DEFAULT_K_MARGIN):
    """LPV angular corridor: keep ``|cross-track to FAC| ≤ k · halfwidth(node)``, as TWO smooth rows.

    The final approach course is the line through the GARP and the LTP. Combine the cross-track
    from that line with the position-dependent half-width (the converging cone)::

        e_xt      = cross_track(n, e, GARP, LTP)              # signed
        margin    = k · lpv_course_halfwidth(n, e, lpv)      # converging toward the runway
        return (e_xt − margin, −e_xt − margin)               # both ≤ 0  ⇔  |e_xt| ≤ margin

    Returns the pair, NOT ``|e_xt| − margin``: the absolute value's kink at ``e_xt = 0`` (where the
    on-centerline LPV optimum sits) makes the gradient-based NLP fail to converge; two linear rows
    are smooth and equivalent.
    """
    e_xt = geo.cross_track(n, e, lpv.garp_ne, lpv.ltp_ne)
    margin = k * lpv_course_halfwidth(n, e, lpv)
    return e_xt - margin, -e_xt - margin
