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


def box_corridor_violation(n, e, A, B, halfwidth_m: float, k: float = 0.5):
    """RNP box corridor: keep ``|cross-track| ≤ k · halfwidth``, as TWO smooth rows.

    Returns the pair ``(e_xt − margin, −e_xt − margin)`` (``margin = k · halfwidth``); both ``≤ 0``
    iff inside. We do **not** return ``|e_xt| − margin``: the absolute value has a
    non-differentiable kink at ``e_xt = 0`` — exactly where an on-centerline optimum sits — which
    makes the gradient-based NLP (IPOPT) fail to converge. Two linear rows are smooth everywhere
    and equivalent (their max is ``|e_xt| − margin``).

    ``halfwidth_m`` is the segment's containment half-width (``2·RNP`` or ``1·RNP`` — design-doc
    §3.2/§8); ``k`` is the design-margin fraction.

    Worked check: A=(0,0), B=(10,0), node=(3,7), halfwidth=20, k=0.5 → e_xt=−7, margin=10 →
    ``(−17, +3)``? no: ``(e_xt−margin, −e_xt−margin) = (−17, −3)`` → max −3 (inside).
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


def lpv_corridor_violation(n, e, lpv, k: float = 0.5):
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
