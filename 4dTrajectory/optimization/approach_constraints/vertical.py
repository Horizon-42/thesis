"""Vertical constraints — glidepath window (final) and the step-down floor (the rest).

Design-doc §3.1b/§6; LPV glidepath in ``lpv_final_segment.en.html`` §4.

Two flavours:
* **Glidepath window** (final) — keep the altitude in a thin band around the published ~3°
  glidepath (vertical guidance, not "dive and drive").
* **Step-down floor** (feeder / initial / intermediate) — a descending STAIRCASE of "at or
  above" minimum altitudes, plus an optional max-descent-gradient cap.

Final-segment profile (side view); the glidepath is a straight line aimed at TCH above the LTP::

    alt
     │  PFAF ●╲
     │         ╲╲   glidepath  h_GP(d) = TDZE + TCH + d·tan(GPA)   (a straight line in space)
     │           ╲╲ ── + above_m ┐  window the altitude h must stay inside
     │             ╲╲ ── − below_m┘  (below_m kept tight — being LOW is the dangerous side)
     │               ╲● ← TCH above the threshold
     ┼────────────────●─────────────────────────────────────►  d  (horizontal dist. BACK from LTP)
   TDZE              LTP, d=0

Step-down floor (non-final), a staircase vs along-track ``s`` from the segment start::

      alt ▲
     1524 ┤■■■■■■           "stay ≥1524 until s = 2000 m"
      914 ┤      ■■■■■■     "then ≥914 until s = 5000 m"
        0 ┤            ■■■  then the base floor
          └──┬─────┬────► s
            2000  5000

Glossary
--------
Aviation:
    glidepath  the published vertical descent path of the final approach (a straight line in
          space); the aircraft rides it down to the runway.
    GPA   Glidepath Angle (下滑道角) — the descent angle (normally 3.0°). ``lpv.gpa_deg``.
    TDZE  Touchdown Zone Elevation (接地带标高) — the runway-threshold elevation (m MSL); the
          altitude of the LTP. ``lpv.tdze_m``.
    TCH   Threshold Crossing Height (入口穿越高度) — height of the glidepath ABOVE the LTP at
          the threshold (≈50 ft ≈ 15 m). ``lpv.tch_m``.
    LTP / PFAF  Landing Threshold Point / Precise Final Approach Fix (see lateral.py glossary).
    step-down (fix)  a fix imposing an "at or above" minimum crossing altitude inside a segment;
          the set of them forms the staircase floor.
    MOC   Minimum Obstacle Clearance (最小越障余度) — the clearance baked into a minimum
          altitude; ``moc_floor`` is that minimum-altitude floor.
    ft/NM  feet per nautical mile — units of descent gradient (318 ft/NM ≈ 3.0°).
    MSL   Mean Sea Level — the datum altitudes are measured from.

Quantities (all scalar components — a NumPy array OR a CasADi symbol):
    h        altitude of the node (m MSL).
    h_GP(d)  glidepath altitude at distance ``d`` — the centre of the vertical window.
    d        horizontal distance measured BACK from the LTP along the course (0 at the
             threshold, increasing toward the PFAF). Used by the glidepath.
    s        along-track distance from the SEGMENT START. Used by the step-down floor.
    gamma    flight-path angle (radians; negative = descending).
    below_m / above_m  half-widths of the glidepath window (how far below / above the glidepath
             ``h`` may be). ``below_m`` is kept tight — being low is the dangerous side.
    base_floor_m  the floor altitude where no step-down applies (default 0).
    StepDown(s_from_start_m, min_alt_m)  one step-down: "stay ≥ ``min_alt_m`` until you pass
             along-track ``s_from_start_m``".
    max_descent_deg  the steepest descent a segment allows (a positive angle, degrees).

Software:
    *_violation  returns ``g`` with ``g ≤ 0 ⇔ satisfied`` (the IPOPT inequality form). The
             glidepath window returns a TUPLE ``(low, high)`` — two such violations.
    mathx.if_else / mathx.fmax  backend dispatch used by the staircase, so the SAME code lowers
             to ``np.where``/``np.maximum`` (tests) and ``ca.if_else``/``ca.fmax`` (optimizer).
             The staircase is the one place a conditional is needed.
    glidepath_altitude / moc_floor  the two pieces the provided wrappers
             (``glidepath_window_violation`` / ``moc_floor_violation``) call.
"""

from __future__ import annotations

import math

from . import mathx


def glidepath_altitude(d, lpv):
    """⑧ — altitude (m MSL) of the LPV glidepath at horizontal distance ``d`` back from the LTP.

    A straight line in space leaving the threshold at ``TCH`` above the LTP, climbing at the GPA
    (design-doc §3.1b)::

        h_GP(d) = TDZE + TCH + d · tan(GPA)

    ``lpv`` carries ``tdze_m``, ``tch_m``, ``gpa_deg``. The GPA is a **constant**, so
    ``tan(GPA)`` is a constant Python float — multiplying the variable ``d`` by it works for both
    backends (no ``mathx`` needed here). ``d`` is 0 at the threshold, + toward the PFAF.

    Worked check: TDZE=120, TCH=15.24, GPA=3°, d=1000 → ≈ 187.65 m.
    Hint: ``return lpv.tdze_m + lpv.tch_m + d * math.tan(math.radians(lpv.gpa_deg))``.
    """
    return lpv.tdze_m + lpv.tch_m + d * math.tan(math.radians(lpv.gpa_deg))


def glidepath_window_violation(h, d, lpv, below_m=None, above_m=None):
    """(provided) Two-sided glidepath window → ``(low, high)``, each ``≤ 0`` when inside.

    ``h`` must lie in ``[h_GP − below, h_GP + above]``::

        low  = (h_GP − below) − h       # too LOW  (the dangerous side → keep ``below`` tight)
        high = h − (h_GP + above)       # too HIGH

    Returns a tuple of two violation arrays/expressions (backend-agnostic; no stacking). Calls ⑧.
    """
    below = lpv.below_m if below_m is None else below_m
    above = lpv.above_m if above_m is None else above_m
    gp = glidepath_altitude(d, lpv)
    return (gp - below) - h, h - (gp + above)


def moc_floor(s, step_downs, base_floor_m: float = 0.0):
    """⑨ — minimum-altitude floor (m MSL) at along-track ``s`` from the segment start.

    A step-down profile is a **staircase**: the floor is higher early and drops toward the runway
    (design-doc §6.1). Each ``StepDown(s_from_start_m, min_alt_m)`` means *"stay at or above
    ``min_alt`` until you pass along-track ``s_from_start``"*. So the floor at ``s`` is the highest
    crossing-altitude among the step-downs still **ahead of (or at)** ``s``::

        floor(s) = base, then for each step:
            floor = (s ≤ step.s_from_start) ? max(floor, step.min_alt) : floor

    Build it with ``mathx`` so ``s`` may be a NumPy array **or** a CasADi symbol::

        floor = base_floor_m
        for st in step_downs:
            cond  = s <= st.s_from_start_m
            floor = mathx.if_else(cond, mathx.fmax(floor, st.min_alt_m), floor)
        return floor

    Worked check: steps [(2000, 1524), (5000, 914)], base 0 →
    s=3000 → 914 (only the 5000 m step ahead); s=1000 → 1524 (both ahead); s=6000 → 0 (none).
    """
    floor = base_floor_m
    for st in step_downs:
        cond = s <= st.s_from_start_m
        floor = mathx.if_else(cond, mathx.fmax(floor, st.min_alt_m), floor)
    return floor


def moc_floor_violation(h, s, step_downs, base_floor_m: float = 0.0):
    """(provided) Step-down floor constraint: ``floor(s) − h ≤ 0`` (i.e. ``h ≥ floor``). Calls ⑨."""
    return moc_floor(s, step_downs, base_floor_m) - h


def descent_gradient_violation(gamma, max_descent_deg: float):
    """⑩ (bonus) — cap the descent: don't descend steeper than ``max_descent_deg``.

    On a descent ``gamma`` is negative; the constraint is ``gamma ≥ −max_descent`` (shallower /
    level is fine)::

        violation = (−max_descent_rad) − gamma          # ≤ 0 when not too steep

    For the descent caps (design-doc §6.3): intermediate ≈ 3°, initial/feeder ≈ 4.7° (≈ 500 ft/NM).
    ``gamma`` is a variable, ``max_descent_deg`` a constant — plain arithmetic, no ``mathx`` needed.
    Hint: ``return -math.radians(max_descent_deg) - gamma``.
    """
    return -math.radians(max_descent_deg) - gamma
