"""Planar geometry in the target ``(n, e)`` frame — the building blocks of every constraint.

Backend-agnostic by design: a node's coordinates ``n``/``e`` come in as **scalar components**
(each may be a NumPy array of node values, or a CasADi symbol). Write the bodies with plain
``+ - * /`` and the :mod:`approach_constraints.mathx` helpers — then the *same* function produces
a NumPy number for tests and a CasADi expression for the optimizer (that is the "写完即可用"
guarantee).

A leg is the straight centerline from fix ``A`` to fix ``B`` (each a constant ``(2,)`` array
``[n, e]``). For a node ``(n, e)`` we need (design-doc §2):

    along-track  s    = how far along A→B the projection of the node sits
    cross-track  e_xt = signed perpendicular distance of the node from the A→B line

plus a course **bearing** and the **intercept angle** between a track and a course.

HEADING CONVENTION — the ONE convention used everywhere in the modeling plane: the dynamics
model's ``psi`` is **0 = East (+e), counter-clockwise toward North (+n)** (math-ENU;
``aerodynamic_model/casadi_simulator.py``: ``V_east = V·cosγ·cos ψ``, ``V_north = V·cosγ·sin ψ``).
:func:`course_bearing` returns courses in this SAME convention so they compare directly against
``psi``. It is NOT the compass bearing (0 = North, clockwise) — the two agree only at 45°/225°,
and mixing them was a real bug once (KRDU RW32).

────────────────────────────────────────────────────────────────────────────────────────────
The four functions below were the teaching TODOs ①–④ (now implemented). Fixes (``A``, ``B``)
are constants, so ``leg_unit`` may use NumPy; the node coords ``n``/``e`` are variables, so they
combine with only ``+ - * /`` and ``mathx.*``. Each docstring keeps the formula, the convention,
and a worked check.
"""

from __future__ import annotations

import numpy as np

from geokit import RAD2DEG

from . import mathx


def leg_unit(A, B) -> tuple[float, float, float]:
    """(provided) Unit-vector components ``(u_n, u_e)`` and length of the leg ``A → B``.

    ``A``, ``B`` are constant fixes, so plain NumPy is fine here. Raises if ``A == B``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    dn, de = B[0] - A[0], B[1] - A[1]
    length = float(np.hypot(dn, de))
    if length == 0.0:
        raise ValueError("degenerate leg: A and B coincide")
    return dn / length, de / length, length


def along_track(n, e, A, B):
    """① — along-track distance of node ``(n, e)`` projected onto leg ``A → B`` (metres).

    Definition (with the constant unit vector ``û = (u_n, u_e)`` from :func:`leg_unit`)::

        s = (n − A_n)·u_n + (e − A_e)·u_e          # scalar projection of (node − A) onto û, dot product

    ``s = 0`` at A, ``s = ‖B − A‖`` at B; may be negative / beyond the leg (do **not** clamp).

    ``n``/``e`` are scalar components (NumPy array OR CasADi symbol) → use only ``+ − ×`` so it
    works on both backends. Worked check: A=(0,0), B=(10,0), node=(3,7) → û=(1,0), s = 3.

    Hint: ``u_n, u_e, _ = leg_unit(A, B)``; then ``(n - A[0]) * u_n + (e - A[1]) * u_e``.
    """
    u_n, u_e, _ = leg_unit(A, B)
    return (n - A[0]) * u_n + (e - A[1]) * u_e

def cross_track(n, e, A, B):
    """② — **signed** cross-track distance of node ``(n, e)`` from leg ``A → B`` (metres).

    The perpendicular offset, via the 2-D cross product (the sign tells you which side)::

        e_xt = (n − A_n)·u_e − (e − A_e)·u_n        # z-component of (node − A) × û, cross product

    Corridor checks use ``mathx.fabs(e_xt)``; the sign is handy for turn-side logic. ``n``/``e``
    are scalar components → plain ``+ − ×`` only. Worked check: A=(0,0), B=(10,0), node=(3,7) →
    e_xt = −7 (node is 7 m to the +e side; with û=(1,0): 3·0 − 7·1 = −7).

    Hint: ``u_n, u_e, _ = leg_unit(A, B)``; then ``(n - A[0]) * u_e - (e - A[1]) * u_n``.
    """
    u_n, u_e, _ = leg_unit(A, B)
    return (n - A[0]) * u_e - (e - A[1]) * u_n



def course_bearing(A, B) -> float:
    """③ — course of the leg ``A → B`` in the MODEL's heading convention
    (radians, 0 = East/+e, counter-clockwise toward North/+n).

    This is the SAME convention as the dynamics state ``psi`` (``V_east = V·cosγ·cos ψ``,
    ``V_north = V·cosγ·sin ψ``), so the result compares directly against ``psi`` — e.g. in
    :func:`intercept_angle_deg`. It is NOT the compass bearing ``atan2(Δe, Δn)`` (0 = North,
    clockwise); the two agree only at 45°/225°. ``A``, ``B`` are constants, so this returns a
    plain float::

        course = atan2(n_B − n_A, e_B − e_A)

    Worked check: A=(0,0), B=(5,0) (due north) → atan2(5, 0) = +π/2;
    A=(0,0), B=(0,5) (due east) → atan2(0, 5) = 0.
    """
    return mathx.atan2(B[0] - A[0], B[1] - A[1])


def intercept_angle_deg(track_bearing_rad, course_bearing_rad):
    """④ — absolute angle (degrees, 0…180) between a track and a course.

    For the ≤ 30° final-course intercept at the PFAF (design-doc §4.4). Both arguments must be
    in the model convention (0 = East, CCW) — pass the state heading ``psi`` directly and a
    course from :func:`course_bearing`. ``track_bearing_rad`` may be the *variable* heading, so
    use ``mathx.*`` — but treat this as **numeric validation only**: the ``fabs`` kink sits
    exactly at the aligned-heading optimum (the same kink the corridor rows are split in two to
    avoid), so an NLP constraint must use a linear ψ-branch box instead, as
    ``collocation/optimizer.py``'s intercept rows do. Wrap the difference into ``(−π, π]``
    before taking the magnitude (so 350° vs 10° reads as 20°, not 340°)::

        d = atan2(sin(track − course), cos(track − course)); use unit circle trig to wrap into (−π, π]
        angle = |d| · (180/π)

    Worked check: track = 10°, course = 350° → 20°. ``RAD2DEG`` is imported for the last step.
    """
    d = mathx.atan2(mathx.sin(track_bearing_rad - course_bearing_rad), mathx.cos(track_bearing_rad - course_bearing_rad))
    return mathx.fabs(d) * RAD2DEG
