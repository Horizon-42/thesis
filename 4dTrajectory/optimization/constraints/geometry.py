"""Planar geometry in the target ``(n, e)`` frame — the building blocks of every constraint.

Backend-agnostic by design: a node's coordinates ``n``/``e`` come in as **scalar components**
(each may be a NumPy array of node values, or a CasADi symbol). Write the bodies with plain
``+ - * /`` and the :mod:`constraints.mathx` helpers — then the *same* function produces a NumPy
number for tests and a CasADi expression for the optimizer (that is the "写完即可用" guarantee).

A leg is the straight centerline from fix ``A`` to fix ``B`` (each a constant ``(2,)`` array
``[n, e]``). For a node ``(n, e)`` we need (design-doc §2):

    along-track  s    = how far along A→B the projection of the node sits
    cross-track  e_xt = signed perpendicular distance of the node from the A→B line

plus a course **bearing** and the **intercept angle** between a track and a course.

────────────────────────────────────────────────────────────────────────────────────────────
YOU IMPLEMENT the four functions below (TODO ①–④). Fixes (``A``, ``B``) are constants, so
``leg_unit`` may use NumPy; the node coords ``n``/``e`` are variables, so combine them with only
``+ - * /`` and ``mathx.*``. Each docstring has the formula, the convention, and a worked check.
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
    """③ — bearing of the leg ``A → B`` (radians, 0 = North/+n, +clockwise toward +e).

    ``A``, ``B`` are constants, so this returns a plain float::

        bearing = atan2(e_B − e_A, n_B − n_A)

    Worked check: A=(0,0), B=(0,5) (due east) → atan2(5, 0) = +π/2.
    Hint: ``mathx.atan2(B[1] - A[1], B[0] - A[0])`` (or ``np.arctan2`` — both fine for constants).
    """
    return mathx.atan2(B[1] - A[1], B[0] - A[0])


def intercept_angle_deg(track_bearing_rad, course_bearing_rad):
    """④ — absolute angle (degrees, 0…180) between a track and a course.

    For the ≤ 30° final-course intercept at the PFAF (design-doc §4.4). ``track_bearing_rad`` may
    be the *variable* heading (``psi``), so use ``mathx.*``. Wrap the difference into ``(−π, π]``
    before taking the magnitude (so 350° vs 10° reads as 20°, not 340°)::

        d = atan2(sin(track − course), cos(track − course)); use unit circle trig to wrap into (−π, π]
        angle = |d| · (180/π)

    Worked check: track = 10°, course = 350° → 20°. ``RAD2DEG`` is imported for the last step.
    Hint: ``d = mathx.atan2(mathx.sin(t - c), mathx.cos(t - c))``; ``return mathx.fabs(d) * RAD2DEG``.
    """
    d = mathx.atan2(mathx.sin(track_bearing_rad - course_bearing_rad), mathx.cos(track_bearing_rad - course_bearing_rad))
    return mathx.fabs(d) * RAD2DEG
