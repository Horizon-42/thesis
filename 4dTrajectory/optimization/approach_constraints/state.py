"""Decision-state layout — shared with the optimizer's ``*Normalized`` schemes.

A *trajectory* here is a NumPy array of shape ``(K, 7)``; each row is one state node in the
**optimizer's NORMALIZED metric coordinates** (the representation the
``trapezoidalNormalizedFullTransport`` scheme already carries — see CLAUDE.md and
``optimization_constraint_design.md`` §2):

    z = (n, e, h, V, psi, gamma, m)

    n, e    metres from the target anchor (the LTP), in the target ENU-like frame  (see frame.py)
    h       altitude (metres, MSL)
    V       true airspeed (m/s)
    psi     heading (radians; 0 = North, +clockwise toward East)
    gamma   flight-path angle (radians; + = climb)
    m       mass (kg)

Because the optimizer state is *already* ``(n, e, …)``, the constraints in this package read
the horizontal position straight off the decision variables — **no per-node coordinate
transform**. Only the fixes (which arrive in lat/lon degrees) get converted once, in frame.py.

This module is plumbing: index constants + tiny column accessors. Nothing to implement here.
"""

from __future__ import annotations

import numpy as np

# Column indices into a state row / trajectory array.
N, E, H, V, PSI, GAMMA, MASS = range(7)
STATE_DIM = 7


def positions(traj: np.ndarray) -> np.ndarray:
    """The horizontal ``(n, e)`` columns, shape ``(..., 2)``."""
    return np.asarray(traj)[..., (N, E)]


def north(traj: np.ndarray) -> np.ndarray:
    """The north column ``n`` (metres), shape ``(...,)`` — pass as a scalar component."""
    return np.asarray(traj)[..., N]


def east(traj: np.ndarray) -> np.ndarray:
    """The east column ``e`` (metres), shape ``(...,)`` — pass as a scalar component."""
    return np.asarray(traj)[..., E]


def altitudes(traj: np.ndarray) -> np.ndarray:
    """The altitude column ``h`` (metres), shape ``(...,)``."""
    return np.asarray(traj)[..., H]


def headings(traj: np.ndarray) -> np.ndarray:
    """The heading column ``psi`` (radians), shape ``(...,)``."""
    return np.asarray(traj)[..., PSI]


def flight_path_angles(traj: np.ndarray) -> np.ndarray:
    """The flight-path-angle column ``gamma`` (radians), shape ``(...,)``."""
    return np.asarray(traj)[..., GAMMA]
