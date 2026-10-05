"""The motion of an aircraft at a row of a scene (post-training §3; D98): its displacement in the 2 s before the row,
from positions and heights only — never the stored track, ground speed or vertical rate, which are fits that use later
rows (vocabulary §6 item 3, D97).

MIRROR of the prior's motion inputs (`prior.inputs.motion`, prior D25, D60), which prior §7 does not export:
`tests/test_post_scene.py` checks that the two agree. An aircraft's row 0 has no row before it: its motion is unknown
(``known`` False) and every value 0.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Motion(NamedTuple):
    """The motion of rows, each ``[N]``: the velocity east and north and the vertical rate (m/s), the ground speed (m/s),
    the compass direction of motion (degrees in [0, 360)) and whether the motion is known."""

    east_mps: np.ndarray
    north_mps: np.ndarray
    vertical_rate_mps: np.ndarray
    ground_speed_mps: np.ndarray
    track_deg: np.ndarray
    known: np.ndarray


def motion(at: np.ndarray, before: np.ndarray, known: np.ndarray, step_s: float) -> Motion:
    """The motion of rows from their states ``at`` and the states of the 2 s row before each (``before``; both ``[N, 3]``:
    e, n, height), where ``known``."""
    known = np.asarray(known, dtype=bool)
    de, dn, dh = (np.asarray(at, dtype=np.float64) - np.asarray(before, dtype=np.float64)).T
    east, north = np.where(known, de / step_s, 0.0), np.where(known, dn / step_s, 0.0)
    return Motion(east, north, np.where(known, dh / step_s, 0.0), np.hypot(east, north),
                  np.where(known, np.degrees(np.arctan2(de, dn)) % 360.0, 0.0), known)
