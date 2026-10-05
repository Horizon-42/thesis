"""The tokens of the traffic attention (post-training §3, §8 C2; D23, D31, D98): for each other aircraft of a window at
one step, what it is to the commanded aircraft (the edge features) and its own motion, from its recorded state only.

**When.** At each step of the window, one instant for every aircraft: the commanded aircraft's row and each other
aircraft's row at that time. Nothing is carried from step to step, and no value of a later row is read.

**Motion** is each aircraft's displacement in the 2 s before the row (`post.motion`, as the prior's motion inputs). An
aircraft's row 0 has no row before it: its motion is unknown, the pair's ``motion_unknown`` is 1 and every feature that
reads a motion is 0; a commanded aircraft whose motion is unknown or zero has no frame (``front``, ``left`` and the
direction 0).

**Runways.** An aircraft's R is used only where it is in force (D23): a recorded aircraft's after its own first
predicted step, the commanded aircraft's from the row after its first predicted step (`scene.AircraftAt`).

**The features** (`TOKEN_FEATURES`, in order; horizontal distances as asinh(d / `SCALE_M`), heights over
`HEIGHT_SCALE_M`, speeds over `SPEED_SCALE_MPS`, vertical rates over `VERTICAL_RATE_SCALE_MPS`, the time over
`CPA_HORIZON_S`; every scale a constant in SI units, as prior D41):

- ``front``, ``left``, ``height`` — the other aircraft in the commanded aircraft's frame: along its direction of motion,
  to its left; its height less the commanded aircraft's;
- ``clock_ahead``, ``clock_incomparable`` — its position on the approach clock less the commanded aircraft's
  (`runways.approach_clock_m`); 0 and the flag 1 where the two runways in force have other directions or one is not in
  force;
- ``same``, ``single``, ``dependent``, ``independent``, ``unrelated`` — the relation of the two runways in force
  (`Separation.relation`), one-hot; all 0 where one is not in force;
- ``closing`` — the rate at which the horizontal distance closes (positive: closing);
- ``cpa_time``, ``cpa_horizontal``, ``cpa_vertical`` — both flying straight on at their present motion: when the
  horizontal distance is least (clipped to [0, `CPA_HORIZON_S`]), that distance, and the height difference then;
- ``required`` — the distance the rules require between the two as leader and follower (`Separation.distance_nm`: the
  one ahead on the approach clock leads; 0 where the clock cannot compare them);
- ``motion_unknown`` — 1 where either motion is unknown;
- ``ground_speed``, ``vertical_rate``, ``direction_sin``, ``direction_cos`` — the other aircraft's own motion: its
  ground speed and vertical rate, and its direction of motion less the commanded aircraft's (0 without a frame or a
  motion).
"""

from __future__ import annotations

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import (
    DEPENDENT, FAA_RADAR_NM, INDEPENDENT, SAME, SINGLE, UNRELATED, Separation,
)
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.post.motion import motion
from ts_transformer.post.runways import approach_clock_m
from ts_transformer.post.scene import AircraftAt

#: The name of this set of features: a changed feature is a new name (principle 8).
EDGES_SCHEMA = "post-edges-v1"
RELATIONS = (SAME, SINGLE, DEPENDENT, INDEPENDENT, UNRELATED)
EDGE_FEATURES = ("front", "left", "height", "clock_ahead", "clock_incomparable", *RELATIONS, "closing", "cpa_time",
                 "cpa_horizontal", "cpa_vertical", "required", "motion_unknown")
MOTION_FEATURES = ("ground_speed", "vertical_rate", "direction_sin", "direction_cos")
TOKEN_FEATURES = EDGE_FEATURES + MOTION_FEATURES
#: The horizontal scale: the terminal radar minimum (7110.65BB 5-5-4 a/b, `runway_schedule.FAA_RADAR_NM`), around which
#: the judgements between aircraft are made.
SCALE_M = FAA_RADAR_NM * NM_M
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0
VERTICAL_RATE_SCALE_MPS = 10.0
CPA_HORIZON_S = 120.0


def _scaled(distance_m) -> np.ndarray:
    return np.arcsinh(np.asarray(distance_m, dtype=np.float64) / SCALE_M)


def tokens(own: AircraftAt, others: AircraftAt, geometry: AirportGeometry, separation: Separation,
           step_s: float) -> np.ndarray:
    """``[len(others), len(TOKEN_FEATURES)]`` float32: the token of each other aircraft at one step (module docstring);
    ``own`` the commanded aircraft (one)."""
    if len(own) != 1:
        raise ValueError(f"the tokens are read from one commanded aircraft, not {len(own)}")
    index = {name: k for k, name in enumerate(TOKEN_FEATURES)}
    out = np.zeros((len(others), len(TOKEN_FEATURES)), dtype=np.float64)
    if not len(others):
        return out.astype(np.float32)
    mine, theirs = motion(own.at, own.before, own.known, step_s), motion(others.at, others.before, others.known, step_s)
    velocity_i = np.array([mine.east_mps[0], mine.north_mps[0]])
    velocity_j = np.stack((theirs.east_mps, theirs.north_mps), axis=1)
    framed = bool(mine.known[0]) and mine.ground_speed_mps[0] > 0.0
    forward = velocity_i / mine.ground_speed_mps[0] if framed else np.zeros(2)
    left = np.array([-forward[1], forward[0]])
    offset = others.at[:, :2] - own.at[0, :2]
    dh = others.at[:, 2] - own.at[0, 2]
    out[:, index["front"]] = _scaled(offset @ forward)
    out[:, index["left"]] = _scaled(offset @ left)
    out[:, index["height"]] = dh / HEIGHT_SCALE_M
    known = mine.known[0] & theirs.known
    relative = velocity_j - velocity_i
    distance = np.hypot(offset[:, 0], offset[:, 1])
    approach = -(offset * relative).sum(axis=1)
    closing = np.divide(approach, distance, out=np.zeros_like(approach), where=distance > 0.0)
    speed2 = (relative * relative).sum(axis=1)
    cpa = np.clip(np.divide(approach, speed2, out=np.full_like(approach, CPA_HORIZON_S), where=speed2 > 0.0),
                  0.0, CPA_HORIZON_S)
    closest = offset + relative * cpa[:, None]
    climb = theirs.vertical_rate_mps - mine.vertical_rate_mps[0]
    out[:, index["closing"]] = np.where(known, closing / SPEED_SCALE_MPS, 0.0)
    out[:, index["cpa_time"]] = np.where(known, cpa / CPA_HORIZON_S, 0.0)
    out[:, index["cpa_horizontal"]] = np.where(known, _scaled(np.hypot(closest[:, 0], closest[:, 1])), 0.0)
    out[:, index["cpa_vertical"]] = np.where(known, (dh + climb * cpa) / HEIGHT_SCALE_M, 0.0)
    out[:, index["motion_unknown"]] = ~known
    out[:, index["ground_speed"]] = theirs.ground_speed_mps / SPEED_SCALE_MPS
    out[:, index["vertical_rate"]] = theirs.vertical_rate_mps / VERTICAL_RATE_SCALE_MPS
    angle = np.radians(theirs.track_deg - mine.track_deg[0])
    directed = theirs.known & framed
    out[:, index["direction_sin"]] = np.where(directed, np.sin(angle), 0.0)
    out[:, index["direction_cos"]] = np.where(directed, np.cos(angle), 0.0)
    _runways(out, index, own, others, geometry, separation)
    return out.astype(np.float32)


def _runways(out: np.ndarray, index: dict[str, int], own: AircraftAt, others: AircraftAt, geometry: AirportGeometry,
             separation: Separation) -> None:
    """The features that read the runways in force (module docstring), into ``out``."""
    out[:, index["clock_incomparable"]] = 1.0
    r_i = int(own.runway_index[0])
    if r_i < 0:
        return
    ident_i = geometry.candidates[r_i].ident
    along_i = float(approach_clock_m(separation, geometry, r_i, own.at[0, 0], own.at[0, 1]))
    for k in np.flatnonzero(others.runway_index >= 0):
        r_j = int(others.runway_index[k])
        ident_j = geometry.candidates[r_j].ident
        relation = separation.relation(ident_i, ident_j)
        out[k, index[relation]] = 1.0
        if relation == UNRELATED:
            continue
        along = float(approach_clock_m(separation, geometry, r_j, others.at[k, 0], others.at[k, 1])) - along_i
        out[k, index["clock_incomparable"]] = 0.0
        out[k, index["clock_ahead"]] = _scaled(along)
        if along >= 0.0:            # the other aircraft leads
            required = separation.distance_nm(ident_j, others.category[k], ident_i, own.category[0])
        else:
            required = separation.distance_nm(ident_i, own.category[0], ident_j, others.category[k])
        out[k, index["required"]] = _scaled(required * NM_M)
