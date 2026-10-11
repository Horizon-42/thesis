"""The tokens of the traffic attention (post-training §3, §8 C2; D23, D31, D98): for each other aircraft of a window at
one step, what it is to the commanded aircraft (the edge features) and its own motion, from its recorded state only.

**When.** At each step of the window, one instant for every aircraft: the commanded aircraft's row and each other
aircraft's row at that time. Nothing is carried from step to step, and no value of a later row is read.

**Motion** is each aircraft's displacement in the 2 s before the row (`prior.inputs.motion`, prior §7 item 2: the
prior's own motion inputs). An aircraft's row 0 has no row before it: its motion is unknown, the pair's ``motion_unknown`` is 1 and every feature that
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

from typing import Any

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import (
    DEPENDENT, FAA_RADAR_NM, INDEPENDENT, SAME, SINGLE, UNRELATED, Separation,
)
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.prior.inputs import motion
from ts_transformer.post.runways import approach_clock_m
from ts_transformer.post.scene import AircraftAt
from ts_transformer.post.traffic import scene_aircraft

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


class Pairs:
    """The features of a pair that read only its runways in force and its wake categories, each computed once (multi-
    aircraft control D189 (1)): the runways' relation (`Separation.relation` reads the two runways) and the distance the
    rules require, scaled (`Separation.distance_nm` reads the two runways and the two categories)."""

    def __init__(self, separation: Separation) -> None:
        self.separation = separation
        self._relations: dict[tuple[str, str], str] = {}
        self._required: dict[tuple[str, str | None, str, str | None], float] = {}

    def relation(self, a: str, b: str) -> str:
        if (a, b) not in self._relations:
            self._relations[(a, b)] = self.separation.relation(a, b)
        return self._relations[(a, b)]

    def required(self, leader: str, leader_category: str | None, follower: str, follower_category: str | None) -> float:
        """The feature ``required`` of a leader and a follower (module docstring)."""
        key = (leader, leader_category, follower, follower_category)
        if key not in self._required:
            self._required[key] = float(_scaled(self.separation.distance_nm(*key) * NM_M))
        return self._required[key]


def clocks(aircraft: AircraftAt, geometry: AirportGeometry, separation: Separation) -> np.ndarray:
    """``[N]``: each aircraft's position on the approach clock of its R (`runways.approach_clock_m`), NaN without one in
    force (the scene's ``along_m``, `post.traffic.traffic`)."""
    out = np.full(len(aircraft), np.nan)
    for k in np.flatnonzero(aircraft.runway_index >= 0):
        out[k] = float(approach_clock_m(separation, geometry, int(aircraft.runway_index[k]), aircraft.at[k, 0],
                                        aircraft.at[k, 1]))
    return out


def tokens(own: AircraftAt, others: AircraftAt, geometry: AirportGeometry, separation: Separation,
           step_s: float) -> np.ndarray:
    """``[len(others), len(TOKEN_FEATURES)]`` float32: the token of each other aircraft at one step (module docstring).
    ``own`` is the commanded aircraft (one)."""
    if len(own) != 1:
        raise ValueError(f"the tokens are read from one commanded aircraft, not {len(own)}")
    (out,) = each_tokens(own, others, geometry, Pairs(separation), step_s)
    return out


def each_tokens(owns: AircraftAt, others: AircraftAt, geometry: AirportGeometry, pairs: Pairs,
                step_s: float) -> list[np.ndarray]:
    """The tokens of ``others`` as each aircraft of ``owns`` reads them alone, at one step: for each own aircraft,
    `tokens` of it and ``others``, bit for bit (multi-aircraft control S2). The own aircraft do not read each other.
    The motions and approach clocks are calculated one time for all the aircraft."""
    count = len(owns)
    both = scene_aircraft(owns, others)
    views = np.tile(np.arange(count, len(both), dtype=np.int64), (count, 1))
    return window_tokens(both, count, geometry, pairs, motion(both.at, both.before, both.known, step_s),
                         clocks(both, geometry, pairs.separation), views=views)


def window_tokens(aircraft: AircraftAt, commanded: int, geometry: AirportGeometry, pairs: Pairs, moving: Any,
                  clock_m: np.ndarray, views: np.ndarray | None = None) -> list[np.ndarray]:
    """The tokens of every pair of a window at one step (multi-aircraft control D189 (3)): of each of the first
    ``commanded`` aircraft of ``aircraft`` (a window's set: its commanded aircraft, then its recorded ones), the token
    of every other aircraft in its view's order (the recorded ones, then the other commanded ones), ``[len(aircraft) -
    1, len(TOKEN_FEATURES)]`` float32 each; from the set's motions (``moving``, `prior.inputs.motion`'s) and approach
    clocks (`clocks`), computed once for the window, and the pairs' runway features from ``pairs``. Each feature is
    computed over every pair at once, elementwise, but the two products of a pair's offset with the commanded
    aircraft's frame, one matrix product for each commanded aircraft (a stacked product could round another way): each
    token's numbers are those of the aircraft alone. ``views``: ``[commanded, m]``, the other aircraft that each
    commanded aircraft reads, in order (`each_tokens`). None: the view above."""
    count = len(aircraft)
    index = {name: k for k, name in enumerate(TOKEN_FEATURES)}
    # each commanded aircraft's other aircraft, in its view's order
    others = (np.array([[*range(commanded, count), *(c for c in range(commanded) if c != i)]
                        for i in range(commanded)], dtype=np.int64).reshape(commanded, count - 1)
              if views is None else views)
    out = np.zeros((commanded, others.shape[1], len(TOKEN_FEATURES)), dtype=np.float64)
    if not others.shape[1]:
        return list(out.astype(np.float32))
    velocity = np.stack((moving.east_mps, moving.north_mps), axis=1)
    ground = moving.ground_speed_mps
    framed = moving.known[:commanded] & (ground[:commanded] > 0.0)
    forward = np.zeros((commanded, 2))
    forward[framed] = velocity[:commanded][framed] / ground[:commanded][framed, None]
    left = np.stack((-forward[:, 1], forward[:, 0]), axis=1)
    offset = aircraft.at[others, :2] - aircraft.at[:commanded, None, :2]
    dh = aircraft.at[others, 2] - aircraft.at[:commanded, None, 2]
    out[..., index["front"]] = _scaled(np.stack([offset[i] @ forward[i] for i in range(commanded)]))
    out[..., index["left"]] = _scaled(np.stack([offset[i] @ left[i] for i in range(commanded)]))
    out[..., index["height"]] = dh / HEIGHT_SCALE_M
    known = moving.known[:commanded, None] & moving.known[others]
    relative = velocity[others] - velocity[:commanded, None]
    distance = np.hypot(offset[..., 0], offset[..., 1])
    approach = -(offset * relative).sum(axis=-1)
    closing = np.divide(approach, distance, out=np.zeros_like(approach), where=distance > 0.0)
    speed2 = (relative * relative).sum(axis=-1)
    cpa = np.clip(np.divide(approach, speed2, out=np.full_like(approach, CPA_HORIZON_S), where=speed2 > 0.0),
                  0.0, CPA_HORIZON_S)
    closest = offset + relative * cpa[..., None]
    climb = moving.vertical_rate_mps[others] - moving.vertical_rate_mps[:commanded, None]
    out[..., index["closing"]] = np.where(known, closing / SPEED_SCALE_MPS, 0.0)
    out[..., index["cpa_time"]] = np.where(known, cpa / CPA_HORIZON_S, 0.0)
    out[..., index["cpa_horizontal"]] = np.where(known, _scaled(np.hypot(closest[..., 0], closest[..., 1])), 0.0)
    out[..., index["cpa_vertical"]] = np.where(known, (dh + climb * cpa) / HEIGHT_SCALE_M, 0.0)
    out[..., index["motion_unknown"]] = ~known
    out[..., index["ground_speed"]] = ground[others] / SPEED_SCALE_MPS
    out[..., index["vertical_rate"]] = moving.vertical_rate_mps[others] / VERTICAL_RATE_SCALE_MPS
    angle = np.radians(moving.track_deg[others] - moving.track_deg[:commanded, None])
    directed = moving.known[others] & framed[:, None]
    out[..., index["direction_sin"]] = np.where(directed, np.sin(angle), 0.0)
    out[..., index["direction_cos"]] = np.where(directed, np.cos(angle), 0.0)
    for i in range(commanded):
        _runways(out[i], index, aircraft, i, others[i], geometry, pairs, clock_m)
    return list(out.astype(np.float32))


def _runways(out: np.ndarray, index: dict[str, int], aircraft: AircraftAt, i: int, others: np.ndarray,
             geometry: AirportGeometry, pairs: Pairs, clock_m: np.ndarray) -> None:
    """The features that read the runways in force (module docstring) of aircraft ``i`` and its ``others`` (places in
    ``aircraft``, ``clock_m`` their approach clocks), into ``out``."""
    out[:, index["clock_incomparable"]] = 1.0
    r_i = int(aircraft.runway_index[i])
    if r_i < 0:
        return
    ident_i, category_i = geometry.candidates[r_i].ident, aircraft.category[i]
    along_i = clock_m[i]
    for k in np.flatnonzero(aircraft.runway_index[others] >= 0):
        j = int(others[k])
        ident_j = geometry.candidates[int(aircraft.runway_index[j])].ident
        relation = pairs.relation(ident_i, ident_j)
        out[k, index[relation]] = 1.0
        if relation == UNRELATED:
            continue
        along = clock_m[j] - along_i
        out[k, index["clock_incomparable"]] = 0.0
        out[k, index["clock_ahead"]] = _scaled(along)
        if along >= 0.0:            # the other aircraft leads
            out[k, index["required"]] = pairs.required(ident_j, aircraft.category[j], ident_i, category_i)
        else:
            out[k, index["required"]] = pairs.required(ident_i, category_i, ident_j, aircraft.category[j])
