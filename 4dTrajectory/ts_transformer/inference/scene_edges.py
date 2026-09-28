"""The edge features a scene prior reads between aircraft (multi-aircraft design §2.5): for every ordered pair i → j at every
step, what j is to i — where it is in i's frame, where on the approach clock, how their runways relate, how fast they
close, their closest point of approach, and how far apart the rules would have them.

Computed here because it needs the separation rules (`runway_schedule`), which the prior's package does not reach (the
same placement as the loop's masks, design §9 item 18); what needs the runway geometry (each aircraft's position on the
approach clock) is handed in by the caller.

**When** (§2.1, §2.5): at each step, at aircraft i's own row time. Aircraft j is taken from its last row at or before that
instant and carried straight on to it (at most one step: rows hang within half a step of the step), so only the past is
read — except for j's very first row, which may lie up to a step after i's instant and is carried back to it.

**Motion** is each aircraft's displacement from its row before, over the time between — the prior's own node inputs'
motion (`prior.data`) and the design's "two rows' positions"; never the signals' fitted ground speed, track or vertical
rate, least-squares fits over a window centred on the row that read 7.5 s of the future. An aircraft's first row has no
row before it: its motion is unknown, the pair's ``motion_unknown`` is 1 and its motion features (``closing`` and the
closest point of approach) are 0; an unknown j is carried with no motion, and an unknown i has no frame (``front`` and
``left`` 0) — nor has an i that did not move since its row before (89 of the artefact's 12.1 million row pairs; the
prior's node reads such a direction as north). The rate along the approach clock is the same displacement of the clock position, known where the runway
in force is the same on both rows (at the step one changes, j is carried along the clock with none).

**The features** (`EDGE_FEATURES`, in order; horizontal distances scaled by asinh(d / `SCALE_M`), heights by
`HEIGHT_SCALE_M`, speeds by `SPEED_SCALE_MPS`, the time by `CPA_HORIZON_S`):

- ``self`` — 1 for i → i, where everything else is 0;
- ``front``, ``left``, ``height`` — j in i's frame: along i's direction of motion, to its left (0 without a frame),
  j's height less i's;
- ``clock_ahead``, ``clock_incomparable`` — j's position on the approach clock less i's; 0 and the flag 1 when the two
  runways in force have other directions, or one has none yet;
- ``same``, ``single``, ``dependent``, ``independent``, ``unrelated`` — the relation of the runways in force
  (`Separation.relation`), one-hot; all 0 when one has none yet;
- ``closing`` — the rate the horizontal distance closes at (positive: closing);
- ``cpa_time``, ``cpa_horizontal``, ``cpa_vertical`` — both flying straight on at their present motion: when the
  horizontal distance is least (clipped to [0, `CPA_HORIZON_S`]), that distance, and j's height less i's then;
- ``motion_unknown`` — 1 when i's or j's motion is unknown (above);
- ``required`` — the distance the rules require were the two leader and follower (`Separation.distance_nm`: the one
  ahead on the approach clock leads; the one-runway radar minimum or TBL 5-5-2, a dependent pair's stagger, 0 otherwise
  and when not comparable), the distance design §2.5 gives the model (user 2026-09-27, §9 item 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import (
    DEPENDENT, FAA_RADAR_NM, INDEPENDENT, SAME, SINGLE, UNRELATED, Separation,
)

RELATIONS = (SAME, SINGLE, DEPENDENT, INDEPENDENT, UNRELATED)
EDGE_FEATURES = ("self", "front", "left", "height", "clock_ahead", "clock_incomparable", *RELATIONS, "closing",
                 "cpa_time", "cpa_horizontal", "cpa_vertical", "motion_unknown", "required")
#: The horizontal scale: the radar minimum (design §2.5: judgements between aircraft are made around it).
SCALE_M = FAA_RADAR_NM * NM_M
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0
CPA_HORIZON_S = 120.0


@dataclass(frozen=True)
class SceneRows:
    """A scene's aircraft on its steps, ``[A, T]`` each (NaN where absent), each aircraft's rows on consecutive steps
    from its first, one per step: the time of the row hung on the step (epoch seconds), position (airport frame) and
    position on the approach clock (NaN without a runway in force). ``runway`` holds the runway in force per aircraft and
    step — in force BEFORE the step's words, as the prior reads its words (`prior.data.sentence_steps`' ``in_force``:
    said at an earlier row; the words of the step it predicts never leak into its edges) — None before any is said or
    where absent. ``category``: the CWT category per aircraft; None where the record has no type, where the required
    distance is the radar minimum alone (as the judge reads such a pair, `separation.losses` ``wake_known``; the caller
    counts them). Refused unless a present step's position is finite, an aircraft's steps are one run, and a runway is
    given exactly where the clock is."""

    time_s: np.ndarray
    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray
    along_m: np.ndarray
    runway: Sequence[Sequence[str | None]]
    category: Sequence[str | None]

    def __post_init__(self) -> None:
        present = self.present
        for name in ("e_m", "n_m", "height_m"):
            if not np.isfinite(getattr(self, name)[present]).all():
                raise ValueError(f"a present aircraft-step has a non-finite {name}")
        starts = present[:, 0].astype(int) + (np.diff(present.astype(np.int8), axis=1) > 0).sum(axis=1)
        if (starts > 1).any():
            raise ValueError("an aircraft's rows are on consecutive steps, one run from its first")
        said = np.array([[runway is not None for runway in steps] for steps in self.runway], dtype=bool)
        if said.shape != present.shape or (said & ~present).any():
            raise ValueError("a runway is given per aircraft and step, and only where the aircraft is present")
        if not np.array_equal(said, np.isfinite(self.along_m)):
            raise ValueError("an aircraft-step is on the approach clock exactly when a runway is in force")

    @property
    def present(self) -> np.ndarray:
        return ~np.isnan(self.time_s)


class Motion(NamedTuple):
    """Each aircraft's motion on each step from its row before (module docstring), 0 where unknown."""

    velocity_mps: np.ndarray       # [A, T, 2] east, north
    climb_mps: np.ndarray          # [A, T]
    along_rate_mps: np.ndarray     # [A, T]
    known: np.ndarray              # [A, T] bool: a row before it


def motion(rows: SceneRows, runway: np.ndarray) -> Motion:
    """``rows``' motion; ``runway``: the runway in force as an index per aircraft and step, −1 for none."""
    present = rows.present
    known = np.zeros_like(present)
    known[:, 1:] = present[:, 1:] & present[:, :-1]
    along_known = known.copy()
    along_known[:, 1:] &= (runway[:, 1:] == runway[:, :-1]) & (runway[:, 1:] >= 0)
    dt = np.where(known[:, 1:], np.diff(rows.time_s, axis=1), 1.0)

    def rate(values: np.ndarray, where: np.ndarray) -> np.ndarray:
        out = np.zeros(values.shape)
        out[:, 1:] = np.where(where[:, 1:], np.diff(values, axis=1) / dt, 0.0)
        return out

    velocity = np.stack((rate(rows.e_m, known), rate(rows.n_m, known)), axis=-1)
    return Motion(velocity, rate(rows.height_m, known), rate(rows.along_m, along_known), known)


def _scaled(distance_m: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(distance_m) / SCALE_M)


def scene_edges(rows: SceneRows, separation: Separation, scenes: Sequence[int] | None = None) -> np.ndarray:
    """``[T, A, A, len(EDGE_FEATURES)]`` float32 (module docstring); 0 wherever i or j is absent. ``scenes``: which scene
    each aircraft is in, several scenes of one airport stacked on ``rows``' aircraft axis (None: one) — only pairs within
    a scene are read, a pair across two stays 0. Every ordered pair over all its steps at once, all pairs together (each
    value is the per-pair formula's, element for element)."""
    aircraft, steps = rows.time_s.shape
    out = np.zeros((steps, aircraft, aircraft, len(EDGE_FEATURES)), dtype=np.float32)
    present = rows.present
    codes = sorted({r for steps in rows.runway for r in steps if r is not None})
    runway = np.array([[codes.index(r) if r is not None else -1 for r in steps] for steps in rows.runway], dtype=np.int64)
    moving = motion(rows, runway)
    velocity, climb = moving.velocity_mps, moving.climb_mps
    direction = np.arctan2(velocity[..., 0], velocity[..., 1])
    position = np.stack((np.nan_to_num(rows.e_m), np.nan_to_num(rows.n_m)), axis=-1)
    height = np.nan_to_num(rows.height_m)
    index = {name: k for k, name in enumerate(EDGE_FEATURES)}
    step_of, own = np.nonzero(present.T)
    out[step_of, own, own, index["self"]] = 1.0
    group = np.zeros(aircraft, dtype=np.int64) if scenes is None else np.asarray(scenes, dtype=np.int64)
    first_of, second_of = np.nonzero((group[:, None] == group[None, :]) & ~np.eye(aircraft, dtype=bool))
    pair, t = np.nonzero(present[first_of] & present[second_of])
    if not len(pair):
        return out
    i, j = first_of[pair], second_of[pair]
    earlier = np.zeros_like(present)
    earlier[:, 1:] = present[:, :-1]
    # j's last row at or before i's instant; its first row, when after it, carried back
    source = np.where((rows.time_s[j, t] <= rows.time_s[i, t]) | ~earlier[j, t], t, t - 1)
    dt = rows.time_s[i, t] - rows.time_s[j, source]
    framed = moving.known[i, t] & (np.hypot(velocity[i, t, 0], velocity[i, t, 1]) > 0.0)
    known = moving.known[i, t] & moving.known[j, source]
    forward = np.stack((np.sin(direction[i, t]), np.cos(direction[i, t])), axis=-1) * framed[:, None]
    left = np.stack((-forward[:, 1], forward[:, 0]), axis=-1)
    offset = position[j, source] + velocity[j, source] * dt[:, None] - position[i, t]
    dh = height[j, source] + climb[j, source] * dt - height[i, t]
    relative = velocity[j, source] - velocity[i, t]
    edge = np.zeros((len(pair), len(EDGE_FEATURES)), dtype=np.float32)
    edge[:, index["front"]] = _scaled((offset * forward).sum(axis=1))
    edge[:, index["left"]] = _scaled((offset * left).sum(axis=1))
    edge[:, index["height"]] = dh / HEIGHT_SCALE_M
    distance = np.hypot(offset[:, 0], offset[:, 1])
    approach = -(offset * relative).sum(axis=1)
    edge[:, index["closing"]] = np.where(distance > 0.0, approach / np.where(distance > 0.0, distance, 1.0),
                                         0.0) / SPEED_SCALE_MPS
    speed2 = (relative * relative).sum(axis=1)
    cpa = np.where(speed2 > 0.0, np.clip(approach / np.where(speed2 > 0.0, speed2, 1.0), 0.0, CPA_HORIZON_S),
                   CPA_HORIZON_S)
    edge[:, index["cpa_time"]] = cpa / CPA_HORIZON_S
    closest = offset + relative * cpa[:, None]
    edge[:, index["cpa_horizontal"]] = _scaled(np.hypot(closest[:, 0], closest[:, 1]))
    edge[:, index["cpa_vertical"]] = (dh + (climb[j, source] - climb[i, t]) * cpa) / HEIGHT_SCALE_M
    for name in ("closing", "cpa_time", "cpa_horizontal", "cpa_vertical"):
        edge[~known, index[name]] = 0.0
    edge[:, index["motion_unknown"]] = ~known
    along = rows.along_m[j, source] + moving.along_rate_mps[j, source] * dt - rows.along_m[i, t]
    _runways(edge, rows, separation, codes, i, j, runway[i, t], runway[j, source], along, index)
    out[t, i, j] = edge
    return out


def _runways(edge: np.ndarray, rows: SceneRows, separation: Separation, codes: list[str], i: np.ndarray, j: np.ndarray,
             mine: np.ndarray, theirs: np.ndarray, along: np.ndarray, index: dict[str, int]) -> None:
    """The features that read the runways in force (module docstring), into ``edge`` (one row per pair-step: ``i`` →
    ``j``): ``mine`` / ``theirs`` the runway indices on their rows, ``along`` j's position on the clock less i's (NaN
    where either has no runway)."""
    edge[:, index["clock_incomparable"]] = 1.0
    kinds = sorted({c for c in rows.category if c is not None}) + [None]
    kind = np.array([kinds.index(c) for c in rows.category], dtype=np.int64)
    for a, b in {(int(x), int(y)) for x, y in zip(mine, theirs) if x >= 0 and y >= 0}:
        on = (mine == a) & (theirs == b)
        relation = separation.relation(codes[a], codes[b])
        edge[on, index[relation]] = 1.0
        if relation == UNRELATED:
            continue
        edge[on, index["clock_incomparable"]] = 0.0
        edge[on, index["clock_ahead"]] = _scaled(along[on])
        # the required distance reads the two runways and the two categories only
        for mine_kind, their_kind in {(int(x), int(y)) for x, y in zip(kind[i[on]], kind[j[on]])}:
            rows_ab = on & (kind[i] == mine_kind) & (kind[j] == their_kind)
            ahead = along[rows_ab]
            j_leads = separation.distance_nm(codes[b], kinds[their_kind], codes[a], kinds[mine_kind]) * NM_M
            i_leads = separation.distance_nm(codes[a], kinds[mine_kind], codes[b], kinds[their_kind]) * NM_M
            edge[rows_ab, index["required"]] = _scaled(np.where(ahead >= 0.0, j_leads, i_leads))
