"""The edge features a scene prior reads between aircraft (multi-aircraft design §2.5): for every ordered pair i → j at every
step, what j is to i — where it is in i's frame, where on the approach clock, how their runways relate, how fast they
close, their closest point of approach, and how far apart the rules would have them.

Computed here because it needs the separation rules (`runway_schedule`), which the prior's package does not reach (the
same placement as the loop's masks, design §9 item 18); what needs the runway geometry (each aircraft's position on the
approach clock and its rate along it) is handed in by the caller.

**When** (§2.1, §2.5): at each step, at aircraft i's own row time. Aircraft j is taken from its last row at or before that
instant and carried straight on to it (at most one step: rows hang within half a step of the step), so only the past is
read — except for j's very first row, which may lie up to a step after i's instant and is carried back to it. Velocities
are each row's own ground speed, track and vertical rate (the design's "two rows' positions" would leave a first row
without one; these are the same instant's quantities, and a flown aircraft has them too).

**The features** (`EDGE_FEATURES`, in order; horizontal distances scaled by asinh(d / `SCALE_M`), heights by
`HEIGHT_SCALE_M`, speeds by `SPEED_SCALE_MPS`, the time by `CPA_HORIZON_S`):

- ``self`` — 1 for i → i, where everything else is 0;
- ``front``, ``left``, ``height`` — j in i's frame: along i's track, to its left, j's height less i's;
- ``clock_ahead``, ``clock_incomparable`` — j's position on the approach clock less i's; 0 and the flag 1 when the two
  runways in force have other directions, or one has none yet;
- ``same``, ``single``, ``dependent``, ``independent``, ``unrelated`` — the relation of the runways in force
  (`Separation.relation`), one-hot; all 0 when one has none yet;
- ``closing`` — the rate the horizontal distance closes at (positive: closing);
- ``cpa_time``, ``cpa_horizontal``, ``cpa_vertical`` — both flying straight on at their present velocities: when the
  horizontal distance is least (clipped to [0, `CPA_HORIZON_S`]), that distance, and j's height less i's then;
- ``required`` — the distance the rules require were the two leader and follower (`Separation.distance_nm`: the one
  ahead on the approach clock leads; the one-runway radar minimum or TBL 5-5-2, a dependent pair's stagger, 0 otherwise
  and when not comparable), the distance design §2.5 gives the model (user 2026-09-27, §9 item 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import (
    DEPENDENT, FAA_RADAR_NM, INDEPENDENT, SAME, SINGLE, UNRELATED, Separation,
)

RELATIONS = (SAME, SINGLE, DEPENDENT, INDEPENDENT, UNRELATED)
EDGE_FEATURES = ("self", "front", "left", "height", "clock_ahead", "clock_incomparable", *RELATIONS, "closing",
                 "cpa_time", "cpa_horizontal", "cpa_vertical", "required")
#: The horizontal scale: the radar minimum (design §2.5: judgements between aircraft are made around it).
SCALE_M = FAA_RADAR_NM * NM_M
HEIGHT_SCALE_M = 1_000.0
SPEED_SCALE_MPS = 100.0
CPA_HORIZON_S = 120.0


@dataclass(frozen=True)
class SceneRows:
    """A scene's aircraft on its steps, ``[A, T]`` each (NaN where absent): the time of the row hung on the step (epoch
    seconds), position (airport frame), ground speed, track (compass degrees), vertical rate, position on the approach
    clock and rate along it (NaN without a runway in force). ``runway`` holds the runway in force per aircraft and step —
    in force BEFORE the step's words, as the prior reads its words (`prior.data.sentence_steps`' ``in_force``: said at
    an earlier row; the words of the step it predicts never leak into its edges) — None before any is said or where
    absent. ``category``: the CWT category per aircraft; None where the record has no type, where the required distance
    is the radar minimum alone (as the judge reads such a pair, `separation.losses` ``wake_known``; the caller counts
    them). Refused unless a present step's state is finite, and a runway is given exactly where the clock is."""

    time_s: np.ndarray
    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray
    ground_speed_mps: np.ndarray
    track_deg: np.ndarray
    vertical_rate_mps: np.ndarray
    along_m: np.ndarray
    along_rate_mps: np.ndarray
    runway: Sequence[Sequence[str | None]]
    category: Sequence[str | None]

    def __post_init__(self) -> None:
        present = self.present
        for name in ("e_m", "n_m", "height_m", "ground_speed_mps", "track_deg", "vertical_rate_mps"):
            if not np.isfinite(getattr(self, name)[present]).all():
                raise ValueError(f"a present aircraft-step has a non-finite {name}")
        said = np.array([[runway is not None for runway in steps] for steps in self.runway], dtype=bool)
        if said.shape != present.shape or (said & ~present).any():
            raise ValueError("a runway is given per aircraft and step, and only where the aircraft is present")
        if not (np.array_equal(said, np.isfinite(self.along_m)) and np.array_equal(said, np.isfinite(self.along_rate_mps))):
            raise ValueError("an aircraft-step is on the approach clock exactly when a runway is in force")

    @property
    def present(self) -> np.ndarray:
        return ~np.isnan(self.time_s)


def _scaled(distance_m: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(distance_m) / SCALE_M)


def scene_edges(rows: SceneRows, separation: Separation) -> np.ndarray:
    """``[T, A, A, len(EDGE_FEATURES)]`` float32 (module docstring); 0 wherever i or j is absent. Each ordered pair over
    all its steps at once."""
    aircraft, steps = rows.time_s.shape
    out = np.zeros((steps, aircraft, aircraft, len(EDGE_FEATURES)), dtype=np.float32)
    present = rows.present
    track = np.radians(np.nan_to_num(rows.track_deg))
    speed = np.nan_to_num(rows.ground_speed_mps)
    velocity = np.stack((speed * np.sin(track), speed * np.cos(track)), axis=-1)            # [A, T, 2]
    position = np.stack((np.nan_to_num(rows.e_m), np.nan_to_num(rows.n_m)), axis=-1)
    height, climb = np.nan_to_num(rows.height_m), np.nan_to_num(rows.vertical_rate_mps)
    codes = sorted({r for steps in rows.runway for r in steps if r is not None})
    runway = np.array([[codes.index(r) if r is not None else -1 for r in steps] for steps in rows.runway], dtype=np.int64)
    index = {name: k for k, name in enumerate(EDGE_FEATURES)}
    column = np.arange(steps)
    for i in range(aircraft):
        out[present[i], i, i, index["self"]] = 1.0
        for j in range(aircraft):
            both = np.flatnonzero(present[i] & present[j]) if j != i else np.zeros(0, dtype=np.int64)
            if not len(both):
                continue
            earlier = np.zeros(steps, dtype=bool)
            earlier[1:] = present[j, :-1]
            # j's last row at or before i's instant; its first row, when after it, carried back
            source = np.where((rows.time_s[j] <= rows.time_s[i]) | ~earlier, column, column - 1)[both]
            dt = rows.time_s[i, both] - rows.time_s[j, source]
            forward = np.stack((np.sin(track[i, both]), np.cos(track[i, both])), axis=-1)
            left = np.stack((-forward[:, 1], forward[:, 0]), axis=-1)
            offset = position[j, source] + velocity[j, source] * dt[:, None] - position[i, both]
            dh = height[j, source] + climb[j, source] * dt - height[i, both]
            relative = velocity[j, source] - velocity[i, both]
            edge = out[both, i, j]
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
            edge[:, index["cpa_vertical"]] = (dh + (climb[j, source] - climb[i, both]) * cpa) / HEIGHT_SCALE_M
            _runways(edge, rows, separation, codes, i, j, both, source, runway, dt, index)
            out[both, i, j] = edge
    return out


def _runways(edge: np.ndarray, rows: SceneRows, separation: Separation, codes: list[str], i: int, j: int,
             both: np.ndarray, source: np.ndarray, runway: np.ndarray, dt: np.ndarray, index: dict[str, int]) -> None:
    """The features that read the runways in force (module docstring), into ``edge`` (its rows are ``both``)."""
    mine, theirs = runway[i, both], runway[j, source]
    edge[:, index["clock_incomparable"]] = 1.0
    along = rows.along_m[j, source] + rows.along_rate_mps[j, source] * dt - rows.along_m[i, both]
    for a, b in {(int(x), int(y)) for x, y in zip(mine, theirs) if x >= 0 and y >= 0}:
        rows_ab = (mine == a) & (theirs == b)
        relation = separation.relation(codes[a], codes[b])
        edge[rows_ab, index[relation]] = 1.0
        if relation == UNRELATED:
            continue
        edge[rows_ab, index["clock_incomparable"]] = 0.0
        ahead = along[rows_ab]
        edge[rows_ab, index["clock_ahead"]] = _scaled(ahead)
        j_leads = separation.distance_nm(codes[b], rows.category[j], codes[a], rows.category[i]) * NM_M
        i_leads = separation.distance_nm(codes[a], rows.category[i], codes[b], rows.category[j]) * NM_M
        edge[rows_ab, index["required"]] = _scaled(np.where(ahead >= 0.0, j_leads, i_leads))
