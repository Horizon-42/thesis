"""Loss of separation between arrivals at one instant (multi-aircraft design §3.2).

The rules are FAA JO 7110.65BB's, encoded once in `inference.runway_schedule`; this module applies them to a set of
aircraft at one time. What the rules need that is not geometry is handed in by the caller, already computed: each
aircraft's runway in force, its position on the approach clock (metres along its runway's landing direction from the
airport's common origin: the threshold's `Separation.along_nm` less the distance still before it), whether it is
established on its final (the capture of the labeller or the executor), and its CWT category (None: the record has no
type). This package does not reach the instruction language (the architecture test), so the caller measures those.

For each pair, by the runways in force (`Separation.relation`) and who is established:

- **in trail on one final** — both established on the same runway, or a pair separated as one: the horizontal distance
  must be at least the one-runway radar minimum (`Separation.same_nm`) and the directly-behind wake minimum (TBL 5-5-1,
  5-5-4 g); the vertical distance does not count (the order gives only radar separation on the same final approach
  course, 5-9-6 a5). The aircraft behind on the approach clock is responsible (on a tie, the one listed later).
- **dependent parallels, both established**: the diagonal minimum (5-9-6 a2 / a3) unless vertically separated; the
  aircraft behind is responsible.
- **independent parallels, both established**: none (5-9-7).
- **everything else** — not both established, runways of other directions (both established or not), a runway not
  said: the terminal radar minimum (3 NM, 5-5-4 a/b) unless vertically separated (5-5-5, 4-5-1 a). Where exactly one of
  the two is established, the other one — joining — is responsible; otherwise both are.

Separately, when an aircraft is over its threshold, the established aircraft next behind it on the same runway (or a
pair separated as one) must be at least the on-approach wake minimum behind it on the approach clock (TBL 5-5-2,
5-5-4 h); an aircraft not yet established is left out (its place on the approach clock is not its place in the queue).
A category is None when the record has no type: in trail, such a pair is judged on the radar minimum alone and says so
(``wake_known``); at the threshold it is not judged — the caller counts those aircraft. Distances are metres; the
vertical is between the two heights as given.

`Traffic` holds its own contract: an aircraft has a position on the approach clock exactly when a runway is in force,
and an established aircraft has one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from geokit import FT_M, NM_M

from ts_transformer.inference.runway_schedule import (
    CWT_DIRECTLY_BEHIND_NM,
    DEPENDENT,
    FAA_RADAR_NM,
    FAA_VERTICAL_FT,
    INDEPENDENT,
    Separation,
)

IN_TRAIL, DIAGONAL, RADAR_OR_VERTICAL, AT_THRESHOLD = "in_trail", "diagonal", "radar_or_vertical", "at_threshold"
#: The relation of a pair one of which has no runway in force yet (beside `runway_schedule`'s five).
NO_RUNWAY = "no_runway"


@dataclass(frozen=True)
class Traffic:
    """The aircraft in the scene at one instant (only those that count: present and not ended)."""

    e_m: np.ndarray                     # [N] airport frame
    n_m: np.ndarray                     # [N]
    height_m: np.ndarray                # [N]
    runway: tuple[str | None, ...]      # the runway in force; None before one is said
    along_m: np.ndarray                 # [N] position on the approach clock; NaN where no runway is in force
    established: np.ndarray             # [N] bool: established on the final of its runway in force
    category: tuple[str | None, ...]    # CWT category; None: the record has no type

    def __post_init__(self) -> None:
        count = len(self.e_m)
        if not all(len(field) == count for field in (self.n_m, self.height_m, self.runway, self.along_m,
                                                        self.established, self.category)):
            raise ValueError("every field of a Traffic holds one value per aircraft")
        said = np.array([runway is not None for runway in self.runway], dtype=bool)
        if not np.array_equal(said, np.isfinite(self.along_m)):
            raise ValueError("an aircraft is on the approach clock exactly when a runway is in force")
        if np.any(np.asarray(self.established, dtype=bool) & ~said):
            raise ValueError("an established aircraft has a runway in force")


@dataclass(frozen=True)
class Loss:
    """``i < j`` index the pair in `Traffic` order, except at the threshold, where ``i`` is the leader over it and ``j``
    its follower. ``wake_known`` is False only in trail with a category unknown (the other kinds need none)."""

    i: int
    j: int
    kind: str                           # IN_TRAIL, DIAGONAL, RADAR_OR_VERTICAL or AT_THRESHOLD
    relation: str                       # `runway_schedule`'s relation of the two runways in force, or NO_RUNWAY
    required_m: float                   # the horizontal (or, at the threshold, along-clock) minimum
    distance_m: float                   # the horizontal (or along-clock) distance found
    vertical_m: float
    responsible: tuple[int, ...]
    wake_known: bool


def _wake_m(table: dict[tuple[str, str], float], leader: str | None, follower: str | None) -> tuple[float, bool]:
    if leader is None or follower is None:
        return 0.0, False
    return table.get((leader, follower), 0.0) * NM_M, True     # a blank cell of the table sets no wake minimum


def losses(traffic: Traffic, separation: Separation) -> list[Loss]:
    """Every pair that has lost separation at this instant (module docstring)."""
    one_runway_m, radar_m, vertical_min_m = separation.same_nm * NM_M, FAA_RADAR_NM * NM_M, FAA_VERTICAL_FT * FT_M
    out = []
    count = len(traffic.e_m)
    for i in range(count):
        for j in range(i + 1, count):
            horizontal = math.hypot(traffic.e_m[j] - traffic.e_m[i], traffic.n_m[j] - traffic.n_m[i])
            vertical = abs(float(traffic.height_m[j] - traffic.height_m[i]))
            ri, rj = traffic.runway[i], traffic.runway[j]
            relation = separation.relation(ri, rj) if ri is not None and rj is not None else NO_RUNWAY
            both = bool(traffic.established[i] and traffic.established[j])     # established: a runway is in force
            ahead, behind = (i, j) if traffic.along_m[i] >= traffic.along_m[j] else (j, i)
            if both and separation.one_runway(ri, rj):
                wake, known = _wake_m(CWT_DIRECTLY_BEHIND_NM, traffic.category[ahead], traffic.category[behind])
                required = max(one_runway_m, wake)
                if horizontal < required:
                    out.append(Loss(i, j, IN_TRAIL, relation, required, horizontal, vertical, (behind,), known))
                continue
            if both and relation == INDEPENDENT:
                continue
            if both and relation == DEPENDENT:
                required = separation.diagonal_nm[frozenset((ri, rj))] * NM_M
                if horizontal < required and vertical < vertical_min_m:
                    out.append(Loss(i, j, DIAGONAL, relation, required, horizontal, vertical, (behind,), True))
                continue
            if horizontal < radar_m and vertical < vertical_min_m:
                if traffic.established[i] != traffic.established[j]:
                    responsible = (j,) if traffic.established[i] else (i,)
                else:
                    responsible = (i, j)
                out.append(Loss(i, j, RADAR_OR_VERTICAL, relation, radar_m, horizontal, vertical, responsible, True))
    return out


def next_behind(traffic: Traffic, leader: int, separation: Separation) -> int | None:
    """The established aircraft next behind ``leader`` on the approach clock, on its runway or one separated as one."""
    runway = traffic.runway[leader]
    behind = [k for k in range(len(traffic.e_m))
              if k != leader and traffic.established[k] and separation.one_runway(runway, traffic.runway[k])
              and traffic.along_m[k] < traffic.along_m[leader]]
    return max(behind, key=lambda k: traffic.along_m[k]) if behind else None


def wake_at_threshold(traffic: Traffic, leader: int, separation: Separation) -> Loss | None:
    """``leader`` is over its threshold now: the established aircraft next behind it on the same runway (or a pair
    separated as one) must be the on-approach wake minimum behind it on the approach clock (TBL 5-5-2, 5-5-4 h)."""
    follower = next_behind(traffic, leader, separation)
    if follower is None:
        return None
    runway = traffic.runway[leader]
    required, known = _wake_m(separation.wake_nm, traffic.category[leader], traffic.category[follower])
    gap = float(traffic.along_m[leader] - traffic.along_m[follower])
    if not known or gap >= required:
        return None
    vertical = abs(float(traffic.height_m[follower] - traffic.height_m[leader]))
    return Loss(leader, follower, AT_THRESHOLD, separation.relation(runway, traffic.runway[follower]), required, gap,
                vertical, (follower,), True)
