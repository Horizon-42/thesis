"""Loss of separation between arrivals at one instant (multi-aircraft design §3.2).

The rules are FAA JO 7110.65BB's, encoded once in `inference.runway_schedule`; this module applies them to a set of
aircraft at one time. What the rules need that is not geometry is handed in by the caller, already computed: each
aircraft's runway in force, its position on the approach clock (metres along its runway's landing direction from the
airport's common origin: the threshold's `Separation.along_nm` less the distance still before it), its track less that
runway's course and its distance right of the runway's extended centreline (both signed), whether it is established
on its final (the capture of the labeller or the executor), and its CWT category (None: the record has no type). This package does not reach the instruction language
(the architecture test), so the caller measures those.

Two readings of the rules (design §3.2; user 2026-09-27, provisional): the closed loop's checks and the reward use
`VISUAL`, and `IFR` is reported beside it.

- `IFR` — the order's instrument rules as written, below.
- `VISUAL` — the visual-approach reading: in good weather controllers clear arrivals to several runways for visual
  approaches (7-4-4 c). The reading assumes those clearances, and never visual separation (7-2-1), which takes a
  traffic-in-sight report and an instruction to keep it — words the vocabulary does not have (user 2026-09-27). From
  `IFR` it differs in two places, and it never judges a pair `IFR` would not:
  - dependent and independent parallels (2,500 ft apart or more): no minimum between the two runways once BOTH aircraft
    are turned in (`_turned_in`; 7-4-4 c2 a / c3 a: approved separation until each is on a heading "which will
    intercept the extended centerline of the runway at an angle not greater than 30 degrees"; then c2 c / c3 c: no
    other separation with the adjacent centreline); before that, the `IFR` rule. Turned in: its track within
    `FAA_VISUAL_INTERCEPT_MAX_DEG` of its course (the heading read as the track: no wind), and on its own side of the
    midline between the two centrelines. The midline is our reading of "will intercept": an aircraft that has overshot
    its centreline into the other final's half, or is still crossing that half toward its own (c2 / c3 b, d), has not
    intercepted; one drifting a few degrees off its own centreline, or turning in from outside, has — reading it as
    "heading toward its centreline" made every aircraft drifting a degree away from it a loss (readout §2);
  - two aircraft both ESTABLISHED on finals of runways of other directions are not judged (7-4-4 c4; the
    crossing-runway gate at the threshold, 3-10-4, is not modelled).
  A pair under 2,500 ft is judged as one runway, as under `IFR`: 7-4-4 c1 (as amended by N JO 7110.805) clears a
  visual approach there only when the succeeding aircraft keeps visual separation (c1 b). Not encoded: the same-side
  cases of c2 / c3 (b) and (d), which hold the succeeding aircraft until it intercepts the farther centreline or is
  established on the nearer one, and b1's "targets must not touch" (it needs a display scale). The recorded traffic
  breaks the `IFR` reading on parallel and crossing runways every hour at four of the five airports (readout
  `2026-09-27_parallel_runway_separation.md`); ADS-B does not say which approach was a visual one, nor where visual
  separation was applied.

For each pair under `IFR`, by the runways in force (`Separation.relation`) and who is established:

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

Separately, under both readings, when an aircraft is over its threshold, the established aircraft next behind it on the
same runway (or a pair separated as one) must be at least the on-approach wake minimum behind it on the approach clock
(TBL 5-5-2, 5-5-4 h); an aircraft not yet established is left out (its place on the approach clock is not its place in the queue).
A category is None when the record has no type: in trail, such a pair is judged on the radar minimum alone and says so
(``wake_known``); at the threshold it is not judged — the caller counts those aircraft. Distances are metres; the
vertical is between the two heights as given.

`Traffic` holds its own contract: an aircraft has a position on the approach clock, a track less its course (in
[−180°, 180°]) and a distance off its centreline exactly when a runway is in force, and an established aircraft has one.
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
    FAA_VISUAL_INTERCEPT_MAX_DEG,
    INDEPENDENT,
    UNRELATED,
    Separation,
)

IN_TRAIL, DIAGONAL, RADAR_OR_VERTICAL, AT_THRESHOLD = "in_trail", "diagonal", "radar_or_vertical", "at_threshold"
#: The two readings (module docstring).
IFR, VISUAL = "ifr", "visual"
READINGS = (IFR, VISUAL)
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
    track_minus_course_deg: np.ndarray  # [N] its track less its runway's course, −180–180° (+: turned right); NaN: no runway
    right_of_course_m: np.ndarray       # [N] its distance right of its runway's extended centreline; NaN: no runway
    established: np.ndarray             # [N] bool: established on the final of its runway in force
    category: tuple[str | None, ...]    # CWT category; None: the record has no type

    def __post_init__(self) -> None:
        count = len(self.e_m)
        if not all(len(field) == count for field in (self.n_m, self.height_m, self.runway, self.along_m,
                                                        self.track_minus_course_deg, self.right_of_course_m,
                                                        self.established, self.category)):
            raise ValueError("every field of a Traffic holds one value per aircraft")
        said = np.array([runway is not None for runway in self.runway], dtype=bool)
        if not all(np.array_equal(said, np.isfinite(field))
                   for field in (self.along_m, self.track_minus_course_deg, self.right_of_course_m)):
            raise ValueError("an aircraft is on the approach clock, and has a track and a distance off its runway's "
                             "course, exactly when a runway is in force")
        if np.any(np.abs(self.track_minus_course_deg[said]) > 180.0):
            raise ValueError("a track less its course lies in [-180, 180] degrees")
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


def in_trail_m(separation: Separation, ahead: str | None, behind: str | None) -> tuple[float, bool]:
    """In trail on one final (both established on one runway, or a pair separated as one): the horizontal minimum — the
    one-runway radar minimum or the directly-behind wake minimum (TBL 5-5-1, 5-5-4 g), whichever is larger — and whether
    both categories are known (unknown: the radar minimum alone)."""
    wake, known = _wake_m(CWT_DIRECTLY_BEHIND_NM, ahead, behind)
    return max(separation.same_nm * NM_M, wake), known


def _turned_in(traffic: Traffic, k: int, other: str, separation: Separation) -> bool:
    """7-4-4 c2 (a)(1) / c3 (a)(1), beside the final of the parallel runway ``other``: aircraft ``k``'s track is within
    `FAA_VISUAL_INTERCEPT_MAX_DEG` of its course, and it is on its own side of the midline between the two centrelines
    (module docstring)."""
    other_right_m = separation.right_nm[(traffic.runway[k], other)] * NM_M      # the other centreline, right of its own
    toward_other_m = float(traffic.right_of_course_m[k]) * math.copysign(1.0, other_right_m)
    return (abs(float(traffic.track_minus_course_deg[k])) <= FAA_VISUAL_INTERCEPT_MAX_DEG
            and toward_other_m < 0.5 * abs(other_right_m))


def losses(traffic: Traffic, separation: Separation, reading: str) -> list[Loss]:
    """Every pair that has lost separation at this instant under ``reading`` (module docstring)."""
    if reading not in READINGS:
        raise ValueError(f"reading {reading!r} is not one of {READINGS}")
    radar_m, vertical_min_m = FAA_RADAR_NM * NM_M, FAA_VERTICAL_FT * FT_M
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
            if reading == VISUAL:
                if (relation in (DEPENDENT, INDEPENDENT) and _turned_in(traffic, i, rj, separation)
                        and _turned_in(traffic, j, ri, separation)):
                    continue            # 7-4-4 c2 / c3: both turned in, at 30° or less, on their own sides
                if relation == UNRELATED and both:
                    continue            # established on finals of other directions: 3-10-4 not modelled
            if both and separation.one_runway(ri, rj):
                required, known = in_trail_m(separation, traffic.category[ahead], traffic.category[behind])
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
    separated as one) must be the on-approach wake minimum behind it on the approach clock (TBL 5-5-2, 5-5-4 h); the
    same under both readings."""
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
