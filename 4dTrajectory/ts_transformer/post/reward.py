"""The reward of the post-training (post-training §2 item 2; D30, D91, D105): only from the outcome.

| The outcome of the sentence | Reward |
|---|---|
| ``landed``, no go-around, on a runway of the airport's present landing direction | 1 |
| ``landed`` after n go-arounds, on such a runway | 0.9ⁿ |
| every other outcome; every loss of separation that the commanded aircraft answers for | 0 |

**The present landing direction**: a runway within 90° of a runway with a landing in the 30 min before the first
predicted step — the landings of the window's scene (D105), without the commanded aircraft's own (D31), counted by the
prior's index (`LandingIndex.counts_before`, whose window is the same 30 min, prior §7 item 2).
"""

from __future__ import annotations

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import wrap180
from ts_transformer.prior.landings import LandingIndex

#: D30: the reward of a landing after n go-arounds is ``GO_AROUND_FACTOR ** n``.
GO_AROUND_FACTOR = 0.9
#: §2 item 2: a runway within this of a runway with a landing in the 30 min before the first predicted step.
PRESENT_DIRECTION_DEG = 90.0
#: MIRROR of the judge's outcome of a landing (`autopilot.judge.OUTCOMES`; `post/` does not import `autopilot/`, the
#: architecture test): `tests/test_post_window_loop.py` pins the two.
LANDED = "landed"


def present_runways(landings: LandingIndex, geometry: AirportGeometry, time_s: float, without: str) -> np.ndarray:
    """``[candidates]`` bool: the runways of the present landing direction at ``time_s`` (module docstring), from
    ``landings`` without the flight ``without``'s own. With no landing in the 30 min, no runway is of it."""
    counts = landings.counts_before(np.array([time_s]), without=without)[0]
    courses = {c.ident: c.course_deg for c in geometry.candidates}
    landed = [courses[ident] for ident, count in zip(landings.runways, counts) if count > 0]
    return np.array([any(abs(float(wrap180(c.course_deg - course))) <= PRESENT_DIRECTION_DEG for course in landed)
                     for c in geometry.candidates], dtype=bool)


def reward(outcome: str, landed_runway: int | None, go_arounds: int, present: np.ndarray,
           lost_separation: bool) -> float:
    """D30's reward (module docstring) of a sentence: the judge's ``outcome``, the candidate it landed on (None without a
    landing), its go-arounds, the runways of the present landing direction, and whether it ended in a loss of separation
    that the commanded aircraft answers for."""
    if lost_separation or outcome != LANDED:
        return 0.0
    if landed_runway is None:
        raise ValueError("a landed sentence names the runway it landed on")
    return GO_AROUND_FACTOR ** go_arounds if bool(present[landed_runway]) else 0.0
