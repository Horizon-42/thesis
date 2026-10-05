"""Established on the final (post-training §3; D31, D92): a function of one row, the same for every aircraft of a
window, commanded or recorded.

An aircraft is established on the final of its R at a row when (D92):

1. G is false (after "go-around" it is not established until a runway word ends G);
2. it is inside the region of the final of R: inside the FAF and the LPV cone (`prior.procedure.Final.inside`, prior §7
   item 6: the region of the procedure masks, prior D64, and of the rows "on the final", prior D72);
3. its track is within the judge's lined-up angle of the course of R (`VocabularySpec.lined_up_deg`: 30°, 7110.65BB
   5-9-2, TBL 5-9-1).

The track is the direction of the 2 s displacement before the row (`post.motion`), as for every motion of a scene; an
aircraft whose motion is unknown (its row 0) is not established. A row without R in force is not established. Only
the separation judge and the masks read it; it is not an executor law, and it reads no later row (not the capture row,
vocabulary §6 item 3) and no executor state.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import wrap180
from ts_transformer.post.motion import motion
from ts_transformer.post.scene import AircraftAt
from ts_transformer.prior.procedure import Final


def established(aircraft: AircraftAt, go_around: np.ndarray, geometry: AirportGeometry, finals: Sequence[Final],
                spec: VocabularySpec, step_s: float) -> np.ndarray:
    """``[N]`` bool: each aircraft of ``geometry``'s airport established on the final of its R at this row (module
    docstring); ``go_around`` its G, ``finals`` the airport's finals in the candidates' order."""
    if len(finals) != len(geometry.candidates):
        raise ValueError(f"{geometry.code}: one final for each candidate, in their order")
    moving = motion(aircraft.at, aircraft.before, aircraft.known, step_s)
    out = np.zeros(len(aircraft), dtype=bool)
    for k in np.flatnonzero((aircraft.runway_index >= 0) & ~np.asarray(go_around, dtype=bool) & moving.known):
        runway = int(aircraft.runway_index[k])
        course = geometry.candidates[runway].course_deg
        out[k] = (bool(finals[runway].inside(np.array([aircraft.at[k, 0]]), np.array([aircraft.at[k, 1]]))[0])
                  and abs(float(wrap180(moving.track_deg[k] - course))) <= spec.lined_up_deg)
    return out
