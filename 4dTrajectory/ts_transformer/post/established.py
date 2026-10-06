"""Established on the final (post-training §3; D31, D92): a function of one row, the same for every aircraft of a
window, commanded or recorded.

An aircraft is established on the final of its R at a row when (D92):

1. G is false (after "go-around" it is not established until a runway word ends G);
2. it is inside the region of the final of R: inside the FAF and the LPV cone (`prior.procedure.Final.inside`, prior §7
   item 6: the region of the procedure masks, prior D64, and of the rows "on the final", prior D72);
3. its track is within `ESTABLISHED_MAX_ANGLE_DEG` (20°) of the course of R: 7110.65BB 5-9-2, TBL 5-9-1, the row for an
   interception less than 2 NM from the approach gate, which holds inside the FAF (the user's decision, 2026-10-05; the
   judge's lined-up angle, 30°, is the table's other row).

The track is the direction of the 2 s displacement before the row (`prior.inputs.motion`, prior §7 item 2), as for
every motion of a scene; an aircraft whose motion is unknown (its row 0) is not established. A row without R in force is not established. Only
the separation judge and the masks read it; it is not an executor law, and it reads no later row (not the capture row,
vocabulary §6 item 3) and no executor state.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import wrap180
from ts_transformer.prior.inputs import motion
from ts_transformer.post.scene import AircraftAt
from ts_transformer.prior.procedure import Final

#: 7110.65BB 5-9-2 a, TBL 5-9-1, "Distance from interception point to approach gate: Less than 2 miles or triple
#: simultaneous approaches in use — Maximum interception angle: 20 degrees" (`docs/literature/arrival_separation/` §8).
ESTABLISHED_MAX_ANGLE_DEG = 20.0


def established(aircraft: AircraftAt, geometry: AirportGeometry, finals: Sequence[Final], step_s: float) -> np.ndarray:
    """``[N]`` bool: each aircraft of ``geometry``'s airport established on the final of its R at this row (module
    docstring), its G ``aircraft.go_around``; ``finals`` the airport's finals in the candidates' order."""
    if len(finals) != len(geometry.candidates):
        raise ValueError(f"{geometry.code}: one final for each candidate, in their order")
    moving = motion(aircraft.at, aircraft.before, aircraft.known, step_s)
    out = np.zeros(len(aircraft), dtype=bool)
    for k in np.flatnonzero((aircraft.runway_index >= 0) & ~aircraft.go_around & moving.known):
        runway = int(aircraft.runway_index[k])
        course = geometry.candidates[runway].course_deg
        out[k] = (bool(finals[runway].inside(np.array([aircraft.at[k, 0]]), np.array([aircraft.at[k, 1]]))[0])
                  and abs(float(wrap180(moving.track_deg[k] - course))) <= ESTABLISHED_MAX_ANGLE_DEG)
    return out
