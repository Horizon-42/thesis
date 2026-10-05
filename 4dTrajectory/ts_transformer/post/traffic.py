"""The separation judge on v4 (post-training §3, §8 C3; D92): the inputs of `inference.separation` from the aircraft of
a window at one step, and the event of a window.

`inference.separation` reads, for each aircraft, its runway in force, its position on the approach clock, its track
less the course of that runway, its distance right of the runway's extended centreline, whether it is established and
its CWT category. Here they come from vocabulary §6 item 4 (`instructions.airport.relative_to_runway` on the
candidates of the artefact), from "established" (`post.established`, D92) and from the 2 s displacement before the row
(`post.motion`); the reading is `VISUAL` (7110.65BB 7-4-4 c with visual approach clearances, never visual separation).

The commanded aircraft is aircraft 0 of the `Traffic`, the other aircraft follow in the scene's order. **The event of a
window** is the first loss of separation for which the commanded aircraft answers (`commanded_loss`): a pair that
`separation.losses` finds lost with the commanded aircraft among the responsible, or, when another aircraft is over
its threshold, the on-approach wake minimum behind it with the commanded aircraft as its follower
(`separation.wake_at_threshold`). A recorded aircraft is over its threshold at its last step in the air
(`scene.AircraftAt.last_step`: its last row is the last before the observed crossing, so the crossing falls within one
Δ after that step).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.inference.separation import VISUAL, Loss, Traffic, losses, wake_at_threshold
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.post.established import established
from ts_transformer.post.motion import motion
from ts_transformer.post.runways import approach_clock_m
from ts_transformer.post.scene import AircraftAt
from ts_transformer.prior.procedure import Final

#: The reading of the separation judge in the loops of stage C (post-training §3).
READING = VISUAL


def joined(own: AircraftAt, others: AircraftAt) -> AircraftAt:
    """The commanded aircraft (one) and the other aircraft of a step as one set, the commanded aircraft first."""
    if len(own) != 1:
        raise ValueError(f"one commanded aircraft, not {len(own)}")
    return AircraftAt(keys=own.keys + others.keys, at=np.concatenate((own.at, others.at)),
                      before=np.concatenate((own.before, others.before)),
                      known=np.concatenate((own.known, others.known)),
                      runway_index=np.concatenate((own.runway_index, others.runway_index)),
                      category=own.category + others.category, last_step=np.concatenate((own.last_step, others.last_step)),
                      go_around=np.concatenate((own.go_around, others.go_around)))


def traffic(aircraft: AircraftAt, geometry: AirportGeometry, separation: Separation, finals: Sequence[Final],
            step_s: float) -> Traffic:
    """`separation.Traffic` of ``aircraft`` at one step (module docstring), each one's G its ``go_around``."""
    moving = motion(aircraft.at, aircraft.before, aircraft.known, step_s)
    count = len(aircraft)
    along, off_course, right = np.full(count, np.nan), np.full(count, np.nan), np.full(count, np.nan)
    runways: list[str | None] = [None] * count
    for k in np.flatnonzero(aircraft.runway_index >= 0):
        if not moving.known[k]:
            raise ValueError(f"{aircraft.keys[k]} has a runway in force and no motion: R is in force only after the "
                             "first predicted step, 16 s after row 0")
        index = int(aircraft.runway_index[k])
        relative = relative_to_runway(aircraft.at[k, 0], aircraft.at[k, 1], moving.track_deg[k], 0.0,
                                      geometry.candidates[index])
        runways[k] = geometry.candidates[index].ident
        along[k] = float(approach_clock_m(separation, geometry, index, aircraft.at[k, 0], aircraft.at[k, 1]))
        off_course[k] = float(relative.track_minus_course_deg)
        right[k] = float(relative.right_of_course_m)
    return Traffic(e_m=aircraft.at[:, 0].copy(), n_m=aircraft.at[:, 1].copy(), height_m=aircraft.at[:, 2].copy(),
                   runway=tuple(runways), along_m=along, track_minus_course_deg=off_course, right_of_course_m=right,
                   established=established(aircraft, geometry, finals, step_s), category=aircraft.category)


def commanded_loss(scene: Traffic, over_threshold: np.ndarray, separation: Separation) -> Loss | None:
    """The first loss of separation at this step for which the commanded aircraft (aircraft 0) answers (module
    docstring), None without one; ``over_threshold`` ``[N]`` bool, the aircraft over their thresholds now."""
    for loss in losses(scene, separation, READING):
        if 0 in loss.responsible:
            return loss
    for leader in np.flatnonzero(over_threshold):
        loss = wake_at_threshold(scene, int(leader), separation)
        if loss is not None and 0 in loss.responsible:
            return loss
    return None
