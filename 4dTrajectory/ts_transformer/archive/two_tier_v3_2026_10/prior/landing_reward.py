"""The landing reward (prior design §9.3): a sentence the prior said, flown by the executor, earns 1 when the executor
lands it on a runway in the airport's landing direction at the time, else 0; the sentences said to one flight are
compared with each other (the group's mean is the baseline).

**The airport's landing direction** at a flight's first predicted step: the candidate runways with a landing in the
`scene.CONTEXT_WINDOW_S` before it — the landings the prior reads as input (`data.own_context`: the flight's own left
out, the sealed test days never in the pool) — and every candidate whose course is within `DIRECTION_TOLERANCE_DEG` of
one of them; any runway when there was no landing in the window (at night). Not the observed runway: landing where the
flight really landed is imitation, which the data term keeps (design §9.3).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import wrap180
from ts_transformer.prior.data import own_context
from ts_transformer.prior.scene import CONTEXT_WINDOW_S, N_LOOK, Landings, utc_s

#: Runways within this of a runway in use land the same way (opposite ends differ by 180°).
DIRECTION_TOLERANCE_DEG = 90.0
#: MIRROR of the executor's judge's outcome name (`autopilot.judge.OUTCOMES`; the prior never imports the executor).
LANDED = "landed"


def landing_direction(signals: FlightSignals, geometry: AirportGeometry, landings: Landings) -> np.ndarray:
    """``[candidates]`` bool: the runways in the airport's landing direction at the flight's first predicted step."""
    context = own_context(signals, landings)
    first_step = np.array([utc_s(signals.entry_time_utc) + float(signals.time_s[N_LOOK])])
    used = [c.course_deg for c in geometry.candidates
            if context.count_before(first_step, CONTEXT_WINDOW_S, c.ident)[0] > 0]
    if not used:
        return np.ones(len(geometry.candidates), dtype=bool)
    courses = np.array([c.course_deg for c in geometry.candidates])
    return (np.abs(wrap180(courses[:, None] - np.array(used)[None, :])) < DIRECTION_TOLERANCE_DEG).any(axis=1)


def rewards(outcomes: Sequence[str], runways: Sequence[int], allowed: Sequence[np.ndarray]) -> np.ndarray:
    """1 where the sentence landed (the executor's judge) on a runway ``allowed`` for its flight, else 0."""
    return np.array([float(outcome == LANDED and bool(allowed_runways[runway]))
                     for outcome, runway, allowed_runways in zip(outcomes, runways, allowed)])


def group_advantages(rewards_by_flight: np.ndarray) -> np.ndarray:
    """``[flights, K]``: each sentence's reward less its flight's mean (not divided by the spread: a flight with one
    landing in K would weigh as much as an even one)."""
    return rewards_by_flight - rewards_by_flight.mean(axis=1, keepdims=True)
