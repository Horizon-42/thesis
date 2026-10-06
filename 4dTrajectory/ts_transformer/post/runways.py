"""An airport's separation rules on the candidates of the artefact (post-training §3): `inference.runway_schedule`'s
FAA minima for the runway ends of vocabulary §6 item 4, and each aircraft's position on the approach clock.

The approach clock of an aircraft is its runway's threshold position along the course from the airport's common origin
(`Separation.along_nm`) less its distance still before that threshold (`instructions.airport.relative_to_runway`): the
place in the queue that the separation judge and the edge features compare.
"""

from __future__ import annotations

import numpy as np
from geokit import NM_M

from ts_transformer.inference.runway_schedule import Separation, faa_separation
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway

#: `faa_separation` turns distances into times at an approach speed; nothing of stage C reads a time from it (the judge,
#: the edge features and the speed-word mask read distances: `Separation.distance_nm`, the minima), so any speed does.
UNREAD_SPEED_MPS = 70.0


def airport_separation(geometry: AirportGeometry) -> Separation:
    """The FAA arrival minima between ``geometry``'s candidates (by their idents), their thresholds and courses from the
    artefact's candidates (the frame's projection back to latitude and longitude)."""
    targets = {}
    for candidate in geometry.candidates:
        lat, lon = geometry.frame.latlon_from_horizontal(candidate.threshold_e_m, candidate.threshold_n_m)
        targets[candidate.ident] = {"lat": float(lat), "lon": float(lon), "course_deg": candidate.course_deg}
    return faa_separation(targets, speed_mps=UNREAD_SPEED_MPS)


def approach_clock_m(separation: Separation, geometry: AirportGeometry, runway_index: int, e_m, n_m) -> np.ndarray:
    """The position on the approach clock (m) of aircraft at ``(e_m, n_m)`` whose runway is candidate ``runway_index``."""
    candidate = geometry.candidates[runway_index]
    before = relative_to_runway(e_m, n_m, 0.0, 0.0, candidate).before_threshold_m
    return separation.along_nm[candidate.ident] * NM_M - np.asarray(before, dtype=np.float64)
