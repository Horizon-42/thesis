"""What the separation rules need of an aircraft beside its position, measured against its runway.

For an aircraft at ``(n, e)`` with a track (compass degrees) and its runway in force: the distance
still before the threshold along the landing direction, its distance right of the extended
centreline, its track less the runway course, and whether it is ESTABLISHED on the final — the
two-tier design's D92, read here as geometry only: inside the FAF and the LPV cone of the runway,
and its track within 20° of the course. (D92 also needs "no go-around": the optimizer flies none, and
a recorded go-around is not marked here; design §1.2.) A runway without a coded FAF has no final
region, so no aircraft is established on it; :func:`runway_frame` reports which.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flight_scenarios.fas_geometry import course_halfwidth_m, fas_course_geometry
from flight_scenarios.procedure_final import final_approach_fix, rnav_gps_procedure_path

from .frame import TargetFrame

#: Track within this of the course to be established inside the FAF. MIRROR of the two-tier
#: decision D92 (7110.65BB 5-9-2 a, TBL 5-9-1), `ts_transformer/docs/two_tier/design/post_training.md`.
ESTABLISHED_TRACK_MAX_DEG = 20.0

_FAS = fas_course_geometry()      # the runway length is not carried: the 9023 ft FPAP floor


@dataclass(frozen=True)
class Runway:
    ident: str
    threshold_n: float
    threshold_e: float
    course_deg: float               # compass: 0 = north, clockwise
    d_faf_m: float | None           # the FAF's distance before the threshold; None: not coded


def runway_frame(runway_targets: dict, frame: TargetFrame, airport: str, procedure_root: str | Path) -> dict[str, Runway]:
    """Every runway of the manifest in the window frame, with its coded FAF distance."""
    runways = {}
    for ident, target in runway_targets.items():
        n, e = frame.to_ne(target["lat"], target["lon"])
        try:
            rnav_gps_procedure_path(airport, ident, root=procedure_root)
        except ValueError:              # the runway has no RNAV(GPS) procedure: no coded final
            d_faf = None
        else:                           # it has one: its FAF must read (a bad document raises)
            d_faf = final_approach_fix(airport, ident, root=procedure_root).distance_to_threshold_m
        runways[ident] = Runway(ident, float(n), float(e), float(target["course_deg"]), d_faf)
    return runways


def wrap_deg(angle):
    """An angle (degrees) in [-180, 180)."""
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def relative(runway: Runway, n, e, track_deg):
    """``(distance before the threshold, distance right of the centreline, track less course)``."""
    course = np.radians(runway.course_deg)
    dn, de = np.asarray(n) - runway.threshold_n, np.asarray(e) - runway.threshold_e
    along = de * np.sin(course) + dn * np.cos(course)          # + past the threshold
    right = de * np.cos(course) - dn * np.sin(course)
    return -along, right, wrap_deg(np.asarray(track_deg) - runway.course_deg)


def established(runway: Runway, before_m, right_m, track_minus_course_deg):
    """Two-tier D92 as geometry: inside the FAF and the LPV cone, track within 20° of the course. The
    cone (8260.58D) continues past the threshold, narrowing to its apex at the GARP, so an aircraft
    over its threshold — a replay ends a few centimetres past it — is still established."""
    before_m = np.asarray(before_m)
    if runway.d_faf_m is None:
        return np.zeros(before_m.shape, dtype=bool)
    return ((before_m <= runway.d_faf_m)
            & (np.abs(right_m) <= course_halfwidth_m(before_m, _FAS))
            & (np.abs(track_minus_course_deg) <= ESTABLISHED_TRACK_MAX_DEG))
