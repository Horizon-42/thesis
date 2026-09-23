"""The airport frame, the candidate runways, and where an aircraft is relative to each.

The airport frame is the package's airport-anchored chart (`data.coordinate_frames.AirportENUFrame`,
origin at the airport reference point) — the one projection the data plane already uses.
A candidate runway is one landing threshold, described by geometry only: the model points at a
candidate, it never learns an identifier.

The threshold position and the course are the arrival manifest's ``runway_targets`` — the FAA
CIFP runway geometry the modeling plane's own target is built from
(`flight_scenarios.runway_target.threshold_target_state`), so the line a sentence joins is the
line the models are judged against. Only the runway's geometry is read (position, elevation,
true course); the published threshold-crossing height and glidepath are procedure and stay out.
The runway length comes from the runway configuration. Beside the candidates the geometry keeps
every runway end the harvest builds (`trajectory_data_process.harvest.airports.load_airport`), the
set its landing rule measures parallel runways against (`landing_cross_limit_m`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from flight_scenarios.runway_target import airport_reference_point, airport_runways
from geokit import ft_to_m
from ts_transformer.data.coordinate_frames import AirportENUFrame, AirportReference
from ts_transformer.instructions.words import wrap180


@dataclass(frozen=True)
class RunwayCandidate:
    """One landing threshold in the airport frame (metres, compass degrees true)."""

    ident: str
    threshold_e_m: float
    threshold_n_m: float
    course_deg: float
    elevation_m: float
    length_m: float

    def to_dict(self) -> dict[str, Any]:
        return {"ident": self.ident, "threshold_e_m": self.threshold_e_m, "threshold_n_m": self.threshold_n_m,
                "course_deg": self.course_deg, "elevation_m": self.elevation_m, "length_m": self.length_m}


@dataclass(frozen=True)
class RunwayEnd:
    """One runway end as the harvest builds it: threshold position in the airport frame (metres)
    and true course (compass degrees)."""

    ident: str
    threshold_e_m: float
    threshold_n_m: float
    course_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {"ident": self.ident, "threshold_e_m": self.threshold_e_m, "threshold_n_m": self.threshold_n_m,
                "course_deg": self.course_deg}


@dataclass(frozen=True)
class AirportGeometry:
    code: str
    frame: AirportENUFrame
    candidates: tuple[RunwayCandidate, ...]
    #: Every runway end of the airport as the harvest builds it — a superset of the candidates.
    runway_ends: tuple[RunwayEnd, ...]

    def candidate_index(self, ident: str) -> int:
        for index, candidate in enumerate(self.candidates):
            if candidate.ident == ident.upper():
                return index
        raise KeyError(f"{self.code} has no runway {ident!r}; candidates {[c.ident for c in self.candidates]}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code,
                "reference": {"lat": self.frame.lat0, "lon": self.frame.lon0, "elevation_m": self.frame.alt0},
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "runway_ends": [end.to_dict() for end in self.runway_ends]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AirportGeometry:
        reference = data["reference"]
        frame = AirportENUFrame.for_airport(AirportReference(
            code=data["code"], lat=reference["lat"], lon=reference["lon"], elevation_msl_m=reference["elevation_m"]))
        return cls(code=data["code"], frame=frame,
                   candidates=tuple(RunwayCandidate(**item) for item in data["candidates"]),
                   runway_ends=tuple(RunwayEnd(**item) for item in data["runway_ends"]))


def airport_geometry(code: str, runway_targets: dict[str, dict[str, Any]], harvest_runways: Sequence[Any]) -> AirportGeometry:
    """The airport frame, every landing threshold of ``code`` that the arrival manifest's
    ``runway_targets`` publishes (sorted by ident), and every runway end of ``harvest_runways``
    (the harvest's `Runway` objects: ``ident``, ``lat``, ``lon``, ``course_deg``; sorted by ident)."""
    code = code.upper()
    point = airport_reference_point(code)
    frame = AirportENUFrame.for_airport(AirportReference(code=code, lat=point["lat"], lon=point["lon"],
                                                         elevation_msl_m=point["elevation_m"]))
    lengths = {str(threshold["ident"]).upper(): ft_to_m(float(runway["length_ft"]))
               for runway in airport_runways(code) for threshold in runway["thresholds"]}
    candidates = []
    for ident, target in runway_targets.items():
        e, n = frame.horizontal_from_latlon(float(target["lat"]), float(target["lon"]))
        if ident.upper() not in lengths:
            raise KeyError(f"{code} runway {ident} is published but not in the runway configuration")
        candidates.append(RunwayCandidate(
            ident=ident.upper(), threshold_e_m=float(e), threshold_n_m=float(n),
            course_deg=float(target["course_deg"]) % 360.0, elevation_m=float(target["elevation_msl_m"]),
            length_m=lengths[ident.upper()]))
    candidates.sort(key=lambda item: item.ident)
    ends = []
    for runway in harvest_runways:
        e, n = frame.horizontal_from_latlon(float(runway.lat), float(runway.lon))
        ends.append(RunwayEnd(ident=str(runway.ident).upper(), threshold_e_m=float(e), threshold_n_m=float(n),
                              course_deg=float(runway.course_deg) % 360.0))
    ends.sort(key=lambda item: item.ident)
    missing = {c.ident for c in candidates} - {end.ident for end in ends}
    if missing:
        raise KeyError(f"{code} candidates {sorted(missing)} are not runway ends the harvest builds")
    return AirportGeometry(code=code, frame=frame, candidates=tuple(candidates), runway_ends=tuple(ends))


def landing_cross_limit_m(geometry: AirportGeometry, index: int, limit_m: float, parallel_delta_deg: float) -> float:
    """How far off candidate ``index``'s centreline a threshold crossing may lie and still be a
    landing on it: ``limit_m``, and no more than half the across-course spacing to any runway end of
    the airport (`AirportGeometry.runway_ends`, the harvest's set) whose course is within
    ``parallel_delta_deg`` (a parallel runway).

    MIRROR of `trajectory_data_process.harvest.threshold_event._runway_bracket_cross_limit` (the
    harvest's rule, which the evaluator's observed crossings inherit). It is restated here because
    the harvest's function is private to it and projects in its own runway frame;
    `tests/test_instruction_vocabulary.py` checks the two agree on every candidate of every airport."""
    runway = geometry.candidates[index]
    halves = []
    for other in geometry.runway_ends:
        if other.ident == runway.ident or abs(float(wrap180(runway.course_deg - other.course_deg))) > parallel_delta_deg:
            continue
        separation = abs(float(relative_to_runway(np.array([other.threshold_e_m]), np.array([other.threshold_n_m]),
                                                  np.array([other.course_deg]), np.array([0.0]),
                                                  runway).right_of_course_m[0]))
        if separation > 0.0:
            halves.append(separation / 2.0)
    return min([limit_m, *halves])


@dataclass(frozen=True)
class RunwayRelative:
    """Where the aircraft is relative to one candidate, row by row (all arrays)."""

    #: Along the course, metres before the threshold (positive on the approach side).
    before_threshold_m: np.ndarray
    #: Across the course, metres right of the extended centreline (looking along the course).
    right_of_course_m: np.ndarray
    #: Track minus course, degrees in [-180, 180).
    track_minus_course_deg: np.ndarray
    #: Height above the threshold elevation, metres.
    height_above_threshold_m: np.ndarray


def relative_to_runway(e_m, n_m, track_deg, altitude_m, candidate: RunwayCandidate) -> RunwayRelative:
    course = math.radians(candidate.course_deg)
    along_e, along_n = math.sin(course), math.cos(course)      # unit vector along the course
    de = np.asarray(e_m) - candidate.threshold_e_m
    dn = np.asarray(n_m) - candidate.threshold_n_m
    return RunwayRelative(
        before_threshold_m=-(de * along_e + dn * along_n),
        right_of_course_m=de * along_n - dn * along_e,
        track_minus_course_deg=wrap180(np.asarray(track_deg) - candidate.course_deg),
        height_above_threshold_m=np.asarray(altitude_m) - candidate.elevation_m,
    )
