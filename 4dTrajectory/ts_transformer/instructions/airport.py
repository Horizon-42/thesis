"""The airport frame, the candidate runways, and where an aircraft is relative to each.

The airport frame is the package's airport-anchored chart (`data.coordinate_frames.AirportENUFrame`,
origin at the airport reference point) — the one projection the data plane already uses.
A candidate runway is one landing threshold, described by geometry only: the model points at a
candidate, it never learns an identifier.

The candidates are the airport's runway ends with a published vertical path in the FAA CIFP (D78: the LPV line, or the
LNAV/VNAV line where no LPV line is published, D61), decided by no flight; their threshold position, course and elevation
are the harvest's runway geometry (`trajectory_data_process.harvest.airports.load_airport`), the same that the arrival
manifest's ``runway_targets`` — the modeling plane's own target — is written from, so the line a sentence joins is the
line the models are judged against (`instructions.signals.signals_from_series` checks a flight's target against it). The runway length comes from the runway configuration. Each
candidate also carries its published vertical path (`VerticalPath`: the threshold crossing height,
the glidepath angle and the decision altitude, D61), read once from the harvest's runway data when the
geometry is built: the judge's decision-altitude check reads it, the executor's laws never do (D9). Beside the candidates the geometry keeps
every runway end the harvest builds (`trajectory_data_process.harvest.airports.load_airport`), the
set its landing rule measures parallel runways against (`landing_cross_limit_m`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from flight_scenarios.runway_target import airport_reference_point, airport_runways
from geokit import ft_to_m, wgs84_curvature_radii
from ts_transformer.data.coordinate_frames import AirportENUFrame, AirportReference
from ts_transformer.instructions.words import wrap180


@dataclass(frozen=True)
class VerticalPath:
    """One runway end's published vertical path and decision altitude (D61): the FAA CIFP's vertical path and the
    plate's minima as the harvest reads them (`trajectory_data_process.harvest.airports.load_airport`). The decision
    altitude is the harvest's `Runway.decision_height_above_threshold_m`: the LPV line's, or the LNAV/VNAV line's where
    the runway publishes no LPV (KRDU 32, KSMF 35R — Claude's reading of "the LPV DA" for those two candidates)."""

    crossing_height_m: float        # the TCH, m above the threshold
    glidepath_deg: float            # the glidepath angle, deg
    decision_height_m: float        # the DA, m above the threshold


@dataclass(frozen=True)
class RunwayCandidate:
    """One landing threshold in the airport frame (metres, compass degrees true) and its published vertical path."""

    ident: str
    threshold_e_m: float
    threshold_n_m: float
    course_deg: float
    elevation_m: float
    length_m: float
    vertical_path: VerticalPath

    def to_dict(self) -> dict[str, Any]:
        return {"ident": self.ident, "threshold_e_m": self.threshold_e_m, "threshold_n_m": self.threshold_n_m,
                "course_deg": self.course_deg, "elevation_m": self.elevation_m, "length_m": self.length_m,
                "vertical_path": asdict(self.vertical_path)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunwayCandidate:
        return cls(**{**data, "vertical_path": VerticalPath(**data["vertical_path"])})


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

    @property
    def elevation_m(self) -> float:
        """E, the airport's published field elevation (MSL m; ``reference.elevation_m`` of ``candidates.json``): the
        altitude words are heights above it (D58)."""
        return float(self.frame.alt0)

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
                   candidates=tuple(RunwayCandidate.from_dict(item) for item in data["candidates"]),
                   runway_ends=tuple(RunwayEnd(**item) for item in data["runway_ends"]))


def vertical_path(code: str, runway: Any) -> VerticalPath:
    """A candidate's published vertical path from the harvest's `Runway` ``runway``; a runway without a TCH, a glidepath
    or vertically guided minima is refused (vocabulary §4.2): the judge cannot check its decision altitude."""
    if runway.threshold_crossing_height_m is None or runway.published_glidepath_deg is None:
        raise ValueError(f"{code} {runway.ident} publishes no threshold crossing height or glidepath")
    if not runway.published_minima.vertically_guided:
        raise ValueError(f"{code} {runway.ident} publishes no decision altitude ({runway.published_minima.note})")
    return VerticalPath(crossing_height_m=float(runway.threshold_crossing_height_m),
                        glidepath_deg=float(runway.published_glidepath_deg),
                        decision_height_m=float(runway.decision_height_above_threshold_m))


def publishes_vertical_path(runway: Any) -> bool:
    """Whether a harvest `Runway` publishes a vertical path a candidate needs (D78, D61): a threshold crossing height, a
    glidepath and vertically guided minima (`vertical_path`)."""
    return (runway.threshold_crossing_height_m is not None and runway.published_glidepath_deg is not None
            and bool(runway.published_minima.vertically_guided))


def airport_geometry(code: str, harvest_runways: Sequence[Any]) -> AirportGeometry:
    """The airport frame; its candidates — every runway end of ``harvest_runways`` (the harvest's `Runway` objects) that
    publishes a vertical path (`publishes_vertical_path`, D78), with it (`vertical_path`, D61), sorted by ident; and every
    runway end of ``harvest_runways`` (sorted by ident)."""
    code = code.upper()
    point = airport_reference_point(code)
    frame = AirportENUFrame.for_airport(AirportReference(code=code, lat=point["lat"], lon=point["lon"],
                                                         elevation_msl_m=point["elevation_m"]))
    lengths = {str(threshold["ident"]).upper(): ft_to_m(float(runway["length_ft"]))
               for runway in airport_runways(code) for threshold in runway["thresholds"]}
    candidates = []
    for runway in harvest_runways:
        if not publishes_vertical_path(runway):
            continue
        ident = str(runway.ident).upper()
        if ident not in lengths:
            raise KeyError(f"{code} runway {ident} is published but not in the runway configuration")
        e, n = frame.horizontal_from_latlon(float(runway.lat), float(runway.lon))
        candidates.append(RunwayCandidate(
            ident=ident, threshold_e_m=float(e), threshold_n_m=float(n), course_deg=float(runway.course_deg) % 360.0,
            elevation_m=float(runway.elevation_msl_m), length_m=lengths[ident], vertical_path=vertical_path(code, runway)))
    candidates.sort(key=lambda item: item.ident)
    ends = []
    for runway in harvest_runways:
        e, n = frame.horizontal_from_latlon(float(runway.lat), float(runway.lon))
        ends.append(RunwayEnd(ident=str(runway.ident).upper(), threshold_e_m=float(e), threshold_n_m=float(n),
                              course_deg=float(runway.course_deg) % 360.0))
    ends.sort(key=lambda item: item.ident)
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


def curvature_radius_m(lat_deg: float, course_deg: float) -> float:
    """The earth's radius of curvature along a compass course at a latitude: Euler's formula on the WGS84 meridional
    and prime-vertical radii (R49's, `archive/two_tier_v3_2026_10/experiments/instruction_final_approach.py`)."""
    meridional, prime_vertical = wgs84_curvature_radii(lat_deg)
    course = math.radians(course_deg)
    return 1.0 / (math.cos(course) ** 2 / meridional + math.sin(course) ** 2 / prime_vertical)


def published_glidepath_height_m(geometry: AirportGeometry, index: int, before_threshold_m):
    """The height above its threshold of candidate ``index``'s published glidepath at each distance before the
    threshold (vocabulary §6 item 4, D61): its vertical path's TCH and angle on the straight-line reference with the
    earth's radius of curvature along its course at the airport (`glidepath_height_m`). The judge's decision-altitude
    check reads it, and so will the prior's input."""
    candidate = geometry.candidates[index]
    path = candidate.vertical_path
    return glidepath_height_m(before_threshold_m, path.crossing_height_m, path.glidepath_deg,
                              curvature_radius_m(geometry.frame.lat0, candidate.course_deg))


def glidepath_height_m(before_threshold_m, crossing_height_m: float, glidepath_deg: float, radius_m: float):
    """The published glidepath's height above the threshold at each distance before it, as a straight line in space
    (vocabulary §9.3, the "straight line" reference): ``TCH + d · tan(angle) + d² / (2 R)``, R the earth's radius of
    curvature along the course (`curvature_radius_m`) — the flat formula without the last term is up to 31 m low at
    20 km. `published_glidepath_height_m` reads it with a candidate's vertical path."""
    d = np.asarray(before_threshold_m, dtype=np.float64)
    return crossing_height_m + d * math.tan(math.radians(glidepath_deg)) + d ** 2 / (2.0 * radius_m)
