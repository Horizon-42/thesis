"""Published approach geometry, read from the ARINC 424 CIFP Path Point record.

The FAA CIFP ships in ``data/CIFP/<cycle>/FAACIFP18``. Its section P / subsection P
"Path Point" records carry, per LPV approach, the numbers that define where the
aircraft is SUPPOSED to cross: the landing threshold point, the published glidepath
angle, the lateral course width at the threshold, and the threshold crossing height.

WHY THIS MATTERS: the project previously assumed a flat TCH of 15 m for every runway.
Every runway in the harvest fleet publishes MORE than that (15.27-18.11 m), so the
assumption put a systematic 1.5-2.5 m bias straight into the vertical gate -- on a
window only 9.15 m wide. Reading the published value moved the measured crossing at
KSMF from +2.74 m to **+0.61 m**: with the correct TCH, real airline traffic crosses
where the plate says it should, to within half a metre. That agreement is also the best
end-to-end evidence the datum handling and the segment fit are both right.

COLUMN DECODE, AND WHY IT IS TRUSTWORTHY
----------------------------------------
ARINC 424 is fixed-column, and a mis-set offset silently yields plausible numbers. This
decode is pinned by a fact that cannot coincide: **4795 of 4900 records decode a course
width of exactly 106.75 m**, the FAA Formula 3-1-1 minimum course-width value
(350 ft). A wrong offset would
have to land on that constant AND on 3.00 deg for 4227 records at the same time.
``read_path_points`` re-checks both on every load rather than trusting this comment.

A runway with no Path Point record has no LPV procedure (KRDU 14/32, KSMF 35R). Its
vertical path may still be published by its RNAV (GPS) approach's runway leg when that
approach carries LNAV/VNAV minima (``read_approach_verticals``, ``ApproachVertical``);
only a runway with no such approach at all (KRDU 14) is left with TCH None -- NOT
defaulted -- because a crossing with no published path cannot be judged vertically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FT_M = 0.3048

# Fixed-column offsets of the Path Point primary record (continuation "001").
_SECTION = 4
_SUBSECTION = 12
_CONTINUATION = slice(24, 27)
_AIRPORT = slice(6, 10)
_RUNWAY = slice(19, 24)
_LTP_LATITUDE = slice(37, 48)             # N/S + DDMMSSssss
_LTP_LONGITUDE = slice(48, 60)            # E/W + DDDMMSSssss
_LTP_ELLIPSOIDAL_HEIGHT = slice(60, 66)   # 0.1 m
_GLIDEPATH_ANGLE = slice(66, 70)          # 0.01 deg
_COURSE_WIDTH = slice(93, 98)             # 0.01 m
_TCH = slice(102, 108)                    # 0.1 unit
_TCH_UNITS = 108                          # 'F' feet | 'M' metres
_LTP_ORTHOMETRIC_HEIGHT = slice(40, 46)   # continuation "002", 0.1 m

_MIN_RECORD_LENGTH = 110

# The two independent constants that pin the decode (see the module docstring).
_EXPECTED_COURSE_WIDTH_M = 106.75
_EXPECTED_GLIDEPATH_DEG = 3.0
_DECODE_CONFIDENCE = 0.75

# Fixed-column offsets of the approach-procedure records (section P / subsection F),
# 0-based slices of the 132-column ARINC 424 line. cifparse's
# ``records/procedure/widths.py`` names the same columns, which is how they were
# confirmed; the decode is pinned at load time against the Path Points (see
# ``read_approach_verticals``).
_PROCEDURE_ID = slice(13, 19)             # "R32   ", "R05LY " (first letter = type)
_TRANSITION_ID = slice(20, 25)            # blank on the final approach segment
_FIX_ID = slice(29, 34)                   # "RW32 " on the runway leg
_FIX_SECTION = slice(36, 38)              # "PG" = the fix IS the runway
_CONTINUATION_NUMBER = 38                 # "0"/"1" primary, "2".. continuation
_APPLICATION_TYPE = 39                    # continuation records: "W" = approach types
_ALTITUDE_DESCRIPTION = 82                # blank = AT; "+", "-", "B" = a bound, not a crossing
_ALTITUDE_1 = slice(84, 89)               # feet MSL, integer, coded AT the runway fix
_VERTICAL_ANGLE = slice(102, 106)         # 0.01 deg, signed: "-300" = 3.00 deg descent
# The approach-types continuation: <authorized letter, service name> pairs; the third
# slot (62:73) is LNAV, which decides nothing here.
_MINIMA_LPV = slice(40, 51)
_MINIMA_LNAV_VNAV = slice(51, 62)
_MINIMA_AUTHORIZED = "A"
# A procedure's runway-fix altitude and the Path Point's LTP + TCH describe the same
# published point. Altitudes are coded in whole feet and the LTP height in 0.1 m, so a
# pair that agrees sits within a foot (file-wide p99 over 3,455 LPV pairs: 0.8 ft); a
# shifted column is off by hundreds.
_APPROACH_LEG_ALTITUDE_TOLERANCE_FT = 2.0


def _is_rnav_gps_runway_procedure(procedure: str) -> bool:
    """``R`` + a runway number: "R32", "R05LY", "R11-Y". NOT "RNV-A"/"RNVB" (circling
    RNAV (GPS)-A/-B, not runway-aligned), nor "H" (RNP AR), "I" (ILS), "L" (LOC)."""
    return procedure.startswith("R") and procedure[1:2].isdigit()


@dataclass(frozen=True)
class PathPoint:
    """One published LPV final approach.

    ``latitude``/``longitude`` are the **Landing Threshold Point** — where the procedure
    is actually aimed, and therefore the correct along-track origin for judging an
    approach to it. It is NOT always where the OurAirports runway geometry puts the
    threshold: cross-checked over this fleet, the two disagree by up to 61 m, and the
    disagreement is measurable in the flown data. KSMF 35L's LTP is 40.7 m CROSS-track
    from the OurAirports point, and the observed lateral median against the latter was
    41.1 m -- i.e. the entire apparent "lateral error" of that runway's traffic was the
    reference being in the wrong place. KSTL 30L disagrees by 61.4 m almost entirely
    ALONG-track, which on a 3 deg path is ~3.2 m of crossing height.
    """

    airport: str
    runway: str
    latitude: float
    longitude: float
    glidepath_deg: float
    threshold_crossing_height_m: float
    course_width_m: float
    ltp_ellipsoidal_height_m: float
    ltp_orthometric_height_m: float


def read_path_points(
    cifp_file: Path, *, airport: str | None = None
) -> dict[tuple[str, str], PathPoint]:
    """Decode every Path Point record, keyed by ``(airport, runway)`` e.g. ("KRDU","05L").

    Raises when the decode does not reproduce the two pinning constants, because a
    silently shifted column would feed a plausible-but-wrong TCH into the vertical gate
    -- the exact failure mode this module exists to remove.
    """
    variants: dict[tuple[str, str], list[PathPoint]] = {}
    course_widths: list[float] = []
    glidepaths: list[float] = []

    lines = cifp_file.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if len(line) < _MIN_RECORD_LENGTH:
            continue
        if line[_SECTION] != "P" or line[_SUBSECTION] != "P" or line[_CONTINUATION] != "001":
            continue
        if airport is not None and line[_AIRPORT].strip() != airport:
            continue
        continuation = lines[index + 1] if index + 1 < len(lines) else ""
        if (
            len(continuation) < _MIN_RECORD_LENGTH
            or continuation[_SECTION] != "P"
            or continuation[_SUBSECTION] != "P"
            or continuation[_CONTINUATION] != "002"
            or continuation[_AIRPORT] != line[_AIRPORT]
            or continuation[_RUNWAY] != line[_RUNWAY]
        ):
            raise ValueError(
                f"{cifp_file}:{index + 1}: Path Point {line[_AIRPORT].strip()} "
                f"{line[_RUNWAY].strip()} has no matching continuation 002"
            )
        try:
            glidepath = int(line[_GLIDEPATH_ANGLE]) / 100.0
            course_width = int(line[_COURSE_WIDTH]) / 100.0
            tch = int(line[_TCH]) / 10.0
            ltp_height = int(line[_LTP_ELLIPSOIDAL_HEIGHT]) / 10.0
            latitude = _decode_angle(line[_LTP_LATITUDE])
            longitude = _decode_angle(line[_LTP_LONGITUDE])
            ltp_orthometric_height = int(
                continuation[_LTP_ORTHOMETRIC_HEIGHT]
            ) / 10.0
        except (ValueError, IndexError):
            continue  # a continuation/notes record that passed the shape test

        units = line[_TCH_UNITS]
        if units not in ("F", "M"):
            continue
        # Non-LPV path-point families can share the runway key but publish zero vertical
        # guidance. They are not variants of an LPV final approach.
        if glidepath <= 0.0 or tch <= 0.0:
            continue
        key = (line[_AIRPORT].strip(), line[_RUNWAY].strip().removeprefix("RW"))
        course_widths.append(course_width)
        glidepaths.append(glidepath)
        variants.setdefault(key, []).append(
            PathPoint(
                airport=key[0],
                runway=key[1],
                latitude=latitude,
                longitude=longitude,
                glidepath_deg=glidepath,
                threshold_crossing_height_m=tch * (FT_M if units == "F" else 1.0),
                course_width_m=course_width,
                ltp_ellipsoidal_height_m=ltp_height,
                ltp_orthometric_height_m=ltp_orthometric_height,
            ),
        )

    _verify_decode(cifp_file, course_widths, glidepaths)
    points: dict[tuple[str, str], PathPoint] = {}
    for key, records in variants.items():
        first = records[0]
        for record in records[1:]:
            if record != first:
                raise ValueError(
                    f"{cifp_file}: conflicting Path Point variants for "
                    f"{key[0]} runway {key[1]}: {first!r} != {record!r}"
                )
        points[key] = first
    return points


@dataclass(frozen=True)
class ApproachVertical:
    """The published vertical path of a runway's RNAV (GPS) approach, at the runway.

    An LPV runway has this AND a Path Point; a runway whose RNAV approach publishes only
    LNAV/VNAV (Baro-VNAV) and LNAV minima has ONLY this. The final leg of the procedure
    (the leg whose fix is the runway itself) codes the vertical angle and the altitude
    at which the path crosses the runway, so ``crossing_altitude_msl_m`` minus the
    threshold elevation is that procedure's threshold crossing height -- the same
    quantity the Path Point publishes for LPV, from the same file. KRDU RNAV (GPS)
    RWY 32: 3.50 deg, 470 ft at a 425 ft threshold = 45 ft; KSMF RNAV (GPS) Y RWY 35R:
    3.00 deg, 86 ft at 22 ft = 64 ft. Measured over the fleet's 23 LPV runways the leg
    altitude agrees with LTP + Path Point TCH to within a foot, which is what pins
    the column decode (``read_approach_verticals``).

    ``glidepath_deg`` is positive for a descent, like ``PathPoint.glidepath_deg``.
    Either vertical field is None when the leg does not code it (LNAV-only procedures
    may publish no angle).
    """

    airport: str
    runway: str
    procedure: str
    glidepath_deg: float | None
    crossing_altitude_msl_m: float | None
    lpv_minima: bool
    baro_vnav_minima: bool
    # Two RNAV (GPS) procedures to this runway (a Y and a Z) publish DIFFERENT paths:
    # the vertical fields are None and this names the disagreement. Raised only where
    # it matters -- a runway that needs the leg for its vertical path (``airports``);
    # an LPV runway takes its Path Point and never reads it.
    conflict: str | None = None


def read_approach_verticals(
    cifp_file: Path,
    *,
    airport: str,
    path_points: dict[tuple[str, str], PathPoint] | None = None,
) -> dict[tuple[str, str], ApproachVertical]:
    """Decode one airport's RNAV (GPS) approach runway-leg verticals, keyed like Path Points.

    Only runway-aligned RNAV (GPS) procedures are read (``_is_rnav_gps_runway_procedure``):
    RNP AR needs special authorization and may publish a different path, ILS/LOC carry
    their vertical in the localizer record, and the circling RNAV (GPS)-A/-B family is
    not aimed at a runway. A procedure's final segment (blank transition) must code the
    runway fix once, with an exact altitude; two RNAV (GPS) procedures to one runway
    that disagree are returned as a ``conflict`` (see ``ApproachVertical``).

    ``path_points`` (the same airport's decoded Path Points) pins the decode: on every
    runway that has both, the LPV-bearing procedure's angle must equal the Path Point
    glidepath and its runway altitude must equal LTP + TCH within
    ``_APPROACH_LEG_ALTITUDE_TOLERANCE_FT``, on at least ``_DECODE_CONFIDENCE`` of the
    comparable runways -- the same rule the Path Point decode is pinned by. A shifted
    column fails every pair; one runway whose two records genuinely disagree does not
    condemn the airport. An airport with no LPV-bearing procedure has nothing to pin
    against; its LNAV/VNAV legs are then decoded by the column offsets alone.
    """
    lines = cifp_file.read_text(errors="replace").splitlines()
    by_procedure: dict[str, list[str]] = {}
    for line in lines:
        if len(line) < _MIN_RECORD_LENGTH:
            continue
        if line[_SECTION] != "P" or line[_SUBSECTION] != "F":
            continue
        if line[_AIRPORT].strip() != airport:
            continue
        procedure = line[_PROCEDURE_ID].strip()
        if not _is_rnav_gps_runway_procedure(procedure):
            continue
        by_procedure.setdefault(procedure, []).append(line)

    variants: dict[tuple[str, str], list[ApproachVertical]] = {}
    for procedure, records in by_procedure.items():
        runway_leg = None
        lpv = baro_vnav = False
        for line in records:
            if line[_TRANSITION_ID].strip():
                continue  # an approach transition, not the final segment
            if line[_CONTINUATION_NUMBER] in "01":
                if line[_FIX_SECTION] != "PG" or not line[_FIX_ID].startswith("RW"):
                    continue
                if runway_leg is not None:
                    raise ValueError(
                        f"{cifp_file}: procedure {airport} {procedure} codes the runway "
                        "fix twice on its final segment; which leg is the crossing is "
                        "not a choice this reader may make"
                    )
                if line[_ALTITUDE_DESCRIPTION] != " ":
                    raise ValueError(
                        f"{cifp_file}: procedure {airport} {procedure} codes a bounded "
                        f"altitude ({line[_ALTITUDE_DESCRIPTION]!r}) at the runway fix; "
                        "only an exact crossing altitude is a vertical path"
                    )
                runway_leg = line
            elif line[_APPLICATION_TYPE] == "W":
                lpv = lpv or _minima_authorized(line[_MINIMA_LPV], "LPV")
                baro_vnav = baro_vnav or _minima_authorized(
                    line[_MINIMA_LNAV_VNAV], "LNAV/VNAV"
                )
        if runway_leg is None:
            continue
        runway = runway_leg[_FIX_ID].strip().removeprefix("RW")
        variants.setdefault((airport, runway), []).append(
            ApproachVertical(
                airport=airport,
                runway=runway,
                procedure=procedure,
                glidepath_deg=_descent_angle(runway_leg[_VERTICAL_ANGLE]),
                crossing_altitude_msl_m=_altitude_m(runway_leg[_ALTITUDE_1]),
                lpv_minima=lpv,
                baro_vnav_minima=baro_vnav,
            )
        )

    verticals: dict[tuple[str, str], ApproachVertical] = {}
    for key, records in variants.items():
        first = records[0]
        procedures = "/".join(record.procedure for record in records)
        disagreeing = any(
            (record.glidepath_deg, record.crossing_altitude_msl_m)
            != (first.glidepath_deg, first.crossing_altitude_msl_m)
            for record in records[1:]
        )
        verticals[key] = ApproachVertical(
            airport=airport,
            runway=key[1],
            procedure=procedures,
            glidepath_deg=None if disagreeing else first.glidepath_deg,
            crossing_altitude_msl_m=None if disagreeing else first.crossing_altitude_msl_m,
            lpv_minima=any(record.lpv_minima for record in records),
            baro_vnav_minima=any(record.baro_vnav_minima for record in records),
            conflict=(
                f"RNAV (GPS) procedures {procedures} to {airport} runway {key[1]} "
                "publish different vertical paths"
                if disagreeing else None
            ),
        )
    if path_points is not None:
        _verify_approach_decode(cifp_file, verticals, path_points)
    return verticals


def _minima_authorized(field: str, service: str) -> bool:
    return field[0] == _MINIMA_AUTHORIZED and field[1:].strip() == service


def _descent_angle(text: str) -> float | None:
    """``"-300"`` -> 3.0 (degrees of descent); blank or a climb -> None."""
    try:
        angle = int(text) / 100.0
    except ValueError:
        return None
    return -angle if angle < 0.0 else None


def _altitude_m(text: str) -> float | None:
    """A coded altitude in whole feet MSL -> metres; blank or a flight level -> None."""
    try:
        return int(text) * FT_M
    except ValueError:
        return None


def _verify_approach_decode(
    cifp_file: Path,
    verticals: dict[tuple[str, str], ApproachVertical],
    path_points: dict[tuple[str, str], PathPoint],
) -> None:
    """The LPV-bearing procedures' runway legs must describe the Path Points' crossings.

    Judged like the Path Point decode: at least ``_DECODE_CONFIDENCE`` of the comparable
    runways agree (angle equal, altitude within tolerance). A shifted column fails all
    of them; a single runway whose two records genuinely differ (24 of 3,455 pairs
    file-wide) is a data disagreement, not a decode error, and that runway takes its
    Path Point anyway. Raises on vacuity only when there IS an LPV-bearing procedure
    that matched no Path Point -- the keys disagree, which is a decode error too.
    """
    comparable = agreeing = 0
    for key, point in path_points.items():
        vertical = verticals.get(key)
        if vertical is None or not vertical.lpv_minima or vertical.conflict:
            continue
        comparable += 1
        published_m = point.ltp_orthometric_height_m + point.threshold_crossing_height_m
        if (
            vertical.glidepath_deg == point.glidepath_deg
            and vertical.crossing_altitude_msl_m is not None
            and abs(vertical.crossing_altitude_msl_m - published_m)
            <= _APPROACH_LEG_ALTITUDE_TOLERANCE_FT * FT_M
        ):
            agreeing += 1
    if comparable == 0:
        if any(vertical.lpv_minima for vertical in verticals.values()):
            raise ValueError(
                f"{cifp_file}: LPV-bearing RNAV (GPS) procedures matched no Path Point "
                "by runway, so the approach-leg decode is unverified"
            )
        return
    if agreeing / comparable < _DECODE_CONFIDENCE:
        raise ValueError(
            f"{cifp_file}: approach-leg decode agrees with the Path Points on only "
            f"{agreeing}/{comparable} LPV runways (need >= {_DECODE_CONFIDENCE:.0%}). "
            "Check the ARINC 424 revision."
        )


def _decode_angle(text: str) -> float:
    """ARINC 424 packed angle: hemisphere letter + DD[D]MMSSssss, to signed degrees."""
    digits = text[1:]
    fractional_digits = 4 if len(digits) in (10, 11) else 2
    tail = 4 + fractional_digits
    degrees = int(digits[:-tail])
    minutes = int(digits[-tail : -tail + 2])
    seconds = int(digits[-tail + 2 : -fractional_digits])
    seconds += int(digits[-fractional_digits:]) / (10**fractional_digits)
    value = degrees + minutes / 60.0 + seconds / 3600.0
    return -value if text[0] in "SW" else value


def _verify_decode(cifp_file: Path, course_widths: list[float], glidepaths: list[float]) -> None:
    if not course_widths:
        raise ValueError(f"{cifp_file}: no Path Point records decoded — wrong file or cycle?")
    width_hits = sum(1 for w in course_widths if w == _EXPECTED_COURSE_WIDTH_M)
    glidepath_hits = sum(1 for g in glidepaths if g == _EXPECTED_GLIDEPATH_DEG)
    n = len(course_widths)
    if width_hits / n < _DECODE_CONFIDENCE or glidepath_hits / n < _DECODE_CONFIDENCE:
        raise ValueError(
            f"{cifp_file}: Path Point column decode looks wrong — "
            f"course width == {_EXPECTED_COURSE_WIDTH_M} m in {width_hits}/{n} records and "
            f"glidepath == {_EXPECTED_GLIDEPATH_DEG} deg in {glidepath_hits}/{n} "
            f"(expected >= {_DECODE_CONFIDENCE:.0%} of each). Check the ARINC 424 revision."
        )
