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

A runway with no Path Point record has no LPV procedure (KRDU 14/32, for instance). Its
TCH is None -- NOT defaulted -- because an approach with no published LPV cannot be
judged against LPV gates at all.
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
