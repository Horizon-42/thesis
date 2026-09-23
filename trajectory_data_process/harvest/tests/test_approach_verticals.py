"""A runway without LPV can still publish its vertical path: its RNAV (GPS) approach's
runway leg. These pin the decode against the Path Points, the two fleet runways it
recovers (KRDU 32, KSMF 35R), the one it cannot (KRDU 14 has no procedure), and every
runway frame's CIFP threshold and centreline course."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.context import NO_VERTICAL_GUIDANCE_SOURCE, assessment_for_runway
from trajectory_data_process.harvest.airports import (
    APPROACH_LEG_TCH_SOURCE,
    PATH_POINT_TCH_SOURCE,
    RUNWAY_RECORD_COURSE_SOURCE,
    load_airport,
)
from trajectory_data_process.harvest.cifp import (
    FT_M,
    ApproachVertical,
    PathPoint,
    _verify_approach_decode,
    read_approach_verticals,
    read_path_points,
    read_runway_records,
)

CIFP = Path("data/CIFP/CIFP_260806/FAACIFP18")
CONFIG = Path("trajectory_data_process/config/runway_thresholds.json")
FLEET = ("KRDU", "KSJC", "KSTL", "KSMF", "KMSY")

def test_rnav_gps_runway_legs_publish_the_lnav_vnav_path():
    verticals = read_approach_verticals(
        CIFP, airport="KRDU", path_points=read_path_points(CIFP, airport="KRDU")
    )
    rw32 = verticals[("KRDU", "32")]
    assert rw32.procedure == "R32"
    assert rw32.glidepath_deg == 3.5
    assert rw32.crossing_altitude_msl_m == pytest.approx(470.0 * FT_M)
    assert rw32.baro_vnav_minima and not rw32.lpv_minima
    assert ("KRDU", "14") not in verticals            # no RNAV procedure at all
    assert verticals[("KRDU", "05L")].lpv_minima      # the LPV runways carry both lines


def test_load_airport_fills_the_vertical_path_from_the_approach_leg():
    krdu = load_airport("KRDU", config_file=CONFIG, cifp_file=CIFP)
    rw32 = krdu.runway("32")
    assert rw32.lpv_course_width_m is None
    assert rw32.tch_source == APPROACH_LEG_TCH_SOURCE and rw32.baro_vnav_minima
    assert rw32.published_glidepath_deg == 3.5
    # The leg crosses at 470 ft over a 425 ft threshold: the 45 ft the runway record
    # publishes too, but taken from the procedure that flies it.
    assert rw32.threshold_crossing_height_m == pytest.approx(45.0 * FT_M, abs=0.1)
    rw14 = krdu.runway("14")
    assert rw14.threshold_crossing_height_m is None
    assert rw14.tch_source is None and not rw14.baro_vnav_minima
    assert krdu.runway("05L").tch_source == PATH_POINT_TCH_SOURCE

    rw35r = load_airport("KSMF", config_file=CONFIG, cifp_file=CIFP).runway("35R")
    assert rw35r.published_glidepath_deg == 3.0
    assert rw35r.threshold_crossing_height_m == pytest.approx(64.0 * FT_M, abs=0.1)
    assert rw35r.tch_source == APPROACH_LEG_TCH_SOURCE


def test_lnav_vnav_runway_resolves_a_real_vertical_gate():
    krdu = load_airport("KRDU", config_file=CONFIG, cifp_file=CIFP)
    context = assessment_for_runway(krdu.runway("32"))
    assert context.benchmark == "rnp_apch_lnav_vnav_baro"
    assert context.baro_vnav_approved
    assert context.procedure_source == APPROACH_LEG_TCH_SOURCE
    limits = context.limits()
    assert (limits.vertical_lower_m, limits.vertical_upper_m) == (-22.0, 22.0)
    assert limits.vertical_reason is None
    assert context.desired_threshold_altitude_msl_m == pytest.approx(470.0 * FT_M, abs=0.1)

    no_procedure = assessment_for_runway(krdu.runway("14"))
    assert not no_procedure.baro_vnav_approved
    assert no_procedure.procedure_source == NO_VERTICAL_GUIDANCE_SOURCE
    assert no_procedure.limits().vertical_lower_m is None


def test_every_runway_frame_is_the_cifp_ltp_on_the_cifp_centreline():
    """Threshold: the Path Point LTP on an LPV runway, the Runway record's otherwise.
    Course: the centreline through both ends' Runway records -- within 0.5 deg of the
    whole-degree OurAirports heading, and to the hundredth where that heading was off
    (KMSY 11 105.55, not 106; KSMF 35R 0.75, not 1)."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["airports"]
    seen = 0
    for code in FLEET:
        points = read_path_points(CIFP, airport=code)
        records = read_runway_records(CIFP, airport=code, path_points=points)
        for runway in load_airport(code, config_file=CONFIG, cifp_file=CIFP).runways:
            seen += 1
            point = points.get((code, runway.ident))
            if point is not None:
                assert (runway.lat, runway.lon) == (point.latitude, point.longitude)
                assert runway.elevation_msl_m == point.ltp_orthometric_height_m
            else:
                record = records[(code, runway.ident)]
                assert (runway.lat, runway.lon) == (record.latitude, record.longitude)
                assert runway.elevation_msl_m == record.threshold_elevation_msl_m
            assert runway.course_source == RUNWAY_RECORD_COURSE_SOURCE
            [heading] = [
                end["heading_deg"]
                for row in config[code]["runways"] for end in row["thresholds"]
                if end["ident"] == runway.ident
            ]
            assert abs((runway.course_deg - heading + 180.0) % 360.0 - 180.0) < 0.5
    assert seen == 26
    assert load_airport("KMSY", config_file=CONFIG, cifp_file=CIFP).runway("11").course_deg == (
        pytest.approx(105.546, abs=0.01)
    )
    assert load_airport("KSMF", config_file=CONFIG, cifp_file=CIFP).runway("35R").course_deg == (
        pytest.approx(0.752, abs=0.01)
    )


def _path_point(runway: str = "09", *, tch_m: float = 16.0) -> PathPoint:
    return PathPoint(
        airport="KAAA", runway=runway, latitude=0.0, longitude=0.0, glidepath_deg=3.0,
        threshold_crossing_height_m=tch_m, course_width_m=106.75,
        ltp_ellipsoidal_height_m=100.0, ltp_orthometric_height_m=130.0,
    )


def _vertical(runway: str, altitude_m: float, *, angle: float = 3.0, lpv: bool = True,
              conflict: str | None = None) -> ApproachVertical:
    return ApproachVertical("KAAA", runway, f"R{runway}", angle, altitude_m, lpv, True,
                            conflict=conflict)


def test_the_pin_is_a_confidence_rule_over_the_lpv_runways():
    """Three of four agreeing passes (a lone data disagreement does not condemn the
    airport); two of four, a wrong angle everywhere, or an unmatched LPV procedure
    raises -- the shapes a shifted column or a mis-keyed decode produce."""
    points = {("KAAA", rw): _path_point(rw) for rw in ("09", "27", "18", "36")}
    good = {("KAAA", rw): _vertical(rw, 146.0) for rw in ("09", "27", "18")}
    _verify_approach_decode(Path("x"), {**good, ("KAAA", "36"): _vertical("36", 150.0)}, points)
    with pytest.raises(ValueError, match="agrees with the Path Points on only 2/4"):
        _verify_approach_decode(
            Path("x"),
            {**good, ("KAAA", "18"): _vertical("18", 150.0), ("KAAA", "36"): _vertical("36", 150.0)},
            points,
        )
    with pytest.raises(ValueError, match="agrees with the Path Points on only 0/4"):
        _verify_approach_decode(
            Path("x"), {key: _vertical(key[1], 146.0, angle=3.1) for key in points}, points
        )
    with pytest.raises(ValueError, match="unverified"):
        _verify_approach_decode(Path("x"), {("KAAA", "05"): _vertical("05", 146.0)}, points)
    # A conflict is not a comparison; an airport with no LPV-bearing procedure has
    # nothing to pin against and is not an error.
    _verify_approach_decode(
        Path("x"),
        {**good, ("KAAA", "36"): _vertical("36", 999.0, conflict="Y and Z disagree")},
        points,
    )
    _verify_approach_decode(
        Path("x"), {("KAAA", "09"): _vertical("09", 146.0, lpv=False)}, points
    )


def _record(
    procedure: str,
    *,
    fix: str = "RW09 ",
    fix_section: str = "PG",
    transition: str = "     ",
    continuation: str = "0",
    altitude_ft: int | None = 146,
    altitude_description: str = " ",
    angle: str = "-300",
    application: str = " ",
    minima: str = "",
) -> str:
    """One 132-column procedure record: section P / subsection F for airport KAAA."""
    row = [" "] * 132
    row[4] = "P"
    row[6:10] = "KAAA"
    row[12] = "F"
    row[13:19] = f"{procedure:<6}"
    row[20:25] = transition
    row[29:34] = fix
    row[36:38] = fix_section
    row[38] = continuation
    row[39] = application
    if minima:
        row[40:40 + len(minima)] = minima
    row[82] = altitude_description
    if altitude_ft is not None:
        row[84:89] = f"{altitude_ft:05d}"
    row[102:106] = angle
    return "".join(row)


def _leg(procedure: str, altitude_ft: int, *, angle: str = "-300") -> str:
    return _record(procedure, altitude_ft=altitude_ft, angle=angle)


def _write(tmp_path: Path, *lines: str) -> Path:
    cifp = tmp_path / "FAACIFP18"
    cifp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cifp


def test_two_rnav_procedures_to_one_runway_that_disagree_are_a_conflict_not_a_choice(tmp_path):
    cifp = _write(tmp_path, _leg("R09Y", 146), _leg("R09Z", 150))
    [vertical] = read_approach_verticals(cifp, airport="KAAA").values()
    assert vertical.conflict and "publish different vertical paths" in vertical.conflict
    assert vertical.glidepath_deg is None and vertical.crossing_altitude_msl_m is None

    cifp = _write(tmp_path, _leg("R09Y", 146), _leg("R09Z", 146))
    [vertical] = read_approach_verticals(cifp, airport="KAAA").values()
    assert vertical.conflict is None
    assert vertical.procedure == "R09Y/R09Z"
    assert vertical.glidepath_deg == 3.0
    assert vertical.crossing_altitude_msl_m == pytest.approx(146 * FT_M)
    assert not vertical.baro_vnav_minima     # no approach-types continuation was coded


def test_only_the_final_segment_runway_leg_of_a_runway_procedure_is_read(tmp_path):
    """Transition legs, non-runway fixes, circling RNAV (GPS)-A and RNP AR procedures
    are not the crossing; a PRIMARY record whose waypoint-description letter sits in
    the continuation's application column is the leg, not an approach-types record;
    the approach-types continuation sets the minima flags."""
    cifp = _write(
        tmp_path,
        _record("R09", transition="ABCDE", altitude_ft=900),       # transition leg to RW09
        _record("R09", fix="FAFIX", fix_section="PC", altitude_ft=1400),
        _record("R09", application="G", altitude_ft=146),           # the runway leg
        _record("R09", continuation="2", application="W",
                minima="ALPV       ALNAV/VNAV ALNAV      "),
        _record("RNV-A", altitude_ft=900),                          # circling
        _record("H09", altitude_ft=140),                            # RNP AR
    )
    verticals = read_approach_verticals(cifp, airport="KAAA")
    assert list(verticals) == [("KAAA", "09")]
    vertical = verticals[("KAAA", "09")]
    assert vertical.procedure == "R09"
    assert vertical.crossing_altitude_msl_m == pytest.approx(146 * FT_M)
    assert vertical.lpv_minima and vertical.baro_vnav_minima


def test_a_procedure_that_codes_the_runway_fix_twice_or_a_bounded_altitude_raises(tmp_path):
    with pytest.raises(ValueError, match="codes the runway fix twice"):
        read_approach_verticals(
            _write(tmp_path, _leg("R09", 146), _leg("R09", 140)), airport="KAAA"
        )
    with pytest.raises(ValueError, match="bounded altitude"):
        read_approach_verticals(
            _write(tmp_path, _record("R09", altitude_description="+")), airport="KAAA"
        )
