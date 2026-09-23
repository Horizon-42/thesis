from pathlib import Path

import pytest

from geokit import haversine_m

from trajectory_data_process.harvest.airports import (
    RUNWAY_RECORD_POSITION_SOURCE,
    load_airport,
)
from trajectory_data_process.harvest.cifp import read_path_points, read_runway_records


CIFP = Path("data/CIFP/CIFP_260319/FAACIFP18")
CONFIG = Path("trajectory_data_process/config/runway_thresholds.json")


def test_kmsy_path_points_publish_same_point_dual_datum():
    points = read_path_points(CIFP, airport="KMSY")
    expected = {
        "02": (-25.7, 0.4),
        "11": (-25.2, 0.9),
        "20": (-27.0, -0.9),
        "29": (-25.9, 0.2),
    }
    for runway, (hae, msl) in expected.items():
        point = points[("KMSY", runway)]
        assert point.ltp_ellipsoidal_height_m == pytest.approx(hae)
        assert point.ltp_orthometric_height_m == pytest.approx(msl)
        assert hae - msl == pytest.approx(-26.1)


def test_missing_path_point_continuation_raises(tmp_path):
    primary = list(" " * 110)
    primary[4] = "P"
    primary[6:10] = "KMSY"
    primary[12] = "P"
    primary[19:24] = "RW02 "
    primary[24:27] = "001"
    cifp = tmp_path / "FAACIFP18"
    cifp.write_text("".join(primary), encoding="utf-8")

    with pytest.raises(ValueError, match="has no matching continuation 002"):
        read_path_points(cifp, airport="KMSY")


@pytest.mark.parametrize(
    ("code", "non_lpv_idents", "no_vertical_idents", "runway_count"),
    [
        ("KRDU", {"14", "32"}, {"14"}, 6),
        ("KSMF", {"35R"}, set(), 4),
    ],
)
def test_non_lpv_runways_take_the_cifp_runway_record_threshold(
    code, non_lpv_idents, no_vertical_idents, runway_count
):
    """A non-LPV runway's threshold is its CIFP Runway record's LTP; its vertical path
    comes from the RNAV (GPS) approach leg when one publishes LNAV/VNAV minima (KRDU 32,
    KSMF 35R) and stays None when no procedure exists at all (KRDU 14)."""
    airport = load_airport(code, config_file=CONFIG, cifp_file=CIFP)

    assert len(airport.runways) == runway_count
    assert {
        runway.ident for runway in airport.runways if runway.lpv_course_width_m is None
    } == non_lpv_idents
    assert {
        runway.ident
        for runway in airport.runways
        if runway.threshold_crossing_height_m is None
    } == no_vertical_idents
    for ident in non_lpv_idents:
        runway = airport.runway(ident)
        assert runway.position_source == RUNWAY_RECORD_POSITION_SOURCE
        assert runway.vertical_source == "nearest_faa_cifp_path_point_offset"
        assert (runway.threshold_crossing_height_m is None) == (ident in no_vertical_idents)
        assert (runway.published_glidepath_deg is None) == (ident in no_vertical_idents)


def test_ksmf_35r_threshold_is_the_published_ltp_not_the_ourairports_end():
    """RW35R's Runway record, N38410065 W121344964: the point the RNAV (GPS) Y RWY 35R is
    flown to. The OurAirports end sits 39.4 m east of it, which every 35R arrival read as
    a ~41 m lateral miss (383/383 failed the lateral gate before this)."""
    rw35r = load_airport("KSMF", config_file=CONFIG, cifp_file=CIFP).runway("35R")
    assert rw35r.lat == pytest.approx(38 + 41 / 60 + 0.65 / 3600, abs=1e-9)
    assert rw35r.lon == pytest.approx(-(121 + 34 / 60 + 49.64 / 3600), abs=1e-9)
    assert rw35r.elevation_msl_m == pytest.approx(22 * 0.3048)
    assert haversine_m(rw35r.lat, rw35r.lon, 38.683498, -121.580002) == pytest.approx(39.4, abs=0.2)


def test_runway_record_decode_is_pinned_against_the_path_points():
    """Every LPV runway's Runway record agrees with its Path Point LTP (<= 1 m, <= 1 ft)."""
    points = read_path_points(CIFP, airport="KSMF")
    records = read_runway_records(CIFP, airport="KSMF", path_points=points)
    assert set(records) == {("KSMF", ident) for ident in ("17L", "17R", "35L", "35R")}
    for key, point in points.items():
        record = records[key]
        assert haversine_m(record.latitude, record.longitude, point.latitude, point.longitude) < 1.0
        assert abs(record.threshold_elevation_msl_m - point.ltp_orthometric_height_m) <= 0.3048


def test_a_misdecoded_runway_record_fails_the_pin(tmp_path):
    """A decode that yields valid-looking but wrong coordinates (here: one arc-minute of
    latitude, ~1.85 km) is caught by the Path Point comparison, not by a parse error."""
    lines = [
        line for line in CIFP.read_text(errors="replace").splitlines()
        if line[6:10] == "KSMF" and line[4] == "P" and line[12] in "GP"
    ]
    # index 36: the units digit of the latitude's arc-minutes ("N38410065" -> "N38420065")
    misread = [
        line[:36] + str((int(line[36]) + 1) % 10) + line[37:] if line[12] == "G" else line
        for line in lines
    ]
    cifp = tmp_path / "FAACIFP18"
    cifp.write_text("\n".join(misread) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Runway-record decode agrees"):
        read_runway_records(cifp, airport="KSMF", path_points=read_path_points(cifp, airport="KSMF"))
