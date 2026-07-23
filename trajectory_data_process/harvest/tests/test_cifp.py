from pathlib import Path

import pytest

from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.cifp import read_path_points


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
    ("code", "non_lpv_idents", "runway_count"),
    [
        ("KRDU", {"14", "32"}, 6),
        ("KSMF", {"35R"}, 4),
    ],
)
def test_airport_keeps_non_lpv_runways_for_assignment_only(
    code, non_lpv_idents, runway_count
):
    airport = load_airport(code, config_file=CONFIG, cifp_file=CIFP)

    assert len(airport.runways) == runway_count
    assert {
        runway.ident
        for runway in airport.runways
        if runway.threshold_crossing_height_m is None
    } == non_lpv_idents
    for ident in non_lpv_idents:
        runway = airport.runway(ident)
        assert runway.position_source == "runway_geometry"
        assert runway.vertical_source == "nearest_faa_cifp_path_point_offset"
        assert runway.threshold_crossing_height_m is None
        assert runway.published_glidepath_deg is None
