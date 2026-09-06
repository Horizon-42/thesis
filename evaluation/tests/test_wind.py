"""The METAR headwind correction: the observed ground-speed proxy becomes an airspeed
estimate when the field's own wind report is usable, and says so either way."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aircraft.reference_speeds import reference_speed
from evaluation import evaluate_batch, evaluate_record, record_from_dict, speed_gate_bounds
from evaluation.speed_gate import (
    OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID,
    OBSERVED_SPEED_CRITERION_ID,
)
from evaluation.tests.factories import AIRCRAFT_TYPE, assessment_context, observed_track_payload
from evaluation.wind import (
    WIND_MAX_AGE_S,
    WindObservation,
    WindTable,
    load_wind_tables,
    wind_at_landing,
)
from geokit import kt_to_ms

IEM_CSV = """station,valid,drct,sknt,gust
RDU,2026-08-11 23:00,0.00,0.00,M
RDU,2026-08-12 00:10,180.00,10.00,18.00
RDU,2026-08-12 00:51,M,4.00,M
RDU,2026-08-12 01:51,90.00,M,M
RDU,2026-08-12 02:51,45.00,12.00,M
"""
# The factory's landing time is 2026-08-12T00:00:00Z on runway 05L (course 0 deg in
# the fixture context): the nearest report, 00:10, is a 10 kt wind FROM 180 = a
# tailwind; the calm 23:00 report is an hour away and not the nearest.
LANDING = "2026-08-12T00:00:00Z"
# The observed window (the type's published mass range at 1 g), through the gate's
# own function: the tests place ground speeds relative to its lower edge.
_LOWER = speed_gate_bounds(
    reference_speed(AIRCRAFT_TYPE), load_factor=1.0, crossing_mass_kg=None
).lower_ms
_TAIL_MS = kt_to_ms(10.0)


def _table(tmp_path: Path) -> WindTable:
    path = tmp_path / "KRDU" / "asos_2026-08-11_2026-08-12.csv"
    path.parent.mkdir()
    path.write_text(IEM_CSV, encoding="utf-8")
    return WindTable.from_iem_csv([path])


def test_the_iem_csv_is_read_in_the_archive_units_and_missing_speed_rows_are_dropped(tmp_path):
    table = _table(tmp_path)
    assert table.station == "RDU"
    assert [o.valid_utc.strftime("%H:%M") for o in table.observations] == [
        "23:00", "00:10", "00:51", "02:51",
    ]                                       # 01:51 had no speed and says nothing
    tail = table.observations[1]
    assert tail.direction_deg == 180.0
    assert tail.speed_ms == pytest.approx(kt_to_ms(10.0))
    assert tail.gust_ms == pytest.approx(kt_to_ms(18.0))
    assert table.observations[2].direction_deg is None    # variable


def test_nearest_report_within_the_age_limit_else_none(tmp_path):
    table = _table(tmp_path)
    when = datetime(2026, 8, 12, 0, 0, tzinfo=timezone.utc)
    assert table.nearest(when).valid_utc.strftime("%H:%M") == "00:10"
    far = datetime(2026, 8, 12, 1, 40, tzinfo=timezone.utc)   # 49 min from 00:51 / 71 from 02:51
    assert table.nearest(far) is None
    assert table.nearest(far, max_age_s=WIND_MAX_AGE_S * 2) is not None


def test_the_headwind_component_follows_the_from_direction():
    ten_kt = kt_to_ms(10.0)
    on_the_nose = WindObservation(datetime.now(timezone.utc), 360.0, ten_kt, None)
    assert on_the_nose.headwind_ms(0.0) == pytest.approx(ten_kt)
    tail = WindObservation(datetime.now(timezone.utc), 180.0, ten_kt, None)
    assert tail.headwind_ms(0.0) == pytest.approx(-ten_kt)
    cross = WindObservation(datetime.now(timezone.utc), 90.0, ten_kt, None)
    assert cross.headwind_ms(0.0) == pytest.approx(0.0, abs=1e-12)
    quartering = WindObservation(datetime.now(timezone.utc), 45.0, ten_kt, None)
    assert quartering.headwind_ms(0.0) == pytest.approx(ten_kt * math.cos(math.radians(45)))
    assert WindObservation(datetime.now(timezone.utc), None, 0.0, None).headwind_ms(0.0) == 0.0
    assert WindObservation(datetime.now(timezone.utc), None, ten_kt, None).headwind_ms(0.0) is None


def test_wind_at_landing_reports_what_it_used_or_why_not(tmp_path):
    table = _table(tmp_path)
    headwind, block = wind_at_landing(table, LANDING, 0.0)
    assert headwind == pytest.approx(-kt_to_ms(10.0))
    assert block["status"] == "estimated" and block["age_s"] == 600.0
    assert block["direction_deg_true"] == 180.0 and block["station"] == "RDU"
    _, variable = wind_at_landing(table, "2026-08-12T00:50:00Z", 0.0)
    assert variable["status"] == "unavailable" and "variable" in variable["reason"]
    _, stale = wind_at_landing(table, "2026-08-12T01:40:00Z", 0.0)
    assert stale["status"] == "unavailable" and "no report within" in stale["reason"]


def test_load_wind_tables_keys_by_airport_and_tolerates_an_absent_root(tmp_path):
    _table(tmp_path)
    assert list(load_wind_tables(tmp_path)) == ["KRDU"]
    assert load_wind_tables(tmp_path / "nowhere") == {}


def test_a_tailwind_turns_a_passing_ground_speed_into_a_failing_airspeed_estimate(tmp_path):
    """A ground speed 2 m/s above the window's floor with 10 kt (5.1 m/s) on the tail is
    3.1 m/s of air BELOW it: the proxy passes, the estimate fails, and the row's
    margin is the signed distance to the floor."""
    table = _table(tmp_path)
    context = assessment_context()
    ground = _LOWER + 2.0
    proxy = evaluate_record(
        record_from_dict(observed_track_payload(ground_speed_m_s=ground)), context=context
    )
    assert proxy.speed_result == "pass"
    assert proxy.speed_criterion == OBSERVED_SPEED_CRITERION_ID
    assert proxy.wind["status"] == "unavailable" and proxy.crossing_airspeed_estimate_ms is None
    assert proxy.speed_margin_ms == pytest.approx(2.0)

    corrected = evaluate_record(
        record_from_dict(observed_track_payload(ground_speed_m_s=ground)),
        context=context, wind=table,
    )
    assert corrected.speed_criterion == OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
    assert corrected.crossing_airspeed_estimate_ms == pytest.approx(ground - _TAIL_MS)
    assert corrected.speed_result == "fail"
    assert corrected.wind["headwind_ms"] == pytest.approx(-_TAIL_MS)
    assert corrected.speed_margin_ms == pytest.approx(2.0 - _TAIL_MS)
    # The same window frames both: the correction moves the judged value, not the bounds.
    assert corrected.speed_bounds.lower_ms == pytest.approx(proxy.speed_bounds.lower_ms)


def test_the_report_counts_estimated_and_proxy_rows(tmp_path):
    table = _table(tmp_path)
    contexts = {("KRDU", "05L"): assessment_context()}
    stale = observed_track_payload(ground_speed_m_s=72.0)
    stale["source"]["landing_time_utc"] = "2026-08-12T01:40:00Z"     # no report within 30 min
    report = evaluate_batch(
        [
            record_from_dict(observed_track_payload(ground_speed_m_s=72.0)),
            record_from_dict(stale),
        ],
        contexts=contexts,
        winds={"KRDU": table},
    )
    assert report["wind_counts"] == {"estimated": 1, "unavailable": 1}
    assert report["speed_indeterminate_reasons"] == {}
    assert report["speed_result_counts_by_criterion"] == {
        OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID: {"pass": 1, "fail": 0, "indeterminate": 0},
        OBSERVED_SPEED_CRITERION_ID: {"pass": 1, "fail": 0, "indeterminate": 0},
    }
    rows = report["trajectories"]
    assert rows[0]["speed_margin_ms"] is not None and rows[1]["speed_margin_ms"] is not None
    assert rows[0]["bounds"]["speed_criterion"] == OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
    assert rows[0]["crossing_airspeed_estimate_ms"] == pytest.approx(72.0 - kt_to_ms(10.0))
    assert rows[0]["wind"]["status"] == "estimated"
    assert rows[1]["bounds"]["speed_criterion"] == OBSERVED_SPEED_CRITERION_ID
    assert rows[1]["crossing_airspeed_estimate_ms"] is None
    assert rows[1]["wind"]["status"] == "unavailable"
    assert report["crossing_airspeed_estimate_ms"]["max"] == pytest.approx(72.0 - kt_to_ms(10.0))
    methodology = report["methodology"]["terminal_speed"]["observed_wind_correction"]
    assert methodology["criterion"] == OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
    assert "limits" in methodology and "uncertainty_ms" not in methodology


def test_a_calm_report_is_an_estimate_with_zero_correction(tmp_path):
    """289 of KRDU's 2,256 reports are calm: the estimate criterion applies and the
    correction is zero."""
    table = _table(tmp_path)
    calm = observed_track_payload(ground_speed_m_s=70.0)
    calm["source"]["landing_time_utc"] = "2026-08-11T23:05:00Z"   # nearest: the calm 23:00
    result = evaluate_record(record_from_dict(calm), context=assessment_context(), wind=table)
    assert result.speed_criterion == OBSERVED_AIRSPEED_ESTIMATE_CRITERION_ID
    assert result.crossing_airspeed_estimate_ms == pytest.approx(70.0)
    assert result.wind["headwind_ms"] == 0.0 and result.wind["status"] == "estimated"


def test_several_csv_files_join_one_station_and_malformed_files_raise(tmp_path):
    first = tmp_path / "KRDU" / "asos_a.csv"
    first.parent.mkdir()
    first.write_text(IEM_CSV, encoding="utf-8")
    second = tmp_path / "KRDU" / "asos_b.csv"
    second.write_text("station,valid,drct,sknt,gust\nRDU,2026-08-13 00:51,270.00,6.00,M\n", encoding="utf-8")
    table = WindTable.from_iem_csv([first, second])
    assert len(table.observations) == 5 and table.observations[-1].direction_deg == 270.0

    bad_header = tmp_path / "KRDU" / "bad.csv"
    bad_header.write_text("station,valid,drct,sknt\nRDU,2026-08-13 00:51,270.00,6.00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        WindTable.from_iem_csv([bad_header])
    short_row = tmp_path / "KRDU" / "short.csv"
    short_row.write_text("station,valid,drct,sknt,gust\nRDU,2026-08-13 00:51,270.00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed row"):
        WindTable.from_iem_csv([short_row])
    mixed = tmp_path / "KRDU" / "mixed.csv"
    mixed.write_text("station,valid,drct,sknt,gust\nRDU,2026-08-13 00:51,270.00,6.00,M\nSJC,2026-08-13 00:51,270.00,6.00,M\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mixes stations"):
        WindTable.from_iem_csv([mixed])


def test_without_any_wind_table_the_batch_is_judged_on_the_proxy_and_says_so():
    report = evaluate_batch(
        [record_from_dict(observed_track_payload(ground_speed_m_s=72.0))],
        contexts={("KRDU", "05L"): assessment_context()},
    )
    assert report["wind_counts"] == {"estimated": 0, "unavailable": 1}
    assert report["trajectories"][0]["wind"]["status"] == "unavailable"
    assert report["trajectories"][0]["speed_result"] == "pass"
