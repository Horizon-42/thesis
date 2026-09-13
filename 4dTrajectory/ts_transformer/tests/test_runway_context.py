"""The causal runway baselines of runway-intent R0 (`data.runway_context`)."""

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from ts_transformer.data.runway_context import (
    ContextLanding,
    Pick,
    RunwayContext,
    WindReport,
    bearing_sector,
    direction_groups,
    load_metar,
)

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
COURSES = {"05L": 45.0, "05R": 45.0, "23L": 225.0, "23R": 225.0}


def _context(landings=(), winds=(), majority=None, **overrides) -> RunwayContext:
    settings = dict(window=timedelta(minutes=30), sector_window=timedelta(minutes=60),
                    metar_delay=timedelta(minutes=10), calm_kt=3.0)
    settings.update(overrides)
    return RunwayContext(list(landings), COURSES, majority or Counter({"23R": 5, "05L": 3}),
                         list(winds), **settings)


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_direction_groups_join_parallels_and_split_directions():
    groups = direction_groups({**COURSES, "32": 322.0})
    assert groups["05L"] == groups["05R"] != groups["23L"] == groups["23R"]
    assert len(set(groups.values())) == 3


def test_every_rule_reads_only_landings_strictly_before_the_anchor():
    context = _context([ContextLanding(_at(0), "05R")])
    at_the_landing = context.picks(_at(0), sector=0, track_course_deg=45.0)
    assert at_the_landing["B1_active_config"].fallback      # the landing at t is not yet known
    after = context.picks(_at(1), sector=0, track_course_deg=45.0)
    assert after["B1_active_config"].runway == "05R" and not after["B1_active_config"].fallback
    too_old = context.picks(_at(31), sector=0, track_course_deg=45.0)
    assert too_old["B1_active_config"].fallback and too_old["B1_active_config"].runway == "23R"


def test_the_majority_is_the_one_it_is_given_ties_broken_by_name():
    context = _context(majority=Counter({"05L": 4, "23L": 4}))
    assert context.picks(_at(0), sector=0, track_course_deg=0.0)["B0_majority"].runway == "05L"


def test_the_course_gate_keeps_the_rule_to_runways_the_aircraft_is_flying_toward():
    landings = [ContextLanding(_at(-5), "23R"), ContextLanding(_at(-4), "23R"),
                ContextLanding(_at(-3), "05L")]
    picks = _context(landings).picks(_at(0), sector=0, track_course_deg=40.0)
    assert picks["B1_active_config"].runway == "23R"
    assert picks["B2_active_config_gated"].runway == "05L"


def test_same_sector_takes_the_latest_landing_from_that_sector_only():
    landings = [ContextLanding(_at(-20), "05R", sector=2), ContextLanding(_at(-10), "05L", sector=2),
                ContextLanding(_at(-5), "23R", sector=6)]
    picks = _context(landings).picks(_at(0), sector=2, track_course_deg=45.0)
    assert picks["B3_same_sector_last"] == Pick("05L", False)
    unseen = _context(landings).picks(_at(0), sector=4, track_course_deg=45.0)
    assert unseen["B3_same_sector_last"].fallback


def test_the_wind_rule_waits_for_the_report_and_picks_the_headwind_group():
    landings = [ContextLanding(_at(-5), "23L"), ContextLanding(_at(-4), "05R")]
    # wind FROM 050 at 12 kt: the 05 group lands into it
    winds = [WindReport(_at(-5), 50.0, 12.0)]
    before_published = _context(landings, winds).picks(_at(0), sector=0, track_course_deg=45.0)
    assert before_published["B4_wind"].fallback              # observed 5 min ago, 10 min delay
    published = _context(landings, winds).picks(_at(6), sector=0, track_course_deg=45.0)
    assert published["B4_wind"].runway == "05R" and not published["B4_wind"].fallback
    calm = _context(landings, [WindReport(_at(-30), 50.0, 2.0)]).picks(_at(0), sector=0, track_course_deg=45.0)
    assert calm["B4_wind"].fallback


def test_the_wind_rule_skips_a_direction_the_airport_barely_uses():
    courses = {"11": 106.0, "29": 286.0, "20": 195.0}
    context = RunwayContext(
        [], courses, Counter({"11": 60, "29": 39, "20": 1}),
        [WindReport(_at(-60), 190.0, 10.0)],      # straight down runway 20
        window=timedelta(minutes=30), sector_window=timedelta(minutes=60),
        metar_delay=timedelta(minutes=10), calm_kt=3.0,
    )
    pick = context.picks(_at(0), sector=0, track_course_deg=106.0)["B4_wind"]
    assert pick.runway == "11"                   # 20 carries 1 % — out of the choice; 11 heads into it


def test_bearing_sectors_are_eight_by_forty_five_degrees():
    ref_lon, ref_lat = -78.79, 35.88
    assert bearing_sector(ref_lon, ref_lat + 0.1, ref_lon, ref_lat) == 0        # due north
    assert bearing_sector(ref_lon + 0.1, ref_lat, ref_lon, ref_lat) == 2        # due east
    assert bearing_sector(ref_lon, ref_lat - 0.1, ref_lon, ref_lat) == 4        # due south


def test_metar_missing_fields_stay_missing(tmp_path):
    path = tmp_path / "asos.csv"
    path.write_text("station,valid,drct,sknt,gust\nRDU,2026-06-01 11:51,M,4.00,M\n"
                    "RDU,2026-06-01 10:51,150.00,6.00,M\n", encoding="utf-8")
    reports = load_metar([path])
    assert [r.valid.hour for r in reports] == [10, 11]
    assert reports[1].direction_deg is None and reports[1].speed_kt == pytest.approx(4.0)
