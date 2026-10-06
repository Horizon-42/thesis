"""The arrivals roster as the interactive mode reads it: rows only, a day, a block, the tracks a window can share."""

import json
from datetime import date

import pytest

import traffic_roster as roster
from trajectory_data_process.harvest.utc import iso_utc, iso_utc_ms

T0 = 1_800_000_000.0                           # an epoch second; every row is placed relative to it


def row(key, entry_s, duration_s=200.0, runway="05L"):
    return {"flight_key": key, "callsign": key[:3], "icao24": "a00001", "runway": runway,
            "entry_time_utc": iso_utc_ms(T0 + entry_s), "landing_time_utc": iso_utc(T0 + entry_s + duration_s),
            "arrival_duration_s": duration_s}


def write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "x", "records": rows}))
    return path


def test_the_roster_is_read_from_the_manifest_alone(tmp_path):
    [found] = roster.read_roster(write_manifest(tmp_path, [row("AAA_05L", 10.0, 170.5)]))
    assert (found.flight_key, found.callsign, found.runway) == ("AAA_05L", "AAA", "05L")
    assert found.entry_utc_s == pytest.approx(T0 + 10.0) and found.end_utc_s == pytest.approx(T0 + 180.5)
    assert found.landing_utc_s == pytest.approx(T0 + 180.0, abs=1.0)           # the landing stamp has whole seconds


def test_a_flight_without_a_callsign_is_a_row_with_none(tmp_path):
    rows = roster.read_roster(write_manifest(tmp_path, [{**row("NIL_05L", 0.0), "callsign": None}, row("AAA_05L", 10.0)]))
    assert [r.callsign for r in rows] == [None, "AAA"]
    assert [r.flight_key for r in roster.landing_in(rows, T0, T0 + 1000.0)] == ["NIL_05L", "AAA_05L"]


def test_a_utc_day_is_half_open_and_the_landings_in_it_come_in_landing_order(tmp_path):
    day = date(2026, 5, 21)
    start, end = roster.utc_day_bounds_s(day)
    assert end - start == 86400.0
    rows = roster.read_roster(write_manifest(tmp_path, [
        row("LATE_05L", start + 3600 - T0), row("BEFORE_05L", start - 300 - T0),
        row("EARLY_05L", start + 60 - T0), row("NEXT_05L", end - 100 - T0)]))
    assert [r.flight_key for r in roster.landing_in(rows, start, end)] == ["EARLY_05L", "LATE_05L"]
    # NEXT lands at end + 100 s: out; BEFORE lands at start - 100 s: out


def test_the_aircraft_that_can_share_a_window_are_those_in_the_air_during_it(tmp_path):
    rows = roster.read_roster(write_manifest(tmp_path, [
        row("OWN_05L", 0.0),                    # the window is 0 .. 2000 s after its entry
        row("AHEAD_05L", -150.0),               # still in the air (ends at +50 s)
        row("GONE_05L", -400.0),                # landed 200 s before the window
        row("INSIDE_05L", 1500.0),              # enters inside it
        row("LATER_05L", 2500.0),               # enters after it
    ]))
    own = rows[0]
    assert [r.flight_key for r in roster.m1_near(rows, own, 2000.0)] == ["OWN_05L", "AHEAD_05L", "INSIDE_05L"]


def test_a_window_edge_that_float_rounding_could_move_keeps_the_aircraft_loaded(tmp_path):
    rows = roster.read_roster(write_manifest(tmp_path, [
        row("OWN_05L", 0.0), row("TOUCH_05L", -200.0 - 0.4), row("EDGE_05L", 2000.0 + 0.4)]))
    assert {r.flight_key for r in roster.m1_near(rows, rows[0], 2000.0)} == {"OWN_05L", "TOUCH_05L", "EDGE_05L"}


def test_a_block_reads_the_tracks_from_its_first_entry_to_its_last_entry_plus_the_horizon(tmp_path):
    rows = roster.read_roster(write_manifest(tmp_path, [
        row("B1_05L", 0.0), row("B2_05L", 300.0),       # the block (their entries; their landings 200 s later)
        row("AHEAD_05L", -100.0),                       # in the air at the first entry
        row("GONE_05L", -500.0),                        # landed before it
        row("EDGE_05L", 2290.0),                        # enters inside last entry + 2000 s
        row("LATER_05L", 2500.0)]))
    block = [r for r in rows if r.flight_key.startswith("B")]
    assert [r.flight_key for r in roster.m2_near(rows, block, 2000.0)] == [
        "B1_05L", "B2_05L", "AHEAD_05L", "EDGE_05L"]
