"""Stage C: the landings of a window's scene (post-training §3 "Landing context", D105)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.data.day_split import parse_utc
from ts_transformer.instructions.artefact import load_candidates, load_day_split, signals_flights
from ts_transformer.post.landings import roster_key, window_landings
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import (
    INSERTED_SUFFIX, MovedScene, airport_scenes, inserted_window, leader_moved_window, real_windows,
)
from ts_transformer.prior.landings import LANDINGS_WINDOW_S, roster_landings
from ts_transformer.tests.post_support import categories, scene_artefact, straight_in, train_noon

DELTA = 4.0


@pytest.fixture
def built(tmp_path):
    flights = {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(120.0)),
                         straight_in("KXXX:c", train_noon(242.0)), straight_in("KXXX:far", train_noon(7_200.0))]}
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    directory = tmp_path / "art"
    geometries = load_candidates(directory)
    scenes, signals = airport_scenes(directory, "train", spec, DELTA, geometries, categories)
    windows = real_windows(directory, "train", spec, DELTA, scenes, signals)
    records = [{"flight_key": roster_key(f["dataset_id"], f["airport"]), "outcome": "assigned", "runway": f["runway"],
                "landing_time_utc": f["landing_time_utc"]} for f in signals_flights(directory, "train")]
    days = load_day_split(directory)
    roster = roster_landings(records, [c.ident for c in geometries["KXXX"].candidates], days)
    return windows, roster, days, geometries


def _count(index, time_s, without):
    return int(index.counts_before(np.array([time_s]), without=without).sum())


def test_a_real_window_counts_the_rosters_landings(built):
    windows, roster, days, _ = built
    for window in windows:
        assert window_landings(window, roster) is roster


def test_window_a_counts_the_inserted_aircraft_at_its_shifted_time(built):
    windows, roster, days, _ = built
    window = inserted_window(windows[0], np.random.default_rng(1337), landing_shift_s=(-120.0, 120.0), apart_s=3_600.0)
    (key, shift), = window.moved
    landings = window_landings(window, roster)
    source = next(x for x in roster.landings if x.flight_key == "far")
    added = next(x for x in landings.landings if x.flight_key == roster_key(key, "KXXX"))
    assert added.flight_key == "far" + INSERTED_SUFFIX
    assert added.time_s == source.time_s + shift and added.runway == source.runway
    assert len(landings.landings) == len(roster.landings) + 1 and source in landings.landings   # the source stays
    own = roster_key(window.commanded.key, "KXXX")
    just_after = added.time_s + 1.0
    assert _count(landings, just_after, own) == _count(roster, just_after, own) + 1
    assert _count(landings, added.time_s, own) == _count(roster, added.time_s, own)              # not before it lands


def test_window_d_counts_the_moved_aircraft_at_its_moved_time(built):
    windows, roster, days, geometries = built
    window = leader_moved_window(windows[1], airport_separation(geometries["KXXX"]), np.random.default_rng(1337),
                                 shift_s=(-60.0, 60.0))
    (key, shift), = window.moved
    landings = window_landings(window, roster)
    before = next(x for x in roster.landings if x.flight_key == roster_key(key, "KXXX"))
    after = next(x for x in landings.landings if x.flight_key == before.flight_key)
    assert after.time_s == before.time_s + shift and len(landings.landings) == len(roster.landings)
    own = roster_key(window.commanded.key, "KXXX")
    early, late = sorted((before.time_s, after.time_s))
    between = early + 1.0                                     # landed in one of the two indexes, not yet in the other
    assert abs(_count(landings, between, own) - _count(roster, between, own)) == 1
    assert _count(landings, late + 1.0, own) == _count(roster, late + 1.0, own)
    assert _count(landings, late + LANDINGS_WINDOW_S + 1.0, own) == _count(roster, late + LANDINGS_WINDOW_S + 1.0, own)


def test_a_landing_shifted_onto_a_test_day_is_left_out(built):
    windows, roster, days, _ = built
    window = windows[1]
    leader = window.scene.flights[window.scene.index["KXXX:a"]]
    landing = next(x for x in roster.landings if x.flight_key == "a")
    test_noon = parse_utc(f"{days.days['test'][0]}T12:00:00Z").timestamp()
    shift = round((test_noon - landing.time_s) / DELTA) * DELTA
    moved = replace(window, kind="D", scene=MovedScene(window.scene, replaced={"KXXX:a": leader.shifted(shift, DELTA)}),
                    moved=(("KXXX:a", shift),))
    landings = window_landings(moved, roster)
    assert "a" not in {x.flight_key for x in landings.landings}
    assert landings.sealed == roster.sealed + 1


def test_a_window_moves_one_flight(built):
    windows, roster, days, _ = built
    with pytest.raises(ValueError, match="moves one flight"):
        window_landings(replace(windows[0], kind="A", moved=()), roster)


def test_an_inserted_landing_on_a_test_day_is_left_out_and_a_flight_missing_from_the_roster_fails(built):
    windows, roster, days, _ = built
    window = windows[0]
    source = window.scene.flights[window.scene.index["KXXX:far"]]
    landing = next(x for x in roster.landings if x.flight_key == "far")
    shift = round((parse_utc(f"{days.days['test'][0]}T12:00:00Z").timestamp() - landing.time_s) / DELTA) * DELTA
    key = "KXXX:far" + INSERTED_SUFFIX
    inserted = replace(window, kind="A", moved=((key, shift),),
                       scene=MovedScene(window.scene, added=(source.shifted(shift, DELTA, key=key),)))
    landings = window_landings(inserted, roster)
    assert landings.landings == roster.landings and landings.sealed == roster.sealed + 1
    missing = replace(roster, landings=tuple(x for x in roster.landings if x.flight_key != "far"))
    with pytest.raises(KeyError):
        window_landings(inserted, missing)
