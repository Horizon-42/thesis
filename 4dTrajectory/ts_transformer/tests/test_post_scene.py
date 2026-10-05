"""Stage C, C1: windows of recorded traffic and their scenes (post-training §3, §8 C1; D29, D93, C32)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ts_transformer.data.day_split import SealedDay
from ts_transformer.instructions.artefact import load_candidates, load_day_split
from ts_transformer.post import scene as post_scene
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import (
    INSERTED, INSERTED_SUFFIX, LEADER_MOVED, REAL, airport_scenes, census, inserted_window, leader_moved_window,
    near_day_cut, next_ahead, real_windows, recorded,
)
from ts_transformer.tests.post_support import categories, finals, scene_artefact, straight_in, train_noon
from ts_transformer.tests.support import INSTRUCTION_STEP_S

DELTA = 4.0


def _flights():
    """Three train flights onto 09 in a stream 120 s apart, one far later, and a select flight in the same minutes."""
    return {"train": [straight_in("KXXX:a", train_noon(0.0)), straight_in("KXXX:b", train_noon(120.0)),
                      straight_in("KXXX:c", train_noon(242.0)), straight_in("KXXX:far", train_noon(7_200.0))],
            "select": [straight_in("KXXX:s", train_noon(60.0, split="select"))]}


@pytest.fixture
def built(tmp_path):
    flights = _flights()
    spec = scene_artefact(tmp_path / "art", flights, interval_s=DELTA)
    geometries = load_candidates(tmp_path / "art")
    scenes, signals = airport_scenes(tmp_path / "art", "train", spec, DELTA, geometries, categories)
    return tmp_path / "art", spec, scenes, signals, geometries


def test_a_scene_holds_its_split_only_and_never_a_test_day(built, tmp_path):
    directory, spec, scenes, _, geometries = built
    assert [f.key for f in scenes["KXXX"].flights] == ["KXXX:a", "KXXX:b", "KXXX:c", "KXXX:far"]
    select, _ = airport_scenes(directory, "select", spec, DELTA, geometries, categories)
    assert [f.key for f in select["KXXX"].flights] == ["KXXX:s"]
    with pytest.raises(ValueError, match="splits"):
        airport_scenes(directory, "test", spec, DELTA, geometries, categories)
    test_day = straight_in("KXXX:t", train_noon(0.0, split="test"))
    with pytest.raises(SealedDay):
        scene_artefact(tmp_path / "sealed", {"train": [test_day]})


def test_the_other_aircraft_at_a_step_are_exactly_the_flights_in_the_air(built):
    directory, spec, scenes, signals, _ = built
    windows = real_windows(directory, "train", spec, DELTA, scenes, signals)
    assert [w.commanded.key for w in windows] == ["KXXX:a", "KXXX:b", "KXXX:c", "KXXX:far"]
    assert all(w.kind == REAL for w in windows)
    window = windows[1]
    for step in range(0, 200):
        time_s = window.step_s(step)
        expected = [f.key for f in scenes["KXXX"].flights
                    if f.key != window.commanded.key and f.start_s <= time_s <= f.end_s]
        assert list(window.others_at(step).keys) == expected


def test_the_steps_are_on_utc_multiples_of_delta(tmp_path):
    for delta in (2.0, 4.0, 8.0):
        flights = {"train": [straight_in("KXXX:a", train_noon(6.0)), straight_in("KXXX:b", train_noon(30.0))]}
        spec = scene_artefact(tmp_path / f"d{delta:g}", flights, interval_s=delta)
        geometries = load_candidates(tmp_path / f"d{delta:g}")
        scenes, signals = airport_scenes(tmp_path / f"d{delta:g}", "train", spec, delta, geometries, categories)
        for window in real_windows(tmp_path / f"d{delta:g}", "train", spec, delta, scenes, signals):
            for step in range(20):
                assert window.step_s(step) % delta == 0.0
            assert window.first_step_s - window.row0_s == 16.0
            assert window.row0_s - window.commanded.start_s < delta


def test_a_recorded_aircraft_moves_by_its_displacement_only(built):
    directory, spec, scenes, signals, geometries = built
    flight = signals[0]
    changed = replace(flight, track_deg=flight.track_deg + 33.0, ground_speed_mps=flight.ground_speed_mps * 3.0,
                      vertical_rate_mps=-flight.vertical_rate_mps)
    none = np.zeros(flight.n_rows, dtype=bool)
    a = recorded(flight, geometries["KXXX"], DELTA, INSTRUCTION_STEP_S, none, categories)
    b = recorded(changed, geometries["KXXX"], DELTA, INSTRUCTION_STEP_S, none, categories)
    for time_s in (a.start_s, a.start_s + 2.0, a.first_step_s + 40.0, a.end_s):
        assert a.at_step(time_s, DELTA) == b.at_step(time_s, DELTA)
    key, at, before, known, *_ = a.at_step(a.start_s, DELTA)
    assert not known and before == (0.0, 0.0, 0.0)


def test_a_recorded_runway_is_absent_up_to_its_first_predicted_step(built):
    _, _, scenes, _, _ = built
    flight = scenes["KXXX"].flights[0]
    assert flight.first_step_s - 16.0 - flight.start_s < DELTA
    for time_s in np.arange(flight.start_s, flight.first_step_s + 0.1, 2.0):
        assert flight.at_step(float(time_s), DELTA)[4] == -1
    assert flight.at_step(flight.first_step_s + 2.0, DELTA)[4] == 0


def test_window_a_inserts_one_shifted_flight_of_another_time(built):
    directory, spec, scenes, signals, _ = built
    window = real_windows(directory, "train", spec, DELTA, scenes, signals)[0]
    rng = np.random.default_rng(1337)
    inserted = inserted_window(window, rng, landing_shift_s=(-120.0, 120.0), apart_s=3_600.0)
    assert inserted.kind == INSERTED
    (key, shift), = inserted.moved
    assert key == "KXXX:far" + INSERTED_SUFFIX and shift % DELTA == 0.0
    moved = inserted.scene.flight(key)
    assert abs(moved.landing_s - window.commanded.landing_s) <= 120.0 + DELTA
    assert inserted.scene.base is window.scene                       # shared, never copied
    assert key in inserted.others_at(int(round((moved.start_s - window.row0_s) / DELTA)) + 1).keys
    assert inserted_window(window, rng, landing_shift_s=(0.0, 0.0), apart_s=1e9) is None
    with pytest.raises(ValueError, match="real window"):
        inserted_window(inserted, rng, landing_shift_s=(-120.0, 120.0), apart_s=3_600.0)


def test_window_d_moves_the_aircraft_next_ahead(built):
    directory, spec, scenes, signals, geometries = built
    separation = airport_separation(geometries["KXXX"])
    windows = real_windows(directory, "train", spec, DELTA, scenes, signals)
    assert next_ahead(windows[1], separation) == "KXXX:a"
    assert next_ahead(windows[0], separation) is None
    moved = leader_moved_window(windows[1], separation, np.random.default_rng(1337), shift_s=(-60.0, 60.0))
    assert moved.kind == LEADER_MOVED
    (key, shift), = moved.moved
    assert key == "KXXX:a" and shift != 0.0 and shift % DELTA == 0.0
    original = windows[1].scene.flights[0]
    assert moved.scene.flight("KXXX:a").start_s == original.start_s + shift
    for step in range(0, 150):                     # a is in the air at its moved times only
        time_s = moved.step_s(step)
        assert ("KXXX:a" in moved.others_at(step).keys) == (original.start_s + shift <= time_s <= original.end_s + shift)
    assert leader_moved_window(windows[0], separation, np.random.default_rng(1), shift_s=(-60.0, 60.0)) is None


def test_the_census_counts_the_traffic_at_the_first_predicted_step(built):
    directory, spec, scenes, signals, geometries = built
    windows = real_windows(directory, "train", spec, DELTA, scenes, signals)
    counted = census(windows, {"KXXX": airport_separation(geometries["KXXX"])}, load_day_split(directory))["KXXX"]
    assert counted["windows"] == 4
    # a: b not yet in the air at its first step (b enters 120 s later); b: a; c: a and b; far: none
    assert counted["others_at_first_step"]["max"] == 2 and counted["others_at_first_step"]["share_none"] == 0.5
    assert counted["windows_admitting_D"] == 2 and counted["share_with_leader_in_air"] == 0.5
    assert counted["near_day_cut"] == 0


def test_a_window_near_a_cut_between_operating_days_is_counted_when_the_day_beside_is_of_another_split(built):
    directory, spec, scenes, signals, _ = built
    window = real_windows(directory, "train", spec, DELTA, scenes, signals)[0]
    days = load_day_split(directory)
    assert not near_day_cut(window, days, 3_600.0)          # noon: far from the 09Z cut
    seen, seen_before = set(), set()
    for day in days.days["train"]:
        start = post_scene.operating_day_span_s(day)[0]
        early = replace(window, commanded=replace(window.commanded, landing_s=start + 300.0,
                                                  first_step_s=start - 600.0))
        before = post_scene.landing_day_s(start - 1.0) in days.days["train"]
        assert near_day_cut(early, days, 3_600.0) == (not before)
        seen_before.add(before)
        assert not near_day_cut(replace(early, commanded=replace(early.commanded, first_step_s=start + 60.0)),
                                days, 3_600.0)
        end = post_scene.operating_day_span_s(day)[1]
        late = replace(window, commanded=replace(window.commanded, landing_s=end - 100.0, first_step_s=end - 500.0))
        same = post_scene.landing_day_s(end) in days.days["train"]
        assert near_day_cut(late, days, 3_600.0) == (not same)
        assert not near_day_cut(late, days, 50.0)
        seen.add(same)
    assert seen == {True, False} and seen_before == {True, False}


def test_the_census_runner_writes_its_census_and_the_edge_reference(built, tmp_path, monkeypatch):
    import json

    from ts_transformer.experiments import post_windows
    from ts_transformer.repo_layout import REPO_ROOT

    directory = built[0]
    monkeypatch.setattr(post_windows, "airport_scenes",
                        lambda *a, **k: post_scene.airport_scenes(*a, **k, category_of=categories))
    monkeypatch.setattr(post_windows, "airport_finals", lambda geometry, root: finals(geometry))
    out = tmp_path / "census"
    assert post_windows.main(["--instructions", str(directory), "--interval-s", "4", "--out", str(out)]) == 0
    record = json.loads((out / "census.json").read_text())
    assert record["schema"] == post_windows.CENSUS_SCHEMA and set(record["splits"]) == {"train", "select"}
    assert record["splits"]["train"]["airports"]["KXXX"]["windows"] == 4
    assert record["splits"]["select"]["airports"]["KXXX"]["windows"] == 1
    assert record["edges_reference"]["max_difference"] == 0.0 and (out / "conformance" / "edges.npz").is_file()
    lost = record["splits"]["train"]["lost_at_first_step"]["KXXX"]
    assert lost["without_runway"] == lost["with_recorded_runway"] == 0          # 120 s apart: no loss
    sampled = tmp_path / "sampled"
    post_windows.main(["--instructions", str(directory), "--interval-s", "4", "--out", str(sampled),
                       "--splits", "train", "--sample", "2"])
    assert json.loads((sampled / "census.json").read_text())["splits"]["train"]["airports"]["KXXX"]["windows"] == 2
    with pytest.raises(SystemExit):
        post_windows.main(["--instructions", str(directory), "--interval-s", "4", "--out", str(out)])
    forbidden = REPO_ROOT / "4dTrajectory" / "outputs" / "post_windows_test_never_written"
    with pytest.raises(SystemExit):
        post_windows.main(["--instructions", str(directory), "--interval-s", "4", "--out", str(forbidden)])
    assert not forbidden.exists()
    with pytest.raises(SystemExit):
        post_windows.main(["--instructions", str(directory), "--interval-s", "4", "--out", str(tmp_path / "v"),
                           "--splits", "val"])


def test_g_is_true_after_each_labelled_go_around_up_to_its_runway_word():
    from ts_transformer.post.scene import go_around_rows

    assert np.flatnonzero(go_around_rows(20, [3, 10], [6, 12])).tolist() == [4, 5, 6, 11, 12]
    assert not go_around_rows(5, [], []).any()
    with pytest.raises(ValueError, match="later row"):
        go_around_rows(20, [6], [6])
    with pytest.raises(ValueError, match="later row"):
        go_around_rows(20, [6], [20])


def test_a_recorded_aircraft_reads_its_g_from_its_labelled_sentence(built, monkeypatch):
    """The user's decision (2026-10-05): a recorded flight with a sentence has G from its labelled go-around rows; one
    without has G false. A recorded aircraft in G is not established, wherever it is."""
    from ts_transformer.post.established import established
    from ts_transformer.tests.post_support import finals

    directory, spec, _, _, geometries = built
    real = post_scene.load_sentences

    def with_a_go_around(d, split, s, fields):
        data = dict(real(d, split, s, fields))
        if split == "train":           # flight b (signal index 1): a go-around at row 100, its runway said at row 140
            # the sentences in another order than the signals: G must follow the signal index, not the sentence's place
            data["signal_index"] = data["signal_index"][::-1].copy()
            counts = np.zeros(len(data["signal_index"]), dtype=np.int64)
            counts[list(data["signal_index"]).index(1)] = 1
            data["go_around_offsets"] = np.concatenate(([0], np.cumsum(counts)))
            data["go_around_row"], data["runway_again_row"] = np.array([100]), np.array([140])
        return data

    monkeypatch.setattr(post_scene, "load_sentences", with_a_go_around)
    scenes, _ = airport_scenes(directory, "train", spec, DELTA, geometries, categories)
    flights = {f.key: f for f in scenes["KXXX"].flights}
    assert np.flatnonzero(flights["KXXX:b"].go_around).tolist() == list(range(101, 141))
    assert not flights["KXXX:a"].go_around.any() and not flights["KXXX:c"].go_around.any()
    b = flights["KXXX:b"]
    inside = [b.start_s + 2.0 * row for row in (120, 150)]               # on the final, lined up; in G at row 120
    seen = [post_scene.AircraftAt.of([b.at_step(t, DELTA)]) for t in inside]
    assert [bool(s.go_around[0]) for s in seen] == [True, False]
    fin = finals(geometries["KXXX"])
    assert [bool(established(s, geometries["KXXX"], fin, 2.0)[0]) for s in seen] == [False, True]
    # commanded, b carries no labelled G (its G is its words' in the loop); a window refuses a commanded record with G
    windows = real_windows(directory, "train", spec, DELTA, scenes, post_scene.load_signals(directory, "train"))
    assert not windows[1].commanded.go_around.any()
    with pytest.raises(ValueError, match="words in force"):
        replace(windows[1], commanded=b)
    # moved (window D of c, whose leader is b), b keeps G with its rows
    from ts_transformer.post.runways import airport_separation

    moved = leader_moved_window(windows[2], airport_separation(geometries["KXXX"]), np.random.default_rng(3),
                                shift_s=(-60.0, 60.0))
    (key, shift), = moved.moved
    assert key == "KXXX:b"
    assert np.array_equal(moved.scene.flight(key).go_around, b.go_around)
    assert [bool(moved.scene.flight(key).at_step(t + shift, DELTA)[7]) for t in inside] == [True, False]


def test_a_window_that_opens_inside_a_loss_of_separation_is_found(built):
    """C1's census for the user: the commanded aircraft, on its record, at its first predicted step — a copy of itself
    8 s ahead is inside 3 NM and 1,000 ft; the stream 120 s apart is not."""
    from ts_transformer.post.scene import MovedScene
    from ts_transformer.post.traffic import loss_at_first_step

    directory, spec, scenes, signals, geometries = built
    separation, fin = airport_separation(geometries["KXXX"]), finals(geometries["KXXX"])
    windows = real_windows(directory, "train", spec, DELTA, scenes, signals)
    for with_runway in (False, True):
        assert loss_at_first_step(windows[1], separation, fin, INSTRUCTION_STEP_S, recorded_runway=with_runway) is None
    own = windows[1].commanded
    key = own.key + INSERTED_SUFFIX
    ahead = replace(windows[1], kind=INSERTED, moved=((key, -8.0),),
                    scene=MovedScene(windows[1].scene, added=(replace(scenes["KXXX"].flights[1], go_around=np.zeros_like(
                        own.go_around)).shifted(-8.0, DELTA, key=key),)))
    for with_runway in (False, True):
        loss = loss_at_first_step(ahead, separation, fin, INSTRUCTION_STEP_S, recorded_runway=with_runway)
        assert loss is not None and 0 in loss.responsible
