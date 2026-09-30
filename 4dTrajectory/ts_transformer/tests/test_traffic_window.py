"""The prior commanding every aircraft of a window (`experiments/traffic_window`, multi-aircraft design §6.6 step 7): the
windows over a segment, the others a window replays, and the draw of the flights it commands. On the scene-data
fixture (a tmp artefact)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ts_transformer.instructions.words import Words
from ts_transformer.tests.test_traffic_scene_data import _artefact

STEP_S = 2.0
#: Eight labelled flights 200 s apart, each 460 s in the scene: one segment of 1,860 s.
CHAINED = [0.0, 200.0, 400.0, 600.0, 800.0, 1_000.0, 1_200.0, 1_400.0]


def _airport(tmp_path, monkeypatch, entries=CHAINED, labelled=tuple(range(len(CHAINED)))):
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_speaking import scene_airports

    directory, manifest, spec, _ = _artefact(tmp_path, entries, list(labelled))
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    airports, _ = scene_airports(directory, "train", spec, ("KXXX",), None, 2_048)
    return directory, airports, spec


def _keys(*numbers):
    return tuple(f"KXXX:f{n}" for n in numbers)


def test_windows_open_every_ten_minutes_over_a_segment_with_the_flights_entering_in_them(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window import window_tiles

    _, airports, _ = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    first = airport.tracks["KXXX:f0"].first_step_s
    tiles = window_tiles(airport, STEP_S)
    # f0 … f7 enter 0 … 1,400 s after the segment opens; windows [0, 1200), [600, 1800), [1200, 2400) s
    assert tiles == [(first, _keys(0, 1, 2, 3, 4, 5)), (first + 600.0, _keys(3, 4, 5, 6, 7)),
                     (first + 1_200.0, _keys(6, 7))]


def test_a_background_flight_is_never_commanded_and_a_window_without_a_sentence_is_no_tile(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_window import window_tiles

    # f1 and f2 background; f3 an hour later, alone
    _, airports, _ = _airport(tmp_path, monkeypatch, [0.0, 200.0, 400.0, 3_600.0], (0, 3))
    tiles = window_tiles(airports["KXXX"], STEP_S)
    assert [keys for _, keys in tiles] == [_keys(0), _keys(3)]


def test_one_commanded_aircraft_replays_what_its_one_aircraft_scene_does(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_speaking import scene_of
    from ts_transformer.experiments.traffic_window import window_of

    _, airports, _ = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    alone = window_of(airport, 0.0, _keys(3), [120.0], STEP_S)
    scene = scene_of(airport, "KXXX:f3", 120.0, STEP_S)
    # f1 and f2 entered 400 and 200 s before f3 and are still in the air; f4 enters after f3's 136 s
    assert alone.others == scene.others == _keys(1, 2)
    assert alone.scene("KXXX:f3", STEP_S) == scene
    # two commanded: each other's first among the others, the replayed spanning both time limits (f4's reaches f5)
    both = window_of(airport, 0.0, _keys(3, 4), [120.0, 300.0], STEP_S)
    assert both.others == _keys(1, 2, 5)
    assert both.scene("KXXX:f4", STEP_S).others == _keys(3, 1, 2, 5)


def _fake_draw(flies):
    """`replay.draw_flights` over the fixture: the flights in ``flies`` fly on their own dynamics (no rebuilt series:
    a stand-in holding what `replay.batch_of` reads)."""
    from ts_transformer.autopilot import replay
    from ts_transformer.instructions.artefact import load_candidates, load_signals

    def draw(directory, split, candidates, *, per_airport, seed, groups=(replay.OWN,)):
        signals = load_signals(directory, split)
        taken = [i for i in candidates if signals[i].dataset_id in flies]
        series = SimpleNamespace(scenario=SimpleNamespace(initial=SimpleNamespace(m=60_000.0),
                                                          source={"resolved_typecode": "B738"}))
        return replay.Drawn(indices=taken, signals=[signals[i] for i in taken], series=[series] * len(taken),
                            groups=[replay.OWN] * len(taken), geometries=load_candidates(directory),
                            vertical_paths={"KXXX": ()}, description={})

    return draw


def test_the_draw_commands_the_flights_that_fly_and_passes_over_a_window_where_none_does(tmp_path, monkeypatch):
    from ts_transformer.autopilot import replay
    from ts_transformer.experiments.traffic_window import draw_windows

    directory, airports, spec = _airport(tmp_path, monkeypatch)
    # f6 and f7 do not fly: the window holding only them is passed over, the one with f3 … f7 commands f3 … f5
    monkeypatch.setattr(replay, "draw_flights", _fake_draw(set(_keys(0, 1, 2, 3, 4, 5))))
    drawn = draw_windows(directory, "train", spec, Words(spec), airports, per_airport=2, seed=5, step_s=STEP_S)
    assert sorted(c for _, _, c in drawn.openings) == [_keys(0, 1, 2, 3, 4, 5), _keys(3, 4, 5)]
    assert [s.dataset_id for s in drawn.batch.signals] == [k for _, _, c in drawn.openings for k in c]
    members = drawn.members()
    assert [len(r) for r in members] == [len(c) for _, _, c in drawn.openings] and members[1].start == members[0].stop
    counts = drawn.counts["airports"]["KXXX"]
    assert counts["windows"] == 2 and counts["commanded"] == 9 and counts["read"] == 2 + counts["passed_over"]
    assert drawn.batch.readings[0].words.shape[1] == 6
    with pytest.raises(ValueError, match="2 windows with a flight that flies, 3 wanted"):
        draw_windows(directory, "train", spec, Words(spec), airports, per_airport=3, seed=5, step_s=STEP_S)
