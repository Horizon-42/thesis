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


# ---- the loop (step 7.3)

LIMIT_S = 60.0
SHIFTED = ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps", "vertical_rate_mps")


def _physics_of(flights, geometry):
    """The executor's inputs of ``flights`` from their first predicted step, stacked (`test_autopilot._physics`)."""
    import dataclasses as dc

    import torch

    from ts_transformer.prior.scene import N_LOOK
    from ts_transformer.tests.test_autopilot import _physics

    parts = [_physics(dc.replace(f, **{name: getattr(f, name)[N_LOOK:] for name in SHIFTED}), geometry) for f in flights]
    stacked = [type(parts[0][k])(**{field.name: torch.cat([getattr(p[k], field.name) for p in parts])
                                    for field in dc.fields(parts[0][k])}) for k in range(3)]
    return (*stacked, torch.cat([p[3] for p in parts]))


def _scene_airport(tmp_path, monkeypatch):
    """`test_traffic_speaking`'s airport: f0, f2 (background) and f1 entering 0, 10 and 30 s apart on one approach, f3
    an hour later, alone."""
    from ts_transformer.tests.test_traffic_speaking import _airport as speaking_airport

    return speaking_airport(tmp_path, monkeypatch)


def _window_loop(model, airport, signals, spec, commanded, *, alone=False, seed=4, samples=1):
    import torch

    from ts_transformer.experiments.traffic_window import WindowLoop, window_of
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    windows = [window_of(airport, 0.0, keys, [LIMIT_S] * len(keys), STEP_S) for keys in commanded
               for _ in range(samples)]
    flights = [signals[k] for window in windows for k in window.commanded]
    geometry = airport.flights.geometry
    physics = _physics_of(flights, geometry)
    return WindowLoop(model, windows, flights, [geometry] * len(flights), *physics, [LIMIT_S] * len(flights),
                      Words(spec), _params(), None, generator=torch.Generator().manual_seed(seed), temperature=1.0,
                      procedure_masks=ProcedureMasks.none(), alone=alone)


def _traffic_model(spec):
    import torch

    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.tests.test_prior_speaker import _model

    torch.manual_seed(1)
    return with_traffic(_model(Words(spec)), EDGE_FEATURES).eval()


def test_one_commanded_aircraft_a_window_says_flies_and_ends_as_its_one_aircraft_scene_does(tmp_path, monkeypatch):
    import numpy as np
    import torch

    from ts_transformer.autopilot.judge import outcome_of
    from ts_transformer.experiments.prior_free_generation import steps_said
    from ts_transformer.experiments.traffic_speaking import SceneLoop, judged, scene_of, speaking_aircraft
    from ts_transformer.instructions.words import RUNWAY, UNCHANGED
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    keys = ["KXXX:f1", "KXXX:f3", "KXXX:f1"]
    window = _window_loop(model, airport, signals, spec, [(k,) for k in keys])
    geometry = airport.flights.geometry
    scenes = [scene_of(airport, k, LIMIT_S, STEP_S) for k in keys]
    scene = SceneLoop(model, [signals[k] for k in keys], [geometry] * 3, *_physics_of([signals[k] for k in keys], geometry),
                      [LIMIT_S] * 3, Words(spec), _params(), None, scenes=scenes, generator=torch.Generator().manual_seed(4),
                      temperature=1.0, procedure_masks=ProcedureMasks.none())
    assert window.pre == scene.speaker.pre == 15 and len(window.executors) == 1
    while scene.running:
        scene.step()
    while window.running:
        window.step()
    flown, said = scene.executor.flown(), scene.spoken.sentences()
    _, _, executor, _ = window.executors[0]
    window_flown = executor.flown()
    step_rows = int(round(STEP_S / flown.cycle_s))
    for j, got in enumerate(window.results()):
        grid = said[j][: steps_said(flown, j, said.shape[1], step_rows)]
        pointer = grid[:, RUNWAY][grid[:, RUNWAY] != UNCHANGED]
        ended = outcome_of(flown, j, geometry, int(pointer[-1]), spec)
        aircraft = speaking_aircraft(scenes[j], flown, j, grid, ended.outcome, ended.end_row, ended.crossing, -1, STEP_S)
        outcome, end, _ = judged(scenes[j], aircraft, STEP_S)
        assert (got.outcome, got.end) == (outcome, end)
        counted = len(grid) if end is None else min(len(grid), int(round((end["t_s"] - aircraft.first_step_s) / STEP_S)))
        assert got.counted == counted
        assert np.array_equal(got.said[:counted], grid[:counted])
        # the states flown to its judged end
        last = counted * step_rows + 1
        assert torch.equal(window_flown.states[j, :last], flown.states[j, :last])
    # f1 answers for a loss behind f0 at its first predicted step: ended there, it says nothing more and flies on
    first = window.results()[0]
    assert first.outcome == "lost_separation" and first.counted == 0 and (first.said[1:] == UNCHANGED).all()


def test_two_commanded_aircraft_are_judged_together_ended_ones_fly_on_silent_and_stay_in_the_scene(tmp_path,
                                                                                                  monkeypatch):
    from ts_transformer.instructions.words import UNCHANGED
    from ts_transformer.prior.scene import N_LOOK

    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, [("KXXX:f0", "KXXX:f1")])
    # f1 first speaks 15 steps after f0: an executor each
    assert loop.pre == 0 and loop.start.tolist() == [0, 15] and len(loop.executors) == 2
    while loop.running:
        loop.step()
    f0, f1 = loop.results()
    # both still vectored: f0 with the background f2 600 m behind it at its first predicted step, f1 with f0 — passive
    # by then, so answered for as a replayed one — at its own (design §3.4, §9 item 29)
    assert (f0.outcome, f1.outcome) == ("lost_separation", "lost_separation") and (f0.counted, f1.counted) == (0, 0)
    assert (f0.end["with"], f0.end["with_controlled"], f1.end["with"], f1.end["with_controlled"]) == \
        ("KXXX:f2", False, "KXXX:f0", False)
    # passive, both fly on silent to their own end (the time limit) and stay in the model's scene to it
    assert (f0.said[1:] == UNCHANGED).all() and (f1.said[1:] == UNCHANGED).all()
    assert (f0.own, f1.own) == ("timeout", "timeout") and len(f0.said) == len(f1.said) == LIMIT_S / STEP_S
    assert loop.speaker.rows.tolist() == [N_LOOK + 1 + 29] * 2 and loop.in_scene_to.tolist() == [30, 30]
    episode = next(e for e in loop.runs[0].episodes if e["pair"] == ["KXXX:f0", "KXXX:f1"])
    assert episode["ended"] == ["KXXX:f1"] and {"key": "KXXX:f0", "controlled": False} in episode["responsible"]


def test_two_commanded_aircraft_speak_front_first_and_the_later_one_s_masks_read_the_earlier_one(tmp_path,
                                                                                                 monkeypatch):
    from ts_transformer.instructions.words import SPEED

    # two labelled flights 40 s apart on the approach, nothing else: f0 still flies when f1 first speaks
    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0], (0, 1))
    from ts_transformer.instructions.artefact import load_signals

    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    loop = _window_loop(_traffic_model(spec), airports["KXXX"], signals, spec, [("KXXX:f0", "KXXX:f1")])
    asked, leader_speed = [], []
    masks = loop.speaker.masks_of

    def spy(column, chosen, now):
        asked.append((loop.speaker.step, column, now.copy()))
        if now[1] and column == SPEED:
            leader_speed.append(int(loop.speaker.value[0, SPEED]))
        return masks(column, chosen, now)

    loop.speaker.masks_of = spy
    for _ in range(80):
        if loop.running:
            loop.step()
    both = sorted({step for step, _, now in asked if now[1]} & {step for step, _, now in asked if now[0]})
    assert both, "the two never spoke at one step"
    for step in both:
        rounds = [now.tolist() for s, _, now in asked if s == step]
        assert rounds == [[True, False]] * 2 + [[False, True]] * 2      # f0 (in front) first, both columns each
    # f1's speed mask read f0's speed word in force, f0's word of that step included
    assert len(leader_speed) == len([s for s, c, now in asked if now[1] and c == SPEED]) and min(leader_speed) > 0


def test_alone_each_commanded_aircraft_sees_only_itself_and_is_still_judged_in_its_window(tmp_path, monkeypatch):
    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, [("KXXX:f0", "KXXX:f1")], alone=True)
    assert loop.speaker.aircraft == 1 and loop.speaker.mask_columns == ()
    while loop.running:
        loop.step()
    assert [r.outcome for r in loop.results()] == ["lost_separation", "lost_separation"]


def test_a_track_unwrapped_a_state_at_a_time_is_numpy_s_unwrap():
    import numpy as np

    from ts_transformer.experiments.traffic_window import _Unwrapped

    rng = np.random.default_rng(3)
    for trial in range(20):
        p = np.cumsum(rng.normal(0.0, 1.5, size=60))
        p = np.mod(p + np.pi, 2 * np.pi) - np.pi                 # wrapped, with jumps
        if trial == 0:
            p[5] = p[4] + np.pi                                  # an exact half turn either way
            p[9] = p[8] - np.pi
        running = _Unwrapped(p[0])
        ours = np.array([p[0]] + [running.next(x) for x in p[1:]])
        assert np.array_equal(ours, np.unwrap(p))
