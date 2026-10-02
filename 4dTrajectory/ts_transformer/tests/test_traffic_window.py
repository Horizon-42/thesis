"""The prior commanding every aircraft of a window (`experiments/traffic_window`, multi-aircraft design §6.6 step 7): the
windows over a segment, the others a window replays, and the draw of the flights it commands. On the scene-data
fixture (a tmp artefact)."""

from __future__ import annotations

import math
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

    from ts_transformer.experiments import traffic_go_around

    directory, manifest, spec, _ = _artefact(tmp_path, entries, list(labelled))
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    airports, _ = scene_airports(directory, "train", spec, ("KXXX",), None, 2_048)
    # KXXX publishes no procedure: its approach altitude (a go-around's reward reads it) 500 m over each threshold
    geometry = airports["KXXX"].flights.geometry
    monkeypatch.setitem(traffic_go_around._APPROACH_M, "KXXX", tuple(c.elevation_m + 500.0 for c in geometry.candidates))
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


def _window_loop(model, airport, signals, spec, commanded, *, alone=False, seed=4, samples=1, limits=None,
                 procedure_masks=None, given=None, go_around_extra_s=0.0, probing=None, probe_margin=1.5):
    """A `WindowLoop` of ``commanded`` (the keys of each window, ``samples`` times each; ``limits``: each window's time
    limit, `LIMIT_S` by default). No time for a go-around by default: these tests compare the window loop with the
    one-aircraft scene loop, which gives none (the untrained model says go-arounds)."""
    import torch

    from ts_transformer.experiments.traffic_window import WindowLoop, window_of
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    limits = limits or [LIMIT_S] * len(commanded)
    windows = [window_of(airport, 0.0, keys, [limit] * len(keys), STEP_S) for keys, limit in zip(commanded, limits)
               for _ in range(samples)]
    per_aircraft = [limit for keys, limit in zip(commanded, limits) for _ in range(samples) for _ in keys]
    flights = [signals[k] for window in windows for k in window.commanded]
    geometry = airport.flights.geometry
    physics = _physics_of(flights, geometry)
    return WindowLoop(model, windows, flights, [geometry] * len(flights), *physics, per_aircraft, Words(spec),
                      _params(), None, generator=torch.Generator().manual_seed(seed), temperature=1.0,
                      procedure_masks=procedure_masks or ProcedureMasks.none(), alone=alone, given=given,
                      go_around_extra_s=go_around_extra_s, probing=probing, probe_margin=probe_margin)


def _one_aircraft_reference(model, airport, signals, spec, keys, limits, seed, procedure_masks=None, alone=False):
    """The one-aircraft loop over ``keys`` (`SceneLoop`, the others replayed — none heard ``alone``), each read as
    `traffic_free_generation.scene_sentences` reads it: ``[(words to its own end, judged outcome, end, counted,
    states)]``."""
    import dataclasses as dc

    import torch

    from ts_transformer.autopilot.judge import outcome_of
    from ts_transformer.experiments.prior_free_generation import glidepath_stops, steps_said
    from ts_transformer.experiments.traffic_speaking import SceneLoop, judged, scene_of, speaking_aircraft
    from ts_transformer.instructions.words import RUNWAY, UNCHANGED
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params

    procedure_masks = procedure_masks or ProcedureMasks.none()
    geometry = airport.flights.geometry
    flights = [signals[k] for k in keys]
    scenes = [scene_of(airport, k, limit, STEP_S) for k, limit in zip(keys, limits)]
    loop = SceneLoop(model, flights, [geometry] * len(keys), *_physics_of(flights, geometry), limits, Words(spec),
                     _params(), None, scenes=[dc.replace(s, others=()) for s in scenes] if alone else scenes,
                     generator=torch.Generator().manual_seed(seed), temperature=1.0, procedure_masks=procedure_masks)
    while loop.running:
        loop.step()
    flown, said = loop.executor.flown(), loop.spoken.sentences()
    step_rows = int(round(STEP_S / flown.cycle_s))
    grids = [said[j].copy() for j in range(len(keys))]
    stops = (glidepath_stops(flown, grids, [geometry] * len(keys),
                             [procedure_masks.finals[geometry.code]] * len(keys), Words(spec))
             if procedure_masks.altitudes else None)
    out = []
    for j in range(len(keys)):
        stop = -1 if stops is None else int(stops.step[j])
        grid = grids[j]
        if stop >= 0:
            grid = grid[: min(len(grid), int(stops.row[j]) + 1)]
        else:
            grid = grid[: steps_said(flown, j, len(grid), step_rows)]
        pointer = grid[:, RUNWAY][grid[:, RUNWAY] != UNCHANGED]
        ended = outcome_of(flown, j, geometry, int(pointer[-1]), spec)
        aircraft = speaking_aircraft(scenes[j], flown, j, grid, ended.outcome, ended.end_row, ended.crossing, stop,
                                     STEP_S)
        outcome, end, _ = judged(scenes[j], aircraft, STEP_S)
        counted = len(grid) if end is None else min(len(grid), int(round((end["t_s"] - aircraft.first_step_s)
                                                                         / STEP_S)))
        out.append((grid, outcome, end, counted, flown.states[j]))
    return out


def _same_as_reference(loop, reference):
    """Every commanded aircraft of ``loop`` (one a window) against the one-aircraft reading of it."""
    import torch

    for got, (grid, outcome, end, counted, states) in zip(loop.results(), reference):
        assert (got.outcome, got.end, got.counted) == (outcome, end, counted)
        assert (got.said[:counted] == grid[:counted]).all()
        flown = loop.executors[got.group][2].flown()
        last = counted * 2 + 1
        assert torch.equal(flown.states[got.place, :last], states[:last])


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
    # f1 first speaks 15 steps after f0: one executor, f1 starting 15 steps in — two cohorts
    assert loop.pre == 0 and loop.start.tolist() == [0, 15] and len(loop.executors) == 1
    assert loop.cohort.tolist() == [0, 1] and loop.executors[0][2].start.tolist() == [0, 15 * 2]
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
    # (the time limit's state, step 30, is their own last: in the judge's scene and the model's)
    assert loop.speaker.rows.tolist() == [N_LOOK + 1 + 30] * 2 and loop.in_scene_to.tolist() == [30, 30]
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


def test_one_commanded_aircraft_a_window_over_chained_traffic_ends_as_its_scene_does_to_the_last_step(tmp_path,
                                                                                                       monkeypatch):
    """Replayed traffic in the air the whole flight, several windows with different limits in one batch; seeds 1 and
    11 lose separation at the time limit's own state (the review of 7.3: the last step was not judged)."""
    from ts_transformer.instructions.artefact import load_signals

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _traffic_model(spec)
    keys, limits = ["KXXX:f2", "KXXX:f3", "KXXX:f5", "KXXX:f3"], [60.0, 60.0, 60.0, 90.0]
    ends = 0
    for seed in (1, 11):
        loop = _window_loop(model, airport, signals, spec, [(k,) for k in keys], seed=seed, limits=limits)
        while loop.running:
            loop.step()
        reference = _one_aircraft_reference(model, airport, signals, spec, keys, limits, seed)
        _same_as_reference(loop, reference)
        ends += sum(end is not None and end["t_s"] - got.first_s == 60.0
                    for got, (_, _, end, _, _) in zip(loop.results(), reference))
    assert ends >= 2


def test_under_the_procedure_s_altitudes_a_glidepath_stop_is_the_one_aircraft_loop_s(tmp_path, monkeypatch):
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
    from ts_transformer.tests.test_prior_procedure import _final

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    geometry = airport.flights.geometry
    # a glidepath raised so that the model's descents sink under its edge (the review's probe)
    masks = ProcedureMasks((PROCEDURE_ALTITUDES,), {"KXXX": tuple(_final(candidate=c, crossing_m=1_200.0,
                                                                         faf_d_m=40_000.0)
                                                                  for c in geometry.candidates)})
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _traffic_model(spec)
    keys, limits = ["KXXX:f2", "KXXX:f3", "KXXX:f5", "KXXX:f3"], [200.0] * 4
    loop = _window_loop(model, airport, signals, spec, [(k,) for k in keys], seed=1, limits=limits,
                        procedure_masks=masks)
    while loop.running:
        loop.step()
    _same_as_reference(loop, _one_aircraft_reference(model, airport, signals, spec, keys, limits, 1, masks))
    assert "below_glidepath" in [r.own for r in loop.results()]


def test_alone_one_commanded_aircraft_a_window_is_the_one_aircraft_loop_hearing_no_other(tmp_path, monkeypatch):
    from ts_transformer.instructions.artefact import load_signals

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    model = _traffic_model(spec)
    keys, limits = ["KXXX:f2", "KXXX:f3", "KXXX:f5"], [60.0] * 3
    loop = _window_loop(model, airport, signals, spec, [(k,) for k in keys], seed=16, limits=limits, alone=True)
    while loop.running:
        loop.step()
    _same_as_reference(loop, _one_aircraft_reference(model, airport, signals, spec, keys, limits, 16, alone=True))


def test_an_executor_a_first_step_flies_each_aircraft_as_its_own_executor_would(tmp_path, monkeypatch):
    """Two commanded aircraft 40 s apart: an executor each, and each one's states are what its said words flown alone
    give (`autopilot.executor.fly`, the executor's own loop)."""
    import torch

    from ts_transformer.autopilot.executor import fly
    from ts_transformer.autopilot.sentence import Sentences, TimeClock
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.tests.test_autopilot import _params

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0], (0, 1))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, [("KXXX:f0", "KXXX:f1")])
    while loop.running:
        loop.step()
    assert len(loop.executors) == 1 and len(loop.cohorts) == 2         # one executor; each from its own start
    params, geometry = _params(), airport.flights.geometry
    for got, key in zip(loop.results(), ("KXXX:f0", "KXXX:f1")):
        inputs, runways, charts, approach = _physics_of([signals[key]], geometry)
        spoken = loop.executors[got.group][3].sentences()[got.place]
        alone = fly(inputs, Sentences([spoken], Words(spec), device=torch.device("cpu")), TimeClock(params.cycle_s),
                    runways, charts, approach, params, Words(spec),
                    time_limit_s=torch.tensor([LIMIT_S], dtype=torch.float64))
        steps = len(got.said) * 2 + 1
        assert torch.equal(loop.executors[got.group][2].flown().states[got.place, :steps], alone.states[0, :steps])


def test_a_window_is_judged_over_its_own_span_whatever_else_the_batch_holds(tmp_path, monkeypatch):
    """f1's window (f1 ended at its first step, flying on silent to its limit) alone and beside a longer window."""
    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    counts = []
    for commanded, limits in (([("KXXX:f1",)], [LIMIT_S]), ([("KXXX:f1",), ("KXXX:f3",)], [LIMIT_S, 3 * LIMIT_S])):
        loop = _window_loop(model, airport, signals, spec, commanded, limits=limits)
        while loop.running:
            loop.step()
        loop.results()
        counts.append((loop.runs[0].steps_judged, loop.runs[0].scene_seconds))
    assert counts[0] == counts[1] == (31, LIMIT_S)


def test_an_aircraft_at_its_first_predicted_step_is_read_at_its_record_before_its_round(tmp_path, monkeypatch):
    """Two commanded aircraft 40 s apart: when f1 first speaks, f0 (in front) speaks first — its masks read f1 at its
    record (f1 has no runway yet); f1's read f0 at its executor state."""
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.scene import N_LOOK

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0], (0, 1))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, [("KXXX:f0", "KXXX:f1")])
    read = []
    others_at = loop._others_at
    loop._others_at = lambda i, step: (read.append((i, step, len(others_at(i, step).speeds))), others_at(i, step))[1]
    while loop.running:
        loop.step()
    first = int(loop.start[1]) + N_LOOK                         # f1's first predicted step
    at_first = sorted({(i, n) for i, step, n in read if step == first})
    assert at_first == [(0, 1), (1, 1)]


def test_a_crossing_of_any_threshold_plane_in_a_step_is_asked_of_the_judge(tmp_path, monkeypatch):
    """The steps an own end the executor flies past may lie in (the review of 7.3): a threshold plane crossed."""
    import dataclasses as dc

    import numpy as np

    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, [("KXXX:f3",)])
    loop.step()
    loop.step()
    threshold = airport.flights.geometry.candidates[0]
    before, after = loop.states[0][0], loop.states[0][1]
    # runway 09 lands east from its threshold: west of it is before it
    loop.states[0][0] = dc.replace(before, e_m=threshold.threshold_e_m - 50.0, n_m=threshold.threshold_n_m)
    loop.states[0][1] = dc.replace(after, e_m=threshold.threshold_e_m + 50.0, n_m=threshold.threshold_n_m)
    assert loop._crossed(np.array([0]), np.array([0])).tolist() == [True]
    loop.states[0][1] = dc.replace(after, e_m=threshold.threshold_e_m - 10.0, n_m=threshold.threshold_n_m)
    assert loop._crossed(np.array([0]), np.array([0])).tolist() == [False]


# ---- the M3 second-pass runner (step 7.4)

def _patch_runner_physics(monkeypatch, signals, geometry, keys):
    """The runner's executor inputs from the fixture's flights (a replay batch's series are the keys here)."""
    import dataclasses as dc

    import torch

    from ts_transformer.experiments import prior_free_generation
    from ts_transformer.experiments import traffic_window_generation as runner

    physics = {k: _physics_of([signals[k]], geometry) for k in keys}

    def stacked(series, k):
        tables = [physics[key][k] for key in series]
        if k == 3:
            return torch.cat(tables)
        return type(tables[0])(**{f.name: torch.cat([getattr(t, f.name) for t in tables]) for f in dc.fields(tables[0])})

    for module in (runner, prior_free_generation):
        monkeypatch.setattr(module, "flight_inputs", lambda series, device, anchor: stacked(list(series), 0))
        monkeypatch.setattr(module, "_physics", lambda part, device: (stacked(part.series, 1), stacked(part.series, 2),
                                                                      stacked(part.series, 3)))


def _as_drawn(windows, members, batch, limits):
    """The runner's windows as drawn (none augmented)."""
    from ts_transformer.experiments import traffic_window_generation as runner

    count = len(batch.signals)
    return runner.Drawn(windows, [None] * len(windows), members, batch, limits, [None] * count, [None] * count)


def test_the_window_runner_reads_every_commanded_aircraft_four_ways_judged_in_its_window(tmp_path, monkeypatch):
    import numpy as np
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.prior.scene import Landings
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 3_600.0], (0, 1, 2))
    airport = airports["KXXX"]
    geometry = airport.flights.geometry
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    commanded = [("KXXX:f0", "KXXX:f1"), ("KXXX:f2",)]
    keys = [k for keys in commanded for k in keys]
    batch = _batch(airport, signals, spec, keys)
    windows = [window_of(airport, 0.0, c, [LIMIT_S] * len(c), STEP_S) for c in commanded]
    drawn = _as_drawn(windows, [range(0, 2), range(2, 3)], batch, [LIMIT_S] * 3)
    _patch_runner_physics(monkeypatch, signals, geometry, keys)
    times = np.sort([f.presence.landing_s for f in airport.flights.flights.values()])
    every = {"KXXX": Landings(times, {c.ident: times if c.ident == "09" else np.zeros(0) for c in geometry.candidates})}
    model = _traffic_model(spec)
    rows = []
    for source in ("scene", "alone"):
        rows += runner.model_rows(model, drawn, [0, 1], source, Words(spec), _params(), None, every, 2,
                                  generator=torch.Generator().manual_seed(4), temperature=1.0,
                                  procedure_masks=ProcedureMasks.none())
    for source in ("labelled", "recorded"):
        rows += runner.fixed_rows(drawn, [0, 1], source, Words(spec), _params(), every, ProcedureMasks.none())
    by_source = {s: [r for r in rows if r["source"] == s] for s in runner.SOURCES}
    # a row per commanded aircraft and sample (the model's), per commanded aircraft (the fixed paths)
    assert [(r["dataset_id"], r["sample"]) for r in by_source["scene"]] == \
        [("KXXX:f0", 0), ("KXXX:f1", 0), ("KXXX:f0", 1), ("KXXX:f1", 1), ("KXXX:f2", 0), ("KXXX:f2", 1)]
    assert [r["dataset_id"] for r in by_source["recorded"]] == keys and [r["commanded"] for r in by_source["labelled"]] == [2, 2, 1]
    for r in rows:
        # a sentence with a go-around (the untrained model says some) is scored on it: at most 0.9
        assert (r["reward"] in (0.0, 1.0) if r["go_around"] is None else 0.0 <= r["reward"] <= 0.9 + 1e-12)
        assert r["outcome"] and r["ifr_outcome"] and r["flown_s"] >= 0.0
        assert r["with_commanded"] + r["with_replayed"] == r["episodes"]
    assert all(r["mask_steps"] == {"approach": 0, "speed": 0} for r in by_source["alone"])
    # the record lands where it landed; its landing is the one the others' landing times are read against
    assert all(r["outcome"] in ("landed", "lost_separation") for r in by_source["recorded"])
    readout = runner.summaries(rows, augmented=False)
    assert set(readout["pooled"]) == set(runner.SOURCES) and set(readout["window_sizes"]) == {"1", "2"}
    assert readout["pooled"]["scene"]["reward_together"]["pairs"] in (0, 1)
    assert readout["window_sizes"]["1"]["scene"]["reward_together"] == {"pairs": 0, "correlation": None}


def test_rewards_that_go_together_correlate():
    from ts_transformer.experiments.traffic_window_generation import together

    rows = [{"window": 0, "dataset_id": key, "sample": s, "reward": reward}
            for key, rewards in (("a", [1, 0, 1, 0]), ("b", [1, 0, 1, 0]), ("c", [0, 1, 0, 1]))
            for s, reward in enumerate(rewards)]
    got = together(rows)
    # a–b +1, a–c −1, b–c −1 on equal variances: (1 − 1 − 1) / 3
    assert got["pairs"] == 3 and abs(got["correlation"] + 1 / 3) < 1e-12


# ---- the re-review of 7.3 (landing context, landings, own ends, several aircraft a window)

def _pool(airport):
    """Every flight of the fixture's airport landing where it landed: its landing context."""
    import numpy as np

    from ts_transformer.prior.scene import Landings

    presences = [f.presence for f in airport.flights.flights.values()]
    return Landings(np.sort(np.array([p.landing_s for p in presences])),
                    {c.ident: np.sort(np.array([p.landing_s for p in presences if p.runway == c.ident]))
                     for c in airport.flights.geometry.candidates})


def test_a_loop_landing_reaches_a_later_aircraft_s_rows_not_encoded_yet(tmp_path, monkeypatch):
    """The landing context (variant "full"): each commanded aircraft's lacks the other commanded ones' recorded
    landings, and a landing in the loop reaches the rows another reads later — its observed rows too."""
    import torch

    from ts_transformer.experiments.traffic_window import WindowLoop, window_of
    from ts_transformer.inference.scene_edges import EDGE_FEATURES
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.data import rows_inputs
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.prior.model import with_traffic
    from ts_transformer.prior.scene import N_LOOK
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_prior_speaker import _model

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    geometry = airport.flights.geometry
    pool = _pool(airport)
    torch.manual_seed(1)
    model = with_traffic(_model(Words(spec), variant="full"), EDGE_FEATURES).eval()
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    keys = ("KXXX:f0", "KXXX:f1", "KXXX:f2")                 # 0, 200, 400 s apart
    window = window_of(airport, 0.0, keys, [120.0] * 3, STEP_S)
    flights = [signals[k] for k in keys]
    loop = WindowLoop(model, [window], flights, [geometry] * 3, *_physics_of(flights, geometry), [120.0] * 3,
                      Words(spec), _params(), {"KXXX": pool}, generator=torch.Generator().manual_seed(3),
                      temperature=1.0, procedure_masks=ProcedureMasks.none())
    speaker = loop.speaker
    # each context: the pool less the other two's recorded landings and its own
    assert [len(c.times_s) for c in speaker.contexts] == [len(pool.times_s) - 3] * 3
    loop.step()
    # f0 lands in the loop 100 s after its first predicted step, long before f2 enters
    loop.landing_s[0] = loop.window_time_s(0, int(loop.start[0]) + N_LOOK) + 100.0
    loop.speaker.value[0, 0] = 1                              # (its runway in force: the fixture's only one)
    before = speaker.relative[2, : N_LOOK + 1].clone()
    loop._landed(0, 0)
    assert float(loop.landing_s[0]) in speaker.contexts[2].times_s
    f2 = flights[2]
    _, relative = rows_inputs(f2.e_m[None, : N_LOOK + 1], f2.n_m[None, : N_LOOK + 1], f2.altitude_m[None, : N_LOOK + 1],
                              speaker.time_s[: N_LOOK + 1], speaker.entry_s[2:3], 0, geometry, [speaker.contexts[2]])
    assert not torch.equal(before, speaker.relative[2, : N_LOOK + 1])
    assert torch.equal(speaker.relative[2, : N_LOOK + 1, : relative.shape[2]],
                       torch.as_tensor(relative[0], dtype=torch.float32))


def test_a_follower_ended_at_one_landing_is_passive_at_the_next_and_a_passive_leader_is_marked():
    """Paths judged afterwards keeping the ended ones (`Loop(keep_ended=True)`, the window's fixed paths): L1 and L2
    land 0.5 s apart in one step with I 3.5 NM behind both (TBL 5-5-2: 4 NM) — I ended at the first, passive at the
    second; and a passive leader recorded as not controlled."""
    import numpy as np
    from geokit import NM_M

    from ts_transformer.experiments.traffic_loop import Judging, Loop, Run
    from ts_transformer.tests.test_traffic_loop import SEPARATION, _controlled

    gap = 3.5 * NM_M
    t = np.arange(0.0, 100.0 + 1e-9, STEP_S)
    first = _controlled("L1", 0.0, -70.0 * (100.5 - t), landing_s=100.5)
    second = _controlled("L2", 0.0, -70.0 * (101.0 - t), landing_s=101.0, n=1.0)
    t_follower = np.arange(0.0, 160.0 + 1e-9, STEP_S)
    follower = _controlled("I", 0.0, -gap - 70.0 * (101.0 - t_follower), category="I", outcome="timeout")
    run = Loop(SEPARATION, "visual", STEP_S, keep_ended=True).run([first, second, follower], [])
    assert run.ended["I"]["t_s"] == 100.5 and run.ended["I"]["with"] == "L1"
    assert [(a["leader"], a["follower_controlled"]) for a in run.at_threshold] == [("L1", True), ("L2", False)]
    judging = Judging(SEPARATION, "visual", STEP_S, Run("visual"))
    judging.landing(100.5, first, [follower], [], leader_passive=True)
    out = judging.out
    assert out.at_threshold[0]["leader_controlled"] is False and out.ended["I"]["with_controlled"] is False


def test_the_steps_counted_run_to_the_judge_s_end_time_and_an_end_the_executor_flies_past_is_found(tmp_path,
                                                                                                    monkeypatch):
    import dataclasses as dc

    from ts_transformer.experiments import traffic_window

    airport, signals, spec = _scene_airport(tmp_path, monkeypatch)
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, [("KXXX:f3",)])
    for _ in range(4):
        loop.step()
    # nothing ended it: asked, it flies on
    loop._own_ends([0], loop.speaker.step - 1)
    assert not loop.left[0]
    # an uncaptured crossing found in the step just flown: it leaves at the state before it, silent, not appended
    real = traffic_window.outcome_of
    monkeypatch.setattr(traffic_window, "outcome_of", lambda flown, p, geometry, runway, spec: dc.replace(
        real(flown, p, geometry, runway, spec), outcome="crossed_without_capture", end_row=7))
    loop._own_ends([0], loop.speaker.step - 1)
    monkeypatch.setattr(traffic_window, "outcome_of", real)
    assert loop.left[0] and loop.own[0] == "crossed_without_capture" and loop.in_scene_to[0] == 3
    rows = int(loop.speaker.rows[0])
    while loop.running:
        loop.step()
    assert int(loop.speaker.rows[0]) == rows and loop.results()[0].outcome == "crossed_without_capture"
    # the steps counted: the judge's end time on the steps from the first state, as `judged_steps` rounds it
    first_s = loop.states[0][0].t_s
    loop.runs[0].ended["KXXX:f3"] = {"t_s": first_s + 3.4, "kind": "at_threshold", "relation": "same", "with": "x",
                                     "with_controlled": False}
    loop.ended_at[0] = 2
    assert loop.results()[0].counted == 2


def test_several_commanded_aircraft_a_window_keep_the_loop_s_books(tmp_path, monkeypatch):
    """The review's stress probe: windows of one to three commanded aircraft with mixed limits under a raised
    glidepath — the loop ends; every aircraft leaves with an own end, every state it is in the scene at is judged on a
    runway, the speaker holds its rows through its last, it is silent after the judge ends it and every landing is
    checked."""
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.instructions.words import UNCHANGED
    from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
    from ts_transformer.prior.scene import N_LOOK
    from ts_transformer.tests.test_prior_procedure import _final

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 80.0, 200.0, 230.0, 400.0, 600.0, 610.0],
                                 tuple(range(8)))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    masks = ProcedureMasks((PROCEDURE_ALTITUDES,), {"KXXX": tuple(_final(candidate=c, crossing_m=1_200.0,
                                                                         faf_d_m=40_000.0)
                                                                  for c in airport.flights.geometry.candidates)})
    groups = [("KXXX:f0", "KXXX:f1", "KXXX:f2"), ("KXXX:f3", "KXXX:f4"), ("KXXX:f5",), ("KXXX:f6", "KXXX:f7")]
    loop = _window_loop(_traffic_model(spec), airport, signals, spec, groups, seed=0,
                        limits=[100.0, 160.0, 60.0, 130.0], procedure_masks=masks)
    steps = 0
    while loop.running:
        loop.step()
        steps += 1
        assert steps < 2_000
    for i, got in enumerate(loop.results()):
        last = min(int(loop.in_scene_to[i]), len(loop.states[i]) - 1)
        assert loop.left[i] and loop.own[i] is not None and int(loop.judged_state[i]) == last
        assert all(s.runway is not None for s in loop.states[i][: last + 1])
        assert got.counted <= len(got.said) and int(loop.speaker.rows[i]) >= N_LOOK + 1 + last
        assert got.end is None or (got.said[int(loop.ended_at[i]) + 1:] == UNCHANGED).all()
        assert got.landing_s is None or loop.landing_checked[i]
    assert all(run.scene_seconds > 0.0 for run in loop.runs)


def test_a_batch_s_budget_is_what_the_speaker_holds(tmp_path, monkeypatch):
    """One window with a pre-roll (f3: f1, f2 in the air before it), one with a late commanded aircraft (f0 and f4,
    800 s apart): the batch's steps are the longest pre-roll, the latest entry and the rows together (the review of
    7.4: the budget took each window's own sum)."""
    from ts_transformer.experiments.traffic_window_generation import batch_cost, window_size
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.generate import rows_for
    from ts_transformer.prior.scene import N_LOOK

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    commanded, samples = [("KXXX:f3",), ("KXXX:f0", "KXXX:f4")], 2
    from ts_transformer.experiments.traffic_go_around import GO_AROUND_EXTRA_S

    loop = _window_loop(_traffic_model(spec), airport, signals, spec, commanded, samples=samples,
                        go_around_extra_s=GO_AROUND_EXTRA_S)
    sizes = [window_size(window, [LIMIT_S] * len(window.commanded), STEP_S) for window in loop.windows[::samples]]
    rows = rows_for(LIMIT_S + GO_AROUND_EXTRA_S + STEP_S, STEP_S)     # a go-around's time laid out for every aircraft
    held = len(loop.windows) * loop.speaker.aircraft * (int(loop.speaker.start.max()) + rows)
    assert batch_cost(sizes, samples) == held and loop.speaker.step == int(loop.speaker.start.min()) + N_LOOK
    assert sizes[0][1] > 0 and sizes[1][2] == 400 and sum(sizes[0][1:]) < held / len(loop.windows)


def test_a_glidepath_stop_is_read_to_its_judged_end_through_the_runner(tmp_path, monkeypatch):
    """Under a raised glidepath (seed 1: f3 stops at 122 s of its 200): the runner reads the model's aircraft to the
    stop, passive after it, and the labelled paths the same way."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_prior_procedure import _final
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch)
    airport = airports["KXXX"]
    geometry = airport.flights.geometry
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    masks = ProcedureMasks((PROCEDURE_ALTITUDES,), {"KXXX": tuple(_final(candidate=c, crossing_m=1_200.0,
                                                                         faf_d_m=40_000.0)
                                                                  for c in geometry.candidates)})
    keys, limit = ["KXXX:f2", "KXXX:f3", "KXXX:f5", "KXXX:f3"], 200.0
    windows = [window_of(airport, 0.0, (k,), [limit], STEP_S) for k in keys]
    drawn = _as_drawn(windows, [range(j, j + 1) for j in range(4)], _batch(airport, signals, spec, keys), [limit] * 4)
    _patch_runner_physics(monkeypatch, signals, geometry, keys)
    every = {"KXXX": _pool(airport)}
    rows = runner.model_rows(_traffic_model(spec), drawn, [0, 1, 2, 3], "scene", Words(spec), _params(), None, every, 1,
                             generator=torch.Generator().manual_seed(1), temperature=1.0, procedure_masks=masks)
    stopped = [r for r in rows if r["own"] == "below_glidepath"]
    assert stopped and all(r["outcome"] == "below_glidepath" and r["flown_s"] < limit for r in stopped)
    labelled = runner.fixed_rows(drawn, [0, 1, 2, 3], "labelled", Words(spec), _params(), every, masks)
    for r in labelled:
        assert r["flown_s"] <= limit and (r["own"] != "below_glidepath" or r["landing_s"] is None)


def test_the_order_of_a_window_s_landings_and_a_flight_in_two_windows():
    from ts_transformer.experiments.traffic_window_generation import summary

    def row(source, window, key, landing, outcome="landed", sample=None):
        return {"source": source, "window": window, "dataset_id": key, "sample": sample, "landing_s": landing,
                "outcome": outcome, "starts_in_a_loss": False, "flown_s": 100.0, "episodes": 0, "relations": {},
                "with_commanded": 0, "with_replayed": 0, "ended_with": None, "ifr_outcome": outcome, "reward": 1.0,
                "counted": 10, "mask_steps": {"approach": 0, "speed": 0}, "masked_mass": {"approach": 0.0, "speed": 0.0},
                "landed_here": outcome == "landed", "go_around": None}

    # a and b land in the record a first; the labelled words swap them in window 0; c is in two windows
    # the record of c in window 1 is judged a loss (its landing still the one to read against)
    rows = [row("recorded", 0, "a", 100.0), row("recorded", 0, "b", 200.0), row("recorded", 1, "b", 200.0),
            row("recorded", 1, "c", 300.0, outcome="lost_separation"),
            row("labelled", 0, "a", 250.0), row("labelled", 0, "b", 210.0), row("labelled", 1, "b", 190.0),
            row("labelled", 1, "c", 320.0), row("alone", 1, "b", 180.0, sample=0),
            row("alone", 1, "c", 170.0, outcome="lost_separation", sample=0)]
    got = summary(rows)
    assert got["recorded"]["order"] == {"pairs": 2, "swapped": 0}
    # window 0 swapped, window 1 kept; alone, c was ended by the judge: not a landing read
    assert got["labelled"]["order"] == {"pairs": 2, "swapped": 1} and got["alone"]["order"] == {"pairs": 0, "swapped": 0}
    assert got["labelled"]["landing_vs_recorded_s"]["n"] == 4 and got["alone"]["landing_vs_recorded_s"]["n"] == 1


def test_a_loop_landing_enters_the_reward_s_landing_context(monkeypatch):
    import numpy as np

    from ts_transformer.experiments import traffic_window_generation as runner

    base = runner.Landings(np.array([10.0, 20.0]), {"09": np.array([10.0, 20.0]), "27": np.zeros(0)})
    monkeypatch.setattr(runner, "window_landings", lambda window, key, landings, step_s: base)
    got = runner._loop_context(None, "x", {"KXXX": base}, [(15.0, "27")], 2.0)
    assert got.times_s.tolist() == [10.0, 15.0, 20.0] and got.by_runway["27"].tolist() == [15.0]


def _square(number):
    if number == 3:
        raise RuntimeError("three")
    return number * number


def test_reading_processes_read_every_number_and_a_failure_ends_the_run():
    from ts_transformer.experiments.traffic_window_generation import in_processes

    got = sorted((n, v) for n, v, _ in in_processes(2, [0, 1, 2], _square))
    assert got == [(0, 0), (1, 1), (2, 4)]
    with pytest.raises(SystemExit, match="three"):
        list(in_processes(2, [0, 1, 2, 3], _square))


def test_a_batch_reads_the_same_whatever_else_is_read_and_in_whichever_process(tmp_path, monkeypatch):
    """Two loop batches: batch 1 alone, after batch 0 and in the second of two processes reads the same rows."""
    import torch

    from ts_transformer.experiments import traffic_window_generation as runner
    from ts_transformer.experiments.traffic_window import window_of
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import ProcedureMasks
    from ts_transformer.tests.test_autopilot import _params
    from ts_transformer.tests.test_traffic_speaking import _batch

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 3_600.0], (0, 1, 2))
    airport = airports["KXXX"]
    geometry = airport.flights.geometry
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    commanded = [("KXXX:f0", "KXXX:f1"), ("KXXX:f2",)]
    keys = [k for keys in commanded for k in keys]
    windows = [window_of(airport, 0.0, c, [LIMIT_S] * len(c), STEP_S) for c in commanded]
    drawn = _as_drawn(windows, [range(0, 2), range(2, 3)], _batch(airport, signals, spec, keys), [LIMIT_S] * 3)
    _patch_runner_physics(monkeypatch, signals, geometry, keys)
    model, every = _traffic_model(spec), {"KXXX": _pool(airport)}
    cpu = torch.device("cpu")

    def read(number):
        return runner.batch_rows(model, drawn, number, [number], Words(spec), _params(), None, every, 2, seed=5,
                                 temperature=1.0, procedure_masks=ProcedureMasks.none(), device=cpu)

    alone = read(1)
    after = (read(0), read(1))[1]
    forked = {n: rows for n, rows, _ in runner.in_processes(2, [0, 1], read)}
    assert alone == after == forked[1] and all(r["batch"] == 1 for r in alone)
    # a read of the model's scene source only: its rows are the full read's scene rows (each source its own streams)
    scene_only = runner.batch_rows(model, drawn, 1, [1], Words(spec), _params(), None, every, 2, seed=5, temperature=1.0,
                                   procedure_masks=ProcedureMasks.none(), device=cpu, model_sources=("scene",))
    assert scene_only == [r for r in alone if r["source"] != "alone"]
    # and the other way: the alone source alone, read first in a process of its own
    alone_only = {n: rows for n, rows, _ in runner.in_processes(1, [1], lambda number: runner.batch_rows(
        model, drawn, number, [number], Words(spec), _params(), None, every, 2, seed=5, temperature=1.0,
        procedure_masks=ProcedureMasks.none(), device=cpu, model_sources=("alone",)))}[1]
    assert alone_only == [r for r in alone if r["source"] != "scene"]


def _busy_windows(tmp_path, monkeypatch):
    """`test_several_commanded_aircraft_a_window_keep_the_loop_s_books`' windows: one to three commanded aircraft a
    window, mixed limits, a raised glidepath — ``(airport, signals, spec, groups, limits, masks)``."""
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.prior.masks import PROCEDURE_ALTITUDES, ProcedureMasks
    from ts_transformer.tests.test_prior_procedure import _final

    _, airports, spec = _airport(tmp_path, monkeypatch, [0.0, 40.0, 80.0, 200.0, 230.0, 400.0, 600.0, 610.0],
                                 tuple(range(8)))
    airport = airports["KXXX"]
    signals = {s.dataset_id: s for s in load_signals(tmp_path / "artefact", "train")}
    masks = ProcedureMasks((PROCEDURE_ALTITUDES,), {"KXXX": tuple(_final(candidate=c, crossing_m=1_200.0,
                                                                         faf_d_m=40_000.0)
                                                                  for c in airport.flights.geometry.candidates)})
    groups = [("KXXX:f0", "KXXX:f1", "KXXX:f2"), ("KXXX:f3", "KXXX:f4"), ("KXXX:f5",), ("KXXX:f6", "KXXX:f7")]
    return airport, signals, spec, groups, [100.0, 160.0, 60.0, 130.0], masks


def _flown(loop):
    while loop.running:
        loop.step()
    return loop


def _same_loops(a, b, *, to_step=None):
    """Two loops over the same windows: every aircraft's words, outcome, ends and states the same — to its own step
    ``to_step[i]`` only (words before it, states to its start), when given — and every window's judge's books."""
    import torch

    rows = int(round(STEP_S / a.params.cycle_s))                 # executor rows a step
    for i, (x, y) in enumerate(zip(a.results(), b.results())):
        flown_x = a.executors[x.group][2].flown().states[x.place]
        flown_y = b.executors[y.group][2].flown().states[y.place]
        if to_step is None:
            assert (x.outcome, x.own, x.end, x.counted, x.landing_s, x.runway) == \
                   (y.outcome, y.own, y.end, y.counted, y.landing_s, y.runway)
            assert x.said.shape == y.said.shape and (x.said == y.said).all()
            assert torch.equal(flown_x, flown_y)
        else:
            k = int(to_step[i])
            assert (x.said[:k] == y.said[:k]).all()
            assert torch.equal(flown_x[: rows * k + 1], flown_y[: rows * k + 1])
    if to_step is None:
        assert [r.ended for r in a.runs] == [r.ended for r in b.runs]
        assert [r.episodes for r in a.runs] == [r.episodes for r in b.runs]


def test_every_aircraft_given_the_words_it_said_flies_and_is_judged_as_it_was_whatever_the_draws(tmp_path,
                                                                                                    monkeypatch):
    """Design §6.6 step 7.7 item 9: words given in full replay a loop bit for bit, under another stream, never sampled
    nor masked."""
    from ts_transformer.experiments.traffic_window import Given

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    spoken = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks))
    assert any(r.end is not None for r in spoken.results()), "the probe should hold a loss of separation"
    given = [Given(r.said, len(r.said)) for r in spoken.results()]
    again = _window_loop(model, airport, signals, spec, groups, seed=99, limits=limits, procedure_masks=masks,
                         given=given)
    asked = []
    masks_of = again.speaker.masks_of

    def spy(column, chosen, now):
        asked.append(now.copy())
        return masks_of(column, chosen, now)

    again.speaker.masks_of = spy
    _same_loops(spoken, _flown(again))
    assert not any(now.any() for now in asked)


def test_a_given_aircraft_leaves_the_others_draws_as_they_were(tmp_path, monkeypatch):
    """The stream: an aircraft given the words it said changes no other aircraft's draw — under the same seed the
    loop is the one where it spoke."""
    from ts_transformer.experiments.traffic_window import Given

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    spoken = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks))
    results = spoken.results()
    given = [None] * len(results)
    given[1] = Given(results[1].said, len(results[1].said))
    _same_loops(spoken, _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits,
                                            procedure_masks=masks, given=given)))


def test_words_given_to_a_step_then_spoken_keep_everything_before_it(tmp_path, monkeypatch):
    """Rewinding one aircraft (design §6.6 step 7.7 items 7–8): it is given its words to own step ``s`` and speaks from
    there, the others are given theirs in full — everything before ``s`` is as it was; past their words' end the prior
    speaks for them."""
    from ts_transformer.experiments.traffic_window import Given

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    spoken = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks))
    results = spoken.results()
    changer, s = 0, 5
    given = [Given(r.said, s if i == changer else len(r.said)) for i, r in enumerate(results)]
    again = _window_loop(model, airport, signals, spec, groups, seed=7, limits=limits, procedure_masks=masks,
                         given=given)
    sampled = []
    masks_of = again.speaker.masks_of

    def spy(column, chosen, now):
        if now[changer]:
            sampled.append(again._step_of(changer, again.speaker.step))
        return masks_of(column, chosen, now)

    again.speaker.masks_of = spy
    _flown(again)
    # the changer is sampled (masked) from its own step s on, never before
    assert sampled and min(sampled) == s
    # before the changer's own step s every aircraft of its window is where it was (each at its own step then)
    w = results[changer].window
    at = int(spoken.start[changer] + s)
    to_step = [max(0, min(len(r.said), at - int(spoken.start[i]))) if r.window == w else len(r.said)
               for i, r in enumerate(results)]
    _same_loops(spoken, again, to_step=to_step)


def test_a_go_around_gives_its_sentence_more_time_and_its_margins_are_kept(tmp_path, monkeypatch):
    """Multi-aircraft design §6.6 step 8 item 9: the first go-around word extends the aircraft's time limit by the
    go-around's time (its executor's and its cohort's), once; the loop keeps each aircraft's tightest margin a step."""
    import numpy as np

    from ts_transformer.experiments.traffic_go_around import GO_AROUND_EXTRA_S
    from ts_transformer.experiments.traffic_window import Given
    from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, HEADING

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    spoken = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks))
    words = spoken.results()[0].said.copy()
    words[3, APPROACH], words[5, APPROACH] = APPROACH_CLEARED, APPROACH_GO_AROUND
    words[6, HEADING] = Words(spec).heading_index(0.0)
    words[7, APPROACH] = APPROACH_GO_AROUND                     # said again: still the first go-around's time only
    given = [Given(words, 8)] + [None] * (len(spoken.results()) - 1)
    loop = _window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks,
                        given=given, go_around_extra_s=GO_AROUND_EXTRA_S)
    base = float(loop.limit_s[0])
    executor = loop.executors[0][2]
    while loop.speaker.step <= int(loop.first_spoken[0]) + 6:
        loop.step()
    assert loop.go_around_step[0] == 5 and loop.extra_s[0] == GO_AROUND_EXTRA_S
    assert loop.limit_s[0] == base + GO_AROUND_EXTRA_S
    assert float(executor.time_limit_s[loop.place[0]]) == base + GO_AROUND_EXTRA_S
    per_aircraft = np.array([limit for keys, limit in zip(groups, limits) for _ in keys])
    others = np.flatnonzero(loop.go_around_step < 0)
    assert (loop.limit_s[others] == per_aircraft[others]).all() and base == per_aircraft[0]
    cohort = loop.cohort[0]
    assert loop.cohort_cycles[cohort] >= int(np.ceil((base + GO_AROUND_EXTRA_S) / loop.params.cycle_s))
    _flown(loop)
    got = loop.results()[0]
    assert got.go_around == 5 and loop.limit_s[0] == base + GO_AROUND_EXTRA_S
    # its flight is not cut at the old limit: a time limit's end comes at the new one
    assert got.own != "timeout" or (len(loop.states[0]) - 1) * STEP_S > base
    assert np.isfinite(loop.margin).any() and (loop.margin >= 0.0).all()


def test_a_probe_watches_an_established_aircraft_speaking_its_own_words_until_its_margin_is_under_the_trigger(
        tmp_path, monkeypatch):
    """Multi-aircraft design §6.6 step 8 item 10: a probe says a go-around for an aircraft it watches — speaking its own
    words, cleared, established on its final (its executor's capture), no go-around said yet — at the first step its
    tightest margin at its step before was under the trigger (an infinite one: at once); never for one not probed."""
    import numpy as np

    from ts_transformer.experiments.traffic_window import Given
    from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, APPROACH_NOT_CLEARED

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    count = sum(len(g) for g in groups)
    loop = _window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks,
                        probing=np.arange(count) < count - 1, probe_margin=1.5)
    while loop.speaker.step < int(loop.first_spoken.max()) + 3:
        loop.step()
    k = loop.speaker.step - loop.start - 9                      # each aircraft's own step about to be said (N_LOOK 8)
    speaking = k >= 1
    for i in range(count):                                      # every one cleared, established, its margin 1.2 before
        loop.speaker.value[i, APPROACH] = APPROACH_CLEARED + 1
        loop.states[i][-1].captured = True
        loop.margin[i, max(int(k[i]) - 1, 0)] = 1.2
    loop.go_around_step[:] = -1
    expected = np.where(speaking & (np.arange(count) < count - 1), APPROACH_GO_AROUND + 1, -1)
    assert (loop._probes(k, speaking, None)[:, APPROACH] == expected).all()
    # each condition alone stops it
    loop.margin[0, int(k[0]) - 1] = 1.6
    loop.states[1][-1].captured = False
    loop.speaker.value[2, APPROACH] = APPROACH_NOT_CLEARED + 1
    loop.go_around_step[3] = 0
    given = np.full((count, 6), -1)
    given[4] = 0                                                # its line given this step: its own words are not
    got = loop._probes(k, speaking, given)[:, APPROACH]
    assert (got[:5] == -1).all() and (got[5:] == expected[5:]).all()
    # an infinite trigger: whatever the margin
    loop.probe_margin = math.inf
    assert loop._probes(k, speaking, None)[0, APPROACH] == APPROACH_GO_AROUND + 1


def test_probes_change_nothing_for_the_aircraft_they_do_not_fire_for(tmp_path, monkeypatch):
    """With no probe's go-around said, a probed loop is the unprobed one."""
    import numpy as np

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    count = sum(len(g) for g in groups)
    plain = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks))
    probed = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks,
                                 probing=np.ones(count, dtype=bool), probe_margin=1e-9))
    assert all(r.forced is None for r in probed.results())
    _same_loops(plain, probed)


def test_the_margins_kept_are_the_judges_at_each_aircrafts_own_step(tmp_path, monkeypatch):
    """Where the judge found a loss of separation an aircraft was in, its margin at that own step is under 1."""
    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    loop = _flown(_window_loop(_traffic_model(spec), airport, signals, spec, groups, seed=0, limits=limits,
                               procedure_masks=masks))
    found = 0
    for i, got in enumerate(loop.results()):
        for episode in loop.runs[got.window].episodes:
            if got.key in episode["pair"]:
                own = int(round((episode["first_s"] - got.first_s) / STEP_S))
                assert loop.margin[i, own] < 1.0
                found += 1
    assert found


def test_a_go_arounds_time_changes_nothing_for_flights_without_one(tmp_path, monkeypatch):
    """The production loop lays a go-around's time out for every aircraft: every aircraft given words with no go-around
    flies, says and is judged as it does with none laid out."""
    import torch

    from ts_transformer.experiments.traffic_go_around import GO_AROUND_EXTRA_S
    from ts_transformer.experiments.traffic_window import Given
    from ts_transformer.instructions.words import APPROACH, APPROACH_GO_AROUND, UNCHANGED

    airport, signals, spec, groups, limits, masks = _busy_windows(tmp_path, monkeypatch)
    model = _traffic_model(spec)
    spoken = _flown(_window_loop(model, airport, signals, spec, groups, seed=0, limits=limits, procedure_masks=masks))
    given = []
    for r in spoken.results():
        words = r.said.copy()
        words[words[:, APPROACH] == APPROACH_GO_AROUND, APPROACH] = UNCHANGED
        given.append(Given(words, len(words)))
    none, laid_out = (_flown(_window_loop(model, airport, signals, spec, groups, seed=3, limits=limits,
                                          procedure_masks=masks, given=given, go_around_extra_s=extra))
                      for extra in (0.0, GO_AROUND_EXTRA_S))
    assert not any(r.go_around is not None for r in laid_out.results())
    for x, y in zip(none.results(), laid_out.results()):
        assert (x.outcome, x.own, x.end, x.counted, x.landing_s, x.runway) == \
               (y.outcome, y.own, y.end, y.counted, y.landing_s, y.runway)
        assert x.said.shape == y.said.shape and (x.said == y.said).all()
        # the same states; the laid-out executor's cycles past every halt repeat the last one
        flown_x = none.executors[x.group][2].flown().states[x.place]
        flown_y = laid_out.executors[y.group][2].flown().states[y.place]
        assert len(flown_y) >= len(flown_x) and torch.equal(flown_x, flown_y[: len(flown_x)])
        assert (flown_y[len(flown_x):] == flown_x[-1]).all()
    assert [r.ended for r in none.runs] == [r.ended for r in laid_out.runs]
    assert [r.episodes for r in none.runs] == [r.episodes for r in laid_out.runs]


def test_a_line_is_given_to_an_own_step_inside_its_words():
    import numpy as np
    import pytest

    from ts_transformer.experiments.traffic_window import Given

    words = np.zeros((4, 6), dtype=np.int64)
    assert Given(words, 4).until == 4 and Given(words, 0).until == 0
    with pytest.raises(ValueError, match="own step 5 of 4"):
        Given(words, 5)
