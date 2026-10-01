"""The prior commanding one aircraft of a scene (`experiments/traffic_speaking`, multi-aircraft design §6.6 step 3): alone
it says and flies what single-aircraft free generation does; the others enter its edge features; the separation masks
read a leader ahead; the judge ends it at a loss it answers for. On the scene-data fixture (a tmp artefact)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace

import numpy as np
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, SPEED, Words
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import with_traffic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_autopilot import _params, _physics
from ts_transformer.tests.test_prior_speaker import _model
from ts_transformer.tests.test_traffic_scene_data import _artefact

LIMIT_S = 60.0


def _airport(tmp_path, monkeypatch):
    """f0, f2 (background) and f1 entering 0, 10 and 30 s apart on one approach, f3 an hour later, alone."""
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_speaking import scene_airports
    from ts_transformer.instructions.artefact import load_signals

    directory, manifest, spec, _ = _artefact(tmp_path, [0.0, 30.0, 10.0, 3_600.0], [0, 1, 3])
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    airports, _ = scene_airports(directory, "train", spec, ("KXXX",), None, 2_048)
    signals = {s.dataset_id: s for s in load_signals(directory, "train")}
    return airports["KXXX"], signals, spec


def _loop(model, airport, signals, spec, key, seed=4):
    from ts_transformer.experiments.traffic_speaking import SceneLoop, scene_of

    flight = signals[key]
    start = replace(flight, **{name: getattr(flight, name)[N_LOOK:] for name in
                               ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                "vertical_rate_mps")})
    inputs, runways, charts, approach = _physics(start, airport.flights.geometry)
    scene = scene_of(airport, key, LIMIT_S, spec.step_s)
    loop = SceneLoop(model, [flight], [airport.flights.geometry], inputs, runways, charts, approach, [LIMIT_S],
                     Words(spec), _params(), None, scenes=[scene], generator=torch.Generator().manual_seed(seed),
                     temperature=1.0, procedure_masks=ProcedureMasks.none())
    return loop, scene, (inputs, runways, charts, approach)


def test_alone_the_scene_loop_says_and_flies_what_single_aircraft_free_generation_does(tmp_path, monkeypatch):
    from ts_transformer.experiments.prior_free_generation import speak_and_fly

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    single = _model(Words(spec))
    torch.manual_seed(1)
    loop, scene, physics = _loop(with_traffic(single, EDGE_FEATURES).eval(), airport, signals, spec, "KXXX:f3")
    assert scene.others == ()
    while loop.running:
        loop.step()
    flown, said, _, _ = speak_and_fly(single, [signals["KXXX:f3"]], [airport.flights.geometry], *physics, [LIMIT_S],
                                      Words(spec), _params(), None, generator=torch.Generator().manual_seed(4),
                                      temperature=1.0, procedure_masks=ProcedureMasks.none())
    assert np.array_equal(loop.spoken.sentences(), said)
    assert torch.equal(loop.executor.flown().states, flown.states)


def test_the_others_enter_the_edge_features_and_the_masks_are_asked_every_step(tmp_path, monkeypatch):
    airport, signals, spec = _airport(tmp_path, monkeypatch)
    torch.manual_seed(1)
    loop, scene, _ = _loop(with_traffic(_model(Words(spec)), EDGE_FEATURES).eval(), airport, signals, spec, "KXXX:f1")
    assert set(scene.others) == {"KXXX:f0", "KXXX:f2"} and loop.speaker.aircraft == 3
    edges, asked = [], []
    original_edges, original_masks = loop.speaker.edges_of, loop.speaker.masks_of
    loop.speaker.edges_of = lambda first, last: (edges.append(original_edges(first, last)), edges[-1])[1]
    loop.speaker.masks_of = lambda column, chosen: (asked.append(column), original_masks(column, chosen))[1]
    for _ in range(3):
        loop.step()
    first = edges[0]
    # f0 entered 30 s (15 steps) before f1: the pre-roll holds it alone, then both
    assert loop.speaker.pre == 15 and first.shape[2] == 3
    both = first[0, 16:, 0, 1]                                  # f1 → f0 once f1 has moved since its first row
    assert np.abs(both[:, EDGE_FEATURES.index("front")]).min() > 0.0
    assert (both[:, EDGE_FEATURES.index("self")] == 0.0).all() and (first[0, :15, 0].sum() == 0.0)
    assert asked == [APPROACH, SPEED] * 3


def test_the_speed_and_clearance_masks_read_the_leader_ahead_on_the_final(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_speaking import Aircraft, Scene, others_at, speaking_masks

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    words = Words(spec)
    leader = airport.tracks["KXXX:f0"]
    # the leader established, 5 km before the threshold (runway 09 lands east from the origin: along = e)
    before = -leader.along_m
    row = int(np.argmin(np.abs(before - 5_000.0)))
    t_s = float(leader.presence.times_s[row])
    assert t_s >= leader.captured_s
    scene = Scene(airport, "KXXX:f1", t_s - 100.0, ("KXXX:f0",))

    def behind(gap_m, captured=True):
        return Aircraft(float(leader.e_m[row]) - gap_m, float(leader.n_m[row]), 600.0, 90.0, 75.0, captured)

    classes = words.speed_unspecified + 2
    others = others_at(scene, t_s, words)
    close = speaking_masks(scene, SPEED, classes, t_s, behind(6_000.0), 0, words.speed_unspecified, False, 70.0,
                           words, False, others)
    fastest, slowest = int(np.argmax([words.speed_mps(i) for i in range(words.speed_unspecified)])), \
        int(np.argmin([words.speed_mps(i) for i in range(words.speed_unspecified)]))
    assert not close[1 + fastest] and close[1 + slowest]         # too fast closes under the minimum, slow keeps it
    assert close[0] == close[1 + words.speed_unspecified]        # "unchanged" is the word in force ("unspecified")
    # not established: no speed mask; a clearance behind the cleared leader is masked inside the in-trail minimum
    assert speaking_masks(scene, SPEED, classes, t_s, behind(6_000.0, captured=False), 0, None, False, 70.0, words,
                          False, others).all()
    approach_classes = 1 + 3
    assert not speaking_masks(scene, APPROACH, approach_classes, t_s, behind(4_000.0), 0, None, False, 70.0, words,
                              False, others)[APPROACH_CLEARED + 1]
    assert speaking_masks(scene, APPROACH, approach_classes, t_s, behind(8_000.0), 0, None, False, 70.0, words,
                          False, others).all()
    # an aircraft already cleared has nothing to be masked
    assert speaking_masks(scene, APPROACH, approach_classes, t_s, behind(4_000.0), 0, None, True, 70.0, words,
                          False, others).all()


def test_the_judge_ends_the_speaking_aircraft_at_a_loss_it_answers_for(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled
    from ts_transformer.experiments.traffic_speaking import Scene, judged

    airport, _, spec = _airport(tmp_path, monkeypatch)
    leader = airport.tracks["KXXX:f0"]
    captured = np.flatnonzero(leader.presence.times_s >= leader.captured_s)
    times = leader.presence.times_s[captured[:20]]
    step_times = np.round(times[0] / spec.step_s) * spec.step_s + spec.step_s * np.arange(len(times))
    e = np.interp(step_times, leader.presence.times_s, leader.e_m) - 1_000.0         # 1 km behind on the final
    n = np.interp(step_times, leader.presence.times_s, leader.n_m)
    h = np.interp(step_times, leader.presence.times_s, leader.height_m)
    follower = Controlled(airport.flights.flights["KXXX:f1"].presence, step_times, e, n, h, ("09",) * len(e), e,
                          np.zeros(len(e)), n, np.full(len(e), 75.0), np.ones(len(e), dtype=bool),
                          airport.flights.flights["KXXX:f1"].category, "timeout", None)
    scene = Scene(airport, "KXXX:f1", float(step_times[0]) - N_LOOK * spec.step_s, ("KXXX:f0",))
    outcome, end, run = judged(scene, follower, spec.step_s)
    assert outcome == LOST_SEPARATION and end["with"] == "KXXX:f0" and math.isclose(end["t_s"], step_times[0])


def test_a_flown_scene_is_judged_from_its_first_predicted_step_on_its_steps(tmp_path, monkeypatch):
    from ts_transformer.autopilot.judge import OUTCOMES, outcome_of
    from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
    from ts_transformer.experiments.traffic_speaking import judged, speaking_aircraft
    from ts_transformer.instructions.words import RUNWAY

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    torch.manual_seed(1)
    loop, scene, _ = _loop(with_traffic(_model(Words(spec)), EDGE_FEATURES).eval(), airport, signals, spec, "KXXX:f1")
    while loop.running:
        loop.step()
    flown, grid = loop.executor.flown(), loop.spoken.sentences()[0]
    ended = outcome_of(flown, 0, airport.flights.geometry, int(grid[0, RUNWAY]), spec)
    aircraft = speaking_aircraft(scene, flown, 0, grid, ended.outcome, ended.end_row, ended.crossing, -1, spec.step_s)
    step_rows = int(round(spec.step_s / flown.cycle_s))
    assert aircraft.times_s[0] == scene.first_step_s + N_LOOK * spec.step_s
    assert np.allclose(np.diff(aircraft.times_s), spec.step_s) and aircraft.runway == ("09",) * len(aircraft.times_s)
    last = ended.end_row if ended.outcome == "timeout" else ended.end_row - 1
    assert len(aircraft.times_s) == last // step_rows + 1
    assert not aircraft.established[0]                            # the executor starts uncaptured
    outcome, end, run = judged(scene, aircraft, spec.step_s)
    assert outcome in (*OUTCOMES, LOST_SEPARATION) and (end is None) == (outcome != LOST_SEPARATION)


def _batch(airport, signals, spec, keys):
    """A replay batch of the fixture's flights at ``keys`` (no rebuilt series: the physics is the test aircraft's)."""
    from ts_transformer.autopilot import replay
    from ts_transformer.instructions.labeller.read import read_flight

    geometry = airport.flights.geometry
    flights = [signals[k] for k in keys]
    readings = [read_flight(f, geometry, spec, Words(spec)) for f in flights]
    # no rebuilt series: the flight's key stands where its series would be (the tests patch the executor's inputs)
    return replay.Batch(signals=flights, series=list(keys), readings=readings, geometries=[geometry] * len(keys),
                        vertical_paths=[()] * len(keys), approach_ias_mps=[70.0] * len(keys),
                        groups=[replay.OWN] * len(keys), drawn={"split": "train"})


def test_the_runner_reads_the_model_alone_and_in_its_scene_and_the_record(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_free_generation as runner
    from ts_transformer.experiments.traffic_speaking import scene_of
    from ts_transformer.inference.separation import IFR, VISUAL
    from ts_transformer.tests.test_prior_speaker import _repeated

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    keys = ["KXXX:f1", "KXXX:f3"]
    batch = _batch(airport, signals, spec, keys)
    physics = []
    for key in keys:
        flight = signals[key]
        start = replace(flight, **{name: getattr(flight, name)[N_LOOK:] for name in
                                   ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                    "vertical_rate_mps")})
        physics.append(_physics(start, airport.flights.geometry))
    samples = 2
    index = torch.tensor([j for j in range(len(keys)) for _ in range(samples)])

    def stacked(k):
        tables = [p[k] for p in physics]
        if isinstance(tables[0], torch.Tensor):
            return torch.cat(tables)[index]
        joined = type(tables[0])(**{f.name: torch.cat([getattr(t, f.name) for t in tables])
                                    for f in dataclasses.fields(tables[0])})
        return _repeated(joined, index)

    monkeypatch.setattr(runner, "flight_inputs", lambda series, device, anchor: stacked(0))
    monkeypatch.setattr(runner, "_physics", lambda batch, device: (stacked(1), stacked(2), stacked(3)))
    monkeypatch.setattr(runner, "limits_s", lambda batch, params, step_s, augmented: [LIMIT_S] * len(batch.readings))
    scenes = [scene_of(airport, key, LIMIT_S, spec.step_s) for key in keys]
    torch.manual_seed(1)
    model = with_traffic(_model(Words(spec)), EDGE_FEATURES).eval()
    rows = runner.model_rows(model, batch, scenes, scenes, "scene", Words(spec), _params(), None, samples,
                             generator=torch.Generator().manual_seed(4), temperature=1.0,
                             procedure_masks=ProcedureMasks.none())
    assert [r["dataset_id"] for r in rows] == [k for k in keys for _ in range(samples)]
    assert [r["others"] for r in rows] == [2, 2, 0, 0] and all(r["source"] == "scene" for r in rows)
    for r in rows:
        assert set(r["mask_steps"]) == {"approach", "speed"} and r[VISUAL]["outcome"] and r[IFR]["outcome"]
        assert 0 <= r["mask_steps"]["speed"] <= r["judged_steps"] <= r["steps_said"]
    # f1 follows f0 by 30 s on its path: already in a loss it answers for at its first predicted step, ended there
    assert [r["starts_in_a_loss"] for r in rows] == [True, True, False, False]
    assert rows[0][VISUAL]["flown_s"] == 0.0 and rows[0]["judged_steps"] == 0 and rows[2][VISUAL]["flown_s"] > 0.0
    alone = runner.model_rows(model, batch, [replace(s, others=()) for s in scenes], scenes, "alone", Words(spec),
                              _params(), None, samples, generator=torch.Generator().manual_seed(4), temperature=1.0,
                              procedure_masks=ProcedureMasks.none())
    assert all(r["mask_steps"] == {"approach": 0, "speed": 0} and r["speaking_with"] == 0 for r in alone)
    assert [r["others"] for r in alone] == [2, 2, 0, 0]                      # judged against the scene it left
    # the lone flight (no others) flies alone the same in both
    assert [r["outcome"] for r in rows[2:]] == [r["outcome"] for r in alone[2:]]
    recorded = runner.recorded_rows(batch, scenes, Words(spec))
    assert [r["outcome"] for r in recorded] == ["landed", "landed"] or recorded[0][VISUAL]["outcome"] == "lost_separation"
    readout = runner.summary(rows + alone + recorded)
    assert set(readout) == {"scene", "alone", "recorded"} and readout["scene"]["flights"] == 2
    assert readout["scene"]["left_out_starting_in_a_loss"] == 2 and readout["recorded"]["left_out_starting_in_a_loss"] == 1
    assert readout["alone"]["mask_steps_share"] == {"approach": 0.0, "speed": 0.0}


def test_a_closed_loop_lets_its_speaker_go_without_the_cyclic_collector(tmp_path, monkeypatch):
    """The speaker holds the loop's callbacks and the loop holds it: `close` breaks the cycle, so the speaker's past (the
    model's keys and values of every step) is freed as soon as the loop is dropped."""
    import gc
    import weakref

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    torch.manual_seed(1)
    loop, _, _ = _loop(with_traffic(_model(Words(spec)), EDGE_FEATURES).eval(), airport, signals, spec, "KXXX:f1")
    loop.step()
    speaker = weakref.ref(loop.speaker)
    gc.disable()
    try:
        loop.close()
        del loop
        assert speaker() is None
    finally:
        gc.enable()
