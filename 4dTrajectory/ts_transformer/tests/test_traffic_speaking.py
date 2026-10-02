"""One commanded aircraft's view of its scene (`experiments/traffic_speaking`, multi-aircraft design §6.6 step 3): the
separation masks read a leader ahead; the judge ends it at a loss it answers for. The fixture's airport and replay batch,
which the window loop's tests read too. On the scene-data fixture (a tmp artefact). (The window loop flying a commanded
aircraft in its scene: `test_traffic_window.py`.)"""

from __future__ import annotations

import math

import numpy as np

from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, SPEED, Words
from ts_transformer.prior.scene import N_LOOK
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
