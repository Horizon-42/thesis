"""Augmented scenes for the one-commanded loop (`experiments/traffic_augment`, multi-aircraft design §5): a flight moved in
time, the leader found, the qualification, an insertion ahead at a gap and a moved start. On the scene-data fixture."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_autopilot import _physics
from ts_transformer.tests.test_traffic_speaking import LIMIT_S, _airport


def test_a_moved_flight_keeps_its_rows_and_moves_its_times_steps_and_capture(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_augment import moved

    airport, _, spec = _airport(tmp_path, monkeypatch)
    rows, track = airport.flights.flights["KXXX:f0"], airport.tracks["KXXX:f0"]
    new_rows, new_track = moved(rows, track, 37.0, "KXXX:f0+x", spec.step_s)
    assert new_rows.presence.dataset_id == new_track.key == "KXXX:f0+x"
    assert np.allclose(new_rows.presence.times_s - rows.presence.times_s, 37.0) and new_rows.node_rows is rows.node_rows
    assert new_track.captured_s == track.captured_s + 37.0 and new_track.first_step_s % spec.step_s == 0.0
    assert new_rows.presence.landing_s == rows.presence.landing_s + 37.0


def test_the_leader_is_next_ahead_and_a_scene_lost_before_the_prior_speaks_does_not_qualify(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_augment import leader, qualifies
    from ts_transformer.experiments.traffic_speaking import scene_of

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    follows = scene_of(airport, "KXXX:f1", LIMIT_S, spec.step_s)
    assert leader(follows, spec.step_s) == "KXXX:f2"           # lands 20 s before f1, f0 30 s
    assert not qualifies(follows, signals["KXXX:f1"], spec.step_s)
    alone = scene_of(airport, "KXXX:f3", LIMIT_S, spec.step_s)
    assert leader(alone, spec.step_s) is None and qualifies(alone, signals["KXXX:f3"], spec.step_s)


def _inputs(airport, signals, key):
    flight = signals[key]
    start = replace(flight, **{name: getattr(flight, name)[N_LOOK:] for name in
                               ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                "vertical_rate_mps")})
    return _physics(start, airport.flights.geometry)[0]


def test_an_insertion_lands_the_gap_ahead_on_the_clock_and_a_moved_start_moves_the_speaking_rows(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_augment import INSERTED, augment
    from ts_transformer.experiments.traffic_speaking import scene_of

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    scene = scene_of(airport, "KXXX:f3", LIMIT_S, spec.step_s)
    inputs = _inputs(airport, signals, "KXXX:f3")
    windows = {"KXXX": (-1e9, 1e9)}
    out = augment(scene, signals["KXXX:f3"], inputs, ["KXXX:f0", "KXXX:f1"], np.random.default_rng(3), windows,
                  spec.step_s, kinds=("A",))
    assert out is not None and out.kind == "A" and out.augmentation is None
    key = out.drawn["inserted"] + INSERTED
    assert key in out.scene.others and out.scene.rows(key).presence.dataset_id == key
    from ts_transformer.experiments.traffic_census import APPROACH_SPEED_MPS

    gap_s = airport.tracks["KXXX:f3"].presence.landing_s - out.scene.track(key).presence.landing_s
    assert abs(gap_s - out.drawn["gap"] * out.drawn["required_m"] / APPROACH_SPEED_MPS) < 1e-6
    assert out.drawn["gap"] >= 0.5 and out.drawn["gap"] <= 2.0
    moved = augment(scene, signals["KXXX:f3"], inputs, [], np.random.default_rng(3), windows, spec.step_s,
                    kinds=("B",))
    assert moved is not None and moved.kind == "B" and moved.scene is scene
    assert not np.allclose(moved.signals.e_m, signals["KXXX:f3"].e_m)
    # nothing to move: D never applies alone
    assert augment(scene, signals["KXXX:f3"], inputs, [], np.random.default_rng(3), windows, spec.step_s,
                   kinds=("D",)) is None
