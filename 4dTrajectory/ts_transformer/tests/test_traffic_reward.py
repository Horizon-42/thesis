"""Multi-aircraft M4's runner (`experiments/traffic_reward`, design §6.6 step 6): its sentences, cut at their judged ends
and read back as training rows in their scenes, give back the distribution each word was sampled from; the landing gap
reads the landing before on the runway; a round refuses to run short of augmented scenes; the guards exclude a round by
name and the choice takes the earliest within the tie. On the scene-data fixture (a tmp artefact)."""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.test_autopilot import _params, _physics
from ts_transformer.tests.test_prior_speaker import _repeated
from ts_transformer.tests.test_traffic_speaking import LIMIT_S, _batch
from ts_transformer.tests.test_traffic_tuner import _split, _traffic_model

CPU = torch.device("cpu")


def _airport(tmp_path, monkeypatch, entries_s, labelled):
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_speaking import scene_airports
    from ts_transformer.instructions.artefact import load_signals
    from ts_transformer.tests.test_traffic_scene_data import _artefact

    directory, manifest, spec, _ = _artefact(tmp_path, entries_s, labelled)
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    airports, _ = scene_airports(directory, "train", spec, ("KXXX",), None, 2_048)
    signals = {s.dataset_id: s for s in load_signals(directory, "train")}
    return airports["KXXX"], signals, spec


def _patch_physics(monkeypatch, airport, signals, keys, samples):
    """The runner's executor inputs for ``keys`` (each ``samples`` times, flight-major): the test aircraft's."""
    from ts_transformer.experiments import traffic_free_generation, traffic_reward

    physics = []
    for key in keys:
        flight = signals[key]
        start = replace(flight, **{name: getattr(flight, name)[N_LOOK:] for name in
                                   ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                    "vertical_rate_mps")})
        physics.append(_physics(start, airport.flights.geometry))
    index = torch.tensor([j for j in range(len(keys)) for _ in range(samples)])

    def stacked(k):
        tables = [p[k] for p in physics]
        if isinstance(tables[0], torch.Tensor):
            return torch.cat(tables)[index]
        joined = type(tables[0])(**{f.name: torch.cat([getattr(t, f.name) for t in tables])
                                    for f in dataclasses.fields(tables[0])})
        return _repeated(joined, index)

    monkeypatch.setattr(traffic_free_generation, "flight_inputs", lambda series, device, anchor: stacked(0))
    monkeypatch.setattr(traffic_free_generation, "_physics", lambda batch, device: (stacked(1), stacked(2), stacked(3)))
    for module in (traffic_free_generation, traffic_reward):
        monkeypatch.setattr(module, "limits_s", lambda batch, params, step_s, augmented: [LIMIT_S] * len(batch.readings))


def test_a_round_s_sentences_train_on_what_their_words_were_sampled_from(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_free_generation import speaking_batches
    from ts_transformer.experiments.traffic_reward import Round, round_summary, sentence_split, speak
    from ts_transformer.experiments.traffic_speaking import scene_of
    from ts_transformer.experiments.traffic_tuner import scene_layout, scene_logits
    from ts_transformer.prior.train import to_batch

    # f1 follows f0 by 120 s on its approach (in its scene, not in a loss), f2 an hour later alone
    airport, signals, spec = _airport(tmp_path, monkeypatch, [0.0, 120.0, 3_600.0], [0, 1, 2])
    words, samples, keys = Words(spec), 2, ["KXXX:f1", "KXXX:f2"]
    _patch_physics(monkeypatch, airport, signals, keys, samples)
    batch = _batch(airport, signals, spec, keys)
    scenes = [scene_of(airport, key, LIMIT_S, spec.step_s) for key in keys]
    assert scenes[0].others == ("KXXX:f0",) and scenes[1].others == ()
    geometry = airport.flights.geometry
    round_ = Round(batch, scenes, [None, None], ["real", "A"], [np.ones(len(geometry.candidates), dtype=bool)] * 2)
    model = _traffic_model(words)
    recorded, original = [], model.logits

    def spy(h, tokens, valid, chosen, first):
        out = original(h, tokens, valid, chosen, first)
        recorded.append([logit[:, 0, 0].detach().clone() for logit in out])
        return out

    model.logits = spy
    try:
        spoken = speak(model, round_, samples, words, _params(), None, ProcedureMasks.none(),
                       generator=torch.Generator().manual_seed(4), budget=10 ** 9, source="train")
    finally:
        del model.logits
    assert [r["dataset_id"] for r in spoken.rows] == [k for k in keys for _ in range(samples)]
    assert [r["kind"] for r in spoken.rows] == ["real", "real", "A", "A"]
    assert all(r["reward"] == float(r["visual"]["outcome"] == "landed") for r in spoken.rows)
    assert not any(r["starts_in_a_loss"] for r in spoken.rows)
    for s, row in enumerate(spoken.rows):
        assert len(spoken.said[s]) == row["judged_steps"] and len(spoken.positions[s]) == N_LOOK + row["judged_steps"]
        assert all(len(codes) == row["judged_steps"] for codes in spoken.allowed[s].values())
    described = round_summary(round_, spoken, samples)
    assert described["scenes"] == 2 and described["kinds"] == {"real": 1, "A": 1}
    keep = np.arange(len(spoken.rows))
    sentences = sentence_split(round_, spoken, keep, samples, _split([], words, geometry), None, ("KXXX",), spec.step_s)
    single = to_batch(sentences.split, list(keep), CPU)
    with torch.no_grad():
        logits = scene_logits(model, scene_layout(sentences, list(keep), single, spec.step_s))
    # the loop spoke in its batches' order (by scene size: f2 first), the round reads flight-major in its own
    (chunk,) = speaking_batches(scenes, [LIMIT_S] * 2, [len(r.words) for r in batch.readings], samples, 10 ** 9,
                                spec.step_s)
    assert chunk == [1, 0]
    for s, row in enumerate(spoken.rows):
        spoke = chunk.index(s // samples) * samples + s % samples
        for step in range(row["judged_steps"]):
            for c in range(6):
                want, got = recorded[6 * step + c][c][spoke], logits[c][s, 0, N_LOOK + step]
                finite = torch.isfinite(want)
                assert torch.equal(finite, torch.isfinite(got))
                assert torch.allclose(got[finite], want[finite], atol=1e-4), (s, step, c)


def test_the_landing_gap_reads_the_landing_before_on_the_runway(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_reward import landing_gap_s
    from ts_transformer.experiments.traffic_speaking import scene_of

    airport, signals, spec = _airport(tmp_path, monkeypatch, [0.0, 120.0, 3_600.0], [0, 1, 2])
    scene = scene_of(airport, "KXXX:f1", LIMIT_S, spec.step_s)
    own = scene.track("KXXX:f1").presence
    assert landing_gap_s(scene, own.runway, own.landing_s) == pytest.approx(120.0, abs=1.0)
    assert landing_gap_s(scene, own.runway, scene.track("KXXX:f0").presence.landing_s) is None
    alone = scene_of(airport, "KXXX:f2", LIMIT_S, spec.step_s)
    assert landing_gap_s(alone, own.runway, alone.track("KXXX:f2").presence.landing_s) is None


def test_a_round_refuses_to_run_short_of_augmented_scenes(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_reward import first_per_airport

    airport, signals, spec = _airport(tmp_path, monkeypatch, [0.0, 120.0, 3_600.0], [0, 1, 2])
    batch = _batch(airport, signals, spec, ["KXXX:f0", "KXXX:f1", "KXXX:f2"])
    assert first_per_airport(batch, 2, ["KXXX"]) == [0, 1]
    with pytest.raises(ValueError, match="KXXX"):
        first_per_airport(batch, 4, ["KXXX"])
    with pytest.raises(ValueError, match="KYYY"):
        first_per_airport(batch, 1, ["KXXX", "KYYY"])


def _row(number, *, reward, landed=0.9, runway=0.95, lost=0.1, time_ratio=1.0, gap_ratio=1.0, heading=3.0):
    def side(real):
        out = {"free_generation": {"all": {"outcomes": {"landed": landed}, "landed_on_observed_runway": runway,
                                           "words_after_first_per_flight": {"approach": 1.0, "heading": heading,
                                                                            "altitude": 1.0, "angle": 1.0,
                                                                            "speed": 1.0}}},
               "separation": {"lost_separation": lost}, "reward": reward}
        if real:
            out["ordering"] = {"time_ratio": time_ratio, "gap_ratio": gap_ratio}
        return out

    return {"round": number, "real": side(True), "augmented": side(False)}


def test_the_guards_exclude_a_round_by_name_and_the_choice_takes_the_earliest_within_the_tie():
    from ts_transformer.experiments.traffic_reward import guarded_choice

    labelled = {side: {"approach": 1.0, "heading": 3.0, "altitude": 1.0, "angle": 1.0, "speed": 1.0}
                for side in ("real", "augmented")}
    history = [_row(0, reward=0.50), _row(1, reward=0.60), _row(2, reward=0.61), _row(3, reward=0.90, lost=0.12),
               _row(4, reward=0.90, landed=0.85), _row(5, reward=0.90, gap_ratio=1.3), _row(6, reward=0.90, heading=4.0),
               _row(7, reward=0.90, time_ratio=None)]
    kept, excluded = guarded_choice(history, labelled)
    assert kept == 1                                              # 0.61 is within 0.015 of 0.60: the earlier one
    assert excluded == {3: ["lost separation"], 4: ["landed"], 5: ["landing gap"],
                        6: ["real heading words", "augmented heading words"], 7: ["time to land"]}
    assert guarded_choice([_row(0, reward=0.5), _row(1, reward=0.4)], labelled) == (0, {})
