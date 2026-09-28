"""The multi-aircraft post-training's optimiser (`experiments/traffic_tuner`, multi-aircraft design §6.6 step 6): the
loop's edge features, step by step, are the scorer's all at once — with the others' rows off the steps; a sentence scored
in its scene gives back the distribution each of its words was sampled from — with a traffic attention that reads the
others — laid out with sentences of other pre-rolls; at a zero traffic attention it scores as the single-aircraft tuner
does; a batch scored in parts has the gradient of one piece; the traffic attention trains at its own learning rate. On
the scene-data fixture (a tmp artefact)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Split, chain_record, column_classes
from ts_transformer.prior.model import with_traffic
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.train import RewardConfig, RewardTuner
from ts_transformer.tests.test_prior_speaker import _model
from ts_transformer.tests.test_traffic_speaking import _loop

CPU = torch.device("cpu")


def _airport(tmp_path, monkeypatch, entries_s=(0.0, 30.0, 10.0, 3_600.0), labelled=(0, 1, 3), context=False):
    """The scene-data fixture's flights entering ``entries_s`` apart (default: f0, f2 background and f1 on one approach,
    f3 an hour later, alone — `test_traffic_speaking`'s), those at ``labelled`` with a sentence; ``context``: their inputs
    with the landing context, their landings (`_landings`) returned too."""
    from ts_transformer.experiments import traffic_scene_data
    from ts_transformer.experiments.traffic_speaking import scene_airports
    from ts_transformer.instructions.artefact import load_candidates, load_signals
    from ts_transformer.tests.test_traffic_scene_data import _artefact

    directory, manifest, spec, _ = _artefact(tmp_path, list(entries_s), list(labelled))
    monkeypatch.setattr(traffic_scene_data, "arrival_manifest_path", lambda code: manifest)
    signals = {s.dataset_id: s for s in load_signals(directory, "train")}
    landings = _landings(signals, load_candidates(directory)["KXXX"]) if context else None
    airports, _ = scene_airports(directory, "train", spec, ("KXXX",), landings, 2_048)
    return (airports["KXXX"], signals, spec) + ((landings,) if context else ())


def _landings(signals, geometry):
    """The fixture's landings as an airport's landing context."""
    from ts_transformer.data.day_split import parse_utc
    from ts_transformer.prior.scene import Landings

    times = np.sort([parse_utc(s.landing_time_utc).timestamp() for s in signals.values()])
    return {"KXXX": Landings(times, {c.ident: times for c in geometry.candidates})}


def _traffic_model(words, seed=1, reads=True, variant="no-context"):
    """A traffic prior; ``reads``: its traffic attention's output layer moved off zero, so the others change its words."""
    torch.manual_seed(seed)
    model = with_traffic(_model(words, variant=variant), EDGE_FEATURES).eval()
    if reads:
        for layer in model.layers:
            torch.nn.init.normal_(layer.traffic.out.weight, std=0.2)
    return model


def _spoken(model, airport, signals, spec, key, seed=4):
    """``key`` spoken to in its scene: its sentence as a training record, its scene, its positions, and the logits each
    of its words was sampled from (per step, per column: ``[classes]``)."""
    loop, scene, _ = _loop(model, airport, signals, spec, key, seed=seed)
    recorded = []
    original = model.logits

    def spy(h, tokens, valid, chosen, first):
        out = original(h, tokens, valid, chosen, first)
        recorded.append([logit[0, 0, 0].detach().clone() for logit in out])
        return out

    model.logits = spy
    try:
        while loop.running:
            loop.step()
    finally:
        del model.logits
    steps = len(recorded) // 6
    said = loop.spoken.sentences()[0][:steps]
    speaker = loop.speaker
    rows = N_LOOK + steps
    positions = np.column_stack((speaker.e[0, :rows], speaker.n[0, :rows], speaker.h[0, :rows]))
    classes = np.where(said != UNCHANGED, said + 1, 0)
    flight = chain_record(signals[key], positions[:, 0], positions[:, 1], positions[:, 2], said, classes,
                          np.ones(said.shape, dtype=bool), airport.flights.geometry, None, 0, 0, spec.step_s)
    sampled = [[recorded[6 * s + c][c] for c in range(6)] for s in range(steps)]
    loop.close()
    return flight, scene, positions, sampled


def _split(flights, words, geometry):
    table = prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 1)
    return Split(list(flights), ("KXXX",), table, (("09",),), ((90.0,),), column_classes(words, 1), "no-context")


@pytest.mark.parametrize("off_s", [0.0, 0.7, -0.7])
def test_the_loop_s_edge_features_step_by_step_are_the_scorer_s_all_at_once(tmp_path, monkeypatch, off_s):
    """The others' rows sit at their recorded times, up to a second off the speaking aircraft's steps (here f0 and f2
    ``off_s`` off f1's): another aircraft later than the step is then carried forward from its row before at that row's
    motion, which the loop reads two steps back for."""
    from ts_transformer.experiments.traffic_speaking import speaking_edges
    from ts_transformer.inference.scene_edges import EDGE_FEATURES as FEATURES
    from ts_transformer.instructions.words import RUNWAY

    airport, signals, spec = _airport(tmp_path, monkeypatch, (off_s, 30.0, 10.0 + off_s, 3_600.0))
    loop, scene, _ = _loop(_traffic_model(Words(spec)), airport, signals, spec, "KXXX:f1")
    blocks, edges_of = [], loop.speaker.edges_of
    loop.speaker.edges_of = lambda first, last: (blocks.append((first, edges_of(first, last))), blocks[-1][1])[1]
    for _ in range(6):
        loop.step()
    speaker = loop.speaker
    whole = speaking_edges(scene, speaker.e[0, : speaker.rows], speaker.n[0, : speaker.rows], speaker.h[0, : speaker.rows],
                           speaker.in_force[0, 0, : speaker.rows, RUNWAY].numpy(), speaker.pre, spec.step_s)
    assert len(blocks) > 3
    for first, block in blocks:
        assert np.array_equal(block[0], whole[first: first + len(block[0])])
    # the others' motion is read where the speaking aircraft has moved (not "unknown")
    assert (whole[speaker.pre + N_LOOK:, 0, 1, FEATURES.index("motion_unknown")] == 0.0).all()


@pytest.mark.parametrize("off_s", [0.0, 0.7, -0.7])
def test_a_sentence_scored_in_its_scene_gives_back_what_its_words_were_sampled_from(tmp_path, monkeypatch, off_s):
    from ts_transformer.experiments.traffic_tuner import SceneSplit, scene_layout, scene_logits
    from ts_transformer.prior.train import to_batch

    airport, signals, spec = _airport(tmp_path, monkeypatch, (off_s, 30.0, 10.0 + off_s, 3_600.0))
    words = Words(spec)
    model = _traffic_model(words)
    # f1 has two others in the air before it (a pre-roll of 15 steps); f3 is alone (none): scored together, f3's rows
    # sit 15 steps later than they did when it spoke
    spoken = [_spoken(model, airport, signals, spec, key) for key in ("KXXX:f1", "KXXX:f3")]
    assert len(spoken[0][1].others) == 2 and spoken[1][1].others == ()
    sentences = SceneSplit(_split([s[0] for s in spoken], words, airport.flights.geometry), [s[1] for s in spoken],
                           [s[2] for s in spoken])
    batch = to_batch(sentences.split, [0, 1], CPU)
    layout = scene_layout(sentences, [0, 1], batch, spec.step_s)
    assert layout.pre == 15 and layout.inputs["present"].shape[1] == 3
    with torch.no_grad():
        logits = scene_logits(model, layout)
        alone = scene_logits(with_traffic(_model(words), EDGE_FEATURES).eval(), layout)
    moved = 0.0
    for b, (_, _, _, sampled) in enumerate(spoken):
        assert len(sampled) > 5
        for s, columns in enumerate(sampled):
            for c, want in enumerate(columns):
                got = logits[c][b, 0, N_LOOK + s]
                finite = torch.isfinite(want)
                assert torch.equal(finite, torch.isfinite(got))
                assert torch.allclose(got[finite], want[finite], atol=1e-4), (b, s, c)
                if b == 0:
                    moved = max(moved, float((got[finite] - alone[c][b, 0, N_LOOK + s][finite]).abs().max()))
    assert moved > 1e-2                                          # the others did change f1's words


def _round(tmp_path, monkeypatch, reads):
    from ts_transformer.experiments.traffic_scene_data import build_split
    from ts_transformer.experiments.traffic_tuner import SceneSplit

    airport, signals, spec = _airport(tmp_path, monkeypatch)
    words = Words(spec)
    model = _traffic_model(words, reads=reads)
    spoken = [_spoken(model, airport, signals, spec, key, seed=seed) for key in ("KXXX:f1", "KXXX:f3")
              for seed in (4, 5)]
    sentences = SceneSplit(_split([s[0] for s in spoken], words, airport.flights.geometry), [s[1] for s in spoken],
                           [s[2] for s in spoken])
    directory = tmp_path / "artefact"
    data, _ = build_split(directory, "train", spec, ("KXXX",), None, 2_048)
    return model, words, sentences, [b for b in data if b.sample.asks], spec


def test_at_a_zero_traffic_attention_the_scene_tuner_measures_what_the_single_tuner_does(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_tuner import SceneRewardTuner

    model, words, sentences, _, spec = _round(tmp_path, monkeypatch, reads=False)
    reference = _model(words)
    for parameter in reference.parameters():                     # a reference apart from the model
        parameter.data.add_(0.01 * torch.randn_like(parameter))
    single = RewardTuner(_model(words), reference, RewardConfig(), CPU, seed=0).distance(sentences.split)
    scene = SceneRewardTuner(model, reference, RewardConfig(), CPU, seed=0, traffic_learning_rate=3e-4,
                             step_s=spec.step_s).distance(sentences)
    assert single > 1e-4 and scene == pytest.approx(single, rel=1e-4)


@pytest.mark.filterwarnings("ignore:Seems like `optimizer.step\\(\\)` has been overridden")
def test_a_batch_scored_in_parts_has_the_gradient_of_one_piece(tmp_path, monkeypatch):
    from ts_transformer.experiments import traffic_tuner
    from ts_transformer.experiments.traffic_tuner import SceneRewardTuner

    _, words, sentences, data, spec = _round(tmp_path, monkeypatch, reads=True)
    advantages = np.array([0.5, -0.5, 0.25, -0.25])
    gradients = []
    for budget in (10 ** 9, 1):
        monkeypatch.setattr(traffic_tuner, "SCORE_BUDGET", budget)
        trained = _traffic_model(words, reads=True)
        tuner = SceneRewardTuner(trained, _model(words), RewardConfig(), CPU, seed=0, traffic_learning_rate=3e-4,
                                 step_s=spec.step_s)
        assert len(tuner._parts(sentences, [0, 1, 2, 3])) == (1 if budget > 1 else 4)
        seen, step = [], tuner.optimiser.step
        tuner.optimiser.step = lambda: (seen.append({name: p.grad.clone() for name, p in trained.named_parameters()
                                                     if p.grad is not None}), step())[1]
        record = tuner.one_pass(sentences, advantages, data, slots=1)
        assert record["batches"] == 1 and record["sentences"] == 4 and len(seen) == 1
        gradients.append(seen[0])
    assert gradients[0].keys() == gradients[1].keys()
    assert max(float(g.abs().max()) for name, g in gradients[0].items() if ".traffic." not in name) > 0.0
    for name, g in gradients[0].items():
        assert torch.allclose(g, gradients[1][name], rtol=1e-4, atol=1e-7), name


def test_the_traffic_attention_trains_at_its_own_learning_rate(tmp_path, monkeypatch):
    from ts_transformer.experiments.traffic_tuner import SceneRewardTuner, traffic_parameters

    model, words, _, _, spec = _round(tmp_path, monkeypatch, reads=False)
    tuner = SceneRewardTuner(model, _model(words), RewardConfig(), CPU, seed=0, traffic_learning_rate=3e-4,
                             step_s=spec.step_s)
    traffic, rest = tuner.optimiser.param_groups
    assert traffic["lr"] * 20 == pytest.approx(3e-4) and rest["lr"] * 20 == pytest.approx(1e-5)     # warm-up step 1
    ids = [id(p) for group in (traffic, rest) for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert {id(p) for p in traffic["params"]} == {id(p) for p in traffic_parameters(model)}
    with pytest.raises(ValueError, match="reads the aircraft alone"):
        SceneRewardTuner(model, model, RewardConfig(), CPU, seed=0, traffic_learning_rate=3e-4, step_s=spec.step_s)


def test_layers_recomputed_in_the_backward_give_the_same_logits_and_gradients(tmp_path, monkeypatch):
    """`Prior.encode(checkpoint=True)` (the scorer's, for memory) against the layers kept: the same forward and the same
    gradients, in a scene whose traffic attention reads the others."""
    from ts_transformer.experiments.traffic_tuner import SceneSplit, scene_layout, scene_logits
    from ts_transformer.prior.train import to_batch

    airport, signals, spec = _airport(tmp_path, monkeypatch, (0.7, 30.0, 10.7, 3_600.0))
    words = Words(spec)
    flight, scene, positions, _ = _spoken(_traffic_model(words), airport, signals, spec, "KXXX:f1")
    sentences = SceneSplit(_split([flight], words, airport.flights.geometry), [scene], [positions])
    results = []
    for checkpoint in (False, True):
        model = _traffic_model(words)
        layout = scene_layout(sentences, [0], to_batch(sentences.split, [0], CPU), spec.step_s)
        logits = scene_logits(model, layout, checkpoint=checkpoint)
        sum((logit.clamp(min=-1e4) * torch.linspace(0.1, 1.0, logit.shape[-1])).sum() for logit in logits).backward()
        results.append(([logit.detach() for logit in logits],
                        {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}))
    (kept, kept_grad), (again, again_grad) = results
    assert all(torch.equal(a, b) for a, b in zip(kept, again))
    assert kept_grad.keys() == again_grad.keys() and any(".traffic." in name for name in kept_grad)
    for name, grad in kept_grad.items():
        assert torch.allclose(grad, again_grad[name], rtol=1e-5, atol=1e-8), name
