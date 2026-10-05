"""Stage C, C5: the traffic attention (post-training §3, §8 C5; D29, D98)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.post.edges import TOKEN_FEATURES
from ts_transformer.post.traffic_attention import (
    TRAFFIC_ATTENTION_SCHEMA, Traffic, TrafficAttention, TrafficConfig, TrafficTokens, add_traffic_attention,
    parameter_groups,
    traffic_modules, traffic_of,
)
from ts_transformer.prior.batch import collate
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.speaker import Position, Speaker
from ts_transformer.tests.post_support import airport, finals
from ts_transformer.tests.support import instruction_spec, prior_sentence

CPU = torch.device("cpu")
SMALL = {"d_model": 32, "layers": 2, "heads": 4, "feedforward": 64}
CONFIG = TrafficConfig(hidden=16, heads=4)
WORDS = Words(instruction_spec())


def _setup(seed=0, count=3, rows=30, first_step=8):
    torch.manual_seed(seed)
    base = Prior(PriorConfig.from_words(WORDS, "full", **SMALL)).eval()
    rng = np.random.default_rng(seed)
    sentences = [prior_sentence(rng, WORDS, candidates=2, rows=rows, first_step=first_step) for _ in range(count)]
    return base, collate(sentences, CPU), rng


def _traffic(rng, count, rows, most=4, empty_rows=()):
    """Random tokens, a random number of other aircraft at each row (none at ``empty_rows``)."""
    out = []
    for _ in range(count):
        aircraft = []
        for r in range(rows):
            n = 0 if r in empty_rows else int(rng.integers(0, most + 1))
            aircraft.append(rng.normal(size=(n, len(TOKEN_FEATURES))).astype(np.float32))
        out.append(aircraft)
    return traffic_of(out, CPU)


def _with_module(base):
    model = copy.deepcopy(base)
    add_traffic_attention(model, CONFIG)
    return model.eval()


def _randomize(model, seed=7):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in traffic_modules(model):
            for p in module.parameters():
                p.copy_(torch.randn(p.shape, generator=generator) * 0.3)


def test_the_module_added_at_its_start_leaves_every_output_of_the_base():
    base, rows, rng = _setup()
    model = _with_module(base)
    extra = _traffic(rng, 3, rows.present.shape[1])
    with torch.no_grad():
        for got, expected in zip(model(rows, extra), base(rows)):
            assert torch.equal(got, expected)
        past_a, past_b = base.no_past(3, 4), model.no_past(3, 4)
        for r in range(rows.present.shape[1]):                     # row by row, as a speaker extends the cache
            row = rows.between(r, r + 1)
            part = Traffic(extra.tokens[:, r:r + 1], extra.present[:, r:r + 1])
            h_a, _, past_a = base.extend(row, past_a)
            h_b, _, past_b = model.extend(row, past_b, part)
            assert torch.equal(h_a, h_b)


def test_without_other_aircraft_the_module_gives_zero_at_any_weights():
    base, rows, rng = _setup(seed=1)
    model = _with_module(base)
    _randomize(model)
    length = rows.present.shape[1]
    empty = _traffic(rng, 3, length, empty_rows=range(length))
    module = traffic_modules(model)[0]
    x = torch.randn(3, length, SMALL["d_model"])
    assert torch.equal(module(x, empty), torch.zeros_like(x))
    with torch.no_grad():
        for got, expected in zip(model(rows, empty), base(rows)):
            assert torch.equal(got, expected)
    # rows without traffic are untouched when other rows have some: the module adds nothing there
    some = _traffic(rng, 3, length, empty_rows=range(0, length, 2))
    out = module(x, some)
    assert torch.equal(out[:, ::2], torch.zeros_like(out[:, ::2])) and out[:, 1::2].abs().sum() > 0


def test_a_permutation_of_the_other_aircraft_changes_nothing():
    base, rows, rng = _setup(seed=2)
    model = _with_module(base)
    _randomize(model)
    extra = _traffic(rng, 3, rows.present.shape[1], most=5)
    order = torch.as_tensor(rng.permutation(extra.tokens.shape[2]))
    permuted = Traffic(extra.tokens[:, :, order], extra.present[:, :, order])
    with torch.no_grad():
        a, b = model(rows, extra), model(rows, permuted)
        changed = model(rows, Traffic(extra.tokens * 2.0, extra.present))
    for x, y in zip(a, b):
        finite = torch.isfinite(x)
        assert torch.equal(finite, torch.isfinite(y))
        assert torch.allclose(x[finite], y[finite], atol=1e-5, rtol=0.0)      # the summation order only
    assert any(not torch.equal(x, y) for x, y in zip(a, changed))             # and the tokens are read


def test_the_speaker_passes_the_traffic_and_says_the_bases_words_at_the_start():
    base, rows, rng = _setup(seed=3, count=2)
    model = _with_module(base)
    first, length = 8, rows.present.shape[1]
    extra = _traffic(rng, 2, length)
    numbers = np.random.default_rng(4).random((length - first, 2, 5))
    fin = [finals(airport())] * 2
    said = {}
    for name, m, x in (("base", base, None), ("module", model, extra)):
        speaker = Speaker(m, WORDS, fin, capacity=length)
        speaker.observe(rows.between(0, first), [Position(np.full(2, -15_000.0 + 140.0 * r), np.zeros(2),
                                                          np.full(2, 700.0 - 4.0 * r)) for r in range(first)],
                        None if x is None else Traffic(x.tokens[:, :first], x.present[:, :first]))
        out = []
        for r in range(first, length):
            at = Position(np.full(2, -15_000.0 + 140.0 * r), np.zeros(2), np.full(2, 700.0 - 4.0 * r))
            part = None if x is None else Traffic(x.tokens[:, r:r + 1], x.present[:, r:r + 1])
            out.append(speaker.speak(rows.between(r, r + 1), at, numbers[r - first], None, part))
        said[name] = (np.stack(out), np.stack(speaker.drawn_probability))
    assert np.array_equal(said["base"][0], said["module"][0])
    assert np.array_equal(said["base"][1], said["module"][1])


def test_the_traffic_modules_have_their_own_learning_rate():
    base, _, _ = _setup()
    model = _with_module(base)
    groups = parameter_groups(model, prior_lr=1e-4, traffic_lr=1e-3)
    assert [(g["name"], g["lr"]) for g in groups] == [("prior", 1e-4), ("traffic", 1e-3)]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert {id(p) for p in groups[1]["params"]} == {id(p) for m in traffic_modules(model) for p in m.parameters()}
    assert len(traffic_modules(model)) == SMALL["layers"]
    with pytest.raises(ValueError, match="traffic modules"):
        parameter_groups(base, 1e-4, 1e-3)


def test_the_traffic_input_is_padded_and_masked():
    tokens = [[np.ones((2, len(TOKEN_FEATURES)), np.float32), np.zeros((0, len(TOKEN_FEATURES)), np.float32)],
              [np.zeros((0, len(TOKEN_FEATURES)), np.float32), np.full((3, len(TOKEN_FEATURES)), 2.0, np.float32)]]
    traffic = traffic_of(tokens, CPU)
    assert traffic.tokens.shape == (2, 2, 3, len(TOKEN_FEATURES))
    assert traffic.present.tolist() == [[[True, True, False], [False, False, False]],
                                        [[False, False, False], [True, True, True]]]
    assert traffic_of([[np.zeros((0, len(TOKEN_FEATURES)), np.float32)]], CPU).tokens.shape == (1, 1, 1,
                                                                                                 len(TOKEN_FEATURES))
    with pytest.raises(ValueError, match="same number of rows"):
        traffic_of([tokens[0], tokens[1][:1]], CPU)
    with pytest.raises(ValueError, match="heads"):
        TrafficAttention(30, CONFIG, TrafficTokens(30, CONFIG), first=True)
    assert CONFIG.to_dict()["schema"] == TRAFFIC_ATTENTION_SCHEMA


def test_one_token_network_shared_by_the_layers_embeds_a_step_once():
    """D116: every layer's module holds the same token network; a forward pass embeds the tokens once (the first
    layer's module), and the layers after it read what it embedded."""
    base, rows, rng = _setup(seed=5)
    model = _with_module(base)
    _randomize(model)
    modules = traffic_modules(model)
    assert len(modules) == SMALL["layers"] > 1
    assert all(m.tokens is modules[0].tokens for m in modules) and [m.first for m in modules] == [True, False]
    calls = []
    modules[0].tokens.register_forward_hook(lambda module, inputs, output: calls.append(output.shape))
    extra = _traffic(rng, 3, rows.present.shape[1])
    with torch.no_grad():
        first = model(rows, extra)
        assert len(calls) == 1                                           # once for the two layers
        again = model(rows, extra)                                       # the same input again: embedded again, alike
    assert len(calls) == 2 and all(torch.equal(a, b) for a, b in zip(first, again))
    shared = {id(p) for p in modules[0].tokens.parameters()}
    counted = [id(p) for g in parameter_groups(model, 1e-4, 1e-3) for p in g["params"]]
    assert shared <= set(counted) and len(counted) == len(set(counted))  # the shared parameters once
