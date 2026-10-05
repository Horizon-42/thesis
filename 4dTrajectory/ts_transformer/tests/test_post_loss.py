"""Stage C, C7: the loss of the post-training (post-training §2 item 5, §8 C7; D36, D76)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import Words
from ts_transformer.post.edges import TOKEN_FEATURES
from ts_transformer.post.loss import (
    CLIP, PassStart, Samples, data_term, one_pass, pull_to_base, stacked, surrogate, update_loss,
)
from ts_transformer.post.traffic_attention import (
    Traffic, TrafficConfig, add_traffic_attention, parameter_groups, traffic_modules, traffic_of,
)
from ts_transformer.prior.batch import collate, target_classes
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.speaker import Position, Speaker
from ts_transformer.prior.train import batch_nll, masked_log_probability
from ts_transformer.tests.post_support import airport, finals
from ts_transformer.tests.support import instruction_spec, prior_sentence

CPU = torch.device("cpu")
SMALL = {"d_model": 32, "layers": 2, "heads": 4, "feedforward": 64}
WORDS = Words(instruction_spec())
FIRST, LENGTH, COUNT = 8, 26, 3


def _position(r):
    return Position(np.full(COUNT, -15_000.0 + 140.0 * r), np.zeros(COUNT), np.full(COUNT, 700.0 - 4.0 * r))


def _base(seed=0):
    torch.manual_seed(seed)
    return Prior(PriorConfig.from_words(WORDS, "full", **SMALL)).eval()


def _trained(base, seed=5):
    """The base with its traffic modules, their weights random (so that the traffic is read)."""
    model = copy.deepcopy(base)
    add_traffic_attention(model, TrafficConfig(hidden=16, heads=4))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in traffic_modules(model):
            for p in module.parameters():
                p.copy_(torch.randn(p.shape, generator=generator) * 0.2)
    return model.eval()


def _spoken(model, seed=1):
    """``model`` speaks COUNT synthetic sentences from their first predicted step, with traffic: the rows with the
    words said as targets, the speaker's records and its drawn probabilities, the traffic."""
    rng = np.random.default_rng(seed)
    sentences = [prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(COUNT)]
    rows = collate(sentences, CPU)
    tokens = [[rng.normal(size=(int(rng.integers(0, 4)), len(TOKEN_FEATURES))).astype(np.float32)
               for _ in range(LENGTH)] for _ in range(COUNT)]
    traffic = traffic_of(tokens, CPU)
    speaker = Speaker(model, WORDS, [finals(airport())] * COUNT, capacity=LENGTH)
    speaker.observe(rows.between(0, FIRST), [_position(r) for r in range(FIRST)],
                    Traffic(traffic.tokens[:, :FIRST], traffic.present[:, :FIRST]))
    numbers = np.random.default_rng(seed + 1).random((LENGTH - FIRST, COUNT, 5))
    said = np.stack([speaker.speak(rows.between(r, r + 1), _position(r), numbers[r - FIRST], None,
                                   Traffic(traffic.tokens[:, r:r + 1], traffic.present[:, r:r + 1]))
                     for r in range(FIRST, LENGTH)], axis=1)
    targets = rows.targets.clone()
    targets[:, FIRST:] = torch.as_tensor(target_classes(said))
    return rows._replace(targets=targets), speaker.permitted(), np.stack(speaker.drawn_probability, axis=1), traffic


def _samples(rows, permitted, traffic, advantage=None, counted=None):
    shape = rows.asked.shape
    advantage = torch.ones(shape) if advantage is None else advantage
    counted = rows.asked.clone() if counted is None else counted
    return Samples(rows, permitted, traffic, advantage, counted)


def test_at_the_parameters_that_spoke_the_ratio_is_one():
    model = _trained(_base())
    rows, permitted, drawn, traffic = _spoken(model)
    with torch.no_grad():
        log_p = masked_log_probability(model, rows, permitted, traffic)
        start = masked_log_probability(PassStart(model).model, rows, permitted, traffic)
    asked = rows.asked.numpy()
    assert torch.equal(log_p, start)                                        # the frozen copy is the model that spoke
    ratio = np.exp(log_p.numpy()[asked] - np.log(drawn.reshape(-1, 5)))     # against the speaker's own probabilities
    assert np.abs(ratio - 1.0).max() < 1e-5                                  # the float tolerance of §6.4
    loss, words, clipped = surrogate(log_p, start, torch.full(rows.asked.shape, 0.5), rows.asked)
    assert words == int(asked.sum()) * 5 and clipped == 0
    assert loss.item() == pytest.approx(-0.5 * 5)                            # r = 1: −A for each of a row's 5 words


def test_a_word_that_a_record_blocks_takes_no_probability():
    model = _trained(_base())
    rows, permitted, _, traffic = _spoken(model)
    found = [(column, *np.argwhere(~mask)[0]) for column, mask in enumerate(permitted.masks) if (~mask).any()]
    assert found, "some record blocks a word (the grammar or the procedure masks)"
    column, b, said_row, word = found[0]
    r = FIRST + int(said_row)
    targets = rows.targets.clone()
    targets[b, r, column] = int(word)
    with torch.no_grad():
        log_p = masked_log_probability(model, rows._replace(targets=targets), permitted, traffic)
    assert log_p[b, r, column].item() == float("-inf")


def test_with_the_module_at_zero_the_data_term_is_the_priors_teacher_forced_loss():
    base = _base()
    model = copy.deepcopy(base)
    add_traffic_attention(model, TrafficConfig(hidden=16, heads=4))      # at its start: output zero
    rng = np.random.default_rng(3)
    rows = collate([prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(4)], CPU)
    torch.manual_seed(11)
    got = data_term(model, rows)
    base.train()
    torch.manual_seed(11)                                                  # the same dropout masks
    nll, speaks = batch_nll(base, rows)
    base.eval()
    assert torch.equal(got, nll.sum() / speaks)
    assert not model.training                                              # back in eval mode
    _randomize = _trained(base)                                             # random module weights, no scene: zero
    torch.manual_seed(11)
    assert torch.equal(data_term(_randomize, rows), got)


def test_the_surrogate_clips_and_reads_only_the_counted_rows():
    log_start = torch.log(torch.full((2, 3, 5), 0.5))
    counted = torch.tensor([[False, True, True], [True, False, False]])
    advantage = torch.tensor([[9.0, 1.0, 1.0], [-1.0, 7.0, 7.0]])
    moved = log_start + torch.log(torch.tensor(1.5))                       # every ratio 1.5: past the clip
    loss, words, clipped = surrogate(moved, log_start, advantage, counted)
    assert words == 3 * 5 and clipped == 15
    # sample 0 (2 rows): A = 1 > 0, the clip caps r at 1.2; sample 1 (1 row): A = −1 < 0, min(−1.5, −1.2) = −1.5 — the
    # batch's sum over its 3 counted rows, divided by 3 (D115)
    assert loss.item() == pytest.approx((2 * -(1.0 + CLIP) * 5 + 1.5 * 5) / 3)
    log_p = log_start.clone().requires_grad_(True)
    surrogate(log_p, log_start, advantage, counted)[0].backward()
    assert (log_p.grad[~counted] == 0).all() and (log_p.grad[counted] != 0).all()


def test_the_pull_to_the_base_is_zero_at_the_base_and_grows_away_from_it():
    log_b = torch.log(torch.full((2, 3, 5), 0.25))
    counted = torch.ones(2, 3, dtype=torch.bool)
    assert pull_to_base(log_b, log_b, counted).item() == 0.0
    assert pull_to_base(log_b + 0.3, log_b, counted).item() > pull_to_base(log_b + 0.1, log_b, counted).item() > 0.0


def test_one_update_and_one_pass():
    base = _base()
    model = _trained(base)
    rows, permitted, _, traffic = _spoken(model)
    rng = np.random.default_rng(9)
    data = collate([prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(4)], CPU)
    advantage = torch.as_tensor(np.random.default_rng(2).normal(size=rows.asked.shape), dtype=torch.float32)
    samples = _samples(rows, permitted, traffic, advantage)
    parts = update_loss(model, PassStart(model), base, samples, data)
    assert parts.kl.item() > 0.0                                            # the module moved it from the base
    assert parts.clipped == 0 and torch.isfinite(parts.loss)
    optimizer = torch.optim.AdamW(parameter_groups(model, prior_lr=1e-4, traffic_lr=1e-3))
    before = [p.detach().clone() for p in model.parameters()]
    passed = one_pass(model, base, optimizer, [samples, samples], [data, data])
    assert len(passed) == 2 and passed[1].clipped >= 0
    assert any(not torch.equal(a, p) for a, p in zip(before, model.parameters()))
    assert not model.training
    summary = stacked(passed)
    assert summary["words"] == 2 * parts.words and 0.0 <= summary["clipped_share"] <= 1.0
    after = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="one for each"):
        one_pass(model, base, optimizer, [samples], [data, data])          # one data batch for each update
    assert all(torch.equal(a, p) for a, p in zip(after, model.parameters()))   # refused before any update
    nan = torch.zeros(1, 2, 5)
    nan[0, 0] = float("nan")                                                # a row not counted that holds a NaN
    assert torch.isfinite(pull_to_base(nan, torch.zeros(1, 2, 5), torch.tensor([[False, True]])))


def test_samples_refuse_rows_that_are_not_said_or_carry_nothing():
    model = _trained(_base())
    rows, permitted, _, traffic = _spoken(model)
    early = rows.asked.clone()
    early[:, 0] = True
    with pytest.raises(ValueError, match="said"):
        _samples(rows, permitted, traffic, counted=early)
    none = rows.asked.clone()
    none[0] = False
    with pytest.raises(ValueError, match="counts some rows"):
        _samples(rows, permitted, traffic, counted=none)
    with pytest.raises(ValueError, match="beside the rows"):
        _samples(rows, permitted, traffic, advantage=torch.ones(1, 1))


def test_the_surrogate_and_the_kl_refuse_a_model_in_training_mode_and_the_data_term_runs_in_it():
    """D107: `masked_log_probability` refuses a model with any module in training mode; `update_loss` scores the words
    in eval mode (a model left in training mode is put there), refuses a base in training mode, and its data term runs
    with dropout on."""
    base = _base()
    model = _trained(base)
    rows, permitted, _, traffic = _spoken(model)
    traffic_modules(model)[0].train()                             # one module in training mode, the rest in eval
    with pytest.raises(ValueError, match="training"):
        masked_log_probability(model, rows, permitted, traffic)
    rng = np.random.default_rng(9)
    data = collate([prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(4)], CPU)
    start = PassStart(_trained(base))                             # the model that spoke (its weights), in eval mode
    model.train()                                                 # left in training mode: the loss puts it in eval
    parts = update_loss(model, start, base, _samples(rows, permitted, traffic), data)
    assert torch.isfinite(parts.loss) and parts.clipped == 0 and not model.training
    base.train()
    with pytest.raises(ValueError, match="training"):
        update_loss(model, start, base, _samples(rows, permitted, traffic), data)
    base.eval()
    torch.manual_seed(1)
    first = data_term(model, data)
    torch.manual_seed(2)
    assert not torch.equal(first, data_term(model, data))         # dropout on: another seed, another loss



def test_every_counted_row_weighs_the_same_in_the_surrogate_and_the_pull():
    """D115: a sample of one counted row and one of three — each row weighs a quarter, not a sample a half."""
    log_start = torch.log(torch.full((2, 3, 5), 0.5))
    counted = torch.tensor([[True, False, False], [True, True, True]])
    advantage = torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])            # only the short sample's row carries one
    loss, words, _ = surrogate(log_start, log_start, advantage, counted)
    assert words == 4 * 5 and loss.item() == pytest.approx(-4.0 * 5 / 4)
    gap = torch.zeros(2, 3, 5)
    gap[0, 0] = 0.3                                                          # the pull on the short sample's row only
    expected = (5 * (np.exp(-0.3) + 0.3 - 1.0)) / 4
    assert pull_to_base(log_start + gap, log_start, counted).item() == pytest.approx(expected)
