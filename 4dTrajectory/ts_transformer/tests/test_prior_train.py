"""Training the prior (prior design §5; milestone B3's loop, on synthetic sentences): the loss, the batches, the stop on
the select days, and a fold that never reads its held-out airport in training."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
import torch
from torch import nn

from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.prior import train as training
from ts_transformer.prior.batch import collate
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.runs import Run, RunData, held_out_sentences, run_data
from ts_transformer.prior.train import TrainConfig, column_nll, evaluate, length_groups, train
from ts_transformer.tests.support import instruction_spec, prior_sentence

CPU = torch.device("cpu")
SMALL = {"d_model": 32, "layers": 1, "heads": 4, "feedforward": 64, "dropout": 0.0}
AIRPORTS = ("KAAA", "KBBB", "KCCC")


@pytest.fixture(scope="module")
def words():
    return Words(instruction_spec())


class RecordingSource:
    """A `runs.SentenceSource` of synthetic sentences that records every split and airport it is asked for."""

    def __init__(self, words, per_airport=4):
        self.words, self.per_airport, self.asked = words, per_airport, []

    def sentences(self, split, airport):
        self.asked.append((split, airport))
        rng = np.random.default_rng(zlib.crc32(f"{split}:{airport}".encode()))
        return [prior_sentence(rng, self.words, airport=airport, split=split, rows=int(rng.integers(15, 40)))
                for _ in range(self.per_airport)]


def test_a_run_names_its_training_airports():
    assert Run(AIRPORTS).training_airports == AIRPORTS
    assert Run(AIRPORTS, "KBBB").training_airports == ("KAAA", "KCCC")
    with pytest.raises(ValueError, match="not one of"):
        Run(AIRPORTS, "KZZZ")
    with pytest.raises(ValueError, match="training airport"):
        Run(("KAAA",), "KAAA")


def test_a_fold_reads_only_the_train_and_select_days_of_its_training_airports(words):
    source = RecordingSource(words)
    run = Run(AIRPORTS, held_out="KBBB")
    data = run_data(source, run)
    assert sorted(source.asked) == [("select", "KAAA"), ("select", "KCCC"), ("train", "KAAA"), ("train", "KCCC")]
    assert {s.airport for s in data.train + data.select} == {"KAAA", "KCCC"}
    source.asked.clear()
    held_out = held_out_sentences(source, run)
    assert source.asked == [("select", "KBBB")] and {s.airport for s in held_out} == {"KBBB"}
    with pytest.raises(ValueError, match="no held-out airport"):
        held_out_sentences(source, Run(AIRPORTS))


def test_the_run_data_refuses_a_sentence_of_another_airport_or_split(words):
    source = RecordingSource(words)
    run = Run(AIRPORTS, held_out="KBBB")
    train_days, select_days = source.sentences("train", "KAAA"), source.sentences("select", "KAAA")
    with pytest.raises(ValueError, match="train sentences are not"):
        RunData(run, train_days + source.sentences("train", "KBBB"), select_days)
    with pytest.raises(ValueError, match="select sentences are not"):
        RunData(run, train_days, select_days + source.sentences("val", "KAAA"))
    with pytest.raises(ValueError, match="no select sentence"):
        RunData(run, train_days, [])


def test_the_batches_hold_every_sentence_once_within_the_padded_rows():
    lengths = [5, 30, 12, 12, 7, 100, 3]
    groups = length_groups(lengths, 40, np.random.default_rng(0))
    assert sorted(i for group in groups for i in group) == list(range(len(lengths)))
    for group in groups:
        assert len(group) == 1 or max(lengths[i] for i in group) * len(group) <= 40
    assert length_groups(lengths, 40, None) == length_groups(lengths, 40, None)
    # each epoch draws new groups: sentences of one length are not always grouped with the same others
    same = [6] * 12
    rng = np.random.default_rng(1)
    draws = {tuple(sorted(tuple(sorted(g)) for g in length_groups(same, 18, rng))) for _ in range(5)}
    assert len(draws) > 1


def test_the_loss_counts_only_the_asked_rows(words):
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL)).eval()
    rng = np.random.default_rng(3)
    group = [prior_sentence(rng, words, rows=n) for n in (12, 30)]
    rows = collate(group, CPU)
    with torch.no_grad():
        logits = model(rows)
        nll = column_nll(logits, rows)
        expected = torch.zeros(len(COLUMNS))
        for b, sentence in enumerate(group):
            for r in range(sentence.first_step, sentence.rows):
                for c, logit in enumerate(logits):
                    expected[c] -= torch.log_softmax(logit[b, r], -1)[rows.targets[b, r, c]]
    torch.testing.assert_close(nll, expected, rtol=1e-5, atol=1e-4)
    # the per-step loss does not depend on the batches it is summed over
    small, large = evaluate(model, group, 30, CPU), evaluate(model, group, 10_000, CPU)
    assert small["steps"] == large["steps"] == sum(s.rows - s.first_step for s in group)
    assert small["loss_per_step"] == pytest.approx(large["loss_per_step"], rel=1e-6)


def test_the_loop_learns_and_keeps_the_best_select_epoch(words):
    data = run_data(RecordingSource(words, per_airport=6), Run(AIRPORTS))
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    config = TrainConfig(tokens_per_batch=200, learning_rate=3e-3, warmup_steps=5, max_epochs=6, patience=6)
    lines = []
    result = train(model, data, config, CPU, lines.append)
    history = result.history
    assert len(history) == 6 and len(lines) == 6
    assert history[-1]["train_loss_per_step"] < 0.8 * history[0]["train_loss_per_step"]
    losses = [row["select_loss_per_step"] for row in history]
    assert result.best_epoch == 1 + int(np.argmin(losses))
    model.load_state_dict(result.state)
    assert evaluate(model, data.select, 10_000, CPU)["loss_per_step"] == pytest.approx(min(losses), rel=1e-6)


def test_the_stop_reads_only_the_select_days_and_waits_patience_epochs(words, monkeypatch):
    data = run_data(RecordingSource(words), Run(AIRPORTS, held_out="KCCC"))
    read = []
    real = training.evaluate

    def recording(model, sentences, tokens, device):
        read.append([s.flight_key for s in sentences])
        return real(model, sentences, tokens, device)

    monkeypatch.setattr(training, "evaluate", recording)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    # no learning: the select loss never improves after the first epoch
    config = TrainConfig(tokens_per_batch=400, learning_rate=0.0, warmup_steps=1, max_epochs=30, patience=3)
    result = train(model, data, config, CPU, lambda line: None)
    assert len(result.history) == 1 + 3 and result.best_epoch == 1
    assert read == [[s.flight_key for s in data.select]] * 4


def test_the_state_kept_is_the_best_select_epochs_not_the_last(words, monkeypatch):
    """The select losses scripted: epoch 2 is the best, epochs 3–5 worse; the state returned is the weights after epoch
    2, not those the model ends with."""
    data = run_data(RecordingSource(words), Run(AIRPORTS))
    scripted = iter([5.0, 3.0, 4.0, 4.5, 6.0])
    weights = []

    def scripted_evaluate(model, sentences, tokens, device):
        weights.append({name: value.clone() for name, value in model.state_dict().items()})
        return {"loss_per_step": next(scripted), "per_column": {}, "steps": 1}

    monkeypatch.setattr(training, "evaluate", scripted_evaluate)
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    config = TrainConfig(tokens_per_batch=400, learning_rate=1e-2, warmup_steps=1, max_epochs=30, patience=3)
    result = train(model, data, config, CPU, lambda line: None)
    assert result.best_epoch == 2 and len(result.history) == 5
    final = model.state_dict()
    assert any(not torch.equal(result.state[name], final[name]) for name in final)
    assert all(torch.equal(result.state[name], weights[1][name]) for name in final)


def test_a_non_finite_loss_stops_the_training(words):
    data = run_data(RecordingSource(words), Run(AIRPORTS))
    torch.manual_seed(0)
    model = Prior(PriorConfig.from_words(words, "full", **SMALL))
    with torch.no_grad():
        nn.init.constant_(model.heads["speed"].weight, float("nan"))
    with pytest.raises(FloatingPointError, match="epoch 1"):
        train(model, data, TrainConfig(tokens_per_batch=400, max_epochs=2), CPU, lambda line: None)
