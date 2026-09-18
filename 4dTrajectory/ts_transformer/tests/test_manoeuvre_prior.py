"""The prior (`manoeuvre/prior.py`): the batch carries the next-code and landed targets at the
right positions, the model is causal, both variants train under one loss skeleton, decoding
reads the last valid position, and the bigram baseline scores what the prior must beat."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ts_transformer.manoeuvre import prior as pr
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.sequences import CodeSequence

K, Z = 16, 2


def _sequence(codes: list[int], *, landed_fraction: float = 0.3, typecode: str = "A320", key: str = "f") -> CodeSequence:
    count = len(codes)
    rng = np.random.default_rng(len(key) + count)
    return CodeSequence(
        dataset_id=f"KRDU:{key}", flight_id=key, anchor=59, segment_s=60.0,
        start_times=118.0 + 60.0 * np.arange(count), codes=np.asarray(codes, dtype=np.int64),
        z=rng.uniform(-1, 1, size=(count, Z)).astype(np.float32),
        states=rng.normal(size=(count + 1, 6)).astype(np.float32), landed_fraction=landed_fraction,
        typecode=typecode, runway_course_rad=0.8,
    )


def _config(**overrides) -> pr.PriorConfig:
    settings = dict(code_count=K, z_dim=Z, type_count=3, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.0, max_positions=8)
    settings.update(overrides)
    return pr.PriorConfig(**settings)


VOCAB = TypeVocabulary.from_typecodes(["A320", "B738"])


def test_collate_puts_the_next_code_and_the_landing_at_the_right_positions():
    batch = pr.collate([_sequence([3, 5, 7], key="a"), _sequence([1], landed_fraction=0.9, typecode="C56X", key="b")], _config(), VOCAB)
    assert batch.codes_in.tolist() == [[K, 3, 5, 7], [K, 1, K, K]]          # BOS then the codes; padding is BOS too
    assert batch.valid.tolist() == [[True] * 4, [True, True, False, False]]
    assert batch.next_code.tolist() == [[3, 5, 7, pr.IGNORE], [1, pr.IGNORE, pr.IGNORE, pr.IGNORE]]
    assert batch.landed.tolist() == [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
    assert batch.landed_fraction.tolist() == pytest.approx([0.3, 0.9])
    assert batch.type_index.tolist() == [1, 0] and batch.runway.shape == (2, 2)
    with pytest.raises(ValueError, match="continuous targets"):
        pr.collate([_sequence([1])], _config(continuous=True), VOCAB)
    with pytest.raises(ValueError, match="continuous targets"):
        pr.collate([_sequence([1])], _config(), VOCAB, continuous_targets=[np.zeros((1, Z))])
    with pytest.raises(ValueError, match="segments"):
        pr.collate([_sequence(list(range(9)))], _config(), VOCAB)


@pytest.mark.parametrize("continuous", [False, True])
def test_the_prior_is_causal_and_trains_under_one_loss_skeleton(continuous):
    torch.manual_seed(0)
    config = _config(continuous=continuous)
    model = pr.ManoeuvrePrior(config).eval()
    sequences = [_sequence([3, 5, 7, 2], key="a"), _sequence([1, 1], key="b")]
    targets = [seq.z for seq in sequences] if continuous else None
    batch = pr.collate(sequences, config, VOCAB, continuous_targets=targets)
    output = model(batch)
    head = output.next_z if continuous else output.next_logits
    assert head.shape == (2, 5, Z if continuous else K) and output.landed_logit.shape == (2, 5)
    # causal: changing a LATER code leaves every earlier position's output unchanged
    changed = pr.collate([_sequence([3, 5, 9, 2], key="a"), sequences[1]], config, VOCAB, continuous_targets=targets)
    other = model(changed)
    other_head = other.next_z if continuous else other.next_logits
    assert torch.allclose(head[0, :3], other_head[0, :3], atol=1e-6) and not torch.allclose(head[0, 3], other_head[0, 3])
    # padding does not leak: flight b's outputs are the same alone or beside a longer flight
    alone = model(pr.collate([sequences[1]], config, VOCAB, continuous_targets=None if targets is None else [targets[1]]))
    alone_head = alone.next_z if continuous else alone.next_logits
    assert torch.allclose(head[1, :3], alone_head[0, :3], atol=1e-5)
    # the loss has the three parts and a few steps of Adam lower it
    model.train()
    terms = model.loss(output, batch)
    assert set(terms) == {"total", "next", "landed", "landed_fraction"} and terms["total"].requires_grad
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = float(terms["total"])
    for _ in range(40):
        optimizer.zero_grad()
        loss = model.loss(model(batch), batch)["total"]
        loss.backward()
        optimizer.step()
    assert float(loss) < first * 0.5
    if not continuous:
        assert model.eval().next_code_accuracy(model(batch), batch) >= 0.5


def test_decoding_reads_the_last_valid_position():
    torch.manual_seed(1)
    config = _config()
    model = pr.ManoeuvrePrior(config).eval()
    batch = pr.collate([_sequence([3, 5, 7], key="a"), _sequence([1], key="b")], config, VOCAB)
    step = pr.last_position_step(model, batch)
    output = model(batch)
    assert step.code.shape == (2,) and step.probabilities.shape == (2, K)
    assert torch.equal(step.code, torch.stack((output.next_logits[0, 3].argmax(), output.next_logits[1, 1].argmax())))
    assert torch.allclose(step.landed_probability, torch.sigmoid(torch.stack((output.landed_logit[0, 3], output.landed_logit[1, 1]))))
    continuous = pr.ManoeuvrePrior(_config(continuous=True)).eval()
    cbatch = pr.collate([_sequence([3, 5], key="a")], _config(continuous=True), VOCAB, continuous_targets=[np.zeros((2, Z), dtype=np.float32)])
    cstep = pr.last_position_step(continuous, cbatch)
    assert cstep.z.shape == (1, Z) and cstep.code is None
    with pytest.raises(ValueError, match="continuous"):
        continuous.next_code_accuracy(continuous(cbatch), cbatch)


def test_the_bigram_baseline_scores_every_transition_including_the_landing():
    train = [_sequence([0, 1, 2], key=f"t{i}") for i in range(20)] + [_sequence([0, 1], key=f"u{i}") for i in range(5)]
    val = [_sequence([0, 1, 2], key="v")]
    baseline = pr.bigram_nll(train, val, code_count=4, alpha=1.0)
    assert baseline["tokens"] == 4 and baseline["code_tokens"] == 3
    # the JOINT (rows over {codes, LANDED}): BOS→0 26/30 ; 0→1 26/30 ; 1→2 21/30 ; 2→LANDED 21/25
    expected = -(math.log(26 / 30) * 2 + math.log(21 / 30) + math.log(21 / 25))
    assert baseline["nll_per_token"] == pytest.approx(expected / 4)
    # the CONDITIONAL next code (the four code columns renormalised, the landing apart — the
    # prior's own quantity): BOS→0 26/29 ; 0→1 26/29 ; 1→2 21/24
    assert baseline["nll_per_code"] == pytest.approx(-(math.log(26 / 29) * 2 + math.log(21 / 24)) / 3)
    # an unseen transition is smoothed, never infinite
    assert math.isfinite(pr.bigram_nll(train, [_sequence([3, 3], key="w")], code_count=4)["nll_per_token"])


def test_config_round_trips_and_refuses_a_bad_shape():
    config = _config()
    assert pr.PriorConfig.from_dict(config.to_dict()) == config and config.bos == K
    with pytest.raises(ValueError, match="n_heads"):
        _config(d_model=30)


@pytest.mark.parametrize("continuous", [False, True])
def test_fit_keeps_the_epoch_with_the_best_val_next_term_and_stops_early(continuous):
    torch.manual_seed(2)
    config = _config(continuous=continuous)
    rng = np.random.default_rng(0)
    train = [_sequence(list(rng.integers(0, 4, size=3)), key=f"t{i}") for i in range(24)]
    val = [_sequence(list(rng.integers(0, 4, size=3)), key=f"v{i}") for i in range(8)]
    targets = (lambda seqs: [s.z for s in seqs]) if continuous else (lambda seqs: None)
    rows = []
    result = pr.fit(pr.ManoeuvrePrior(config), train, val, VOCAB, train_targets=targets(train), val_targets=targets(val),
                    epochs=30, patience=5, batch_size=8, learning_rate=3e-3, seed=1, device=torch.device("cpu"), log=rows.append)
    assert len(result.history) == len(rows) <= 30 and result.best_epoch <= len(result.history)
    assert result.best_val_next == pytest.approx(min(row["val"]["next"] for row in result.history))
    assert set(result.history[0]["train"]) == {"total", "next", "landed", "landed_fraction"}
    val_keys = {"next", "landed", "landed_fraction", "total", "landed_accuracy"} | (set() if continuous else {"next_code_accuracy"})
    assert set(result.history[0]["val"]) == val_keys
    # the kept weights reproduce the kept epoch's val number
    model = pr.ManoeuvrePrior(config)
    model.load_state_dict(result.state_dict)
    again = pr.evaluate(model, val, targets(val), VOCAB, batch_size=8, device=torch.device("cpu"))
    assert again["next"] == pytest.approx(result.best_val_next, abs=1e-6)


def test_the_flip_rate_counts_disagreements_between_adjacent_asks_on_the_same_segment():
    torch.manual_seed(3)
    config = _config()
    model = pr.ManoeuvrePrior(config).eval()
    sequences = [_sequence([3, 5, 7, 2], key="a"), _sequence([1], key="b"), _sequence([], key="c")]
    result = pr.flip_rate(model, sequences, VOCAB, batch_size=2, device=torch.device("cpu"))
    # flight a has three (t, t+2) pairs (c_2, c_3, c_4); b and c have none
    assert result["pairs"] == 3 and 0 <= result["flips"] <= 3 and result["rate"] == result["flips"] / 3
    # a prior whose top-1 for c_{t+1} IS the truth's cannot flip: overfit one flight and check
    batch = pr.collate([sequences[0]], config, VOCAB)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        loss = model.loss(model(batch), batch)["total"]
        loss.backward()
        optimizer.step()
    model.eval()
    if model.next_code_accuracy(model(batch), batch) == 1.0:
        assert pr.flip_rate(model, [sequences[0]], VOCAB, batch_size=1, device=torch.device("cpu"))["flips"] == 0
    with pytest.raises(ValueError, match="continuous"):
        pr.flip_rate(pr.ManoeuvrePrior(_config(continuous=True)), sequences, VOCAB, batch_size=2, device=torch.device("cpu"))


def test_the_flip_rate_replaces_exactly_one_code_per_ask_and_matches_a_brute_force_loop():
    torch.manual_seed(5)
    config = _config()
    model = pr.ManoeuvrePrior(config).eval()
    rng = np.random.default_rng(1)
    sequences = [_sequence(list(rng.integers(0, K, size=n)), key=f"f{i}") for i, n in enumerate((5, 3, 6, 1))]
    result = pr.flip_rate(model, sequences, VOCAB, batch_size=4, device=torch.device("cpu"))
    # brute force: one flight at a time, one position at a time, ONLY that position's code replaced
    flips = pairs = 0
    with torch.no_grad():
        for sequence in sequences:
            batch = pr.collate([sequence], config, VOCAB)
            own = model(batch).next_logits.argmax(dim=-1)[0]
            for t in range(sequence.length - 1):           # c_{t+2} exists for t ≤ T-2
                codes = batch.codes_in.clone()
                codes[0, t + 1] = own[t]
                assert int((codes != batch.codes_in).sum()) <= 1
                rolled = pr.PriorBatch(codes, batch.states, batch.valid, batch.next_code, batch.landed, batch.landed_fraction,
                                       batch.next_z, batch.type_index, batch.runway)
                ahead = model(rolled).next_logits.argmax(dim=-1)[0, t + 1]
                flips += int(ahead != own[t + 1])
                pairs += 1
    assert result["pairs"] == pairs == 4 + 2 + 5 + 0 and result["flips"] == flips


def test_a_batch_of_zero_segment_flights_has_a_finite_loss_and_evaluate_refuses_nothing():
    config = _config()
    model = pr.ManoeuvrePrior(config)
    batch = pr.collate([_sequence([], key="a"), _sequence([], key="b")], config, VOCAB)
    terms = model.loss(model(batch), batch)
    assert all(torch.isfinite(value) for value in terms.values()) and float(terms["next"]) == 0.0
    with pytest.raises(ValueError, match="at least one"):
        pr.evaluate(model, [], None, VOCAB, batch_size=2, device=torch.device("cpu"))


def test_a_rolled_sequence_feeds_the_flown_codes_and_targets_the_truths_by_time_index():
    """Plan §2.7 step 1: the input is what was flown, the label is the truth's code at the same
    absolute time; where the truth has no full segment left the label is the landing."""
    from ts_transformer.manoeuvre.sequences import rolled_sequence
    truth = _sequence([3, 5, 7], key="a")                                  # T = 3
    flown_states = np.random.default_rng(0).normal(size=(6, 6)).astype(np.float32)
    # the executor flew FIVE rounds (past the truth's three): the rolled sequence keeps T + 1 = 4
    rolled = rolled_sequence(truth, [9, 9, 1, 2, 4], flown_states)
    assert rolled.rolled and rolled.length == 4 and rolled.codes.tolist() == [9, 9, 1, 2]
    assert rolled.targets.tolist() == [3, 5, 7] and rolled.states.shape == (5, 6)
    batch = pr.collate([rolled, truth], _config(), VOCAB)
    assert batch.codes_in[0].tolist() == [K, 9, 9, 1, 2]                  # BOS + the flown codes
    assert batch.next_code[0].tolist() == [3, 5, 7, pr.IGNORE, pr.IGNORE]  # the truth's, by time index
    assert batch.landed[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]           # landed from position T on
    assert batch.next_code[1].tolist() == [3, 5, 7, pr.IGNORE, pr.IGNORE] and batch.landed[1].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    # a shorter flown history keeps every flown position and still lands after the truth's end
    short = rolled_sequence(truth, [9], flown_states[:2])
    short_batch = pr.collate([short], _config(), VOCAB)
    assert short_batch.next_code[0].tolist() == [3, 5] and short_batch.landed[0].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="boundary states"):
        rolled_sequence(truth, [9, 9], flown_states[:2])
