"""The instruction sequences and the instruction prior (two-tier v3 stage B, B′-dev3): the
sentence with states and next-word targets, a rolled sequence's flown states and truth targets,
the batch, the causal factorised model, its loss, the readings (top-k, flips, misses, joint
coverage, landing) and the hold / bigram baselines. Untrained on synthetic data: mechanics
only, never a number."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, PREDICTION_CONTROL, TSConfig, recipe_settings
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import instruction_prior as pr
from ts_transformer.manoeuvre import instruction_sequences as sq
from ts_transformer.manoeuvre import instructions as ins
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.tests.support import AIRPORT, RUNWAY


def _config() -> TSConfig:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(prediction_output=PREDICTION_CONTROL, control_horizon_s=20.0, n_segments=2, control_imitation_loss_weight=0.0,
                         final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
                         e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1))
    return TSConfig(**settings)


#: The cohort lands on TWO thresholds: the runway word (D62) is a real column here, not a
#: constant every flight shares, so the head's classes and `shift_words` are exercised.
OTHER_RUNWAY = "23R"
RUNWAYS = ins.RunwayVocabulary.from_idents([RUNWAY, OTHER_RUNWAY])


@pytest.fixture(scope="module")
def world():
    vocabulary = ins.Vocabulary()
    config_ts = _config()
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=5) + synthetic_arrivals(AIRPORT, OTHER_RUNWAY, n_flights=2, seed=5)
    series, _report = build_series(flights, config_ts, airport=AIRPORT)
    assert {ins.flight_runway(item) for item in series} == set(RUNWAYS.idents)
    readings = [ins.read_instructions(item, vocabulary, RUNWAYS) for item in series]
    sequences = [sq.flight_sequence(item, reading) for item, reading in zip(series, readings, strict=True)]
    types = TypeVocabulary.from_typecodes(item.typecode for item in sequences)
    config = pr.PriorConfig(words=ins.word_counts(vocabulary, RUNWAYS), type_count=types.size, vocabulary_sha256=vocabulary.sha256,
                            d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, max_positions=64)
    return vocabulary, series, readings, sequences, types, config


def test_the_truth_sequence_carries_the_words_the_states_and_the_next_words(world):
    _vocabulary, series, readings, sequences, _types, _config = world
    item, reading, sequence = series[0], readings[0], sequences[0]
    assert sequence.length == len(reading.positions_s) and sequence.words.shape == (sequence.length, 5)   # five kinds since D62
    assert np.array_equal(sequence.targets[:-1], reading.words[1:]) and np.array_equal(sequence.targets[-1], reading.words[-1])
    assert sequence.states.shape == (sequence.length, len(sq.STATE_TOKEN_FEATURES))
    first = sq.state_token(item.values[0], item.target_chart)
    assert np.allclose(sequence.states[0], first, atol=1e-6) and abs(first[4] ** 2 + first[5] ** 2 - 1.0) < 1e-5
    assert sequence.typecode == str(item.scenario.aircraft.code) and sequence.ends_at_landing
    # the position is relative to the threshold, whatever the chart's origin
    shifted = np.array(item.values[0], dtype=np.float64)
    shifted[:3] += 5000.0
    assert np.allclose(sq.state_token(shifted, np.asarray(item.target_chart) + 5000.0), first, atol=1e-6)
    still = np.array(item.values[0], dtype=np.float64)
    still[3:5] = 0.0
    with pytest.raises(ValueError, match="has no course"):
        sq.state_token(still, item.target_chart)
    with pytest.raises(ValueError, match="is not series"):
        sq.flight_sequence(series[1], reading)


def test_a_rolled_sequence_reads_flown_states_and_targets_the_truth_s_next_words(world):
    _vocabulary, series, readings, sequences, _types, _config = world
    item, reading, truth = series[0], readings[0], sequences[0]
    said = truth.words[:5]
    rolled = sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values, said, item.target_chart)
    assert rolled.length == 5 and np.array_equal(rolled.words, said) and np.allclose(rolled.states, truth.states[:5])
    assert np.array_equal(rolled.targets, truth.targets[:5])                    # on the truth's own path the targets agree
    assert not rolled.ends_at_landing                                           # a prefix: its last position has a next
    whole = sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values, truth.words, item.target_chart)
    assert whole.ends_at_landing
    with pytest.raises(ValueError, match="a rolled prefix covers"):
        sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values, np.zeros((truth.length + 1, 5), dtype=np.int64), item.target_chart)
    with pytest.raises(ValueError, match="is not sequence"):
        sq.rolled_sequence(truth, readings[1], item.supervision_times, item.supervision_values, said, item.target_chart)


def test_the_batch_shifts_the_intercept_and_marks_the_next_and_the_landing(world):
    _vocabulary, _series, _readings, sequences, types, config = world
    batch = pr.collate(sequences[:2], config, types)
    longest = max(item.length for item in sequences[:2])
    assert batch.words.shape == (2, longest, 5) and batch.targets.shape == (2, longest, 5) and batch.states.shape == (2, longest, 6)
    for row, item in enumerate(sequences[:2]):
        n = item.length
        assert bool(batch.valid[row, :n].all()) and not bool(batch.valid[row, n:].any())
        assert batch.words[row, :n, 3].min() >= 0                                     # NO_INTERCEPT → row 0
        assert np.array_equal(batch.words[row, :n].numpy(), pr.shift_words(item.words))
        assert np.array_equal(pr.unshift_words(batch.words[row, :n].numpy()), item.words)
        # the RUNWAY column is never shifted: its word is already ≥ 0 and indexes its table
        assert np.array_equal(batch.words[row, :n, 4].numpy(), item.words[:, 4])
        assert batch.words[row, :n, 4].max() < config.classes("runway")
        assert bool((batch.targets[row, n - 1] == pr.IGNORE).all()) and batch.landed[row, n - 1] == 1.0 and batch.landed[row, : n - 1].sum() == 0
        assert bool(batch.has_next[row, : n - 1].all()) and not bool(batch.has_next[row, n - 1 :].any())
    assert batch.runway.shape == (2, 2) and batch.type_index.shape == (2,)
    # a rolled PREFIX: its last position has a next word and no landing
    prefix = sq.InstructionSequence(**{**sequences[0].__dict__, "positions_s": sequences[0].positions_s[:4], "words": sequences[0].words[:4],
                                       "states": sequences[0].states[:4], "targets": sequences[0].targets[:4], "ends_at_landing": False})
    batch = pr.collate([prefix], config, types)
    assert bool(batch.has_next[0].all()) and batch.landed.sum() == 0 and np.array_equal(batch.targets[0].numpy(), pr.shift_words(prefix.targets))
    with pytest.raises(ValueError, match="the prior holds"):
        pr.collate(sequences[:1], pr.PriorConfig(**{**config.to_dict(), "max_positions": 3}), types)


def test_the_model_is_causal_and_factorised_and_the_loss_has_one_term_per_kind(world):
    _vocabulary, _series, _readings, sequences, types, config = world
    torch.manual_seed(0)
    model = pr.InstructionPrior(config).eval()
    batch = pr.collate(sequences[:2], config, types)
    output = model(batch)
    for kind in ins.INSTRUCTION_KINDS:
        assert output.logits[kind].shape == (2, batch.words.shape[1], config.classes(kind))
    assert output.landed_logit.shape == (2, batch.words.shape[1])
    terms = model.loss(output, batch)
    assert set(terms) == {*ins.INSTRUCTION_KINDS, "landed", "next", "total"} and torch.isfinite(terms["total"])
    assert torch.isclose(terms["next"], sum(terms[k] for k in ins.INSTRUCTION_KINDS))
    # causality: perturbing the input at position 4 leaves every logit before it unchanged
    words = batch.words.clone()
    words[:, 4, 0] = (words[:, 4, 0] + 1) % config.classes("heading")
    later = model(pr.PriorBatch(words, batch.states, batch.valid, batch.targets, batch.landed, batch.type_index, batch.runway))
    assert torch.allclose(later.logits["speed"][:, :4], output.logits["speed"][:, :4], atol=1e-5)
    assert not torch.allclose(later.logits["speed"][:, 4:6], output.logits["speed"][:, 4:6], atol=1e-5)


def test_the_joint_rank_is_the_truth_tuple_s_place_among_the_product_candidates(world):
    kinds = ins.INSTRUCTION_KINDS
    classes = {kind: world[5].classes(kind) for kind in kinds}
    targets = torch.tensor([[[3, 2, 5, 0, 1], [pr.IGNORE] * 5]])                       # one position with a next
    has_next = targets[..., 0] != pr.IGNORE
    logits = {kind: torch.zeros(1, 2, classes[kind]) for kind in kinds}
    for column, kind in enumerate(kinds):
        logits[kind][0, 0, targets[0, 0, column]] = 5.0                              # the truth is every kind's argmax
    assert pr._joint_ranks(logits, targets, has_next).tolist() == [0]
    logits["speed"][0, 0, 7] = 6.0                                                    # one better speed word: rank 1
    assert pr._joint_ranks(logits, targets, has_next).tolist() == [1]
    logits["altitude"][0, 0] = torch.arange(11.0) * 10.0                              # the truth (2) falls out of altitude's top-8
    assert pr._joint_ranks(logits, targets, has_next).tolist() == [pr.JOINT_SEARCH ** len(kinds)]
    # a tie with the truth counts as beaten (never an optimistic rank)
    logits = {kind: torch.zeros(1, 2, classes[kind]) for kind in kinds}
    assert pr._joint_ranks(logits, targets, has_next).tolist() == [pr.JOINT_SEARCH ** len(kinds)] or pr._joint_ranks(logits, targets, has_next).tolist()[0] > 0


def test_fit_keeps_the_best_val_epoch_and_evaluate_reads_every_rate(world):
    _vocabulary, _series, _readings, sequences, types, config = world
    torch.manual_seed(0)
    model = pr.InstructionPrior(config)
    rows = []
    result = pr.fit(model, sequences[:3], sequences[3:], types, epochs=2, patience=5, batch_size=2, learning_rate=1e-3,
                    seed=0, device=torch.device("cpu"), log=rows.append)
    assert len(result.history) == 2 == len(rows) and result.best_epoch in (1, 2) and not result.stopped_early
    assert set(result.state_dict) == set(model.state_dict()) and rows[0]["val"]["joint_top_k"] is None   # not read per epoch
    reading = pr.evaluate(model, sequences[3:], types, batch_size=2, device=torch.device("cpu"))
    assert reading["positions_with_next"] == sequences[3].length - 1
    for kind in ins.INSTRUCTION_KINDS:
        assert 0.0 <= reading["top1"][kind] <= 1.0
        # a top-k at or past the kind's class count is 1 by construction and reads as ABSENT;
        # the runway's classes are the cohort's thresholds, so even its top-2 can be absent
        for k in ("2", "4", "8"):
            assert (reading["top_k"][kind][k] is None) == (config.classes(kind) <= int(k))
        available = [reading["top_k"][kind][k] for k in ("2", "4", "8") if reading["top_k"][kind][k] is not None]
        assert available == sorted(available) and all(reading["top1"][kind] <= v <= 1.0 for v in available)
        for name in ("flip_rate", "miss_rate", "change_recall"):
            assert reading[name][kind] is None or 0.0 <= reading[name][kind] <= 1.0
    assert reading["joint_top_k"]["2"] <= reading["joint_top_k"]["4"] <= reading["joint_top_k"]["8"] <= 1.0
    assert 0.0 <= reading["landed_accuracy"] <= 1.0 and reading["total"] == pytest.approx(reading["next"] + reading["landed"])
    assert reading["landed_share"] == pytest.approx(1 / sequences[3].length) and reading["landed_recall"] in (0.0, 1.0)
    # the readings are batch-size invariant over flights of different lengths (the padding masks)
    together = pr.evaluate(model, sequences, types, batch_size=4, device=torch.device("cpu"))
    alone = pr.evaluate(model, sequences, types, batch_size=1, device=torch.device("cpu"))
    for name in ("next", "landed", "positions_with_next", "landed_accuracy"):
        assert together[name] == pytest.approx(alone[name], rel=1e-5)
    for name in ("top1", "flip_rate", "miss_rate", "change_recall", "change_share"):
        assert together[name] == alone[name]
    assert together["joint_top_k"] == alone["joint_top_k"]
    # early stop: a frozen model never improves after its first epoch
    torch.manual_seed(0)
    frozen = pr.fit(pr.InstructionPrior(config), sequences[:3], sequences[3:], types, epochs=4, patience=1, batch_size=2,
                    learning_rate=0.0, seed=0, device=torch.device("cpu"))
    assert frozen.stopped_early and frozen.best_epoch == 1 and len(frozen.history) == 2
    with pytest.raises(ValueError, match="no epoch improved"):
        pr.fit(pr.InstructionPrior(config), sequences[:3], sequences[3:], types, epochs=0, patience=1, batch_size=2,
               learning_rate=1e-3, seed=0, device=torch.device("cpu"))
    # a one-position sequence: no next word, the landing only; nothing divides by zero, the rates read as absent
    one = sq.InstructionSequence(**{**sequences[0].__dict__, "positions_s": sequences[0].positions_s[:1], "words": sequences[0].words[:1],
                                    "states": sequences[0].states[:1], "targets": sequences[0].targets[:1]})
    only = pr.evaluate(model, [one], types, batch_size=1, device=torch.device("cpu"))
    assert only["positions_with_next"] == 0 and only["next"] is None and only["top1"]["heading"] is None and only["landed_share"] == 1.0


def test_the_hold_baseline_scores_a_sentence_that_never_changes_as_perfectly_held(world):
    _vocabulary, _series, _readings, sequences, _types, config = world
    still = sq.InstructionSequence(
        dataset_id="d", flight_id="f", positions_s=np.arange(6) * 10.0,
        words=np.tile(np.array([[18, 3, 6, -1, 1]]), (6, 1)), states=np.zeros((6, 6), dtype=np.float32),
        targets=np.tile(np.array([[18, 3, 6, -1, 1]]), (6, 1)), typecode="B738", runway_course_rad=0.0, ends_at_landing=True,
    )
    baseline = pr.hold_baseline(sequences, [still], config)
    assert all(baseline["hold_accuracy"][kind] == 1.0 for kind in ins.INSTRUCTION_KINDS) and baseline["positions_with_next"] == 5
    assert all(baseline["bigram_nll"][kind] > 0.0 for kind in ins.INSTRUCTION_KINDS)
    assert baseline["bigram_nll"]["next"] == pytest.approx(sum(baseline["bigram_nll"][k] for k in ins.INSTRUCTION_KINDS))
    real = pr.hold_baseline(sequences[:3], sequences[3:], config)
    assert real["positions_with_next"] == sequences[3].length - 1
    with pytest.raises(ValueError, match="words are the counts"):
        pr.PriorConfig(words={"heading": 36}, type_count=1, vocabulary_sha256="x")
    with pytest.raises(ValueError, match="words are the counts"):                    # the kinds' ORDER is the contract
        pr.PriorConfig(words={"altitude": 11, "heading": 36, "speed": 23, "intercept": 3, "runway": 2}, type_count=1, vocabulary_sha256="x")
    assert pr.PriorConfig.from_dict(config.to_dict()) == config


def test_the_runner_s_table_and_cohort_door(world):
    from types import SimpleNamespace
    from ts_transformer.experiments.instruction_prior import render
    from ts_transformer.experiments.support import cohort_splits
    _vocabulary, _series, _readings, sequences, types, config = world
    torch.manual_seed(0)
    model = pr.InstructionPrior(config).eval()
    readings = pr.evaluate(model, sequences[3:], types, batch_size=2, device=torch.device("cpu"))
    baseline = pr.hold_baseline(sequences[:3], sequences[3:], config)
    table = render(readings, baseline, limit=3)
    assert all(kind in table for kind in ins.INSTRUCTION_KINDS) and "PREFIX of 3" in table and "joint top-K" in table
    payload = {"split": {"train": ["a", "b", "c"], "val": ["d", "e"]}}
    cohort = SimpleNamespace(train_flight_ids=["b", "a"], val_flight_ids=["e"])
    assert cohort_splits(payload, cohort) == {"train": ["b", "a"], "val": ["e"]}
    assert cohort_splits(payload, cohort, limit=1) == {"train": ["b"], "val": ["e"]}
    with pytest.raises(ValueError, match="not in the executor's val split"):
        cohort_splits(payload, SimpleNamespace(train_flight_ids=["a"], val_flight_ids=["a"]))
