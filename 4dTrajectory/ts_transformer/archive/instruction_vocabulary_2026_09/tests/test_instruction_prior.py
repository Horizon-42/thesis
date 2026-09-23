"""The instruction sequences and the instruction prior (two-tier v3 stage B, B′-dev3): the
sentence with states and next-word targets, a rolled sequence's flown states and truth targets,
the batch, the causal factorised model, its loss, the readings (top-k, flips, misses, joint
coverage, landing) and the hold / bigram baselines. Untrained on synthetic data: mechanics
only, never a number."""

from __future__ import annotations

from dataclasses import replace

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
#: The class LABEL is airport-qualified (idents collide across a pooled cohort: KSJC and KSTL both
#: have 12L); the bare ident is what the synthetic generator takes, so the two are kept apart.
RUNWAYS = ins.RunwayVocabulary.from_idents([f"{AIRPORT}:{RUNWAY}", f"{AIRPORT}:{OTHER_RUNWAY}"])


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


def _prefix(sequence, count: int):
    """The first ``count`` events of a sentence, as a legal prefix.

    Slicing alone is illegal since D72: the sliced last event would still say `landed` while the
    prefix does not end the flight. A real caller has to set both, so the helper does.
    """
    terminal = ins.INSTRUCTION_KINDS.index("terminal")
    words = sequence.words[:count].copy()
    words[-1, terminal] = ins.TERMINAL_CONTINUE
    return sq.InstructionSequence(**{**sequence.__dict__, "positions_s": sequence.positions_s[:count],
                                     "words": words, "states": sequence.states[:count],
                                     "targets": sequence.targets[:count], "ends_at_landing": False})


def test_the_truth_sequence_carries_the_words_the_states_and_the_next_words(world):
    _vocabulary, series, readings, sequences, _types, _config = world
    item, reading, sequence = series[0], readings[0], sequences[0]
    assert sequence.length == len(reading.event_times_s) and sequence.words.shape == (sequence.length, 6)   # five kinds since D62
    assert np.array_equal(sequence.targets[:-1], reading.words[1:]) and np.array_equal(sequence.targets[-1], reading.words[-1])
    assert sequence.states.shape == (sequence.length, len(sq.STATE_TOKEN_FEATURES))
    origin = sq.airport_origin(item)          # D66: the AIRPORT reference, not the threshold
    first = sq.state_token(item.values[0], origin)
    assert np.allclose(sequence.states[0], first, atol=1e-6) and abs(first[4] ** 2 + first[5] ** 2 - 1.0) < 1e-5
    assert sequence.typecode == str(item.scenario.aircraft.code) and sequence.ends_at_landing
    # the position is relative to the threshold, whatever the chart's origin
    shifted = np.array(item.values[0], dtype=np.float64)
    shifted[:3] += 5000.0
    assert np.allclose(sq.state_token(shifted, np.asarray(origin) + 5000.0), first, atol=1e-6)
    still = np.array(item.values[0], dtype=np.float64)
    still[3:5] = 0.0
    with pytest.raises(ValueError, match="has no course"):
        sq.state_token(still, origin)
    with pytest.raises(ValueError, match="is not series"):
        sq.flight_sequence(series[1], reading)


def test_a_rolled_sequence_is_matched_by_TIME_and_takes_its_ending_from_its_own_terminal_word(world):
    """D55's shape, and the two defects a review found in its first version.

    The loop chooses its own gaps — that is what the duration word is for — so its events are NOT
    at the truth's instants and matching targets by INDEX would be wrong. And its ending is its own
    terminal word, not how many events it happened to say.
    """
    _vocabulary, series, readings, sequences, _types, _config = world
    item, reading, truth = series[0], readings[0], sequences[0]
    origin = sq.airport_origin(item)
    terminal = ins.INSTRUCTION_KINDS.index("terminal")
    half = max(1, truth.length // 2)

    # flown at the truth's own instants and stopping short: the prefix continues, and each target
    # is the truth's NEXT event
    said = truth.words[:half].copy()
    said[-1, terminal] = ins.TERMINAL_CONTINUE
    times = truth.positions_s[:half]
    rolled = sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values, said, times, origin)
    assert rolled.length == half and not rolled.ends_at_landing
    assert np.array_equal(rolled.targets, reading.words[1 : half + 1])

    # the same words said LATER than the truth said them: the targets must follow the TIME, so a
    # said event after the truth's k-th event targets the (k+1)-th, not the index-matched one
    if truth.length > 2:                     # needs a next event to move past
        shifted = times + float(np.diff(reading.event_times_s).max())
        shifted = np.minimum(shifted, reading.event_times_s[-1])
        late = sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values, said, shifted, origin)
        assert not np.array_equal(late.targets, rolled.targets)

    # the whole sentence, ending on its own terminal word
    whole = sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values,
                               truth.words, truth.positions_s, origin)
    assert whole.ends_at_landing and int(truth.words[-1, terminal]) == ins.TERMINAL_LANDED

    # `rolled_sequence` DERIVES the ending from the terminal word, so it cannot disagree; the
    # guard is on the type itself, for anyone building one directly (D72: ONE answer)
    lying = truth.words.copy()
    lying[-1, terminal] = ins.TERMINAL_CONTINUE
    with pytest.raises(ValueError, match="D72 made these ONE answer"):
        sq.InstructionSequence(**{**truth.__dict__, "words": lying})
    with pytest.raises(ValueError, match="strictly increasing"):
        sq.rolled_sequence(truth, reading, item.supervision_times, item.supervision_values,
                           truth.words[:2], np.array([5.0, 5.0]), origin)
    with pytest.raises(ValueError, match="is not sequence"):
        sq.rolled_sequence(truth, readings[1], item.supervision_times, item.supervision_values, said, times, origin)


def test_the_batch_marks_which_events_have_a_next_and_where_the_sentence_lands(world):
    _vocabulary, _series, _readings, sequences, types, config = world
    batch = pr.collate(sequences[:2], config, types)
    longest = max(item.length for item in sequences[:2])
    assert batch.words.shape == (2, longest, 6) and batch.targets.shape == (2, longest, 6) and batch.states.shape == (2, longest, 6)
    for row, item in enumerate(sequences[:2]):
        n = item.length
        assert bool(batch.valid[row, :n].all()) and not bool(batch.valid[row, n:].any())
        assert batch.words[row, :n].min() >= 0        # D73: no column carries a 'nothing said' value
        assert np.array_equal(batch.words[row, :n].numpy(), pr.shift_words(item.words))
        assert np.array_equal(pr.unshift_words(batch.words[row, :n].numpy()), item.words)
        # the RUNWAY column is never shifted: its word is already ≥ 0 and indexes its table
        assert np.array_equal(batch.words[row, :n, 3].numpy(), item.words[:, 3])
        assert batch.words[row, :n, 3].max() < config.classes("runway")
        # D72: the last event of a landing sentence has no target — its own terminal word says it lands
        assert bool((batch.targets[row, n - 1] == pr.IGNORE).all())
        # D72: the landing is the last event's own terminal word, not a separate label
        assert int(item.words[-1, ins.INSTRUCTION_KINDS.index("terminal")]) == ins.TERMINAL_LANDED
        assert bool(batch.has_next[row, : n - 1].all()) and not bool(batch.has_next[row, n - 1 :].any())
    assert batch.type_index.shape == (2,)      # D66: no runway-course context token any more
    assert not hasattr(batch, "runway"), "the runway course must not come back as context"
    # a rolled PREFIX: its last position has a next word and no landing
    prefix = _prefix(sequences[0], min(4, sequences[0].length))
    batch = pr.collate([prefix], config, types)
    assert bool(batch.has_next[0].all()) and np.array_equal(batch.targets[0].numpy(), pr.shift_words(prefix.targets))
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
    terms = model.loss(output, batch)
    assert set(terms) == {*ins.INSTRUCTION_KINDS, "next", "total"} and torch.isfinite(terms["total"])
    assert torch.isclose(terms["next"], sum(terms[k] for k in ins.INSTRUCTION_KINDS))
    # causality: perturbing the input at the LAST event leaves every logit before it unchanged
    words = batch.words.clone()
    last = words.shape[1] - 1
    words[:, last, 0] = (words[:, last, 0] + 1) % config.classes("heading")
    later = model(pr.PriorBatch(words, batch.states, batch.valid, batch.targets, batch.type_index))
    assert torch.allclose(later.logits["speed"][:, :last], output.logits["speed"][:, :last], atol=1e-5)
    assert not torch.allclose(later.logits["speed"][:, last:], output.logits["speed"][:, last:], atol=1e-5)


def test_the_joint_rank_is_the_truth_tuple_s_place_among_the_product_candidates(world):
    kinds = ins.INSTRUCTION_KINDS
    classes = {kind: world[5].classes(kind) for kind in kinds}
    targets = torch.tensor([[[3, 2, 5, 0, 1, 0], [pr.IGNORE] * 6]])                    # one event with a next
    has_next = targets[..., 0] != pr.IGNORE
    logits = {kind: torch.zeros(1, 2, classes[kind]) for kind in kinds}
    for column, kind in enumerate(kinds):
        logits[kind][0, 0, targets[0, 0, column]] = 5.0                              # the truth is every kind's argmax
    assert pr._joint_ranks(logits, targets, has_next).tolist() == [0]
    logits["speed"][0, 0, min(7, classes["speed"] - 1)] = 6.0                                                    # one better speed word: rank 1
    assert pr._joint_ranks(logits, targets, has_next).tolist() == [1]
    # A kind with MORE classes than the joint search is wide, or the truth cannot fall out of it:
    # the vertical word has six modes and they all fit inside the top-8. The heading has 72.
    logits["heading"][0, 0] = torch.arange(float(classes["heading"])) * 10.0          # the truth (3) falls out of heading's top-8
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
    assert reading["total"] == pytest.approx(reading["next"])   # D72: no separate landing term any more
    assert reading["top1"]["terminal"] is None or 0.0 <= reading["top1"]["terminal"] <= 1.0
    # the readings are batch-size invariant over flights of different lengths (the padding masks)
    together = pr.evaluate(model, sequences, types, batch_size=4, device=torch.device("cpu"))
    alone = pr.evaluate(model, sequences, types, batch_size=1, device=torch.device("cpu"))
    for name in ("next", "positions_with_next"):
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
    # a one-EVENT sentence that lands: no next word at all, so nothing divides by zero and every
    # rate reads as absent. (A one-event PREFIX would still have a next — it continues.)
    terminal = ins.INSTRUCTION_KINDS.index("terminal")
    words = sequences[0].words[:1].copy()
    words[0, terminal] = ins.TERMINAL_LANDED
    one = sq.InstructionSequence(**{**sequences[0].__dict__, "positions_s": sequences[0].positions_s[:1],
                                    "words": words, "states": sequences[0].states[:1],
                                    "targets": sequences[0].targets[:1], "ends_at_landing": True})
    only = pr.evaluate(model, [one], types, batch_size=1, device=torch.device("cpu"))
    assert only["positions_with_next"] == 0 and only["next"] is None and only["top1"]["heading"] is None


def test_the_hold_baseline_scores_a_sentence_that_never_changes_as_perfectly_held(world):
    _vocabulary, _series, _readings, sequences, _types, config = world
    still = sq.InstructionSequence(
        dataset_id="d", flight_id="f", positions_s=np.arange(6) * 10.0,
        words=np.tile(np.array([[18, 3, 6, 1, 5, ins.TERMINAL_CONTINUE]]), (6, 1)), states=np.zeros((6, 6), dtype=np.float32),
        targets=np.tile(np.array([[18, 3, 6, 1, 5, ins.TERMINAL_CONTINUE]]), (6, 1)), typecode="B738", ends_at_landing=False,
    )
    baseline = pr.hold_baseline(sequences, [still], config)
    assert all(baseline["hold_accuracy"][kind] == 1.0 for kind in ins.INSTRUCTION_KINDS) and baseline["positions_with_next"] == 5
    assert all(baseline["bigram_nll"][kind] > 0.0 for kind in ins.INSTRUCTION_KINDS)
    assert baseline["bigram_nll"]["next"] == pytest.approx(sum(baseline["bigram_nll"][k] for k in ins.INSTRUCTION_KINDS))
    real = pr.hold_baseline(sequences[:3], sequences[3:], config)
    assert real["positions_with_next"] == sequences[3].length - 1
    with pytest.raises(ValueError, match="words are the counts"):
        pr.PriorConfig(words={"heading": 36}, type_count=1, vocabulary_sha256="x")
    # the kinds' ORDER is the contract: the prior embeds and scores by COLUMN INDEX, so a dict
    # holding every kind but in another order sizes the wrong head. The case has to carry all six
    # or it would also pass on a missing-key check and stop isolating the ordering rule.
    every = {kind: 2 for kind in ins.INSTRUCTION_KINDS}
    swapped = {kind: every[kind] for kind in (ins.INSTRUCTION_KINDS[1], ins.INSTRUCTION_KINDS[0], *ins.INSTRUCTION_KINDS[2:])}
    assert set(swapped) == set(every) and tuple(swapped) != tuple(every)
    with pytest.raises(ValueError, match="words are the counts"):
        pr.PriorConfig(words=swapped, type_count=1, vocabulary_sha256="x")
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


def test_the_prior_takes_exactly_one_cohort_door(tmp_path):
    """The prior needs TRACKS, not a trained executor, and a pooled cohort has no executor at all
    — so the runner takes either the checkpoint door or the manifests one, and refuses neither and
    both before it reads anything."""
    from ts_transformer.experiments import instruction_prior as runner
    tail = ["--vocabulary", str(tmp_path / "v.json"), "--cohort", str(tmp_path / "c.json"), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        runner.main(tail)
    with pytest.raises(SystemExit):
        runner.main(["--executor", str(tmp_path / "none.pt"), "--airports", "KRDU", *tail])


def test_the_state_tokens_are_anchored_at_the_AIRPORT_not_at_the_landing_threshold(world):
    """D66's whole point, and the one thing that makes the runway head a prediction.

    Threshold-anchored coordinates are centred on the very runway the prior is asked to name, so
    its runway head would score ~1.0 while predicting nothing. The origin must sit somewhere the
    runway does NOT determine, and the shift must be bounded by the airport's own extent.
    """
    _vocabulary, series, _readings, _sequences, _types, _config = world
    item = series[0]
    origin = sq.airport_origin(item)
    threshold = np.asarray(item.target_chart)
    offset = float(np.hypot(*(origin[:2] - threshold[:2])))
    assert offset > 100.0, "the origin still sits on the threshold: the runway head reads its own answer"
    assert offset < 10_000.0, f"the origin is {offset:.0f} m from the threshold, beyond any airport's extent"
    # and it does not MOVE with the runway: each flight has its own chart, so the numbers differ
    # between flights, but within one chart the origin must ignore where the threshold is
    moved = replace(item, scenario=replace(item.scenario,
                                           target=replace(item.scenario.target,
                                                          latitude=item.scenario.target.latitude + 0.01)))
    assert np.allclose(sq.airport_origin(moved), origin), "the origin followed the threshold"
    assert not np.allclose(np.asarray(moved.target_chart), threshold), "the fixture did not move the threshold"
