"""The prior's network (prior design §2, §3, §7; milestone B2): row by row as a whole, times by their differences only,
candidates in any order and number, the first predicted step's masks, the place for an added module, the checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ts_transformer.instructions.words import COLUMNS, RUNWAY, Words
from ts_transformer.prior.batch import (
    CANDIDATE_FEATURES, OWN_FEATURES,
    RUNWAY_FIXED_CLASSES, RUNWAY_GO_AROUND_CLASS, RUNWAY_UNCHANGED_CLASS, SentenceRows, collate, require_words,
    target_classes,
)
from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA, load_checkpoint, save_checkpoint
from ts_transformer.prior.model import Prior, PriorConfig, WORD_COLUMNS
from ts_transformer.tests.support import instruction_spec, prior_sentence

CPU = torch.device("cpu")
SMALL = {"d_model": 32, "layers": 2, "heads": 4, "feedforward": 64}


@pytest.fixture(scope="module")
def words():
    return Words(instruction_spec())


def small_prior(words, seed=0, variant="full", **shape):
    torch.manual_seed(seed)
    return Prior(PriorConfig.from_words(words, variant, **{**SMALL, **shape})).eval()


def sentences(words, count=3, seed=1, **kwargs):
    rng = np.random.default_rng(seed)
    return [prior_sentence(rng, words, rows=20 + 7 * i, **kwargs) for i in range(count)]


def test_the_columns_and_their_values_come_from_the_spec(words):
    config = PriorConfig.from_words(words, "full")
    counts = words.class_counts()
    assert config.word_values == tuple(counts[name] for name in WORD_COLUMNS)
    model = Prior(config)
    for name in WORD_COLUMNS:
        assert model.heads[name].out_features == 1 + counts[name]
    with pytest.raises(ValueError, match="even width"):
        PriorConfig.from_words(words, "full", d_model=128, heads=6)
    with pytest.raises(ValueError, match="variant"):
        PriorConfig.from_words(words, "no-such-variant")


def test_the_vocabulary_words_become_the_heads_classes():
    targets = np.array([[-1, -1, -1, -1, -1], [-2, 0, 3, 1, 4], [2, 5, -1, -1, -1]])
    assert target_classes(targets).tolist() == [
        [RUNWAY_UNCHANGED_CLASS, 0, 0, 0, 0], [RUNWAY_GO_AROUND_CLASS, 1, 4, 2, 5], [2 + RUNWAY_FIXED_CLASSES, 6, 0, 0, 0]]


def test_row_by_row_gives_what_the_whole_sentence_gives(words):
    model = small_prior(words)
    rows = collate(sentences(words), CPU)
    with torch.no_grad():
        h, tokens = model.encode(rows)
        whole = model.logits(h, tokens, rows.valid, rows.targets, rows.first)
        past = model.no_past(rows.present.shape[0], rows.present.shape[1])
        pieces = []
        for start, end in ((0, 1), (1, 9), (9, 10), (10, rows.present.shape[1])):
            part = rows.between(start, end)
            h_part, tokens_part, past = model.extend(part, past)
            pieces.append(model.logits(h_part, tokens_part, part.valid, part.targets, part.first))
    for column in range(len(COLUMNS)):
        row_by_row = torch.cat([piece[column] for piece in pieces], dim=1)
        finite = torch.isfinite(whole[column])
        assert torch.equal(finite, torch.isfinite(row_by_row))
        torch.testing.assert_close(row_by_row[finite], whole[column][finite], rtol=1e-5, atol=1e-5)


def test_a_past_that_outgrows_its_room_grows_and_gives_the_same_rows(words):
    """The speaker's cache grows as a sentence needs (`Past.grown`): row by row from a room of 2 rows, the logits are
    those of a room for the whole sentence, and the room is at least the rows."""
    model = small_prior(words)
    rows = collate(sentences(words), CPU)
    count = rows.present.shape[1]
    with torch.no_grad():
        outputs = []
        for capacity in (count, 2):
            past, pieces = model.no_past(rows.present.shape[0], capacity), []
            for r in range(count):
                part = rows.between(r, r + 1)
                h_part, tokens_part, past = model.extend(part, past)
                pieces.append(model.logits(h_part, tokens_part, part.valid, part.targets, part.first)[0])
            outputs.append(torch.cat(pieces, dim=1))
            assert all(p.keys.shape[2] >= count and p.rows == count for p in past)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0, equal_nan=True)


def test_a_shift_of_every_time_changes_no_output(words):
    model = small_prior(words)
    rows = collate(sentences(words), CPU)
    # 1e6 s: the rows' times stay exact in float32, the angles (1e6 rad) need float64
    shifted = rows._replace(time_s=rows.time_s + 1.0e6)
    with torch.no_grad():
        before, after = model(rows), model(shifted)
    for a, b in zip(before, after):
        finite = torch.isfinite(a)
        torch.testing.assert_close(b[finite], a[finite], rtol=1e-5, atol=1e-5)
    # the time does count: a row read as later changes what attends to it
    stretched = rows._replace(time_s=rows.time_s * 2.0)
    with torch.no_grad():
        assert not torch.allclose(model(stretched)[1], before[1])


def permuted(sentence: SentenceRows, order: np.ndarray) -> SentenceRows:
    """The same flight with its candidates in ``order`` (new slot i holds old candidate order[i])."""
    new_index = np.argsort(order)
    runway = sentence.runway_in_force.copy()
    runway[runway >= 0] = new_index[runway[runway >= 0]]
    targets = sentence.targets.copy()
    said = targets[:, RUNWAY] >= 0
    targets[said, RUNWAY] = new_index[targets[said, RUNWAY]]
    return SentenceRows(**{**sentence.__dict__, "candidates": sentence.candidates[:, order],
                           "runway_in_force": runway, "targets": targets})


def test_a_permutation_of_the_candidates_permutes_the_runway_scores_and_nothing_else(words):
    model = small_prior(words)
    sentence = sentences(words, count=1, candidates=5)[0]
    order = np.array([3, 0, 4, 1, 2])
    with torch.no_grad():
        before = model(collate([sentence], CPU))
        after = model(collate([permuted(sentence, order)], CPU))
    for column in range(len(COLUMNS)):
        expected = before[column]
        if column == RUNWAY:
            expected = torch.cat((expected[..., :RUNWAY_FIXED_CLASSES],
                                  expected[..., RUNWAY_FIXED_CLASSES:][..., torch.as_tensor(order)]), dim=-1)
        finite = torch.isfinite(expected)
        assert torch.equal(finite, torch.isfinite(after[column]))
        torch.testing.assert_close(after[column][finite], expected[finite], rtol=1e-5, atol=1e-5)


def test_any_number_of_candidates_and_padding_change_nothing(words):
    """Airports of 1, 2 and 9 candidates in one batch (more than any of the five has), each sentence's logits as it has
    them alone: neither a padded candidate slot nor a padded row is read."""
    model = small_prior(words)
    rng = np.random.default_rng(4)
    group = [prior_sentence(rng, words, candidates=k, rows=n) for k, n in ((1, 25), (2, 40), (9, 18))]
    with torch.no_grad():
        together = model(collate(group, CPU))
        for b, sentence in enumerate(group):
            alone = model(collate([sentence], CPU))
            for column in range(len(COLUMNS)):
                width = alone[column].shape[-1]
                part = together[column][b, : sentence.rows, :width]
                finite = torch.isfinite(alone[column][0])
                assert torch.equal(finite, torch.isfinite(part))
                torch.testing.assert_close(part[finite], alone[column][0][finite], rtol=1e-5, atol=1e-5)
                if column == RUNWAY:
                    assert torch.isinf(together[column][b, :, width:]).all()


def test_the_first_predicted_step_says_a_word_in_every_column(words):
    model = small_prior(words)
    rows = collate(sentences(words), CPU)
    with torch.no_grad():
        logits = model(rows)
    for column, logit in enumerate(logits):
        at_first = logit[rows.first]
        assert torch.isneginf(at_first[:, 0]).all()
        assert torch.isfinite(logit[~rows.first][:, 0]).all()
        if column == RUNWAY:
            assert torch.isneginf(at_first[:, RUNWAY_GO_AROUND_CLASS]).all()
        assert torch.isfinite(at_first[:, -1]).all()


def changed(sentence, **arrays):
    return SentenceRows(**{**sentence.__dict__, **arrays})


def test_a_sentence_is_refused_at_the_boundary(words):
    sentence = sentences(words, count=1, candidates=2)[0]
    first = sentence.first_step
    targets = sentence.targets.copy()
    targets[first, 2] = -1
    with pytest.raises(ValueError, match="every column"):
        changed(sentence, targets=targets)
    # D23: the runway (or any word) in force at or before the first predicted step would give its answer away
    runway = sentence.runway_in_force.copy()
    runway[: first + 1] = sentence.targets[first, RUNWAY]
    with pytest.raises(ValueError, match="in force at or before"):
        changed(sentence, runway_in_force=runway)
    in_force = sentence.words_in_force.copy()
    in_force[0, 1] = 2
    with pytest.raises(ValueError, match="in force at or before"):
        changed(sentence, words_in_force=in_force)
    runway = sentence.runway_in_force.copy()
    runway[first + 3] = 2
    with pytest.raises(ValueError, match="not one of its 2 candidates"):
        changed(sentence, runway_in_force=runway)
    for bad in (2, -3):
        targets = sentence.targets.copy()
        targets[first + 2, RUNWAY] = bad
        with pytest.raises(ValueError, match="runway word"):
            changed(sentence, targets=targets)
    no_motion = OWN_FEATURES.index("no_motion")
    for row, column, value in ((0, no_motion, 0.0), (5, no_motion, 1.0), (0, OWN_FEATURES.index("ground_speed"), 0.7)):
        own = sentence.own.copy()
        own[row, column] = value
        with pytest.raises(ValueError, match="D60"):
            changed(sentence, own=own)
    vectors = sentence.candidates.copy()
    vectors[0, 1, CANDIDATE_FEATURES.index("motion_minus_course_cos")] = 1.0
    with pytest.raises(ValueError, match="D60"):
        changed(sentence, candidates=vectors)
    with pytest.raises(ValueError, match="seconds from row 0"):
        changed(sentence, time_s=sentence.time_s + 1.76e9)
    own = sentence.own.copy()
    own[3, 1] = np.nan
    with pytest.raises(ValueError, match="own is not finite"):
        changed(sentence, own=own)
    targets = sentence.targets.copy()
    targets[first + 1, 4] = words.class_counts()["speed"]
    with pytest.raises(ValueError, match="speed beyond"):
        require_words([changed(sentence, targets=targets)], PriorConfig.from_words(words, "full").word_values)
    require_words([sentence], PriorConfig.from_words(words, "full").word_values)


def test_a_head_reads_the_words_of_the_earlier_columns_of_its_row_only(words):
    """Teacher forcing hands each head the truth of the columns before it at its own row: a change of one word of a row
    changes the logits of the later columns of that row and nothing else (no input, no other row, not its own)."""
    model = small_prior(words)
    sentence = sentences(words, count=1, candidates=4)[0]
    rows = collate([sentence], CPU)
    row = sentence.first_step + 3
    with torch.no_grad():
        h, _ = model.encode(rows)
        before = model(rows)
        for column in range(len(COLUMNS)):
            targets = rows.targets.clone()
            was = int(targets[0, row, column])
            targets[0, row, column] = (2 if was != 2 else 3) if column == RUNWAY else (1 if was != 1 else 2)
            assert targets[0, row, column] != was
            other = rows._replace(targets=targets)
            assert torch.equal(model.encode(other)[0], h)
            after = model(other)
            for c in range(len(COLUMNS)):
                moved = ~torch.isclose(after[c], before[c], rtol=0, atol=0).all(dim=-1)[0]
                expected = torch.zeros_like(moved)
                expected[row] = c > column
                assert torch.equal(moved, expected), (column, c)


class Zero(nn.Module):
    """An added module whose output layer starts at zero, reading the caller's ``extra``."""

    def __init__(self, d):
        super().__init__()
        self.read = nn.Linear(d + 1, d)
        self.out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, extra):
        return self.out(torch.tanh(self.read(torch.cat((x, extra.expand(*x.shape[:-1], 1)), dim=-1))))


def test_an_added_module_with_a_zero_output_changes_nothing_until_it_learns(words):
    model = small_prior(words)
    rows = collate(sentences(words), CPU)
    extra = torch.ones(1, 1, 1)
    with torch.no_grad():
        before = model(rows)
        model.add_at_each_layer(lambda i: Zero(SMALL["d_model"]))
        after = model(rows, extra)
    for a, b in zip(before, after):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="added module already"):
        model.add_at_each_layer(lambda i: Zero(SMALL["d_model"]))
    loss = sum(logit[rows.asked].logsumexp(-1).sum() for logit in model(rows, extra))
    loss.backward()
    assert all(layer.added.out.weight.grad.abs().sum() > 0 for layer in model.layers)
    nn.init.normal_(model.layers[0].added.out.weight)
    with torch.no_grad():
        assert not torch.equal(model(rows, extra)[1], before[1])


def test_the_checkpoint_opens_only_for_its_artefact(words, tmp_path):
    model = small_prior(words, variant="constants")
    identity = {"spec_sha256": "a" * 64, "day_split": {"train": ["2026-05-01"]}, "candidates": {"KXXX": []},
                "sentences": {"train": "b" * 64}}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, model.state_dict(), identity=identity, run={"airports": ["KXXX"], "held_out": None},
                    train_config={"seed": 1})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, model.state_dict(), identity=identity, run={}, train_config={})
    with pytest.raises(ValueError, match="plain JSON"):
        save_checkpoint(tmp_path / "tuple.pt", model, model.state_dict(), identity={"airports": ("KXXX",)}, run={},
                        train_config={})
    with pytest.raises(TypeError, match="not JSON serializable"):
        save_checkpoint(tmp_path / "numpy.pt", model, model.state_dict(), identity={"rows": np.int64(5)}, run={},
                        train_config={})
    loaded = load_checkpoint(path, identity)
    assert loaded.model.config == model.config and loaded.run["airports"] == ["KXXX"]
    rows = collate(sentences(words, variant="constants"), CPU)
    with torch.no_grad():
        for a, b in zip(model(rows), loaded.model(rows)):
            assert torch.equal(a, b)
    with pytest.raises(ValueError, match=r"\['spec_sha256'\] differ"):
        load_checkpoint(path, {**identity, "spec_sha256": "c" * 64})
    payload = torch.load(path, weights_only=True)
    other = tmp_path / "other.pt"
    torch.save({**payload, "schema": "ts-prior-checkpoint-v3"}, other)
    with pytest.raises(ValueError, match=CHECKPOINT_SCHEMA):
        load_checkpoint(other, identity)


def test_a_loops_row_is_the_sentences_row(words):
    """§7 item 2: the row a loop builds for a speaker (`row_tensors`) is the row `collate` gives the same sentence —
    every input; the targets are not the loop's to give."""
    from ts_transformer.prior.batch import row_tensors

    rng = np.random.default_rng(9)
    group = [prior_sentence(rng, words, candidates=k, rows=n) for k, n in ((2, 20), (5, 26))]
    whole = collate(group, CPU)
    r = 12
    row = row_tensors(np.array([s.time_s[r] for s in group]), np.stack([s.own[r] for s in group]),
                      [s.candidates[r] for s in group], np.array([s.runway_in_force[r] for s in group]),
                      np.array([s.go_around[r] for s in group]), np.stack([s.heading_in_force[r] for s in group]),
                      np.stack([s.words_in_force[r] for s in group]), np.stack([s.since[r] for s in group]),
                      np.array([s.first_step == r for s in group]), np.array([r >= s.first_step for s in group]), CPU)
    expected = whole.between(r, r + 1)
    for name in row._fields:
        if name != "targets":
            assert torch.equal(getattr(row, name), getattr(expected, name)), name
