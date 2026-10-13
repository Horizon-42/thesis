"""Stage C, C6: branch training (post-training §2 item 9, §8 C6; D37, D94) — on A26's one-flight synthetic artefact (the
fixture of `test_post_window_loop`)."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.conformance import STATE_BOUND_M as EXECUTOR_BOUND_M
from ts_transformer.experiments import post_branches
from ts_transformer.experiments.post_branches import branch_round, same_prefix
from ts_transformer.experiments.post_window_loop import start_move_of
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.post.branches import (
    STATE_BOUND_M, Group, Sentence, branch_points, continuation_numbers, first_numbers, samples,
)
from ts_transformer.post.edges import TOKEN_FEATURES
from ts_transformer.post.loss import one_pass
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, MovedScene
from ts_transformer.post.traffic_attention import parameter_groups
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, _window_loop, _with_module, setup  # noqa: F401


def _round(s, model, windows, continuations=2, round_=0, places=(0,)):
    return branch_round(model, lambda flights: s["moved_loop"](start_move_of(windows[0])), windows, list(places),
                        {0: s["stored"]}, s["flights"],
                        s["geometries"], {s["geometry"].code: s["roster"]}, s["finals"], s["words"], interval_s=DELTA,
                        variant="full",
                        edges_reference=s["reference"], faults={s["geometry"].code: {}}, device=CPU, seed=1337,
                        round_=round_, split="train",
                        continuations=continuations)


def _ahead(window):
    """The window with the flight itself inserted 8 s ahead (window A): lost at its first row flown."""
    own = window.scene.flights[0]
    key = own.key + INSERTED_SUFFIX
    return replace(window, kind=INSERTED, moved=((key, -8.0),),
                   scene=MovedScene(window.scene, added=(own.shifted(-8.0, DELTA, key=key),)))


def test_the_numbers_and_the_branch_points():
    a, b = first_numbers(1, 2, 3).random(5), first_numbers(1, 2, 3).random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, first_numbers(1, 2, 4).random(5))
    assert not np.array_equal(continuation_numbers(1, 2, 3, 40, 0).random(5), continuation_numbers(1, 2, 3, 40, 1).random(5))
    assert not np.array_equal(continuation_numbers(1, 2, 3, 40, 0).random(5), continuation_numbers(1, 2, 3, 70, 0).random(5))
    assert branch_points(4, 70, 4.0) == [4, 34, 64]                   # the first predicted step, then every 120 s
    assert branch_points(4, 5, 4.0) == [4] and branch_points(4, 4, 4.0) == []
    with pytest.raises(ValueError, match="whole number"):
        branch_points(4, 70, 7.0)
    assert STATE_BOUND_M == EXECUTOR_BOUND_M                          # the mirror of the executor's bound


def test_a_continuation_with_the_first_sentences_numbers_says_and_flies_the_first_sentence(setup):
    """D94, D37 item 4: the state copied at a branch point, continued with the first sentence's numbers."""
    s = setup
    model = _with_module(s["base"])
    first = _window_loop(s, model, s["windows"])
    (result,) = first.run([first_numbers(1337, 0, 0)])
    second = _window_loop(s, model, s["windows"])
    numbers = first_numbers(1337, 0, 0)
    while second.speaking.observing:
        second.observe()
    branch = second.speaking.start + 6
    while second.speaking.t < branch and second.speaking.alive.any():
        second.step(np.stack([numbers.random(len(COLUMNS))]))
    assert second.speaking.alive.all(), "the first sentence is longer than the branch point"
    (again,) = second.copy([0]).finish([copy.deepcopy(numbers)])
    assert np.array_equal(again.words, result.words) and again.outcome == result.outcome
    assert same_prefix(again.states, result.states, len(result.states), STATE_BOUND_M)
    assert second.speaking.t == branch                                 # the loop copied is unchanged


def test_the_executor_flies_the_same_states_with_and_without_gradients(setup):
    """D37 item 6."""
    s = setup
    model = _with_module(s["base"])
    out = []
    for grad in (False, True):
        with torch.set_grad_enabled(grad):
            (result,) = _window_loop(s, model, s["windows"]).run([first_numbers(1337, 0, 0)])
        out.append(result)
    assert np.array_equal(out[0].words, out[1].words) and np.array_equal(out[0].states, out[1].states)


def test_a_round_gives_groups_after_each_branch_point_and_their_samples(setup):
    s = setup
    model = _with_module(s["base"])
    round_ = _round(s, model, [_ahead(s["windows"][0])])
    (first,) = round_.first
    assert first.reward == 0.0 and round_.spoken_again == [0] and round_.differed == []
    start = s["stored"].rows.start
    assert [g.branch for g in round_.groups] == branch_points(start, first.loss_step, DELTA) == [start]
    (group,) = round_.groups
    assert len(group.continuations) == 2 and np.array_equal(group.first.rows.targets[start:], first.words)
    # every sentence lost: no gradient — a group whose rewards are all the same gives no sample
    assert not group.informative
    with pytest.raises(ValueError, match="no informative group"):
        samples(round_.groups, CPU)
    # rewards that differ: the advantage, on the rows from the branch point only
    rewarded = replace(group, continuations=(replace(group.continuations[0], reward=1.0), group.continuations[1]))
    batch = samples([rewarded], CPU)
    assert batch.counted.shape[0] == 3
    assert (~batch.counted[:, :start]).all() and batch.counted[:, start].all()
    expected = torch.tensor([0.0, 1.0, 0.0]) - 1.0 / 3.0
    assert torch.allclose(batch.advantage[:, start], expected)
    assert (batch.advantage[~batch.counted] == 0).all()
    assert torch.equal(samples([rewarded], CPU, scale=0.25).advantage, batch.advantage * 0.25)    # D195: a group's weight
    # the samples go through the loss of C7
    optimizer = torch.optim.AdamW(parameter_groups(model, prior_lr=1e-4, traffic_lr=1e-3))
    data = batch.rows
    (parts,) = one_pass(model, s["base"], optimizer, [([batch], data)])
    assert torch.isfinite(parts.loss)


@pytest.mark.parametrize("first, continuations, side", [
    (0.0, [0.0] * 7 + [1.0], 1),              # one lucky continuation against eight
    (4.0, [3.0] * 8, 1),                      # the first sentence the one good one
    (5.0, [5.0, 4.0, 5.0, 4.0, 5.0, 5.0, 5.0, 5.0], 2),
    (0.0, [1.0] * 6 + [0.0] * 2, 3),
    (0.0, [1.0] * 4 + [0.0] * 4, 4),
    (0.9, [0.9] * 8, 0),                      # all the same: no side
    (7.0, [1.0, 1.0] + [0.0] * 6, 1),         # the two at the mean (1) are on neither side: 1 above, 6 below
    (0.1, [0.3, 0.4, 0.4], 1),                # 0.3 at the mean within SIDE_TOLERANCE (computed 0.30000000000000004)
])
def test_a_group_s_minority_side_is_the_smaller_count_either_side_of_its_mean(first, continuations, side):
    """D195: `Group.minority`, the count of the sentences on the smaller side of the group's mean reward."""
    def sentence(reward):
        return Sentence(rows=None, permitted=None, tokens=[], reward=reward)

    group = Group(window=0, branch=0, first=sentence(first), continuations=tuple(sentence(r) for r in continuations))
    assert group.minority == side


def test_a_window_whose_second_pass_differs_is_counted_and_gives_no_sample(setup, monkeypatch):
    """D94: the second pass is checked against the first up to the last branch point."""
    s = setup
    model = _with_module(s["base"])
    calls = []

    def numbers(seed, round_, window):
        calls.append(window)
        return np.random.default_rng([seed, round_, window, len(calls)])      # the second pass draws other numbers

    monkeypatch.setattr(post_branches, "first_numbers", numbers)
    round_ = _round(s, model, [_ahead(s["windows"][0])])
    assert round_.spoken_again == [0]
    # lost at its first row flown, the window has one branch point, its first predicted step: nothing said before it
    # can differ, so it is kept
    assert round_.differed == [] and len(round_.groups) == 1
    # the window alone flies on past its first predicted step: words said before a later branch point differ
    alone = _round(s, model, s["windows"])
    (first,) = alone.first
    assert first.reward < 1.0 and first.loss is None, "the synthetic window does not land under a random prior"
    start = s["stored"].rows.start
    points = branch_points(start, start + len(first.words), DELTA)
    assert len(points) > 1, "the first sentence lasts past a second branch point"
    assert alone.spoken_again == [0] and alone.differed == [0] and alone.groups == []



def test_a_branch_points_copies_are_released_once_its_groups_are_taken(setup, monkeypatch):
    """D183 (4): a branch point's copies (their cache) are released at the end of its block, before the next point's
    copies are made; the groups keep none of their tensors."""
    import weakref

    from ts_transformer.experiments.post_window_loop import WindowLoop

    s = setup
    model = _with_module(s["base"])
    copies, caches = [], []
    real_copy, real_finish = WindowLoop.copy, WindowLoop.finish

    def released():
        return all(r() is None for r in copies + caches)

    def copy(self, windows):
        assert released(), "an earlier branch point's copies are alive at the next point"
        out = real_copy(self, windows)
        copies.append(weakref.ref(out))
        return out

    def finish(self, numbers):
        out = real_finish(self, numbers)
        if any(r() is self for r in copies):                       # the cache as flown (it may have grown)
            caches.extend(weakref.ref(t) for p in self.speaking.speaker.past for t in (p.keys, p.values, p.present))
        return out

    monkeypatch.setattr(WindowLoop, "copy", copy)
    monkeypatch.setattr(WindowLoop, "finish", finish)
    round_ = _round(s, model, s["windows"])
    assert round_.differed == [] and len(round_.groups) >= 2 and len(copies) == len(round_.groups)
    assert caches and released()


def test_a_windows_numbers_are_keyed_by_its_place_in_the_round(setup, monkeypatch):
    """D94: a round's windows are flown in several batches; each draws by its place in the round, not in its batch."""
    s = setup
    model = _with_module(s["base"])
    drawn = []
    real = post_branches.first_numbers
    monkeypatch.setattr(post_branches, "first_numbers", lambda seed, r, w: drawn.append(w) or real(seed, r, w))
    round_ = _round(s, model, [_ahead(s["windows"][0])], places=(7,))
    assert set(drawn) == {7} and round_.spoken_again == [7] and [g.window for g in round_.groups] == [7]
    with pytest.raises(ValueError, match="place in the round"):
        _round(s, model, [_ahead(s["windows"][0])], places=(7, 8))


def test_with_segments_each_row_of_a_first_sentence_is_in_one_group_and_a_continuation_counts_its_segment_only():
    """D170 (b): with ``segment_rows`` a group counts only the rows from its branch point to the next one (or the
    event): every said row of a first sentence is counted in exactly one of its groups, and a continuation's rows after
    its segment in none; without it, each group counts from its point to the event (the code before D170)."""
    from ts_transformer.tests.test_post_loss import FIRST, LENGTH, WORDS, _base, _spoken, _trained
    from ts_transformer.tests.support import prior_sentence

    rng = np.random.default_rng(4)
    rows = [prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(3)]
    _, permitted, _, _ = _spoken(_trained(_base()))
    tokens = [np.zeros((0, len(TOKEN_FEATURES)), dtype=np.float32)] * LENGTH

    def sentence(i, reward):
        return Sentence(rows[i], permitted.select([i]), tokens, reward)

    every = 5                                                       # rows: 20 s at Δ = 4 s
    points = branch_points(FIRST, LENGTH, DELTA, every_s=every * DELTA)
    assert points == [8, 13, 18, 23]
    groups = [Group(0, b, sentence(0, 0.0), (sentence(1, 1.0), sentence(2, 0.0))) for b in points]
    whole = [samples([g], CPU).counted for g in groups]
    cut = [samples([g], CPU, segment_rows=every).counted for g in groups]
    firsts = torch.stack([c[0] for c in cut]).sum(dim=0)
    assert (firsts[FIRST:] == 1).all() and (firsts[:FIRST] == 0).all()                # each said row in one group
    for b, c in zip(points, cut):
        assert c[1, b:b + every].all() and not c[1, b + every:].any() and not c[1, :b].any()   # a continuation's segment
    assert [int(w[0].sum()) for w in whole] == [LENGTH - b for b in points]            # without it: to the event
    with pytest.raises(ValueError, match="whole number"):
        branch_points(FIRST, LENGTH, DELTA, every_s=6.0)


def test_segment_credit_ends_at_the_varied_aircrafts_next_branch_point_where_it_comes_first():
    """D200: a later aircraft of stage D's grid has its first point off the window's grid, its next point less than
    ``segment_rows`` rows after it: that group counts its rows up to that next point only, every said row of the first
    sentence then in exactly one group; a group without a next point (its last) counts its full segment."""
    from ts_transformer.tests.test_post_loss import FIRST, LENGTH, WORDS, _base, _spoken, _trained
    from ts_transformer.tests.support import prior_sentence

    rng = np.random.default_rng(4)
    rows = [prior_sentence(rng, WORDS, candidates=2, rows=LENGTH, first_step=FIRST) for _ in range(3)]
    _, permitted, _, _ = _spoken(_trained(_base()))
    tokens = [np.zeros((0, len(TOKEN_FEATURES)), dtype=np.float32)] * LENGTH

    def sentence(i, reward):
        return Sentence(rows[i], permitted.select([i]), tokens, reward)

    every = 5
    points = [FIRST, FIRST + 2, FIRST + 7, FIRST + 12]                      # its own first step, then the grid
    nexts = points[1:] + [None]
    groups = [Group(0, b, sentence(0, 0.0), (sentence(1, 1.0), sentence(2, 0.0)), next_branch=n)
              for b, n in zip(points, nexts)]
    cut = [samples([g], CPU, segment_rows=every).counted for g in groups]
    firsts = torch.stack([c[0] for c in cut]).sum(dim=0)
    assert (firsts[FIRST:FIRST + 17] == 1).all() and not firsts[:FIRST].any() and not firsts[FIRST + 17:].any()
    assert cut[0][1, FIRST:FIRST + 2].all() and not cut[0][1, FIRST + 2:].any()      # up to its next point only
    assert cut[3][1, FIRST + 12:FIRST + 17].all() and not cut[3][1, FIRST + 17:].any()  # the last: its segment
    unbounded = [samples([g], CPU).counted for g in groups]                   # no segment credit: to the event
    assert [int(u[0].sum()) for u in unbounded] == [LENGTH - b for b in points]


def test_a_groups_next_branch_point_is_its_aircrafts_next_point(setup):
    """D200: `branch_round` gives each group its varied aircraft's next branch point (a Δ row), None at its last;
    stage C's points are a segment apart, so its segment credit is the same as before. Stage C's points every row,
    the first three kept (three groups)."""
    s = setup
    model = _with_module(s["base"])
    every_row = post_branches.stage_c_rules(1337, 0, DELTA, DELTA)
    rules = replace(every_row, varied=lambda *args: [(v, p[:3]) for v, p in every_row.varied(*args)])
    (window,) = s["windows"]
    found = branch_round(model, lambda flights: s["moved_loop"](start_move_of(window)), [window], [0], {0: s["stored"]},
                         s["flights"], s["geometries"], {s["geometry"].code: s["roster"]}, s["finals"], s["words"],
                         interval_s=DELTA, variant="full", edges_reference=s["reference"],
                         faults={s["geometry"].code: {}}, device=CPU, seed=1337, round_=0, split="train",
                         continuations=2, rules=rules)
    branches = [g.branch for g in found.groups]
    assert len(branches) == 3 and branches == sorted(branches)
    assert [g.next_branch for g in found.groups] == [b + 1 for b in branches[:-1]] + [None]



def test_an_aircraft_ended_at_its_first_predicted_step_before_a_row_flown_has_its_observed_rows():
    """Stage D's branch copy (MC6, 2026-10-09): in a loop of later joins there is no observe phase, so an aircraft the
    copy ends at its first predicted step has flown no row; its states are its observed rows, two-dimensional."""
    import numpy as np

    from ts_transformer.experiments.prior_speaking_loop import SpeakingLoop

    loop = object.__new__(SpeakingLoop)
    loop.observed = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    loop._flown = [[], [np.ones(6), np.ones(6) * 2]]
    loop.alive, loop.ended = np.array([False, False]), np.array([True, True])
    loop.left = {}                                        # neither has left the loop (S6)
    assert np.array_equal(loop.states(0), loop.observed[0])
    assert loop.states(1).shape == (5, 6) and np.array_equal(loop.states(1)[3:], [[1.0] * 6, [2.0] * 6])
