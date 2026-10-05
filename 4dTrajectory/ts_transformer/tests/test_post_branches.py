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
from ts_transformer.instructions.words import COLUMNS
from ts_transformer.post.branches import (
    STATE_BOUND_M, branch_points, continuation_numbers, first_numbers, samples,
)
from ts_transformer.post.loss import one_pass
from ts_transformer.post.scene import INSERTED, INSERTED_SUFFIX, MovedScene
from ts_transformer.post.traffic_attention import parameter_groups
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, _window_loop, _with_module, setup  # noqa: F401


def _round(s, model, windows, continuations=2, round_=0, places=(0,)):
    return branch_round(model, lambda flights: s["loop"](), windows, list(places), {0: s["stored"]}, s["flights"],
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
    # the samples go through the loss of C7
    optimizer = torch.optim.AdamW(parameter_groups(model, prior_lr=1e-4, traffic_lr=1e-3))
    data = batch.rows
    (parts,) = one_pass(model, s["base"], optimizer, [batch], [data])
    assert torch.isfinite(parts.loss)


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
