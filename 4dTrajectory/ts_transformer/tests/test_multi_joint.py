"""Stage D, MC8: version 2 (multi-aircraft control §6.4; D153, D212) — the module beside the traffic attention that reads
the other commanded aircraft of a window at the same tick (`multi/joint.py`), the window loop's window input, the
sentences of a loop placed on a window's ticks (`prior.batch.collate(…, starts=)`) and the window samples of the loss
(`post.branches.samples`, `post.loss`) — on the synthetic window of three commanded aircraft of stage D's loop tests."""

from __future__ import annotations

import copy as copying
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.instructions.words import COLUMNS
from ts_transformer.multi.joint import JointAttention, JointConfig, add_joint_attention, joint_modules
from ts_transformer.multi.tokens import TOKENS_SCHEMA, TokenPart
from ts_transformer.post.branches import Group, Placed, Sentence, Valued, samples
from ts_transformer.post.edges import TOKEN_FEATURES
from ts_transformer.post.loss import Samples, base_log_probability
from ts_transformer.post.traffic_attention import Traffic, add_token_part, parameter_groups, traffic_of
from ts_transformer.prior.batch import collate
from ts_transformer.prior.speaker import SAID
from ts_transformer.prior.train import masked_log_probability
from ts_transformer.tests.test_post_window_loop import CPU, _with_module, setup  # noqa: F401
from ts_transformer.tests.test_post_window_multi import JOINS, _multi, _numbers

TICKS = 40


def _version_2(base, joint_weights: bool):
    """Version 1's model (the traffic attention with weights, the token part at zero) and, beside it, version 2's
    module: at zero, or with weights of its own (``joint_weights``)."""
    model = _with_module(base)
    add_token_part(model, TokenPart.width, TOKENS_SCHEMA)
    add_joint_attention(model, JointConfig(heads=4))
    if joint_weights:
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for module in joint_modules(model):
                for p in module.parameters():
                    p.copy_(torch.randn(p.shape, generator=generator) * 0.3)
    return model.eval()


def _options(s, joint=True):
    return dict(token_part=TokenPart(s["words"]), part_width=TokenPart.width, joint=joint)


#: the tick at which `_flown` ends the first aircraft (``ended``), the others flown on: the anchor ended while the
#: second is said and before the third joins
ENDED_AT = JOINS[1] + 2


def _flown(loop, numbers, until=TICKS, extras=None, ended=False):
    """Step ``loop`` to tick ``until`` (or its end), each row said with its own numbers; ``extras``: each tick's input
    of the added modules (and the rows joined and flown before it), kept; ``ended``: the first aircraft ended at tick
    `ENDED_AT` (as a loss of separation ends it), the others flown on."""
    step = loop.speaking.step

    def kept(numbers_, caller=None, extra=None):
        if extras is not None:
            extras.append((loop.speaking.t, extra, loop.speaking.joined().copy(), loop.speaking.alive.copy()))
        return step(numbers_, caller, extra)

    loop.speaking.step = kept
    while loop.speaking.alive.any() and loop.speaking.t < until:
        if ended and loop.speaking.t == ENDED_AT:
            assert loop.speaking.alive.all()
            loop.speaking.end(np.arange(len(loop.order)) == 0)
        roles = loop.speaking.roles()
        loop.step(np.stack([n.random(len(COLUMNS)) if r == SAID else np.zeros(len(COLUMNS))
                            for n, r in zip(numbers, roles)]))
    return loop


def _window_group(loop, b, joins=JOINS):
    """A group of two window samples of row ``b`` of ``loop`` (its sentence and its window's other sentences, each at
    its join tick), rewards 1 and 0: every row of ``b`` from its first predicted step counted."""
    made = dict(zip(range(len(loop.order)), loop.samples("train")))
    others = tuple(Placed(*made[c], joins[c]) for c in range(len(loop.order)) if c != b)
    one = Sentence(*made[b], 1.0, window=others, join=joins[b])
    return Group(0, loop.speaking.start, one, (Sentence(*made[b], 0.0, window=others, join=joins[b]),))


# ---- the sentences of a loop on a window's ticks (prior §7 item 4)
def test_sentences_placed_at_their_starts_are_the_batch_moved_and_without_starts_todays(setup):
    """D212 (3): no starts, or every start 0, is today's batch bit for bit; with starts each sentence's rows are its own
    from its start, the rows before it not present and not asked, its first predicted step moved with it."""
    s = setup
    _, loop_of = _multi(s)
    loop = _flown(loop_of(_with_module(s["base"])), _numbers(len(JOINS)))
    sentences = loop.speaking.sentences("train")
    today = collate(sentences, CPU)
    for same in (collate(sentences, CPU, None), collate(sentences, CPU, [0] * len(JOINS))):
        assert all(torch.equal(a, b) for a, b in zip(today, same))
    placed = collate(sentences, CPU, list(JOINS))
    for b, (j, sentence) in enumerate(zip(JOINS, sentences)):
        n = sentence.rows
        for name, value in zip(today._fields, today):
            if name != "valid":
                assert torch.equal(getattr(placed, name)[b, j:j + n], value[b, :n]), name
        assert not placed.present[b, :j].any() and not placed.asked[b, :j].any() and not placed.first[b, :j].any()
        assert not placed.present[b, j + n:].any()
    with pytest.raises(ValueError, match="a start of 0 or more"):
        collate(sentences, CPU, [0, -1, 0])


# ---- the module and its input
def test_the_module_reads_only_the_keys_of_its_own_window_never_itself():
    """D153, D212 (1): a row's output moves with a key row of its window, not with a row of its window that is no key,
    another window's row or itself; a row without a key, and a batch without windows, add zero."""
    torch.manual_seed(0)
    module = JointAttention(16, JointConfig(heads=4))
    with torch.no_grad():
        module.out.weight.normal_()
    x = torch.randn(5, 3, 16)
    windows = torch.tensor([[0, 1, 2], [3, 4, -1]])
    key = torch.ones(5, 3, dtype=torch.bool)
    key[2, 1] = False                                       # row 2 is no key at tick 1
    extra = Traffic(torch.zeros(5, 3, 1, len(TOKEN_FEATURES)), torch.zeros(5, 3, 1, dtype=torch.bool), None,
                    windows, key)
    with torch.no_grad():
        out = module(x, extra)
        for row, tick, moves in ((2, 1, False), (2, 0, True), (3, 1, False), (0, 1, False)):
            moved = x.clone()
            moved[row, tick] += torch.randn(16)               # not a shift of every feature (the norm drops it)
            changed = (module(moved, extra)[0, tick] - out[0, tick]).abs().max() > 1e-6
            assert bool(changed) == moves, (row, tick)
        alone = key.clone()
        alone[[1, 2]] = False                                  # row 0's window has no other key
        assert torch.equal(module(x, Traffic(extra.tokens, extra.present, None, windows, alone))[0],
                           torch.zeros(3, 16))
        assert torch.equal(module(x, Traffic(extra.tokens, extra.present)), torch.zeros_like(x))
        assert torch.equal(module(x, None), torch.zeros_like(x))


def test_the_modules_parameters_learn_with_the_traffic_modules(setup):
    """D212 (4): the module's parameters are in the traffic group of `parameter_groups`, every parameter in one group."""
    model = _version_2(setup["base"], joint_weights=False)
    groups = {g["name"]: {id(p) for p in g["params"]} for g in parameter_groups(model, 1e-5, 1e-4)}
    joint = {id(p) for m in joint_modules(model) for p in m.parameters()}
    assert joint and joint <= groups["traffic"] and not joint & groups["prior"]
    assert len(groups["prior"]) + len(groups["traffic"]) == len({id(p) for p in model.parameters()})


@pytest.mark.parametrize("ended", [False, True])
def test_the_window_input_holds_each_window_and_as_keys_its_rows_joined_and_not_ended(setup, ended):
    """D212 (1), (2): at every tick the loop's window input holds the window's rows the speaker holds and, as keys, the
    rows joined and still flown — never an absent one, never one ended (its later rows are not its sentence's): with
    the first aircraft ended at `ENDED_AT`, the others flown on."""
    s = setup
    _, loop_of = _multi(s)
    extras = []
    loop = _flown(loop_of(_version_2(s["base"], joint_weights=False), **_options(s)), _numbers(len(JOINS)),
                  extras=extras, ended=ended)
    absent = gone = 0
    for t, extra, joined, alive in extras:
        assert extra.windows.tolist() == [[0, 1, 2]]
        assert extra.key[:, 0].tolist() == list(joined & alive), t
        absent += int((~joined).sum())
        gone += int((joined & ~alive).any() and (joined & alive).any())
    assert absent and len(extras) > JOINS[-1] and loop.speaking.t > JOINS[-1]
    assert bool(gone) == ended                          # an ended aircraft beside one flown on: not vacuous
    seen = []
    _flown(loop_of(_with_module(s["base"])), _numbers(len(JOINS)), extras=seen)
    assert seen and all(extra.windows is None for _, extra, _, _ in seen)


def test_with_the_module_at_zero_the_loop_says_version_1s_words(setup):
    """MC8: version 2's module starts at zero, so with it and its window input the loop says and flies, aircraft by
    aircraft, what version 1 says and flies, bit for bit."""
    s = setup
    _, loop_of = _multi(s)
    one = _with_module(s["base"])
    add_token_part(one, TokenPart.width, TOKENS_SCHEMA)
    two = copying.deepcopy(one)
    add_joint_attention(two, JointConfig(heads=4))
    a = loop_of(one.eval(), **_options(s, joint=False)).run(_numbers(len(JOINS)))
    b = loop_of(two.eval(), **_options(s)).run(_numbers(len(JOINS)))
    for x, y in zip(a, b):
        assert np.array_equal(x.words, y.words) and np.array_equal(x.states, y.states)
        assert (x.outcome, x.reward, x.loss_step, x.other) == (y.outcome, y.reward, y.loss_step, y.other)


@pytest.mark.parametrize("ended", [False, True])
def test_a_window_samples_log_probability_is_the_speakers(setup, ended):
    """MC8: a window sample (every sentence of the window at its join tick, its window input) gives each aircraft's
    words said the probability the speaker drew them from, within the float tolerance of post-training §6.4 — with the
    module's weights not at zero, so that the aircraft read one another (the same rows without the window read
    otherwise); also with the first aircraft ended while the others are flown on (it is no key after its end in the
    speaker's loop and in the sample)."""
    s = setup
    _, loop_of = _multi(s)
    model = _version_2(s["base"], joint_weights=True)
    loop = _flown(loop_of(model, **_options(s)), _numbers(len(JOINS)), until=10 ** 6, ended=ended)
    assert not ended or loop.speaking.start + len(loop.speaking.said(0)) == ENDED_AT
    speaker = loop.speaking.speaker
    said_rows = np.stack(speaker.said_rows, axis=1)                                  # [B, rows]
    drawn = np.stack(speaker.drawn_probability, axis=1)                              # [B, rows, 5]
    read_apart = 0
    for b in range(len(JOINS)):
        words = len(loop.speaking.said(b))
        expected = drawn[b][said_rows[b]][:words]                                   # its sentence's rows said
        window = samples([_window_group(loop, b)], CPU, TokenPart.width)
        with torch.no_grad():
            log_p = masked_log_probability(model, window.rows, window.permitted, window.traffic)
        owner = window.counted[0]
        assert int(owner.sum()) == words and len(expected) == words
        ratio = np.exp(log_p[0][owner].numpy() - np.log(expected))
        assert np.abs(ratio - 1.0).max() < 1e-5, b
        alone = Traffic(window.traffic.tokens, window.traffic.present, window.traffic.part)
        with torch.no_grad():
            apart = masked_log_probability(model, window.rows, window.permitted, alone)
        read_apart += int((apart[0][owner] - log_p[0][owner]).abs().max() > 1e-4)
    assert read_apart                                    # not vacuous: the module changes what they say


def test_no_aircraft_reads_a_word_said_to_another_in_its_own_tick(setup):
    """D148, D153: the aircraft of a tick are said in one call and read one another's rows before their words, so other
    words drawn for one aircraft at a tick leave every other aircraft's probabilities at that tick (and before) as
    they were."""
    s = setup
    _, loop_of = _multi(s)
    model = _version_2(s["base"], joint_weights=True)
    c, tick = 2, JOINS[2] + 4                     # the last aircraft at its first predicted step: every column said
    runs = []
    for shifted in (False, True):
        numbers = _numbers(len(JOINS))
        loop = loop_of(model, **_options(s))
        while loop.speaking.alive.any() and loop.speaking.t <= tick:
            roles = loop.speaking.roles()
            row = np.stack([n.random(len(COLUMNS)) if r == SAID else np.zeros(len(COLUMNS))
                            for n, r in zip(numbers, roles)])
            if shifted and loop.speaking.t == tick:
                row[c] = (row[c] + 0.5) % 1.0
            loop.step(row)
        runs.append(loop)
    first, second = runs
    assert len(first.speaking.said(c)) == len(second.speaking.said(c)) == 1
    last = len(first.speaking.speaker.said_rows) - 1                   # the record of tick ``tick``
    a, b = first.speaking.speaker, second.speaking.speaker
    for other in (0, 1):
        for r in range(last + 1):
            assert np.array_equal(a.drawn_probability[r][other], b.drawn_probability[r][other]), (other, r)
    assert not np.array_equal(first.speaking.said(c)[-1], second.speaking.said(c)[-1])   # not vacuous


# ---- the samples and the loss
def test_window_samples_count_only_their_own_rows_and_keep_whole_windows(setup):
    """D212 (3): a window sample counts its own aircraft's rows from the branch point (its join tick on), the other rows
    of its window none; a window that counts none is refused; a selection keeps whole windows."""
    s = setup
    _, loop_of = _multi(s)
    loop = _flown(loop_of(_version_2(s["base"], joint_weights=False), **_options(s)), _numbers(len(JOINS)))
    window = samples([_window_group(loop, 1)], CPU, TokenPart.width)
    assert window.traffic.windows.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert torch.equal(window.traffic.key, window.rows.present)
    rows = window.counted.any(dim=1).tolist()
    assert rows == [True, False, False, True, False, False]
    assert not window.counted[0, :JOINS[1] + loop.speaking.start].any()
    assert window.advantage[0].max() == 0.5 and window.advantage[3].min() == -0.5
    kept = window.select([3, 4, 5])
    assert kept.traffic.windows.tolist() == [[0, 1, 2]] and torch.equal(kept.counted, window.counted[3:])
    with pytest.raises(ValueError, match="keeps whole windows"):
        window.select([0, 1])
    with pytest.raises(ValueError, match="every window sample counts some rows"):
        Samples(window.rows, window.permitted, window.traffic, window.advantage,
                window.counted & torch.tensor([True, False, False, False, False, False])[:, None])


def test_d220_a_valued_window_sample_carries_its_advantage_row_by_row_at_its_join_tick(setup):
    """D220 with window samples: the unit's first sentence counts its own rows from its earliest point (moved to the
    window's ticks by its join tick), its advantage on each the segment's, 0 on the rows before its join tick and on
    its window's other sentences; its group's continuation the group's advantage. A group alone gives the window
    samples of before, bit for bit."""
    s = setup
    _, loop_of = _multi(s)
    loop = _flown(loop_of(_version_2(s["base"], joint_weights=False), **_options(s)), _numbers(len(JOINS)))
    group = _window_group(loop, 1)                                # rewards 1 (the first sentence) and 0
    start, join = loop.speaking.start, JOINS[1]
    unit = Valued(0, 1, group.first, (start, start + 3), (0.25, 0.75), (group,))
    window = samples([unit], CPU, TokenPart.width)
    assert window.traffic.windows.tolist() == [[0, 1, 2], [3, 4, 5]]
    counted, advantage = window.counted, window.advantage
    assert not counted[0, :join + start].any() and counted[0, join + start:].any()
    assert torch.equal(advantage[0, join + start:join + start + 3], 0.5 * counted[0, join + start:join + start + 3])
    assert torch.equal(advantage[0, join + start + 3:], 0.25 * counted[0, join + start + 3:])
    assert not advantage[0, :join + start].any() and not advantage[[1, 2, 4, 5]].any()
    assert torch.equal(advantage[3], -0.5 * counted[3].to(torch.float32))       # 0 less the mean of 1 and 0
    alone = samples([group], CPU, TokenPart.width)
    before = torch.tensor([0.5, 0, 0, -0.5, 0, 0], dtype=torch.float32)[:, None] * alone.counted   # before D220
    assert torch.equal(alone.advantage.view(torch.int32), before.view(torch.int32))


def test_the_base_reads_only_the_counted_sentences_of_window_samples(setup):
    """D212 (3): the pull's base log-probability of window samples is the base's on the sentences with counted rows (as
    read in full, within the float tolerance) and the model's own elsewhere (no pull); of samples alone it is the full
    read, bit for bit."""
    s = setup
    _, loop_of = _multi(s)
    model = _version_2(s["base"], joint_weights=True)
    loop = _flown(loop_of(model, **_options(s)), _numbers(len(JOINS)))
    window = samples([_window_group(loop, 1)], CPU, TokenPart.width)
    base = s["base"].eval()
    with torch.no_grad():
        log_p = masked_log_probability(model, window.rows, window.permitted, window.traffic)
        full = masked_log_probability(base, window.rows, window.permitted)
        read = base_log_probability(base, window, log_p)
    owners = window.counted.any(dim=1)
    assert torch.allclose(read[owners], full[owners], atol=1e-5)
    assert torch.equal(read[~owners], log_p[~owners])
    alone = samples([Group(0, loop.speaking.start, Sentence(*loop.samples("train", [1])[0], 1.0),
                           (Sentence(*loop.samples("train", [1])[0], 0.0),))], CPU, TokenPart.width)
    with torch.no_grad():
        assert torch.equal(base_log_probability(base, alone, log_p[:2]),
                           masked_log_probability(base, alone.rows, alone.permitted))


# ---- the round and the campaign
@pytest.mark.parametrize("branching", ["grid", "majority"])
def test_a_round_with_whole_windows_gives_each_sentence_its_window_and_its_pass_reaches_the_module(setup, branching):
    """D212 (3): with ``whole_windows`` a round's groups are version 1's (D143's grid, `branch_round`, or D192's
    majority point, `backward_round`), each sentence holding its window's other sentences at their join ticks; with the module at zero a window sample gives its own rows version 1's sample's
    log-probabilities (within the float tolerance), and an update of window samples reaches the module (its output
    layer), one of samples alone does not. (Every aircraft of the synthetic window loses separation, so every group's
    rewards are 0: one continuation is given a reward of 1, which only sets the advantages.)"""
    from ts_transformer.experiments.post_branches import Rules, backward_round, branch_round
    from ts_transformer.multi import credit
    from ts_transformer.post.loss import PassStart, update_step
    from ts_transformer.tests.test_post_window_loop import DELTA

    s = setup
    # the majority's search varies every aircraft with the joins of `test_multi_backward`'s searches
    joins = JOINS if branching == "grid" else (0, 60, 90)
    multi, loop_of = _multi(s, joins=joins)
    parts = loop_of.parts
    model = _version_2(s["base"], joint_weights=False)
    common = dict(interval_s=DELTA, variant="full", edges_reference=s["reference"],
                  faults={s["geometry"].code: {}}, device=CPU, split="train", continuations=2,
                  rules=Rules(**credit.rules(1337, 0, DELTA)), loop_options=_options(s))
    data = (model, parts["start"], [multi], [5], parts["sentences"], parts["flights"], s["geometries"], s["finals"],
            s["words"])
    if branching == "grid":
        alone, whole = (branch_round(*data, **common, seed=1337, round_=0, whole_windows=w) for w in (False, True))
    else:
        alone, whole = (backward_round(*data, **common, majority=2, whole_windows=w) for w in (False, True))
    assert len(alone.groups) == len(whole.groups) and whole.groups

    def informative(group):
        return replace(group, continuations=(replace(group.continuations[0], reward=1.0), *group.continuations[1:]))

    cut = 0
    for a, w in zip(map(informative, alone.groups), map(informative, whole.groups)):
        assert (a.window, a.branch, a.varied, a.next_branch) == (w.window, w.branch, w.varied, w.next_branch)
        for x, y in zip(a.sentences, w.sentences):
            assert not x.window and y.join == joins[w.varied]
            assert [p.join for p in y.window] == [j for m, j in enumerate(joins) if m != w.varied]
            assert np.array_equal(x.rows.time_s, y.rows.time_s) and (x.reward, x.until) == (y.reward, y.until)
        apart, together = samples([a], CPU, TokenPart.width), samples([w], CPU, TokenPart.width)
        with torch.no_grad():
            p_apart = masked_log_probability(model, apart.rows, apart.permitted, apart.traffic)
            p_together = masked_log_probability(model, together.rows, together.permitted, together.traffic)
        owners = together.counted.any(dim=1)
        assert torch.allclose(p_apart[apart.counted], p_together[owners][together.counted[owners]], atol=1e-5)
        # D200's segment credit: the rows counted to the segment's end are version 1's, moved by the join
        segment, segment_together = (samples([g], CPU, TokenPart.width, segment_rows=2) for g in (a, w))
        moved = torch.zeros_like(segment_together.counted[owners])
        own = segment.counted.shape[1]
        moved[:, joins[w.varied]:joins[w.varied] + own] = segment.counted
        assert torch.equal(segment_together.counted[owners], moved)
        cut += int(segment.counted.sum() < apart.counted.sum())
    assert cut                                          # the segment ends before the sentence's end somewhere
    # a later aircraft's group: its counted rows have the anchor beside them (the anchor's end before the others join)
    k = max(range(len(whole.groups)), key=lambda i: whole.groups[i].varied)
    group = informative(whole.groups[k])
    assert group.varied > 0
    sentences = [y.rows for y in group.sentences]
    reached = []
    for chosen in (informative(alone.groups[k]), group):
        model.zero_grad(set_to_none=True)
        update_step(model, PassStart(model), s["base"].eval(), [samples([chosen], CPU, TokenPart.width)],
                    collate(sentences[:1], CPU))
        grads = [m.out.weight.grad for m in joint_modules(model)]
        reached.append(any(g is not None and bool(g.abs().sum() > 0) for g in grads))
    assert reached == [False, True]
    # MC9: the group's samples in pieces of a window each (`Samples.pieces`) give the parts and the gradient of the
    # group in one piece, to float rounding (the data term with the same random numbers)
    one = samples([group], CPU, TokenPart.width)
    windows = one.pieces()
    assert len(windows) == len(one.traffic.windows) > 1 and all(len(p.traffic.windows) == 1 for p in windows)
    assert sum(int(p.counted.sum()) for p in windows) == int(one.counted.sum())
    assert samples([informative(alone.groups[k])], CPU, TokenPart.width).pieces()[0].traffic.windows is None
    out = []
    for pieces in ([one], windows):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        parts = update_step(model, PassStart(model), s["base"].eval(), pieces, collate(sentences[:1], CPU))
        out.append((parts, [None if p.grad is None else p.grad.clone() for p in model.parameters()]))
    (whole, expected), (apart, got) = out
    for name in ("loss", "surrogate", "kl", "data"):
        assert torch.allclose(getattr(apart, name), getattr(whole, name), atol=1e-6, rtol=1e-6), name
    assert (apart.words, apart.clipped) == (whole.words, whole.clipped)
    assert all((g is None) == (e is None) and (e is None or torch.allclose(g, e, atol=1e-6, rtol=1e-5))
               for g, e in zip(got, expected))


def test_a_round_of_version_2_end_to_end(setup, tmp_path, monkeypatch):
    """D212 (5): a campaign with ``joint_heads`` is version 2: its record's and checkpoints' formats, the module's shape
    in the identity, the module at zero at its start (the start is version 1's, bit for bit, beside it); `round_model`
    opens the round and refuses version 1's settings; `multi_settings_of` takes a record whose format is its settings'
    version's only."""
    from ts_transformer.experiments import multi_train, post_train
    from ts_transformer.experiments.multi_train import (
        MULTI_CAMPAIGN_SCHEMA, MULTI_CAMPAIGN_SCHEMA_V2, MULTI_CHECKPOINT_SCHEMA_V2, MULTI_CLAIM_READER, MultiSettings,
        multi_settings_of, stage_d,
    )
    from ts_transformer.multi.windows import REAL_KIND
    from ts_transformer.tests.test_multi_train import _multi_settings
    from ts_transformer.tests.test_post_branches import _ahead
    from ts_transformer.tests.test_post_train import _context, short_round
    from dataclasses import asdict

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    window = replace(_ahead(s["windows"][0]), span_s=1.0)
    monkeypatch.setattr(multi_train, "draw_windows",
                        lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    one = _multi_settings(rounds=1, start=None, traffic_hidden=16, traffic_heads=4)
    two = replace(one, joint_heads=4)
    assert (one.campaign_schema, two.campaign_schema) == (MULTI_CAMPAIGN_SCHEMA, MULTI_CAMPAIGN_SCHEMA_V2)
    first, _ = stage_d().start(context, one)
    second, _ = stage_d().start(context, two)
    held = second.state_dict()
    assert all(torch.equal(held[n], v) for n, v in first.state_dict().items())            # version 1's start beside it
    assert {n for n in held if ".beside." in n} and all(not held[n].any() for n in held if ".beside.out." in n)
    out = tmp_path / "stage_d"
    paths = {key: str(tmp_path / "inputs" / key) for key in post_train.INPUT_PATHS}
    post_train.open_campaign(out, {**paths, "settings": asdict(two), "smoke": True}, {"head": "x", "dirty": False}, {},
                             schema=two.campaign_schema, reader=MULTI_CLAIM_READER, settings_type=MultiSettings)
    post_train.run_campaign(out, two, context, stage=stage_d())
    identity = torch.load(out / "round_0" / "checkpoint.pt", weights_only=False)["identity"]
    assert identity["schema"] == MULTI_CHECKPOINT_SCHEMA_V2 and identity["joint"] == JointConfig(4).to_dict()
    model = multi_train.round_model(context, two, out, 0)
    assert not model.training and joint_modules(model)
    with pytest.raises(ValueError, match="another start, base"):
        multi_train.round_model(context, one, out, 0)
    record = json.loads((out / "campaign.json").read_text())
    assert record["schema"] == MULTI_CAMPAIGN_SCHEMA_V2 and multi_settings_of(record) == two
    with pytest.raises(ValueError, match="whose settings are"):
        multi_settings_of({**record, "schema": MULTI_CAMPAIGN_SCHEMA})
