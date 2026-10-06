"""Stage C's window loop generalised for windows of several commanded aircraft (post-training §9 item 8;
multi-aircraft control D144–D148, D150, D152): on A26's synthetic flight (the fixture of `test_post_window_loop`), a
window whose anchor is the flight and whose other commanded aircraft are copies of it under their own keys, each
joining the loop a few ticks behind it (one stored sentence, one start state: the executor reads no absolute time)."""

from __future__ import annotations

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.start import NO_MOVE, Loop
from ts_transformer.experiments import post_window_loop
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION, WindowLoop
from ts_transformer.experiments.prior_speaking_loop import flight_numbers
from ts_transformer.instructions.words import COLUMNS, UNCHANGED
from ts_transformer.post.edges import TOKEN_FEATURES
from ts_transformer.post.landings import roster_key
from ts_transformer.post.reward import LANDED
from ts_transformer.post.scene import Joined
from ts_transformer.prior.landings import Landing, LandingIndex
from ts_transformer.prior.speaker import ABSENT, MOST_GO_AROUNDS, SAID
from ts_transformer.tests import test_start
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, _with_module, setup  # noqa: F401

#: The join steps of the window's commanded aircraft: the anchor, a copy 30 s behind it, one 60 s behind.
JOINS = (0, 8, 15)


def _utc_text(time_s):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(time_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _multi(s, joins=JOINS, shifts=None):
    """A window of ``len(joins)`` commanded aircraft (the anchor and copies joining ``joins`` steps after it; with
    ``shifts``, copy k recorded ``joins[k] − shifts[k]`` steps after the anchor and moved ``shifts[k]`` steps earlier, a
    compressed window's), its flights, sentences, observed rows and roster (each one's own landing in it), and
    ``loop_of(model, **options)``: its window loop."""
    shifts = [0] * len(joins) if shifts is None else list(shifts)
    (window,) = s["windows"]
    anchor = window.commanded
    flights, sentences, observed = {}, {}, {}
    _, _, observed0 = s["moved_loop"](NO_MOVE)
    joined = []
    for k, step in enumerate(joins):
        recorded_step = step - shifts[k]                            # its record's row 0 before the move
        record = anchor if k == 0 else anchor.shifted(step * DELTA, DELTA, key=f"{anchor.key}+c{k}")
        flights[k] = {**s["flights"][0], "dataset_id": record.key,
                      "entry_time_utc": _utc_text(_utc_s(s["flights"][0]["entry_time_utc"]) + recorded_step * DELTA)}
        sentences[k], observed[k] = s["stored"], observed0[0]
        if k:
            joined.append(Joined(record, k, shift_s=shifts[k] * DELTA))
    multi = replace(window, joined=tuple(joined))
    code = s["geometry"].code
    idents = s["roster"].runways
    landing = next(x for x in s["roster"].landings if x.flight_key == s["key"])
    landings = list(s["roster"].landings) + [Landing(landing.time_s + (step - shifts[k]) * DELTA, landing.runway,
                                                     roster_key(flights[k]["dataset_id"], code))
                                             for k, step in enumerate(joins) if k]
    roster = LandingIndex(idents, tuple(sorted(landings, key=lambda x: (x.time_s, x.flight_key))), 0, s["roster"].days)
    one, _, _ = s["moved_loop"](NO_MOVE)
    inputs, limit = one.executor.inputs, float(one.executor.time_limit_s[0])

    def start(chosen):
        """The start of a closed loop of the flights ``chosen`` (`branch_round`'s ``start_loop``): copies of the flight,
        each at its join step."""
        count = len(chosen)
        many = FlightInputs(**{f.name: getattr(inputs, f.name).expand(count, *getattr(inputs, f.name).shape[1:]).clone()
                               for f in fields(FlightInputs)})
        loop = Loop(many, [s["geometry"]] * count, [test_start.A320_IAS] * count, [limit] * count, test_start._params(),
                    s["words"], interval_s=DELTA, most_go_arounds=MOST_GO_AROUNDS, device=CPU,
                    join_ticks=np.array([joins[i] for i in chosen]))
        return loop, list(chosen), {i: observed[i] for i in chosen}

    def loop_of(model, **options):
        loop, order, seen = start(list(range(len(joins))))
        return WindowLoop(model, loop, order, [multi], sentences, flights, s["geometries"], {code: roster},
                          s["finals"], s["words"], interval_s=DELTA, variant="full", edges_reference=s["reference"],
                          faults={code: {}}, observed=seen, device=CPU, **options)

    loop_of.parts = dict(start=start, sentences=sentences, flights=flights, rosters={code: roster})
    return multi, loop_of


def _utc_s(text):
    from ts_transformer.prior.landings import utc_s

    return utc_s(text)


def _numbers(count):
    return [flight_numbers(7, 0, k) for k in range(count)]


def test_a_follower_that_answers_for_its_loss_is_silent_and_flies_on_until_the_window_ends(setup):
    """D144, D145: the copies join on the anchor's path behind it; at the tick a copy joins it is in its observed rows
    and answers for nothing, while the anchor, said, answers the loss (neither established: both responsible) and
    becomes silent — reward 0, `LOST_SEPARATION`, "unchanged" in every column from its next row, flown on its words in
    force, never answering again; the next copy answers at its own said rows; the window ends when every commanded
    aircraft is done or silent (here all silent, the last when it is first said), the silent ones halted then."""
    s = setup
    _, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    results = loop.run(_numbers(len(JOINS)))
    assert len(results) == len(JOINS) and loop.silent.all()
    start = loop.speaking.start
    anchor = results[0]
    assert anchor.outcome == LOST_SEPARATION and anchor.reward == 0.0 and anchor.loss_step == JOINS[1]
    assert anchor.other == loop.records[1].key                     # the copy that joined, in its observed rows
    for b, result in enumerate(results):
        assert result.outcome == LOST_SEPARATION and result.loss is not None and result.index == b
        own = result.loss_step - JOINS[b]                            # its own row of the loss
        assert own >= start + 1                                      # answered only once it was said
        said = loop.speaking.said(b)
        assert (said[own - start:] == UNCHANGED).all()               # silent from the row after it answered
        assert loop.end_step(b) == own
    assert results[2].loss_step - JOINS[2] == start + 1             # the last copy, at its first row said
    assert not loop.speaking.alive.any() and loop.speaking.ended.all()


def test_each_aircraft_reads_the_others_of_its_window_once_they_have_joined(setup):
    """D150, D152: a commanded aircraft's tokens hold its window's other commanded aircraft from the tick they join (an
    absent one is never a token) after its recorded ones; a token part of the caller is given, for each row, the rows
    of its other aircraft (None for a recorded one), and its columns follow the edge features."""
    s = setup
    _, loop_of = _multi(s)
    seen = []

    def part(loop, b, others):
        seen.append((loop.speaking.t, b, list(others)))
        return np.full((len(others), 2), float(b), dtype=np.float32)

    model = _with_module(s["base"])
    from ts_transformer.post.traffic_attention import add_token_part

    add_token_part(model, 2, "test-part-v1")
    loop = loop_of(model.eval(), token_part=part, part_width=2)
    numbers = _numbers(len(JOINS))
    while loop.speaking.alive.any() and loop.speaking.t < JOINS[2] + 6:
        roles = loop.speaking.roles()
        loop.step(np.stack([n.random(len(COLUMNS)) if r == SAID else np.zeros(len(COLUMNS))
                            for n, r in zip(numbers, roles)]))
    assert not len(s["windows"][0].others_at(10))                  # the synthetic window has no recorded aircraft
    for t, b, others in seen:
        joined = [c for c, j in enumerate(JOINS) if j <= t and c != b]
        assert None not in others and others == sorted(others) and set(others) <= set(joined), (t, b)
        assert t >= JOINS[b]                                        # an absent row has no tokens
    tokens = loop._tokens
    for b, join in enumerate(JOINS):
        assert len(tokens[b]) == loop.speaking.t - join                # its own rows only
        assert all(row.shape[1] == len(TOKEN_FEATURES) + 2 for row in tokens[b])
        first = tokens[b][0]
        assert len(first) == sum(1 for c, j in enumerate(JOINS) if c != b and j <= join)
        assert (first[:, len(TOKEN_FEATURES):] == float(b)).all()


def test_a_landing_in_the_loop_counts_for_the_other_aircraft_of_its_window_from_its_crossing(setup, monkeypatch):
    """D147 item 2: a commanded aircraft that the executor ends `landed` adds its landing, on its landed runway at its
    crossing time, to the other commanded aircraft's landings of its window (not its own), and not when it is silent."""
    s = setup
    _, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    crossing = {"at_row": 37.5, "runway_index": 0, "cross_m": 0.0, "height_m": 15.0}
    monkeypatch.setattr(loop.speaking.loop, "outcome", lambda b: SimpleNamespace(outcome=LANDED, crossing=crossing))
    before = [index.digest() for index in loop.landings]
    loop._landed(np.array([True, False, False]))
    time_s = loop.records[0].first_step_s + 37.5 * loop.speaking.loop.params.cycle_s
    for b in (1, 2):
        added = [x for x in loop.landings[b].landings if x.flight_key == loop.keys[0]]
        assert added == [Landing(time_s, s["geometry"].candidates[0].ident, loop.keys[0])]
    assert loop.landings[0].digest() == before[0]
    loop.silent[1] = True
    after = [index.digest() for index in loop.landings]
    loop._landed(np.array([False, True, False]))                  # a silent aircraft's landing adds nothing
    assert [index.digest() for index in loop.landings] == after


def test_a_copy_of_a_window_of_several_aircraft_says_and_flies_what_it_does(setup):
    """D94 with several commanded aircraft: a copy of the window taken while its last aircraft has not joined, flown on
    with the same numbers, says and flies what the original does, aircraft by aircraft."""
    s = setup
    _, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    numbers = _numbers(len(JOINS))
    for _ in range(JOINS[2] - 2):
        roles = loop.speaking.roles()
        loop.step(np.stack([n.random(len(COLUMNS)) if r == SAID else np.zeros(len(COLUMNS))
                            for n, r in zip(numbers, roles)]))
    assert loop.speaking.roles()[2] == ABSENT
    states = [n.bit_generator.state for n in numbers]
    copy = loop.copy([0, 0])
    assert copy.members == [[0, 1, 2], [3, 4, 5]] and copy.window_of.tolist() == [0, 0, 0, 1, 1, 1]
    original = loop.finish(numbers)
    twins = [np.random.default_rng() for _ in range(6)]
    for k, twin in enumerate(twins):
        twin.bit_generator.state = states[k % 3]
    copied = copy.finish(twins)
    for k in range(6):
        a, b = copied[k], original[k % 3]
        assert np.array_equal(a.words, b.words) and a.outcome == b.outcome and a.loss_step == b.loss_step
        assert np.allclose(a.states, b.states, rtol=0.0, atol=STATE_BOUND_M)


def test_a_callers_rule_of_who_answers_is_asked_with_the_windows_commanded_aircraft_first(setup):
    """D145's place (post-training §9 item 3): the loop asks the caller's rule, for each loss of a window, which of its
    commanded aircraft answer, with the judged set holding the commanded aircraft still flown first; a rule that
    answers nothing leaves every aircraft spoken."""
    s = setup
    _, loop_of = _multi(s)
    asked = []

    def nobody(window, step, loss, aircraft, commanded):
        asked.append((step, commanded, aircraft.keys[:commanded]))
        return ()

    loop = loop_of(_with_module(s["base"]), answering=nobody)
    loop.run(_numbers(len(JOINS)))
    assert asked and not loop.silent.any()
    keys = [r.key for r in loop.records]
    for step, commanded, held in asked:
        assert list(held) == [k for k in keys if k in held] and commanded == len(held)


def test_the_window_loop_refuses_a_loop_not_started_at_the_windows_join_steps(setup):
    """D150: the loop's join ticks must be the windows' join steps, and every flight of the loop one window's."""
    s = setup
    multi, _ = _multi(s)
    with pytest.raises(ValueError, match="join ticks are not the windows' join steps"):
        _start_wrong(s, multi)


def _start_wrong(s, multi):
    one, order, observed = s["moved_loop"](NO_MOVE)
    loop = one.copy([0, 0, 0])                                     # every join tick 0
    return WindowLoop(s["base"], loop, [0, 1, 2], [multi], {k: s["stored"] for k in range(3)}, {}, s["geometries"],
                      {}, s["finals"], s["words"], interval_s=DELTA, variant="full", edges_reference=s["reference"],
                      faults={}, observed={}, device=CPU)


def test_a_compressed_windows_moved_aircraft_reads_its_inputs_at_its_moved_time(setup):
    """Multi-aircraft control D146, §6.3 item 2: in a compressed window a commanded aircraft moved earlier by its shift
    joins at its moved row 0 and reads its landings at its moved time (its entry time moved), its own landing at its
    moved time too; its states and words are those of the window without the move, joined at the same tick."""
    s = setup
    shift = -4                                                      # recorded 4 steps later, moved 4 steps earlier
    _, plain_of = _multi(s)
    _, moved_of = _multi(s, shifts=(0, shift, 0))
    plain, moved = plain_of(_with_module(s["base"])), moved_of(_with_module(s["base"]))
    entry = moved.speaking.rows_of.entry_utc_s
    assert entry[1] == plain.speaking.rows_of.entry_utc_s[1]        # its moved entry: where the plain copy enters
    own = [x for x in moved.landings[1].landings if x.flight_key == moved.keys[1]]
    plain_own = [x for x in plain.landings[1].landings if x.flight_key == plain.keys[1]]
    assert len(own) == 1 and own[0].time_s == plain_own[0].time_s   # its own landing at its moved time
    a, b = plain.run(_numbers(len(JOINS))), moved.run(_numbers(len(JOINS)))
    for x, y in zip(a, b):
        assert np.array_equal(x.words, y.words) and np.array_equal(x.states, y.states) and x.outcome == y.outcome


def test_another_commanded_aircrafts_token_is_its_state_at_the_tick(setup):
    """D152, D148, review_guide §5 (an aircraft reads the others at step t or before): a commanded aircraft's token of
    another commanded aircraft is the edge features of that aircraft's state at the same tick — the state it is flown
    from in that row, never a later one."""
    from ts_transformer.post.edges import tokens
    from ts_transformer.post.traffic import scene_aircraft  # noqa: F401  (the scene's order: recorded, then commanded)

    s = setup
    _, loop_of = _multi(s)
    loop = loop_of(_with_module(s["base"]))
    numbers = _numbers(len(JOINS))
    checked = 0
    while loop.speaking.alive.any() and loop.speaking.t < JOINS[2] + 3:
        t = loop.speaking.t
        joined = loop.speaking.joined()
        owns = {b: loop._own(b) for b in range(len(JOINS)) if joined[b] and loop.speaking.alive[b]}
        said_now = loop.speaking.said_now()
        loop.step(np.stack([n.random(len(COLUMNS)) if said else np.zeros(len(COLUMNS))
                            for n, said in zip(numbers, said_now)]))
        for b, own in owns.items():
            others = [owns[c] for c in owns if c != b]
            if not others:
                continue
            from ts_transformer.experiments.post_window_loop import _concat

            g = loop.geometries[b]
            expected = tokens(own, _concat(others), g, loop.separations[g.code], loop.step_s)
            assert np.array_equal(loop._tokens[b][t - JOINS[b]], expected), (t, b)
            checked += 1
    assert checked >= 5


def _round_of(s, model, rules, continuations=2):
    """`branch_round` of the window of several aircraft, under ``rules``."""
    from ts_transformer.experiments.post_branches import branch_round

    multi, loop_of = _multi(s)
    parts = loop_of.parts
    return branch_round(model, parts["start"], [multi], [5], parts["sentences"], parts["flights"], s["geometries"],
                        parts["rosters"], s["finals"], s["words"], interval_s=DELTA, variant="full",
                        edges_reference=s["reference"], faults={s["geometry"].code: {}}, device=CPU, seed=1337,
                        round_=0, split="train", continuations=continuations, rules=rules)


def _rules(continuation, varied_points):
    from ts_transformer.experiments.post_branches import Rules

    return Rules(first=lambda place, member: np.random.default_rng([1, place, member]), continuation=continuation,
                 reward=lambda results: float(sum(r.reward for r in results)),
                 again=lambda results: sum(r.reward for r in results) < len(results), varied=varied_points)


def test_a_continuation_of_a_varied_aircraft_on_its_own_numbers_repeats_the_first_sentence(setup):
    """D142, D94 with several commanded aircraft: at a branch point of a varied aircraft, its continuations copy the
    window; every other aircraft goes on from its own first-sentence stream, so a continuation whose varied aircraft is
    given its own first-sentence stream too (advanced past the rows it said) says and flies the first sentence, for
    every aircraft — the group's sentences equal the first, its rewards the window's sum, nothing differs; the group
    names its varied aircraft, its branch point in that aircraft's own rows, and counts its rows only up to its event."""
    s = setup
    start = 4                                                        # the observed rows at Δ = 4 s
    asked = []

    def continuation(place, member, varied, tick, k):
        asked.append((member, varied, tick, k))
        own = np.random.default_rng([1, place, member])
        own.random((tick - JOINS[member] - start, len(COLUMNS)))     # past the rows it said before the tick
        return own

    def varied(window, loop, rows):
        return [(1, [JOINS[1] + start + 1])]                          # copy 1, one tick after its first predicted step

    finished = []
    real_finish = WindowLoop.finish

    def finish(loop, numbers):                                       # every pass's ends, the copies' included
        ends = real_finish(loop, numbers)
        finished.append(ends)
        return ends

    import ts_transformer.experiments.post_window_loop as module

    original = module.WindowLoop.finish
    module.WindowLoop.finish = finish
    try:
        round_ = _round_of(s, _with_module(s["base"]), _rules(continuation, varied))
    finally:
        module.WindowLoop.finish = original
    assert round_.spoken_again == [5] and round_.differed == []
    first_ends, copies = finished[0], finished[1]                    # the first pass, then the continuations
    assert len(copies) == 2 * len(JOINS)
    for k, end in enumerate(copies):                                 # every aircraft of each continuation
        mine = first_ends[k % len(JOINS)]
        assert np.array_equal(end.words, mine.words) and end.outcome == mine.outcome and end.reward == mine.reward
        assert np.allclose(end.states, mine.states, rtol=0.0, atol=STATE_BOUND_M)
    assert {(m, v) for m, v, _, _ in asked} == {(1, 1)}               # only the varied aircraft is asked for numbers
    (group,) = round_.groups
    assert group.varied == 1 and group.branch == start + 1 and group.window == 5
    first = group.first
    for sentence in group.continuations:
        assert np.array_equal(sentence.rows.targets, first.rows.targets) and sentence.reward == first.reward
        assert sentence.until == first.until
    assert first.reward == sum(r.reward for r in round_.first)
    assert first.until is not None and first.until <= len(first.rows.time_s)


def test_a_varied_aircraft_on_new_numbers_gives_the_group_of_its_own_rows(setup):
    """D142: a continuation of the varied aircraft on new numbers changes its words after the branch point; the
    samples of the group count only its rows from the branch point to its event (`Sentence.until`), the advantage the
    window's reward less the group's mean."""
    from dataclasses import replace as dc_replace

    from ts_transformer.post.branches import samples

    s = setup
    start = 4
    rules = _rules(lambda place, member, v, tick, k: np.random.default_rng([9, place, member, tick, k]),
                   lambda window, loop, rows: [(0, [start + 1])])
    round_ = _round_of(s, _with_module(s["base"]), rules, continuations=3)
    (group,) = round_.groups
    assert group.varied == 0 and len(group.continuations) == 3
    assert any(not np.array_equal(c.rows.targets, group.first.rows.targets) for c in group.continuations)
    rewarded = dc_replace(group, continuations=(dc_replace(group.continuations[0], reward=group.first.reward + 1.0),
                                                *group.continuations[1:]))
    batch = samples([rewarded], CPU)
    for k, sentence in enumerate(rewarded.sentences):
        counted = batch.counted[k].numpy()
        rows = np.flatnonzero(counted)
        assert rows.min() >= group.branch and rows.max() < sentence.until


def test_the_check_of_the_second_pass_reads_an_aircraft_ended_before_the_point_and_one_not_yet_said(setup):
    """D94 for every aircraft: an aircraft that the executor ended before the window's last branch point is compared on
    its whole flight (the second pass says and flies the first's: nothing differs), and one whose only branch point is
    its own first predicted step has nothing said to compare — neither makes the window differ, nor fails."""
    s = setup
    start = 4
    pairs = {}

    def at_its_end(window, loop, rows):                              # the later aircraft, one row before its own end
        b = rows[1]
        pairs["points"] = [JOINS[1] + loop.end_step(b) - 1]
        return [(1, pairs["points"])]

    rules = _rules(lambda place, member, v, tick, k: np.random.default_rng([9, place, member, tick, k]), at_its_end)
    from ts_transformer.experiments.post_branches import branch_round

    def run(rules, joins):
        multi, loop_of = _multi(s, joins=joins)
        parts = loop_of.parts
        return branch_round(_with_module(s["base"]), parts["start"], [multi], [5], parts["sentences"],
                            parts["flights"], s["geometries"], parts["rosters"], s["finals"], s["words"],
                            interval_s=DELTA, variant="full", edges_reference=s["reference"],
                            faults={s["geometry"].code: {}}, device=CPU, seed=1337, round_=0, split="train",
                            continuations=2, rules=rules)

    late = run(rules, (0, 200))
    every = int(round(DELTA / s["words"].spec.step_s))
    assert len(late.first[0].states) < pairs["points"][0] * every + 1      # the anchor ended before the point
    assert late.spoken_again == [5] and late.differed == [] and len(late.groups) == 1
    first = run(_rules(lambda place, member, v, tick, k: np.random.default_rng([9, place, member, tick, k]),
                       lambda window, loop, rows: [(1, [40 + start])]), (0, 40))
    assert first.differed == []
