"""Stage D, MC12: branch points found backward from each event (multi-aircraft control D182; `multi/backward.py`,
`experiments.post_branches.backward_round`): the event, the candidate points and the judgement of a continuation; the
search's points, its stop, the fine point, point B (and B at the decision point: one group) and its bound on trials
whose first avoiding point is known; on the synthetic window, the points tried and chosen, the continuations of a point
not chosen flown only to the event + 60 s, a group's W of D141; the copies held on another device; the setting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.experiments import post_branches
from ts_transformer.experiments.post_branches import Rules, backward_round
from ts_transformer.experiments.post_window_loop import WindowLoop, start_move_of
from ts_transformer.multi import backward, credit
from ts_transformer.multi.backward import END, GO_AROUND, LOSS, Event, Trial, TRY, avoids, event, points, search
from ts_transformer.multi.credit import Aircraft
from ts_transformer.tests.test_post_window_loop import CPU, DELTA, _with_module, setup  # noqa: F401

STEP = 8                    # 32 s at Δ = 4 s
FINE = 4                    # 16 s


def test_the_candidate_points_go_back_from_the_event_every_32_s_at_most_8_and_not_before_the_first_step():
    long = Event(200, END, outcome="timeout")
    assert points(long, 10, DELTA) == [200 - STEP * j for j in range(1, 9)]
    assert points(Event(40, END, outcome="timeout"), 10, DELTA) == [32, 24, 16]
    assert points(Event(17, END, outcome="timeout"), 10, DELTA) == []          # no point: no group (no early point)
    assert backward.judged_until(long, DELTA) == 215                           # the event + 60 s
    with pytest.raises(ValueError, match="no whole number"):
        backward.ticks_of(32.0, 6.0)


def test_the_event_and_its_kind():
    """D182 (1): D143's event — its own end or the earliest loss it is in — a loss with its other aircraft; a landing
    after a go-around its first go-around when earlier."""
    a = Aircraft("A", 0.0, 4, 30, 30, "B")                         # A answered for a loss with B at 30
    b = Aircraft("B", 1.0, 4, 60, None, None)
    c = Aircraft("C", 0.9, 4, 80, None, None)                       # landed after a go-around
    assert event([a, b, c], 0, "lost_separation", None) == Event(30, LOSS, partner="B")
    assert event([a, b, c], 1, "landed", None) == Event(30, LOSS, partner="A")       # B is the other of A's loss
    assert event([a, b, c], 2, "landed", 50) == Event(50, GO_AROUND)
    assert event([a, b, c], 2, "timeout", None) == Event(80, END, outcome="timeout")
    assert event([a, b, Aircraft("C", 0.9, 4, 20, None, None)], 2, "landed", 50) == Event(20, END, outcome="landed")


def test_a_continuation_avoids_its_event():
    """D182 (3): for every kind of event, v answers for no loss by the judged tick; then a loss's pair keeps its
    separation, a go-around is not said, another end's outcome does not happen. A worse end does not avoid an event."""
    loss, ga, end = Event(30, LOSS, partner="B"), Event(50, GO_AROUND), Event(80, END, outcome="timeout")
    assert avoids(loss, answered=None, partner_answered_v=False, ended=None, go_around=True)
    assert not avoids(loss, answered="B", partner_answered_v=False, ended=None, go_around=False)
    assert not avoids(loss, answered="X", partner_answered_v=False, ended=None, go_around=False)   # any loss of v
    assert not avoids(loss, answered=None, partner_answered_v=True, ended=None, go_around=False)
    assert avoids(ga, answered=None, partner_answered_v=False, ended="landed", go_around=False)
    assert not avoids(ga, answered=None, partner_answered_v=False, ended="landed", go_around=True)
    assert not avoids(ga, answered="B", partner_answered_v=False, ended="lost_separation", go_around=False)
    assert avoids(end, answered=None, partner_answered_v=False, ended=None, go_around=True)
    assert avoids(end, answered=None, partner_answered_v=False, ended="landed", go_around=False)
    assert not avoids(end, answered=None, partner_answered_v=False, ended="timeout", go_around=False)
    assert not avoids(end, answered="B", partner_answered_v=False, ended="lost_separation", go_around=False)
    assert not avoids(end, answered="B", partner_answered_v=False, ended=None, go_around=False)    # its loss, still flown


def _searched(candidates, judged):
    """`search`'s steps on ``candidates``, a point tried answered by ``judged(point)``; its requests in order."""
    calls, steps, answer = [], search(candidates, FINE), None
    while True:
        try:
            kind, p = steps.send(answer)
        except StopIteration as stop:
            return stop.value, calls
        calls.append(("trial" if kind == TRY else "chosen", p))
        answer = judged(p) if kind == TRY else None


def test_the_search_stops_at_the_first_avoiding_point_and_tries_the_fine_point_and_point_b():
    """D182 (4)–(7) on trials whose first avoiding point is known: points 0–1 do not avoid; point 2 avoids only with a
    go-around; point 4 avoids without one (B); the fine point after point 2 avoids."""
    candidates = [100 - STEP * j for j in range(1, 9)]               # 92, 84, …, 36
    decision, b = candidates[2], candidates[4]

    def judged(p):
        if p == decision:
            return [Trial(False, False), Trial(True, True)]
        if p == decision + FINE:
            return [Trial(True, False)]
        if p == b:
            return [Trial(True, False), Trial(False, False)]
        if p < decision:
            return [Trial(True, True)]                                # avoids, with a go-around: not B
        return [Trial(False, False)] * 2

    found, calls = _searched(candidates, judged)
    assert (found.decision, found.fine, found.b) == (decision, decision + FINE, b)
    assert found.groups == (decision, decision + FINE, b)
    assert found.tried == (92, 84, 76, 80, 68, 60)
    assert calls == [("trial", 92), ("trial", 84), ("trial", 76), ("chosen", 76), ("trial", 80), ("chosen", 80),
                     ("trial", 68), ("trial", 60), ("chosen", 60)]     # a point chosen right after it is tried


def test_b_at_the_decision_point_is_one_group_and_a_fine_point_that_does_not_avoid_none():
    candidates = [100 - STEP * j for j in range(1, 9)]
    found, calls = _searched(candidates, lambda p: [Trial(p <= 84, False)])
    assert (found.decision, found.fine, found.b, found.groups) == (84, None, None, (84,))
    assert found.tried == (92, 84, 88) and ("chosen", 88) not in calls


def test_an_event_no_point_avoids_gives_no_group_and_b_stays_within_the_bound():
    candidates = [100 - STEP * j for j in range(1, 9)]
    found, calls = _searched(candidates, lambda p: [Trial(False, False)])
    assert found.groups == () and found.tried == tuple(candidates) and not [c for c in calls if c[0] == "chosen"]
    found, _ = _searched(candidates, lambda p: [Trial(p <= 52, True)])          # avoids only with a go-around
    assert (found.decision, found.b) == (52, None) and found.tried == (92, 84, 76, 68, 60, 52, 56, 44, 36)


def _backward(s, model, rules, continuations=2):
    (window,) = s["windows"]
    return backward_round(model, lambda flights: s["moved_loop"](start_move_of(window)), [window], [0],
                          {0: s["stored"]}, s["flights"], s["geometries"], {s["geometry"].code: s["roster"]}, s["finals"],
                          s["words"], interval_s=DELTA, variant="full", edges_reference=s["reference"],
                          faults={s["geometry"].code: {}}, device=CPU, split="train", continuations=continuations,
                          rules=rules)


def test_a_round_found_backward_on_a_window_whose_first_avoiding_point_is_known(setup, monkeypatch):
    """MC12 on the synthetic window (its one aircraft ends at its time limit, W < 1) with the judgement replaced: the
    continuations avoid from the third point back on, without a go-around. The points tried are the search's; the
    decision point gives the one group (B is it; its fine point does not avoid); every point not chosen is flown to the
    event + 60 s only and the chosen one to the window's end; the group's sentences have the W of D141 of their ends."""
    s = setup
    model = _with_module(s["base"])
    rules = Rules(**credit.rules(1337, 0, DELTA))
    reached, finished = [], []
    events = {}

    def judged(loop, own, v, happened, point):
        events["event"] = happened
        reached.append((point, int(loop.speaking.t - loop.origins[own[0]])))     # in its window's ticks
        return [Trial(point <= happened.tick - 3 * STEP, False) for _ in own]

    real_results = WindowLoop.results

    def results(self):
        out = real_results(self)
        finished.append(out)
        return out

    monkeypatch.setattr(post_branches, "_trials", judged)
    monkeypatch.setattr(WindowLoop, "results", results)
    found = _backward(s, model, rules)
    happened = events["event"]
    assert happened.kind == END and found.spoken_again == [0] and found.differed == []
    until = backward.judged_until(happened, DELTA)
    decision = happened.tick - 3 * STEP
    assert [p for p, _ in reached] == [happened.tick - STEP, happened.tick - 2 * STEP, decision, decision + FINE]
    assert all(t <= until for _, t in reached)                         # every point tried: to the event + 60 s at most
    (group,) = found.groups
    assert group.branch == decision and group.varied == 0 and found.depths_s == [3 * 32.0]
    assert (found.searches, found.found, found.to_end, found.to_event) == (1, 1, 2, 3 * 2)
    *_, ends = finished                                                # the decision point's continuations, to the end
    assert [c.reward for c in group.continuations] == [credit.window_reward([r.reward]) for r in ends]
    assert all(not np.isnan(c.reward) for c in group.continuations)


def test_a_round_found_backward_judges_its_continuations_by_what_they_did(setup):
    """The real judgement on the synthetic window: each group's point is a candidate point or a fine point, its depth
    the seconds to the event; the counts add up."""
    s = setup
    found = _backward(s, _with_module(s["base"]), Rules(**credit.rules(1337, 0, DELTA)))
    assert found.searches == 1 and len(found.groups) == len(found.depths_s) <= 3
    assert found.to_end == 2 * len(found.groups) and found.to_event % 2 == 0
    for group, depth in zip(found.groups, found.depths_s):
        assert depth % 16.0 == 0 and 16.0 <= depth <= 8 * 32.0


def _round_trip_on_the_gpu(directory: str) -> None:
    """The body of the GPU test below, in a process of its own: the synthetic window's loop flown three rows, copied to
    the GPU (every tensor checked there), copied back, and flown to its end beside the loop copied."""
    import copy as copying
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

    from ts_transformer.experiments.post_branches import _batch_loops
    from ts_transformer.tests.test_post_window_loop import window_setup

    s = window_setup(Path(directory), MonkeyPatch())
    model = _with_module(s["base"])
    (window,) = s["windows"]
    window_loop, sources = _batch_loops(model, lambda flights: s["moved_loop"](start_move_of(window)), [window], [0],
                                        {0: s["stored"]}, s["flights"], s["geometries"],
                                        {s["geometry"].code: s["roster"]}, s["finals"], s["words"], interval_s=DELTA,
                                        variant="full", edges_reference=s["reference"],
                                        faults={s["geometry"].code: {}}, device=CPU,
                                        rules=Rules(**credit.rules(1337, 0, DELTA)), loop_options=None)
    loop = window_loop([0])
    numbers = sources(loop, [0])
    while loop.speaking.observing:
        loop.observe()
    for _ in range(3):
        loop.step(post_branches._draws(loop, numbers))
    away = loop.copy([0], device=torch.device("cuda"))
    tensors = [p.keys for p in away.speaking.speaker.past] + [away.speaking.loop.executor.state,
                                                               away.speaking.loop.spoken.value]
    assert all(t.device.type == "cuda" for t in tensors)
    back = away.copy([0], device=CPU)
    twin = [copying.deepcopy(n) for n in numbers]
    a, b = loop.finish(numbers), back.finish(twin)
    assert np.array_equal(a[0].words, b[0].words) and np.array_equal(a[0].states, b[0].states)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_copy_held_on_another_device_and_brought_back_flies_and_says_as_its_original(tmp_path):
    """D182: the second pass's copies held in the host's memory and flown on the GPU later — a window loop copied to
    another device holds every tensor there; copied back, it says and flies what the loop copied does, bit for bit.
    Run in a new Python process: a test process that has used the GPU cannot fork speaking workers later
    (`post_train.Speakers` refuses it), and other tests of the same process do."""
    import os
    import subprocess
    import sys

    code = ("from ts_transformer.tests.test_multi_backward import _round_trip_on_the_gpu; "
            f"_round_trip_on_the_gpu({str(tmp_path)!r})")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)})
    assert done.returncode == 0, done.stderr[-3000:]
    assert not torch.cuda.is_initialized()


def test_the_branching_setting_defaults_to_the_grid_and_a_resume_compares_it(tmp_path):
    from dataclasses import asdict, replace

    from ts_transformer.experiments import post_train
    from ts_transformer.experiments.multi_train import MULTI_CAMPAIGN_SCHEMA, MULTI_CLAIM_READER, MultiSettings
    from ts_transformer.tests.test_multi_train import _multi_settings

    settings = _multi_settings()
    assert settings.branching == backward.GRID
    with pytest.raises(ValueError, match="none of"):
        replace(settings, branching="sideways")
    recorded = asdict(settings)
    del recorded["branching"]                                        # a record before D182 reads as the grid
    assert MultiSettings(**recorded) == settings
    options = dict(schema=MULTI_CAMPAIGN_SCHEMA, reader=MULTI_CLAIM_READER, settings_type=MultiSettings)
    paths = {key: str(tmp_path / "inputs" / key) for key in post_train.INPUT_PATHS}
    out = tmp_path / "campaign"
    post_train.open_campaign(out, {**paths, "settings": recorded, "smoke": True}, {"head": "x", "dirty": False}, {},
                             **options)
    post_train.open_campaign(out, {**paths, "settings": asdict(settings), "smoke": True}, {"head": "y", "dirty": False},
                             {}, **options)
    with pytest.raises(SystemExit, match="other inputs or settings"):
        post_train.open_campaign(out, {**paths, "settings": asdict(replace(settings, branching=backward.BACKWARD)),
                                       "smoke": True}, {"head": "y", "dirty": False}, {}, **options)


def test_stage_ds_batch_record_holds_the_searches_with_backward_only(setup, tmp_path, monkeypatch):
    """MC12: a batch of stage D spoken backward records each group's span and depth, the searches and the
    continuations flown to the event + 60 s and to the end; with the grid its record is the one before D182 (the same
    keys); window_behaviour (R68) captures either round."""
    from dataclasses import replace

    from ts_transformer.experiments import multi_train, window_behaviour
    from ts_transformer.multi.windows import REAL_KIND
    from ts_transformer.tests.test_multi_train import _multi_settings, _stage_c
    from ts_transformer.tests.test_post_train import _context, short_round

    s = setup
    short_round(monkeypatch, s)
    context = _context(s)
    settings = _multi_settings(start=_stage_c(s, tmp_path, context), continuations=2)
    (window,) = s["windows"]
    window = replace(window, span_s=1.0)                         # the real window: its aircraft ends at its time limit
    monkeypatch.setattr(multi_train, "draw_windows", lambda context, settings, rng: ([window], {"drawn": {REAL_KIND: 1}}))
    stage = multi_train.stage_d()
    model = stage.start(context, settings)[0].eval()
    records = {}
    for branching in backward.BRANCHINGS:
        chosen = replace(settings, branching=branching)
        windows, _ = stage.draw(context, chosen, 0)
        (tmp_path / branching).mkdir()
        with window_behaviour.captured_rounds() as found:
            records[branching] = stage.speak_batch(model, context, windows, [0], chosen, 0, tmp_path / branching, 0)
        (spoken,) = found
        assert isinstance(spoken, post_branches.BackwardRound) == (branching == backward.BACKWARD)
    plain = {"groups", "informative_groups", "spoken_again", "differed", "reward_sum", "faulty_steps",
             "losses_reading_fault", "outcomes"}
    assert set(records[backward.GRID]) == plain
    searched = records[backward.BACKWARD]
    assert set(searched) == plain | {"group_depths_s", "searches", "searches_found", "continuations_to_event",
                                     "continuations_to_end", "search_s", "search_steps", "search_rows"}
    assert searched["search_steps"] > 0 and searched["search_rows"] >= 2 * searched["search_steps"]   # K = 2 a step
    assert searched["searches"] == 1 and len(searched["group_depths_s"]) == searched["groups"]
    assert all(span == 1.0 for span, _ in searched["group_depths_s"])
    assert searched["continuations_to_end"] == 2 * searched["groups"]
    assert post_branches.branch_round is multi_train.branch_round and multi_train.backward_round is backward_round


def test_the_searches_of_a_batch_step_together_and_find_what_each_finds_alone(setup, monkeypatch):
    """D188: on the window of three commanded aircraft (`test_post_window_multi._multi`), the searches of its varied
    aircraft stepped together in one loop find what each finds searched alone (its aircraft the only one varied): the
    same groups (point, varied aircraft, depth; each continuation's words, reward and bound). Each point tried is judged
    at its own event + 60 s, or where its continuations ended before; each point chosen is flown to its window's end."""
    from ts_transformer.tests.test_post_window_multi import _multi

    s = setup
    multi, loop_of = _multi(s, joins=(0, 30, 60))
    parts = loop_of.parts
    model = _with_module(s["base"])
    rules = Rules(**credit.rules(1337, 0, DELTA))
    judged, real = [], post_branches._trials

    def trials(loop, own, v, happened, point):
        alive = loop.speaking.alive[[b for w in own for b in loop.members[w]]].any()
        judged.append((v, point, int(loop.speaking.t - loop.origins[own[0]]), backward.judged_until(happened, DELTA),
                       bool(alive)))
        return real(loop, own, v, happened, point)

    monkeypatch.setattr(post_branches, "_trials", trials)

    def round_of(varied):
        return backward_round(model, parts["start"], [multi], [5], parts["sentences"], parts["flights"],
                              s["geometries"], parts["rosters"], s["finals"], s["words"], interval_s=DELTA,
                              variant="full", edges_reference=s["reference"], faults={s["geometry"].code: {}},
                              device=CPU, split="train", continuations=2, rules=replace(rules, varied=varied))

    together = round_of(rules.varied)
    assert together.searches == 3 and len({g.varied for g in together.groups}) == 3       # every aircraft varied
    for v, point, tick, until, alive in judged:                  # each point tried at its own judged tick
        assert tick == until or (tick < until and not alive)
    assert together.to_end == 2 * len(together.groups) and together.to_event == 2 * (len(judged) - len(together.groups))
    alones = [round_of(lambda window, loop, rows, results, v=v: [x for x in rules.varied(window, loop, rows, results)
                                                                  if x[0] == v]) for v in range(3)]
    # the groups in the order of the searches, then of their points chosen
    assert ([(g.varied, g.branch) for g in together.groups]
            == [(h.varied, h.branch) for one in alones for h in one.groups])
    for v, alone in enumerate(alones):
        ours = [(g, d) for g, d in zip(together.groups, together.depths_s) if g.varied == v]
        assert alone.searches == 1 and len(alone.groups) == len(ours) > 0
        for (g, depth), h, other in zip(ours, alone.groups, alone.depths_s):
            assert (g.window, g.branch, g.varied, depth) == (h.window, h.branch, h.varied, other)
            for x, y in zip(g.continuations, h.continuations):
                assert np.array_equal(x.rows.targets, y.rows.targets) and (x.reward, x.until) == (y.reward, y.until)
                assert all(np.array_equal(a, b) for a, b in zip(x.permitted.masks, y.permitted.masks))   # same shape
