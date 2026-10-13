"""Stage D, MC12: branch points found backward from each event (multi-aircraft control D182; `multi/backward.py`,
`experiments.post_branches.backward_round`): the event, the candidate points and the judgement of a continuation; the
search's points, its stop, the fine point, point B (and B at the decision point: one group) and its bound on trials
whose first avoiding point is known; the majority point in B's place (D192); on the synthetic window, the points tried
and chosen, the continuations of a point not chosen flown only to the event + 60 s, a group's W of D141; the copies held
on another device; the setting."""

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
from ts_transformer.post.branches import EQUAL, MINORITY_SIDE
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


def _searched(candidates, judged, majority=None):
    """`search`'s steps on ``candidates``, a point tried answered by ``judged(point)``; its requests in order."""
    calls, steps, answer = [], search(candidates, FINE, majority), None
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


def _avoiding(n, go_around=False):
    """Eight trials of a point, the first ``n`` avoiding (with a go-around if ``go_around``)."""
    return [Trial(j < n, go_around and j < n) for j in range(8)]


def test_the_majority_point_is_the_first_point_behind_the_decision_point_where_5_of_8_avoid():
    """D192 on trials whose avoiding counts are known: the decision point has 1 of 8; the next point back 4; the one
    after 5, all with a go-around (counted: no go-around condition) — the majority point; the fine point avoids. On the
    same trials D182 chooses the point of 4 (B: avoiding with no go-around)."""
    candidates = [100 - STEP * j for j in range(1, 9)]               # 92, 84, …, 36
    counts = {92: 0, 84: 0, 76: 1, 80: 2, 68: 4, 60: 5}
    found, calls = _searched(candidates, lambda p: _avoiding(counts[p], go_around=p in (76, 60)), majority=5)
    assert (found.decision, found.fine, found.b, found.majority) == (76, 80, None, 60)
    assert found.groups == (76, 80, 60) and found.kinds == (backward.DECISION, backward.FINE, backward.MAJORITY_POINT)
    assert found.tried == (92, 84, 76, 80, 68, 60)
    assert calls[-2:] == [("trial", 60), ("chosen", 60)]
    point_b, _ = _searched(candidates, lambda p: _avoiding(counts[p], go_around=p in (76, 60)))
    assert point_b.b == 68 and point_b.majority is None                          # D182's B on the same trials


def test_a_decision_point_where_5_of_8_avoid_is_the_majority_point_and_one_group():
    candidates = [100 - STEP * j for j in range(1, 9)]
    found, calls = _searched(candidates, lambda p: _avoiding(8 if p <= 84 else 0, go_around=True), majority=5)
    assert (found.decision, found.fine, found.majority) == (84, None, 84)
    assert found.groups == (84,) and found.kinds == (backward.DECISION,)
    assert found.tried == (92, 84, 88) and [c for c in calls if c[0] == "chosen"] == [("chosen", 84)]


def test_no_majority_point_within_the_bound_leaves_the_decision_points_groups():
    candidates = [100 - STEP * j for j in range(1, 9)]
    found, _ = _searched(candidates, lambda p: _avoiding(4 if p <= 88 else 0), majority=5)
    assert (found.decision, found.fine, found.majority, found.groups) == (84, 88, None, (84, 88))
    assert found.tried == (92, 84, 88, 76, 68, 60, 52, 44, 36)               # every point back, not past the bound


def _backward(s, model, rules, continuations=2, majority=None):
    (window,) = s["windows"]
    return backward_round(model, lambda flights: s["moved_loop"](start_move_of(window)), [window], [0],
                          {0: s["stored"]}, s["flights"], s["geometries"], {s["geometry"].code: s["roster"]}, s["finals"],
                          s["words"], interval_s=DELTA, variant="full", edges_reference=s["reference"],
                          faults={s["geometry"].code: {}}, device=CPU, split="train", continuations=continuations,
                          rules=rules, majority=majority)


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

    def results(self, rows=None):
        out = real_results(self, rows)
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


def test_a_majority_point_out_of_reach_ends_its_continuations_early_and_finds_the_same(setup, monkeypatch):
    """D194 (3) on the synthetic window with the judgement replaced, 4 continuations, the majority 3: the event's point
    avoids 0, the decision point behind it 1 (its fine point 0), the next point 1 and the one behind it 4 (the
    majority point). The point between has 3 failed at its first step (`_failed` replaced too), so its majority is out
    of reach (`multi.backward.out_of_reach`): it leaves the joint loop there, not at the judged tick. The groups, their
    kinds and depths, and every count but the steps and rows flown, are those of the search that flies it to the judged
    tick."""
    s = setup
    model = _with_module(s["base"])
    rules = Rules(**credit.rules(1337, 0, DELTA))
    reached = {}

    def avoiding(point, happened):
        back = (happened.tick - point) // STEP if (happened.tick - point) % STEP == 0 else None
        return {1: 0, 2: 1, 3: 1, 4: 4}.get(back, 0)

    def judged(loop, own, v, happened, point):
        reached.setdefault(point, []).append(int(loop.speaking.t - loop.origins[own[0]]))
        return [Trial(i < avoiding(point, happened), False) for i in range(len(own))]

    def failed(loop, own, part):
        return len(own) - avoiding(part.point, part.search.happened)

    monkeypatch.setattr(post_branches, "_trials", judged)
    monkeypatch.setattr(post_branches, "_failed", failed)
    early = _backward(s, model, rules, continuations=4, majority=3)
    early_reached, reached = reached, {}
    with monkeypatch.context() as patched:
        patched.setattr(backward, "out_of_reach", lambda *args: False)
        full = _backward(s, model, rules, continuations=4, majority=3)
    tick = max(early_reached) + STEP                                   # the event's tick (its point avoids 0)
    behind = tick - 3 * STEP
    assert early.kinds == full.kinds == [backward.DECISION, backward.MAJORITY_POINT]
    assert [g.branch for g in early.groups] == [g.branch for g in full.groups] == [tick - 2 * STEP, tick - 4 * STEP]
    assert [[c.reward for c in g.continuations] for g in early.groups] == \
        [[c.reward for c in g.continuations] for g in full.groups]
    assert (early.depths_s, early.majority_depths_s) == (full.depths_s, full.majority_depths_s)
    assert (early.searches, early.found, early.to_event, early.to_end) == \
        (full.searches, full.found, full.to_event, full.to_end)
    assert early_reached[behind] == [behind] and reached[behind][0] > behind      # judged at its first step, not later
    assert early.search_steps <= full.search_steps and early.search_rows < full.search_rows


def test_a_point_whose_continuations_all_failed_ends_them_early_and_finds_the_same(setup, monkeypatch):
    """D194 (3) for the points the decision point and the fine point are searched with (point B's search, 4
    continuations): the two points before the decision point and the fine point avoid with none, and all 4 have failed
    at their first step (`_failed` replaced), so each leaves the joint loop there; the decision point avoids with 2. The
    groups, their depths and every count but the steps and rows flown are those of the search that flies each point to
    the judged tick."""
    s = setup
    model = _with_module(s["base"])
    rules = Rules(**credit.rules(1337, 0, DELTA))
    reached = {}

    def avoiding(point, happened):
        return 2 if happened.tick - point == 3 * STEP else 0

    def judged(loop, own, v, happened, point):
        reached.setdefault(point, []).append(int(loop.speaking.t - loop.origins[own[0]]))
        return [Trial(i < avoiding(point, happened), False) for i in range(len(own))]

    def failed(loop, own, part):
        return len(own) - avoiding(part.point, part.search.happened)

    monkeypatch.setattr(post_branches, "_trials", judged)
    monkeypatch.setattr(post_branches, "_failed", failed)
    early = _backward(s, model, rules, continuations=4)
    early_reached, reached = reached, {}
    with monkeypatch.context() as patched:
        patched.setattr(backward, "out_of_reach", lambda *args: False)
        full = _backward(s, model, rules, continuations=4)
    tick = max(early_reached) + STEP                                   # the event's tick (its point avoids 0)
    decision = tick - 3 * STEP
    assert [g.branch for g in early.groups] == [g.branch for g in full.groups] == [decision]
    assert [[c.reward for c in g.continuations] for g in early.groups] == \
        [[c.reward for c in g.continuations] for g in full.groups]
    assert (early.depths_s, early.searches, early.found, early.to_event, early.to_end) == \
        (full.depths_s, full.searches, full.found, full.to_event, full.to_end)
    for point in (tick - STEP, tick - 2 * STEP):                       # judged at their first step, not later
        assert early_reached[point] == [point] and reached[point][0] > point
    fine = decision + FINE
    assert early_reached[fine] == [fine] and reached[fine][0] > fine
    assert early_reached[decision] == reached[decision]                # the decision point flown to the judged tick
    assert early.search_steps <= full.search_steps and early.search_rows < full.search_rows


@pytest.mark.parametrize("kind", [LOSS, END, GO_AROUND])
def test_a_continuation_counted_failed_early_is_one_the_judgement_fails(kind):
    """D194 (3): `_failed` counts a continuation only where `_trials` on the same state judges it not to avoid the event
    (so the search's answer cannot change), and counts each failure that stays to the judged tick: a loss v answered
    for, the event's partner answering for a loss with v (a loss event), v ended with the event's outcome (an end). A
    go-around said is left to the judged tick."""
    from types import SimpleNamespace

    from ts_transformer.experiments.post_branches import _Part, _Search, _failed, _trials
    from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED

    cases = ["v answered", "partner answered", "ended so", "ended otherwise", "alive", "go-around"]
    loss = [object() if c == "v answered" else None for c in cases for _ in (0, 1)]
    other = ["P" if c == "v answered" else None for c in cases for _ in (0, 1)]
    for w, c in enumerate(cases):
        if c == "partner answered":
            loss[2 * w + 1], other[2 * w + 1] = object(), f"V{w}"
    alive = np.array([c not in ("ended so", "ended otherwise") or m == 1 for c in cases for m in (0, 1)])
    said = {2 * w: np.full((3, len(COLUMNS)), UNCHANGED) for w in range(len(cases))}
    said[2 * cases.index("go-around")][1, RUNWAY] = RUNWAY_GO_AROUND
    loop = SimpleNamespace(
        members=[[2 * w, 2 * w + 1] for w in range(len(cases))], loss=loss, other=other,
        records=[SimpleNamespace(key=f"V{w}" if m == 0 else "P") for w in range(len(cases)) for m in (0, 1)],
        own_row=lambda b, point: 0,
        speaking=SimpleNamespace(alive=alive, ended=np.zeros(len(alive), bool), start=0, said=lambda b: said[b],
                                 outcome=lambda b: SimpleNamespace(outcome="X" if cases[b // 2] == "ended so" else "Y")))
    happened = Event(40, kind, partner="P" if kind == LOSS else None, outcome="X" if kind == END else None)
    part = _Part(_Search(0, 0, 0, happened, 55, iter(())), 8, len(cases), [])
    failed = {c: _failed(loop, [w], part) for w, c in enumerate(cases)}
    judged = {c: not _trials(loop, [w], 0, happened, 8)[0].avoids for w, c in enumerate(cases)}
    assert all(failed[c] <= judged[c] for c in cases)
    assert failed == {"v answered": 1, "partner answered": int(kind == LOSS), "ended so": int(kind == END),
                      "ended otherwise": 0, "alive": 0, "go-around": 0}
    assert _failed(loop, range(len(cases)), part) == sum(failed.values())


def test_a_point_is_out_of_reach_once_too_few_continuations_are_left_to_choose_it():
    """D194 (3): 5 of 8 make the majority, so 4 failed put it out of reach and 3 do not; a point that needs 1 avoiding
    continuation is out of reach once all 8 failed."""
    assert backward.out_of_reach(4, 8, 5) and not backward.out_of_reach(3, 8, 5)
    assert backward.out_of_reach(1, 2, 2) and not backward.out_of_reach(0, 2, 2)
    assert backward.out_of_reach(8, 8, 1) and not backward.out_of_reach(7, 8, 1)


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


def _executor_kept_on_the_cpu(directory: str) -> None:
    """The body of the GPU test below. It runs in a separate process. The speaker of the synthetic window is on the GPU.
    The loop is flown two times: once with the executor started on the CPU, once with the executor on the GPU. Each
    time the loop flies three rows, is copied to the host (D182), is joined with itself on the GPU (D188), and flies
    to its end."""
    import copy as copying
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

    from ts_transformer.experiments.post_branches import _batch_loops
    from ts_transformer.tests.test_post_window_loop import window_setup

    s = window_setup(Path(directory), MonkeyPatch())
    gpu = torch.device("cuda")
    model = _with_module(s["base"]).to(gpu)
    (window,) = s["windows"]

    def flown(executor: torch.device):
        window_loop, sources = _batch_loops(model, lambda flights: s["moved_loop"](start_move_of(window), executor),
                                            [window], [0], {0: s["stored"]}, s["flights"], s["geometries"],
                                            {s["geometry"].code: s["roster"]}, s["finals"], s["words"],
                                            interval_s=DELTA, variant="full", edges_reference=s["reference"],
                                            faults={s["geometry"].code: {}}, device=gpu,
                                            rules=Rules(**credit.rules(1337, 0, DELTA)),
                                            loop_options={"executor_device": None if executor == gpu else executor})
        loop = window_loop([0])
        numbers = sources(loop, [0])
        while loop.speaking.observing:
            loop.observe()
        for _ in range(3):
            loop.step(post_branches._draws(loop, numbers))
        held = loop.copy([0], device=CPU)
        joint = WindowLoop.together([(held, [0]), (loop, [0])], gpu, None)
        executor_on = {t.device.type for t in (held.speaking.loop.executor.state, joint.speaking.loop.executor.state,
                                               joint.speaking.loop.spoken.value)}
        speaker_on = {p.keys.device.type for p in joint.speaking.speaker.past}
        return executor_on, speaker_on, joint.finish([copying.deepcopy(n) for n in numbers * 2])

    executor_on, speaker_on, on_cpu = flown(CPU)
    assert executor_on == {"cpu"} and speaker_on == {"cuda"}
    executor_on, _, on_gpu = flown(gpu)
    assert executor_on == {"cpu", "cuda"}                    # with the speaker: held on the host, joined on the GPU
    assert len(on_cpu) == len(on_gpu) == 2
    for a, b in zip(on_cpu, on_gpu):
        assert np.array_equal(a.words, b.words) and a.outcome == b.outcome
        assert np.allclose(a.states, b.states, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_an_executor_started_on_the_cpu_stays_there_beside_a_speaker_on_the_gpu(tmp_path):
    """Multi-aircraft control S1. The executor starts on the CPU (``executor_device``) and the speaker is on the GPU.
    The executor stays on the CPU through the host copies and the joins of a backward batch. The loop says the same
    words, and flies the same states to 1e-6 m, as a loop whose executor follows the speaker. The test runs in a new
    Python process, as the test above."""
    import os
    import subprocess
    import sys

    code = ("from ts_transformer.tests.test_multi_backward import _executor_kept_on_the_cpu; "
            f"_executor_kept_on_the_cpu({str(tmp_path)!r})")
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


def test_stage_ds_batch_record_holds_the_searches_found_backward_only(setup, tmp_path, monkeypatch):
    """MC12: a batch of stage D spoken backward records each group's span and depth, the searches and the
    continuations flown to the event + 60 s and to the end; with the majority point (D192) besides each group's kind
    and each majority point's span and depth; with the grid its record is the one before D182 (the same keys);
    window_behaviour (R68) captures every round."""
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
        chosen = replace(settings, branching=branching,              # D195's record beside the majority point's
                         group_weight=MINORITY_SIDE if branching == backward.MAJORITY else EQUAL)
        windows, _ = stage.draw(context, chosen, 0)
        (tmp_path / branching).mkdir()
        with window_behaviour.captured_rounds() as found:
            records[branching] = stage.speak_batch(model, context, windows, [0], chosen, 0, tmp_path / branching, 0)
        (spoken,) = found
        assert isinstance(spoken, post_branches.BackwardRound) == (branching != backward.GRID)
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
    majority = records[backward.MAJORITY]
    assert set(majority) == set(searched) | {"group_kinds", "majority_depths_s", "group_minorities"}
    written = torch.load(tmp_path / backward.MAJORITY / "groups_0.pt", weights_only=False)
    assert majority["group_minorities"] == [g.minority for g in written] and len(written) == majority["informative_groups"]
    assert len(majority["group_kinds"]) == majority["groups"]
    assert majority["group_kinds"].count(backward.DECISION) == majority["searches_found"] == searched["searches_found"]
    assert len(majority["majority_depths_s"]) <= majority["searches_found"]
    assert all(span == 1.0 for span, _ in majority["majority_depths_s"])
    assert post_branches.branch_round is multi_train.branch_round and multi_train.backward_round is backward_round


def test_the_searches_of_a_batch_step_together_and_find_what_each_finds_alone(setup, monkeypatch):
    """D188: on the window of three commanded aircraft (`test_post_window_multi._multi`), the searches of its varied
    aircraft stepped together in one loop find what each finds searched alone (its aircraft the only one varied): the
    same groups (point, varied aircraft, depth; each continuation's words, reward and bound). Each point tried is judged
    at its own event + 60 s, or where its continuations ended before, or where every one failed (D194 (3)); each point
    chosen is flown to its window's end."""
    from ts_transformer.tests.test_post_window_multi import _multi

    s = setup
    # every aircraft searched, two of them in a loss (ts-executor-spec-v11: with joins (0, 30, 60) the third lost
    # separation at its first step and had no branch point)
    multi, loop_of = _multi(s, joins=(0, 40, 80))
    parts = loop_of.parts
    model = _with_module(s["base"])
    rules = Rules(**credit.rules(1337, 0, DELTA))
    judged, real = [], post_branches._trials

    def trials(loop, own, v, happened, point):
        alive = loop.speaking.alive[[b for w in own for b in loop.members[w]]].any()
        out = real(loop, own, v, happened, point)
        judged.append((v, point, int(loop.speaking.t - loop.origins[own[0]]), backward.judged_until(happened, DELTA),
                       bool(alive) and any(t.avoids for t in out)))
        return out

    monkeypatch.setattr(post_branches, "_trials", trials)

    def round_of(varied):
        return backward_round(model, parts["start"], [multi], [5], parts["sentences"], parts["flights"],
                              s["geometries"], parts["rosters"], s["finals"], s["words"], interval_s=DELTA,
                              variant="full", edges_reference=s["reference"], faults={s["geometry"].code: {}},
                              device=CPU, split="train", continuations=2, rules=replace(rules, varied=varied))

    together = round_of(rules.varied)
    assert together.searches == 3 and len({g.varied for g in together.groups}) == 3       # every aircraft varied
    for v, point, tick, until, open_ in judged:                  # each point tried at its own judged tick
        assert tick == until or (tick < until and not open_)
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


def test_a_joint_loops_cache_has_room_for_the_steps_to_its_next_rebuild(setup, monkeypatch):
    """D190 (1): after a rebuild the joint loop's cache has room for the steps to its next rebuild by its parts' schedule
    (a point tried: to its judged tick; the decision point's copy: to the fine point; D183's step where every part is a
    point chosen), and flown to the next rebuild it never grows where the room was so given (a point chosen at its
    judged tick that goes on alone: the loop rebuilt as it is)."""
    from ts_transformer.prior.model import CACHE_STEP_ROWS, Past
    from ts_transformer.tests.test_post_window_multi import _multi

    s = setup
    multi, loop_of = _multi(s, joins=(0, 30, 60))
    parts = loop_of.parts
    rules = Rules(**credit.rules(1337, 0, DELTA))
    built, real_together, real_step, real_grown = [], WindowLoop.together, WindowLoop.step, Past.grown

    def together(parts, device, room, spent=None):
        out = real_together(parts, device, room, spent)
        past = out.speaking.speaker.past[0]
        built.append({"loop": out, "room": room, "capacity": past.keys.shape[2] - (past.rows - past.start), "steps": 0,
                      "grown": 0})
        return out

    def step(self, numbers):
        if built and self is built[-1]["loop"]:
            built[-1]["steps"] += 1
        return real_step(self, numbers)

    def grown(self, capacity):
        if built:                                        # once the joint loop is flown, the only one that says a row
            built[-1]["grown"] += 1
        return real_grown(self, capacity)

    monkeypatch.setattr(WindowLoop, "together", staticmethod(together))
    monkeypatch.setattr(WindowLoop, "step", step)
    monkeypatch.setattr(Past, "grown", grown)
    backward_round(_with_module(s["base"]), parts["start"], [multi], [5], parts["sentences"], parts["flights"],
                   s["geometries"], parts["rosters"], s["finals"], s["words"], interval_s=DELTA, variant="full",
                   edges_reference=s["reference"], faults={s["geometry"].code: {}}, device=CPU, split="train",
                   continuations=2, rules=rules)
    assert len(built) > 1
    for b in built:
        assert b["capacity"] == (CACHE_STEP_ROWS if b["room"] is None else b["room"])
        if b["room"] is not None:
            assert b["steps"] <= b["room"] and b["grown"] == 0
    assert any(b["room"] is not None and b["steps"] == b["room"] for b in built)     # a rebuild by the schedule


def test_a_row_no_longer_flown_leaves_a_joint_loop_and_keeps_its_end(setup):
    """Multi-aircraft control D193 (3), S6. A join keeps in the speaker and the executor only the rows still flown. A row
    that has ended leaves them, and the loop keeps its record. Flown on with the same numbers, the joined loop says the
    words of the loop it came from and flies its states within the executor's bound (D181 (57)). Each row gets the same
    end and the same sentence (its inputs, its permitted words and its tokens); a row that left gets them from its
    record. A copy of a row that left has its record too."""
    import copy as copying

    from ts_transformer.autopilot.conformance import STATE_BOUND_M
    from ts_transformer.tests.test_post_window_multi import _multi, _numbers

    s = setup
    # its second row loses separation, flies on silent and ends first, while the others fly (ts-executor-spec-v11:
    # with joins (0, 10, 20) its third row ended first and the window with it)
    _, loop_of = _multi(s, joins=(0, 10, 60))
    loop = loop_of(_with_module(s["base"]))
    numbers = _numbers(3)
    while loop.speaking.alive.all():                     # flown until a row ends while the others fly
        loop.step(post_branches._draws(loop, numbers))
    gone = ~loop.speaking.alive
    assert gone.any() and loop.speaking.alive.any()
    joint = WindowLoop.together([(loop, [0])], CPU, None)
    assert np.array_equal(joint.speaking.held, ~gone) and sorted(joint.speaking.left) == np.flatnonzero(gone).tolist()
    assert len(joint.speaking.speaker.in_force) == int((~gone).sum())           # the speaker holds the others only
    copied = joint.copy([0])
    assert np.array_equal(copied.speaking.held, joint.speaking.held)
    twin = [copying.deepcopy(n) for n in numbers]           # the same streams, copied before either is drawn
    ours, theirs = loop.finish(numbers), joint.finish(twin)
    for a, b in zip(ours, theirs):
        assert np.array_equal(a.words, b.words) and (a.outcome, a.reward, a.loss_step) == (b.outcome, b.reward,
                                                                                          b.loss_step)
        assert a.states.shape == b.states.shape and np.allclose(a.states, b.states, rtol=0.0, atol=STATE_BOUND_M)
    for (x, px, tx), (y, py, ty) in zip(loop.samples("train"), joint.samples("train")):
        assert np.array_equal(x.targets, y.targets) and np.array_equal(x.time_s, y.time_s)
        assert np.allclose(x.own, y.own, rtol=0.0, atol=1e-6)
        assert all(np.array_equal(a, b) for a, b in zip(px.masks, py.masks)) and np.array_equal(px.time_s, py.time_s)
        assert len(tx) == len(ty) and all(np.allclose(a, b, rtol=0.0, atol=1e-6) for a, b in zip(tx, ty))
    left = int(np.flatnonzero(gone)[0])
    assert copied.speaking.left[left] is joint.speaking.left[left]                # a copy keeps the record


def test_a_join_of_loops_with_rows_that_left_keeps_their_records(setup):
    """S6: a joint loop joined again, with itself twice (a source with rows that left, and a source used twice), says
    and flies what the loop it came from does and gives each row the same end and sentence. A loop whose rows have all
    ended, joined, holds no speaker or executor and gives each row its end and sentence from its records, as does a copy
    of it."""
    import copy as copying

    from ts_transformer.autopilot.conformance import STATE_BOUND_M
    from ts_transformer.tests.test_post_window_multi import _multi, _numbers

    s = setup
    # its second row loses separation, flies on silent and ends first, while the others fly (ts-executor-spec-v11:
    # with joins (0, 10, 20) its third row ended first and the window with it)
    _, loop_of = _multi(s, joins=(0, 10, 60))
    loop = loop_of(_with_module(s["base"]))
    numbers = _numbers(3)
    while loop.speaking.alive.all():
        loop.step(post_branches._draws(loop, numbers))
    joint = WindowLoop.together([(loop, [0])], CPU, None)
    again = WindowLoop.together([(joint, [0]), (joint, [0])], CPU, None)
    assert len(again.speaking.left) == 2 * len(joint.speaking.left)
    assert all(again.speaking.left[b] is joint.speaking.left[b] for b in joint.speaking.left)
    twin = [copying.deepcopy(n) for n in numbers + numbers]
    ours, theirs = loop.finish(numbers), again.finish(twin)
    for a, b in zip(ours + ours, theirs):
        assert np.array_equal(a.words, b.words) and (a.outcome, a.reward) == (b.outcome, b.reward)
        assert np.allclose(a.states, b.states, rtol=0.0, atol=STATE_BOUND_M)
    for (x, px, _), (y, py, _) in zip(loop.samples("train") * 2, again.samples("train")):
        assert np.array_equal(x.targets, y.targets) and np.array_equal(px.time_s, py.time_s)
    ended = WindowLoop.together([(loop, [0])], CPU, None)                 # every row ended: none held
    assert ended.speaking.speaker is None and ended.speaking.loop is None and not ended.speaking.held.any()
    for other in (ended, ended.copy([0])):
        for a, b in zip(loop.results(), other.results()):
            assert np.array_equal(a.words, b.words) and np.array_equal(a.states, b.states) and a.reward == b.reward
        for (x, px, tx), (y, py, ty) in zip(loop.samples("train"), other.samples("train")):
            assert np.array_equal(x.own, y.own) and all(np.array_equal(a, b) for a, b in zip(px.masks, py.masks))
            assert all(np.array_equal(a, b) for a, b in zip(tx, ty))
