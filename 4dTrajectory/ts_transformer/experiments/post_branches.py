"""One round's samples by branch training (post-training §2 item 9, §8 C6; D37, D94): a module shared by the runners of
stage C, not a runner.

1. **The first pass**: every window is spoken one time (`post_window_loop.WindowLoop`), each with its own random
   numbers (`post.branches.first_numbers`: the seed, the round, the window's place in the round, which the caller gives:
   a round's windows are flown in several batches). A window whose reward is 1 gives no sample.
2. **The second pass**: the windows whose reward is less than 1 are spoken again with the random numbers of their
   first sentences. At each branch point (`post.branches.branch_points`), every such window whose event is later is
   copied `CONTINUATIONS` times (`WindowLoop.copy`: the executor's loop, the speaker, the window's own state), and the
   continuations of that branch point of all windows are flown as one batch, each with its own numbers
   (`continuation_numbers`).
3. **The check**: the second pass says the first pass's words and flies its states within `STATE_BOUND_M` up to each
   window's last branch point; a window that differs is counted, reported and gives no sample (D94).

The windows of one batch command different flights (a loop holds each flight once): the caller splits a round's windows
into such batches. A window spoken again is halted after its last branch point.

**A window of several commanded aircraft** (multi-aircraft control D141–D143, D149; post-training §9 item 9): the
caller gives the rules (`Rules`) — each aircraft's numbers of the first sentence and of a continuation, the window's
reward, whether a window is spoken again, and the aircraft it varies with their branch points (ticks of the window's
grid). A continuation of a varied aircraft v copies the window: v draws the continuation's numbers, every other
aircraft goes on drawing from its own first-sentence stream from where it was at the branch point (D142); the group is
v's sentences with the window's rewards, each counted up to v's event (`post.branches.Sentence.until`). The check of
D94 holds for every aircraft of a window spoken again. `stage_c_rules` are stage C's: its one aircraft, D94's numbers,
its reward, spoken again below 1, its branch points from its first predicted step to its event.
"""

from __future__ import annotations

import copy as copying
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.start import Loop
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION, WindowLoop, WindowResult
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.words import COLUMNS, RUNWAY, RUNWAY_GO_AROUND, Words
from ts_transformer.multi import backward, credit
from ts_transformer.post.branches import (
    BRANCH_EVERY_S, CONTINUATIONS, STATE_BOUND_M, Group, Sentence, branch_points, continuation_numbers, first_numbers,
)
from ts_transformer.post.reward import LANDED
from ts_transformer.post.scene import Window
from ts_transformer.prior.landings import LandingIndex
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final


@dataclass(frozen=True)
class BranchRound:
    """One batch of a round: the first pass's ends (in the batch's order of rows), the branch groups (`Group.window`: the
    window's place in the round), and the windows spoken again and those whose second pass differed from their first
    (places in the round)."""

    first: list[WindowResult]
    groups: list[Group]
    spoken_again: list[int]
    differed: list[int]


@dataclass(frozen=True)
class Rules:
    """A caller's rules of branch training (module docstring), for a window at its place in the round:

    - ``first(place, member)``: the numbers of a commanded aircraft's first sentence;
    - ``continuation(place, member, varied, tick, k)``: the numbers of continuation ``k`` of the aircraft ``varied``
      at the branch point ``tick``, for its aircraft ``member`` (``varied``'s: new numbers; the caller's other aircraft
      go on from their first-sentence streams and are not asked);
    - ``reward(results)``: the window's reward from its aircraft's ends;
    - ``again(results)``: whether the window is spoken again;
    - ``varied(window, loop, rows, results)``: the aircraft it varies (places among its commanded aircraft) and each
      one's branch points (ticks of the window's grid), from the first pass's ``loop``, the window's ``rows`` in it and
      their ends ``results``."""

    first: Callable[[int, int], np.random.Generator]
    continuation: Callable[[int, int, int, int, int], np.random.Generator]
    reward: Callable[[Sequence[WindowResult]], float]
    again: Callable[[Sequence[WindowResult]], bool]
    varied: Callable[[Window, WindowLoop, Sequence[int], Sequence[WindowResult]], list[tuple[int, list[int]]]]


def stage_c_rules(seed: int, round_: int, interval_s: float, every_s: float = BRANCH_EVERY_S) -> Rules:
    """Stage C's rules (module docstring; D37, D94): one aircraft, its reward, spoken again below 1, its branch points from
    its first predicted step to its event, every ``every_s`` (D37's, or the campaign's `Settings.branch_every_s`, P55)."""
    def varied(window: Window, loop: WindowLoop, rows: Sequence[int], results: Sequence[WindowResult]
               ) -> list[tuple[int, list[int]]]:
        (b,) = rows
        return [(0, branch_points(loop.speaking.start, loop.end_step(b), interval_s, every_s))]

    return Rules(first=lambda place, member: first_numbers(seed, round_, place),
                 continuation=lambda place, member, v, tick, k: continuation_numbers(seed, round_, place, tick, k),
                 reward=lambda results: results[0].reward, again=lambda results: results[0].reward < 1.0,
                 varied=varied)


def same_prefix(a: np.ndarray, b: np.ndarray, rows: int, bound_m: float) -> bool:
    """Whether the first ``rows`` state rows of ``a`` and ``b`` (`STATE_COLUMNS`) agree: their positions and heights
    within ``bound_m``."""
    if len(a) < rows or len(b) < rows:
        return False
    return bool(np.max(np.abs(a[:rows, :3] - b[:rows, :3]), initial=0.0) <= bound_m)


def _draws(loop: WindowLoop, numbers: Sequence[np.random.Generator]) -> np.ndarray:
    """The next row's numbers of every row of ``loop``: five from each row's own source where it is said (the same
    draws as it says its rows alone)."""
    said_now = loop.speaking.said_now()
    return np.stack([n.random(len(COLUMNS)) if said else np.zeros(len(COLUMNS)) for n, said in zip(numbers, said_now)])


def branch_round(model: Prior, start_loop: Callable[[Sequence[int]], tuple[Loop, list[int], dict[int, np.ndarray]]],
                 windows: Sequence[Window],
                 places: Sequence[int], sentences: Mapping[int, ClosedLoopSentence], flights: Mapping[int, Mapping[str, Any]],
                 geometries: Mapping[str, AirportGeometry], rosters: Mapping[str, LandingIndex],
                 finals: Mapping[str, Sequence[Final]], words: Words, *, interval_s: float, variant: str,
                 edges_reference: Path, faults: Mapping[str, Mapping[str, frozenset[int]]], device: torch.device,
                 seed: int, round_: int, split: str,
                 continuations: int = CONTINUATIONS, rules: Rules | None = None,
                 loop_options: Mapping[str, Any] | None = None) -> BranchRound:
    """One batch ``windows`` of a round (module docstring), ``places`` their places in the round (the key of their random
    numbers); ``start_loop`` starts the closed loop of given flights (their places in the split's signals) with their
    windows' moves and join steps (`autopilot.start.Start.moved`, `post_window_loop.start_move_of`) and returns it with
    its order and the observed rows it gave back; ``rules`` the caller's (`stage_c_rules` when not given);
    ``loop_options`` the window loop's options (its rule of who answers, its token part)."""
    rules = stage_c_rules(seed, round_, interval_s) if rules is None else rules
    window_loop, sources = _batch_loops(model, start_loop, windows, places, sentences, flights, geometries, rosters,
                                        finals, words, interval_s=interval_s, variant=variant,
                                        edges_reference=edges_reference, faults=faults, device=device, rules=rules,
                                        loop_options=loop_options)
    everything = list(range(len(windows)))
    first = window_loop(everything)
    results = first.run(sources(first, everything))
    ends = [[results[b] for b in rows] for rows in first.members]
    again = [w for w in everything if rules.again(ends[w])]
    if not again:
        return BranchRound(results, [], [], [])
    #: each window spoken again (its place in ``again``): its varied aircraft and their branch points (ticks)
    varied = {k: rules.varied(windows[w], first, first.members[w], ends[w]) for k, w in enumerate(again)}
    read = sorted({first.members[again[k]][v] for k in varied for v, points in varied[k] if points})
    first_samples = dict(zip(read, first.samples(split, read)))       # the varied aircraft's only (D181 (58))
    second = window_loop(again)
    numbers = sources(second, again)
    while second.speaking.observing:
        second.observe()
    groups: list[tuple[int, Group]] = []
    last = {k: max((p[-1] for _, p in v if p), default=-1) for k, v in varied.items()}
    final = max([second.speaking.start] + [t for t in last.values() if t >= 0])
    while True:
        t = second.speaking.t
        due = [(k, v) for k in range(len(again)) for v, points in varied[k] if t in points]
        if due:
            copies = second.copy([k for k, _ in due for _ in range(continuations)])
            drawn = []
            for k, v in due:
                rows = second.members[k]
                for j in range(continuations):
                    for m, b in enumerate(rows):
                        drawn.append(rules.continuation(places[again[k]], m, v, t, j) if m == v
                                     else copying.deepcopy(numbers[b]))
            ends_k = copies.finish(drawn)
            read = [copies.members[i * continuations + j][v] for i, (_, v) in enumerate(due) for j in range(continuations)]
            made = dict(zip(read, copies.samples(split, read)))
            for i, (k, v) in enumerate(due):
                w = again[k]
                b_first = first.members[w][v]
                rows, permitted, tokens = first_samples[b_first]
                kept = []
                for j in range(continuations):
                    window = i * continuations + j
                    members = copies.members[window]
                    reward = rules.reward([ends_k[b] for b in members])
                    b_copy = members[v]
                    kept.append(Sentence(*made[b_copy], reward, _until(copies, b_copy, len(windows[w].joined))))
                join = int(second.speaking.join_ticks[second.members[k][v]])
                groups.append((k, Group(places[w], t - join, Sentence(rows, permitted, tokens, rules.reward(ends[w]),
                                                                       _until(first, b_first, len(windows[w].joined))),
                                        tuple(kept), varied=v)))
            del copies, drawn, ends_k, made     # released before the next point's copies (D183 (4)): one set alive
            done = np.zeros(len(second.order), dtype=bool)
            for k in range(len(again)):
                if last[k] >= 0 and last[k] <= t:
                    done[second.members[k]] = True
            if (done & second.speaking.alive).any():           # past its last branch point: nothing more to copy
                second.speaking.end(done & second.speaking.alive)
        if t >= final or not second.speaking.alive.any():
            break
        second.step(_draws(second, numbers))
    differed = _differed(first, second, again, last, places)
    return BranchRound(results, [g for _, g in groups if g.window not in differed], [places[w] for w in again],
                       differed)


def _batch_loops(model: Prior, start_loop: Callable[[Sequence[int]], tuple[Loop, list[int], dict[int, np.ndarray]]],
                 windows: Sequence[Window], places: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                 flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                 rosters: Mapping[str, LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                 interval_s: float, variant: str, edges_reference: Path,
                 faults: Mapping[str, Mapping[str, frozenset[int]]], device: torch.device, rules: Rules,
                 loop_options: Mapping[str, Any] | None
                 ) -> tuple[Callable[[Sequence[int]], WindowLoop],
                            Callable[[WindowLoop, Sequence[int]], list[np.random.Generator]]]:
    """A batch's loops (`branch_round`'s module docstring; refused unless its windows command different flights in the
    order of their signals and have each its own place): the window loop of some of its windows (their places in the
    batch) and each row's first-sentence numbers in such a loop."""
    options = dict(loop_options or {})
    commanded = [i for w in windows for i in w.signal_indices]
    if [w.signal_index for w in windows] != sorted({w.signal_index for w in windows}) or \
            len(set(commanded)) != len(commanded):
        raise ValueError("the windows of a batch command different flights, in the order of their places in the signals")
    if len(places) != len(windows) or len(set(places)) != len(places):
        raise ValueError("one place in the round for each window of the batch, each its own")

    def window_loop(batch: Sequence[int]) -> WindowLoop:
        loop, started, observed = start_loop(sorted(i for b in batch for i in windows[b].signal_indices))
        return WindowLoop(model, loop, started, [windows[b] for b in batch], sentences, flights, geometries, rosters,
                          finals, words, interval_s=interval_s, variant=variant, edges_reference=edges_reference,
                          faults=faults, observed=observed, device=device, **options)

    def sources(loop: WindowLoop, batch: Sequence[int]) -> list[np.random.Generator]:
        """Each row's first-sentence numbers (its window's place, its place among the window's aircraft)."""
        return [rules.first(places[batch[int(loop.window_of[b])]], int(loop.member_of[b])) for b in range(len(loop.order))]

    return window_loop, sources


def _differed(first: WindowLoop, second: WindowLoop, again: Sequence[int], last: Mapping[int, int],
              places: Sequence[int]) -> list[int]:
    """The check of D94 (module docstring): the places of the windows spoken again whose second pass (``second``, the
    windows ``again``) did not say the first pass's words or fly its states within `STATE_BOUND_M` up to each one's
    last point (``last``, by its place in ``again``; below 0: none)."""
    differed = []
    for k, w in enumerate(again):
        if last[k] < 0:
            continue
        for b_second, b_first in zip(second.members[k], first.members[w]):
            join = int(second.speaking.join_ticks[b_second])
            said = last[k] - join - second.speaking.start
            if said < 0 or (said == 0 and join > 0):             # nothing said to it before the point
                continue
            ours, theirs = second.speaking.states(b_second), first.speaking.states(b_first)
            if said == 0 and len(ours) == len(second.speaking.observed[b_second]):
                continue                    # ended at its first predicted step before a row flown: nothing to compare
            # an aircraft the executor ended before the point: its whole flight, the same length in both passes
            rows = min((last[k] - join) * second.every + 1, len(theirs))
            ended = rows < (last[k] - join) * second.every + 1
            if not (np.array_equal(second.speaking.said(b_second)[:said], first.speaking.said(b_first)[:said])
                    and same_prefix(ours, theirs, rows, STATE_BOUND_M) and (not ended or len(ours) == len(theirs))):
                differed.append(places[w])
                break
    return differed


def _until(loop: WindowLoop, b: int, joined: int) -> int | None:
    """Row ``b``'s event in its own rows (`WindowLoop.end_step`) as a sentence's bound; None for a stage C window (its
    rows end there)."""
    return loop.end_step(b) if joined else None


@dataclass(frozen=True)
class BackwardRound(BranchRound):
    """A batch of a round whose branch points are found backward (`backward_round`; multi-aircraft control D182): the
    branch round's parts, and each group's depth (the seconds from its point to its event, in the groups' order), the
    searches made (one a varied aircraft with a candidate point) and those that found a decision point, the
    continuations flown only to the event + 60 s and those flown to their window's end; and, as information on what
    the search costs (its batches are one point of one window, MC12): its seconds, its loop steps and the rows they
    flew (each step's rows summed)."""

    depths_s: list[float]
    searches: int
    found: int
    to_event: int
    to_end: int
    search_s: float
    search_steps: int
    search_rows: int


def backward_round(model: Prior, start_loop: Callable[[Sequence[int]], tuple[Loop, list[int], dict[int, np.ndarray]]],
                   windows: Sequence[Window], places: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                   flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                   rosters: Mapping[str, LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                   interval_s: float, variant: str, edges_reference: Path,
                   faults: Mapping[str, Mapping[str, frozenset[int]]], device: torch.device, split: str,
                   continuations: int, rules: Rules, loop_options: Mapping[str, Any] | None = None,
                   host: torch.device = torch.device("cpu")) -> BackwardRound:
    """One batch ``windows`` of a round of stage D with its branch points found backward (multi-aircraft control D182;
    `multi.backward`): the first pass and the windows spoken again as `branch_round`'s (``rules``: stage D's,
    `multi.credit.rules`); each varied aircraft's event and candidate points (`multi.backward.event`, `points`); the
    second pass copies each window spoken again at its candidate points and holds the copies on ``host`` (D94's check as
    `branch_round`'s, up to each window's latest point); then each varied aircraft of a window that did not differ is
    searched (`multi.backward.search`), one point's ``continuations`` on ``device`` at a time: a point's copies flown
    to the event + 60 s and judged (`_trials`), a point chosen flown on to its window's end and made a group (D142: its
    W of D141, v's rows from the point). The fine point is the decision point's copy flown on its first sentences'
    numbers (every aircraft's) for 16 s. How the continuations are batched (stage D's reading, MC12): one point of one
    window at a time, its K continuations one batch."""
    window_loop, sources = _batch_loops(model, start_loop, windows, places, sentences, flights, geometries, rosters,
                                        finals, words, interval_s=interval_s, variant=variant,
                                        edges_reference=edges_reference, faults=faults, device=device, rules=rules,
                                        loop_options=loop_options)
    everything = list(range(len(windows)))
    first = window_loop(everything)
    results = first.run(sources(first, everything))
    ends = [[results[b] for b in rows] for rows in first.members]
    again = [w for w in everything if rules.again(ends[w])]
    if not again:
        return BackwardRound(results, [], [], [], [], 0, 0, 0, 0, 0.0, 0, 0)
    #: each window spoken again (its place in ``again``): its varied aircraft, their events and candidate points
    plans: dict[int, list[tuple[int, backward.Event, list[int]]]] = {}
    for k, w in enumerate(again):
        rows = first.members[w]
        aircraft = credit.aircraft_of(first, rows, ends[w])
        plans[k] = []
        for v, _ in rules.varied(windows[w], first, rows, ends[w]):
            end = ends[w][v]
            go_arounds = np.flatnonzero(end.words[:, RUNWAY] == RUNWAY_GO_AROUND)
            first_go_around = (aircraft[v].first_step + int(go_arounds[0])
                               if end.outcome == LANDED and len(go_arounds) else None)
            happened = backward.event(aircraft, v, end.outcome, first_go_around)
            plans[k].append((v, happened, backward.points(happened, aircraft[v].first_step, interval_s)))
    needed = {k: sorted({p for _, _, points in plan for p in points}) for k, plan in plans.items()}
    second = window_loop(again)
    numbers = sources(second, again)
    while second.speaking.observing:
        second.observe()
    held: dict[tuple[int, int], tuple[WindowLoop, list[np.random.Generator]]] = {}
    last = {k: max(ticks, default=-1) for k, ticks in needed.items()}
    final = max([second.speaking.start] + [t for t in last.values() if t >= 0])
    while True:
        t = second.speaking.t
        for k, ticks in needed.items():
            if t in ticks:                        # the window at its candidate point, held on the host (D182)
                held[(k, t)] = (second.copy([k], device=host),
                                [copying.deepcopy(numbers[b]) for b in second.members[k]])
        done = np.zeros(len(second.order), dtype=bool)
        for k in range(len(again)):
            if 0 <= last[k] <= t:
                done[second.members[k]] = True
        if (done & second.speaking.alive).any():           # past its latest point: nothing more to copy
            second.speaking.end(done & second.speaking.alive)
        if t >= final or not second.speaking.alive.any():
            break
        second.step(_draws(second, numbers))
    differed = _differed(first, second, again, last, places)
    del second
    read = sorted({first.members[again[k]][v] for k, plan in plans.items() for v, _, points in plan if points})
    first_samples = dict(zip(read, first.samples(split, read)))
    groups: list[Group] = []
    depths: list[float] = []
    counted = {"searches": 0, "found": 0, "to_event": 0, "to_end": 0, "steps": 0, "rows": 0}
    search_s = 0.0

    def flown(loop: WindowLoop, until: int, streams: Sequence[np.random.Generator]) -> None:
        """``loop`` flown to the tick ``until`` (or its end, if earlier) on ``streams``, its steps and rows counted."""
        while loop.speaking.t < until and loop.speaking.alive.any():
            loop.step(_draws(loop, streams))
            counted["steps"] += 1
            counted["rows"] += len(loop.order)
    fine_ticks = backward.ticks_of(backward.FINE_S, interval_s)
    for k, w in enumerate(again):
        for v, happened, points in plans[k] if places[w] not in differed else []:
            if not points:
                continue
            until = backward.judged_until(happened, interval_s)
            b_first = first.members[w][v]
            tried: dict[str, Any] = {}

            def trial(p: int) -> list[backward.Trial]:
                tried.clear()                        # the point tried before: the device holds one point's copies
                if (k, p) in held:
                    state, streams = held[(k, p)]
                else:                                # the fine point: the decision point's copy flown on 16 s
                    state, streams = held[(k, p - fine_ticks)]
                    state, streams = state.copy([0], device=device), [copying.deepcopy(n) for n in streams]
                    flown(state, p, streams)
                copies = state.copy([0] * continuations, device=device)
                drawn = [rules.continuation(places[w], m, v, p, j) if m == v else copying.deepcopy(streams[m])
                         for j in range(continuations) for m in range(len(streams))]
                flown(copies, until, drawn)
                counted["to_event"] += continuations
                tried.update(point=p, copies=copies, drawn=drawn)
                return _trials(copies, v, happened, p)

            def chosen(p: int) -> None:
                copies, drawn = tried["copies"], tried["drawn"]
                before = copies.speaking.t
                ends_k = copies.finish(drawn)
                counted["steps"] += copies.speaking.t - before
                counted["rows"] += (copies.speaking.t - before) * len(copies.order)
                rows_v = [copies.members[j][v] for j in range(continuations)]
                made = dict(zip(rows_v, copies.samples(split, rows_v)))
                kept = tuple(Sentence(*made[b], rules.reward([ends_k[c] for c in copies.members[j]]),
                                      _until(copies, b, len(windows[w].joined)))
                             for j, b in enumerate(rows_v))
                join = int(first.speaking.join_ticks[b_first])
                rows, permitted, tokens = first_samples[b_first]
                groups.append(Group(places[w], p - join, Sentence(rows, permitted, tokens, rules.reward(ends[w]),
                                                                  _until(first, b_first, len(windows[w].joined))),
                                    kept, varied=v))
                depths.append((happened.tick - p) * interval_s)
                counted["to_event"] -= continuations
                counted["to_end"] += continuations

            began = time.perf_counter()
            found = backward.search(points, fine_ticks, trial, chosen)
            search_s += time.perf_counter() - began
            tried.clear()
            counted["searches"] += 1
            counted["found"] += found.decision is not None
        for key in [key for key in held if key[0] == k]:     # its copies released once its searches are done
            del held[key]
    return BackwardRound(results, groups, [places[w] for w in again], differed, depths, counted["searches"],
                         counted["found"], counted["to_event"], counted["to_end"], search_s, counted["steps"],
                         counted["rows"])


def _trials(copies: WindowLoop, v: int, happened: backward.Event, point: int) -> list[backward.Trial]:
    """Each continuation of a point (``copies``: its windows, v the varied aircraft's place among each one's aircraft)
    judged at the event + 60 s or its end (`multi.backward.avoids`): the loss v answered for, whether the event's partner
    answered for one with v, v's outcome if it ended, and whether v said a go-around from ``point`` on."""
    out = []
    for rows in copies.members:
        b = rows[v]
        key = copies.records[b].key
        if copies.loss[b] is not None:
            ended: str | None = LOST_SEPARATION
        elif not copies.speaking.alive[b] and not copies.speaking.ended[b]:
            ended = copies.speaking.loop.outcome(b).outcome
        else:
            ended = None
        partner = any(copies.loss[c] is not None and copies.other[c] == key and copies.records[c].key == happened.partner
                      for c in rows)
        said = copies.speaking.said(b)
        own = point - int(copies.speaking.join_ticks[b]) - copies.speaking.start   # the point's row of v's sentence
        go_around = bool((said[max(own, 0):, RUNWAY] == RUNWAY_GO_AROUND).any())
        out.append(backward.Trial(backward.avoids(happened, answered=copies.other[b] if copies.loss[b] is not None
                                                  else None, partner_answered_v=partner, ended=ended,
                                                  go_around=go_around), go_around))
    return out
