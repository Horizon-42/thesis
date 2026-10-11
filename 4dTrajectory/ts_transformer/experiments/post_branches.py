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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Mapping, Sequence

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
    draws as it says its rows alone). A row that left the loop draws none (S6): nothing reads its source after."""
    said_now = loop.speaking.said_now() & loop.speaking.held
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
    the search costs (its points stepped together in one loop, D188): its seconds, the loop's steps and the rows they
    flew (each step's rows held in the loop, summed: S6); each group's kind (`multi.backward.Found.kinds`, in the
    groups' order), and the depth of each majority point found (D192; one a search that found one, in the searches'
    order)."""

    depths_s: list[float]
    searches: int
    found: int
    to_event: int
    to_end: int
    search_s: float
    search_steps: int
    search_rows: int
    kinds: list[str]
    majority_depths_s: list[float]


@dataclass
class _Search:
    """One varied aircraft's search in a backward batch (`backward_round`): its window spoken again (``k`` its place in
    ``again``, ``w`` in the batch), the aircraft ``v``, its event and judged tick, the search's steps
    (`multi.backward.search`), the part flown for the point it tries, its groups in the order chosen, and what it
    found once done."""

    k: int
    w: int
    v: int
    happened: backward.Event
    until: int
    steps: Generator[tuple[str, int], Sequence[backward.Trial] | None, backward.Found]
    trying: _Part | None = None
    groups: list[tuple[Group, float] | None] = field(default_factory=list)
    found: backward.Found | None = None
    #: the decision point once chosen (the search's first point chosen)
    decision: int | None = None


@dataclass
class _Part:
    """Windows of the joint loop that one search flies (`backward_round`): a point's continuations, flown to the judged
    tick and, once chosen (``chosen``: the place of its group among its search's), on to their window's end; or
    (``base``) the decision point's copy flown on its first-sentence numbers to the fine point. Its ``windows`` and
    each of their rows' numbers, in order."""

    search: _Search
    point: int
    windows: int
    streams: list[np.random.Generator]
    base: bool = False
    chosen: int | None = None
    #: a point tried (D194 (3)): the continuations that must avoid the event for the search to choose it (the majority
    #: behind the decision point in a search for the majority point, else 1); the part ends once that is out of reach
    #: (`multi.backward.out_of_reach`). None: the decision point's copy (``base``)
    need: int | None = None


def backward_round(model: Prior, start_loop: Callable[[Sequence[int]], tuple[Loop, list[int], dict[int, np.ndarray]]],
                   windows: Sequence[Window], places: Sequence[int], sentences: Mapping[int, ClosedLoopSentence],
                   flights: Mapping[int, Mapping[str, Any]], geometries: Mapping[str, AirportGeometry],
                   rosters: Mapping[str, LandingIndex], finals: Mapping[str, Sequence[Final]], words: Words, *,
                   interval_s: float, variant: str, edges_reference: Path,
                   faults: Mapping[str, Mapping[str, frozenset[int]]], device: torch.device, split: str,
                   continuations: int, rules: Rules, loop_options: Mapping[str, Any] | None = None,
                   host: torch.device = torch.device("cpu"), majority: int | None = None) -> BackwardRound:
    """One batch ``windows`` of a round of stage D with its branch points found backward (multi-aircraft control D182;
    `multi.backward`): the first pass and the windows spoken again as `branch_round`'s (``rules``: stage D's,
    `multi.credit.rules`); each varied aircraft's event and candidate points (`multi.backward.event`, `points`); the
    second pass copies each window spoken again at its candidate points and holds the copies on ``host`` (D94's check as
    `branch_round`'s, up to each window's latest point). Then every varied aircraft of a window that did not differ is
    searched (`multi.backward.search`; ``majority``: its majority point in place of point B, D192), the searches
    stepped together on ``device`` (D188):

    - **One loop.** The joint loop holds every search's current point: its ``continuations`` copies of the point's held
      window, v on numbers of its own, the other aircraft on their first-sentence numbers from the point. A step says
      and flies one row of every window in it, each window at its own tick (`WindowLoop.together`).
    - **When a continuation stops.** A point's continuations are flown to the event + 60 s (or their end, if earlier),
      then judged (`_trials`) and the search sent their trials; a point not chosen leaves the loop there. A point is
      judged as soon as too few of its continuations are left to choose it (D194 (3), `_failed`): every one failed,
      or, behind the decision point in a search for the majority point, the majority out of reach. The search's
      answer is the same.
    - **A point chosen** goes on in the same loop to its window's end, beside the points then tried, and its group is
      made when its windows end (D142: its W of D141, v's rows from the point).
    - **Joining.** Before a step, the parts that ended leave and the searches' next points join: a point from its held
      copy; the fine point from the decision point's copy, flown first in the loop for 16 s on every aircraft's
      first-sentence numbers, its continuations then copied from it.

    The groups are in the order of the searches (the windows', then their varied aircraft's) and, within a search, of
    their points chosen; a window's held copies are freed once its searches are done."""
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
        return BackwardRound(results, [], [], [], [], 0, 0, 0, 0, 0.0, 0, 0, [], [])
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
    fine_ticks = backward.ticks_of(backward.FINE_S, interval_s)
    searches = [_Search(k, w, v, happened, backward.judged_until(happened, interval_s),
                        backward.search(points, fine_ticks, majority))
                for k, w in enumerate(again) if places[w] not in differed
                for v, happened, points in plans[k] if points]
    left = {k: sum(s.k == k for s in searches) for k in range(len(again))}
    for key in [key for key in held if not left[key[0]]]:        # a window searched for nothing: its copies freed
        del held[key]
    counted = {"found": 0, "to_event": 0, "to_end": 0, "steps": 0, "rows": 0}
    joining: list[tuple[_Part, WindowLoop, list[int]]] = []     # parts to join the loop, each from its source's windows

    def tried(s: _Search, point: int, streams: Sequence[np.random.Generator], source: WindowLoop, window: int) -> None:
        """Point ``point``'s continuations of search ``s`` to join the loop: copies of ``source``'s ``window``, v on
        its own numbers (D166 (26)), every other aircraft on ``streams`` (its first sentence's, from the point)."""
        drawn = [rules.continuation(places[s.w], m, s.v, point, j) if m == s.v else copying.deepcopy(streams[m])
                 for j in range(continuations) for m in range(len(streams))]
        behind = majority is not None and s.decision is not None and point < s.decision
        s.trying = _Part(s, point, continuations, drawn, need=majority if behind else 1)
        joining.append((s.trying, source, [window] * continuations))
        counted["to_event"] += continuations

    def ask(s: _Search, answer: Sequence[backward.Trial] | None) -> None:
        """Search ``s`` sent ``answer`` (the trials of the point it tried; None to start it) and stepped on to the next
        point it tries: a point chosen stays in the loop for its group; a point to try joins it (the fine point first
        as the decision point's copy flown to it)."""
        while True:
            try:
                kind, point = s.steps.send(answer)
            except StopIteration as stop:
                s.found = stop.value
                counted["found"] += stop.value.decision is not None
                left[s.k] -= 1
                if not left[s.k]:                         # the window's searches are done: its copies freed
                    for key in [key for key in held if key[0] == s.k]:
                        del held[key]
                return
            if kind == backward.CHOOSE:
                if s.decision is None:
                    s.decision = point
                s.trying.chosen = len(s.groups)
                s.groups.append(None)
                counted["to_event"] -= continuations
                counted["to_end"] += continuations
                answer = None
                continue
            if (s.k, point) in held:
                state, streams = held[(s.k, point)]
                tried(s, point, streams, state, 0)
            else:                                         # the fine point: the decision point's copy flown on first
                state, streams = held[(s.k, point - fine_ticks)]
                s.trying = _Part(s, point, 1, [copying.deepcopy(n) for n in streams], base=True)
                joining.append((s.trying, state, [0]))
            return

    def group(part: _Part, joint: WindowLoop, own: Sequence[int]) -> tuple[Group, float]:
        """The group of a point chosen whose windows ``own`` (in the joint loop) have ended, and its depth: read from
        the joint loop, each row's end and sentence its own (`WindowLoop.results`, `samples`; S6)."""
        s = part.search
        members = [joint.members[w] for w in own]
        rows = [b for rows_w in members for b in rows_w]
        ended = dict(zip(rows, joint.results(rows)))
        rows_v = [rows_w[s.v] for rows_w in members]
        made = dict(zip(rows_v, joint.samples(split, rows_v)))
        kept = tuple(Sentence(*made[b], rules.reward([ended[c] for c in rows_w]),
                              _until(joint, b, len(windows[s.w].joined))) for rows_w, b in zip(members, rows_v))
        b_first = first.members[s.w][s.v]
        join = int(first.speaking.join_ticks[b_first])
        sentence, permitted, tokens = first_samples[b_first]
        return (Group(places[s.w], part.point - join, Sentence(sentence, permitted, tokens, rules.reward(ends[s.w]),
                                                               _until(first, b_first, len(windows[s.w].joined))),
                      kept, varied=s.v), (s.happened.tick - part.point) * interval_s)

    def settled(joint: WindowLoop | None, flying: Sequence[_Part]) -> list[tuple[_Part, range]]:
        """The parts of the joint loop still flown, each with its windows in it: a part at its end is judged (a point
        tried), made into the fine point's continuations (the decision point's copy), or made into its group (a point
        chosen)."""
        out, at = [], 0
        for part in flying:
            own, at = range(at, at + part.windows), at + part.windows
            s = part.search
            tick = joint.speaking.t - int(joint.origins[own[0]])
            alive = bool(joint.speaking.alive[[b for w in own for b in joint.members[w]]].any())
            if part.base:
                if alive and tick < part.point:
                    out.append((part, own))
                else:
                    tried(s, part.point, part.streams, joint, own[0])
                continue
            if part.chosen is None:
                hopeless = backward.out_of_reach(_failed(joint, own, part), continuations, part.need)
                if alive and tick < s.until and not hopeless:
                    out.append((part, own))
                    continue
                ask(s, _trials(joint, own, s.v, s.happened, part.point))
                if part.chosen is None:                   # not chosen: its continuations end here
                    continue
            if alive:
                out.append((part, own))
            else:
                s.groups[part.chosen] = group(part, joint, own)
        return out

    def scheduled(part: _Part, loop: WindowLoop, window: int) -> int | None:
        """The steps part ``part`` (its first window ``window`` in ``loop``) is flown before it leaves the joint loop at
        the latest: a point tried, to its judged tick; the decision point's copy, to the fine point; None for a point
        chosen (flown to its window's end, not known)."""
        tick = loop.speaking.t - int(loop.origins[window])
        if part.base:
            return part.point - tick
        return part.search.until - tick if part.chosen is None else None

    began = time.perf_counter()
    for s in searches:
        ask(s, None)
    joint: WindowLoop | None = None
    flying: list[_Part] = []
    room: int | None = None             # the steps the joint loop's cache has room for (None: D183's, as it grows)
    while True:
        while True:                     # the parts ended leave, the searches' next points join, until none does
            kept = settled(joint, flying)
            if not joining and len(kept) == len(flying) and room != 0:
                break
            # the joint loop's own windows (those kept, and the fine points' copies of a decision point's) one part of
            # it, then the points from their held copies; the joint loop replaced is freed as it is copied
            ours = [(part, own) for part, source, own in joining if source is joint]
            theirs = [(part, source, own) for part, source, own in joining if source is not joint]
            mine = [w for _, own in kept for w in own] + [w for _, own in ours for w in own]
            selected = ([(joint, mine)] if mine else []) + [(source, own) for _, source, own in theirs]
            flying = [part for part, _ in kept] + [part for part, _ in ours] + [part for part, _, _ in theirs]
            # the cache's room: the steps to the next rebuild, at the first part's end by its schedule (D190 (1)); a
            # point chosen there that goes on with no part leaving or joining is rebuilt with the others all the same
            steps = [scheduled(part, joint, own[0]) for part, own in kept + ours] + \
                [scheduled(part, source, own[0]) for part, source, own in theirs]
            room = min((n for n in steps if n is not None), default=None)
            joining.clear()
            joint = WindowLoop.together(selected, device, room, spent=joint if mine else None) if flying else None
        if not flying:
            break
        joint.step(_draws(joint, [n for part in flying for n in part.streams]))
        room = None if room is None else room - 1
        counted["steps"] += 1
        counted["rows"] += int(joint.speaking.held.sum())
    search_s = time.perf_counter() - began
    del held
    groups = [g for s in searches for g in s.groups]
    majorities = [(s.happened.tick - s.found.majority) * interval_s for s in searches if s.found.majority is not None]
    return BackwardRound(results, [g for g, _ in groups], [places[w] for w in again], differed,
                         [depth for _, depth in groups], len(searches), counted["found"], counted["to_event"],
                         counted["to_end"], search_s, counted["steps"], counted["rows"],
                         [kind for s in searches for kind in s.found.kinds], majorities)


def _failed(loop: WindowLoop, own: Sequence[int], part: _Part) -> int:
    """The continuations of ``part``'s point (its windows ``own`` in ``loop``, as `_trials` reads them) that already do
    not avoid the event, by what stays to the judged tick (`multi.backward.out_of_reach`): v answered for a loss, the
    event's partner answered for one with v, or v ended with the event's outcome. A go-around said is read at the judged
    tick only (`_trials`)."""
    v, happened = part.search.v, part.search.happened
    out = 0
    for w in own:
        rows = loop.members[w]
        b = rows[v]
        if loop.loss[b] is not None:
            out += 1
        elif happened.kind == backward.LOSS:
            key = loop.records[b].key
            out += any(loop.loss[c] is not None and loop.other[c] == key and loop.records[c].key == happened.partner
                       for c in rows)
        elif happened.kind == backward.END:
            out += (not loop.speaking.alive[b] and not loop.speaking.ended[b]
                    and loop.speaking.outcome(b).outcome == happened.outcome)
    return out


def _trials(loop: WindowLoop, own: Sequence[int], v: int, happened: backward.Event, point: int) -> list[backward.Trial]:
    """Each continuation of a point (``own``: its windows in ``loop``, v the varied aircraft's place among each one's
    aircraft) judged at the event + 60 s or its end (`multi.backward.avoids`): the loss v answered for, whether the
    event's partner answered for one with v, v's outcome if it ended, and whether v said a go-around from ``point``
    on."""
    out = []
    for w in own:
        rows = loop.members[w]
        b = rows[v]
        key = loop.records[b].key
        if loop.loss[b] is not None:
            ended: str | None = LOST_SEPARATION
        elif not loop.speaking.alive[b] and not loop.speaking.ended[b]:
            ended = loop.speaking.outcome(b).outcome
        else:
            ended = None
        partner = any(loop.loss[c] is not None and loop.other[c] == key and loop.records[c].key == happened.partner
                      for c in rows)
        said = loop.speaking.said(b)
        mine = loop.own_row(b, point) - loop.speaking.start         # the point's row of v's sentence
        go_around = bool((said[max(mine, 0):, RUNWAY] == RUNWAY_GO_AROUND).any())
        out.append(backward.Trial(backward.avoids(happened, answered=loop.other[b] if loop.loss[b] is not None
                                                  else None, partner_answered_v=partner, ended=ended,
                                                  go_around=go_around), go_around))
    return out
