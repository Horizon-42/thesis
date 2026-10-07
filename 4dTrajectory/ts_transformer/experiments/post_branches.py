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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.start import Loop
from ts_transformer.experiments.post_window_loop import WindowLoop, WindowResult
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.post.branches import (
    CONTINUATIONS, STATE_BOUND_M, Group, Sentence, branch_points, continuation_numbers, first_numbers,
)
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


def stage_c_rules(seed: int, round_: int, interval_s: float) -> Rules:
    """Stage C's rules (module docstring; D37, D94): one aircraft, its reward, spoken again below 1, its branch points from
    its first predicted step to its event."""
    def varied(window: Window, loop: WindowLoop, rows: Sequence[int], results: Sequence[WindowResult]
               ) -> list[tuple[int, list[int]]]:
        (b,) = rows
        return [(0, branch_points(loop.speaking.start, loop.end_step(b), interval_s))]

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

    everything = list(range(len(windows)))
    first = window_loop(everything)
    results = first.run(sources(first, everything))
    ends = [[results[b] for b in rows] for rows in first.members]
    again = [w for w in everything if rules.again(ends[w])]
    if not again:
        return BranchRound(results, [], [], [])
    first_samples = first.samples(split)
    #: each window spoken again (its place in ``again``): its varied aircraft and their branch points (ticks)
    varied = {k: rules.varied(windows[w], first, first.members[w], ends[w]) for k, w in enumerate(again)}
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
            made = copies.samples(split)
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
            done = np.zeros(len(second.order), dtype=bool)
            for k in range(len(again)):
                if last[k] >= 0 and last[k] <= t:
                    done[second.members[k]] = True
            if (done & second.speaking.alive).any():           # past its last branch point: nothing more to copy
                second.speaking.end(done & second.speaking.alive)
        if t >= final or not second.speaking.alive.any():
            break
        second.step(_draws(second, numbers))
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
            # an aircraft the executor ended before the point: its whole flight, the same length in both passes
            rows = min((last[k] - join) * second.every + 1, len(theirs))
            ended = rows < (last[k] - join) * second.every + 1
            if not (np.array_equal(second.speaking.said(b_second)[:said], first.speaking.said(b_first)[:said])
                    and same_prefix(ours, theirs, rows, STATE_BOUND_M) and (not ended or len(ours) == len(theirs))):
                differed.append(places[w])
                break
    return BranchRound(results, [g for _, g in groups if g.window not in differed], [places[w] for w in again],
                       differed)


def _until(loop: WindowLoop, b: int, joined: int) -> int | None:
    """Row ``b``'s event in its own rows (`WindowLoop.end_step`) as a sentence's bound; None for a stage C window (its
    rows end there)."""
    return loop.end_step(b) if joined else None
