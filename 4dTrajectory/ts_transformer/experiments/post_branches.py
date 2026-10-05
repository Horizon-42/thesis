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
"""

from __future__ import annotations

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
    """One batch of a round: the first pass's ends (in the batch's order), the branch groups (`Group.window`: the window's
    place in the round), and the windows spoken again and those whose second pass differed from their first (places in
    the round)."""

    first: list[WindowResult]
    groups: list[Group]
    spoken_again: list[int]
    differed: list[int]


def same_prefix(a: np.ndarray, b: np.ndarray, rows: int, bound_m: float) -> bool:
    """Whether the first ``rows`` state rows of ``a`` and ``b`` (`STATE_COLUMNS`) agree: their positions and heights
    within ``bound_m``."""
    if len(a) < rows or len(b) < rows:
        return False
    return bool(np.max(np.abs(a[:rows, :3] - b[:rows, :3]), initial=0.0) <= bound_m)


def branch_round(model: Prior, start_loop: Callable[[Sequence[int]], tuple[Loop, list[int]]], windows: Sequence[Window],
                 places: Sequence[int], sentences: Mapping[int, ClosedLoopSentence], flights: Mapping[int, Mapping[str, Any]],
                 geometries: Mapping[str, AirportGeometry], rosters: Mapping[str, LandingIndex],
                 finals: Mapping[str, Sequence[Final]], words: Words, *, interval_s: float, variant: str,
                 edges_reference: Path, device: torch.device, seed: int, round_: int, split: str,
                 continuations: int = CONTINUATIONS) -> BranchRound:
    """One batch ``windows`` of a round (module docstring), ``places`` their places in the round (the key of their random
    numbers); ``start_loop`` starts the closed loop of given flights (their places in the split's signals,
    `autopilot.start.start`) and returns it with its order."""
    order = [w.signal_index for w in windows]
    if order != sorted(set(order)):
        raise ValueError("the windows of a batch command different flights, in the order of their places in the signals")
    if len(places) != len(windows) or len(set(places)) != len(places):
        raise ValueError("one place in the round for each window of the batch, each its own")

    def window_loop(batch: Sequence[int]) -> WindowLoop:
        loop, started = start_loop([order[b] for b in batch])
        return WindowLoop(model, loop, started, [windows[b] for b in batch], sentences, flights, geometries, rosters,
                          finals, words, interval_s=interval_s, variant=variant, edges_reference=edges_reference,
                          device=device)

    everything = list(range(len(windows)))
    first = window_loop(everything)
    results = first.run([first_numbers(seed, round_, places[b]) for b in everything])
    again = [b for b in everything if results[b].reward < 1.0]
    if not again:
        return BranchRound(results, [], [], [])
    first_samples = first.samples(split)
    points = {k: branch_points(first.speaking.start, first.end_step(b), interval_s) for k, b in enumerate(again)}
    second = window_loop(again)
    numbers = [first_numbers(seed, round_, places[b]) for b in again]
    while second.speaking.observing:
        second.observe()
    groups: list[tuple[int, Group]] = []
    last = max((p[-1] for p in points.values() if p), default=second.speaking.start)
    while True:
        t = second.speaking.t
        due = [k for k in range(len(again)) if t in points[k]]
        if due:
            copies = second.copy([k for k in due for _ in range(continuations)])
            ends = copies.finish([continuation_numbers(seed, round_, places[again[k]], t, j)
                                  for k in due for j in range(continuations)])
            made = copies.samples(split)
            for i, k in enumerate(due):
                b = again[k]
                rows, permitted, tokens = first_samples[b]
                kept = [Sentence(*made[i * continuations + j], ends[i * continuations + j].reward)
                        for j in range(continuations)]
                groups.append((k, Group(places[b], t, Sentence(rows, permitted, tokens, results[b].reward),
                                        tuple(kept))))
            done = np.array([bool(points[k]) and points[k][-1] <= t for k in range(len(again))])
            if (done & second.speaking.alive).any():           # past its last branch point: nothing more to copy
                second.speaking.end(done & second.speaking.alive)
        if t >= last or not second.speaking.alive.any():
            break
        second.step(np.stack([n.random(len(COLUMNS)) for n in numbers]))
    differed = []
    for k, b in enumerate(again):
        if not points[k]:
            continue
        said = points[k][-1] - second.speaking.start
        rows = points[k][-1] * second.every + 1
        if not (np.array_equal(second.speaking.said(k)[:said], first.speaking.said(b)[:said])
                and same_prefix(second.speaking.states(k), first.speaking.states(b), rows, STATE_BOUND_M)):
            differed.append(places[b])
    return BranchRound(results, [g for _, g in groups if g.window not in differed], [places[b] for b in again],
                       differed)
