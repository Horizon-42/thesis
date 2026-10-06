"""The window's reward and its credit (multi-aircraft control §2.2, §2.3, §3.3; D141–D143): stage D's rules of branch
training (post-training §9 item 6, `experiments.post_branches.Rules`, which `rules` gives as its fields).

- **The window's reward** W (`window_reward`, D141): the sum of its commanded aircraft's rewards (post-training D30,
  each with its own landings, D140); not divided by their count. With one commanded aircraft, that aircraft's reward.
- **Spoken again** (D143): a window whose W is less than its count of commanded aircraft.
- **The varied aircraft** (`varied`, D143): each commanded aircraft with r < 1, and each commanded aircraft of a loss
  of separation that one of them answers for (the other aircraft of that loss, even with r = 1).
- **Their branch points** (`points`, D143): a varied aircraft v's own first predicted step and the points of the
  window's grid after it — the anchor's first predicted step and every `BRANCH_EVERY_S` after it — before v's event:
  the step of its own end, or of the earliest loss that it is in (answering or not), whichever is earlier. Ticks of
  the window's grid (the loop's ticks). With one commanded aircraft, stage C's points (post-training D37).
- **The numbers** (§3.3, D142): each commanded aircraft's own stream, from the seed, the round, the window's place in
  the round and the aircraft's place in the window (`first_numbers`); a continuation's from these, the branch point and
  k (`continuation_numbers`, for the varied aircraft only: the others go on from their own first-sentence streams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ts_transformer.post.branches import branch_points


@dataclass(frozen=True)
class Aircraft:
    """One commanded aircraft of a window's first sentence, as the credit reads it: its key, its reward, its first
    predicted step and its end (ticks: the row of its loss, or the row after the last one said to it), and the loss it
    answered for (its tick and the other aircraft's key; None without one)."""

    key: str
    reward: float
    first_step: int
    end: int
    loss_step: int | None
    other: str | None


def window_reward(rewards: Sequence[float]) -> float:
    """W (D141): the sum of the window's commanded aircraft's rewards."""
    return float(sum(rewards))


def spoken_again(rewards: Sequence[float]) -> bool:
    """Whether a window is spoken again (D143): its W below its count of commanded aircraft."""
    return window_reward(rewards) < len(rewards)


def varied(aircraft: Sequence[Aircraft]) -> list[int]:
    """The places of the varied aircraft (module docstring, D143), in order."""
    places = {a.key: k for k, a in enumerate(aircraft)}
    low = [k for k, a in enumerate(aircraft) if a.reward < 1.0]
    out = set(low)
    for k in low:
        other = aircraft[k].other
        if other in places:
            out.add(places[other])
    return sorted(out)


def event(aircraft: Sequence[Aircraft], v: int) -> int:
    """The tick of aircraft ``v``'s event (module docstring): its end, or the earliest loss that it is in."""
    key = aircraft[v].key
    losses = [a.loss_step for a in aircraft if a.loss_step is not None and (a.key == key or a.other == key)]
    return min([aircraft[v].end] + losses)


def points(aircraft: Sequence[Aircraft], v: int, interval_s: float) -> list[int]:
    """The branch points of aircraft ``v`` (module docstring), ticks: its first predicted step, then the window's grid
    (from the anchor's first predicted step, ``aircraft[0]``'s) after it, before its event."""
    first, end = aircraft[v].first_step, event(aircraft, v)
    grid = [t for t in branch_points(aircraft[0].first_step, end, interval_s) if t > first]
    return ([first] if first < end else []) + grid


def first_numbers(seed: int, round_: int, place: int, member: int) -> np.random.Generator:
    """The numbers of commanded aircraft ``member``'s first sentence in the window at ``place`` of round ``round_``."""
    return np.random.default_rng([seed, round_, place, member])


def continuation_numbers(seed: int, round_: int, place: int, member: int, tick: int, k: int) -> np.random.Generator:
    """The numbers of continuation ``k`` of the varied aircraft ``member`` at the branch point ``tick``."""
    return np.random.default_rng([seed, round_, place, member, tick, k])


def aircraft_of(loop: Any, rows: Sequence[int], results: Sequence[Any]) -> list[Aircraft]:
    """A window's commanded aircraft (`Aircraft`) from its first sentence: ``loop`` the window loop that spoke it
    (post-training §9 item 5), ``rows`` the window's rows in it, ``results`` their ends (`WindowResult`)."""
    out = []
    for b, result in zip(rows, results):
        join = int(loop.speaking.join_ticks[b])
        out.append(Aircraft(key=loop.records[b].key, reward=float(result.reward),
                            first_step=join + int(loop.speaking.start), end=join + int(loop.end_step(b)),
                            loss_step=result.loss_step, other=result.other))
    return out


def rules(seed: int, round_: int, interval_s: float) -> dict[str, Any]:
    """Stage D's rules of branch training (module docstring), as the fields of `experiments.post_branches.Rules`."""
    def chosen(window: Any, loop: Any, rows: Sequence[int], results: Sequence[Any]) -> list[tuple[int, list[int]]]:
        aircraft = aircraft_of(loop, rows, results)
        return [(v, points(aircraft, v, interval_s)) for v in varied(aircraft)]

    return dict(first=lambda place, member: first_numbers(seed, round_, place, member),
                continuation=lambda place, member, v, tick, k: continuation_numbers(seed, round_, place, member, tick, k),
                reward=lambda results: window_reward([r.reward for r in results]),
                again=lambda results: spoken_again([r.reward for r in results]),
                varied=chosen)
