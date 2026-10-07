"""Branch training (post-training §2 item 9, §8 C6; D37, D94): the random numbers of a round, the branch points, the
branch groups and their advantage, and the samples they give the loss.

**Random numbers** (D94): a window's first sentence draws from the seed, the round and the window; a continuation from
the seed, the round, the window, the branch point and k. Each is a numpy generator of its own, five uniform numbers a
row said, as free generation draws them.

**Branch points** (D37): the first predicted step and every `BRANCH_EVERY_S` after it, before the event that ended
the first sentence (t_E: the step of its loss of separation, of its judged outcome or of its time limit).

**A branch group** is the first sentence and the K continuations of one branch point; they differ only after it. The
advantage of each is its reward minus the mean of the group, on the rows from the branch point on (the words after the
branch point, §2 item 9 point 5); a group whose rewards are all the same gives no sample. A first sentence in several
groups gives a sample in each.

**A window of several commanded aircraft** (multi-aircraft control D142, D143, D149; post-training §9 item 9): a group
names the aircraft it varies (``Group.varied``, its place among the window's commanded aircraft; stage C's: 0, its one
aircraft), its sentences are that aircraft's, its rewards the window's (the caller's rule), and each sentence counts
its rows only from the branch point up to the aircraft's event (``Sentence.until``: the row of its end or of its loss,
from which it is silent, D144); stage C's sentences end at their event (no bound). The numbers of each aircraft are the
caller's rule (`experiments.post_branches.branch_round`); stage C's are D94's, below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.post.loss import Samples
from ts_transformer.post.traffic_attention import traffic_of
from ts_transformer.prior.batch import collate
from ts_transformer.prior.speaker import Permitted

#: D37: the branch points are the first predicted step and every 120 s after it.
BRANCH_EVERY_S = 120.0
#: D94: the continuations of each branch point.
CONTINUATIONS = 8
#: The bound on a state flown again, m (D94: the second pass against the first). MIRROR of
#: `autopilot.conformance.STATE_BOUND_M` (vocabulary D97 (3)), which stage C's runners do not import (the
#: architecture test): `tests/test_post_branches.py` pins the two.
STATE_BOUND_M = 1e-6


def first_numbers(seed: int, round_: int, window: int) -> np.random.Generator:
    """The random numbers of window ``window``'s first sentence in round ``round_`` (D94)."""
    return np.random.default_rng([seed, round_, window])


def continuation_numbers(seed: int, round_: int, window: int, branch: int, k: int) -> np.random.Generator:
    """The random numbers of continuation ``k`` of window ``window`` at branch point ``branch`` (its Δ row, D94)."""
    return np.random.default_rng([seed, round_, window, branch, k])


def landed_numbers(seed: int, round_: int, window: int, draw: int) -> np.random.Generator:
    """The random numbers of draw ``draw`` of window ``window`` in round ``round_`` of a campaign that trains on the landed
    sentences (P49): draw 0 is the window's first sentence (`first_numbers`), draw d a stream of its own (its fourth
    number, 2**30, is no Δ row of a continuation's)."""
    return first_numbers(seed, round_, window) if draw == 0 else np.random.default_rng([seed, round_, window, 1 << 30,
                                                                                        draw])


def branch_points(start: int, end_step: int, interval_s: float) -> list[int]:
    """The branch points of a first sentence (module docstring), as Δ rows: the first predicted step ``start`` and every
    `BRANCH_EVERY_S` after it, before its event at Δ row ``end_step``."""
    every = BRANCH_EVERY_S / interval_s
    if abs(every - round(every)) > 1e-9:
        raise ValueError(f"{BRANCH_EVERY_S:g} s is not a whole number of {interval_s:g} s rows")
    return list(range(start, end_step, int(round(every))))


@dataclass(frozen=True)
class Sentence:
    """One sentence of a group: its rows with the words said as targets (`prior.batch.SentenceRows`), the speaker's
    records of them, its tokens at each row and its reward."""

    rows: Any
    permitted: Permitted
    tokens: list[np.ndarray]
    reward: float
    #: the row of its event (its end, or the row from which it is silent), after which no row is counted; None: its rows
    #: end there (stage C's)
    until: int | None = None


@dataclass(frozen=True)
class Group:
    """A branch group (module docstring): the window, its branch point (a Δ row), the first sentence and the
    continuations."""

    window: int
    branch: int
    first: Sentence
    continuations: tuple[Sentence, ...]
    #: the aircraft it varies: its place among the window's commanded aircraft (module docstring)
    varied: int = 0

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        return (self.first, *self.continuations)

    def advantages(self) -> np.ndarray:
        """Each sentence's reward minus the mean of the group's."""
        rewards = np.array([s.reward for s in self.sentences], dtype=np.float64)
        return rewards - rewards.mean()

    @property
    def informative(self) -> bool:
        """Whether its rewards differ (a group whose rewards are all the same gives no sample)."""
        rewards = [s.reward for s in self.sentences]
        return max(rewards) > min(rewards)


def samples(groups: Sequence[Group], device: torch.device, part_width: int = 0) -> Samples:
    """The loss's batch of the samples of ``groups`` (`post.loss.Samples`): every sentence of every informative group,
    its advantage on the rows from its group's branch point on up to its event (``Sentence.until``; and only there), the
    speaker's records joined and the tokens padded to the longest sentence (with a caller's token part of
    ``part_width``, `traffic_attention.traffic_of`)."""
    items = [(sentence, advantage, group.branch) for group in groups if group.informative
             for sentence, advantage in zip(group.sentences, group.advantages())]
    if not items:
        raise ValueError("no informative group: no sample")
    rows = collate([s.rows for s, _, _ in items], device)
    length = rows.asked.shape[1]
    empty = np.zeros((0, items[0][0].tokens[0].shape[1]), dtype=np.float32)
    tokens = [s.tokens + [empty] * (length - len(s.tokens)) for s, _, _ in items]
    steps = torch.arange(length, device=device)[None, :]
    after = steps >= torch.tensor([b for _, _, b in items], device=device)[:, None]
    counted = rows.asked & after
    if any(s.until is not None for s, _, _ in items):
        until = torch.tensor([length if s.until is None else s.until for s, _, _ in items], device=device)[:, None]
        counted = counted & (steps < until)
    advantage = torch.tensor([a for _, a, _ in items], dtype=torch.float32, device=device)[:, None] * counted
    return Samples(rows, Permitted.join([s.permitted for s, _, _ in items]), traffic_of(tokens, device, part_width),
                   advantage, counted)


@dataclass(frozen=True)
class Landed:
    """A window's kept sentence in a campaign that trains on the landed sentences (P49): the window (its place in the
    round) and its landed sentence of the highest reward among its draws."""

    window: int
    sentence: Sentence


def landed_samples(kept: Sequence[Landed], device: torch.device) -> Samples:
    """The loss's batch of kept sentences (`post.loss.landed_step`): every row a sentence said counts (its words are
    the targets), the speaker's records joined and the tokens padded to the longest. The advantage is not read by that
    loss; it is the counted rows, so the batch is a `Samples`."""
    if not kept:
        raise ValueError("no kept sentence: no sample")
    rows = collate([k.sentence.rows for k in kept], device)
    length = rows.asked.shape[1]
    empty = np.zeros((0, kept[0].sentence.tokens[0].shape[1]), dtype=np.float32)
    tokens = [k.sentence.tokens + [empty] * (length - len(k.sentence.tokens)) for k in kept]
    return Samples(rows, Permitted.join([k.sentence.permitted for k in kept]), traffic_of(tokens, device),
                   rows.asked.to(torch.float32), rows.asked)
