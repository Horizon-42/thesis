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


@dataclass(frozen=True)
class Group:
    """A branch group (module docstring): the window, its branch point (a Δ row), the first sentence and the
    continuations."""

    window: int
    branch: int
    first: Sentence
    continuations: tuple[Sentence, ...]

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


def samples(groups: Sequence[Group], device: torch.device) -> Samples:
    """The loss's batch of the samples of ``groups`` (`post.loss.Samples`): every sentence of every informative group,
    its advantage on the rows from its group's branch point on (and only there), the speaker's records joined and the
    tokens padded to the longest sentence."""
    items = [(sentence, advantage, group.branch) for group in groups if group.informative
             for sentence, advantage in zip(group.sentences, group.advantages())]
    if not items:
        raise ValueError("no informative group: no sample")
    rows = collate([s.rows for s, _, _ in items], device)
    length = rows.asked.shape[1]
    empty = np.zeros((0, items[0][0].tokens[0].shape[1]), dtype=np.float32)
    tokens = [s.tokens + [empty] * (length - len(s.tokens)) for s, _, _ in items]
    after = torch.arange(length, device=device)[None, :] >= torch.tensor([b for _, _, b in items], device=device)[:, None]
    counted = rows.asked & after
    advantage = torch.tensor([a for _, a, _ in items], dtype=torch.float32, device=device)[:, None] * counted
    return Samples(rows, Permitted.join([s.permitted for s, _, _ in items]), traffic_of(tokens, device), advantage,
                   counted)
