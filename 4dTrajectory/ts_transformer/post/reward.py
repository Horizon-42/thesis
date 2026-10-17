"""The reward of the post-training (post-training §2 item 2; D30, D91, D207): only from the outcome.

| The outcome of the sentence | Reward |
|---|---|
| ``landed``, no go-around | 1 |
| ``landed`` after n go-arounds | 0.9ⁿ |
| every other outcome; every loss of separation that the commanded aircraft answers for | 0 |

The runway of the landing does not change the reward (D207): a landing on any candidate runway counts.
"""

from __future__ import annotations

#: D30: the reward of a landing after n go-arounds is ``GO_AROUND_FACTOR ** n``.
GO_AROUND_FACTOR = 0.9
#: MIRROR of the judge's outcome of a landing (`autopilot.judge.OUTCOMES`; `post/` does not import `autopilot/`, the
#: architecture test): `tests/test_post_window_loop.py` pins the two.
LANDED = "landed"


def reward(outcome: str, go_arounds: int, lost_separation: bool) -> float:
    """D30's reward (module docstring) of a sentence: the judge's ``outcome``, its go-arounds, and whether it ended in
    a loss of separation that the commanded aircraft answers for."""
    if lost_separation or outcome != LANDED:
        return 0.0
    return GO_AROUND_FACTOR ** go_arounds
