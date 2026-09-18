"""The plan token: what the control decoder is told about the plan it flies, one fused token.

One key (`PLAN_TOKEN_KEY`), one width per ``plan_conditioning`` value (`plan_token_width`).
The two-tier v2 tokens (``truth-next``, ``waypoints``) are archived with that design
(`archive/two_tier_v2_2026_09/plan_token_v2.py`); the manoeuvre-token plan
(`docs/2026-09-18_manoeuvre_token_plan.zh.md` §2.6) adds its ``manoeuvre-code`` token here in
P3.1 — the decoded descriptor of the next segment, the same at training (the truth's) and at
deployment (the prior's). Until then the only value is ``off`` and no token is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ts_transformer.config import PLAN_CONDITIONING_OFF, TSConfig

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The dynamics-context key the token rides under (beside `condition` and `cta_s`).
PLAN_TOKEN_KEY = "plan_token"


def plan_token_width(config: TSConfig) -> int:
    """The token's width under this run's ``plan_conditioning`` (the decoder's input size)."""
    if config.plan_conditioning == PLAN_CONDITIONING_OFF:
        return 0
    raise ValueError(f"no plan token is built for plan_conditioning={config.plan_conditioning!r}")


def training_plan_token(series: FlightSeries, anchor: int, config: TSConfig) -> np.ndarray:
    """The TRUTH's token at an observed ``anchor`` under this run's ``plan_conditioning`` — what
    a training row and `predict` hand the decoder."""
    raise ValueError(f"no plan token is built for plan_conditioning={config.plan_conditioning!r}")


def probe_plan_token(config: TSConfig) -> np.ndarray:
    """A PRESENT token of this run's width with nothing defined in it (the batch-size probe)."""
    raise ValueError(f"no plan token is built for plan_conditioning={config.plan_conditioning!r}")


__all__ = ["PLAN_TOKEN_KEY", "plan_token_width", "probe_plan_token", "training_plan_token"]
