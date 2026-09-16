"""The plan token: the plan path's target vector at an anchor, fused into the control decoder.

Two-tier feasibility T1 (`docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §10.3): a
control model re-asked every 30 s on its own rollout, told the PLAN it is flying. The plan is
the plan path's own target vector (`outputs.plan.labels.TARGETS`: the operating parameters and
the next instruction in runway axes about the anchor) — the one definition the plan head is
trained on, so the head's prediction can later stand where the truth's stands now (T2).

``plan_conditioning = truth-next`` reads the TRUTH's plan: at an observed anchor through the
extractors (`extract_plan` + `targets_from_labels`, exactly the plan head's label there), at a
rolled pose through a `TruthExpert` read in time order. It reads the future, like
``cta=given``: the protocol-C oracle, never a prediction result, and the run name says so.

The token is ``[values / SCALE_VECTOR · valid, valid, next_is_join, present]``: an entry the
track does not define is 0 with its valid bit 0, and the ABSENT token (no plan, or the plan
dropped in training) is all zeros — ``present`` 0 is what separates "no plan" from a plan whose
entries all happen to be 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.labels import SCALE_VECTOR, TARGETS, PlanTargets, targets_from_labels
from ts_transformer.outputs.plan.skeleton import SkeletonCache

if TYPE_CHECKING:
    from ts_transformer.data.dataset import FlightSeries

#: The dynamics-context key the token rides under (beside `condition` and `cta_s`).
PLAN_TOKEN_KEY = "plan_token"
#: values, valid bits, the no-fix flag, the present flag.
PLAN_TOKEN_WIDTH = 2 * len(TARGETS) + 2


def plan_token(targets: PlanTargets | None) -> np.ndarray:
    """The token of one target vector; None is the absent token."""
    token = np.zeros(PLAN_TOKEN_WIDTH, dtype=np.float32)
    if targets is None:
        return token
    width = len(TARGETS)
    valid = np.asarray(targets.valid, dtype=np.float32)
    token[:width] = np.asarray(targets.values, dtype=np.float32) / SCALE_VECTOR * valid
    token[width : 2 * width] = valid
    token[2 * width] = float(targets.next_is_join)
    token[2 * width + 1] = 1.0
    return token


def truth_plan_token(series: FlightSeries, anchor: int, skeletons: SkeletonCache) -> np.ndarray:
    """The truth's plan read at an observed ``anchor`` — the plan head's label there."""
    skeleton = skeletons.for_series(series)
    return plan_token(targets_from_labels(extract_plan(series, anchor, skeleton), series, anchor, skeleton))


__all__ = ["PLAN_TOKEN_KEY", "PLAN_TOKEN_WIDTH", "plan_token", "truth_plan_token"]
