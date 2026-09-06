"""Stable train-only candidate ordering for capacity diagnostics.

``select_outer_train_series``, which opened exactly one outer-train flight and returned a
``TrainOnlySelection``, was deleted in the T3 review (2026-09-07): its only caller had gone
to ``archive/oracle_teacher_2026_08/`` in T2, and a loader nothing loads through is a
split-discipline claim nothing checks. What is left is the ordering itself, which is pure.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def rank_outer_train_candidates(
    train_keys: Sequence[str], *, ranking_namespace: str, split_seed: int
) -> list[str]:
    """Stable train-only candidate order independent from optimizer randomness."""
    return sorted(
        train_keys,
        key=lambda key: hashlib.sha256(
            f"{ranking_namespace}:{split_seed}:{key}".encode()
        ).digest(),
    )
