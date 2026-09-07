"""The remaining-path anchor grid — ONE definition, two consumers.

The A0 curve (``run_ts_anytime_curve.py``) replays a trained checkpoint from a grid of
later anchors; the ``anchor-grid-common-grid-ade`` checkpoint-selection metric
(:mod:`validation`) scores the validation split at a subset of the SAME grid. If the two
ever drifted apart, "the anytime curve improved" and "this epoch was selected on the
anytime curve" would be claims about different anchors. So the bins, the future floor, the
per-flight anchor rule and the fixed-at-L−1 stratum rule live here and nowhere else — the
runner and the metric both import them.

Anytime-prediction design
(``docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md``) §2.3, §六 1–2.

**Why remaining PATH and not time**: it is the covariate the NEAR / FAR strata are already
cut on (``approach_difficulty.remaining_path_m``, whose per-sample form
``remaining_path_profile_m`` this module reads rather than restating), and it is comparable
across flights of different speeds.

**The per-flight anchor rule** (:func:`bin_anchor`): the observed sample whose remaining
path is CLOSEST to the bin value. The sample is chosen on remaining path ALONE and only
then tested for admissibility — a full lookback (``anchor >= seq_len - 1``) and at least
``min_future_s`` of truth after it — so a flight whose closest sample cannot be an anchor
has NO reading at that bin, rather than a reading taken somewhere else on its track.

**The stratum rule** (:func:`strata_fixed_at_l1`): the label is computed once at the L−1
evaluation anchor and reused at every bin. Relabelling per bin would drop each flight out
of the vectored stratum exactly when it rolled out on the centreline, and the curve would
be measuring the survivors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from approach_difficulty import (
    approach_difficulty,
    remaining_path_profile_m,
    strata_masks,
)
from dataset import FlightSeries, truth_duration_s

#: The full measurement grid, in km of remaining path (design §2.3). The A0 runner's
#: default; a whole curve, read from far to near.
DEFAULT_ANCHOR_GRID_KM = (20, 16, 12, 8, 6, 4, 2)

#: Truth required after an anchor for the bin to hold a reading. A bin nearer than this
#: many seconds of flight is empty BY CONSTRUCTION, not by accident: at approach speed
#: 2 km is ≈ 27 s and 4 km ≈ 53 s of remaining truth, both under this floor.
DEFAULT_GRID_MIN_FUTURE_S = 60.0

#: The bins the checkpoint-selection metric scores, beside L−1 (A1).
#:
#: These are the READABLE, NON-PARTIAL bins of the full grid, and the choice is made from
#: A0's own coverage numbers rather than by taste:
#:
#: * **20 km** holds ≈ 35 % of the cohort — most flights enter the 25 km arrival slice
#:   with less than 20 km of path left to fly — which is below the ``partial`` threshold
#:   (0.5) the A0 verdicts already refuse to read;
#: * **4 km and 2 km** are partial-to-empty under :data:`DEFAULT_GRID_MIN_FUTURE_S`
#:   (≈ 53 s and ≈ 27 s of truth left), so a selection value would be scored on whichever
#:   handful of slow flights happened to clear the floor;
#: * **16 / 12 / 8 / 6 km** are the four that carry the whole cohort and are where the
#:   A0-random arm's geometry gain was measured (chamfer p50 −114…−524 m against the fixed
#:   arm at every anchor).
#:
#: A selection metric must not move with coverage, so it reads only these four.
VALIDATION_ANCHOR_GRID_KM = (16, 12, 8, 6)

#: The same four bins in metres — what every anchor rule below takes.
VALIDATION_ANCHOR_GRID_M = tuple(
    float(km) * 1000.0 for km in VALIDATION_ANCHOR_GRID_KM
)


def remaining_path_profiles(series: Sequence[FlightSeries]) -> list[np.ndarray]:
    """The per-sample remaining path of each flight — computed once, read at every bin."""
    return [remaining_path_profile_m(item) for item in series]


def bin_anchor(series: FlightSeries, profile: np.ndarray, target_m: float, *,
               seq_len: int, min_future_s: float) -> int | None:
    """The flight's anchor for one remaining-path bin, or None when the bin is empty.

    The sample is chosen on remaining path ALONE and only then tested for eligibility: a
    flight whose closest sample cannot be an anchor has no reading at that bin, rather than
    a reading taken somewhere else on its track.
    """
    anchor = int(np.argmin(np.abs(profile - target_m)))
    if anchor < seq_len - 1:
        return None
    if truth_duration_s(series, anchor) < min_future_s:
        return None
    return anchor


def anchors_for_bin(series: Sequence[FlightSeries], profiles: Sequence[np.ndarray],
                    target_m: float, *, seq_len: int,
                    min_future_s: float) -> dict[int, int]:
    """``{flight index: its anchor}`` for the flights that HAVE one at ``target_m``.

    Insertion order is the cohort's order, and a flight with no admissible anchor is
    ABSENT rather than present with a sentinel — every consumer's coverage count is then
    just ``len(...)``.
    """
    anchors: dict[int, int] = {}
    for index, (item, profile) in enumerate(zip(series, profiles, strict=True)):
        anchor = bin_anchor(
            item, profile, target_m, seq_len=seq_len, min_future_s=min_future_s
        )
        if anchor is not None:
            anchors[index] = anchor
    return anchors


def strata_fixed_at_l1(series: Sequence[FlightSeries], keys: Sequence[str], *,
                       seq_len: int) -> dict[str, np.ndarray]:
    """The stratum masks, computed ONCE at the L−1 anchor and valid at every bin.

    §六 1: a flight that is vectored at the evaluation anchor and established at 8 km stays
    in the vectored stratum at 8 km.
    """
    anchor_l1 = seq_len - 1
    difficulty = {
        key: approach_difficulty(item, anchor_l1).to_dict()
        for key, item in zip(keys, series, strict=True)
    }
    return strata_masks(difficulty, list(keys))
