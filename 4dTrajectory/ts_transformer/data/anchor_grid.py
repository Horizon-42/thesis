"""The remaining-path anchor grid — ONE definition, three consumers.

The A0 curve (``experiments/anytime_curve.py``) replays a trained checkpoint from a grid of
later anchors; the ``anchor-grid-common-grid-ade`` checkpoint-selection metric
(:mod:`validation`) scores the validation split at a subset of the SAME grid. If the two
ever drifted apart, "the anytime curve improved" and "this epoch was selected on the
anytime curve" would be claims about different anchors. So the future floor, the per-flight
anchor rule and the fixed-at-L−1 stratum rule live here and nowhere else — the runner and
the metric both import them.

The third consumer is TRAINING (``random_train_anchor_sampling``, A0.b), and it sits on the
far side of an import edge: ``dataset`` cannot import this module, because this module
imports ``dataset``. So the grid's VALUES — :data:`DEFAULT_ANCHOR_GRID_KM`, the strata they
cut when read as edges, the per-flight draw law — live in the leaf :mod:`anchor_strata` and
are RE-EXPORTED here, the same objects, so every reading consumer keeps one import site
while the training sampler reads the same numbers at module scope.

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

**The stratum rule** (:func:`strata_fixed_at_anchor`): the label is computed once at the
evaluation anchor (the checkpoint's fixed anchor, `config.default_anchor`: L−1 unless the run
carries an `anchor_floor_index`) and reused at every bin. Relabelling per bin would drop each flight out
of the vectored stratum exactly when it rolled out on the centreline, and the curve would
be measuring the survivors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ts_transformer.data.approach_difficulty import (
    approach_difficulty,
    remaining_path_profile_m,
    strata_masks,
)
# Re-exported, not restated: `anchor_strata` is the leaf both this module and `dataset` read
# the remaining-path values from, so these names are the SAME objects on both sides of the
# edge (`tests/test_anchor_grid.py` asserts identity, not equality).
from ts_transformer.data.anchor_strata import (  # noqa: F401 — the grid's public surface lives here
    DEFAULT_ANCHOR_GRID_KM,
    REMAINING_PATH_STRATA_EDGES_M,
    REMAINING_PATH_STRATA_LABELS,
    bin_label,
    remaining_path_strata,
    remaining_path_uniform_offset,
)
from ts_transformer.data.dataset import FlightSeries, truth_duration_s

#: Truth required after an anchor for the bin to hold a reading. A bin nearer than this
#: many seconds of flight is empty BY CONSTRUCTION, not by accident: at approach speed
#: 2 km is ≈ 27 s and 4 km ≈ 53 s of remaining truth, both under this floor.
DEFAULT_GRID_MIN_FUTURE_S = 60.0

#: A bin holding less than this share of a cohort is `partial`: the flights missing from it
#: are exactly the ones whose geometry never reached it, so its value describes a
#: SUBCOHORT. A0's verdicts already refuse to read such a point (§六 2) and the selection
#: metric drops it — one threshold, both consumers.
PARTIAL_COVERAGE = 0.5

#: The bins the checkpoint-selection metric CONSIDERS, beside L−1 (A1). Which of them it
#: actually scores is decided per cohort at plan build, against :data:`PARTIAL_COVERAGE`.
#:
#: These are the candidates, chosen from measured coverage rather than by taste. On the
#: real KRDU validation cohort (1404 flights, 60 s floor):
#:
#: * **20 km — 33.7 %**, excluded from the tuple outright: most flights enter the 25 km
#:   arrival slice with less than 20 km of path left to fly (the MEDIAN flight has
#:   **13.4 km** left at L−1);
#: * **16 km — 37 %**: a candidate the coverage gate DROPS on this cohort. It stays in the
#:   tuple because coverage is a property of the airport and the split, not of the code —
#:   an airport whose arrivals enter further out would read it;
#: * **12 km — 78.7 %**, **8 km and 6 km — 99.7 %**: the bins that carry this cohort, and
#:   where the A0-random arm's geometry gain was measured (chamfer p50 −114…−524 m against
#:   the fixed arm at every anchor);
#: * **4 km and 2 km** are partial-to-empty under :data:`DEFAULT_GRID_MIN_FUTURE_S`
#:   (≈ 53 s and ≈ 27 s of truth left), so a value there would be scored on whichever
#:   handful of slow flights cleared the floor.
#:
#: A selection metric must not move with coverage, which is why there is a gate rather than
#: a hand-picked list: the surviving set is a property of the COHORT, hence identical
#: across every arm trained on the same split.
VALIDATION_ANCHOR_GRID_KM = (16, 12, 8, 6)

#: The same candidate bins in metres — what every anchor rule below takes.
VALIDATION_ANCHOR_GRID_M = tuple(
    float(km) * 1000.0 for km in VALIDATION_ANCHOR_GRID_KM
)


def remaining_path_profiles(series: Sequence[FlightSeries]) -> list[np.ndarray]:
    """The per-sample remaining path of each flight — computed once, read at every bin."""
    return [remaining_path_profile_m(item) for item in series]


def bin_anchor(series: FlightSeries, profile: np.ndarray, target_m: float, *,
               seq_len: int, min_future_s: float,
               minimum_anchor_index: int | None = None) -> int | None:
    """The flight's anchor for one remaining-path bin, or None when the bin is empty.

    The sample is chosen on remaining path ALONE and only then tested for eligibility: a
    flight whose closest sample cannot be an anchor has no reading at that bin, rather than
    a reading taken somewhere else on its track. ``minimum_anchor_index`` is the same
    experiment-supplied common floor `dataset.window_anchors` takes, and raises the
    lookback requirement the same way.
    """
    anchor = int(np.argmin(np.abs(profile - target_m)))
    if anchor < max(seq_len - 1, minimum_anchor_index or 0):
        return None
    if truth_duration_s(series, anchor) < min_future_s:
        return None
    return anchor


def anchors_for_bin(series: Sequence[FlightSeries], profiles: Sequence[np.ndarray],
                    target_m: float, *, seq_len: int, min_future_s: float,
                    minimum_anchor_index: int | None = None) -> dict[int, int]:
    """``{flight index: its anchor}`` for the flights that HAVE one at ``target_m``.

    Insertion order is the cohort's order, and a flight with no admissible anchor is
    ABSENT rather than present with a sentinel — every consumer's coverage count is then
    just ``len(...)``.

    The truth's end admission reads is the SUPERVISION rows' (``dataset.truth_duration_s``: the observed track closed
    to the threshold). The displacement readouts (`inference.receding`, `experiments.lead_time_error`) read the
    OBSERVED rows, which stop a median 6 s / 380 m short at KRDU — so a flight admitted with ``min_future_s`` of truth
    can have less readable truth; those readouts count it (``truth_shorter_than_horizon``) rather than raise.
    """
    anchors: dict[int, int] = {}
    for index, (item, profile) in enumerate(zip(series, profiles, strict=True)):
        anchor = bin_anchor(
            item, profile, target_m, seq_len=seq_len, min_future_s=min_future_s,
            minimum_anchor_index=minimum_anchor_index,
        )
        if anchor is not None:
            anchors[index] = anchor
    return anchors


def difficulty_at_anchor(series: Sequence[FlightSeries], keys: Sequence[str], *,
                         anchor: int) -> dict[str, dict]:
    """Every flight's stratum covariates at the evaluation ``anchor`` (`approach_difficulty`), by key —
    what `strata_fixed_at_anchor` stratifies, for a readout that keeps them per flight."""
    return {key: approach_difficulty(item, anchor).to_dict() for key, item in zip(keys, series, strict=True)}


def strata_fixed_at_anchor(series: Sequence[FlightSeries], keys: Sequence[str], *,
                           anchor: int) -> dict[str, np.ndarray]:
    """The stratum masks, computed ONCE at the evaluation ``anchor`` and valid at every bin
    (and every later lead).

    §六 1: a flight that is vectored at the evaluation anchor and established at 8 km stays
    in the vectored stratum at 8 km. The anchor is the caller's `default_anchor(config)` —
    a checkpoint trained at a common floor is judged there, and stratified there (it was
    ``seq_len - 1`` until 2026-09-16, when the floor made the two differ).
    """
    return strata_masks(difficulty_at_anchor(series, keys, anchor=anchor), list(keys))
