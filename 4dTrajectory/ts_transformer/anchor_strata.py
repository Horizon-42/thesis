"""The remaining-path axis: the grid's values, the strata they cut, and the draw law.

A LEAF module — numpy plus :mod:`approach_difficulty`, no ``dataset`` and no torch — because
its two consumers sit on opposite sides of an import edge. :mod:`anchor_grid` (which imports
``dataset`` for ``FlightSeries`` and ``truth_duration_s``) re-exports everything here, so
"the grid" still has one import site for every reading consumer; ``dataset`` imports this
module directly, at module scope, for the training-anchor draw.

The point of the split is that the numbers are the SAME numbers. The anchors a model is
trained at, the bins a re-anchored curve is read at, and the strata an epoch's anchors are
counted in all come from :data:`DEFAULT_ANCHOR_GRID_KM` — read as targets by
``anchor_grid.bin_anchor``, as edges by :data:`REMAINING_PATH_STRATA_EDGES_M`. Two modules
that merely agreed today would make "the curve improved at 12 km" and "training saw 12 km"
claims about different lengths.

**Remaining PATH, not time**: it is the covariate the NEAR / FAR strata are already cut on
(``approach_difficulty.remaining_path_m``, whose per-sample form ``remaining_path_profile_m``
this module reads rather than restating), and it is comparable across flights of different
speeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from approach_difficulty import remaining_path_profile_m

if TYPE_CHECKING:  # `dataset` imports this module; the type is an annotation only
    from dataset import FlightSeries

#: The full measurement grid, in km of remaining path (design §2.3). The A0 runner's
#: default; a whole curve, read from far to near. Re-exported by :mod:`anchor_grid`, which
#: is where every reading consumer imports it from.
DEFAULT_ANCHOR_GRID_KM = (20, 16, 12, 8, 6, 4, 2)


def bin_label(target_m: float) -> str:
    """The published name of a bin: ``"16km"``, never a bare ``16000`` that reads as an ADE."""
    return f"{target_m / 1000:g}km"


#: The strata an epoch's drawn anchors are COUNTED in — the same grid values read as EDGES
#: rather than as targets, in ascending metres.
#:
#: They are bookkeeping, **not the sampling law**. A0.b's first draft did draw a stratum
#: uniformly and then a sample inside it; measured on 700 real KRDU validation flights that
#: moved training TOWARD the runway (mean remaining path 12.2 → 8.0 km, share ≥ 20 km
#: 16.9 % → 4.1 %), because the grid gives the near end four 2-km strata and the far end ONE
#: open stratum spanning 20–123 km — an eighth of the probability for a quarter of the
#: approach and a third of the anchors. The law is now
#: :func:`remaining_path_uniform_offset` — uniform over the flight's own remaining-path
#: range — and these edges only name the columns the histogram is reported in.
REMAINING_PATH_STRATA_EDGES_M = tuple(
    sorted(float(km) * 1000.0 for km in DEFAULT_ANCHOR_GRID_KM)
)

#: The published name of each stratum, ascending, one per interval. Spelled through
#: :func:`bin_label` so the km rendering keeps one owner. Seven edges, EIGHT strata: the two
#: open ends are strata of their own, so every anchor is counted in exactly one column.
REMAINING_PATH_STRATA_LABELS = (
    f"<{bin_label(REMAINING_PATH_STRATA_EDGES_M[0])}",
    *(
        f"{bin_label(low)}-{bin_label(high)}"
        for low, high in zip(
            REMAINING_PATH_STRATA_EDGES_M, REMAINING_PATH_STRATA_EDGES_M[1:]
        )
    ),
    f">={bin_label(REMAINING_PATH_STRATA_EDGES_M[-1])}",
)


def remaining_path_strata(series: "FlightSeries") -> np.ndarray:
    """The remaining-path stratum index of EVERY observed sample of one flight: ``[N]``.

    ``np.digitize`` over :data:`REMAINING_PATH_STRATA_EDGES_M` on the same profile the bins
    are chosen from, so a sample's stratum and its bin are two readings of one length. Index
    ``i`` is :data:`REMAINING_PATH_STRATA_LABELS` ``[i]``: 0 is under the first edge, the
    last is at or beyond the last edge.
    """
    return np.digitize(remaining_path_profile_m(series), REMAINING_PATH_STRATA_EDGES_M)


def remaining_path_uniform_offset(remaining_path_m: np.ndarray, unit_draw: float) -> int:
    """Pick one of a flight's admissible anchors, uniformly in REMAINING PATH.

    ``remaining_path_m`` is the remaining path at each anchor the flight actually stores, in
    that order; ``unit_draw`` ∈ [0, 1) is the flight's own draw for this epoch. The returned
    offset indexes that array.

    The law is: place ``unit_draw`` uniformly across the flight's OWN admissible span
    ``[min, max]`` and take the nearest admissible anchor — equal weight per kilometre of
    remaining path, rather than per sample. Drawing per sample is uniform in TIME, and time
    and path are not proportional along an approach: the aircraft is slow near the runway, so
    a kilometre there holds more samples than a kilometre at 25 km and collects more draws.

    Nearest, not bracketing, because the anchors are not a regular lattice — they thin out
    where the aircraft is fast and where eligibility removed samples — and a flight with a
    single admissible anchor must still return it.
    """
    target = float(
        remaining_path_m.min()
        + unit_draw * (remaining_path_m.max() - remaining_path_m.min())
    )
    return int(np.argmin(np.abs(remaining_path_m - target)))
