"""final_approach — the single producer of flown final-approach geometry.

``trajectory_data_process`` first uses a source-valid threshold bracket when one
exists to select the runway and physical inbound pass. It then fits one robust
three-dimensional final segment for both threshold coordinates. Evaluation consumes
that policy-free event; it does not import or rerun the fitter.

The two decisions remain separate:

  * the harvest asks **which runway** — a RELATIVE comparison (``assign_runway``
    takes the arg-min over candidate runways) that must never reject a track for
    flying badly;
  * evaluation asks **how good** — benchmark/runway bounds applied to the
    serialized crossing and uncertainty, without fitting trajectory samples.

That split is load-bearing. If the harvest filtered on the quality criterion that
evaluation later reports, every surviving track would pass by construction. So
this package returns FACTS ONLY: it exposes no ``established`` flag, no
pass/fail verdict, and no regulation constant. Thresholds live with their consumer.

Why the fit still exists: many crowd-sourced ADS-B tracks stop before the threshold,
and raw bracket altitude could not be validated consistently across airports. A
bracket therefore anchors a pass but does not become one component of a hybrid point.
See ``FIT_MODEL_OPTIMIZATION.md`` for the metadata experiments and uncertainty design.

Datum contract (stated once, enforced nowhere): ``TrackPoint.alt_m`` and
``RunwayFrame.elevation_m`` MUST share a vertical datum. The harvest works in
ellipsoidal height (HAE, as the sensor reported); the modeling plane works in MSL.
This package never converts -- it subtracts, so a caller that mixes the two gets a
~33 m error with no warning. Assignment reads only cross-track and is datum-free.
"""

from __future__ import annotations

from final_approach.assign import (
    AMBIGUITY_MARGIN_M,
    Assignment,
    LandingScreen,
    Outcome,
    assign_runway,
    landing_screen_reason,
)
from final_approach.fit import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_SPAN_M,
    DEFAULT_WINDOW_M,
    LATERAL_FIT_MODEL_FLOOR_95_M,
    LineFit,
    SegmentFit,
    fit_final_segment,
)
from final_approach.frame import (
    Projected,
    RunwayFrame,
    TrackPoint,
)

__all__ = [
    "TrackPoint",
    "Projected",
    "RunwayFrame",
    "LineFit",
    "SegmentFit",
    "fit_final_segment",
    "DEFAULT_WINDOW_M",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_MIN_SPAN_M",
    "LATERAL_FIT_MODEL_FLOOR_95_M",
    "LandingScreen",
    "Assignment",
    "Outcome",
    "assign_runway",
    "landing_screen_reason",
    "AMBIGUITY_MARGIN_M",
]
