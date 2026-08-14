"""final_approach — the single producer of flown final-approach geometry.

``trajectory_data_process`` projects a track into runway-aligned frames and fits
the final segment to select the runway. Its threshold-event producer then uses a
valid measured threshold bracket for lateral position when available and a
producer-side final-segment fit for vertical height. Evaluation consumes that
policy-free event; it does not import or rerun the fitter.

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

Why the fit still exists: it supplies the runway-assignment geometry, and many
crowd-sourced ADS-B tracks stop before the threshold. Those unbracketed tracks need
producer-side extrapolation. Even when positions bracket the threshold, OpenSky
``geoaltitude`` is not guaranteed to share the position's update time; the event
therefore combines direct lateral interpolation with a fitted vertical intercept.
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
)
from final_approach.fit import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_SPAN_M,
    DEFAULT_WINDOW_M,
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
    "LandingScreen",
    "Assignment",
    "Outcome",
    "assign_runway",
    "AMBIGUITY_MARGIN_M",
]
