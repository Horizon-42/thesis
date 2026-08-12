"""final_approach — the single producer of flown final-approach geometry.

``trajectory_data_process`` projects a track into a runway-aligned frame, fits
the final segment once, selects the runway, and serializes a policy-free
threshold event. Evaluation consumes that event; it does not import or rerun
the fitter.

The two decisions remain separate:

  * the harvest asks **which runway** — a RELATIVE comparison (``assign_runway``
    takes the arg-min over candidate runways) that must never reject a track for
    flying badly;
  * evaluation asks **how good** — benchmark/runway bounds applied to the
    serialized crossing and uncertainty.

That split is load-bearing. If the harvest filtered on the quality criterion that
evaluation later reports, every surviving track would pass by construction. So
this package returns FACTS ONLY: it exposes no ``established`` flag, no
pass/fail verdict, and no regulation constant. Thresholds live with their consumer.

Why the fit exists at all: ADS-B coverage from crowd-sourced ground receivers stops
before touchdown (measured on 996 KRDU arrivals: 970 end short of the threshold, a
median 325 m out and still ~135 ft up, at a clean 1 Hz cadence right to the cut).
The last sample therefore records where reception ended, not where the aircraft
crossed. Fitting the flown segment and extrapolating recovers the crossing; the
recovered glidepath (3.02-3.11 deg across five airports) is the self-check that it
recovers a real approach rather than inventing one.

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
