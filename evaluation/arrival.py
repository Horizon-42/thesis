"""The arrival event: where a trajectory actually reached its target.

A SOLVE AND AN OBSERVATION DO NOT ARRIVE THE SAME WAY
-----------------------------------------------------
``metrics.final_state_deviation`` measures ``states[-1]`` against ``target_state``. For a
SOLVE that is exact: the trajectory terminates at its target by construction. For an
OBSERVATION it is wrong, and not by a little -- 970 of 996 KRDU arrivals end a median
325 m short of the threshold and still ~135 ft up, because crowd-sourced ADS-B receivers
lose the aircraft before touchdown. Grading those on ``states[-1]`` measures where
reception stopped. It scored real, completed airline landings at **0.9%** on the vertical
gate.

So the arrival event is dispatched on the record's subject rather than assumed:

    optimized / predicted   ->  states[-1] vs target_state   (unchanged, byte for byte)
    observed                ->  the fitted final approach, extrapolated to the threshold

On the same KRDU data that raises the vertical gate from 0.9% to **49.2%**, and the fit
self-validates: the recovered glidepath is 3.02-3.11 deg at all five airports.

WHY A NEW SUBJECT FIELD
-----------------------
``controls == []`` cannot distinguish an observation from a prediction -- the ts exporter
writes reference-shaped records too. So the subject is explicit on ``source.subject``.
Absent, it defaults to ``optimized``; that default is safe for a specific reason, not by
luck: ``optimized`` and ``predicted`` share one code path, so a mislabel between them
cannot change any number. Only ``observed`` alters behaviour, and only the observed
writer emits it.

NOT-ESTABLISHED IS A RESULT, NOT A DROP
---------------------------------------
An observed track whose final approach cannot be fitted has no arrival to measure. That
is reported as ``not_established`` and counted, never silently extrapolated and never
dropped: the established rate is 21-54% on real data, far too large a bucket to hide.
It is also kept distinct from the harvest's ``unassignable`` -- that one means the
receiver lost the aircraft, this one means the approach was not stabilised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from final_approach import RunwayFrame, TrackPoint, fit_final_segment
from final_approach.fit import DEFAULT_WINDOW_M
from geokit import haversine_m

from evaluation.records import TrajectoryRecord

Subject = Literal["optimized", "predicted", "observed"]

DEFAULT_SUBJECT: Subject = "optimized"

# Established-on-final criteria, applied to a SegmentFit. These are quality thresholds
# and they live here, with the consumer -- deliberately not in ``final_approach``, whose
# assignment step must never filter on them (see that package's docstring).
MAX_CROSS_TRACK_M = 400.0
GLIDEPATH_RANGE_DEG = (2.0, 4.5)
MAX_VERTICAL_RMS_M = 6.0


@dataclass(frozen=True)
class EstablishedCriteria:
    """What makes an observed final approach measurable.

    ``max_vertical_rms_m`` is an RMS, not a max-residual, on purpose: a max-residual test
    is decided by a single bad sample. Measured on KRDU, ``max <= 15 m`` threw away 66
    otherwise clean approaches whose median fitted glidepath was 3.02 deg -- 12% of the
    usable data discarded by one blip each.
    """

    window_m: tuple[float, float] = DEFAULT_WINDOW_M
    max_cross_track_m: float = MAX_CROSS_TRACK_M
    glidepath_range_deg: tuple[float, float] = GLIDEPATH_RANGE_DEG
    max_vertical_rms_m: float = MAX_VERTICAL_RMS_M


@dataclass(frozen=True)
class ArrivalDeviation:
    """How far the arrival missed the target, and how well that is known.

    ``lateral_sigma_m`` / ``vertical_sigma_m`` are None for a solve (a computed final
    state has no measurement uncertainty) and populated for an observation, where the
    crossing is a fitted quantity. They are what make a gate verdict readable: on real
    data the 95% interval straddles a gate boundary for the majority of flights, so a
    bare pass rate overstates its own precision.
    """

    lateral_m: float
    vertical_m: float
    speed_ms: float
    heading_rad: float
    flight_time_s: float

    extrapolated: bool
    lateral_sigma_m: float | None = None
    vertical_sigma_m: float | None = None
    glidepath_deg: float | None = None
    extrapolation_m: float | None = None


@dataclass(frozen=True)
class ArrivalOutcome:
    """Either a deviation, or the reason there is none."""

    deviation: ArrivalDeviation | None
    established: bool | None  # None for subjects where the notion does not apply
    reason: str | None = None


def subject_of(record: TrajectoryRecord) -> Subject:
    """The record's subject, defaulting to ``optimized`` (see the module docstring)."""
    subject = record.source.get("subject", DEFAULT_SUBJECT)
    if subject not in ("optimized", "predicted", "observed"):
        raise ValueError(
            f"unknown subject {subject!r}; expected 'optimized', 'predicted' or 'observed'"
        )
    return subject


def arrival_deviation(
    record: TrajectoryRecord,
    *,
    criteria: EstablishedCriteria = EstablishedCriteria(),
) -> ArrivalOutcome:
    """Measure where this trajectory arrived, by the semantics its subject deserves."""
    if subject_of(record) != "observed":
        return ArrivalOutcome(deviation=final_state_deviation(record), established=None)
    return _observed_arrival(record, criteria)


def final_state_deviation(record: TrajectoryRecord) -> ArrivalDeviation:
    """states[-1] vs target_state — the solve/prediction measure (solved records only).

    The single definition of the final-state deviation; ``evaluation`` re-exports it.
    """
    final, target = record.states[-1], record.target_state
    return ArrivalDeviation(
        lateral_m=haversine_m(final["lat"], final["lon"], target["lat"], target["lon"]),
        vertical_m=final["alt"] - target["alt"],
        speed_ms=final["V"] - target["V"],
        heading_rad=math.remainder(final["psi"] - target["psi"], math.tau),
        flight_time_s=final["t"],
        extrapolated=False,
    )


def _observed_arrival(
    record: TrajectoryRecord, criteria: EstablishedCriteria
) -> ArrivalOutcome:
    """Fit the flown final approach and extrapolate it to the threshold."""
    course = record.source.get("runway_course_deg")
    if course is None:
        raise ValueError(
            f"observed record {record.source.get('id')!r} has no source.runway_course_deg; "
            "the observed writer must stamp it (evaluation cannot read the runway config)"
        )
    target = record.target_state
    frame = RunwayFrame(
        ident=str(record.source.get("runway", "?")),
        lat=target["lat"],
        lon=target["lon"],
        elevation_m=target["alt"],  # target altitude IS threshold elevation + published TCH
        course_deg=float(course),
    )
    fit = fit_final_segment(
        [TrackPoint(s["lat"], s["lon"], s["alt"]) for s in record.states],
        frame,
        window_m=criteria.window_m,
    )
    if fit is None:
        return ArrivalOutcome(
            None, established=False,
            reason=f"no final segment in [{criteria.window_m[0]:.0f}, {criteria.window_m[1]:.0f}] m",
        )

    unmet = _unmet_criteria(fit, criteria)
    if unmet:
        return ArrivalOutcome(None, established=False, reason="; ".join(unmet))

    final = record.states[-1]
    return ArrivalOutcome(
        deviation=ArrivalDeviation(
            # The frame is anchored AT the target, so the fitted crossing offsets are the
            # deviations -- no second distance computation, hence no second convention.
            lateral_m=abs(fit.cross_at_threshold_m),
            vertical_m=fit.height_at_threshold_m,
            # Ungated context, taken from the last observed sample rather than
            # extrapolated: neither is compared against a limit, and extrapolating speed
            # over the truncated tail would amplify noise for no gain.
            speed_ms=final["V"] - target["V"],
            heading_rad=math.remainder(final["psi"] - target["psi"], math.tau),
            flight_time_s=final["t"],
            extrapolated=True,
            lateral_sigma_m=fit.cross.sigma_at_zero,
            vertical_sigma_m=fit.height.sigma_at_zero,
            glidepath_deg=fit.glidepath_deg,
            extrapolation_m=fit.extrapolation_m,
        ),
        established=True,
    )


def _unmet_criteria(fit, criteria: EstablishedCriteria) -> list[str]:
    unmet: list[str] = []
    if fit.median_abs_cross_m > criteria.max_cross_track_m:
        unmet.append(
            f"median cross-track {fit.median_abs_cross_m:.0f} m > {criteria.max_cross_track_m:.0f} m"
        )
    low, high = criteria.glidepath_range_deg
    if not low <= fit.glidepath_deg <= high:
        unmet.append(f"glidepath {fit.glidepath_deg:.2f} deg outside [{low}, {high}]")
    if fit.height.rms_residual_m > criteria.max_vertical_rms_m:
        unmet.append(
            f"vertical RMS {fit.height.rms_residual_m:.1f} m > {criteria.max_vertical_rms_m:.1f} m"
        )
    return unmet
