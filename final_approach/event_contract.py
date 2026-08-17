"""The policy-free runway-threshold event: its discriminators and its schema.

The producer (``trajectory_data_process.harvest.threshold_event``) and the evaluator
(``evaluation.arrival``) live in different packages on purpose. This module is the
seam between them: the discriminator strings, plus the ONE function that decides
whether a serialized event is well formed. Both sides call it, so the schema cannot
drift between the side that writes an event and the side that grades it -- which is
exactly what two hand-rolled validators used to allow.

Identity is deliberately NOT checked here. The producer holds a ``Runway`` and the
evaluator an ``AssessmentContext``; each binds the event to its own frame and says so
in its own terms. This module makes no reference to either package.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

EVENT_SCHEMA_VERSION = "runway-threshold-event-v1"
DIRECT_EVENT_METHOD = "direct_linear_bracket"
CENSORED_EVENT_METHOD = "censored_robust_line"
NO_EVENT_METHOD = "none"

ESTIMATED_OBSERVABILITY_BY_METHOD = {
    DIRECT_EVENT_METHOD: "within_observed_support",
    CENSORED_EVENT_METHOD: "right_censored",
}
UNAVAILABLE_OBSERVABILITIES = frozenset({"invalid_support", "unavailable"})

# Present and finite on every estimated event, whichever estimator produced it.
_ESTIMATED_NUMBERS = (
    "threshold_crossing_lat",
    "threshold_crossing_lon",
    "threshold_crossing_altitude_m",
    "signed_cross_track_m",
    "extrapolation_distance_m",
)


def validate_event(event: Mapping[str, Any]) -> str:
    """Validate one serialized event and return its ``status``.

    Raises ``ValueError`` naming the offending field. Anything a rebuild would fix
    says ``--reclassify-existing``, because regenerating the event is the only cure:
    these payloads are measured artifacts, not values a consumer may repair.
    """
    if event.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise ValueError(
            f"threshold event must use {EVENT_SCHEMA_VERSION}; run --reclassify-existing"
        )
    status = event.get("status")
    method = event.get("method")
    observability = event.get("observability")

    if status == "unavailable":
        if method != NO_EVENT_METHOD or observability not in UNAVAILABLE_OBSERVABILITIES:
            raise ValueError("unavailable threshold event has invalid discriminators")
        if not isinstance(event.get("unavailable_reason"), str):
            raise ValueError("unavailable threshold event requires a reason")
        return status
    if status != "estimated":
        raise ValueError(f"threshold event has invalid status {status!r}")

    # ``.get(method)`` alone is not enough: an event carrying NEITHER field compares
    # None != None -> False and would be accepted, then fall through to the censored
    # branch and be graded as a real crossing. The method must be one this contract
    # knows before its observability is worth comparing.
    expected_observability = ESTIMATED_OBSERVABILITY_BY_METHOD.get(method)
    if expected_observability is None or expected_observability != observability:
        raise ValueError(
            "unsupported threshold-event method/observability "
            f"{method!r}/{observability!r}; run --reclassify-existing"
        )
    if event.get("altitude_datum") != "hae":
        raise ValueError("threshold event altitude_datum must be 'hae'")
    _require_source_sample_range(event.get("source_sample_range"))
    for key in _ESTIMATED_NUMBERS:
        _finite(event, key)
    if event.get("uncertainty") != {"status": "uncalibrated"}:
        raise ValueError("threshold event uncertainty must be explicitly uncalibrated")

    # The two estimators are distinguished by geometry, not only by their name: a
    # direct event interpolates inside observed support (fraction in (0, 1], nothing
    # extrapolated), a censored one extrapolates past the last sample (no fraction).
    extrapolation_m = float(event["extrapolation_distance_m"])
    if method == DIRECT_EVENT_METHOD:
        _finite(event, "event_time_s")
        fraction = _finite(event, "interpolation_fraction")
        if not 0.0 < fraction <= 1.0 or extrapolation_m != 0.0:
            raise ValueError("direct threshold event has invalid interpolation geometry")
    elif (
        event.get("event_time_s") is not None
        or event.get("interpolation_fraction") is not None
        or extrapolation_m <= 0.0
    ):
        raise ValueError("censored threshold event has invalid extrapolation geometry")
    return status


def _require_source_sample_range(value: Any) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or value[0] < 0
        or value[1] < value[0]
    ):
        raise ValueError("threshold event has an invalid source_sample_range")


def _finite(event: Mapping[str, Any], key: str) -> float:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"threshold event {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"threshold event {key} must be finite")
    return number
