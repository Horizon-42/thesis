"""Record-level crossing markers: WHERE a trajectory's threshold crossing lives.

An evaluation record's states either CONTAIN the crossing (a solve terminates at
its target; an ADS-B track with a validated bracket straddles the plane) or END
BEFORE it (right-censored coverage). The marker serialized under
``source["crossing_span"]`` says which case the reader is holding and where to
look, so grading is one shared interpolation over the states instead of one
reading path per producer:

  ``{"kind": "measured_bracket", "left_index": i, "fraction": f}``
      The instrument-selected direct bracket — the harvest event's
      source-validated, minimum-|cross-track| choice, REPRODUCED from the event
      rather than re-derived, so the crossing graded is the crossing that chose
      the runway.

  ``{"kind": "fitted_tail", "start_index": n, ...}``
      States from ``start_index`` on are INFERRED (the censored final-approach
      fit), appended after the measured samples; the crossing is the appended
      row. ``final_time_s`` keeps pointing at the last MEASURED row, so no
      flight-time or Δt-vs-observed statistic moves.

Computed records carry no marker: their crossing is derived by the evaluator
from the raw states, so the artifact under test cannot author the quantity it
is graded on.
"""

from __future__ import annotations

from typing import Any, Mapping

from final_approach.crossing import (
    CROSSING_SPAN_KEY,
    FITTED_TAIL_KIND,
    MEASURED_BRACKET_KIND,
)

__all__ = [
    "CROSSING_SPAN_KEY",
    "FITTED_TAIL_KIND",
    "MEASURED_BRACKET_KIND",
    "V_SOURCE_EVENT_GROUND_SPEED",
    "V_SOURCE_LAST_MEASURED",
    "TIME_SOURCE_TRAPEZOIDAL",
    "crossing_span_from_event",
]

# v sources for the fitted-tail crossing row, most-trusted first. The row's V is a
# GROUND speed either way (audit-only; evaluation's airspeed gate never reads it) —
# the source is recorded so a reader can tell a measured-reported extrapolation from
# the last-establishied-sample carry-over used when the event fitted no speed.
V_SOURCE_EVENT_GROUND_SPEED = "event_ground_speed"
V_SOURCE_LAST_MEASURED = "last_measured_sample"

# The appended row needs a time only because states are timed; nothing grades on it
# and final_time_s does not point at it. Trapezoidal over the deceleration:
# dt = extrapolation / mean(V_last, V_crossing).
TIME_SOURCE_TRAPEZOIDAL = "trapezoidal_ground_speed"


def crossing_span_from_event(
    event: Mapping[str, Any],
    states: list[dict[str, float]],
    *,
    hae_minus_msl_m: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """Resolve one validated ESTIMATED event into ``(marker, rows_to_append)``.

    ``states`` are the record's MSL state rows, one per stored track sample in
    order (``state_samples_from_track`` guarantees the 1:1 alignment the event's
    ``source_sample_range`` indexes rely on). Direct events yield an empty
    append; censored events yield the single inferred crossing row.

    The caller has already validated the event (``validate_event`` + the frame
    binding); this function trusts its payload and refuses only what would make
    the marker structurally wrong.
    """
    if event.get("status") != "estimated":
        raise ValueError("crossing spans exist only for estimated events")

    extrapolation_m = float(event["extrapolation_distance_m"])
    if extrapolation_m == 0.0:
        left_index, right_index = event["source_sample_range"]
        if not 0 <= left_index < right_index < len(states):
            raise ValueError(
                f"event sample range [{left_index}, {right_index}] does not fit "
                f"a {len(states)}-state record"
            )
        return (
            {
                "kind": MEASURED_BRACKET_KIND,
                "left_index": left_index,
                "fraction": float(event["interpolation_fraction"]),
            },
            [],
        )

    last = states[-1]
    ground_speed = event.get("crossing_ground_speed_m_s")
    if ground_speed is not None:
        crossing_v = float(ground_speed)
        v_source = V_SOURCE_EVENT_GROUND_SPEED
    else:
        crossing_v = float(last["V"])
        v_source = V_SOURCE_LAST_MEASURED
    mean_v = (float(last["V"]) + crossing_v) / 2.0
    if mean_v <= 0.0:
        raise ValueError("cannot time the fitted crossing with a non-positive speed")
    crossing_row = {
        "t": float(last["t"]) + extrapolation_m / mean_v,
        "lat": float(event["threshold_crossing_lat"]),
        "lon": float(event["threshold_crossing_lon"]),
        # The event stores the crossing as broadcast (HAE); the record is MSL.
        "alt": float(event["threshold_crossing_altitude_m"]) - hae_minus_msl_m,
        "V": crossing_v,
        # Heading and path angle are not separately estimated at the crossing; the
        # last established values carry, exactly the quantities the pre-span reader
        # reported for observed rows.
        "psi": float(last["psi"]),
        "gamma": float(last["gamma"]),
        "m": float(last["m"]),
    }
    return (
        {
            "kind": FITTED_TAIL_KIND,
            "start_index": len(states),
            "v_source": v_source,
            "time_source": TIME_SOURCE_TRAPEZOIDAL,
            "extrapolation_m": extrapolation_m,
        },
        [crossing_row],
    )
