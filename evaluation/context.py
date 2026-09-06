"""Resolve evaluation-owned assessment context from authoritative airport data."""

from __future__ import annotations

from collections.abc import Mapping

from evaluation.records import TrajectoryRecord
from evaluation.thresholds import AssessmentContext
from trajectory_data_process.harvest.airports import (
    Airport,
    Runway,
    threshold_frame_fingerprint,
)

ContextKey = tuple[str, str]

# ``procedure_source`` for a non-LPV runway whose RNAV approach publishes no
# vertical path at all (KRDU 14): there is no procedure to cite.
NO_VERTICAL_GUIDANCE_SOURCE = "no_published_vertical_guidance"


def assessment_for_runway(
    runway: Runway,
    *,
    benchmark: str | None = None,
) -> AssessmentContext:
    """Create policy context without adding it to a trajectory artifact.

    LPV is selected only when the CIFP exposes an LPV course width. The sole
    fallback is RNP APCH LNAV/VNAV, and its ±22 m vertical limit applies only when
    the runway's own RNAV (GPS) approach publishes an LNAV/VNAV (Baro-VNAV) line of
    minima AND a threshold path -- TCH and glidepath decoded from that approach's
    runway leg (``harvest.cifp.read_approach_verticals``; KRDU 32 and KSMF 35R).
    Both facts come from the runway data, never from a caller's say-so. A runway
    with neither keeps its lateral verdict and grades vertical indeterminate.
    """
    selected = benchmark or (
        "lpv" if runway.lpv_course_width_m is not None
        else "rnp_apch_lnav_vnav_baro"
    )
    if selected == "lpv" and runway.lpv_course_width_m is None:
        raise ValueError(f"{runway.airport} {runway.ident}: no LPV FAS course width")
    return AssessmentContext(
        benchmark=selected,  # type: ignore[arg-type]
        airport=runway.airport,
        runway=runway.ident,
        threshold_lat=runway.lat,
        threshold_lon=runway.lon,
        runway_course_deg=runway.course_deg,
        runway_width_m=runway.width_m,
        runway_source=runway.width_source,
        runway_source_cycle=runway.runway_source_cycle,
        procedure_source=(
            runway.position_source if selected == "lpv"
            else runway.tch_source or NO_VERTICAL_GUIDANCE_SOURCE
        ),
        procedure_source_cycle=runway.procedure_source_cycle,
        threshold_elevation_hae_m=runway.elevation_hae_m,
        threshold_elevation_msl_m=runway.elevation_msl_m,
        threshold_crossing_height_m=runway.threshold_crossing_height_m,
        lpv_course_width_m=(
            runway.lpv_course_width_m if selected == "lpv" else None
        ),
        baro_vnav_approved=(runway.baro_vnav_minima if selected != "lpv" else False),
        threshold_frame_fingerprint=threshold_frame_fingerprint(runway),
    )


def contexts_for_airport(airport: Airport) -> dict[ContextKey, AssessmentContext]:
    return {
        (airport.code, runway.ident): assessment_for_runway(runway)
        for runway in airport.runways
    }


def resolve_context(
    record: TrajectoryRecord,
    contexts: Mapping[ContextKey, AssessmentContext],
) -> AssessmentContext:
    key = (record.airport, record.runway)
    try:
        return contexts[key]
    except KeyError as exc:
        raise ValueError(f"no assessment context for {key[0]} runway {key[1]}") from exc
