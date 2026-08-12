"""Resolve evaluation-owned assessment context from authoritative airport data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from evaluation.records import TrajectoryRecord
from evaluation.thresholds import AssessmentContext
from trajectory_data_process.harvest.airports import Airport, Runway

ContextKey = tuple[str, str]


def assessment_for_runway(
    runway: Runway,
    *,
    benchmark: str | None = None,
    baro_vnav_approved: bool = False,
) -> AssessmentContext:
    """Create policy context without adding it to a trajectory artifact.

    LPV is selected only when the CIFP exposes an LPV course width.  The sole
    fallback is RNP APCH LNAV/VNAV, and its vertical limit remains unavailable
    unless the caller explicitly confirms approved Baro-VNAV applicability.
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
        runway_course_deg=runway.course_deg,
        runway_width_m=runway.width_m,
        runway_source=runway.width_source,
        runway_source_cycle=runway.runway_source_cycle,
        procedure_source=(
            runway.position_source if selected == "lpv" else "faa_terminal_procedure"
        ),
        procedure_source_cycle=runway.procedure_source_cycle,
        lpv_lateral_fsd_m=(
            runway.lpv_course_width_m if selected == "lpv" else None
        ),
        # Deliberately indeterminate until licensed RTCA scaling is validated.
        lpv_vertical_fsd_m=None,
        baro_vnav_approved=(baro_vnav_approved if selected != "lpv" else False),
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
    airport = record.source.get("arr_airport") or record.source.get("airport")
    runway = record.source.get("runway")
    if not isinstance(airport, str) or not isinstance(runway, str):
        raise ValueError(
            f"record {record.path or record.source.get('id')!r} requires "
            "source.arr_airport and source.runway"
        )
    key = (airport.upper(), runway)
    try:
        context = contexts[key]
    except KeyError as exc:
        raise ValueError(f"no assessment context for {key[0]} runway {key[1]}") from exc
    return context


def used_contexts(
    records: Iterable[TrajectoryRecord],
    contexts: Mapping[ContextKey, AssessmentContext],
) -> list[AssessmentContext]:
    unique: dict[ContextKey, AssessmentContext] = {}
    for record in records:
        context = resolve_context(record, contexts)
        unique[(context.airport, context.runway)] = context
    return [unique[key] for key in sorted(unique)]
