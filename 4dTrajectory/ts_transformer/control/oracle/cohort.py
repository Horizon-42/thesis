"""Audited multi-flight outer-train selection for oracle-teacher experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import TSConfig
from dataset import (
    BuildReport,
    FlightSeries,
    arrival_data_provenance,
    build_series,
    flight_keys_by_split,
    load_flight_dicts,
    split_name_for_dataset_id,
)
from train_only_diagnostics import rank_outer_train_candidates


@dataclass(frozen=True)
class OracleTeacherCohort:
    series: tuple[FlightSeries, ...]
    report: BuildReport
    provenance: dict
    split_keys: dict[str, set[str]]
    ranked_candidates_opened: tuple[str, ...]


def select_outer_train_cohort(
    manifest: Path,
    config: TSConfig,
    *,
    airport: str,
    cohort_size: int,
) -> OracleTeacherCohort:
    """Open a deterministic train-only candidate pool and return usable flights."""
    if cohort_size < 1:
        raise ValueError("oracle teacher cohort_size must be positive")
    provenance = arrival_data_provenance([manifest])
    split_keys = flight_keys_by_split(provenance, config)
    ranked = rank_outer_train_candidates(
        split_keys["train"],
        ranking_namespace="inverse-dynamics-oracle-teacher-v1",
        split_seed=config.resolved_split_seed,
    )
    opened = tuple(ranked[: cohort_size * 4])
    flights = load_flight_dicts([manifest], include_flight_keys=set(opened))
    built, report = build_series(
        flights,
        config,
        airport=airport,
        aircraft_type=config.aircraft_type,
    )
    indexed = {item.dataset_id: item for item in built}
    selected = tuple(indexed[key] for key in opened if key in indexed)[:cohort_size]
    if len(selected) != cohort_size:
        raise ValueError(
            f"only {len(selected)}/{cohort_size} ranked outer-train flights are usable"
        )
    if any(split_name_for_dataset_id(item.dataset_id, config) != "train" for item in selected):
        raise RuntimeError("oracle teacher cohort escaped outer-train")
    return OracleTeacherCohort(
        series=selected,
        report=report,
        provenance=provenance,
        split_keys=split_keys,
        ranked_candidates_opened=opened,
    )
