"""Audited data selection shared by train-only capacity diagnostics.

These helpers inspect split identities from the manifest but open trajectory values for
exactly one outer-train flight.  They intentionally provide no validation/test loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

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


@dataclass(frozen=True)
class TrainOnlySelection:
    series: FlightSeries
    report: BuildReport
    provenance: dict
    split_keys: dict[str, set[str]]


def rank_outer_train_candidates(
    train_keys: Sequence[str], *, ranking_namespace: str, split_seed: int
) -> list[str]:
    """Stable train-only candidate order independent from optimizer randomness."""
    return sorted(
        train_keys,
        key=lambda key: hashlib.sha256(
            f"{ranking_namespace}:{split_seed}:{key}".encode()
        ).digest(),
    )


def select_outer_train_series(
    manifest: Path,
    config: TSConfig,
    *,
    airport: str,
    requested_id: str | None,
    ranking_namespace: str,
) -> TrainOnlySelection:
    """Open one usable outer-train flight without opening val/test trajectory values."""
    provenance = arrival_data_provenance([manifest])
    split_keys = flight_keys_by_split(provenance, config)
    train_keys = split_keys["train"]
    if requested_id is not None:
        requested_id = (
            requested_id
            if requested_id.startswith(f"{airport}:")
            else f"{airport}:{requested_id}"
        )
        if requested_id not in train_keys:
            raise ValueError(f"requested flight {requested_id!r} is not in outer-train")
        candidates = [requested_id]
    else:
        candidates = rank_outer_train_candidates(
            train_keys,
            ranking_namespace=ranking_namespace,
            split_seed=config.resolved_split_seed,
        )

    for dataset_id in candidates:
        flights = load_flight_dicts([manifest], include_flight_keys={dataset_id})
        series, report = build_series(
            flights,
            config,
            airport=airport,
            aircraft_type=config.aircraft_type,
        )
        if series:
            item = series[0]
            if split_name_for_dataset_id(item.dataset_id, config) != "train":
                raise RuntimeError("selected diagnostic trajectory escaped outer-train")
            return TrainOnlySelection(item, report, provenance, split_keys)
        if requested_id is not None:
            raise ValueError(
                f"requested outer-train flight {requested_id!r} is unusable: "
                f"{report.format()}"
            )
    raise ValueError("no usable outer-train flight found for the diagnostic")
