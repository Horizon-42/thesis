"""Same-validation-cohort comparison for A/B/C intent experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from dataset import (
    arrival_data_provenance,
    build_series,
    load_flight_dicts,
    require_matching_data_provenance,
)
from development_cohorts import load_development_cohort
from models import resolve_device
from train import evaluate_fixed_anchor_series, load_checkpoint


def compare_checkpoints(
    *,
    data: str | Path,
    cohort_path: str | Path,
    checkpoints: Sequence[str | Path],
    labels: Sequence[str],
    output_path: str | Path,
    device_name: str,
) -> dict:
    cohort = load_development_cohort(cohort_path)
    validation_ids = set(cohort.val_flight_ids)
    provenance = arrival_data_provenance(data)
    flights = load_flight_dicts(data, include_flight_keys=validation_ids)
    results = []
    reference_config = None
    for label, checkpoint in zip(labels, checkpoints, strict=True):
        model, config, normalizer, payload = load_checkpoint(checkpoint)
        require_matching_data_provenance(payload, provenance)
        if not validation_ids <= set(payload["split"]["val"]):
            raise ValueError(f"{label} checkpoint does not contain the comparison cohort")
        if reference_config is None:
            reference_config = config.to_dict()
        elif config.to_dict() != reference_config:
            raise ValueError("A/B/C checkpoints do not use the same model recipe")
        series, report = build_series(
            flights,
            config,
            aircraft_type=config.aircraft_type,
        )
        if {item.dataset_id for item in series} != validation_ids:
            raise ValueError(f"{label} could not rebuild the full comparison cohort")
        ordered = sorted(series, key=lambda item: item.dataset_id)
        evaluation = evaluate_fixed_anchor_series(
            model.to(resolve_device(device_name)),
            ordered,
            normalizer,
            config,
            resolve_device(device_name),
            split_name="shared-validation-cohort",
        )
        checkpoint_path = Path(checkpoint)
        results.append({
            "label": label,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "training_flights": {
                "train": len(payload["split"]["train"]),
                "val": len(payload["split"]["val"]),
            },
            "build_report": report.to_dict(),
            "shared_validation": evaluation,
        })
    document = {
        "schema_version": "ts-approach-intent-comparison-v1",
        "cohort": {
            "name": cohort.name,
            "validation_flights": len(cohort.val_flight_ids),
            "artifact_sha256": hashlib.sha256(
                Path(cohort_path).read_bytes()
            ).hexdigest(),
        },
        "outer_test_tracks_opened": False,
        "experiments": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document
