"""Serializable clustering result and exact development cohorts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dataset import FlightSeries
from development_cohorts import write_development_cohort

from .model import ApproachClusterModel


APPROACH_CLUSTER_SCHEMA = "ts-approach-clusters-v1"


def _assignments(
    series: Sequence[FlightSeries], labels: np.ndarray
) -> dict[str, int]:
    return {
        item.dataset_id: int(label) for item, label in zip(series, labels, strict=True)
    }


def write_clustering_artifacts(
    output_dir: str | Path,
    *,
    airport: str,
    runway: str,
    manifest_path: str | Path,
    config: dict[str, Any],
    feature_points: int,
    pca_components: int,
    cluster_seed: int,
    model: ApproachClusterModel,
    candidates: list[dict[str, Any]],
    train_series: Sequence[FlightSeries],
    train_labels: np.ndarray,
    val_series: Sequence[FlightSeries],
    val_labels: np.ndarray,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_k = len(model.centers)
    target_cluster = int(np.argmax(np.bincount(train_labels, minlength=selected_k)))
    train_assignments = _assignments(train_series, train_labels)
    val_assignments = _assignments(val_series, val_labels)
    manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    document = {
        "schema_version": APPROACH_CLUSTER_SCHEMA,
        "airport": airport,
        "runway": runway,
        "source": {
            "arrival_manifest_sha256": manifest_sha256,
            "split_seed": (
                config["seed"]
                if config["split_seed"] is None
                else config["split_seed"]
            ),
            "outer_test_tracks_opened": False,
        },
        "feature": {
            "channels": ["e", "n"],
            "parameterization": "equal-horizontal-arc",
            "points": feature_points,
            "anchor_index": config["seq_len"] - 1,
            "pca_components": pca_components,
        },
        "selection": {
            "scope": "train-only",
            "seed": cluster_seed,
            "candidates": candidates,
            "selected_k": selected_k,
            "target_cluster": target_cluster,
        },
        "model": model.to_dict(),
        "assignments": {
            "train": train_assignments,
            "val": val_assignments,
        },
    }
    cluster_path = output / "approach_clusters.json"
    cluster_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    qfu_path = output / "qfu_cohort.json"
    write_development_cohort(
        qfu_path,
        name=f"{airport}-{runway}-all-approaches",
        train_flight_ids=list(train_assignments),
        val_flight_ids=list(val_assignments),
        selection={
            "kind": "qfu",
            "airport": airport,
            "runway": runway,
        },
    )
    cluster_cohort_path = output / "dominant_cluster_cohort.json"
    write_development_cohort(
        cluster_cohort_path,
        name=f"{airport}-{runway}-cluster-{target_cluster}",
        train_flight_ids=[
            key for key, label in train_assignments.items() if label == target_cluster
        ],
        val_flight_ids=[
            key for key, label in val_assignments.items() if label == target_cluster
        ],
        selection={
            "kind": "approach-cluster",
            "airport": airport,
            "runway": runway,
            "cluster_artifact_sha256": hashlib.sha256(
                cluster_path.read_bytes()
            ).hexdigest(),
            "cluster": target_cluster,
        },
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": "ts-approach-cluster-summary-v1",
        "airport": airport,
        "runway": runway,
        "selected_k": selected_k,
        "target_cluster": target_cluster,
        "qfu_flights": {
            "train": len(train_series),
            "val": len(val_series),
        },
        "target_cluster_flights": {
            "train": int(np.sum(train_labels == target_cluster)),
            "val": int(np.sum(val_labels == target_cluster)),
        },
        "candidates": candidates,
        "outer_test_tracks_opened": False,
    }, indent=2), encoding="utf-8")
    return {
        "clusters": cluster_path,
        "qfu_cohort": qfu_path,
        "cluster_cohort": cluster_cohort_path,
        "summary": summary_path,
    }
