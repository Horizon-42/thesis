"""Command-line construction of train-only approach clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import TSConfig
from dataset import (
    arrival_data_provenance,
    build_series,
    flight_keys_by_split,
    load_flight_dicts,
    split_by_flight,
)

from .artifacts import write_clustering_artifacts
from .features import horizontal_arc_features
from .model import fit_cluster_candidates


def _cluster_counts(value: str) -> tuple[int, ...]:
    return tuple(int(token) for token in value.split(","))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build train-only approach-path clusters and development cohorts"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--recipe", required=True, help="history.json carrying TSConfig")
    parser.add_argument("--runway", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-points", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--clusters", type=_cluster_counts, default=(2, 3, 4))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    config = TSConfig.from_dict(recipe["config"])
    provenance = arrival_data_provenance(args.data)
    airport = provenance["manifests"][0]["airport"]
    split_keys = flight_keys_by_split(provenance, config)
    development_keys = set(split_keys["train"] + split_keys["val"])
    flights = load_flight_dicts(args.data, include_flight_keys=development_keys)
    series, report = build_series(flights, config)
    print(report.format())
    train, val, leaked_test = split_by_flight(series, config)
    if leaked_test:
        raise ValueError("clustering loaded outer-test trajectory series")

    runway = args.runway.strip().upper()
    train = [item for item in train if item.scenario.source["runway"] == runway]
    val = [item for item in val if item.scenario.source["runway"] == runway]
    anchor_index = config.seq_len - 1
    train_features = horizontal_arc_features(
        train, anchor_index=anchor_index, points=args.feature_points
    )
    val_features = horizontal_arc_features(
        val, anchor_index=anchor_index, points=args.feature_points
    )
    model, train_labels, candidates = fit_cluster_candidates(
        train_features,
        cluster_counts=args.clusters,
        pca_components=args.pca_components,
        seed=args.seed,
    )
    val_labels = model.predict(val_features)
    paths = write_clustering_artifacts(
        args.output_dir,
        airport=airport,
        runway=runway,
        manifest_path=args.data,
        config=config.to_dict(),
        feature_points=args.feature_points,
        pca_components=args.pca_components,
        cluster_seed=args.seed,
        model=model,
        candidates=candidates,
        train_series=train,
        train_labels=train_labels,
        val_series=val,
        val_labels=val_labels,
    )
    selected = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(
        f"selected K={selected['selected_k']} dominant cluster "
        f"{selected['target_cluster']}: "
        f"train={selected['target_cluster_flights']['train']} "
        f"val={selected['target_cluster_flights']['val']}"
    )
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0
