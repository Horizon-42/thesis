from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(TS_ROOT), str(REPO_ROOT)]

from approach_clustering import evaluation as clustering_evaluation
from approach_clustering.artifacts import write_clustering_artifacts
from approach_clustering.features import horizontal_arc_feature
from approach_clustering.model import (
    ApproachClusterModel,
    _silhouette_score,
    fit_cluster_candidates,
)
from development_cohorts import DevelopmentCohort


def test_module_cli_bootstraps_repository_dependencies() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "approach_clustering", "--help"],
        cwd=TS_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "{build,compare}" in result.stdout


def test_horizontal_arc_feature_removes_sampling_speed() -> None:
    coarse = SimpleNamespace(supervision_values=np.array([
        [0.0, 0.0], [5.0, 0.0], [10.0, 0.0]
    ]))
    uneven = SimpleNamespace(supervision_values=np.array([
        [0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [10.0, 0.0]
    ]))

    np.testing.assert_allclose(
        horizontal_arc_feature(coarse, anchor_index=0, points=6),
        horizontal_arc_feature(uneven, anchor_index=0, points=6),
    )


def test_cluster_selection_finds_two_separated_path_families() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(loc=-4.0, scale=0.15, size=(40, 12))
    right = rng.normal(loc=4.0, scale=0.15, size=(20, 12))
    features = np.concatenate((left, right), axis=0)

    model, labels, candidates = fit_cluster_candidates(
        features,
        cluster_counts=(2, 3),
        pca_components=4,
        seed=11,
    )

    assert len(model.centers) == 2
    assert sorted(np.bincount(labels).tolist()) == [20, 40]
    assert {row["clusters"] for row in candidates} == {2, 3}
    np.testing.assert_array_equal(model.predict(features), labels)


def test_singleton_cluster_has_zero_silhouette() -> None:
    values = np.array([[0.0], [1.0], [10.0]])
    labels = np.array([0, 0, 1])

    score = _silhouette_score(values, labels)

    assert score == pytest.approx((0.9 + (8.0 / 9.0) + 0.0) / 3.0)


def test_clustering_artifact_preserves_zero_split_seed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    model = ApproachClusterModel(
        feature_mean=np.zeros(2),
        feature_scale=np.ones(2),
        pca_components=np.eye(2),
        centers=np.array([[0.0, 0.0], [1.0, 1.0]]),
    )

    paths = write_clustering_artifacts(
        tmp_path / "clusters",
        airport="KSJC",
        runway="30L",
        manifest_path=manifest,
        config={"seed": 1337, "split_seed": 0, "seq_len": 60},
        feature_points=32,
        pca_components=2,
        cluster_seed=17,
        model=model,
        candidates=[],
        train_series=[
            SimpleNamespace(dataset_id="KSJC:train-0"),
            SimpleNamespace(dataset_id="KSJC:train-1"),
        ],
        train_labels=np.array([0, 1]),
        val_series=[SimpleNamespace(dataset_id="KSJC:val-0")],
        val_labels=np.array([0]),
    )

    document = json.loads(paths["clusters"].read_text(encoding="utf-8"))
    assert document["source"]["split_seed"] == 0


def test_comparison_rebuilds_with_checkpoint_aircraft_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cohort = DevelopmentCohort(
        name="shared-validation",
        train_flight_ids=("KSJC:train",),
        val_flight_ids=("KSJC:val",),
        selection={"kind": "qfu"},
    )
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort.to_dict()), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(
        aircraft_type="B738",
        to_dict=lambda: {"aircraft_type": "B738"},
    )
    payload = {
        "split": {"train": ["KSJC:train"], "val": ["KSJC:val"]},
    }
    series = [SimpleNamespace(dataset_id="KSJC:val")]
    report = SimpleNamespace(to_dict=lambda: {"built": 1})
    captured: dict[str, object] = {}

    class Model:
        def to(self, _device):
            return self

    monkeypatch.setattr(
        clustering_evaluation,
        "arrival_data_provenance",
        lambda _data: {"schema_version": "test"},
    )
    monkeypatch.setattr(
        clustering_evaluation,
        "load_flight_dicts",
        lambda _data, include_flight_keys: [{}],
    )
    monkeypatch.setattr(
        clustering_evaluation,
        "load_checkpoint",
        lambda _path: (Model(), config, object(), payload),
    )
    monkeypatch.setattr(
        clustering_evaluation, "require_matching_data_provenance", lambda *_args: None
    )

    def build_with_aircraft_type(_flights, _config, *, aircraft_type=None):
        captured["aircraft_type"] = aircraft_type
        return series, report

    monkeypatch.setattr(clustering_evaluation, "build_series", build_with_aircraft_type)
    monkeypatch.setattr(clustering_evaluation, "resolve_device", lambda _name: "cpu")
    monkeypatch.setattr(
        clustering_evaluation,
        "evaluate_fixed_anchor_series",
        lambda *_args, **_kwargs: {"metrics": {}},
    )

    clustering_evaluation.compare_checkpoints(
        data=tmp_path / "manifest.json",
        cohort_path=cohort_path,
        checkpoints=[checkpoint],
        labels=["B738"],
        output_path=tmp_path / "comparison.json",
        device_name="cpu",
    )

    assert captured["aircraft_type"] == "B738"


def test_development_cohort_accepts_only_locked_train_and_validation() -> None:
    cohort = DevelopmentCohort(
        name="sample",
        train_flight_ids=("A:train",),
        val_flight_ids=("A:val",),
        selection={"kind": "qfu"},
    )
    splits = {
        "train": ["A:train"],
        "val": ["A:val"],
        "test": ["A:test"],
    }

    assert cohort.development_flight_ids(splits) == {"A:train", "A:val"}

    invalid = DevelopmentCohort(
        name="invalid",
        train_flight_ids=("A:test",),
        val_flight_ids=("A:val",),
        selection={"kind": "qfu"},
    )
    with pytest.raises(ValueError, match="train roster"):
        invalid.development_flight_ids(splits)
