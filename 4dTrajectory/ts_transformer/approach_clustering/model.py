"""Small deterministic PCA + K-means model for approach geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ApproachClusterModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_components: np.ndarray
    centers: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized @ self.pca_components.T

    def predict(self, features: np.ndarray) -> np.ndarray:
        projected = self.transform(features)
        distances = np.sum(
            np.square(projected[:, None, :] - self.centers[None, :, :]), axis=2
        )
        return np.argmin(distances, axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "pca_components": self.pca_components.tolist(),
            "centers": self.centers.tolist(),
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> ApproachClusterModel:
        return cls(**{
            name: np.asarray(document[name], dtype=np.float64)
            for name in (
                "feature_mean", "feature_scale", "pca_components", "centers"
            )
        })


def _fit_projector(features: np.ndarray, components: int) -> tuple[np.ndarray, ...]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (features - mean) / scale
    _left, _singular, right = np.linalg.svd(standardized, full_matrices=False)
    basis = right[:components]
    return mean, scale, basis, standardized @ basis.T


def _initial_centers(
    values: np.ndarray, clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = [values[int(rng.integers(len(values)))]]
    nearest = np.sum(np.square(values - centers[0]), axis=1)
    for _ in range(1, clusters):
        probabilities = nearest / nearest.sum()
        centers.append(values[int(rng.choice(len(values), p=probabilities))])
        nearest = np.minimum(
            nearest,
            np.sum(np.square(values - centers[-1]), axis=1),
        )
    return np.asarray(centers)


def _fit_kmeans(
    values: np.ndarray,
    clusters: int,
    *,
    seed: int,
    n_init: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for initialization in range(n_init):
        rng = np.random.default_rng(seed + initialization)
        centers = _initial_centers(values, clusters, rng)
        for _ in range(100):
            distances = np.sum(
                np.square(values[:, None, :] - centers[None, :, :]), axis=2
            )
            labels = np.argmin(distances, axis=1)
            updated = np.stack([values[labels == label].mean(axis=0)
                                for label in range(clusters)])
            if np.allclose(updated, centers, rtol=0.0, atol=1e-8):
                centers = updated
                break
            centers = updated
        distances = np.sum(
            np.square(values[:, None, :] - centers[None, :, :]), axis=2
        )
        labels = np.argmin(distances, axis=1)
        inertia = float(distances[np.arange(len(values)), labels].sum())
        candidate = (centers, labels, inertia)
        if best is None or inertia < best[2]:
            best = candidate
    assert best is not None
    return best


def _silhouette_score(
    values: np.ndarray, labels: np.ndarray, *, max_samples: int = 2000
) -> float:
    if len(values) > max_samples:
        indices = np.linspace(0, len(values) - 1, max_samples, dtype=np.int64)
        values = values[indices]
        labels = labels[indices]
    squared = (
        np.sum(np.square(values), axis=1)[:, None]
        + np.sum(np.square(values), axis=1)[None, :]
        - 2.0 * values @ values.T
    )
    distances = np.sqrt(np.maximum(squared, 0.0))
    scores = np.empty(len(values), dtype=np.float64)
    cluster_ids = np.unique(labels)
    for index, label in enumerate(labels):
        own = labels == label
        own[index] = False
        if not own.any():
            scores[index] = 0.0
            continue
        within = float(distances[index, own].mean())
        nearest = min(
            float(distances[index, labels == other].mean())
            for other in cluster_ids if other != label
        )
        scores[index] = (nearest - within) / max(within, nearest)
    return float(scores.mean())


def fit_cluster_candidates(
    features: np.ndarray,
    *,
    cluster_counts: Sequence[int],
    pca_components: int,
    seed: int,
) -> tuple[ApproachClusterModel, np.ndarray, list[dict[str, Any]]]:
    mean, scale, basis, projected = _fit_projector(features, pca_components)
    candidates = []
    fitted = {}
    for clusters in cluster_counts:
        centers, labels, inertia = _fit_kmeans(
            projected, clusters, seed=seed
        )
        score = _silhouette_score(projected, labels)
        candidates.append({
            "clusters": clusters,
            "silhouette": score,
            "inertia": inertia,
            "counts": np.bincount(labels, minlength=clusters).tolist(),
        })
        fitted[clusters] = (centers, labels)
    selected = max(candidates, key=lambda row: (row["silhouette"], -row["clusters"]))
    centers, labels = fitted[selected["clusters"]]
    return (
        ApproachClusterModel(mean, scale, basis, centers),
        labels,
        candidates,
    )
