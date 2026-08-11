"""Train-only approach-path clustering for development cohort experiments."""

__all__ = (
    "ApproachClusterModel",
    "fit_cluster_candidates",
    "horizontal_arc_feature",
    "horizontal_arc_features",
)


def __getattr__(name: str):
    """Load public helpers lazily so ``python -m`` can bootstrap first."""
    if name in {"ApproachClusterModel", "fit_cluster_candidates"}:
        from . import model

        return getattr(model, name)
    if name in {"horizontal_arc_feature", "horizontal_arc_features"}:
        from . import features

        return getattr(features, name)
    raise AttributeError(name)
