"""A scene sample as the prior reads it (multi-aircraft design §2.3, §2.5): every aircraft that appears in a sample on the
sample's steps, each at its own rows, which steps count toward the loss, and the tensors of a batch of samples.

A sample's steps run from the first step of any of its aircraft (a flight carried in from before the cut starts at its
row 0, so its earlier rows are context) to its last loss step. Each aircraft (`Node`) sits on the steps its rows hang on,
consecutively from its first (`scene.hang`), and keeps its own row numbers (§2.1): the position embedding and the first
predicted step read them. What counts toward the loss: a flight with a sentence, at the rows its own sentence asks
(from its first predicted step) that fall on the sample's loss steps; a background aircraft never (its words are
"none"). Rows past the sample's last step are left out (they belong to the next sample).

The edge features are not here: they need the separation rules, which this package does not reach
(`inference.scene_edges`, computed by the caller for each batch and handed to `batch_tensors`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

#: The most aircraft a sample's aircraft axis holds (design §2.3): the most any training-day sample or window held in the
#: M0 census (`outputs/POOLED/traffic/census_20260927`, KRDU 18). A sample of another split holding more is refused, not
#: trimmed.
A_MAX = 18


@dataclass(frozen=True)
class Node:
    """One aircraft of a sample: its rows' inputs (as `data.Flight` holds them: ``features [R, …]``, ``relative
    [R, K, …]``, ``static``, ``in_force`` / ``since`` / ``targets`` / ``asked`` ``[R, 6]``), the sample step its row 0
    hangs on, and whether it has a sentence."""

    key: str
    speaking: bool
    first_step: int
    features: np.ndarray
    relative: np.ndarray
    static: np.ndarray
    in_force: np.ndarray
    since: np.ndarray
    targets: np.ndarray
    asked: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class SceneSample:
    """A sample on its ``steps`` steps: its airport, its aircraft (each placed from its ``first_step``; rows past the last
    step are left out) and the loss on steps ``[loss_from, loss_to)``. Its arrays are laid out only when a batch is
    formed (`placed`): a flight carried across a cut is in two samples, and its rows are held once."""

    airport: int
    nodes: tuple[Node, ...]
    steps: int
    loss_from: int
    loss_to: int

    def __post_init__(self) -> None:
        if len(self.nodes) > A_MAX:
            raise ValueError(f"a sample of {len(self.nodes)} aircraft; the aircraft axis holds {A_MAX} (design §2.3)")
        for node in self.nodes:
            if not 0 <= node.first_step < self.steps:
                raise ValueError(f"{node.key} starts at step {node.first_step}, outside the sample's {self.steps}")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(node.key for node in self.nodes)

    def kept(self, node: Node) -> int:
        """How many of ``node``'s rows fall on the sample's steps."""
        return min(node.rows, self.steps - node.first_step)

    def asked(self, node: Node) -> np.ndarray:
        """``[kept, 6]``: what the loss counts on ``node``'s kept rows — its own sentence's asked rows on the loss steps,
        nothing for a background aircraft."""
        kept = self.kept(node)
        steps = np.arange(node.first_step, node.first_step + kept)
        on_loss = (steps >= self.loss_from) & (steps < self.loss_to)
        return node.asked[:kept] & on_loss[:, None] & node.speaking

    @property
    def asks(self) -> bool:
        return any(self.asked(node).any() for node in self.nodes)


def placed(samples: Sequence[SceneSample], slots: int) -> dict[str, np.ndarray]:
    """The samples laid out on one grid, padded to the most aircraft and steps among them (padding absent) and to
    ``slots`` candidate runways (the model's): per sample, aircraft and step — present, own row, the inputs, the targets
    and what the loss counts; the keys `train.to_batch` gives a single-aircraft batch, without the edge features."""
    count = len(samples)
    aircraft = max(len(s.nodes) for s in samples)
    steps = max(s.steps for s in samples)
    first = samples[0].nodes[0]

    def zeros(shape: tuple[int, ...], dtype) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    out = {"features": zeros((count, aircraft, steps, first.features.shape[1]), np.float32),
           "relative": zeros((count, aircraft, steps, slots, first.relative.shape[2]), np.float32),
           "static": zeros((count, aircraft, first.static.shape[0]), np.float32),
           "in_force": zeros((count, aircraft, steps, 6), np.int64), "since": zeros((count, aircraft, steps, 6), np.float32),
           "targets": zeros((count, aircraft, steps, 6), np.int64), "asked": zeros((count, aircraft, steps, 6), bool),
           "present": zeros((count, aircraft, steps), bool), "rows": zeros((count, aircraft, steps), np.int64),
           "airport": np.array([s.airport for s in samples], dtype=np.int64)}
    for b, sample in enumerate(samples):
        for a, node in enumerate(sample.nodes):
            kept = sample.kept(node)
            span = slice(node.first_step, node.first_step + kept)
            out["present"][b, a, span], out["rows"][b, a, span] = True, np.arange(kept)
            for name in ("features", "in_force", "since", "targets"):
                out[name][b, a, span] = getattr(node, name)[:kept]
            out["relative"][b, a, span, : node.relative.shape[1]] = node.relative[:kept]
            out["static"][b, a] = node.static
            out["asked"][b, a, span] = sample.asked(node)
    return out


def batch_tensors(samples: Sequence[SceneSample], edges: Sequence[np.ndarray], slots: int, device: torch.device
                  ) -> dict[str, torch.Tensor]:
    """A batch of samples (`placed`) with each sample's edge features (``[T, A, A, E]``, padded as the samples): the
    batch `model.Prior` reads."""
    out = placed(samples, slots)
    count, aircraft, steps = out["present"].shape
    out["edges"] = np.zeros((count, steps, aircraft, aircraft, edges[0].shape[-1]), dtype=np.float32)
    for b, edge in enumerate(edges):
        t, a = edge.shape[:2]
        out["edges"][b, :t, :a, :a] = edge
    return {name: torch.as_tensor(value, device=device) for name, value in out.items()}
