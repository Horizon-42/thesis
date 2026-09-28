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
    """A sample on its ``steps`` steps: its airport, its aircraft (``[A]``), and per aircraft and step — present, own row,
    the inputs, the targets and what the loss counts (``[A, T, …]``)."""

    airport: int
    keys: tuple[str, ...]
    speaking: tuple[bool, ...]
    present: np.ndarray            # [A, T] bool
    rows: np.ndarray               # [A, T] int64, 0 where absent
    features: np.ndarray           # [A, T, F] float32
    relative: np.ndarray           # [A, T, K, R] float32
    static: np.ndarray             # [A, S] float32
    in_force: np.ndarray           # [A, T, 6] int64
    since: np.ndarray              # [A, T, 6] float32
    targets: np.ndarray            # [A, T, 6] int64
    asked: np.ndarray              # [A, T, 6] bool

    @property
    def steps(self) -> int:
        return self.present.shape[1]


def scene_sample(airport: int, nodes: Sequence[Node], steps: int, loss_from: int, loss_to: int) -> SceneSample:
    """``nodes`` placed on ``steps`` steps, the loss on steps ``[loss_from, loss_to)`` (module docstring)."""
    if len(nodes) > A_MAX:
        raise ValueError(f"a sample of {len(nodes)} aircraft; the aircraft axis holds {A_MAX} (design §2.3)")
    count = len(nodes)
    first = nodes[0]
    present = np.zeros((count, steps), dtype=bool)
    rows = np.zeros((count, steps), dtype=np.int64)
    features = np.zeros((count, steps, first.features.shape[1]), dtype=np.float32)
    relative = np.zeros((count, steps, *first.relative.shape[1:]), dtype=np.float32)
    in_force = np.zeros((count, steps, 6), dtype=np.int64)
    since = np.zeros((count, steps, 6), dtype=np.float32)
    targets = np.zeros((count, steps, 6), dtype=np.int64)
    asked = np.zeros((count, steps, 6), dtype=bool)
    loss = np.zeros(steps, dtype=bool)
    loss[loss_from:loss_to] = True
    for a, node in enumerate(nodes):
        if node.first_step < 0:
            raise ValueError(f"{node.key} starts before the sample's first step")
        kept = min(node.rows, steps - node.first_step)
        span = slice(node.first_step, node.first_step + kept)
        present[a, span], rows[a, span] = True, np.arange(kept)
        features[a, span], relative[a, span] = node.features[:kept], node.relative[:kept]
        in_force[a, span], since[a, span], targets[a, span] = node.in_force[:kept], node.since[:kept], node.targets[:kept]
        if node.speaking:
            asked[a, span] = node.asked[:kept] & loss[span, None]
    return SceneSample(airport, tuple(n.key for n in nodes), tuple(n.speaking for n in nodes), present, rows, features,
                       relative, np.stack([n.static for n in nodes]).astype(np.float32), in_force, since, targets, asked)


def batch_tensors(samples: Sequence[SceneSample], edges: Sequence[np.ndarray], slots: int, device: torch.device
                  ) -> dict[str, torch.Tensor]:
    """A batch of samples padded to the most aircraft and steps among them (padding absent) and to ``slots`` candidate
    runways (the model's), with each sample's edge features (``[T, A, A, E]``): the batch `model.Prior` reads — the
    keys `train.to_batch` gives a single-aircraft one."""
    count = len(samples)
    aircraft = max(len(s.keys) for s in samples)
    steps = max(s.steps for s in samples)
    width = edges[0].shape[-1]
    first = samples[0]

    def zeros(shape: tuple[int, ...], dtype) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    out = {"features": zeros((count, aircraft, steps, first.features.shape[2]), np.float32),
           "relative": zeros((count, aircraft, steps, slots, first.relative.shape[3]), np.float32),
           "static": zeros((count, aircraft, first.static.shape[1]), np.float32),
           "in_force": zeros((count, aircraft, steps, 6), np.int64), "since": zeros((count, aircraft, steps, 6), np.float32),
           "targets": zeros((count, aircraft, steps, 6), np.int64), "asked": zeros((count, aircraft, steps, 6), bool),
           "present": zeros((count, aircraft, steps), bool), "rows": zeros((count, aircraft, steps), np.int64),
           "edges": zeros((count, steps, aircraft, aircraft, width), np.float32),
           "airport": np.array([s.airport for s in samples], dtype=np.int64)}
    for b, (sample, edge) in enumerate(zip(samples, edges)):
        a, t = len(sample.keys), sample.steps
        for name in ("features", "in_force", "since", "targets", "asked", "present", "rows"):
            out[name][b, :a, :t] = getattr(sample, name)
        out["relative"][b, :a, :t, : sample.relative.shape[2]] = sample.relative
        out["static"][b, :a] = sample.static
        out["edges"][b, :t, :a, :a] = edge
    return {name: torch.as_tensor(value, device=device) for name, value in out.items()}
